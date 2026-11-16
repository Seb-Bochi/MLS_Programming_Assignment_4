"""Measured inference, storage, robustness and repeated-seed evaluation."""
from __future__ import annotations

import inspect
import time
from pathlib import Path
import numpy as np
from scipy import stats
from common import (tf, get_config, get_data, set_seed, save_model, MemoryMonitor,
                    model_weight_bytes, make_lite_interpreter)


class InferenceRunner:
    """Common logits interface for Keras and actual TensorFlow Lite artifacts."""
    def __init__(self, model):
        self.model = model
        self.lite = isinstance(model, (str, Path, bytes, bytearray))
        if self.lite:
            kwargs = {"model_content": bytes(model)} if isinstance(model, (bytes, bytearray)) else {"model_path": str(model)}
            self.interpreter = make_lite_interpreter(**kwargs)
            self.interpreter.allocate_tensors()
        else:
            self.call = tf.function(lambda x: model(x, training=False), reduce_retracing=True)

    def __call__(self, images):
        images = np.asarray(images, dtype=np.float32)
        if not self.lite:
            return self.call(tf.convert_to_tensor(images)).numpy()
        details = self.interpreter.get_input_details()[0]
        if tuple(details["shape"]) != images.shape:
            self.interpreter.resize_tensor_input(details["index"], images.shape, strict=False)
            self.interpreter.allocate_tensors()
            details = self.interpreter.get_input_details()[0]
        if np.issubdtype(details["dtype"], np.integer):
            scale, zero = details["quantization"]
            if scale <= 0:
                raise ValueError("Integer input has no quantization scale")
            limits = np.iinfo(details["dtype"])
            inputs = np.clip(np.round(images / scale + zero), limits.min, limits.max).astype(details["dtype"])
        else:
            inputs = images.astype(details["dtype"])
        self.interpreter.set_tensor(details["index"], inputs)
        self.interpreter.invoke()
        output = self.interpreter.get_output_details()[0]
        logits = self.interpreter.get_tensor(output["index"]).astype(np.float32)
        if np.issubdtype(output["dtype"], np.integer):
            scale, zero = output["quantization"]
            logits = (logits - zero) * scale
        return logits


def estimate_flops(model):
    """Conv/Dense arithmetic per image, counting a multiply-add as two FLOPs.

    These are dense operation counts, not sparsity or bit-width speed estimates.
    Activation, normalization and preprocessing operations are omitted.
    """
    if not isinstance(model, tf.keras.Model):
        return None
    total = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            total += estimate_flops(layer)
        elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            _, h, w, channels = layer.output_shape
            total += 2 * h * w * channels * np.prod(layer.kernel_size)
        elif isinstance(layer, tf.keras.layers.Conv2D):
            _, h, w, channels = layer.output_shape
            total += 2 * h * w * channels * np.prod(layer.kernel_size) * layer.input_shape[-1] // layer.groups
        elif isinstance(layer, tf.keras.layers.Dense):
            total += 2 * layer.input_shape[-1] * layer.units
    return int(total)


def expected_calibration_error(probabilities, labels, bins=15):
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == np.asarray(labels).reshape(-1)
    ids = np.minimum((confidence * bins).astype(int), bins - 1)
    return float(sum(np.mean(ids == i) * abs(np.mean(correct[ids == i]) - np.mean(confidence[ids == i]))
                     for i in range(bins) if np.any(ids == i)))


def prediction_metrics(logits, labels):
    labels = np.asarray(labels).reshape(-1).astype(int)
    probabilities = tf.nn.softmax(logits, axis=-1).numpy()
    return {"accuracy": float(np.mean(probabilities.argmax(1) == labels)),
            "ece": expected_calibration_error(probabilities, labels),
            "negative_log_likelihood": float(-np.log(np.maximum(probabilities[np.arange(len(labels)), labels], 1e-8)).mean()),
            "brier_score": float(np.square(probabilities - np.eye(probabilities.shape[1])[labels]).sum(1).mean())}


class CompressionEvaluator:
    def __init__(self, baseline_model=None, power_watts=None):
        self.metrics = {key: [] for key in (
            "accuracy", "model_size_mb", "inference_latency_ms", "memory_usage_mb",
            "flops_reduction", "compression_ratio", "energy_consumption")}
        self.baseline_model = baseline_model
        self.baseline_metrics = None
        self.power_watts = power_watts
        if power_watts is not None and power_watts <= 0:
            raise ValueError("power_watts must be positive")

    def benchmark_compressed_model(self, model, test_data, model_name):
        """Benchmark real artifacts; first call establishes the baseline.

        Supply validation data for model selection and test data for final results.
        Peak RSS is sampled process memory, including other retained models.
        """
        x, y = test_data
        if len(x) == 0 or len(x) != len(y):
            raise ValueError("A nonempty aligned (images, labels) pair is required")
        if self.baseline_metrics is None and self.baseline_model is not None and model is not self.baseline_model:
            self.benchmark_compressed_model(self.baseline_model, test_data, "baseline")
        cfg = get_config()
        runner = InferenceRunner(model)
        if isinstance(model, tf.keras.Model):
            artifact_path = save_model(model, f"benchmark_{model_name}")
            size = Path(artifact_path).stat().st_size
            weight_bytes = model_weight_bytes(model)
        elif isinstance(model, (bytes, bytearray)):
            size, artifact_path, weight_bytes = len(model), None, None
        else:
            artifact_path = str(model)
            size, weight_bytes = Path(model).stat().st_size, None
        with MemoryMonitor() as memory:
            outputs = [runner(x[start:start + cfg.batch_size])
                       for start in range(0, len(x), cfg.batch_size)]
            logits = np.concatenate(outputs)
            score = prediction_metrics(logits, y)
            latencies = {}
            for label, images in (("single", x[:1]), ("batch", x[:cfg.batch_size])):
                for _ in range(3):
                    runner(images)
                samples = []
                for _ in range(cfg.benchmark_repeats):
                    start = time.perf_counter()
                    runner(images)  # materializes numpy, synchronizing GPU execution
                    samples.append((time.perf_counter() - start) * 1000)
                latencies[label] = {"median": float(np.median(samples)),
                                    "p95": float(np.percentile(samples, 95))}
        flops = estimate_flops(model)
        baseline = self.baseline_metrics
        batch_count = min(cfg.batch_size, len(x))
        metrics = {
            "model_name": model_name, "test_accuracy": score["accuracy"],
            "model_size_mb": size / 2**20, "artifact_path": artifact_path,
            "weight_size_mb": None if weight_bytes is None else weight_bytes / 2**20,
            "single_inference_ms": latencies["single"]["median"],
            "single_p95_ms": latencies["single"]["p95"],
            "batch_inference_ms": latencies["batch"]["median"],
            "batch_size": batch_count,
            "throughput_images_per_second": 1000 * batch_count / latencies["batch"]["median"],
            "peak_memory_mb": memory.peak_bytes / 2**20,
            "incremental_memory_mb": (memory.peak_bytes - memory.start_bytes) / 2**20,
            "flops": flops,
            "flops_reduction": (1 - flops / baseline["flops"])
                if baseline and flops is not None and baseline["flops"] else (0.0 if flops else None),
            "compression_ratio": baseline["model_size_mb"] / (size / 2**20) if baseline else 1.0,
            "accuracy_loss_percent": 100 * (baseline["test_accuracy"] - score["accuracy"]) if baseline else 0.0,
            "energy_consumption": self.power_watts * latencies["single"]["median"] / 1000
                if self.power_watts is not None else None,
            "energy_method": "user supplied constant watts times seconds (joules/image)"
                if self.power_watts is not None else "unavailable: no power measurement",
            "ece": score["ece"], "negative_log_likelihood": score["negative_log_likelihood"],
            "brier_score": score["brier_score"],
            "execution_backend": "TFLite CPU, 1 thread; built-in kernels without delegates on Windows"
                if runner.lite else "TensorFlow default device/threads",
            "flops_method": "Conv/Dense estimate; multiply-add=2; excludes preprocessing/BN/activation",
            "memory_method": "sampled process RSS; includes runtime and retained models",
        }
        metrics["compression_target_met"] = metrics["compression_ratio"] >= 10 and metrics["accuracy_loss_percent"] < 5
        if baseline is None:
            self.baseline_metrics = dict(metrics)
        for key, value in (("accuracy", metrics["test_accuracy"]), ("model_size_mb", metrics["model_size_mb"]),
                           ("inference_latency_ms", metrics["single_inference_ms"]),
                           ("memory_usage_mb", metrics["peak_memory_mb"]),
                           ("flops_reduction", metrics["flops_reduction"]),
                           ("compression_ratio", metrics["compression_ratio"]),
                           ("energy_consumption", metrics["energy_consumption"])):
            self.metrics[key].append(value)
        return metrics

    def cross_validate_compression(self, compression_function, num_seeds=3):
        """Repeated-seed experiment on a fixed split (not k-fold CV)."""
        if num_seeds < 2:
            raise ValueError("At least two seeds are needed for uncertainty")
        rows = []
        for offset in range(num_seeds):
            seed = get_config().seed + offset
            set_seed(seed)
            result = compression_function(seed=seed) if "seed" in inspect.signature(compression_function).parameters else compression_function()
            if "final_accuracy" not in result and "test_accuracy" not in result:
                raise ValueError("compression_function must return final_accuracy or test_accuracy")
            rows.append(result)
        scores = np.array([row.get("final_accuracy", row.get("test_accuracy")) for row in rows])
        mean, std = float(scores.mean()), float(scores.std(ddof=1))
        margin = float(stats.t.ppf(0.975, num_seeds - 1) * std / np.sqrt(num_seeds))
        significance = {"pvalue": None, "reason": "No matched per-seed baseline_accuracy supplied"}
        if all("baseline_accuracy" in row for row in rows):
            differences = scores - np.array([row["baseline_accuracy"] for row in rows])
            significance = {"test": "paired two-sided t-test over seeds", "pvalue":
                float(stats.ttest_1samp(differences, 0).pvalue) if np.std(differences) > 0 else None,
                "mean_difference": float(differences.mean()),
                "caution": "Small seed counts have limited statistical power"}
        return {"runs": rows, "mean_accuracy": mean, "std_accuracy": std,
                "confidence_interval_95": [mean - margin, mean + margin],
                "significance": significance, "method": "fixed data split, different training seeds"}

    def analyze_compression_robustness(self, compressed_model):
        """Corruption accuracy, calibration and per-class failure rates."""
        _, _, _, _, x, y, _ = get_data()
        runner = InferenceRunner(compressed_model)
        cfg = get_config()
        rng = np.random.default_rng(cfg.seed)
        def evaluate(images):
            logits = np.concatenate([runner(images[i:i + cfg.batch_size])
                                     for i in range(0, len(images), cfg.batch_size)])
            return prediction_metrics(logits, y), logits
        clean, logits = evaluate(x)
        corruptions = {}
        for sigma in (5, 15, 30):
            noisy = np.clip(x.astype(np.float32) + rng.normal(0, sigma, x.shape), 0, 255)
            corruptions[f"gaussian_sigma_{sigma}"] = evaluate(noisy)[0]
        for brightness in (-30, 30):
            shifted = np.clip(x.astype(np.float32) + brightness, 0, 255)
            corruptions[f"brightness_{brightness}"] = evaluate(shifted)[0]
        y = np.asarray(y).reshape(-1)
        per_class = {int(label): float(np.mean(logits.argmax(1)[y == label] == label))
                     for label in np.unique(y)}
        result = {"clean": clean, "corruptions": corruptions, "per_class_accuracy": per_class,
                  "worst_classes": sorted(per_class, key=per_class.get)[:10],
                  "evaluation_split": "test", "corruption_scale": "raw pixel range 0..255"}
        if self.baseline_model is not None:
            baseline = InferenceRunner(self.baseline_model)
            base_logits = np.concatenate([baseline(x[i:i + cfg.batch_size])
                                          for i in range(0, len(x), cfg.batch_size)])
            result["baseline_clean"] = prediction_metrics(base_logits, y)
            result["ece_change"] = clean["ece"] - result["baseline_clean"]["ece"]
        return result
