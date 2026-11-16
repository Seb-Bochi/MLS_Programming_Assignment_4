"""Part 4: measured compression interactions and pipeline ordering.

All pipeline selection uses validation data. A distillation stage changes the
architecture; constraints introduced earlier are therefore reapplied explicitly
to its student, rather than silently disappearing with the teacher graph.
"""

from itertools import permutations
from pathlib import Path
import time

import numpy as np

from common import (get_config, get_data, clone_model, fit_model, accuracy,
                    save_model, set_seed, MemoryMonitor)
from baseline_model import create_student_model
from part1_pruning import (prune_and_finetune, magnitude_prune, FixedMaskCallback,
                           layer_sparsity, _global_sparsity)
from part2_quantization import (quantize_model, packed_weights, export_tflite,
                                materialize_quantized_weights)
from part3_distillation import distill_student


def pareto_frontier(records, cost_key="model_size_mb", accuracy_key="validation_accuracy"):
    """Return nondominated names, maximizing accuracy and minimizing cost.

    Equal points both remain nondominated. Missing or nonfinite measurements are
    excluded; neither absent latency nor absent energy is interpreted as zero.
    """
    points = []
    for name, row in records.items():
        score = row.get(accuracy_key, row.get("test_accuracy"))
        cost = row.get(cost_key)
        if score is not None and cost is not None and np.isfinite(score) and np.isfinite(cost):
            points.append((name, float(score), float(cost)))
    winners = []
    for name, score, cost in points:
        dominated = any(other_score >= score and other_cost <= cost
                        and (other_score > score or other_cost < cost)
                        for other_name, other_score, other_cost in points if other_name != name)
        if not dominated:
            winners.append(name)
    return sorted(winners, key=lambda name: records[name][cost_key])


class _StudentMask(FixedMaskCallback):
    """Bind masks to the student, never the trainer's similarly named teacher."""

    def __init__(self, student, masks):
        super().__init__(masks)
        self.student = student

    def set_model(self, model):
        super().set_model(self.student)


def _stability(histories):
    differences = []
    increases = 0
    for history in histories.values():
        losses = np.asarray(history.get("val_loss", []), dtype=float)
        if len(losses) > 1:
            change = np.diff(losses)
            differences.extend(change.tolist())
            increases += int(np.sum(change > 0))
    return {"validation_loss_step_std": float(np.std(differences)) if differences else None,
            "validation_loss_increases": increases,
            "measured_within_stages": True}


class CompressionInteractionAnalyzer:
    def __init__(self, baseline_model):
        self.baseline_model = baseline_model
        self.interaction_results = {}
        self._records = {}
        self._evaluator = None
        self._baseline = None

    def _prepare_evaluator(self):
        if self._evaluator is None:
            from compression_evaluator import CompressionEvaluator
            self._evaluator = CompressionEvaluator()
            _, _, x_val, y_val, _, _, _ = get_data()
            self._baseline = dict(self._evaluator.benchmark_compressed_model(
                self.baseline_model, (x_val, y_val), "baseline"))
            self._baseline["validation_accuracy"] = self._baseline.pop("test_accuracy")
            self._baseline["evaluation_split"] = "validation"

    def _run_pipeline(self, ordering):
        ordering = tuple(ordering)
        if not ordering or len(set(ordering)) != len(ordering) or set(ordering) - {"pruning", "quantization", "distillation"}:
            raise ValueError("Ordering must contain unique supported compression techniques.")
        if ordering in self._records:
            return self._records[ordering]
        self._prepare_evaluator()
        config = get_config()
        set_seed(config.seed)
        name = "interaction_" + "_then_".join(ordering)
        model = clone_model(self.baseline_model)
        masks, histories, stages = {}, {}, []
        active = set()
        training_seconds = 0.0
        start = time.perf_counter()
        with MemoryMonitor() as memory:
            for step, technique in enumerate(ordering, 1):
                stage_name = f"{name}_stage{step}"
                if technique == "pruning":
                    result = prune_and_finetune(model, target_sparsity=0.7,
                                               schedule="polynomial", name=stage_name)
                    model = result["model"]
                    masks = magnitude_prune(model, 0.7)
                    history = result["training_history"]
                    duration = result["training_time_seconds"]
                    details = {"target_sparsity": 0.7, "schedule": "polynomial"}
                elif technique == "quantization":
                    model = quantize_model(model, bits=8, calibration_data=get_data()[-1])
                    callbacks = [FixedMaskCallback(masks)] if masks else []
                    training = fit_model(model, learning_rate=1e-4,
                                         callbacks=callbacks, name=stage_name)
                    if callbacks:
                        callbacks[0].enforce()
                    history, duration = training["history"], training["training_time_seconds"]
                    details = {"bits": 8, "method": "calibrated reference QAT",
                               "native_integer_execution": False}
                else:
                    teacher = model
                    student = create_student_model(num_classes=int(teacher.output_shape[-1]))
                    if "quantization" in active:
                        student = quantize_model(student, bits=8, calibration_data=get_data()[-1])
                    masks = magnitude_prune(student, 0.7) if "pruning" in active else {}
                    callbacks = [_StudentMask(student, masks)] if masks else []
                    result = distill_student(teacher, student=student, temperature=4.0,
                                             callbacks=callbacks, name=stage_name)
                    model = result["model"]
                    if callbacks:
                        callbacks[0].enforce()
                    history, duration = result["training_history"], result["training_time_seconds"]
                    details = {"teacher": "preceding pipeline model", "temperature": 4.0,
                               "reapplied_student_constraints": sorted(active),
                               "student_parameters": int(student.count_params())}
                active.add(technique)
                histories[technique] = history
                training_seconds += duration
                stages.append({"technique": technique, "training_time_seconds": duration,
                               "validation_accuracy": accuracy(model), **details})
        model_path = save_model(model, name)
        _, _, x_val, y_val, _, _, _ = get_data()
        metrics = self._evaluator.benchmark_compressed_model(model, (x_val, y_val), name)
        metrics["validation_accuracy"] = metrics.pop("test_accuracy")
        metrics["evaluation_split"] = "validation"
        artifact = {}
        if "quantization" in active:
            packed_path = packed_weights(model, name)
            packed_mb = Path(packed_path).stat().st_size / 2 ** 20
            tflite_path = export_tflite(materialize_quantized_weights(model), name,
                                        mode="int8", calibration_data=get_data()[-1])
            lite_metrics = dict(self._evaluator.benchmark_compressed_model(
                tflite_path, (x_val, y_val), name + "_native_int8"))
            lite_metrics["validation_accuracy"] = lite_metrics.pop("test_accuracy")
            lite_metrics["evaluation_split"] = "validation"
            artifact = {"packed_weights_path": packed_path, "packed_storage_mb": packed_mb,
                        "packed_storage_compression_ratio": self._baseline["model_size_mb"] / packed_mb,
                        "packed_runtime_note": "Packed disk storage expands into FP32 reference execution.",
                        "tflite_path": tflite_path, "tflite_metrics": lite_metrics,
                        "tflite_conversion_note": "Quantized kernels are baked; Lite recalibrates activation scales. Accuracy is measured again."}
        result = {
            "model": model, "model_path": model_path, "ordering": list(ordering),
            "final_accuracy": metrics["validation_accuracy"],
            "compression_ratio": metrics["compression_ratio"],
            "training_time_seconds": training_seconds,
            "pipeline_time_seconds": time.perf_counter() - start,
            "peak_training_rss_mb": memory.peak_bytes / 2 ** 20,
            "training_rss_increase_mb": max(0, memory.peak_bytes - memory.start_bytes) / 2 ** 20,
            "convergence_stability": _stability(histories), "training_history": histories,
            "metrics": metrics, "stages": stages,
            "sparsity_achieved": _global_sparsity(layer_sparsity(model)),
            **artifact,
        }
        self._records[ordering] = result
        return result

    def comprehensive_compression_analysis(self):
        """Evaluate requested combinations plus single-technique controls."""
        controls = {technique: self._run_pipeline((technique,))
                    for technique in ("pruning", "quantization", "distillation")}
        requested = {
            "pruning_then_quantization": ("pruning", "quantization"),
            "quantization_then_pruning": ("quantization", "pruning"),
            "pruning_with_distillation": ("pruning", "distillation"),
            "quantization_with_distillation": ("quantization", "distillation"),
            "all_three_combined": ("pruning", "quantization", "distillation"),
        }
        combinations = {name: self._run_pipeline(order) for name, order in requested.items()}
        baseline_accuracy = self._baseline["validation_accuracy"]
        interactions = {}
        for name, order in requested.items():
            expected = baseline_accuracy - sum(baseline_accuracy - controls[t]["final_accuracy"] for t in order)
            observed = combinations[name]["final_accuracy"]
            interactions[name] = {
                "additive_reference_accuracy": expected,
                "observed_accuracy": observed,
                "interaction_effect_percentage_points": 100 * (observed - expected),
                "interpretation": "Positive means higher accuracy than additive individual losses; descriptive, not causal.",
            }
        forward = combinations["pruning_then_quantization"]
        reverse = combinations["quantization_then_pruning"]
        combinations["ordering_sensitivity_analysis"] = {
            "pruning_quantization_accuracy_difference": forward["final_accuracy"] - reverse["final_accuracy"],
            "pruning_quantization_training_time_difference_seconds": forward["training_time_seconds"] - reverse["training_time_seconds"],
            "preferred_pair_order": (forward if forward["final_accuracy"] >= reverse["final_accuracy"] else reverse)["ordering"],
        }
        combinations["individual_controls"] = controls
        combinations["interaction_effects"] = interactions
        combinations["baseline"] = self._baseline
        combinations["optimal_pipelines"] = self._recommendations()
        self.interaction_results = combinations
        return combinations

    def _metric_rows(self):
        self._prepare_evaluator()
        rows = {"baseline": self._baseline}
        for order, result in self._records.items():
            name = " -> ".join(order)
            rows[name] = result["metrics"]
            if "tflite_metrics" in result:
                rows[name + " [native INT8]"] = result["tflite_metrics"]
        return rows

    def _recommendations(self):
        rows = self._metric_rows()
        minimum = self._baseline["validation_accuracy"] - 0.05
        acceptable = {name: row for name, row in rows.items() if row["validation_accuracy"] > minimum}
        if not acceptable:
            return {"status": "No evaluated candidate satisfies the accuracy budget."}
        return {
            "size_priority": min(acceptable, key=lambda name: acceptable[name]["model_size_mb"]),
            "latency_priority": min(acceptable, key=lambda name: acceptable[name]["single_inference_ms"]),
            "accuracy_priority": max(acceptable, key=lambda name: acceptable[name]["validation_accuracy"]),
            "tenfold_size_target_candidates": [name for name, row in acceptable.items() if row["compression_ratio"] >= 10],
            "accuracy_budget_percentage_points": 5,
            "selection_split": "validation",
        }

    def pareto_frontier_analysis(self):
        """Find measured accuracy/size and accuracy/latency frontiers and plot them."""
        if not self._records:
            self.comprehensive_compression_analysis()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows = self._metric_rows()
        size = pareto_frontier(rows, "model_size_mb")
        latency = pareto_frontier(rows, "single_inference_ms")
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, cost, frontier, label in zip(axes, ("model_size_mb", "single_inference_ms"),
                                             (size, latency), ("Serialized model size (MiB)", "Single inference (ms)")):
            ax.scatter([row[cost] for row in rows.values()],
                       [row["validation_accuracy"] * 100 for row in rows.values()], alpha=0.7)
            ax.plot([rows[name][cost] for name in frontier],
                    [rows[name]["validation_accuracy"] * 100 for name in frontier], "r.-")
            for name in frontier:
                ax.annotate(name, (rows[name][cost], rows[name]["validation_accuracy"] * 100), fontsize=6)
            ax.set(xlabel=label, ylabel="Validation accuracy (%)")
            ax.grid(alpha=0.2)
        fig.tight_layout()
        path = Path(get_config().output_dir) / "interaction_pareto_frontiers.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        result = {
            "accuracy_vs_size_frontier": {name: rows[name] for name in size},
            "accuracy_vs_latency_frontier": {name: rows[name] for name in latency},
            "non_dominated_solutions": {"size": size, "latency": latency},
            "dominated_solutions": {"size": sorted(set(rows) - set(size)), "latency": sorted(set(rows) - set(latency))},
            "use_case_recommendations": self._recommendations(), "plot_path": str(path),
        }
        self.interaction_results["pareto_analysis"] = result
        return result

    def compression_pipeline_optimization(self):
        """Evaluate all six orderings with matched seeds and per-stage budgets."""
        results = {" -> ".join(order): self._run_pipeline(order)
                   for order in permutations(("pruning", "quantization", "distillation"))}
        ranked = sorted(results, key=lambda name: (-results[name]["final_accuracy"],
                                                  results[name]["training_time_seconds"]))
        outcome = {"orderings": results, "ranked_by_accuracy_then_training_time": ranked,
                   "best_accuracy_ordering": ranked[0],
                   "fastest_training_ordering": min(results, key=lambda name: results[name]["training_time_seconds"]),
                   "accuracy_range_percentage_points": 100 * (max(r["final_accuracy"] for r in results.values())
                                                               - min(r["final_accuracy"] for r in results.values())),
                   "selection_split": "validation"}
        self.interaction_results["pipeline_optimization"] = outcome
        return outcome

    def failure_mode_analysis(self):
        """Flag observed accuracy, size and latency failures against the baseline."""
        if not self._records:
            self.comprehensive_compression_analysis()
        baseline = self._baseline
        failures = {}
        for order, result in self._records.items():
            metrics = result["metrics"]
            problems, guidance = [], []
            if baseline["validation_accuracy"] - metrics["validation_accuracy"] >= 0.05:
                problems.append("Accuracy loss reaches or exceeds five percentage points.")
                guidance.append("Reduce pruning sparsity, increase student capacity, or extend recovery training.")
            if metrics["compression_ratio"] < 10:
                problems.append("The serialized executable model misses the tenfold size target.")
                guidance.append("Use a smaller student and a verified INT8 export; distinguish packed storage from execution memory.")
            if metrics["single_inference_ms"] >= baseline["single_inference_ms"]:
                problems.append("No measured single-sample latency improvement.")
                guidance.append("Prefer physical channel removal and target-supported integer kernels over masked dense kernels.")
            if not np.isfinite(metrics["validation_accuracy"]):
                problems.append("Nonfinite accuracy measurement.")
                guidance.append("Inspect numerical stability, calibration ranges and training logs.")
            failures[" -> ".join(order)] = {"observed_failures": problems,
                                            "mitigations_to_test": list(dict.fromkeys(guidance)),
                                            "metrics": metrics}
        result = {"pipelines": failures,
                  "limitations": ["Single-seed differences are descriptive until repeated-seed evaluation.",
                                  "Process RSS includes TensorFlow caches and other retained models.",
                                  "Reference QAT uses floating-point kernels; packed storage does not imply integer speedups.",
                                  "Student architecture replacement explicitly reapplies preceding constraints."],
                  "selection_split": "validation"}
        self.interaction_results["failure_modes"] = result
        return result
