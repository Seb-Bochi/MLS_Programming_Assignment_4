"""Structured and unstructured pruning experiments from Part 1.

Unstructured masks reduce nonzero weights; ordinary TensorFlow dense kernels do
not automatically exploit that sparsity. Structured pruning physically removes
ResNet bottleneck channels while preserving the widths of residual additions.
"""

from pathlib import Path
import gzip
import re
import time

import numpy as np

from common import (
    tf, get_config, get_data, clone_model, fit_model, accuracy, save_model,
    model_weight_bytes, set_seed,
)


def _prunable_weights(model):
    """Yield unique convolution/dense kernels, including nested models."""
    seen = set()

    def visit(parent):
        for layer in parent.layers:
            if isinstance(layer, tf.keras.Model):
                yield from visit(layer)
            for attribute in ("kernel", "depthwise_kernel", "pointwise_kernel"):
                variable = getattr(layer, attribute, None)
                if variable is not None and id(variable) not in seen:
                    seen.add(id(variable))
                    yield layer, attribute, variable

    return list(visit(model))


def _key(layer, attribute):
    return f"{layer.name}/{attribute}"


def _validate_fraction(value, name):
    if not 0 <= value < 1:
        raise ValueError(f"{name} must be in [0, 1).")


def magnitude_prune(model, target_sparsity, previous_masks=None):
    """Apply exact global magnitude pruning; preserve an existing sparse mask.

    Returns masks keyed by ``layer_name/kernel_attribute``. Biases and batch
    normalization parameters are excluded from the sparsity denominator.
    """
    _validate_fraction(target_sparsity, "target_sparsity")
    entries = _prunable_weights(model)
    if not entries:
        raise ValueError("The model has no convolution or dense kernels to prune.")
    previous_masks = previous_masks or {}
    scores = []
    previously_pruned = 0
    for layer, attribute, variable in entries:
        values = np.abs(variable.numpy()).ravel()
        old_mask = previous_masks.get(_key(layer, attribute))
        if old_mask is not None:
            if old_mask.shape != tuple(variable.shape):
                raise ValueError("A previous pruning mask has an incompatible shape.")
            values = values.copy()
            values[np.asarray(old_mask).ravel() == 0] = -np.inf
            previously_pruned += int(np.count_nonzero(old_mask == 0))
        scores.append(values)
    scores = np.concatenate(scores)
    count = int(np.floor(target_sparsity * scores.size))
    if count < previously_pruned:
        raise ValueError("The requested sparsity would undo a previous pruning mask.")
    flat_mask = np.ones(scores.size, dtype=np.float32)
    if count:
        flat_mask[np.argpartition(scores, count - 1)[:count]] = 0
    masks, offset = {}, 0
    for layer, attribute, variable in entries:
        size = int(np.prod(variable.shape))
        mask = flat_mask[offset:offset + size].reshape(variable.shape)
        masks[_key(layer, attribute)] = mask
        variable.assign(variable * tf.cast(mask, variable.dtype))
        offset += size
    return masks


class FixedMaskCallback(tf.keras.callbacks.Callback):
    """Keep zeroed weights zero before inference and after every optimizer step."""

    def __init__(self, masks):
        super().__init__()
        self.masks = masks

    def enforce(self):
        for layer, attribute, variable in _prunable_weights(self.model):
            mask = self.masks.get(_key(layer, attribute))
            if mask is not None:
                variable.assign(variable * tf.cast(mask, variable.dtype))

    def on_train_begin(self, logs=None):
        self.enforce()

    def on_train_batch_begin(self, batch, logs=None):
        self.enforce()

    def on_train_batch_end(self, batch, logs=None):
        self.enforce()

    def on_train_end(self, logs=None):
        self.enforce()


class MagnitudePruningCallback(FixedMaskCallback):
    """Epoch-wise polynomial, linear gradual, or immediate constant pruning."""

    def __init__(self, target_sparsity, epochs, schedule="polynomial"):
        super().__init__({})
        _validate_fraction(target_sparsity, "target_sparsity")
        if schedule not in {"polynomial", "constant", "gradual"}:
            raise ValueError("schedule must be 'polynomial', 'constant', or 'gradual'.")
        self.target_sparsity = target_sparsity
        self.epochs = max(1, int(epochs))
        self.schedule = schedule
        self.sparsity_history = []

    def on_epoch_begin(self, epoch, logs=None):
        progress = 1.0 if self.epochs == 1 else epoch / (self.epochs - 1)
        if self.schedule == "constant":
            fraction = 1.0
        elif self.schedule == "polynomial":
            fraction = 1.0 - (1.0 - progress) ** 3
        else:
            fraction = progress
        target = self.target_sparsity * fraction
        self.masks = magnitude_prune(self.model, target, self.masks)
        self.sparsity_history.append(float(target))


def layer_sparsity(model):
    result = {}
    for layer, attribute, variable in _prunable_weights(model):
        values = variable.numpy()
        zeros = int(np.count_nonzero(values == 0))
        result[_key(layer, attribute)] = {
            "layer_type": type(layer).__name__,
            "weights": int(values.size),
            "zero_weights": zeros,
            "sparsity": zeros / values.size,
            "dense_weight_bytes": int(values.nbytes),
            "gzip_weight_bytes": len(gzip.compress(values.tobytes(), mtime=0)),
        }
    return result


def _global_sparsity(analysis):
    denominator = sum(row["weights"] for row in analysis.values())
    return sum(row["zero_weights"] for row in analysis.values()) / max(1, denominator)


def prune_and_finetune(model, target_sparsity=0.7, schedule="polynomial",
                       epochs=None, name="pruned"):
    """Clone, prune, and fine-tune without modifying the input model."""
    epochs = get_config().finetune_epochs if epochs is None else int(epochs)
    if epochs < 1:
        raise ValueError("Pruning fine-tuning requires at least one epoch.")
    pruned = clone_model(model)
    pruned.trainable = True
    callback = MagnitudePruningCallback(target_sparsity, epochs, schedule)
    training = fit_model(pruned, epochs=epochs, learning_rate=1e-4,
                         callbacks=[callback], name=name)
    # Early stopping/checkpoint restoration may restore an earlier, less sparse
    # epoch. Project onto the final target before evaluation and serialization.
    callback.masks = magnitude_prune(pruned, target_sparsity)
    callback.enforce()
    analysis = layer_sparsity(pruned)
    training["history"]["scheduled_sparsity"] = callback.sparsity_history
    result = {
        "model": pruned,
        "final_accuracy": accuracy(pruned),
        "sparsity_achieved": _global_sparsity(analysis),
        "layer_sparsity_analysis": analysis,
        "training_history": training["history"],
        "training_time_seconds": training["training_time_seconds"],
        "model_path": save_model(pruned, name),
        "evaluation_split": "validation",
        "storage_note": "Dense Keras storage; gzip_weight_bytes is compressed kernel payload only.",
    }
    return result


def _filter_importance(model, layers, metric):
    if metric not in {"l1_norm", "l2_norm", "gradient", "taylor"}:
        raise ValueError("Unknown filter importance metric.")
    if metric in {"l1_norm", "l2_norm"}:
        scores = {}
        for layer in layers:
            kernel = layer.kernel.numpy()
            axes = tuple(range(kernel.ndim - 1))
            scores[layer.name] = (np.sum(np.abs(kernel), axis=axes)
                                  if metric == "l1_norm"
                                  else np.sqrt(np.sum(kernel ** 2, axis=axes)))
        return scores

    _, _, x_val, y_val, _, _, _ = get_data()
    count = min(len(x_val), get_config().sensitivity_samples)
    scores = {layer.name: np.zeros(layer.filters, np.float64) for layer in layers}
    variables = [layer.kernel for layer in layers]
    batch_size = min(get_config().batch_size, 16)
    for start in range(0, count, batch_size):
        x = tf.convert_to_tensor(x_val[start:start + batch_size], dtype=tf.float32)
        y = tf.reshape(y_val[start:start + batch_size], [-1])
        with tf.GradientTape() as tape:
            for variable in variables:
                tape.watch(variable)
            logits = model(x, training=False)
            losses = tf.keras.losses.sparse_categorical_crossentropy(y, logits, from_logits=True)
            loss = tf.reduce_mean(losses)
        gradients = tape.gradient(loss, variables)
        for layer, gradient in zip(layers, gradients):
            if gradient is None:
                raise ValueError(f"No gradient reaches {layer.name}; cannot rank its filters.")
            values = gradient.numpy()
            if metric == "taylor":
                values = values * layer.kernel.numpy()
            scores[layer.name] += np.sum(np.abs(values), axis=(0, 1, 2)) * len(x)
    return {name: score / max(count, 1) for name, score in scores.items()}


def structured_prune_model(model, target_reduction=0.5, importance_metric="l1_norm"):
    """Physically remove internal channels of standard Keras ResNet50 blocks.

    The two internal convolutions are independently ranked. Their output slices
    propagate through batch normalization and into the following convolution's
    input dimension. The third convolution and shortcut keep their output width,
    so every residual addition remains shape compatible.
    """
    _validate_fraction(target_reduction, "target_reduction")
    pattern = re.compile(r"^(conv[2-5]_block\d+)_[12]_conv$")
    layers = [layer for layer in model.layers
              if isinstance(layer, tf.keras.layers.Conv2D) and pattern.match(layer.name)]
    if not layers:
        raise ValueError("Structured channel removal expects the flat Keras ResNet50 baseline.")
    scores = _filter_importance(model, layers, importance_metric)
    indices, removed = {}, {}
    for layer in layers:
        keep = max(1, int(round(layer.filters * (1 - target_reduction))))
        indices[layer.name] = np.sort(np.argsort(scores[layer.name], kind="stable")[-keep:])
        removed[layer.name] = layer.filters - keep

    def clone_layer(layer):
        config = layer.get_config()
        if layer.name in indices:
            config["filters"] = len(indices[layer.name])
        return layer.__class__.from_config(config)

    pruned = tf.keras.models.clone_model(model, clone_function=clone_layer)
    for original in model.layers:
        weights = original.get_weights()
        if not weights:
            continue
        copied = [value.copy() for value in weights]
        match = re.match(r"^(conv[2-5]_block\d+)_([123])_(conv|bn)$", original.name)
        if match:
            prefix, number, kind = match.groups()
            if kind == "conv":
                if number == "1":
                    output = indices[f"{prefix}_1_conv"]
                    copied[0] = np.take(copied[0], output, axis=3)
                    if len(copied) > 1:
                        copied[1] = copied[1][output]
                elif number == "2":
                    incoming = indices[f"{prefix}_1_conv"]
                    output = indices[f"{prefix}_2_conv"]
                    copied[0] = np.take(np.take(copied[0], incoming, axis=2), output, axis=3)
                    if len(copied) > 1:
                        copied[1] = copied[1][output]
                else:
                    copied[0] = np.take(copied[0], indices[f"{prefix}_2_conv"], axis=2)
            elif number in {"1", "2"}:
                selected = indices[f"{prefix}_{number}_conv"]
                copied = [value[selected] for value in copied]
        pruned.get_layer(original.name).set_weights(copied)
    changes = {
        "parameters_before": int(model.count_params()),
        "parameters_after": int(pruned.count_params()),
        "weight_bytes_before": int(model_weight_bytes(model)),
        "weight_bytes_after": int(model_weight_bytes(pruned)),
        "removed_internal_filters": int(sum(removed.values())),
        "residual_output_channels_preserved": True,
        "requested_internal_channel_reduction": float(target_reduction),
        "importance_metric": importance_metric,
        "retained_filter_indices": {name: index.tolist() for name, index in indices.items()},
    }
    return pruned, removed, changes


def _latency(layer, tensor, repeats=5):
    for _ in range(2):
        layer(tensor, training=False).numpy()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        layer(tensor, training=False).numpy()
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def _dense_sparse_comparison(dense, sparse):
    """Measure representative identical-shape layers using normal dense kernels."""
    candidates = [layer for layer, attribute, _ in _prunable_weights(dense)
                  if attribute == "kernel" and not isinstance(layer.input_shape, list)]
    count = min(6, len(candidates))
    selected = np.unique(np.linspace(0, len(candidates) - 1, count).astype(int)) if count else []
    rows = {}
    for index in selected:
        layer = candidates[index]
        shape = [1 if dimension is None else int(dimension) for dimension in layer.input_shape]
        tensor = tf.ones(shape, tf.float32)
        dense_ms = _latency(layer, tensor, repeats=3 if get_config().quick else 10)
        sparse_ms = _latency(sparse.get_layer(layer.name), tensor,
                             repeats=3 if get_config().quick else 10)
        rows[layer.name] = {"dense_ms": dense_ms, "masked_dense_kernel_ms": sparse_ms,
                            "observed_speedup": dense_ms / max(sparse_ms, 1e-12)}
    return {"layer_timings": rows,
            "note": "Both paths use TensorFlow dense kernels. Zeros do not imply sparse execution."}


def _history_analysis(result):
    history = result["training_history"]
    losses = np.asarray(history.get("val_loss", history.get("loss", [])), dtype=float)
    accuracies = np.asarray(history.get("val_accuracy", []), dtype=float)
    return {
        "epochs_run": int(len(losses)),
        "best_validation_epoch": int(np.argmax(accuracies) + 1) if len(accuracies) else None,
        "validation_loss_step_std": float(np.std(np.diff(losses))) if len(losses) > 1 else 0.0,
        "validation_loss_increases": int(np.sum(np.diff(losses) > 0)),
        "training_time_seconds": result["training_time_seconds"],
    }


def _subset_accuracy(model, x, y):
    correct = 0
    for start in range(0, len(x), get_config().batch_size):
        logits = model(x[start:start + get_config().batch_size], training=False).numpy()
        labels = np.asarray(y[start:start + get_config().batch_size]).reshape(-1)
        correct += int(np.sum(np.argmax(logits, axis=-1) == labels))
    return correct / max(1, len(x))


class PruningComparator:
    def __init__(self, base_model_path):
        from baseline_model import ResNetPreprocess
        self.base_model = tf.keras.models.load_model(
            base_model_path, compile=False,
            custom_objects={"Compression>ResNetPreprocess": ResNetPreprocess})
        self.pruning_results = {}

    def magnitude_based_pruning(self, target_sparsity=0.7, pruning_schedule="polynomial"):
        """Implement magnitude-based unstructured pruning and fine-tuning."""
        results = prune_and_finetune(self.base_model, target_sparsity, pruning_schedule,
                                    name=f"magnitude_{pruning_schedule}_{target_sparsity:g}")
        results["dense_vs_sparse_performance"] = _dense_sparse_comparison(self.base_model, results["model"])
        self.pruning_results["magnitude_pruning"] = results
        return results

    def structured_pruning(self, target_reduction=0.5, importance_metric="l1_norm"):
        """Remove internal bottleneck channels and fine-tune the smaller graph."""
        model, removed, changes = structured_prune_model(
            self.base_model, target_reduction, importance_metric)
        model.trainable = True
        name = f"structured_{importance_metric}_{target_reduction:g}"
        training = fit_model(model, learning_rate=1e-4, name=name)
        results = {
            "model": model,
            "filters_removed_per_layer": removed,
            "architecture_changes": changes,
            "final_accuracy": accuracy(model),
            "model_size_reduction": 1 - model_weight_bytes(model) / model_weight_bytes(self.base_model),
            "training_history": training["history"],
            "training_time_seconds": training["training_time_seconds"],
            "model_path": save_model(model, name),
            "evaluation_split": "validation",
        }
        self.pruning_results["structured_pruning"] = results
        return results

    def gradual_vs_oneshot_pruning(self, target_sparsity=0.7):
        """Compare matched initializations and training budgets for two schedules."""
        set_seed(get_config().seed)
        gradual = prune_and_finetune(self.base_model, target_sparsity, "gradual", name="gradual_pruning")
        set_seed(get_config().seed)
        oneshot = prune_and_finetune(self.base_model, target_sparsity, "constant", name="oneshot_pruning")
        convergence = {"gradual": _history_analysis(gradual), "oneshot": _history_analysis(oneshot)}
        comparison = {
            "gradual_pruning": gradual,
            "oneshot_pruning": oneshot,
            "convergence_analysis": convergence,
            "stability_metrics": {
                name: {key: value for key, value in row.items() if key.startswith("validation_loss")}
                for name, row in convergence.items()
            },
            "accuracy_difference": gradual["final_accuracy"] - oneshot["final_accuracy"],
            "time_overhead_seconds": gradual["training_time_seconds"] - oneshot["training_time_seconds"],
        }
        self.pruning_results["gradual_vs_oneshot"] = comparison
        return comparison

    def analyze_pruning_sensitivity(self):
        """Independently mask each kernel at three sparsities on validation data."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        _, _, x_val, y_val, _, _, _ = get_data()
        count = min(len(x_val), get_config().sensitivity_samples)
        x, y = x_val[:count], y_val[:count]
        model = clone_model(self.base_model)
        baseline = _subset_accuracy(model, x, y)
        sparsities = (0.3, 0.5, 0.7)
        rows = {}
        for layer, attribute, variable in _prunable_weights(model):
            original = variable.numpy()
            scores = np.abs(original).ravel()
            row = {}
            try:
                for sparsity in sparsities:
                    mask = np.ones(scores.size, dtype=original.dtype)
                    remove = int(scores.size * sparsity)
                    if remove:
                        mask[np.argpartition(scores, remove - 1)[:remove]] = 0
                    variable.assign(original * mask.reshape(original.shape))
                    measured = _subset_accuracy(model, x, y)
                    row[str(sparsity)] = {"accuracy": measured, "accuracy_drop": baseline - measured}
            finally:
                variable.assign(original)
            rows[_key(layer, attribute)] = row
        names = list(rows)
        matrix = [[rows[name][str(s)]["accuracy_drop"] * 100 for s in sparsities] for name in names]
        fig, ax = plt.subplots(figsize=(7, max(4, len(names) * 0.16)))
        plotted = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
        ax.set_yticks(range(len(names)), names, fontsize=6)
        ax.set_xticks(range(len(sparsities)), [f"{s:.0%}" for s in sparsities])
        ax.set_xlabel("Independently imposed kernel sparsity")
        ax.set_title("Layer sensitivity: validation accuracy loss (percentage points)")
        fig.colorbar(plotted, ax=ax)
        fig.tight_layout()
        plot_path = Path(get_config().output_dir) / "pruning_sensitivity.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        ranked = sorted(names, key=lambda name: rows[name]["0.7"]["accuracy_drop"], reverse=True)
        result = {"baseline_accuracy": baseline, "validation_samples": count,
                  "layer_sensitivity": rows, "most_sensitive": ranked[:5],
                  "least_sensitive": list(reversed(ranked[-5:])), "heatmap_path": str(plot_path)}
        self.pruning_results["sensitivity"] = result
        return result

    def lottery_ticket_hypothesis_test(self):
        """Iteratively prune, reset to saved initialization, and retrain.

        For an ImageNet-initialized baseline, this tests rewinding to the initial
        transfer-learning weights, not the original random ImageNet initialization.
        A same-mask random-initialization control is trained at the final sparsity.
        """
        config = get_config()
        candidates = [Path(config.output_dir) / "checkpoints" / "baseline_initial.weights.h5",
                      Path("checkpoints") / "baseline_initial.weights.h5"]
        initial_path = next((path for path in candidates if path.exists()), None)
        if initial_path is None:
            raise FileNotFoundError("Train the baseline first to save checkpoints/baseline_initial.weights.h5.")
        initial = clone_model(self.base_model)
        initial.load_weights(initial_path)
        initial_weights = initial.get_weights()
        current = clone_model(self.base_model)
        masks, rounds = {}, []
        baseline_accuracy = accuracy(self.base_model)
        count = 2 if config.quick else 3
        for iteration in range(count):
            target = 1 - 0.7 ** (iteration + 1)
            masks = magnitude_prune(current, target, masks)
            current.set_weights(initial_weights)
            current.trainable = True
            callback = FixedMaskCallback(masks)
            training = fit_model(current, epochs=config.baseline_epochs, learning_rate=1e-3,
                                 callbacks=[callback], name=f"lottery_ticket_round_{iteration + 1}")
            callback.enforce()
            measured = accuracy(current)
            rounds.append({"iteration": iteration + 1, "sparsity": _global_sparsity(layer_sparsity(current)),
                           "accuracy": measured, "accuracy_drop": baseline_accuracy - measured,
                           "matches_baseline_within_one_percentage_point": measured >= baseline_accuracy - 0.01,
                           "training_history": training["history"],
                           "training_time_seconds": training["training_time_seconds"]})
        set_seed(config.seed + 1)
        random_control = tf.keras.models.clone_model(self.base_model)
        random_control.trainable = True
        control_callback = FixedMaskCallback(masks)
        control_training = fit_model(random_control, epochs=config.baseline_epochs, learning_rate=1e-3,
                                     callbacks=[control_callback], name="lottery_random_control")
        control_callback.enforce()
        result = {
            "model": current,
            "final_accuracy": rounds[-1]["accuracy"],
            "baseline_accuracy": baseline_accuracy,
            "iterations": rounds,
            "model_path": save_model(current, "lottery_ticket"),
            "initialization_path": str(initial_path),
            "random_initialization_control": {
                "model": random_control, "final_accuracy": accuracy(random_control),
                "training_history": control_training["history"],
                "training_time_seconds": control_training["training_time_seconds"],
                "model_path": save_model(random_control, "lottery_random_control"),
            },
            "interpretation": "Saved baseline-start weights are rewound; pretrained weights remain pretrained.",
            "evaluation_split": "validation",
        }
        self.pruning_results["lottery_ticket"] = result
        return result
