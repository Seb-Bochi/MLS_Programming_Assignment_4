"""Knowledge distillation experiments for CIFAR-100.

All networks return logits. Hyperparameters are selected on the validation split;
the held-out test split is reserved for ``CompressionEvaluator``.
"""

from pathlib import Path
import time

import numpy as np

from common import (
    tf, get_config, get_data, make_dataset, clone_model, fit_model,
    accuracy, save_model, set_seed,
)
from baseline_model import create_student_model


def _feature_layers(model, count=3):
    """Find spatial features without depending on equal teacher/student widths."""
    preferred = ["stage1_features", "stage2_features", "stage3_features"]
    if not all(any(layer.name == name for layer in model.layers) for name in preferred):
        preferred = ["conv2_block3_out", "conv3_block4_out", "conv4_block6_out"]
    names = {layer.name for layer in model.layers}
    if all(name in names for name in preferred):
        return [model.get_layer(name) for name in preferred]
    candidates = []
    for layer in model.layers:
        try:
            shape = layer.output.shape
            if len(shape) == 4 and not isinstance(layer, tf.keras.layers.InputLayer):
                candidates.append(layer)
        except (AttributeError, TypeError):
            continue
    if not candidates:
        raise ValueError("Feature/attention distillation requires spatial feature layers.")
    indices = np.unique(np.linspace(0, len(candidates) - 1, min(count, len(candidates))).astype(int))
    return [candidates[index] for index in indices]


def _attention(features, spatial_shape):
    features = tf.image.resize(features, spatial_shape, method="bilinear")
    spatial_energy = tf.reduce_mean(tf.square(features), axis=-1)
    flat = tf.reshape(spatial_energy, [tf.shape(spatial_energy)[0], -1])
    return tf.math.l2_normalize(flat, axis=-1, epsilon=1e-8)


class _BestStudent(tf.keras.callbacks.Callback):
    """Restore the best student without copying an entire frozen teacher."""

    def __init__(self, student):
        super().__init__()
        self.student = student
        self.best = -np.inf
        self.weights = None

    def on_epoch_end(self, epoch, logs=None):
        score = (logs or {}).get("val_accuracy")
        if score is not None and score > self.best:
            self.best = float(score)
            self.weights = self.student.get_weights()

    def on_train_end(self, logs=None):
        if self.weights is not None:
            self.student.set_weights(self.weights)


class _Distiller(tf.keras.Model):
    """Response KD, normalized spatial attention, or learned feature adapters."""

    def __init__(self, teachers, student, temperature, alpha, mode, feature_weight):
        super().__init__(name="distillation_trainer")
        self.teachers = list(teachers)
        self.student = student
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.mode = mode
        self.feature_weight = float(feature_weight)
        self.adapters = []
        self.student_extractor = None
        self.teacher_extractor = None
        if mode != "response":
            if len(teachers) != 1:
                raise ValueError("Feature/attention transfer expects one teacher.")
            student_layers = _feature_layers(student)
            teacher_layers = _feature_layers(teachers[0])
            count = min(len(student_layers), len(teacher_layers))
            student_layers, teacher_layers = student_layers[-count:], teacher_layers[-count:]
            self.student_extractor = tf.keras.Model(
                student.inputs, [layer.output for layer in student_layers] + [student.output]
            )
            self.teacher_extractor = tf.keras.Model(
                teachers[0].inputs, [layer.output for layer in teacher_layers] + [teachers[0].output]
            )
            if mode == "feature":
                for index, (student_layer, teacher_layer) in enumerate(zip(student_layers, teacher_layers)):
                    adapter = tf.keras.layers.Conv2D(
                        int(teacher_layer.output.shape[-1]), 1, use_bias=False,
                        name=f"feature_adapter_{index}",
                    )
                    adapter.build((None, None, None, int(student_layer.output.shape[-1])))
                    self.adapters.append(adapter)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.hard_tracker = tf.keras.metrics.Mean(name="student_loss")
        self.soft_tracker = tf.keras.metrics.Mean(name="distillation_loss")
        self.feature_tracker = tf.keras.metrics.Mean(name="transfer_loss")
        self.accuracy_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return [self.loss_tracker, self.hard_tracker, self.soft_tracker,
                self.feature_tracker, self.accuracy_tracker]

    def call(self, inputs, training=False):
        return self.student(inputs, training=training)

    def train_step(self, data):
        images, labels = data
        labels = tf.reshape(labels, [-1])
        teacher_features = []
        if self.teacher_extractor is not None:
            outputs = self.teacher_extractor(images, training=False)
            teacher_features = [tf.stop_gradient(value) for value in outputs[:-1]]
            teacher_logits = [tf.stop_gradient(outputs[-1])]
        else:
            teacher_logits = [tf.stop_gradient(teacher(images, training=False)) for teacher in self.teachers]
        # Average probability distributions, not logits, for an ensemble teacher.
        soft_targets = tf.add_n([
            tf.nn.softmax(logits / self.temperature, axis=-1) for logits in teacher_logits
        ]) / len(teacher_logits)
        with tf.GradientTape() as tape:
            if self.student_extractor is not None:
                outputs = self.student_extractor(images, training=True)
                student_features, student_logits = outputs[:-1], outputs[-1]
            else:
                student_features = []
                student_logits = self.student(images, training=True)
            hard_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
                labels, student_logits, from_logits=True
            ))
            log_predictions = tf.nn.log_softmax(student_logits / self.temperature, axis=-1)
            soft_loss = tf.reduce_mean(tf.reduce_sum(
                soft_targets * (tf.math.log(tf.maximum(soft_targets, 1e-8)) - log_predictions),
                axis=-1,
            )) * self.temperature ** 2
            transfer_loss = tf.constant(0.0, dtype=tf.float32)
            if self.mode == "attention":
                terms = []
                for student_features_i, teacher_features_i in zip(student_features, teacher_features):
                    spatial_shape = tf.shape(student_features_i)[1:3]
                    terms.append(tf.reduce_mean(tf.reduce_sum(tf.square(
                        _attention(student_features_i, spatial_shape)
                        - _attention(teacher_features_i, spatial_shape)
                    ), axis=-1)))
                transfer_loss = tf.add_n(terms) / len(terms)
            elif self.mode == "feature":
                terms = []
                for adapter, student_features_i, teacher_features_i in zip(
                    self.adapters, student_features, teacher_features
                ):
                    projected = adapter(student_features_i)
                    target = tf.image.resize(teacher_features_i, tf.shape(projected)[1:3])
                    # Channel normalization prevents large teacher activations
                    # from overwhelming the supervised/KL objectives.
                    terms.append(tf.reduce_mean(tf.square(
                        tf.math.l2_normalize(projected, axis=-1, epsilon=1e-8)
                        - tf.math.l2_normalize(target, axis=-1, epsilon=1e-8)
                    )))
                transfer_loss = tf.add_n(terms) / len(terms)
            loss = self.alpha * hard_loss + (1.0 - self.alpha) * soft_loss
            loss += self.feature_weight * transfer_loss
            if self.student.losses:
                loss += tf.add_n(self.student.losses)
        variables = list(self.student.trainable_variables)
        for adapter in self.adapters:
            variables.extend(adapter.trainable_variables)
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients([
            (gradient, variable) for gradient, variable in zip(gradients, variables)
            if gradient is not None
        ])
        self.loss_tracker.update_state(loss)
        self.hard_tracker.update_state(hard_loss)
        self.soft_tracker.update_state(soft_loss)
        self.feature_tracker.update_state(transfer_loss)
        self.accuracy_tracker.update_state(labels, student_logits)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        images, labels = data
        labels = tf.reshape(labels, [-1])
        logits = self.student(images, training=False)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True
        ))
        self.loss_tracker.update_state(loss)
        self.accuracy_tracker.update_state(labels, logits)
        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy_tracker.result()}


def distill_student(teacher, student=None, temperature=4.0, alpha=0.5,
                    mode="response", feature_weight=1.0, epochs=None,
                    name=None, callbacks=None):
    """Train a student and return its validation score, history and elapsed time.

    ``alpha`` is the supervised-loss weight; the soft KL term is multiplied by
    ``temperature**2``. Teachers are always called in inference mode and their
    variables are excluded from the optimizer. Only the student is exported.
    """
    if temperature <= 0 or not 0 <= alpha <= 1 or feature_weight < 0:
        raise ValueError("Require temperature > 0, alpha in [0, 1], and feature_weight >= 0.")
    if mode not in {"response", "attention", "feature"}:
        raise ValueError("mode must be 'response', 'attention', or 'feature'.")
    config = get_config()
    teachers = list(teacher) if isinstance(teacher, (list, tuple)) else [teacher]
    if not teachers:
        raise ValueError("At least one teacher is required.")
    student = student if student is not None else create_student_model()
    if any(int(item.output_shape[-1]) != int(student.output_shape[-1]) for item in teachers):
        raise ValueError("Teacher and student must have the same number of output classes.")
    epochs = config.distill_epochs if epochs is None else int(epochs)
    if epochs < 1:
        raise ValueError("epochs must be at least one.")
    trainer = _Distiller(teachers, student, temperature, alpha, mode, feature_weight)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    best_student = _BestStudent(student)
    training_callbacks = [
        best_student,
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, epochs // 8), min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=max(4, epochs // 4),
            restore_best_weights=False,
        ),
    ] + list(callbacks or [])
    start = time.perf_counter()
    history = trainer.fit(
        make_dataset("train", augment=True),
        validation_data=make_dataset("val"), epochs=epochs,
        callbacks=training_callbacks, verbose=config.verbose,
    ).history
    duration = time.perf_counter() - start
    result = {
        "model": student,
        "final_accuracy": accuracy(student, split="val"),
        "training_history": {key: [float(value) for value in values] for key, values in history.items()},
        "training_time_seconds": duration,
        "temperature": float(temperature), "alpha": float(alpha),
        "mode": mode, "feature_weight": float(feature_weight),
        "teacher_count": len(teachers), "selection_split": "validation",
    }
    if mode == "feature":
        result["adapter_parameter_count"] = int(sum(adapter.count_params() for adapter in trainer.adapters))
        result["adapters_used_during_training_only"] = True
    if name:
        result["artifact_path"] = str(save_model(student, name))
    return result


def _soft_statistics(model, temperature, images):
    outputs = []
    for start in range(0, len(images), get_config().batch_size):
        outputs.append(tf.nn.softmax(model(images[start:start + get_config().batch_size], training=False)
                                     / float(temperature), axis=-1).numpy())
    probabilities = np.concatenate(outputs)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-8)), axis=-1)
    return {"mean_entropy_nats": float(entropy.mean()),
            "mean_max_probability": float(probabilities.max(axis=-1).mean()),
            "effective_classes": float(np.exp(entropy).mean())}


def _response_agreement(teacher, student, images):
    agreements = []
    divergences = []
    for start in range(0, len(images), get_config().batch_size):
        batch = images[start:start + get_config().batch_size]
        p = tf.nn.softmax(teacher(batch, training=False), axis=-1).numpy()
        q = tf.nn.softmax(student(batch, training=False), axis=-1).numpy()
        agreements.extend((p.argmax(-1) == q.argmax(-1)).tolist())
        divergences.extend(np.sum(p * (np.log(np.maximum(p, 1e-8))
                                      - np.log(np.maximum(q, 1e-8))), axis=-1).tolist())
    return {"prediction_agreement": float(np.mean(agreements)),
            "teacher_to_student_kl": float(np.mean(divergences))}


def _spatial_analysis(teacher, student, images):
    teacher_layers, student_layers = _feature_layers(teacher), _feature_layers(student)
    count = min(len(teacher_layers), len(student_layers))
    teacher_layers, student_layers = teacher_layers[-count:], student_layers[-count:]
    teacher_features = tf.keras.Model(teacher.inputs, [layer.output for layer in teacher_layers])
    student_features = tf.keras.Model(student.inputs, [layer.output for layer in student_layers])
    differences = [[] for _ in range(count)]
    for start in range(0, len(images), get_config().batch_size):
        batch = images[start:start + get_config().batch_size]
        teacher_values, student_values = teacher_features(batch, training=False), student_features(batch, training=False)
        if count == 1:
            teacher_values = teacher_values if isinstance(teacher_values, (list, tuple)) else [teacher_values]
            student_values = student_values if isinstance(student_values, (list, tuple)) else [student_values]
        for index, (teacher_value, student_value) in enumerate(zip(teacher_values, student_values)):
            shape = tf.shape(student_value)[1:3]
            distances = tf.reduce_sum(tf.square(_attention(teacher_value, shape)
                                                - _attention(student_value, shape)), axis=-1)
            differences[index].extend(distances.numpy().tolist())
    return {f"{t.name} -> {s.name}": {"mean_squared_attention_distance": float(np.mean(values))}
            for t, s, values in zip(teacher_layers, student_layers, differences)}


class DistillationFramework:
    def __init__(self, teacher_model, student_architecture):
        self.teacher = teacher_model
        self.student_arch = student_architecture
        self.distillation_results = {}
        if isinstance(student_architecture, tf.keras.Model):
            self._initial_student = clone_model(student_architecture)
        elif callable(student_architecture):
            self._initial_student = student_architecture()
        elif student_architecture == "mobile":
            self._initial_student = create_student_model(num_classes=int(teacher_model.output_shape[-1]))
        else:
            raise ValueError("student_architecture must be 'mobile', a model, or a model factory.")

    def _student(self):
        return clone_model(self._initial_student)

    def _analysis_images(self):
        images = get_data()[2]
        return images[:min(len(images), get_config().sensitivity_samples)]

    def temperature_optimization(self, temperature_range=(1, 20), num_trials=10):
        """Grid-search temperature with identical student initialization/budgets."""
        if len(temperature_range) != 2 or temperature_range[0] <= 0 or temperature_range[1] < temperature_range[0]:
            raise ValueError("temperature_range must contain an ordered positive interval.")
        if num_trials < 1:
            raise ValueError("num_trials must be positive.")
        config = get_config()
        trials, curve, statistics, transfer = {}, {}, {}, {}
        images = self._analysis_images()
        for temperature in np.linspace(*temperature_range, num=int(num_trials)):
            set_seed(config.seed)
            key = f"{temperature:.6g}"
            result = distill_student(self.teacher, self._student(), temperature=float(temperature),
                                     name=f"distilled_temperature_{key.replace('.', '_')}")
            trials[key] = result
            curve[key] = result["final_accuracy"]
            statistics[key] = _soft_statistics(self.teacher, temperature, images)
            transfer[key] = _response_agreement(self.teacher, result["model"], images)
        best = max(curve, key=curve.get)
        self.optimal_temperature = float(best)
        results = {
            "optimal_temperature": float(best),
            "temperature_accuracy_curve": curve,
            "soft_target_analysis": statistics,
            "knowledge_transfer_metrics": transfer,
            "trials": trials,
            "best_model": trials[best]["model"],
            "selection_split": "validation",
        }
        import matplotlib.pyplot as plt
        directory = Path(config.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot([float(key) for key in curve], list(curve.values()), marker="o")
        axis.set(xlabel="Distillation temperature", ylabel="Validation accuracy",
                 title="Temperature selection (fixed initialization and training budget)")
        figure.tight_layout()
        path = directory / "temperature_optimization.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        results["plot_path"] = str(path)
        self.distillation_results["temperature_optimization"] = results
        return results

    def _intermediate_student(self, fraction):
        """Approximate a fraction of teacher parameters with the mobile family."""
        teacher_count = self.teacher.count_params()
        target_count = int(teacher_count * fraction)
        final_count = self._initial_student.count_params()
        if target_count <= final_count:
            raise ValueError("Intermediate parameter budgets must exceed the final student's size.")
        base = create_student_model(num_classes=int(self.teacher.output_shape[-1]))
        width = max(0.125, np.sqrt(target_count / base.count_params()))
        candidate = base
        # Most weights scale quadratically with channel width. Correct the
        # estimate twice because classification/depthwise weights scale linearly.
        for _ in range(3):
            candidate = create_student_model(width=float(width), num_classes=int(self.teacher.output_shape[-1]))
            width *= np.sqrt(target_count / candidate.count_params())
        return candidate, target_count

    def progressive_distillation(self, intermediate_sizes=[0.75, 0.5, 0.25]):
        """Distill through decreasing parameter budgets, then the target student."""
        fractions = [float(value) for value in intermediate_sizes]
        if any(not 0 < value < 1 for value in fractions) or any(
            left <= right for left, right in zip(fractions, fractions[1:])
        ):
            raise ValueError("intermediate_sizes must be strictly decreasing fractions in (0, 1).")
        config = get_config()
        temperature = getattr(self, "optimal_temperature", 4.0)
        current_teacher = self.teacher
        intermediates, chain, preservation = {}, {}, {}
        images = self._analysis_images()
        for index, fraction in enumerate(fractions):
            set_seed(config.seed)
            model, target_count = self._intermediate_student(fraction)
            key = f"stage_{index + 1}"
            result = distill_student(current_teacher, model, temperature=temperature,
                                     name=f"progressive_{key}")
            result.update({"target_teacher_parameter_fraction": fraction,
                           "target_parameter_count": target_count,
                           "actual_parameter_count": model.count_params(),
                           "actual_teacher_parameter_fraction": model.count_params() / self.teacher.count_params()})
            intermediates[key] = model
            chain[key] = result
            preservation[key] = _response_agreement(self.teacher, model, images)
            current_teacher = model
        set_seed(config.seed)
        final = distill_student(current_teacher, self._student(), temperature=temperature,
                                name="progressive_final_student")
        chain["final_student"] = final
        preservation["final_student"] = _response_agreement(self.teacher, final["model"], images)
        set_seed(config.seed)
        direct = distill_student(self.teacher, self._student(), temperature=temperature,
                                 name="direct_distilled_student")
        results = {
            "intermediate_models": intermediates,
            "progressive_chain_results": chain,
            "direct_distillation_comparison": {
                "direct": direct, "progressive": final,
                "accuracy_difference_percentage_points": 100 * (final["final_accuracy"] - direct["final_accuracy"]),
                "progressive_total_training_seconds": sum(item["training_time_seconds"] for item in chain.values()),
                "direct_training_seconds": direct["training_time_seconds"],
                "comparison_note": "Same target student initialization and final-stage epochs; progressive pays for additional stages.",
            },
            "knowledge_preservation_analysis": preservation,
        }
        self.distillation_results["progressive_distillation"] = results
        return results

    def attention_transfer(self):
        """Compare response KD to response KD plus normalized spatial attention."""
        config = get_config()
        temperature = getattr(self, "optimal_temperature", 4.0)
        set_seed(config.seed)
        vanilla = distill_student(self.teacher, self._student(), temperature=temperature,
                                  name="vanilla_distilled_student")
        set_seed(config.seed)
        attention = distill_student(self.teacher, self._student(), temperature=temperature,
                                    mode="attention", feature_weight=10.0,
                                    name="attention_distilled_student")
        images = self._analysis_images()
        results = {
            "attention_maps_analysis": {
                "teacher_layers": [layer.name for layer in _feature_layers(self.teacher)],
                "student_layers": [layer.name for layer in _feature_layers(attention["model"])],
                "definition": "L2-normalized spatial map of channel-mean squared activations",
            },
            "attention_transfer_loss": attention["training_history"].get("transfer_loss", []),
            "combined_distillation_results": {
                "attention": attention, "vanilla": vanilla,
                "accuracy_difference_percentage_points": 100 * (attention["final_accuracy"] - vanilla["final_accuracy"]),
            },
            "spatial_pattern_analysis": {
                "vanilla": _spatial_analysis(self.teacher, vanilla["model"], images),
                "attention": _spatial_analysis(self.teacher, attention["model"], images),
            },
        }
        self.distillation_results["attention_transfer"] = results
        return results

    def feature_matching_distillation(self):
        """Match multiple feature stages with trainable 1x1 channel adapters."""
        config = get_config()
        temperature = getattr(self, "optimal_temperature", 4.0)
        set_seed(config.seed)
        response = distill_student(self.teacher, self._student(), temperature=temperature,
                                   name="feature_response_control")
        set_seed(config.seed)
        feature = distill_student(self.teacher, self._student(), temperature=temperature,
                                  mode="feature", feature_weight=10.0,
                                  name="feature_distilled_student")
        results = {
            "response_distillation": response,
            "feature_matching": feature,
            "matched_teacher_layers": [layer.name for layer in _feature_layers(self.teacher)],
            "matched_student_layers": [layer.name for layer in _feature_layers(feature["model"])],
            "accuracy_difference_percentage_points": 100 * (feature["final_accuracy"] - response["final_accuracy"]),
            "adapter_note": "Learned 1x1 adapters participate in optimization and are discarded at inference.",
        }
        self.distillation_results["feature_matching"] = results
        return results

    def self_distillation_experiments(self):
        """Train independent student-family teachers and distill their ensemble."""
        config = get_config()
        teacher_count = 2 if config.quick else 3
        temperature = getattr(self, "optimal_temperature", 4.0)
        teachers, teacher_results = [], {}
        for index in range(teacher_count):
            set_seed(config.seed + index)
            if isinstance(self.student_arch, str) and self.student_arch == "mobile":
                teacher = create_student_model(num_classes=int(self.teacher.output_shape[-1]))
            elif callable(self.student_arch) and not isinstance(self.student_arch, tf.keras.Model):
                teacher = self.student_arch()
            else:
                # clone_model normally preserves weights; fresh architecture
                # cloning here deliberately runs seeded initializers instead.
                teacher = tf.keras.models.clone_model(self._initial_student)
            fitted = fit_model(teacher, epochs=config.distill_epochs, learning_rate=1e-3,
                               name=f"self_teacher_{index + 1}")
            teachers.append(teacher)
            teacher_results[f"teacher_{index + 1}"] = {
                "model": teacher, "seed": config.seed + index,
                "final_accuracy": accuracy(teacher, split="val"),
                "training_history": fitted["history"],
                "training_time_seconds": fitted["training_time_seconds"],
                "artifact_path": str(save_model(teacher, f"self_teacher_{index + 1}")),
            }
        comparisons, marginal = {}, {}
        previous = None
        for count in range(1, teacher_count + 1):
            set_seed(config.seed)
            result = distill_student(teachers[:count], self._student(), temperature=temperature,
                                     name=f"self_distilled_{count}_teachers")
            comparisons[str(count)] = result
            if previous is not None:
                marginal[str(count)] = 100 * (result["final_accuracy"] - previous)
            previous = result["final_accuracy"]
        results = {
            "independent_teachers": teacher_results,
            "single_teacher_self_distillation": comparisons["1"],
            "ensemble_distillation": comparisons,
            "diminishing_returns_percentage_points": marginal,
            "teacher_training_total_seconds": sum(item["training_time_seconds"] for item in teacher_results.values()),
            "comparison_note": "Teachers and students share an architecture; ensemble size varies at fixed student initialization and budget.",
        }
        self.distillation_results["self_distillation"] = results
        return results
