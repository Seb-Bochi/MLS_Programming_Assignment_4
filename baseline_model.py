"""Baseline creation, CIFAR-100 preparation and reproducible training."""
from __future__ import annotations

import time
import numpy as np
from sklearn.model_selection import train_test_split
from common import (tf, get_config, set_seed, set_data, make_dataset, compile_model,
                    accuracy, save_json, clone_model)


@tf.keras.utils.register_keras_serializable(package="Compression")
class ResNetPreprocess(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.keras.applications.resnet50.preprocess_input(tf.cast(inputs, tf.float32))


def create_baseline_model(input_shape=(32, 32, 3), num_classes=100):
    """ResNet-50 with ImageNet initialization, batch norm and head dropout.

    Inputs are RGB pixels in [0,255]. Preprocessing is inside the saved graph.
    The >75% target is evaluated after training, never assumed.
    """
    cfg = get_config()
    set_seed(cfg.seed)
    inputs = tf.keras.Input(input_shape, name="images")
    x = tf.keras.layers.Resizing(cfg.image_size, cfg.image_size, name="resize")(inputs)
    x = ResNetPreprocess(name="resnet_preprocess")(x)
    backbone = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet" if cfg.imagenet_weights else None,
        input_tensor=x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(backbone.output)
    x = tf.keras.layers.Dropout(0.3, name="head_dropout")(x)
    logits = tf.keras.layers.Dense(num_classes, name="logits",
                                  kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    return compile_model(tf.keras.Model(inputs, logits, name="cifar100_resnet50"))


def create_student_model(width=1.0, num_classes=100, input_shape=(32, 32, 3), depth=1):
    """Compact separable-convolution student; width/depth form the NAS space."""
    if width <= 0 or depth < 1:
        raise ValueError("width and depth must be positive")
    inputs = tf.keras.Input(input_shape, name="images")
    x = tf.keras.layers.Rescaling(1.0 / 255, name="normalize")(inputs)
    for stage, channels in enumerate((32, 64, 128), 1):
        filters = max(4, int(round(channels * width / 4)) * 4)
        for block in range(depth):
            # Explicit depthwise/pointwise layers allow weight quantization separately.
            x = tf.keras.layers.DepthwiseConv2D(3, padding="same", use_bias=False,
                strides=2 if stage > 1 and block == 0 else 1,
                name=f"stage{stage}_{block}_depthwise")(x)
            x = tf.keras.layers.BatchNormalization(name=f"stage{stage}_{block}_dw_bn")(x)
            x = tf.keras.layers.ReLU(name=f"stage{stage}_{block}_dw_relu")(x)
            x = tf.keras.layers.Conv2D(filters, 1, use_bias=False,
                name=f"stage{stage}_{block}_pointwise")(x)
            x = tf.keras.layers.BatchNormalization(name=f"stage{stage}_{block}_bn")(x)
            x = tf.keras.layers.ReLU(name=f"stage{stage}_features" if block == depth - 1
                                    else f"stage{stage}_{block}_relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="student_pool")(x)
    x = tf.keras.layers.Dropout(0.2, name="student_dropout")(x)
    logits = tf.keras.layers.Dense(num_classes, name="logits")(x)
    return compile_model(tf.keras.Model(inputs, logits, name="mobile_student"))


def prepare_compression_datasets():
    """Return train/validation/test arrays and 1000 training-only calibration images.

    The official test set is untouched. Validation is stratified 10% of training.
    Raw uint8 storage avoids duplicating CIFAR in RAM; graphs normalize internally.
    Training augmentation (crop/flip) is performed lazily in make_dataset.
    """
    cfg = get_config()
    (images, labels), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
    labels, y_test = labels.reshape(-1), y_test.reshape(-1)
    indexes = np.arange(len(images))
    train_idx, val_idx = train_test_split(indexes, test_size=0.1,
        random_state=cfg.seed, stratify=labels)
    def subset(idx, y, limit):
        if limit is None or limit >= len(idx):
            return idx
        if limit < 100:
            return np.random.default_rng(cfg.seed).choice(idx, limit, replace=False)
        selected, _ = train_test_split(idx, train_size=limit,
            random_state=cfg.seed, stratify=y[idx])
        return selected
    train_idx = subset(train_idx, labels, cfg.train_limit)
    val_idx = subset(val_idx, labels, cfg.val_limit)
    test_idx = subset(np.arange(len(x_test)), y_test, cfg.test_limit)
    x_train, y_train = images[train_idx], labels[train_idx]
    x_val, y_val = images[val_idx], labels[val_idx]
    x_test, y_test = x_test[test_idx], y_test[test_idx]
    calibration_idx = subset(np.arange(len(x_train)), y_train,
                             min(cfg.calibration_samples, len(x_train)))
    calibration_data = x_train[calibration_idx]
    data = (x_train, y_train, x_val, y_val, x_test, y_test, calibration_data)
    set_data(data)
    folder = cfg.output_dir / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "data_split.npz", train_indices=train_idx,
                        validation_indices=val_idx, test_indices=test_idx,
                        calibration_train_indices=calibration_idx, seed=cfg.seed)
    return data


def train_baseline_model():
    """Warm up the head, fine-tune all layers, restore the best validation model."""
    cfg = get_config()
    model = create_baseline_model()
    folder = cfg.output_dir / "checkpoints"
    folder.mkdir(parents=True, exist_ok=True)
    # Saved BEFORE optimization: transfer-learning initialization for lottery tickets.
    model.save_weights(str(folder / "baseline_initial.weights.h5"))
    start = time.perf_counter()
    histories = {}
    if cfg.imagenet_weights and cfg.warmup_epochs:
        for layer in model.layers:
            layer.trainable = layer.name in ("logits", "head_dropout")
        compile_model(model, 1e-3)
        warmup = model.fit(make_dataset("train", True), validation_data=make_dataset("val"),
                           epochs=cfg.warmup_epochs, verbose=cfg.verbose)
        histories["warmup"] = warmup.history
    for layer in model.layers:
        layer.trainable = True
    compile_model(model, 1e-4 if cfg.imagenet_weights else 1e-3)
    best_path = folder / "baseline_best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(str(best_path), monitor="val_accuracy",
            mode="max", save_best_only=True, save_weights_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=5,
            factor=0.5, min_lr=1e-7),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max",
            patience=15, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(str(folder / "baseline_training.csv")),
    ]
    history = model.fit(make_dataset("train", True), validation_data=make_dataset("val"),
                        epochs=cfg.baseline_epochs, callbacks=callbacks, verbose=cfg.verbose)
    model.load_weights(str(best_path))
    histories["finetune"] = history.history
    clone_model(model).save(cfg.output_dir / "baseline_model.keras")
    result = {"training_history": histories, "training_time_seconds": time.perf_counter() - start,
              "validation_accuracy": accuracy(model, "val"),
              "test_accuracy": accuracy(model, "test"), "quick_run": cfg.quick,
              "initialization": "ImageNet transfer" if cfg.imagenet_weights else "random"}
    result["baseline_target_met"] = result["test_accuracy"] > 0.75
    save_json(result, cfg.output_dir / "results" / "baseline_training.json")
    print(f"Baseline test accuracy: {result['test_accuracy']:.2%}; target >75%: "
          f"{result['baseline_target_met']}")
    return model


if __name__ == "__main__":
    from common import configure
    configure()
    prepare_compression_datasets()
    train_baseline_model()
