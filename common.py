"""Small shared utilities; experiment classes keep the assignment's public API."""
from __future__ import annotations

import json
import os
import random
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_HOME", str(Path(__file__).parent / ".keras"))
import numpy as np
import psutil
import tensorflow as tf


@dataclass
class ExperimentConfig:
    seed: int = 42
    batch_size: int = 64
    baseline_epochs: int = 100
    warmup_epochs: int = 5
    finetune_epochs: int = 10
    distill_epochs: int = 30
    output_dir: Path = Path(".")
    quick: bool = False
    sensitivity_samples: int = 256
    calibration_samples: int = 1000
    verbose: int = 2
    imagenet_weights: bool = True
    image_size: int = 96
    benchmark_repeats: int = 20
    train_limit: int | None = None
    val_limit: int | None = None
    test_limit: int | None = None


_CONFIG = ExperimentConfig()
_DATA = None


def get_config():
    return _CONFIG


def configure(config=None, **kwargs):
    global _CONFIG, _DATA
    _CONFIG = config or ExperimentConfig(**kwargs)
    _CONFIG.output_dir = Path(_CONFIG.output_dir)
    for name in ("", "checkpoints", "compressed_models", "results"):
        (_CONFIG.output_dir / name).mkdir(parents=True, exist_ok=True)
    _DATA = None
    set_seed(_CONFIG.seed)
    return _CONFIG


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    if sys.version_info >= (3, 12):
        # tf-keras 2.16 passes 1e9 (float) to Random.randint. Python 3.12
        # disallows that. Restrict the compatibility fix to Keras's own RNG;
        # do not monkey-patch Python random or edit installed dependencies.
        import tf_keras
        if tf_keras.__version__ == "2.16.0":
            from tf_keras.src import backend as legacy_backend
            class IntegerSeedRandom(random.Random):
                def randint(self, a, b):
                    return super().randint(int(a), int(b))
            legacy_backend._SEED_GENERATOR.generator = IntegerSeedRandom(seed)


def set_data(data):
    """Inject arrays for tests, or cache the stratified CIFAR-100 split."""
    global _DATA
    _DATA = tuple(data)


def get_data():
    if _DATA is None:
        from baseline_model import prepare_compression_datasets
        prepare_compression_datasets()
    return _DATA


def split_data(split="val"):
    indexes = {"train": (0, 1), "val": (2, 3), "test": (4, 5)}
    if split not in indexes:
        raise ValueError("split must be train, val, or test")
    data = get_data()
    i, j = indexes[split]
    return data[i], data[j]


def make_dataset(split="train", augment=False):
    x, y = split_data(split)
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if split == "train":
        ds = ds.shuffle(min(len(x), 10000), seed=get_config().seed)
    def transform(image, label):
        image = tf.cast(image, tf.float32)
        if augment:
            image = tf.pad(image, [[4, 4], [4, 4], [0, 0]], mode="REFLECT")
            image = tf.image.random_crop(image, [32, 32, 3])
            image = tf.image.random_flip_left_right(image)
        return image, tf.cast(label, tf.int32)
    ds = ds.map(transform, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(get_config().batch_size).prefetch(tf.data.AUTOTUNE)


def compile_model(model, learning_rate=1e-4, clipnorm=None):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate, clipnorm=clipnorm),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


def clone_model(model):
    cloned = tf.keras.models.clone_model(model)
    cloned.set_weights(model.get_weights())
    return cloned


def safe_name(name):
    import hashlib
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))
    return value if len(value) <= 90 else value[:78] + "_" + hashlib.sha1(value.encode()).hexdigest()[:10]


def fit_model(model, epochs=None, learning_rate=1e-4, callbacks=None, name=None,
              clipnorm=None):
    cfg = get_config()
    epochs = cfg.finetune_epochs if epochs is None else epochs
    if epochs < 1:
        raise ValueError("epochs must be positive")
    compile_model(model, learning_rate, clipnorm=clipnorm)
    extra = list(callbacks or [])
    if name:
        folder = cfg.output_dir / "checkpoints"
        folder.mkdir(parents=True, exist_ok=True)
        extra += [tf.keras.callbacks.ModelCheckpoint(
            str(folder / f"{safe_name(name)}.weights.h5"),
            monitor="val_accuracy", save_best_only=True, save_weights_only=True)]
    extra += [tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7)]
    start = time.perf_counter()
    history = model.fit(make_dataset("train", augment=True),
                        validation_data=make_dataset("val"), epochs=epochs,
                        callbacks=extra, verbose=cfg.verbose)
    result = {"history": history.history,
              "training_time_seconds": time.perf_counter() - start}
    if name:
        save_json(result, cfg.output_dir / "results" / f"{safe_name(name)}_history.json")
    return result


def accuracy(model, split="val"):
    correct = total = 0
    for x, y in make_dataset(split):
        logits = model(x, training=False)
        correct += int(tf.reduce_sum(tf.cast(
            tf.argmax(logits, axis=-1, output_type=tf.int32) == tf.reshape(y, [-1]),
            tf.int32)).numpy())
        total += len(y)
    return correct / total


def model_weight_bytes(model):
    return int(sum(w.numpy().nbytes for w in model.weights))


def make_lite_interpreter(**kwargs):
    """Use a stable CPU interpreter for the pinned Windows TensorFlow build.

    Its default XNNPACK delegate can access-violate on float residual graphs.
    Built-in kernels retain native INT8 execution without that delegate.
    """
    options = {"num_threads": 1, **kwargs}
    if os.name == "nt":
        options["experimental_op_resolver_type"] = (
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    return tf.lite.Interpreter(**options)


def save_model(model, name):
    folder = get_config().output_dir / "compressed_models"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{safe_name(name).removesuffix('.keras')}.keras"
    # A fresh clone has no optimizer slots, allowing fair inference artifact sizes.
    clone_model(model).save(path)
    return str(path)


def json_ready(value):
    if isinstance(value, tf.keras.Model):
        return {"model_name": value.name, "parameters": value.count_params()}
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"byte_count": len(value)}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2, allow_nan=False), encoding="utf-8")
    return str(path)


class MemoryMonitor:
    """Sample process RSS; this includes the Python runtime and retained models."""
    def __enter__(self):
        self.process = psutil.Process()
        self.start_bytes = self.peak_bytes = self.process.memory_info().rss
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(0.01):
                self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak_bytes = max(self.peak_bytes, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def environment_metadata():
    import platform
    return {"python": platform.python_version(), "tensorflow": tf.__version__,
            "platform": platform.platform(), "devices": [str(x) for x in tf.config.list_physical_devices()],
            "config": asdict(get_config()), "size_units": "MiB (2**20 bytes)",
            "accuracy_units": "fraction; losses additionally in percentage points"}
