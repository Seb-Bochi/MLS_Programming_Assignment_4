"""Small numerical tests for distillation's training semantics (no downloads)."""
import numpy as np
import pytest

from common import tf, configure, set_data
from part3_distillation import _Distiller, distill_student


def tiny_model(width=1, batch_norm=False):
    inputs = tf.keras.Input((32, 32, 3))
    x = tf.keras.layers.Rescaling(1 / 255)(inputs)
    for index, channels in enumerate((2, 3, 4), 1):
        x = tf.keras.layers.Conv2D(channels * width, 1, activation="relu",
                                   name=f"stage{index}_features")(x)
        if batch_norm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.AveragePooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(3)(x)
    return tf.keras.Model(inputs, outputs)


@pytest.fixture
def data(tmp_path):
    configure(output_dir=tmp_path, batch_size=4, distill_epochs=1, verbose=0,
              sensitivity_samples=4, quick=True, imagenet_weights=False)
    rng = np.random.default_rng(123)
    images = rng.integers(0, 256, size=(8, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(8, dtype=np.int32) % 3
    set_data((images, labels, images[:4], labels[:4], images[:4], labels[:4], images[:4]))
    return images, labels


def test_temperature_squared_and_teacher_unchanged(data):
    images, labels = data
    teacher, student = tiny_model(2, batch_norm=True), tiny_model()
    teacher_before = [value.copy() for value in teacher.get_weights()]
    student_before = [value.copy() for value in student.get_weights()]
    temperature = 4.0
    q = tf.nn.softmax(teacher(images, training=False) / temperature)
    log_p = tf.nn.log_softmax(student(images, training=True) / temperature)
    expected_kl = tf.reduce_mean(tf.reduce_sum(
        q * (tf.math.log(tf.maximum(q, 1e-8)) - log_p), axis=-1
    )) * temperature ** 2
    trainer = _Distiller([teacher], student, temperature, 0.0, "response", 0.0)
    trainer.compile(optimizer=tf.keras.optimizers.SGD(0.01))
    metrics = trainer.train_step((images, labels))
    np.testing.assert_allclose(metrics["distillation_loss"].numpy(), expected_kl.numpy(),
                               atol=1e-6, rtol=1e-5)
    for before, after in zip(teacher_before, teacher.get_weights()):
        np.testing.assert_array_equal(before, after)
    assert any(not np.array_equal(before, after)
               for before, after in zip(student_before, student.get_weights()))
    assert all(np.isfinite(float(value)) for value in metrics.values())


@pytest.mark.parametrize("mode", ["feature", "attention"])
def test_spatial_transfer_and_adapter_gradients(data, mode):
    images, labels = data
    teacher, student = tiny_model(2), tiny_model()
    trainer = _Distiller([teacher], student, 2.0, 0.5, mode, 10.0)
    trainer.compile(optimizer=tf.keras.optimizers.Adam(0.01))
    teacher_before = [value.copy() for value in teacher.get_weights()]
    adapters_before = [value.numpy().copy() for adapter in trainer.adapters for value in adapter.weights]
    result = trainer.train_step((images, labels))
    assert np.isfinite(float(result["transfer_loss"]))
    for before, after in zip(teacher_before, teacher.get_weights()):
        np.testing.assert_array_equal(before, after)
    if mode == "feature":
        after = [value.numpy() for adapter in trainer.adapters for value in adapter.weights]
        assert adapters_before
        assert any(not np.array_equal(left, right) for left, right in zip(adapters_before, after))


def test_one_epoch_distillation_exports_student_only(data):
    teacher, student = tiny_model(2), tiny_model()
    result = distill_student(teacher, student, epochs=1, name="test_student")
    assert result["model"] is student
    assert 0 <= result["final_accuracy"] <= 1
    assert len(result["training_history"]["val_accuracy"]) == 1
    loaded = tf.keras.models.load_model(result["artifact_path"])
    assert loaded.count_params() == student.count_params()
    assert all("adapter" not in layer.name for layer in loaded.layers)
    np.testing.assert_allclose(loaded(data[0], training=False).numpy(),
                               student(data[0], training=False).numpy(), atol=1e-6)


@pytest.mark.parametrize("kwargs", [{"temperature": 0}, {"alpha": 2}, {"mode": "unknown"}])
def test_invalid_distillation_options(data, kwargs):
    with pytest.raises(ValueError):
        distill_student(tiny_model(2), tiny_model(), **kwargs)
