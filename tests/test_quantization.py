"""Small numerical checks; no CIFAR download or long training is required."""

from pathlib import Path

import numpy as np
import pytest

import common
from common import tf
from part2_quantization import (
    QuantizationPipeline, QuantizedConv2D, QuantizedDepthwiseConv2D,
    quantize_model, packed_weights, load_packed_model, export_tflite,
    tflite_predict, _ClipBinaryLatentWeights,
)


@pytest.fixture
def quantization_case(tmp_path):
    previous_config, previous_data = common.get_config(), common._DATA
    common.configure(output_dir=tmp_path, batch_size=4, finetune_epochs=1,
                     sensitivity_samples=4, calibration_samples=4, verbose=0)
    rng = np.random.default_rng(7)
    x = rng.integers(0, 256, size=(8, 32, 32, 3), dtype=np.uint8)
    y = np.arange(8, dtype=np.int32) % 3
    common.set_data((x, y, x[:4], y[:4], x[4:], y[4:], x[:4]))
    inputs = tf.keras.Input((32, 32, 3))
    h = tf.keras.layers.Rescaling(1 / 255)(inputs)
    h = tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu", name="stem")(h)
    h = tf.keras.layers.DepthwiseConv2D(3, padding="same", activation="relu", name="depthwise")(h)
    h = tf.keras.layers.Conv2D(6, 1, activation="relu", name="pointwise")(h)
    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    outputs = tf.keras.layers.Dense(3, name="classifier")(h)
    model = tf.keras.Model(inputs, outputs)
    yield model, x
    common.configure(previous_config)
    common.set_data(previous_data) if previous_data is not None else setattr(common, "_DATA", None)


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
def test_reference_kernels_have_finite_trainable_gradients(quantization_case, bits):
    model, x = quantization_case
    quantized = quantize_model(model, bits, calibration_data=x[:4])
    assert isinstance(quantized.get_layer("stem"), QuantizedConv2D)
    assert isinstance(quantized.get_layer("depthwise"), QuantizedDepthwiseConv2D)
    with tf.GradientTape() as tape:
        logits = quantized(x[:4].astype(np.float32), training=True)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            np.arange(4) % 3, logits, from_logits=True))
    gradients = tape.gradient(loss, quantized.trainable_weights)
    assert logits.shape == (4, 3)
    assert all(gradient is not None and np.isfinite(gradient.numpy()).all() for gradient in gradients)
    assert any(np.any(gradient.numpy() != 0) for gradient in gradients)


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
def test_packed_archive_preserves_reference_predictions(quantization_case, bits):
    model, x = quantization_case
    quantized = quantize_model(model, bits, calibration_data=x[:4])
    path = packed_weights(quantized, f"roundtrip_{bits}")
    restored = load_packed_model(path)
    np.testing.assert_allclose(quantized(x[:2]), restored(x[:2]), rtol=2e-5, atol=2e-5)
    assert Path(path).stat().st_size > 0


def test_keras_archive_reloads_custom_quantized_layers(quantization_case):
    model, x = quantization_case
    quantized = quantize_model(model, {"stem": 8, "depthwise": 2,
                                       "pointwise": 4, "classifier": 8},
                               calibration_data=x[:4])
    path = common.save_model(quantized, "mixed_roundtrip")
    restored = tf.keras.models.load_model(path, compile=False)
    np.testing.assert_allclose(quantized(x[:2]), restored(x[:2]), rtol=1e-6, atol=1e-6)


def test_full_integer_tflite_runs_with_integer_io(quantization_case):
    model, x = quantization_case
    path = export_tflite(model, "tiny_int8", calibration_data=x[:4])
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    assert interpreter.get_input_details()[0]["dtype"] == np.int8
    assert interpreter.get_output_details()[0]["dtype"] == np.int8
    predictions = tflite_predict(path, x[:4])
    assert predictions.shape == (4, 3)
    assert np.isfinite(predictions).all()


def test_sensitivity_includes_frozen_and_depthwise_kernels(quantization_case):
    model, _ = quantization_case
    model.get_layer("stem").trainable = False
    scores = QuantizationPipeline(model)._gradient_sensitivity()
    assert set(scores) == {"stem", "depthwise", "pointwise", "classifier"}
    assert all(np.isfinite(score) and score >= 0 for score in scores.values())
    assert scores["stem"] > 0


def test_invalid_precision_is_rejected(quantization_case):
    model, _ = quantization_case
    with pytest.raises(ValueError, match="bit widths"):
        quantize_model(model, bits=3)


def test_packer_rejects_non_finite_quantized_weights(quantization_case):
    model, x = quantization_case
    quantized = quantize_model(model, 1, calibration_data=x[:4])
    kernel = quantized.get_layer("stem").kernel
    damaged = kernel.numpy()
    damaged.flat[0] = np.nan
    kernel.assign(damaged)
    with pytest.raises(ValueError, match="non-finite"):
        packed_weights(quantized, "damaged_binary")


def test_binary_training_guard_clips_latent_weights(quantization_case):
    model, x = quantization_case
    quantized = quantize_model(model, 1, calibration_data=x[:4])
    kernel = quantized.get_layer("stem").kernel
    kernel.assign(tf.fill(kernel.shape, 3.0))
    callback = _ClipBinaryLatentWeights()
    callback.set_model(quantized)
    callback.on_train_batch_end(0)
    assert np.isfinite(kernel.numpy()).all()
    assert np.max(np.abs(kernel.numpy())) <= 1.0


@pytest.mark.parametrize("mode", ["float32", "dynamic"])
def test_float_tflite_residual_graph_executes(quantization_case, mode):
    _, x = quantization_case
    inputs = tf.keras.Input((32, 32, 3))
    values = tf.keras.layers.Rescaling(1 / 255)(inputs)
    residual = tf.keras.layers.Conv2D(3, 3, padding="same")(values)
    values = tf.keras.layers.Add()([values, residual])
    values = tf.keras.layers.GlobalAveragePooling2D()(values)
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(3)(values))
    path = export_tflite(model, f"residual_{mode}", mode=mode)
    predictions = tflite_predict(path, x[:2])
    assert predictions.shape == (2, 3)
    assert np.isfinite(predictions).all()
