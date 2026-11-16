"""Small, offline checks for pruning invariants and residual graph surgery."""
import numpy as np
import pytest

from common import tf
from part1_pruning import (magnitude_prune, FixedMaskCallback, MagnitudePruningCallback,
                           structured_prune_model, layer_sparsity, _global_sparsity)


def _dense_model():
    inputs = tf.keras.Input((4,))
    outputs = tf.keras.layers.Dense(2, use_bias=False, name="dense")(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.get_layer("dense").set_weights([np.arange(1, 9, dtype=np.float32).reshape(4, 2)])
    return model


def _bottleneck_model():
    inputs = tf.keras.Input((8, 8, 8))
    x = tf.keras.layers.Conv2D(4, 1, name="conv2_block1_1_conv")(inputs)
    x = tf.keras.layers.BatchNormalization(name="conv2_block1_1_bn")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(4, 3, padding="same", name="conv2_block1_2_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="conv2_block1_2_bn")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(8, 1, name="conv2_block1_3_conv")(x)
    x = tf.keras.layers.BatchNormalization(name="conv2_block1_3_bn")(x)
    x = tf.keras.layers.Add()([inputs, x])
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(3)(x))


def test_exact_global_mask_and_enforcement():
    model = _dense_model()
    masks = magnitude_prune(model, 0.5)
    np.testing.assert_array_equal(model.get_layer("dense").kernel.numpy().reshape(-1)[:4], 0)
    assert _global_sparsity(layer_sparsity(model)) == 0.5
    callback = FixedMaskCallback(masks)
    callback.set_model(model)
    model.compile(optimizer="sgd", loss="mse")
    model.train_on_batch(np.ones((2, 4), np.float32), np.zeros((2, 2), np.float32))
    callback.on_train_batch_end(0)
    np.testing.assert_array_equal(model.get_layer("dense").kernel.numpy().reshape(-1)[:4], 0)
    with pytest.raises(ValueError):
        magnitude_prune(model, 0.25, masks)


@pytest.mark.parametrize("schedule", ["constant", "polynomial", "gradual"])
def test_schedule_reaches_requested_sparsity(schedule):
    model = _dense_model()
    callback = MagnitudePruningCallback(0.75, 3, schedule)
    callback.set_model(model)
    for epoch in range(3):
        callback.on_epoch_begin(epoch)
    assert _global_sparsity(layer_sparsity(model)) == 0.75
    assert callback.sparsity_history[-1] == 0.75


@pytest.mark.parametrize("metric", ["l1_norm", "l2_norm"])
def test_structured_channels_and_residual_dependency(metric):
    model = _bottleneck_model()
    pruned, removed, changes = structured_prune_model(model, 0.5, metric)
    assert pruned.count_params() < model.count_params()
    assert pruned(np.ones((2, 8, 8, 8), np.float32)).shape == (2, 3)
    assert removed["conv2_block1_1_conv"] == 2
    first = changes["retained_filter_indices"]["conv2_block1_1_conv"]
    second = changes["retained_filter_indices"]["conv2_block1_2_conv"]
    expected = np.take(np.take(model.get_layer("conv2_block1_2_conv").kernel.numpy(), first, 2), second, 3)
    np.testing.assert_array_equal(pruned.get_layer("conv2_block1_2_conv").kernel.numpy(), expected)
    assert pruned.get_layer("conv2_block1_3_conv").filters == 8


def test_zero_structured_pruning_preserves_predictions():
    model = _bottleneck_model()
    pruned, _, _ = structured_prune_model(model, 0.0)
    x = np.ones((1, 8, 8, 8), np.float32)
    np.testing.assert_allclose(model(x).numpy(), pruned(x).numpy(), atol=1e-6)


@pytest.mark.parametrize("metric", ["gradient", "taylor"])
def test_gradient_ranking_uses_connected_validation_gradients(monkeypatch, metric):
    import common
    rng = np.random.default_rng(7)
    x = rng.normal(size=(4, 8, 8, 8)).astype(np.float32)
    y = np.array([0, 1, 2, 0], np.int32)
    monkeypatch.setattr(common, "_DATA", (x, y, x, y, x, y, x))
    monkeypatch.setattr(common.get_config(), "sensitivity_samples", 4)
    monkeypatch.setattr(common.get_config(), "batch_size", 2)
    model = _bottleneck_model()
    pruned, _, _ = structured_prune_model(model, 0.5, metric)
    assert np.isfinite(pruned(x).numpy()).all()
    assert pruned.count_params() < model.count_params()


def test_pareto_duplicates_and_dominated_points():
    from part4_interaction_analysis import pareto_frontier
    rows = {"a": {"validation_accuracy": .8, "model_size_mb": 2},
            "b": {"validation_accuracy": .7, "model_size_mb": 3},
            "c": {"validation_accuracy": .9, "model_size_mb": 4},
            "d": {"validation_accuracy": .8, "model_size_mb": 2}}
    assert set(pareto_frontier(rows)) == {"a", "c", "d"}


def test_distillation_mask_is_bound_to_student_only():
    from part4_interaction_analysis import _StudentMask
    teacher, student = _dense_model(), _dense_model()
    original_teacher = teacher.get_weights()
    masks = magnitude_prune(student, 0.5)
    trainer = tf.keras.Model()
    trainer.teacher = teacher
    trainer.student = student
    callback = _StudentMask(student, masks)
    callback.set_model(trainer)
    callback.enforce()
    for original, actual in zip(original_teacher, teacher.get_weights()):
        np.testing.assert_array_equal(original, actual)
    assert _global_sparsity(layer_sparsity(student)) == 0.5
