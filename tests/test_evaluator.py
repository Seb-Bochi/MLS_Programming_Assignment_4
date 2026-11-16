import numpy as np
import pytest
from common import tf, configure, set_data
from compression_evaluator import CompressionEvaluator, estimate_flops, expected_calibration_error
from part4_interaction_analysis import pareto_frontier


def test_benchmark_uses_real_size_and_baseline(tmp_path):
    configure(output_dir=tmp_path, batch_size=2, benchmark_repeats=2, verbose=0)
    inputs = tf.keras.Input((4,))
    model = tf.keras.Model(inputs, tf.keras.layers.Dense(2, use_bias=False)(inputs))
    model.layers[-1].set_weights([np.array([[1, 0], [0, 1], [0, 0], [0, 0]], np.float32)])
    images = np.array([[2, 0, 0, 0], [0, 2, 0, 0]], np.float32)
    evaluator = CompressionEvaluator()
    metrics = evaluator.benchmark_compressed_model(model, (images, np.array([0, 1])), "baseline")
    assert metrics["test_accuracy"] == 1
    assert metrics["compression_ratio"] == 1
    assert metrics["energy_consumption"] is None
    assert metrics["model_size_mb"] > 0
    assert metrics["single_inference_ms"] > 0
    assert metrics["flops"] == 16
    assert metrics["accuracy_loss_percent"] == 0


def test_calibration_and_pareto_ties():
    probabilities = np.array([[1., 0.], [0., 1.]])
    assert expected_calibration_error(probabilities, [0, 1]) == 0
    rows = {"a": {"validation_accuracy": .8, "model_size_mb": 5},
            "tie": {"validation_accuracy": .8, "model_size_mb": 5},
            "bad": {"validation_accuracy": .7, "model_size_mb": 6},
            "small": {"validation_accuracy": .6, "model_size_mb": 1}}
    assert set(pareto_frontier(rows)) == {"a", "tie", "small"}


def test_seed_summary_paired_comparison(tmp_path):
    configure(output_dir=tmp_path)
    evaluator = CompressionEvaluator()
    result = evaluator.cross_validate_compression(lambda seed: {
        "final_accuracy": .8 + (seed % 3) * .001, "baseline_accuracy": .75}, 3)
    assert result["mean_accuracy"] == pytest.approx(.801)
    assert result["std_accuracy"] > 0
    assert result["significance"]["pvalue"] < .05
