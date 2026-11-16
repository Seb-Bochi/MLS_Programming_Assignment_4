"""Constraint handling and honest hardware/energy reporting."""
from application_scenarios import ApplicationScenarioAnalysis, generate_compression_guidelines


def record(name, accuracy=0.73, size=4.0, latency=10.0, loss=3.0, throughput=500):
    return {"model_name": name, "test_accuracy": accuracy, "model_size_mb": size,
            "single_inference_ms": latency, "accuracy_loss_percent": loss,
            "peak_memory_mb": 120.0, "throughput_images_per_second": throughput,
            "batch_size": 8, "batch_inference_ms": 16.0,
            "energy_consumption": None}


def test_scenarios_reject_infeasible_candidates():
    scenarios = ApplicationScenarioAnalysis({
        "too_large": record("too_large", size=100),
        "too_inaccurate": record("too_inaccurate", loss=7),
    })
    choice = scenarios.mobile_deployment_optimization()["optimal_compression_pipeline"]
    assert choice["model_name"] is None
    assert len(choice["rejected_candidates"]) == 2


def test_latency_and_accuracy_units_and_cloud_throughput():
    scenarios = ApplicationScenarioAnalysis({
        "accurate": record("accurate", accuracy=0.75, loss=1, throughput=400),
        "fast": record("fast", accuracy=0.72, loss=4, throughput=900),
        "loss_exceeds_five_pp": record("loss_exceeds_five_pp", accuracy=0.70, loss=6, throughput=2000),
    })
    mobile = scenarios.mobile_deployment_optimization()
    assert mobile["optimal_compression_pipeline"]["model_name"] == "accurate"
    assert mobile["battery_life_impact"]["host_energy_estimate_mj"] is None
    cloud = scenarios.cloud_inference_optimization()
    assert cloud["throughput_optimization"]["model_name"] == "fast"
    assert cloud["cost_benefit_analysis"]["estimated_cost_per_million_inferences"] is None


def test_energy_estimate_conversion_and_optional_cost():
    metrics = record("candidate")
    metrics["energy_consumption"] = 0.002
    scenarios = ApplicationScenarioAnalysis({"candidate": metrics}, cloud_hourly_cost=1.8)
    mobile = scenarios.mobile_deployment_optimization()
    assert mobile["battery_life_impact"]["host_energy_estimate_mj"] == 2.0
    cloud = scenarios.cloud_inference_optimization()
    assert cloud["cost_benefit_analysis"]["estimated_cost_per_million_inferences"] == 1.0


def test_guidelines_do_not_claim_measured_success(tmp_path):
    output = tmp_path / "guidelines.md"
    result = generate_compression_guidelines(output_path=output)
    text = output.read_text(encoding="utf-8")
    assert set(result) == {"mobile_deployment", "edge_computing", "cloud_inference"}
    assert "not guaranteed outcomes" in text
    assert "percentage points" in text
