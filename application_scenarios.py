"""Select compression candidates under explicit application budgets.

Latency and RSS are measurements of the benchmark host. They are screening
proxies, not measurements of a phone, embedded board, or production server.
"""
from __future__ import annotations

import math
from pathlib import Path


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class ApplicationScenarioAnalysis:
    def __init__(self, compressed_models, mobile_size_mb=10.0, mobile_latency_ms=33.0,
                 edge_size_mb=20.0, edge_latency_ms=20.0, edge_memory_mb=256.0,
                 max_accuracy_loss_pp=5.0, cloud_hourly_cost=None):
        self.compressed_models = compressed_models
        self.budgets = {
            "mobile_size_mb": float(mobile_size_mb), "mobile_latency_ms": float(mobile_latency_ms),
            "edge_size_mb": float(edge_size_mb), "edge_latency_ms": float(edge_latency_ms),
            "edge_memory_mb": float(edge_memory_mb), "max_accuracy_loss_pp": float(max_accuracy_loss_pp),
        }
        if any(value <= 0 or not math.isfinite(value) for value in self.budgets.values()):
            raise ValueError("Application budgets must be positive finite values.")
        if cloud_hourly_cost is not None and (cloud_hourly_cost < 0 or not math.isfinite(cloud_hourly_cost)):
            raise ValueError("cloud_hourly_cost must be finite and nonnegative.")
        self.cloud_hourly_cost = cloud_hourly_cost
        self.records = {}
        for name, value in compressed_models.items():
            if not isinstance(value, dict):
                continue
            metrics = value.get("metrics", value)
            if isinstance(metrics, dict) and _number(metrics.get("test_accuracy")):
                self.records[str(name)] = dict(metrics, model_name=metrics.get("model_name", str(name)))
        self.measurement_caveat = (
            "Host latency, throughput and process RSS are screening proxies. Re-benchmark the exact "
            "exported artifact with the target device/runtime; fake-quantized floating-point graphs "
            "do not establish native INT4/binary speed or memory benefits."
        )

    @staticmethod
    def _throughput(metrics):
        for key in ("throughput_images_per_second", "throughput_samples_per_second", "throughput_samples_per_sec", "throughput"):
            value = metrics.get(key)
            if _number(value) and value > 0:
                return float(value)
        batch_size, duration = metrics.get("batch_size"), metrics.get("batch_inference_ms")
        if _number(batch_size) and _number(duration) and batch_size > 0 and duration > 0:
            return float(batch_size) * 1000.0 / float(duration)
        return None

    def _selection(self, constraints, objective):
        candidates, rejected = {}, {}
        for name, metrics in self.records.items():
            reasons = []
            for metric, maximum in constraints.items():
                observed = metrics.get(metric)
                if not _number(observed):
                    reasons.append(f"Missing finite {metric}")
                elif observed > maximum:
                    reasons.append(f"{metric}={observed:.4g} exceeds {maximum:.4g}")
            if objective == "throughput" and self._throughput(metrics) is None:
                reasons.append("Missing measured batch throughput")
            if reasons:
                rejected[name] = reasons
            else:
                candidates[name] = metrics
        if objective == "throughput":
            ordered = sorted(candidates, key=lambda name: (
                -self._throughput(candidates[name]), -candidates[name]["test_accuracy"]
            ))
        else:
            ordered = sorted(candidates, key=lambda name: (
                -candidates[name]["test_accuracy"], candidates[name].get("single_inference_ms", math.inf),
                candidates[name].get("model_size_mb", math.inf),
            ))
        best = ordered[0] if ordered else None
        return {
            "model_name": best,
            "metrics": candidates[best] if best is not None else {},
            "feasible_candidates": ordered,
            "rejected_candidates": rejected,
            "constraints": constraints,
            "objective": "highest throughput within budgets" if objective == "throughput"
                         else "highest accuracy within budgets, then lowest latency and size",
            "status": "candidate_selected" if best is not None else "no_measured_candidate_meets_all_budgets",
            "measurement_caveat": self.measurement_caveat,
        }

    def mobile_deployment_optimization(self):
        """Screen for app footprint and response time, then maximize accuracy."""
        selection = self._selection({
            "model_size_mb": self.budgets["mobile_size_mb"],
            "single_inference_ms": self.budgets["mobile_latency_ms"],
            "accuracy_loss_percent": self.budgets["max_accuracy_loss_pp"],
        }, "accuracy")
        metrics = selection["metrics"]
        energy_joules = metrics.get("energy_consumption")
        energy = float(energy_joules) * 1000 if _number(energy_joules) else None
        return {
            "optimal_compression_pipeline": selection,
            "battery_life_impact": {
                "status": "not_measured_on_mobile_hardware",
                "host_energy_estimate_mj": energy if _number(energy) else None,
                "energy_method": metrics.get("energy_method"),
                "recommendation": "Measure device joules per request and duty cycle; neither file size nor host latency predicts battery lifetime.",
            },
            "thermal_considerations": {
                "status": "not_measured",
                "recommendation": "Run sustained target-device workloads and monitor throttling, temperature and p95 latency.",
            },
            "user_experience_metrics": {
                "host_single_inference_ms": metrics.get("single_inference_ms"),
                "artifact_size_mib": metrics.get("model_size_mb"),
                "response_budget_ms": self.budgets["mobile_latency_ms"],
                "app_model_budget_mib": self.budgets["mobile_size_mb"],
                "runtime_checks": [
                    "Confirm CPU/GPU/NPU delegate support for every exported operator.",
                    "Measure startup, preprocessing and end-to-end p95 response time separately.",
                ],
            },
        }

    def edge_device_optimization(self):
        """Filter size, host process RSS, latency and accuracy constraints."""
        selection = self._selection({
            "model_size_mb": self.budgets["edge_size_mb"],
            "single_inference_ms": self.budgets["edge_latency_ms"],
            "peak_memory_mb": self.budgets["edge_memory_mb"],
            "accuracy_loss_percent": self.budgets["max_accuracy_loss_pp"],
        }, "accuracy")
        return {
            "memory_constrained_optimization": selection,
            "real_time_performance": {
                "host_single_inference_ms": selection["metrics"].get("single_inference_ms"),
                "deadline_ms": self.budgets["edge_latency_ms"],
                "hard_real_time_guarantee": False,
                "required_validation": "Measure worst-case/p95 latency including I/O and preprocessing on the target device.",
            },
            "device_scalability": {
                "selection_scope": "The measured host only; device rankings can change with operator support.",
                "required_profiles": ["CPU-only", "integer accelerator when available", "memory-constrained runtime"],
            },
            "deployment_considerations": {
                "offline_ready": "Exported weights and preprocessing must be packaged locally; verify operator availability without downloads.",
                "memory_caveat": "peak_memory_mb is sampled process RSS, including Python/runtime and retained models; it is not isolated tensor-arena memory.",
                "size_units": "MiB (2**20 bytes)",
                "accuracy_loss_units": "percentage points",
            },
        }

    def cloud_inference_optimization(self):
        """Maximize measured batch throughput under an accuracy-loss budget."""
        selection = self._selection({
            "accuracy_loss_percent": self.budgets["max_accuracy_loss_pp"],
        }, "throughput")
        metrics = selection["metrics"]
        throughput = self._throughput(metrics)
        single_ms, batch_size, batch_ms = (
            metrics.get("single_inference_ms"), metrics.get("batch_size"), metrics.get("batch_inference_ms")
        )
        batching_speedup = None
        if all(_number(value) and value > 0 for value in (single_ms, batch_size, batch_ms)):
            batching_speedup = float(single_ms * batch_size / batch_ms)
        estimated_cost = None
        if throughput and self.cloud_hourly_cost is not None:
            estimated_cost = self.cloud_hourly_cost * 1_000_000 / (throughput * 3600)
        return {
            "throughput_optimization": selection,
            "batch_processing_efficiency": {
                "throughput_samples_per_second": throughput,
                "batch_size": batch_size,
                "batch_latency_ms": batch_ms,
                "speedup_over_serial_single_inference": batching_speedup,
                "caveat": "Measured inference only; batching queue delay and production concurrency are not included.",
            },
            "cost_benefit_analysis": {
                "hourly_instance_cost": self.cloud_hourly_cost,
                "estimated_cost_per_million_inferences": estimated_cost,
                "assumptions": "When a cost is supplied: full utilization at measured throughput; excludes I/O, orchestration and training amortization.",
            },
            "serving_infrastructure_impact": {
                "measurement_caveat": self.measurement_caveat,
                "recommended_checks": [
                    "Sweep batch size and concurrency on the production accelerator.",
                    "Compare dense, structured and integer kernels using the actual serving runtime.",
                    "Retain preprocessing and class order with the model artifact.",
                ],
            },
        }


def generate_compression_guidelines(evaluation_results=None, output_path=None):
    """Write practical qualitative guidance, optionally attaching measured picks.

    This is the requested guidelines deliverable, not a technical/LaTeX report.
    No universal accuracy, energy, latency or compression gain is assumed.
    """
    guidelines = {
        "mobile_deployment": {
            "priority_metrics": ["model_size", "inference_latency"],
            "recommended_techniques": ["compact student distillation", "structured channel pruning", "native INT8 PTQ or QAT"],
            "technique_ordering": ["choose student architecture", "distill/fine-tune", "structured prune if needed", "fine-tune", "calibrate and quantize", "validate on device"],
            "expected_trade_offs": {
                "accuracy": "Select temperature, width and pruning strength on validation data; use QAT if INT8 PTQ loses too much accuracy.",
                "size": "Report actual exported bytes. Packed INT4/binary weights are storage experiments unless a compatible execution backend is supplied.",
                "latency": "Native operator/delegate support and activation traffic determine gains; sparse weights alone do not reduce dense-kernel work.",
                "battery": "Measure device energy and duty cycle before estimating battery impact.",
            },
        },
        "edge_computing": {
            "priority_metrics": ["energy_consumption", "memory_usage"],
            "recommended_techniques": ["small student architecture search", "structured pruning", "full-integer quantization"],
            "technique_ordering": ["set flash/RAM/deadline budgets", "search compact students", "distill", "prune and fine-tune", "calibrate INT8", "profile target runtime"],
            "expected_trade_offs": {
                "memory": "Measure activation/tensor-arena peak separately from model file size and Python process RSS.",
                "latency": "Validate deadline tails under sustained load; a mean host latency is not a real-time guarantee.",
                "portability": "Package preprocessing and supported operators for offline operation; unsupported operators may force a slower fallback.",
            },
        },
        "cloud_inference": {
            "priority_metrics": ["throughput", "accuracy"],
            "recommended_techniques": ["distillation", "hardware-aligned structured pruning", "backend-supported low precision"],
            "technique_ordering": ["choose serving batch/concurrency", "distill", "compare pruning and quantization orders", "benchmark actual backend", "select a Pareto-efficient candidate"],
            "expected_trade_offs": {
                "throughput": "Choose from measured batch throughput subject to an accuracy budget; smaller files do not imply higher throughput.",
                "cost": "Amortize extra compression training and evaluate serving utilization before claiming cost savings.",
                "accuracy": "Use the same dataset split and multiple seeds; do not choose hyperparameters using held-out test accuracy.",
            },
        },
    }
    if evaluation_results:
        scenarios = ApplicationScenarioAnalysis(evaluation_results)
        guidelines["mobile_deployment"]["measured_selection"] = scenarios.mobile_deployment_optimization()["optimal_compression_pipeline"]
        guidelines["edge_computing"]["measured_selection"] = scenarios.edge_device_optimization()["memory_constrained_optimization"]
        guidelines["cloud_inference"]["measured_selection"] = scenarios.cloud_inference_optimization()["throughput_optimization"]
    if output_path is None:
        from common import get_config
        output_path = Path(get_config().output_dir) / "compression_guidelines.md"
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Compression technique selection guidelines", "",
        "Use validation data to choose hyperparameters and report held-out test results only after selection. "
        "The assignment targets are >75% baseline test accuracy and at least 10x actual artifact compression "
        "with less than 5 percentage points of accuracy loss; these are measured goals, not guaranteed outcomes.", "",
        "All size metrics use MiB (2**20 bytes). `accuracy_loss_percent` denotes percentage points. "
        "Host measurements screen candidates; final mobile, edge and cloud recommendations require the intended hardware/runtime.", "",
    ]
    for key, scenario in guidelines.items():
        lines.extend([f"## {key.replace('_', ' ').title()}", "",
                      "Priorities: " + ", ".join(scenario["priority_metrics"]) + ".", "",
                      "Candidate techniques: " + "; ".join(scenario["recommended_techniques"]) + ".", "",
                      "Suggested sequence to test: " + " → ".join(scenario["technique_ordering"]) + ".", ""])
        for metric, text in scenario["expected_trade_offs"].items():
            lines.append(f"- **{metric.title()}:** {text}")
        lines.append("")
        if "measured_selection" in scenario:
            selection = scenario["measured_selection"]
            lines.extend([
                "Measured host candidate: " + (f"`{selection['model_name']}`." if selection["model_name"]
                                               else "none meets every configured budget."),
                "", f"Selection rule: {selection['objective']}.", "",
            ])
    lines.extend([
        "## Interaction and failure checks", "",
        "- Compare compression orderings at a common starting checkpoint and record each stage's training cost.",
        "- Recalibrate after pruning or distillation; previously estimated activation ranges can become stale.",
        "- Keep pruning masks enforced during fine-tuning; otherwise zeroed weights can regrow.",
        "- Inspect layer sensitivity before applying severe pruning or low-bit quantization to every layer.",
        "- Treat INT4 and XNOR-style floating-point reference kernels as algorithm studies; report their measured runtime and packed storage separately.",
        "- Report accuracy, confidence calibration and corruption robustness together; a compact network can preserve accuracy while becoming overconfident.",
        "- Count preprocessing, calibration provenance, initialization and random seeds as part of reproducibility.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")
    return guidelines
