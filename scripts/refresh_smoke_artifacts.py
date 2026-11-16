"""Rebenchmark existing smoke models after evaluator changes; no retraining."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from common import configure, tf, save_json, set_data
import part2_quantization  # register custom saved layers
from compression_evaluator import CompressionEvaluator
from visualization_tools import export_benchmarks, create_compression_visualizations
from application_scenarios import ApplicationScenarioAnalysis, generate_compression_guidelines

folder = Path("outputs/smoke")
metadata = json.loads((folder / "results/environment.json").read_text())
cfg = configure(**metadata["config"])
rng = np.random.default_rng(cfg.seed)
images = rng.integers(0, 256, (24, 32, 32, 3), dtype=np.uint8)
labels = np.arange(24, dtype=np.int32) % 100
set_data((images[:12], labels[:12], images[12:18], labels[12:18], images[18:], labels[18:], images[:6]))
records = json.loads((folder / "performance_benchmarks.json").read_text())
original_names = set(records)
experiments = json.loads((folder / "results/all_experiments.json").read_text())
search = experiments.get("nas", {}).get("architecture_search", {})
for trial, row in search.get("trials", {}).items():
    name = f"nas_architecture_search_trials_{trial}_model"
    if trial != search.get("best", {}).get("trial") and row.get("artifact_path"):
        records.setdefault(name, {"artifact_path": row["artifact_path"]})
evaluator = CompressionEvaluator()
only_missing = "--only-missing" in sys.argv
results = {name: row for name, row in records.items() if name in original_names} if only_missing else {}
if only_missing:
    evaluator.baseline_metrics = dict(records["baseline"])
for name, record in records.items():
    if only_missing and name in original_names:
        continue
    path = record["artifact_path"]
    model = path if path.endswith(".tflite") else tf.keras.models.load_model(path, compile=False)
    results[name] = evaluator.benchmark_compressed_model(model, (images[18:], labels[18:]), name)
export_benchmarks(results, include_workbook=False)
create_compression_visualizations(results, experiments)
generate_compression_guidelines(results, folder / "compression_guidelines.md")
scenarios = ApplicationScenarioAnalysis(results)
save_json({"mobile": scenarios.mobile_deployment_optimization(),
           "edge": scenarios.edge_device_optimization(),
           "cloud": scenarios.cloud_inference_optimization()}, folder / "results/application_scenarios.json")
print(f"Refreshed {len(results)} saved model benchmarks and plots.")
