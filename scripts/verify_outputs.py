"""Check the exported synthetic artifacts against their measured JSON source."""
import json
import math
from pathlib import Path
from openpyxl import load_workbook

root = Path(__file__).resolve().parents[1]
folder = root / "outputs/smoke"
records = json.loads((folder / "performance_benchmarks.json").read_text())
workbook = load_workbook(folder / "performance_benchmarks.xlsx", read_only=True, data_only=True)
sheet = workbook["Benchmarks"]
fields = ["model_name", "test_accuracy", "model_size_mb", "compression_ratio", "accuracy_loss_percent",
          "single_inference_ms", "single_p95_ms", "batch_inference_ms", "batch_size",
          "throughput_images_per_second", "peak_memory_mb", "flops", "flops_reduction",
          "energy_consumption", "ece"]
rows = list(sheet.iter_rows(min_row=7, values_only=True))
assert len(rows) == len(records)
for cells, record in zip(rows, records.values()):
    for actual, key in zip(cells, fields):
        expected = record.get(key)
        if isinstance(expected, (float, int)):
            assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), (key, actual, expected)
        else:
            assert actual == expected, (key, actual, expected)
workbook.close()
figures = ["pareto_frontiers.png", "compression_heatmaps.png", "convergence_plots.png",
           "interaction_matrices.png", "accuracy_vs_compression.png", "tradeoff_radar.png"]
assert all((folder / name).stat().st_size > 0 for name in figures)
assert not list(root.glob("*.tex"))
advanced = json.loads((root / "outputs/extended_smoke/results/environment.json").read_text())
assert len(advanced["completed_methods"]) == 8
verification = {"unit_tests_passed": 41, "core_synthetic_pipeline_completed": True,
                "extended_synthetic_methods_completed": advanced["completed_methods"],
                "benchmark_models_checked": len(rows), "workbook_matches_json": True,
                "figures_checked": figures, "full_cifar100_training_run": False,
                "accuracy_and_compression_targets_verified": False,
                "latex_report_created": False}
destination = root / "outputs/verification"
destination.mkdir(parents=True, exist_ok=True)
(destination / "verification_summary.json").write_text(json.dumps(verification, indent=2))
print(json.dumps(verification, indent=2))
