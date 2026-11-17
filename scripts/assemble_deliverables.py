"""Assemble downloaded DTU runs, replacing Part 2 and deduplicating identical models.

Run from the repository root before removing the downloaded output directories.
No training or new benchmark measurements are performed.
"""
import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    if path.suffix == ".keras":
        with zipfile.ZipFile(path) as archive:
            h.update(json.dumps(json.loads(archive.read("config.json")), sort_keys=True).encode())
            h.update(archive.read("model.weights.h5"))
    else:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b""):
                h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-workbook", action="store_true",
                        help="Export JSON/CSV only; optionally use export_benchmarks.mjs for XLSX.")
    args = parser.parse_args()
    full = ROOT / "outputs/dtu_30015147"
    fixed = ROOT / "outputs/dtu_part2_fixed"
    assert full.is_dir() and fixed.is_dir(), "Both downloaded DTU runs are required."
    assert not (ROOT / "baseline_model.keras").exists(), "Deliverables already assembled."
    artifacts = {p.relative_to(full).as_posix(): p for folder in ("compressed_models", "checkpoints")
                 for p in (full / folder).iterdir() if p.is_file()}
    artifacts.update({p.relative_to(fixed).as_posix(): p for folder in ("compressed_models", "checkpoints")
                      for p in (fixed / folder).iterdir() if p.is_file()})
    artifacts["baseline_model.keras"] = full / "baseline_model.keras"
    aliases, seen, selected = {}, {}, {}
    # Prefer descriptive original filenames, and the root baseline, to benchmark copies.
    priority = lambda item: (item[0] != "baseline_model.keras", "benchmark_" in item[0], item[0])
    for relative, source in sorted(artifacts.items(), key=priority):
        key = (source.suffix, digest(source))
        if source.suffix == ".keras" and key in seen:
            aliases[relative] = seen[key]
        else:
            aliases[relative] = relative
            seen[key] = relative
            selected[relative] = source
    assert digest(full / "baseline_model.keras") == digest(fixed / "baseline_model.keras")
    for relative, source in selected.items():
        target = ROOT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists(), f"Refusing to overwrite {target}"
        shutil.copy2(source, target)

    def portable(value):
        if isinstance(value, dict):
            return {k: portable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [portable(v) for v in value]
        if isinstance(value, str) and value.startswith("/zhome/"):
            for marker in ("/outputs/dtu_30015147", "/outputs/dtu_part2_fixed"):
                if marker in value:
                    relative = value.split(marker, 1)[1].lstrip("/")
                    return aliases.get(relative, relative or ".")
        return value

    records = read(full / "performance_benchmarks.json")
    corrected = read(fixed / "performance_benchmarks.json")
    for name, row in records.items():
        row["source_job"] = "30015147"
    for name, row in corrected.items():
        if name != "baseline":
            row["source_job"] = "30040324"
            records[name] = row
    records = portable(records)
    experiments = read(full / "results/all_experiments.json")
    experiments["quantization"] = read(fixed / "results/all_experiments.json")["quantization"]
    experiments = portable(experiments)

    skip = {"all_experiments.json", "environment.json", "guidelines.json", "application_scenarios.json"}
    results = {p.name: p for p in (full / "results").glob("*.json") if p.name not in skip}
    for p in (fixed / "results").glob("*.json"):
        if p.name.startswith(("quantization_", "qat_", "extreme_")):
            results[p.name] = p
    for name, source in results.items():
        write(ROOT / "results" / name, portable(read(source)))
    write(ROOT / "results/all_experiments.json", experiments)
    write(ROOT / "results/robustness_part2.json", portable(read(fixed / "results/robustness.json")))
    environment = portable(read(full / "results/environment.json"))
    environment["consolidation"] = {
        "full_run_job": "30015147", "quantization_replacement_job": "30040324",
        "note": "Part 2 rows replaced, other measured rows preserved. No measurements rerun locally.",
        "artifact_paths": "relative to repository root",
    }
    write(ROOT / "results/environment.json", environment)
    write(ROOT / "results/artifact_aliases.json", aliases)
    for source_root, job in ((full, "30015147"), (fixed, "30040324")):
        provenance = ROOT / "results/provenance" / job
        provenance.mkdir(parents=True, exist_ok=True)
        for name in ("performance_benchmarks.json", "results/environment.json", "results/all_experiments.json",
                     f"environment_{job}.txt"):
            shutil.copy2(source_root / name, provenance / Path(name).name)
    logs = ROOT / "results/provenance/logs"
    logs.mkdir(exist_ok=True)
    for p in (ROOT / "outputs/hpc_logs").iterdir():
        if p.is_file():
            shutil.copy2(p, logs / p.name)

    from common import configure, save_json
    from visualization_tools import export_benchmarks, create_compression_visualizations
    from application_scenarios import ApplicationScenarioAnalysis, generate_compression_guidelines
    configure(output_dir=ROOT)
    export_benchmarks(records, include_workbook=not args.no_workbook)
    create_compression_visualizations(records, experiments)
    for name in ("pruning_sensitivity.png", "temperature_optimization.png", "interaction_pareto_frontiers.png"):
        shutil.copy2(full / name, ROOT / name)
    guidelines = generate_compression_guidelines(records, ROOT / "compression_guidelines.md")
    save_json(guidelines, ROOT / "results/guidelines.json")
    scenarios = ApplicationScenarioAnalysis(records)
    save_json({"mobile": scenarios.mobile_deployment_optimization(),
               "edge": scenarios.edge_device_optimization(),
               "cloud": scenarios.cloud_inference_optimization()}, ROOT / "results/application_scenarios.json")
    failures = {name: row for name, row in records.items()
                if row.get("accuracy_loss_percent", 0) > 5}
    save_json({"note": "Measured test losses above five percentage points; includes corrected Part 2.",
               "models": failures}, ROOT / "results/compression_failure_summary.json")
    # Verify every active model reference; raw provenance intentionally retains original HPC paths.
    for name, row in records.items():
        assert (ROOT / row["artifact_path"]).is_file(), (name, row["artifact_path"])
    summary = {"benchmark_rows": len(records), "model_files_kept": len(selected),
               "duplicate_model_aliases": sum(k != v for k, v in aliases.items()),
               "target_passing_models": [name for name, r in records.items()
                                          if r["compression_ratio"] >= 10 and r["accuracy_loss_percent"] < 5],
               "missing_documents": ["compression_analysis_report.pdf", "interaction_analysis_summary.pdf"]}
    write(ROOT / "results/deliverable_audit.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
