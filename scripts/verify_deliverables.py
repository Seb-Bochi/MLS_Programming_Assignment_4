"""Audit packaged artifacts without training or downloading data."""
import io
import json
import math
import zipfile
from pathlib import Path

import h5py
import numpy as np
from openpyxl import load_workbook
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors, counts = [], {}
    paths = list((ROOT / "compressed_models").glob("*"))
    paths += list((ROOT / "checkpoints").glob("*"))
    paths += list((ROOT / "results").rglob("*.json"))
    paths += list(ROOT.glob("*.json")) + list(ROOT.glob("*.png"))
    paths += [ROOT / "baseline_model.keras", ROOT / "performance_benchmarks.xlsx"]

    def finite_h5(stream):
        with h5py.File(stream, "r") as h5:
            def check(name, item):
                if isinstance(item, h5py.Dataset) and item.dtype.kind in "fc":
                    assert np.isfinite(item[()]).all(), f"Non-finite tensor: {name}"
            h5.visititems(check)

    for path in paths:
        if path.name == ".gitkeep":
            continue
        try:
            assert path.stat().st_size > 0, "Empty file"
            counts[path.suffix] = counts.get(path.suffix, 0) + 1
            if path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"),
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            elif path.suffix in (".keras", ".xlsx", ".npz"):
                with zipfile.ZipFile(path) as archive:
                    assert archive.testzip() is None, "ZIP CRC failure"
                    if path.suffix == ".keras":
                        finite_h5(io.BytesIO(archive.read("model.weights.h5")))
                if path.suffix == ".npz":
                    with np.load(path, allow_pickle=False) as archive:
                        for key in archive.files:
                            a = archive[key]
                            if a.dtype.kind in "fc":
                                assert np.isfinite(a).all(), f"Non-finite array: {key}"
            elif path.suffix == ".h5":
                finite_h5(path)
            elif path.suffix == ".tflite":
                with path.open("rb") as f:
                    assert f.read(8)[4:] == b"TFL3", "Invalid Lite header"
            elif path.suffix == ".png":
                with Image.open(path) as im:
                    im.verify()
        except Exception as exc:
            errors.append({"path": str(path.relative_to(ROOT)), "error": str(exc)})

    records = json.loads((ROOT / "performance_benchmarks.json").read_text())
    missing_references = []

    def check_references(value):
        if isinstance(value, dict):
            for child in value.values():
                check_references(child)
        elif isinstance(value, list):
            for child in value:
                check_references(child)
        elif isinstance(value, str) and value.startswith(("compressed_models/", "checkpoints/")):
            if not (ROOT / value).is_file():
                missing_references.append(value)

    for path in (ROOT / "results").glob("*.json"):
        if path.name != "artifact_aliases.json":
            check_references(json.loads(path.read_text()))
    check_references(records)
    assert not missing_references, sorted(set(missing_references))
    workbook = load_workbook(ROOT / "performance_benchmarks.xlsx", read_only=True, data_only=True)
    sheet = workbook["Benchmarks"]
    rows = list(sheet.iter_rows(min_row=7, values_only=True))
    assert len(rows) == len(records), (len(rows), len(records))
    fields = ["model_name", "test_accuracy", "model_size_mb", "compression_ratio", "accuracy_loss_percent",
              "single_inference_ms", "single_p95_ms", "batch_inference_ms", "batch_size",
              "throughput_images_per_second", "peak_memory_mb", "flops", "flops_reduction",
              "energy_consumption", "ece", "source_job", "compression_target_met"]
    for cells, record in zip(rows, records.values()):
        for cell, field in zip(cells, fields):
            expected = record.get(field)
            if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                assert math.isclose(cell, expected, rel_tol=1e-10, abs_tol=1e-10), (field, cell, expected)
            else:
                assert cell == expected, (field, cell, expected)
    workbook.close()
    summary = {"integrity_errors": errors, "checked_files_by_extension": counts,
               "missing_model_references": missing_references, "workbook_matches_json": True,
               "benchmark_rows": len(records),
               "missing_documents": [name for name in ("compression_analysis_report.pdf",
                                                         "interaction_analysis_summary.pdf")
                                     if not (ROOT / name).is_file()]}
    (ROOT / "results/integrity_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
