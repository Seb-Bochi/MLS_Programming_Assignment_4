"""Plots and tabular exports generated only from measured experiment results."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from common import get_config, save_json, json_ready
from application_scenarios import generate_compression_guidelines


def _walk(value, prefix=""):
    if isinstance(value, dict):
        yield prefix, value
        for key, child in value.items():
            if isinstance(child, dict):
                yield from _walk(child, f"{prefix}/{key}".strip("/"))


def _frontier(rows, cost):
    return [row for row in rows if not any(
        other[cost] <= row[cost] and other["test_accuracy"] >= row["test_accuracy"]
        and (other[cost] < row[cost] or other["test_accuracy"] > row["test_accuracy"])
        for other in rows)]


def create_compression_visualizations(evaluation_results, experiment_results=None):
    """Create the skeleton's figures, plus a multi-metric radar comparison.

    evaluation_results maps names to benchmark dictionaries. experiment_results
    contains histories, per-layer sparsity and interaction records when available.
    Missing analyses are omitted rather than replaced by fabricated data.
    """
    folder = get_config().output_dir
    folder.mkdir(parents=True, exist_ok=True)
    rows = list(evaluation_results.values())
    if not rows:
        raise ValueError("No benchmark results supplied")
    sns.set_theme(style="whitegrid", context="notebook")
    paths = {}
    def save(fig, name):
        if get_config().quick:
            fig.suptitle("Quick run: execution checks only", color="#835F24", fontsize=11)
        fig.tight_layout()
        path = folder / name
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths[name] = str(path)
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        ax.scatter(row["compression_ratio"], 100 * row["test_accuracy"], s=35)
    ax.set(xlabel="Actual artifact compression ratio", ylabel="Test accuracy (%)",
           title="Accuracy and stored model size", xscale="log")
    save(fig, "accuracy_vs_compression.png")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, cost, label in zip(axes, ("model_size_mb", "single_inference_ms"),
                              ("Inference artifact size (MiB)", "Single-image latency (ms)")):
        ax.scatter([r[cost] for r in rows], [100 * r["test_accuracy"] for r in rows], alpha=0.65)
        front = sorted(_frontier(rows, cost), key=lambda r: r[cost])
        ax.plot([r[cost] for r in front], [100 * r["test_accuracy"] for r in front], "o-", color="#136F63")
        ax.set(xlabel=label, ylabel="Test accuracy (%)", xscale="log",
               ylim=(0, max(1, min(100, 100 * max(r["test_accuracy"] for r in rows) + 3))))
    save(fig, "pareto_frontiers.png")
    experiments = experiment_results or {}
    layer_rows = {}
    histories = {}
    for path, value in _walk(experiments):
        if "layer_sparsity_analysis" in value:
            layer_rows[path] = {name: item["sparsity"] for name, item in value["layer_sparsity_analysis"].items()}
        history = value.get("training_history", value.get("history"))
        if isinstance(history, dict) and isinstance(history.get("val_accuracy"), (list, tuple)):
            histories[path] = history
    if layer_rows:
        names = sorted(set().union(*(item.keys() for item in layer_rows.values())))
        matrix = [[item.get(name, np.nan) for item in layer_rows.values()] for name in names]
        fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(layer_rows)), max(5, len(names) * 0.17)))
        sns.heatmap(matrix, vmin=0, vmax=1, cmap="Blues", yticklabels=names,
                    xticklabels=[name.rsplit("/", 1)[-1].replace("_", " ") for name in layer_rows],
                    cbar_kws={"label": "Zero kernel fraction"}, ax=ax)
        ax.tick_params(axis="y", labelsize=6)
        ax.tick_params(axis="x", labelsize=10, rotation=0)
        save(fig, "compression_heatmaps.png")
    if histories:
        fig, ax = plt.subplots(figsize=(11, 6))
        for index, (name, history) in enumerate(histories.items()):
            ax.plot(np.arange(1, len(history["val_accuracy"]) + 1),
                    100 * np.asarray(history["val_accuracy"]), marker="o", markersize=3,
                    color=plt.get_cmap("tab20")(index % 20),
                    label=" / ".join(name.split("/")[-2:]).replace("_", " "))
        best_accuracy = max(max(h["val_accuracy"]) for h in histories.values())
        ax.set(xlabel="Epoch", ylabel="Validation accuracy (%)", title="Training convergence",
               ylim=(0, max(1, min(100, 100 * best_accuracy + 3))))
        ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
        save(fig, "convergence_plots.png")
    interaction = experiments.get("interaction", {})
    matrix_rows = []
    matrix_names = []
    for name, record in interaction.items():
        if isinstance(record, dict):
            metric = record.get("metrics", record)
            score = metric.get("validation_accuracy", metric.get("test_accuracy", record.get("final_accuracy")))
            if score is not None and "compression_ratio" in metric:
                matrix_names.append(name)
                matrix_rows.append([100 * score, metric["compression_ratio"], metric["single_inference_ms"]])
    if matrix_rows:
        values = np.asarray(matrix_rows)
        fig, axes = plt.subplots(1, 3, figsize=(12, max(4, 0.45 * len(matrix_rows))))
        for i, (ax, title) in enumerate(zip(axes, ("Validation accuracy (%)", "Artifact compression (x)", "Latency (ms)"))):
            sns.heatmap(values[:, i:i + 1], annot=True, fmt=".2f", cmap="Blues", cbar=False,
                        yticklabels=matrix_names if i == 0 else False, xticklabels=[title], ax=ax)
        save(fig, "interaction_matrices.png")
    selected = sorted(rows, key=lambda row: row["test_accuracy"], reverse=True)[:5]
    dimensions = ("test_accuracy", "compression_ratio", "single_inference_ms", "peak_memory_mb")
    values = np.asarray([[r[k] for k in dimensions] for r in selected], dtype=float)
    for i in (2, 3):
        values[:, i] = 1 / np.maximum(values[:, i], 1e-9)
    values /= np.maximum(values.max(axis=0), 1e-9)
    angles = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw={"projection": "polar"})
    for row, scores in zip(selected, values):
        ax.plot(np.r_[angles, angles[0]], np.r_[scores, scores[0]], label=row["model_name"])
    ax.set_xticks(angles, ["Accuracy", "Compression", "Inverse latency", "Inverse RSS"])
    ax.set_ylim(0, 1)
    ax.set_title("Relative scores within selected models (higher is better)", pad=25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.15, 1), fontsize=7)
    save(fig, "tradeoff_radar.png")
    return paths


def export_benchmarks(evaluation_results, include_workbook=True):
    """Portable Python runtime export; the XLSX dependency is openpyxl."""
    folder = get_config().output_dir
    save_json(evaluation_results, folder / "performance_benchmarks.json")
    rows = [json_ready(row) for row in evaluation_results.values()]
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (folder / "performance_benchmarks.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    if not include_workbook:
        return str(folder / "performance_benchmarks.csv")
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Benchmarks"
    fields = ["model_name", "test_accuracy", "model_size_mb", "compression_ratio",
              "accuracy_loss_percent", "single_inference_ms", "single_p95_ms",
              "batch_inference_ms", "batch_size", "throughput_images_per_second",
              "peak_memory_mb", "flops", "flops_reduction", "energy_consumption", "ece"]
    headers = ["Model", "Accuracy", "Artifact size (MiB)", "Compression (x)", "Accuracy loss (pp)",
               "Single latency (ms)", "Single p95 (ms)", "Batch latency (ms)", "Batch size",
               "Throughput (images/s)", "Process RSS (MiB)", "Estimated FLOPs", "FLOPs reduction",
               "Energy estimate (J/image)", "ECE"]
    sheet.append(["Compression performance benchmarks"])
    sheet.append(["Smoke runs use synthetic data. Check results/environment.json for run provenance."])
    sheet.append(["Sizes include the inference artifact. RSS includes the runtime; blank energy means unmeasured."])
    sheet.append(headers)
    for row in rows:
        sheet.append([row.get(field) for field in fields])
    sheet.freeze_panes = "B5"
    sheet.auto_filter.ref = f"A4:O{sheet.max_row}"
    for cell in sheet[4]:
        cell.font = Font(color="FFFFFF", bold=True)
        cell.fill = PatternFill("solid", fgColor="173C50")
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    sheet.row_dimensions[4].height = 32
    for index in range(1, len(fields) + 1):
        sheet.column_dimensions[get_column_letter(index)].width = 24 if index > 1 else 58
    for row in sheet.iter_rows(min_row=5):
        for cell in row[1:]:
            cell.number_format = "0.000"
        for index in (1, 12, 14):
            row[index].number_format = "0.00%"
    path = folder / "performance_benchmarks.xlsx"
    workbook.save(path)
    return str(path)
