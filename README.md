# CIFAR-100 model compression pipeline

This implementation follows the supplied skeleton's functions, classes, method names,
result dictionaries and six-step execution flow. The requested code is split into the
listed deliverable files. Small shared utilities live in `common.py`; `main.py` runs
the pipeline. `neural_architecture_search.py` supplies the architecture search named in
the objective but missing from the skeleton. The training pipeline does not generate reports.
See `DELIVERABLES_CHECKLIST.md` for the current submission audit and remaining gaps.

## Consolidated DTU results

The repository-root model files, benchmark table and figures combine the full DTU
job **30015147** with the corrected Part 2 job **30040324**. Quantization results
come from the second job; pruning, distillation, interactions and NAS retain their
original measurements. Summaries and plots are regenerated from those records;
assembly does not retrain models or rerun measurements.

`results/baseline_training.json` records **80.83% test accuracy**, exceeding the
75% baseline requirement. The original benchmark measures 80.81%; these are separate
evaluations of the baseline. **No measured model satisfies both at least 10x size
reduction and less than five percentage points of accuracy loss.** The best test
accuracy among executable artifacts above 10x reduction is 55.35% (the 180.43x NAS
student). Corrected QAT INT8 TFLite reaches 81.06% test accuracy at 3.92x reduction.
The corrected binary experiment completes but reaches only 1.64% validation accuracy.
These results must not be represented as meeting the combined compression target.

Active JSON artifact paths are relative to this repository. Exact duplicate Keras
exports share a canonical file, with aliases recorded in
`results/artifact_aliases.json`; all distinct model variants are retained.
Original benchmarks, configurations, package versions and scheduler logs remain in
`results/provenance/`, where original HPC paths are preserved as provenance.

`scripts/assemble_deliverables.py` reconstructs the consolidated artifacts from the
two downloaded folders `outputs/dtu_30015147/` and `outputs/dtu_part2_fixed/`, plus
`outputs/hpc_logs/`. It regenerates plots and tabular summaries without training.
Use it only in a clean destination with those downloads available; it refuses to
overwrite an already assembled `baseline_model.keras`. The submission's
`performance_benchmarks.xlsx` contains the consolidated benchmark records.

`interaction_analysis_summary.pdf` summarizes the measured Part 4 combinations,
and ordering comparisons in two tables. Rebuild it without training using
`python scripts/build_interaction_summary.py` (optional dependency: `reportlab==4.4.9`).
The main `compression_analysis_report.pdf` remains missing under the earlier
instruction not to create that report. Multiple-seed evaluation is implemented but has no saved
full-data experimental results; the downloaded runs use seed 42.

## Setup

Use **Python 3.10–3.12**. The dependencies pin TensorFlow 2.16.2 and legacy Keras 2
(`tf-keras` 2.16.0) for compatibility. `common.py` sets `TF_USE_LEGACY_KERAS=1`
before importing TensorFlow. Import project modules before importing TensorFlow in
a notebook. TensorFlow 2.16 defaults to Keras 3; see the
[Keras compatibility guide](https://keras.io/getting_started/).
For Python 3.12, `set_seed` includes a narrowly scoped workaround for legacy
Keras 2.16's floating-point `randint` bound; it preserves deterministic seeds
without modifying installed dependencies or Python's global random module.

Windows PowerShell:

The local `.venv` used for verification is already installed in this workspace.
The first two commands below are for creating a fresh environment.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Linux or WSL:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

The first real run downloads CIFAR-100 and, by default, ImageNet weights into
`.keras/`. Native Windows uses CPU with this TensorFlow version. For NVIDIA GPU
training, use a compatible Linux/WSL installation and `tensorflow[and-cuda]==2.16.2`;
follow the [official TensorFlow installation instructions](https://www.tensorflow.org/install/pip).
Full training and all comparisons are substantial GPU workloads. Several ResNet-50
variants and optimizer states may be resident at once; use `--parts` to run families
separately when memory is limited.

## Run

```powershell
# Tiny synthetic end-to-end execution check: no dataset or weight downloads.
.\.venv\Scripts\python.exe main.py --smoke --quiet

# One epoch on a stratified CIFAR-100 subset; tests wiring, not accuracy targets.
.\.venv\Scripts\python.exe main.py --quick

# Full baseline, core comparisons, interactions and architecture search in a fresh directory.
.\.venv\Scripts\python.exe main.py --output-dir outputs/new_full

# Also run sensitivity, lottery tickets, dynamic quantization, feature/ensemble KD,
# all six technique orders, and corruption/calibration evaluation.
.\.venv\Scripts\python.exe main.py --extended --output-dir outputs/new_extended

# Train the baseline separately, then reuse it in subsequent commands.
.\.venv\Scripts\python.exe main.py --baseline-only --output-dir outputs/new_pruning_quantization
.\.venv\Scripts\python.exe main.py --parts pruning quantization --output-dir outputs/new_pruning_quantization
```

Useful controls: `--output-dir`, `--seed`, `--batch-size`, `--baseline-epochs`,
`--finetune-epochs`, `--distill-epochs`, `--no-pretrained`, `--retrain`,
`--power-watts` and `--no-workbook`. Run `python main.py --help` for details.
An existing baseline in the selected output directory is reused unless `--retrain`
is passed. Use a new output directory when changing data/configuration; each run
replaces that directory's aggregate results and benchmark exports. Per-family runs
do not merge old results. `--quick` writes to `outputs/quick`; `--smoke` writes to
`outputs/smoke`, keeping those artifacts separate from full-run deliverables.

## DTU HPC submission

Copy the Python files, `requirements.txt`, `setup_dtu_hpc.sh` and `run_dtu_hpc.sh`
to a project directory on DTU. Do not copy the Windows `.venv` or synthetic
`outputs/` directory. Use shared storage with enough space for the
Linux environment and many saved model variants.

From that project directory on DTU:

```bash
module avail python3
# Default: python3/3.10.13. Override if a different 3.10–3.12 module is available:
# PYTHON_MODULE=python3/<version> bash setup_dtu_hpc.sh
bash setup_dtu_hpc.sh
bsub < run_dtu_hpc.sh
bjobs
```

The batch script requests one exclusive A100, four CPU slots, 64 GB total host RAM,
and 24 hours. It runs all experiments, including the extended methods, and writes
artifacts to `outputs/dtu_<job-id>/`. Scheduler logs are `compression_<job-id>.out`
and `.err`; a combined log and package versions are also saved with the artifacts.
It verifies GPU execution before starting training. No report is produced.

For a short real-data wiring check before the full experiment:

```bash
MODE=quick bsub -W 2:00 < run_dtu_hpc.sh
```

To train the baseline in its own job, then reuse the **completed** baseline:

```bash
export OUTPUT_DIR="$PWD/outputs/dtu_full"
MODE=baseline bsub < run_dtu_hpc.sh
# After the baseline job finishes successfully:
MODE=all bsub < run_dtu_hpc.sh
```

Optional exported settings: `BATCH_SIZE`, `BASELINE_EPOCHS`, `FINETUNE_EPOCHS`,
`DISTILL_EPOCHS`, `SEED`, and `EXTENDED=0` to omit the additional comparisons.
`PARTS=quantization` selects Part 2; `BASELINE_SOURCE` can point to a previously
trained `baseline_model.keras` to copy into a fresh `OUTPUT_DIR`.
Use the same seed/configuration when reusing a baseline. Do not run two jobs into
the same output directory. A 24-hour allocation is a limit, not a guarantee that
the complete experiment suite finishes. Completed files persist if the job times
out; the pipeline reuses a completed baseline but does not automatically resume
an interrupted training stage from its checkpoint.

The scripts follow DTU's [LSF GPU job instructions](https://www.hpc.dtu.dk/?page_id=2759),
[per-slot memory requests](https://www.hpc.dtu.dk/?page_id=1416), and
[TensorFlow CUDA installation guidance](https://www.hpc.dtu.dk/?page_id=3755).
If `gpua100` is unavailable to your account, change the queue directive or override
it with `bsub -q <your-GPU-queue> < run_dtu_hpc.sh`.

## Implementation map

| File | Skeleton functionality |
|---|---|
| `baseline_model.py` | ResNet-50, CIFAR-100 splits, augmentation, checkpoints, LR scheduling, training |
| `part1_pruning.py` | Magnitude schedules, physical bottleneck channel removal, sensitivity, lottery tickets |
| `part2_quantization.py` | Mixed bits, PTQ/QAT, INT4/binary references, native INT8, dynamic quantization, error propagation |
| `part3_distillation.py` | Temperature search, progressive KD, attention/feature matching, self/ensemble distillation |
| `part4_interaction_analysis.py` | Individual controls, combinations, ordering, Pareto frontiers and failure analysis |
| `compression_evaluator.py` | Accuracy, real artifact sizes, latency, RSS, FLOPs, seed statistics and robustness |
| `visualization_tools.py` | Plots, benchmark JSON/CSV/XLSX, guideline function re-export |
| `application_scenarios.py` | Mobile, edge and cloud constraints; practical guidelines |
| `neural_architecture_search.py` | Equal-epoch width/depth grid search under a parameter budget |
| `common.py`, `main.py` | Shared reproducibility utilities and the six-step runner |

## Data and measurement conventions

- CIFAR-100 training data is split into 45,000 training and 5,000 validation samples
  with class stratification. The official 10,000-image test set remains separate.
  Calibration uses 1,000 stratified training images by default. Quick mode reduces
  these counts. Saved split indexes and seeds make the selection reproducible.
- Arrays retain raw RGB pixels in `[0,255]`. Training applies random crops/flips;
  baseline resizing and ResNet preprocessing are inside the model. Students contain
  their own `1/255` rescaling. Do not normalize inputs a second time.
- Networks output **logits**. Classification losses use `from_logits=True` and KD
  uses temperature-scaled KL with the required `T²` factor. Teacher predictions use
  inference mode. Hyperparameters, pruning sensitivities and architecture candidates
  use validation data. Test data is used for final evaluation.
- Baseline initialization is saved before optimization. With ImageNet initialization,
  the lottery-ticket experiment is a **transfer-learning rewind experiment**, not
  a claim about tickets in an entirely randomly initialized baseline.
- Unstructured pruning enforces zero masks after each optimizer update. Dense Keras
  kernels still execute dense operations. Structured pruning physically reduces
  internal ResNet bottleneck channels and slices dependent BN/conv tensors; residual
  output widths are kept compatible. The requested channel reduction is not the same
  as total parameter reduction.
- Native TFLite INT8 artifacts have real quantized input/output handling and measured
  accuracy. Dynamic-range conversion permits float activations. Custom 8/4/2-bit
  QAT and binary/XNOR-style layers execute **FP32 reference kernels** with quantized
  values; they do not provide native INT4/XNOR hardware acceleration.
- `.packed.npz` files actually bit-pack quantized weights. `load_packed_model(path)`
  reconstructs a runnable reference model, expanding those weights into FP32 memory.
  Packed storage, dense Keras size and executable TFLite size are reported separately.
  QAT-to-Lite export bakes quantized kernels and recalibrates activations; its accuracy
  is measured again rather than assumed identical to the QAT reference.
- Sizes use MiB (`2**20` bytes), despite the skeleton's `_mb` names. Model sizes come
  from saved inference artifacts without optimizer slots. Accuracy is a fraction;
  `accuracy_loss_percent` is an absolute **percentage-point** loss. The target is
  `compression_ratio >= 10` and `accuracy_loss_percent < 5`.
- Latency is the median of warmed, synchronized executions. Single-image p95, batch
  latency and throughput are also recorded. TFLite uses one CPU thread; TensorFlow
  uses its configured device/threads. Backend differences are included in records.
  The pinned native Windows build disables default TFLite delegates because its
  XNNPACK delegate can crash on floating-point residual graphs. Built-in kernels
  still execute the exported INT8 models.
- RSS includes the Python runtime and other resident models and is not an isolated
  device-memory measurement. FLOPs estimate Conv/Dense arithmetic (multiply-add = 2)
  and exclude preprocessing, BN and activation work. TFLite FLOPs are unavailable,
  rather than guessed from file size. Low-bit/sparse execution is not assumed to
  reduce dense FLOPs.
- Energy stays blank unless `--power-watts` is supplied. That value gives a rough
  constant-power estimate, not a measured battery or thermal result. Deployment
  recommendations use host measurements as proxies and require target-device checks.
- `cross_validate_compression` runs training seeds on a fixed split, not k-fold CV.
  Pass a function returning `final_accuracy` (or `test_accuracy`); include a matched
  per-seed `baseline_accuracy` to obtain a paired significance test.

## Generated files

A successful full run writes:

```text
baseline_model.keras
checkpoints/                     # initial/best weights, split indexes, training logs
compressed_models/               # .keras, .tflite and .packed.npz variants
results/                         # configs, histories, analyses, scenario recommendations
performance_benchmarks.json
performance_benchmarks.csv
performance_benchmarks.xlsx
compression_guidelines.md
accuracy_vs_compression.png
pareto_frontiers.png
compression_heatmaps.png
convergence_plots.png
interaction_matrices.png
tradeoff_radar.png
```

Individual methods generate additional sensitivity, temperature and Pareto plots.
Figures requiring omitted experiment families are omitted. No technical report,
interaction-summary PDF or `.tex` file is produced.

The root artifacts now contain real CIFAR-100 measurements. The baseline accuracy
target is met; the combined 10x/<5-point target remains unmet. Synthetic and
one-epoch runs are execution checks and must be kept separate from this evidence.

## Validation

### Repeated training seeds

`scripts/evaluate_seeds.py` runs INT8 reference QAT from the same trained baseline
with seeds 42, 43 and 44 on a fixed CIFAR-100 split. It saves each model/history,
test accuracy, mean, sample standard deviation, 95% t confidence interval and a
paired comparison against the fixed baseline. This measures QAT training variability,
not baseline retraining variability or every compression family. Three seeds give
limited statistical power. The measurements remain missing until this job completes.

On DTU, from the project directory (substitute a valid notification address):

```bash
bsub -N -u YOUR_EMAIL_ADDRESS \
  -env "all,MODE=seeds,BASELINE_SOURCE=$PWD/outputs/dtu_30015147/baseline_model.keras" \
  < run_dtu_hpc.sh
```

Upload `scripts/evaluate_seeds.py` and the updated `run_dtu_hpc.sh` first. The job
uses its own `outputs/dtu_<job-id>/` directory and writes `seed_results.json` there.

Run `python scripts/verify_deliverables.py` to check saved archive integrity,
finite model/checkpoint tensors, portable model references and workbook values.
This check reads the packaged results without training or downloading data.
`results/cleanup_record.json` records the recovery location for removed obsolete
files. The ignored `.venv/` is a local execution environment, not a submission file.

The last completed offline test run contains 43 passing numerical and behavioral
checks. It covers data-split isolation, baseline graph construction, exact pruning
masks, residual-safe channel surgery, custom quantization gradients and serialization,
integer TFLite execution, unchanged teacher weights, feature-adapter updates,
benchmark arithmetic, Pareto ties, and scenario constraints. These tests do not train
CIFAR-100 to the accuracy target. The binary regression tests cover latent-weight
clipping and rejection of non-finite quantized kernels during archive packing.
The completed DTU jobs provide the separate full-data experiment evidence.
