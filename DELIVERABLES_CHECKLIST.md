# Deliverables checklist

Audit of the consolidated repository: **19 of 21 listed deliverables are present;
two PDF documents are missing.** Supporting utilities, tests and provenance are
retained because they are needed to execute, verify or reproduce the deliverables.

| Required deliverable | Status / evidence |
|---|---|
| `baseline_model.py` | Present: baseline architecture, data preparation and training. |
| `part1_pruning.py` | Present: pruning implementations and comparisons. |
| `part2_quantization.py` | Present: quantization and corrected binary-training safeguards. |
| `part3_distillation.py` | Present: knowledge-distillation experiments. |
| `part4_interaction_analysis.py` | Present: combinations, ordering, Pareto and failure analyses. |
| `compression_evaluator.py` | Present: metrics, repeated-seed utility and robustness. |
| `visualization_tools.py` | Present: measured-result plots and benchmark exports. |
| `application_scenarios.py` | Present: mobile, edge and cloud recommendations. |
| `requirements.txt` | Present: Python dependencies. |
| `README.md` | Present: setup, execution, provenance and limitations. |
| `baseline_model.keras` | Present: 80.83% test accuracy in `results/baseline_training.json`; exceeds 75%. |
| `compressed_models/` | Present: distinct compressed variants; duplicate exports resolve through `results/artifact_aliases.json`. |
| `checkpoints/` | Present: initial/best baseline weights, split indices and experiment checkpoints. |
| `compression_analysis_report.pdf` | **Missing:** required 3–4 page technical report; excluded by the earlier instruction not to create reports. |
| `compression_guidelines.md` | Present: regenerated from consolidated measured results. |
| `performance_benchmarks.xlsx` | Present: consolidated metrics, with corrected Part 2 results. |
| `interaction_analysis_summary.pdf` | **Missing:** required interaction summary; excluded by the earlier instruction not to create reports. |
| `pareto_frontiers.png` | Present: accuracy versus size and latency. |
| `compression_heatmaps.png` | Present: layer-wise compression impact. |
| `convergence_plots.png` | Present: training convergence comparisons. |
| `interaction_matrices.png` | Present: technique-interaction comparisons. |

## Remaining substantive gaps

1. **The combined compression target is unmet.** No saved benchmark achieves both
   at least 10x actual artifact compression and less than five percentage points
   of test-accuracy loss. The best test accuracy among executable artifacts above
   10x reduction is 55.35%, at 180.43x reduction (NAS width 1.5, depth 2), a loss of
   25.46 points against the original benchmark baseline. Corrected QAT INT8 TFLite
   reaches 81.06% test accuracy but only 3.92x compression. Packed pruning-plus-INT8
   alternatives preserve accuracy but reach approximately 8.1x compression.
2. **The two PDFs above remain to be written.** File existence and successful jobs
   do not substitute for these documents or establish that all grading criteria
   have been met.
3. **Repeated-seed experimental evidence is absent.** The fuller assignment
   skeleton includes multiple-seed statistics/significance. The function is
   implemented, but the saved full-data runs use seed 42 and do not provide that
   evaluation. This is additional to the 21-item file checklist.

## Interpretation of the evidence

The baseline training evaluation reports 80.83% **test** accuracy; the original
benchmark evaluation reports 80.81%. Both exceed the requirement. The corrected
binary experiment reports 1.64% **validation** accuracy, so it is retained as an
experimental failure rather than a successful compressed-model recommendation.
INT4/XNOR layers run floating-point reference kernels: packed size reductions do
not establish native low-bit acceleration or runtime-memory savings.

Full-run evidence comes from DTU job `30015147`; corrected Part 2 evidence comes
from job `30040324`. The consolidated tables and plots replace old standalone
Part 2 records with the corrected run. Other experiment families retain their
original measurements. `results/provenance/` preserves source records and logs;
active model paths are repository-relative.
