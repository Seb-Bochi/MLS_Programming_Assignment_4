#!/usr/bin/env bash
# DTU uses LSF (#BSUB), not Slurm (#SBATCH).
# Submit from the project directory: bsub < run_dtu_hpc.sh
#BSUB -J mls_compression
#BSUB -q gpua100
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o compression_%J.out
#BSUB -e compression_%J.err

set -euo pipefail
# Four CPU slots x 16 GiB/slot reserve 64 GiB host RAM; GPU memory is separate.
# Change the queue directive above if your account uses a different GPU queue.
if [[ -z "${LSB_JOBID:-}" ]]; then
    echo "Submit with 'bsub < run_dtu_hpc.sh'; do not train on a login node." >&2
    exit 1
fi
PROJECT_DIR="${PROJECT_DIR:-${LS_SUBCWD:-$PWD}}"
cd "$PROJECT_DIR"
PROJECT_DIR="$PWD"
[[ -f main.py ]] || { echo "Submit from the directory containing main.py." >&2; exit 1; }
VENV_DIR="$PROJECT_DIR/.venv-hpc"
[[ -f "$VENV_DIR/dtu-python-module.txt" ]] || {
    echo "Run 'bash setup_dtu_hpc.sh' first." >&2; exit 1;
}

if ! type module >/dev/null 2>&1; then
    [[ ! -f /etc/profile.d/modules.sh ]] || source /etc/profile.d/modules.sh
fi
module purge
PYTHON_MODULE="$(cat "$VENV_DIR/dtu-python-module.txt")"
module load "$PYTHON_MODULE"
source "$VENV_DIR/bin/activate"
export PYTHONNOUSERSITE=1
export TF_USE_LEGACY_KERAS=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_NUM_INTRAOP_THREADS="${LSB_DJOB_NUMPROC:-4}"
export TF_NUM_INTEROP_THREADS=2
export OMP_NUM_THREADS="$TF_NUM_INTRAOP_THREADS"
export OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export KERAS_HOME="$PROJECT_DIR/.keras"
# Preserve LSF's CUDA_VISIBLE_DEVICES; never select an unallocated GPU.

# Make pip-installed NVIDIA libraries/tools visible to the dynamic loader/XLA.
mapfile -t CUDA_PATHS < <(python - <<'PY'
from pathlib import Path
import site
roots = [Path(p) / "nvidia" for p in site.getsitepackages()]
print(":".join(str(p) for root in roots for p in sorted(root.glob("*/lib"))))
print(":".join(str(p) for root in roots for p in sorted(root.glob("*/bin"))))
print(next((str(root / "cuda_nvcc") for root in roots
            if (root / "cuda_nvcc/nvvm/libdevice").is_dir()), ""))
PY
)
if [[ -n "${CUDA_PATHS[0]:-}" ]]; then
    export LD_LIBRARY_PATH="${CUDA_PATHS[0]}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -n "${CUDA_PATHS[1]:-}" ]]; then
    export PATH="${CUDA_PATHS[1]}:$PATH"
fi
if [[ -n "${CUDA_PATHS[2]:-}" ]]; then
    export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_cuda_data_dir=${CUDA_PATHS[2]}"
fi

MODE="${MODE:-all}"                 # all | baseline | quick
EXTENDED="${EXTENDED:-1}"           # 1 includes every optional experiment
PARTS="${PARTS:-}"                  # Optional space-separated main.py --parts values
BASELINE_SOURCE="${BASELINE_SOURCE:-}"  # Optional existing baseline_model.keras
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/outputs/dtu_${LSB_JOBID}}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
if [[ -n "$BASELINE_SOURCE" && ! -f "$OUTPUT_DIR/baseline_model.keras" ]]; then
    [[ -f "$BASELINE_SOURCE" ]] || {
        echo "BASELINE_SOURCE does not exist: $BASELINE_SOURCE" >&2
        exit 1
    }
    cp "$BASELINE_SOURCE" "$OUTPUT_DIR/baseline_model.keras"
fi
# Keep scheduler output in the submission directory and a complete log with artifacts.
exec > >(tee -a "$OUTPUT_DIR/job_${LSB_JOBID}.log") 2>&1
trap 'status=$?; printf "Job %s finished with exit status %s\n" "$LSB_JOBID" "$status"; exit "$status"' EXIT
printf 'Job: %s\nHost: %s\nMode: %s\nArtifacts: %s\n' "$LSB_JOBID" "$(hostname)" "$MODE" "$OUTPUT_DIR"
date -Is
nvidia-smi
python -m pip freeze > "$OUTPUT_DIR/environment_${LSB_JOBID}.txt"

# Fail promptly if GPU libraries or allocation are wrong; do not fall back to CPU.
python - <<'PY'
from common import tf
gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise SystemExit("No TensorFlow GPU detected. Check allocation, driver and CUDA libraries.")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
with tf.device("/GPU:0"):
    value = tf.linalg.matmul(tf.ones((64, 64)), tf.ones((64, 64)))
    value.numpy()
if "GPU" not in value.device:
    raise SystemExit("GPU preflight operation was placed on CPU.")
print("TensorFlow:", tf.__version__, "GPUs:", gpus, "device:", value.device)
PY

if [[ "$MODE" == seeds ]]; then
    [[ -f "$OUTPUT_DIR/baseline_model.keras" ]] || {
        echo 'Set BASELINE_SOURCE to an existing trained baseline.' >&2; exit 1;
    }
    python -u scripts/evaluate_seeds.py --baseline "$OUTPUT_DIR/baseline_model.keras" \
        --output-dir "$OUTPUT_DIR" --epochs "${FINETUNE_EPOCHS:-10}" --num-seeds 3
    exit 0
fi

args=(--output-dir "$OUTPUT_DIR" --batch-size "${BATCH_SIZE:-64}"
      --baseline-epochs "${BASELINE_EPOCHS:-100}"
      --finetune-epochs "${FINETUNE_EPOCHS:-10}"
      --distill-epochs "${DISTILL_EPOCHS:-30}" --seed "${SEED:-42}")
if [[ -n "$PARTS" ]]; then
    read -r -a selected_parts <<< "$PARTS"
    args+=(--parts "${selected_parts[@]}")
fi
case "$MODE" in
    all) [[ "$EXTENDED" != 1 ]] || args+=(--extended) ;;
    baseline) args+=(--baseline-only) ;;
    quick) args+=(--quick) ;;
    *) echo "MODE must be all, baseline, or quick." >&2; exit 1 ;;
esac
printf 'Command:'; printf ' %q' python -u main.py "${args[@]}"; printf '\n'
python -u main.py "${args[@]}"
echo "Completed. Models, checkpoints, benchmarks and figures: $OUTPUT_DIR"
