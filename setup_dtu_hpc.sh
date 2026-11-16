#!/usr/bin/env bash
# Run once on DTU after copying the source files (not the Windows .venv).
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if ! type module >/dev/null 2>&1; then
    [[ ! -f /etc/profile.d/modules.sh ]] || source /etc/profile.d/modules.sh
fi
if ! type module >/dev/null 2>&1; then
    echo "The DTU module command is unavailable. Run this on the cluster." >&2
    exit 1
fi

# Change with: PYTHON_MODULE=python3/<available-version> bash setup_dtu_hpc.sh
# Check available names with: module avail python3
PYTHON_MODULE="${PYTHON_MODULE:-python3/3.10.13}"
module purge
module load "$PYTHON_MODULE"
export PYTHONNOUSERSITE=1
python3 - <<'PY'
import sys
if not (3, 10) <= sys.version_info[:2] <= (3, 12):
    raise SystemExit("Select a Python 3.10, 3.11 or 3.12 module.")
PY

VENV_DIR="$PROJECT_DIR/.venv-hpc"
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Expected a Linux environment at $VENV_DIR; do not copy the Windows .venv." >&2
    exit 1
fi
"$VENV_DIR/bin/python" -m pip install --upgrade pip
# The CUDA extra provides matching user-space CUDA/cuDNN libraries. No CUDA
# system modules are loaded, avoiding a mixture of library versions.
"$VENV_DIR/bin/python" -m pip install -r requirements.txt 'tensorflow[and-cuda]==2.16.2'
"$VENV_DIR/bin/python" -m pip check
printf '%s\n' "$PYTHON_MODULE" > "$VENV_DIR/dtu-python-module.txt"

export TF_USE_LEGACY_KERAS=1
export KERAS_HOME="$PROJECT_DIR/.keras"
export MPLBACKEND=Agg
mkdir -p "$KERAS_HOME"
# Fetch data and pretrained weights now so the batch job need not download them.
# This loads the architecture on CPU; it does not train on the login node.
CUDA_VISIBLE_DEVICES='' "$VENV_DIR/bin/python" - <<'PY'
from common import tf
tf.keras.datasets.cifar100.load_data(label_mode="fine")
tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=(96, 96, 3))
print("CIFAR-100 and ImageNet weights are cached.")
PY
echo "Setup complete. Submit from this directory: bsub < run_dtu_hpc.sh"
