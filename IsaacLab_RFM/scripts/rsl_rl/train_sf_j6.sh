#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/phi5090ii/UMI-ON-TRON/conda_envs/isaaclab_tron/bin/python}"
TASK="Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-v0"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-j6_eef}"
GPU_ID="${GPU_ID:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PROJECT_DIR}/rsl_rl:${PROJECT_DIR}/source/ext_loco${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${PYTHON_BIN%/bin/python}/lib/python3.10/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export WBC_LOG_ROOT="${WBC_LOG_ROOT:-${PROJECT_DIR}/logs/rsl_rl/ImplicitOneStageARXR5Arm}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python interpreter is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

mkdir -p "${WBC_LOG_ROOT}"
cd "${PROJECT_DIR}"

echo "[train] task=${TASK}"
echo "[train] EEF frame=link6 (J6 child body)"
echo "[train] GPU=${GPU_ID}, num_envs=${NUM_ENVS}, iterations=${MAX_ITERATIONS}, seed=${SEED}"
echo "[train] logs=${WBC_LOG_ROOT}"

exec "${PYTHON_BIN}" scripts/rsl_rl/ios_train.py \
    --task "${TASK}" \
    --headless \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${MAX_ITERATIONS}" \
    --seed "${SEED}" \
    --run_name "${RUN_NAME}" \
    "$@"
