#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

RUN_NAME_TO_LOAD="2026-07-30_00-17-56"
CHECKPOINT_NAME="model_19999.pt"
LOG_ROOT="${PROJECT_DIR}/logs/rsl_rl/ImplicitOneStageARXR5Arm"
CHECKPOINT_PATH="${LOG_ROOT}/${RUN_NAME_TO_LOAD}/${CHECKPOINT_NAME}"

# This runner interprets max_iterations as the number of additional iterations.
# For example, ADDITIONAL_ITERATIONS=20000 resumes near iteration 19999 and
# trains until approximately iteration 39998.
ADDITIONAL_ITERATIONS="${ADDITIONAL_ITERATIONS:-20000}"
NUM_ENVS="${NUM_ENVS:-8192}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-resume_model_19999}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT_PATH}" >&2
    exit 1
fi

if [[ ! "${ADDITIONAL_ITERATIONS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: ADDITIONAL_ITERATIONS must be a positive integer." >&2
    exit 2
fi

export WBC_LOG_ROOT="${LOG_ROOT}"
export MAX_ITERATIONS="${ADDITIONAL_ITERATIONS}"
export NUM_ENVS
export GPU_ID
export SEED
export RUN_NAME

echo "[resume] checkpoint=${CHECKPOINT_PATH}"
echo "[resume] additional_iterations=${ADDITIONAL_ITERATIONS}"
echo "[resume] new_run_name=${RUN_NAME}"

exec "${SCRIPT_DIR}/train_sf_j6.sh" \
    --load_run "${RUN_NAME_TO_LOAD}" \
    --checkpoint "${CHECKPOINT_NAME}" \
    "$@"
