#!/usr/bin/env bash

# Train the scale-matched ungated reward ablation derived from a shared-state rollout.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
PYTHON_BIN="${PYTHON_BIN:-/data/miniforge3/envs/isaaclab_umi_on_tron/bin/python}"
GPU_ID="${GPU_ID:-3}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
SEED="${SEED:-42}"
STAMP="${LAUNCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="ablation_matched_additive_gpu${GPU_ID}_${STAMP}"
SESSION="matched_additive_${STAMP}"
LOG_ROOT="${WBC_LOG_ROOT:-${PROJECT_DIR}/logs/rsl_rl}"
CONSOLE_DIR="${LOG_ROOT}/matched_additive_launcher_${STAMP}"
USD_DIR="${PROJECT_DIR}/evaluation/usd_cache/${RUN_NAME}"
OVERRIDES_FILE="${PROJECT_DIR}/evaluation/reward_matching_baseline_20260805/matched_additive_overrides.txt"

export PYTHONPATH="${PROJECT_DIR}/source/ext_loco:${PROJECT_DIR}/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 1; }
[[ -s "${OVERRIDES_FILE}" ]] || { echo "Missing overrides: ${OVERRIDES_FILE}" >&2; exit 1; }
mapfile -t MATCHED_OVERRIDES <"${OVERRIDES_FILE}"

run_training() {
    mkdir -p "${CONSOLE_DIR}" "${USD_DIR}"
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
    cd "${SCRIPT_DIR}"
    exec "${PYTHON_BIN}" ios_train.py \
        --headless \
        --device=cuda:0 \
        --task=Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-v0 \
        --run_name="${RUN_NAME}" \
        --seed="${SEED}" \
        --asset_usd_dir="${USD_DIR}" \
        --num_envs="${NUM_ENVS}" \
        --max_iterations="${MAX_ITERATIONS}" \
        --logger=wandb \
        "${MATCHED_OVERRIDES[@]}" \
        "agent.wandb_run_name=${RUN_NAME}"
}

if [[ "${1:-}" == "--foreground" ]]; then
    run_training
fi

command -v tmux >/dev/null || { echo "tmux is required" >&2; exit 1; }
if nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
    echo "GPU ${GPU_ID} already has a compute process." >&2
    exit 1
fi
mkdir -p "${CONSOLE_DIR}"
printf -v COMMAND \
    'exec env GPU_ID=%q NUM_ENVS=%q MAX_ITERATIONS=%q SEED=%q LAUNCH_STAMP=%q PYTHON_BIN=%q WBC_LOG_ROOT=%q bash %q --foreground >%q 2>&1' \
    "${GPU_ID}" "${NUM_ENVS}" "${MAX_ITERATIONS}" "${SEED}" "${STAMP}" "${PYTHON_BIN}" "${LOG_ROOT}" \
    "${SCRIPT_PATH}" "${CONSOLE_DIR}/${RUN_NAME}.log"
tmux new-session -d -s "${SESSION}" -c "${SCRIPT_DIR}" "${COMMAND}"
echo "Started ${SESSION}"
echo "Run: ${RUN_NAME}"
echo "Log: ${CONSOLE_DIR}/${RUN_NAME}.log"
