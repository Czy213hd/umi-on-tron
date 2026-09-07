#!/usr/bin/env bash

# Full privileged-state Oracle: the actor directly consumes the complete critic
# observation. This is a simulator-only performance upper bound, not deployable.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
EXT_LOCO_DIR="${PROJECT_DIR}/source/ext_loco"

export WBC_LOG_ROOT="${WBC_LOG_ROOT:-${PROJECT_DIR}/logs/rsl_rl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${EXT_LOCO_DIR}:${PROJECT_DIR}/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="$(command -v "${PYTHON_BIN:-python}")"
TASK="Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-v0"
GPU_ID="${GPU_ID:-0}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
SEED="${ABLATION_SEED:-42}"
STAMP="${LAUNCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="ablation_privileged_oracle_gpu${GPU_ID}_${STAMP}"
CONSOLE_DIR="${WBC_LOG_ROOT}/ablation_launcher_${STAMP}"
CONSOLE_LOG="${CONSOLE_DIR}/${RUN_NAME}.log"
USD_DIR="/tmp/IsaacLab/${STAMP}_privileged_oracle_gpu${GPU_ID}"

mkdir -p "${CONSOLE_DIR}"
cd "${SCRIPT_DIR}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
exec "${PYTHON_BIN}" ios_train.py \
    --headless \
    --device=cuda:0 \
    --task="${TASK}" \
    --run_name="${RUN_NAME}" \
    --seed="${SEED}" \
    --asset_usd_dir="${USD_DIR}" \
    --num_envs="${NUM_ENVS}" \
    --max_iterations="${MAX_ITERATIONS}" \
    --logger=wandb \
    "env.commands.EE_pose.ranges.pos_z=[0.2,2.0]" \
    agent.ppo_algorithm.use_latent=false \
    agent.ppo_algorithm.use_privileged_actor=true \
    "agent.wandb_run_name=${RUN_NAME}" \
    >"${CONSOLE_LOG}" 2>&1
