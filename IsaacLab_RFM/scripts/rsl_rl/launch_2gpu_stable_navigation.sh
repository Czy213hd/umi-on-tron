#!/usr/bin/env bash

# Two low-speed, minimally shaped fine-tunes from the full baseline.
# GPU1: gentle navigation; GPU2: slower navigation plus walking-only arm tuck.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
EXT_LOCO_DIR="${PROJECT_DIR}/source/ext_loco"

export WBC_LOG_ROOT="${WBC_LOG_ROOT:-${PROJECT_DIR}/logs/rsl_rl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${EXT_LOCO_DIR}:${PROJECT_DIR}/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniforge3/envs/isaaclab_umi_on_tron/bin/python}"
TASK="Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-v0"
BASELINE_RUN="2026-08-03_17-09-34_ablation_full_baseline_gpu0_20260803_170924"
STAMP="${LAUNCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-8000}"
SEED="${ABLATION_SEED:-42}"
CONSOLE_DIR="${WBC_LOG_ROOT}/stable_navigation_launcher_${STAMP}"

mkdir -p "${CONSOLE_DIR}"
cd "${SCRIPT_DIR}"

PIDS=()
for gpu in 1 2; do
    if [[ "${gpu}" == 1 ]]; then
        variant="gentle"
        overrides=(
            env.rewards.track_EE_pb.weight=8.0
            env.rewards.base_target_heading_alignment.weight=0.8
            env.rewards.target_directed_base_velocity_exp.weight=2.0
            env.rewards.target_directed_base_velocity_exp.params.max_speed=0.35
            env.rewards.target_directed_base_velocity_exp.params.slowdown_distance=1.0
            env.rewards.action_rate_l2.weight=-0.9
        )
    else
        variant="graceful_arm_tuck"
        overrides=(
            env.rewards.track_EE_pb.weight=6.0
            env.rewards.base_target_heading_alignment.weight=1.0
            env.rewards.target_directed_base_velocity_exp.weight=2.5
            env.rewards.target_directed_base_velocity_exp.params.max_speed=0.25
            env.rewards.target_directed_base_velocity_exp.params.slowdown_distance=1.2
            env.rewards.action_rate_l2.weight=-1.0
            env.rewards.walking_arm_deviation_l2.weight=-0.4
        )
    fi

    run_name="ablation_stable_navigation_${variant}_gpu${gpu}_${STAMP}"
    console_log="${CONSOLE_DIR}/${run_name}.log"
    usd_dir="/tmp/IsaacLab/${STAMP}_stable_navigation_gpu${gpu}"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        exec "${PYTHON_BIN}" ios_train.py \
            --headless \
            --device=cuda:0 \
            --task="${TASK}" \
            --run_name="${run_name}" \
            --seed="${SEED}" \
            --asset_usd_dir="${usd_dir}" \
            --num_envs="${NUM_ENVS}" \
            --max_iterations="${MAX_ITERATIONS}" \
            --logger=wandb \
            --load_run="${BASELINE_RUN}" \
            --checkpoint=model_19999.pt \
            agent.ppo_algorithm.learning_rate=1.0e-4 \
            "env.commands.EE_pose.ranges.pos_z=[0.2,2.0]" \
            "${overrides[@]}" \
            "agent.wandb_run_name=${run_name}"
    ) >"${console_log}" 2>&1 &
    PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
