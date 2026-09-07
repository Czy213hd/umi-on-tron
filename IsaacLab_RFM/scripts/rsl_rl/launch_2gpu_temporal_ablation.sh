#!/usr/bin/env bash

# GPU0: GRU-only (latest observation projection + GRU)
# GPU1: Transformer-only (Transformer + identity recurrent stage)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
EXT_LOCO_DIR="${PROJECT_DIR}/source/ext_loco"

export WBC_LOG_ROOT="${WBC_LOG_ROOT:-${PROJECT_DIR}/logs/rsl_rl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${EXT_LOCO_DIR}:${PROJECT_DIR}/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-/data/miniforge3/envs/isaaclab_umi_on_tron/bin/python}"
TASK="Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-v0"
NUM_ENVS="${NUM_ENVS:-8192}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
SEED="${ABLATION_SEED:-42}"
STAMP="${LAUNCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
FOREGROUND=false

if [[ "${1:-}" == "--foreground" ]]; then
    FOREGROUND=true
elif (( $# > 0 )); then
    echo "Usage: $0 [--foreground]" >&2
    exit 2
fi

if [[ "${FOREGROUND}" == false && -z "${TMUX:-}" ]]; then
    session="temporal_ablation_${STAMP}"
    printf -v cmd \
        'exec env PYTHON_BIN=%q PYTHONPATH=%q WBC_LOG_ROOT=%q NUM_ENVS=%q MAX_ITERATIONS=%q ABLATION_SEED=%q LAUNCH_STAMP=%q bash %q --foreground' \
        "${PYTHON_BIN}" "${PYTHONPATH}" "${WBC_LOG_ROOT}" "${NUM_ENVS}" \
        "${MAX_ITERATIONS}" "${SEED}" "${STAMP}" "${SCRIPT_PATH}"
    tmux new-session -d -s "${session}" -c "${SCRIPT_DIR}" "${cmd}"
    echo "Started tmux session ${session}"
    exit 0
fi

busy="$(nvidia-smi -i 0,1 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
[[ -z "${busy}" ]] || { echo "ERROR: GPU0/1 already busy: ${busy}" >&2; exit 1; }

console_dir="${WBC_LOG_ROOT}/temporal_ablation_launcher_${STAMP}"
mkdir -p "${console_dir}"
pids=()

launch() {
    local gpu="$1" name="$2" contact_class="$3" gru_class="$4"
    local run_name="ablation_${name}_gpu${gpu}_${STAMP}"
    echo "Launching GPU${gpu}: ${run_name}"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        exec "${PYTHON_BIN}" "${SCRIPT_DIR}/ios_train.py" \
            --headless --device=cuda:0 --task="${TASK}" \
            --run_name="${run_name}" --seed="${SEED}" \
            --asset_usd_dir="/tmp/IsaacLab/${STAMP}_${name}_gpu${gpu}" \
            --num_envs="${NUM_ENVS}" --max_iterations="${MAX_ITERATIONS}" \
            --logger=wandb \
            "env.commands.EE_pose.ranges.pos_z=[0.2,2.0]" \
            "agent.contactNet.class_name=${contact_class}" \
            "agent.gru.class_name=${gru_class}" \
            "agent.wandb_run_name=${run_name}"
    ) >"${console_dir}/${run_name}.log" 2>&1 &
    pids+=("$!")
}

launch 0 gru_only LastObservationEncoder GRUWrapper
launch 1 transformer_only SimplifiedContactNetModel IdentityGRUWrapper

sleep 15
for pid in "${pids[@]}"; do
    kill -0 "${pid}" 2>/dev/null || {
        echo "ERROR: training process ${pid} exited during startup; inspect ${console_dir}" >&2
        exit 1
    }
done
echo "Both temporal ablations survived startup. Logs: ${console_dir}"

status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
