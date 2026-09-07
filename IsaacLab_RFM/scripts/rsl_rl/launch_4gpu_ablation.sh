#!/usr/bin/env bash

# Four independent, single-GPU ablations (not DDP).
# GPU0: full model; GPU1: no latent; GPU2: DreamWaQ-style CENet;
# GPU3: additive rewards with all active phase/safety gates disabled.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
EXT_LOCO_DIR="${PROJECT_DIR}/source/ext_loco"

export WBC_LOG_ROOT="${WBC_LOG_ROOT:-${PROJECT_DIR}/logs/rsl_rl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${EXT_LOCO_DIR}:${PROJECT_DIR}/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="$(command -v "${PYTHON_BIN:-python}")"
TASK="Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-v0"
NUM_ENVS=8192
MAX_ITERATIONS=20000
SEED="${ABLATION_SEED:-42}"
EE_POS_Z_MIN=0.2
EE_POS_Z_MAX=2.0
GPU_CSV="${GPU_IDS:-0,1,2,3}"
FOREGROUND=false
CHECK_CONFIG=false

while (( $# > 0 )); do
    case "$1" in
        --foreground) FOREGROUND=true ;;
        --check-config) CHECK_CONFIG=true ;;
        --gpus) GPU_CSV="$2"; shift ;;
        --gpus=*) GPU_CSV="${1#*=}" ;;
        *) echo "Usage: $0 [--foreground] [--check-config] [--gpus 0,1,2,3]" >&2; exit 2 ;;
    esac
    shift
done

if [[ ! "${GPU_CSV}" =~ ^[0-3](,[0-3])*$ ]]; then
    echo "ERROR: --gpus must contain unique experiment/GPU ids from 0,1,2,3." >&2
    exit 2
fi
IFS=',' read -r -a GPUS <<< "${GPU_CSV}"
declare -A SEEN=()
for gpu in "${GPUS[@]}"; do
    [[ -z "${SEEN[$gpu]:-}" ]] || { echo "ERROR: duplicate GPU ${gpu}." >&2; exit 2; }
    SEEN[$gpu]=1
done

EXPERIMENT_NAMES=(full_baseline no_latent dreamwaq_cenet additive_reward)

build_overrides() {
    local gpu="$1"
    local output_name="$2"
    local -n output_ref="${output_name}"
    output_ref=("env.commands.EE_pose.ranges.pos_z=[${EE_POS_Z_MIN},${EE_POS_Z_MAX}]")
    case "${gpu}" in
        0) ;;
        1)
            output_ref+=("agent.ppo_algorithm.use_latent=false")
            ;;
        2)
            output_ref+=(
                "agent.contactNet.class_name=CENet"
                "agent.gru.class_name=IdentityGRUWrapper"
            )
            ;;
        3)
            output_ref+=(
                "env.rewards.safety_exp.params.use_gates=false"
                "env.rewards.track_EE_position_exp.params.use_gates=false"
                "env.rewards.track_EE_orientation_fine_exp.params.use_gates=false"
                "env.rewards.track_EE_pb.params.use_walking_gate=false"
                "env.rewards.track_EE_pb.params.use_safety_gate=false"
                "env.rewards.foot_flat_l2.params.use_standing_gate=false"
                "env.rewards.feet_contacts_reg.params.use_standing_gate=false"
                "env.rewards.feet_contacts_reg.params.use_position_gate=false"
            )
            ;;
    esac
}

print_matrix() {
    echo "Four-GPU ablation: ${NUM_ENVS} envs, ${MAX_ITERATIONS} iterations, seed ${SEED}"
    for gpu in "${GPUS[@]}"; do
        local_overrides=()
        build_overrides "${gpu}" local_overrides
        echo "GPU ${gpu}: ${EXPERIMENT_NAMES[$gpu]}"
        printf '  %s\n' "${local_overrides[@]}"
    done
}

validate_source() {
    local origin
    origin="$("${PYTHON_BIN}" -c 'import importlib.util, os; s=importlib.util.find_spec("ext_loco"); print(os.path.realpath(s.origin) if s and s.origin else "")')"
    [[ "${origin}" == "${EXT_LOCO_DIR}/"* ]] || {
        echo "ERROR: ext_loco resolves outside this checkout: ${origin:-not found}" >&2
        return 1
    }
    # PPO imports Isaac Lab utilities that become available after AppLauncher
    # starts Kit, so preflight only imports the standalone torch modules here.
    "${PYTHON_BIN}" -c 'from isaaclab.app import AppLauncher; from rsl_rl.modules import CENet, IdentityGRUWrapper'
}

preflight() {
    validate_source
    command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found." >&2; return 1; }
    local gpu_count
    gpu_count="$(nvidia-smi -L | wc -l)"
    for gpu in "${GPUS[@]}"; do
        (( gpu < gpu_count )) || { echo "ERROR: GPU ${gpu} unavailable (${gpu_count} detected)." >&2; return 1; }
    done
    local busy
    busy="$(nvidia-smi -i "${GPU_CSV}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
    [[ -z "${busy}" ]] || { echo "ERROR: selected GPUs already have compute processes: ${busy}" >&2; return 1; }
}

if [[ "${CHECK_CONFIG}" == true ]]; then
    validate_source
    print_matrix
    exit 0
fi

STAMP="${LAUNCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
SESSION="ablation_${STAMP}"

if [[ "${FOREGROUND}" == false && -z "${TMUX:-}" ]]; then
    command -v tmux >/dev/null || { echo "ERROR: tmux not found; use --foreground." >&2; exit 1; }
    preflight
    printf -v TMUX_CMD \
        'exec env PYTHON_BIN=%q PYTHONPATH=%q WBC_LOG_ROOT=%q PYTORCH_CUDA_ALLOC_CONF=%q ABLATION_SEED=%q LAUNCH_STAMP=%q bash %q --foreground --gpus %q' \
        "${PYTHON_BIN}" "${PYTHONPATH}" "${WBC_LOG_ROOT}" "${PYTORCH_CUDA_ALLOC_CONF}" \
        "${SEED}" "${STAMP}" "${SCRIPT_PATH}" "${GPU_CSV}"
    tmux new-session -d -s "${SESSION}" -c "${SCRIPT_DIR}" "${TMUX_CMD}"
    echo "Started tmux session ${SESSION}."
    echo "Attach: tmux attach -t ${SESSION}"
    echo "Logs:   ${WBC_LOG_ROOT}/ablation_launcher_${STAMP}"
    exit 0
fi

command -v flock >/dev/null || { echo "ERROR: flock not found." >&2; exit 1; }
LOCK_FDS=()
for gpu in "${GPUS[@]}"; do
    lock_path="${TMPDIR:-/tmp}/umi_ablation_gpu${gpu}.lock"
    exec {lock_fd}>"${lock_path}"
    flock -n "${lock_fd}" || { echo "ERROR: GPU ${gpu} is reserved by another ablation launcher." >&2; exit 1; }
    LOCK_FDS+=("${lock_fd}")
done
preflight
print_matrix

CONSOLE_DIR="${WBC_LOG_ROOT}/ablation_launcher_${STAMP}"
[[ ! -e "${CONSOLE_DIR}" ]] || { echo "ERROR: ${CONSOLE_DIR} already exists." >&2; exit 1; }
mkdir -p "${CONSOLE_DIR}"

PIDS=()
cleanup() {
    local status=$?
    if (( status != 0 )); then
        for pid in "${PIDS[@]:-}"; do kill "${pid}" 2>/dev/null || true; done
        echo "Ablation launcher failed; logs preserved in ${CONSOLE_DIR}." >&2
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "${SCRIPT_DIR}"
for gpu in "${GPUS[@]}"; do
    experiment="${EXPERIMENT_NAMES[$gpu]}"
    run_name="ablation_${experiment}_gpu${gpu}_${STAMP}"
    console_log="${CONSOLE_DIR}/${run_name}.log"
    usd_dir="/tmp/IsaacLab/${STAMP}_ablation_gpu${gpu}"
    overrides=()
    build_overrides "${gpu}" overrides
    echo "Launching GPU ${gpu}: ${experiment} -> ${console_log}"
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
            "${overrides[@]}" \
            "agent.wandb_run_name=${run_name}"
    ) >"${console_log}" 2>&1 &
    PIDS+=("$!")
done

sleep 10
for index in "${!PIDS[@]}"; do
    kill -0 "${PIDS[$index]}" 2>/dev/null || {
        echo "ERROR: GPU ${GPUS[$index]} exited during startup; inspect ${CONSOLE_DIR}." >&2
        exit 1
    }
done
echo "All ${#PIDS[@]} runs survived initial startup."

status=0
for pid in "${PIDS[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
