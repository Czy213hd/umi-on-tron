#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ISAACLAB_RFM_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
EXT_LOCO_SOURCE_DIR="${ISAACLAB_RFM_DIR}/source/ext_loco"

export WBC_LOG_ROOT="${WBC_LOG_ROOT:-${ISAACLAB_RFM_DIR}/logs/rsl_rl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# The conda environment may have an editable ext_loco install pointing at a
# different worktree. Put this checkout first so the sweep always trains the
# code next to this launcher.
export PYTHONPATH="${EXT_LOCO_SOURCE_DIR}:${ISAACLAB_RFM_DIR}/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="$(command -v "${PYTHON_BIN:-python}")"
STARTUP_GRACE_SECONDS="${STARTUP_GRACE_SECONDS:-180}"
task="Template-Isaac-EEPose-Flat-Limx-SF-Tron1A-v0"

foreground=false
check_config=false

selected_gpu_csv="${GPU_IDS:-0,1,2,3}"

while (( $# > 0 )); do
    case "$1" in
        --foreground)
            foreground=true
            ;;
        --check-config)
            check_config=true
            ;;
        --gpus)
            if (( $# < 2 )); then
                echo "ERROR: --gpus requires a comma-separated GPU list (for example, 0,1)." >&2
                exit 2
            fi
            selected_gpu_csv="$2"
            shift
            ;;
        --gpus=*)
            selected_gpu_csv="${1#*=}"
            ;;
        *)
            echo "Usage: $0 [--foreground] [--check-config] [--gpus GPU_IDS]" >&2
            exit 2
            ;;
    esac
    shift
done

if [[ ! "${selected_gpu_csv}" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]]; then
    echo "ERROR: GPU_IDS must be a comma-separated list of non-negative GPU indices (for example, 0,1)." >&2
    exit 2
fi
IFS=',' read -r -a ACTIVE_GPUS <<< "${selected_gpu_csv}"
declare -A seen_gpus=()
for gpu in "${ACTIVE_GPUS[@]}"; do
    if [[ -n "${seen_gpus[$gpu]:-}" ]]; then
        echo "ERROR: GPU ${gpu} is listed more than once in --gpus." >&2
        exit 2
    fi
    seen_gpus[$gpu]=1
done
worker_count="${#ACTIVE_GPUS[@]}"

if [[ ! "${STARTUP_GRACE_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: STARTUP_GRACE_SECONDS must be a positive integer." >&2
    exit 2
fi

# Every enabled, non-zero reward is explicit here. This makes each run
# reproducible even if the task's Python defaults change later.
readonly -a reward_terms=(
    safety_exp
    track_EE_position_exp
    track_EE_orientation_exp
    track_EE_pb
    base_target_heading_alignment
    dof_weighted_torques_l2
    dof_weighted_power_l1
    action_rate_l2
    base_pitch_exp
    dof_vel_ankle_l2
    dof_vel_non_ankle_l2
    dof_vel_arm_l2
    dof_non_ankle_pos_limits
    feet_contacts_reg
    knee_contacts
    foot_flat_l2
    foot_slip_l2
    legs_min_separation
    base_height
    termination_penalty
)

# Per-GPU reward sweep. Array index == physical GPU index.
#                                               GPU 0    GPU 1    GPU 2    GPU 3
safety_exp_weights=(                              2.0      2.0      2.0      2.0 )
track_EE_position_exp_weights=(                   6.0      6.0      6.0      6.0 )
track_EE_orientation_exp_weights=(               12.0      9.0      3.0      6.0 )
track_EE_pb_weights=(                            15.0     15.0     15.0     15.0 )
base_target_heading_alignment_weights=(           0.5      0.5      1.0      0.5 )
dof_weighted_torques_l2_weights=(              -4.0e-5  -4.0e-5  -4.0e-5  -4.0e-5 )
dof_weighted_power_l1_weights=(                -2.5e-4  -2.5e-4  -2.5e-4  -2.5e-4 )
action_rate_l2_weights=(                         -0.7     -0.7     -0.7     -0.7 )
base_pitch_exp_weights=(                         -0.5     -0.5     -1.0     -0.5 )
dof_vel_ankle_l2_weights=(                     -5.0e-4  -5.0e-4  -5.0e-4  -5.0e-4 )
dof_vel_non_ankle_l2_weights=(                 -5.0e-4  -5.0e-4  -5.0e-4  -5.0e-4 )
dof_vel_arm_l2_weights=(                       -5.0e-4  -5.0e-4  -5.0e-4  -5.0e-4 )
dof_non_ankle_pos_limits_weights=(              -10.0    -10.0    -10.0    -10.0 )
feet_contacts_reg_weights=(                       0.5      0.5      0.5      0.5 )
knee_contacts_weights=(                       -1000.0  -1000.0  -1000.0  -1000.0 )
foot_flat_l2_weights=(                           -2.0     -2.0     -2.0     -2.0 )
foot_slip_l2_weights=(                           -2.0     -2.0     -2.0     -2.0 )
legs_min_separation_weights=(                    -7.0     -7.0     -7.0     -7.0 )
base_height_weights=(                            -1.0     -1.0     -1.0     -1.0 )
termination_penalty_weights=(                 -1000.0  -1000.0  -1000.0  -1000.0 )
pb_use_walking_gate=(                            true     true     true     true )

readonly reward_term_csv="$(IFS=,; printf '%s' "${reward_terms[*]}")"
readonly configured_gpu_count=4

# Action scale groups:
#   legs: abad/hip/knee/ankle on both legs
#   J1-3: proximal arm joints
#   J4-6: distal arm joints
leg_action_scales=(                     0.6    0.6    0.6    0.6 )
j1_3_action_scales=(                    0.3    0.3    0.3    0.3 )
j4_6_action_scales=(                    0.2    0.2    0.2    0.2 )

num_envs=8192
max_iterations=20000

reward_weight_for_gpu() {
    local term="$1"
    local gpu="$2"
    local weight_array_name="${term}_weights"
    local -n weights_ref="${weight_array_name}"

    printf '%s' "${weights_ref[$gpu]}"
}

validate_matrix_row() {
    local row_name="$1"
    local -n row_ref="${row_name}"

    if (( ${#row_ref[@]} != configured_gpu_count )); then
        echo "ERROR: ${row_name} must contain exactly ${configured_gpu_count} GPU values; found ${#row_ref[@]}." >&2
        return 1
    fi
}

validate_experiment_matrix() {
    local term

    for term in "${reward_terms[@]}"; do
        validate_matrix_row "${term}_weights"
    done
    validate_matrix_row pb_use_walking_gate
    validate_matrix_row leg_action_scales
    validate_matrix_row j1_3_action_scales
    validate_matrix_row j4_6_action_scales
}

validate_experiment_matrix
for gpu in "${ACTIVE_GPUS[@]}"; do
    if (( gpu >= configured_gpu_count )); then
        echo "ERROR: GPU ${gpu} has no configured sweep parameters; configured GPU indices are 0-$(( configured_gpu_count - 1 ))." >&2
        exit 2
    fi
done

build_reward_overrides() {
    local gpu="$1"
    local output_name="$2"
    local term
    local -n output_ref="${output_name}"

    output_ref=()
    for term in "${reward_terms[@]}"; do
        output_ref+=("env.rewards.${term}.weight=$(reward_weight_for_gpu "${term}" "${gpu}")")
    done
    output_ref+=("env.rewards.track_EE_pb.params.use_walking_gate=${pb_use_walking_gate[$gpu]}")
}

print_reward_config() {
    local gpu="$1"
    local indent="$2"
    local override
    local -a reward_overrides=()

    build_reward_overrides "${gpu}" reward_overrides
    for override in "${reward_overrides[@]}"; do
        printf '%s%s\n' "${indent}" "${override#env.rewards.}"
    done
}

build_action_scale_override() {
    local gpu="$1"

    # Hydra requires square brackets in dictionary keys to be escaped.
    printf 'env.actions.joint_pos.scale={abad_\\[RL\\]_Joint|hip_\\[RL\\]_Joint|knee_\\[RL\\]_Joint|ankle_\\[RL\\]_Joint:%s,J\\[1-3\\]:%s,J\\[4-6\\]:%s}' \
        "${leg_action_scales[$gpu]}" \
        "${j1_3_action_scales[$gpu]}" \
        "${j4_6_action_scales[$gpu]}"
}

check_training_source() {
    local ext_loco_origin

    ext_loco_origin="$("${PYTHON_BIN}" -c \
        'import importlib.util, os; spec = importlib.util.find_spec("ext_loco"); print(os.path.realpath(spec.origin) if spec and spec.origin else "")')"
    if [[ "${ext_loco_origin}" != "${EXT_LOCO_SOURCE_DIR}/"* ]]; then
        echo "ERROR: ext_loco resolves outside this checkout: ${ext_loco_origin:-not found}" >&2
        echo "Expected it under: ${EXT_LOCO_SOURCE_DIR}" >&2
        return 1
    fi
}

validate_action_scale_override() {
    local gpu="$1"
    local override

    override="$(build_action_scale_override "${gpu}")"
    "${PYTHON_BIN}" - \
        "${override}" \
        "${leg_action_scales[$gpu]}" \
        "${j1_3_action_scales[$gpu]}" \
        "${j4_6_action_scales[$gpu]}" <<'PY'
import math
import sys

from hydra.core.override_parser.overrides_parser import OverridesParser

override, leg, j1_3, j4_6 = sys.argv[1:]
parsed = OverridesParser.create().parse_override(override)
if parsed.get_key_element() != "env.actions.joint_pos.scale":
    raise SystemExit(f"unexpected Hydra key: {parsed.get_key_element()}")

actual = parsed.value()
expected = {
    "abad_[RL]_Joint|hip_[RL]_Joint|knee_[RL]_Joint|ankle_[RL]_Joint": float(leg),
    "J[1-3]": float(j1_3),
    "J[4-6]": float(j4_6),
}
if actual.keys() != expected.keys():
    raise SystemExit(f"unexpected action scale groups: {actual}")
for key, expected_value in expected.items():
    actual_value = float(actual[key])
    if not math.isfinite(actual_value) or actual_value <= 0.0:
        raise SystemExit(f"action scale must be finite and positive: {key}={actual_value}")
    if not math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=1.0e-12):
        raise SystemExit(f"action scale mismatch: {key}={actual_value}, expected {expected_value}")
PY
}

validate_reward_overrides() {
    local gpu="$1"
    local -a reward_overrides=()

    build_reward_overrides "${gpu}" reward_overrides
    "${PYTHON_BIN}" - \
        "${reward_term_csv}" \
        "${reward_overrides[@]}" <<'PY'
import math
import sys

from hydra.core.override_parser.overrides_parser import OverridesParser

expected_terms = sys.argv[1].split(",")
expected_keys = tuple(
    [f"env.rewards.{term}.weight" for term in expected_terms]
    + ["env.rewards.track_EE_pb.params.use_walking_gate"]
)
overrides = OverridesParser.create().parse_overrides(sys.argv[2:])
actual_keys = tuple(override.get_key_element() for override in overrides)
if actual_keys != expected_keys:
    raise SystemExit(f"unexpected reward override keys: {actual_keys}")

for term, override in zip(expected_terms, overrides[:-1]):
    value = override.value()
    if isinstance(value, bool):
        raise SystemExit(f"reward weight must be numeric, not boolean: {term}={value!r}")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"reward weight must be numeric: {term}={value!r}") from exc
    if not math.isfinite(value) or value == 0.0:
        raise SystemExit(f"reward weight must be finite and non-zero: {term}={value}")

if not isinstance(overrides[-1].value(), bool):
    raise SystemExit(f"PB walking-gate override is not boolean: {overrides[-1].value()!r}")
PY
}

verify_experiment_yaml() {
    local env_yaml="$1"
    local gpu="$2"
    local expected_usd_dir="$3"
    local -a reward_overrides=()

    build_reward_overrides "${gpu}" reward_overrides
    "${PYTHON_BIN}" - \
        "${env_yaml}" \
        "${expected_usd_dir}" \
        "${leg_action_scales[$gpu]}" \
        "${j1_3_action_scales[$gpu]}" \
        "${j4_6_action_scales[$gpu]}" \
        "${reward_term_csv}" \
        "${reward_overrides[@]}" <<'PY'
import math
import sys

import yaml
from hydra.core.override_parser.overrides_parser import OverridesParser

env_yaml, expected_usd_dir, leg, j1_3, j4_6, reward_term_csv, *reward_override_args = sys.argv[1:]
expected_reward_terms = reward_term_csv.split(",")
parsed_reward_overrides = OverridesParser.create().parse_overrides(reward_override_args)
expected_reward_keys = tuple(
    [f"env.rewards.{term}.weight" for term in expected_reward_terms]
    + ["env.rewards.track_EE_pb.params.use_walking_gate"]
)
actual_reward_keys = tuple(
    override.get_key_element() for override in parsed_reward_overrides
)
if actual_reward_keys != expected_reward_keys:
    raise SystemExit(f"unexpected reward override keys: {actual_reward_keys}")

expected_reward_weights = {
    term: float(override.value())
    for term, override in zip(expected_reward_terms, parsed_reward_overrides[:-1])
}
expected_walking_gate = parsed_reward_overrides[-1].value()

# Isaac Lab's dump_yaml emits trusted Python-specific tags (for example tuples
# and slices), which SafeLoader and FullLoader cannot reconstruct.
with open(env_yaml, encoding="utf-8") as stream:
    cfg = yaml.unsafe_load(stream)

try:
    actual_usd_dir = cfg["scene"]["robot"]["spawn"]["usd_dir"]
except (KeyError, TypeError) as exc:
    raise SystemExit(f"{env_yaml}: missing scene.robot.spawn.usd_dir") from exc
if actual_usd_dir != expected_usd_dir:
    raise SystemExit(
        f"{env_yaml}: scene.robot.spawn.usd_dir={actual_usd_dir!r}, "
        f"expected {expected_usd_dir!r}"
    )

try:
    actual = cfg["actions"]["joint_pos"]["scale"]
except (KeyError, TypeError) as exc:
    raise SystemExit(f"{env_yaml}: missing actions.joint_pos.scale") from exc

expected_grouped = {
    "abad_[RL]_Joint|hip_[RL]_Joint|knee_[RL]_Joint|ankle_[RL]_Joint": float(leg),
    "J[1-3]": float(j1_3),
    "J[4-6]": float(j4_6),
}
expected_expanded = {
    "abad_L_Joint": float(leg),
    "abad_R_Joint": float(leg),
    "hip_L_Joint": float(leg),
    "hip_R_Joint": float(leg),
    "knee_L_Joint": float(leg),
    "knee_R_Joint": float(leg),
    "ankle_L_Joint": float(leg),
    "ankle_R_Joint": float(leg),
    "J1": float(j1_3),
    "J2": float(j1_3),
    "J3": float(j1_3),
    "J4": float(j4_6),
    "J5": float(j4_6),
    "J6": float(j4_6),
}
if actual.keys() == expected_grouped.keys():
    expected = expected_grouped
elif actual.keys() == expected_expanded.keys():
    expected = expected_expanded
else:
    raise SystemExit(f"{env_yaml}: unexpected action scale groups: {actual}")
for key, expected_value in expected.items():
    actual_value = float(actual[key])
    if not math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=1.0e-12):
        raise SystemExit(
            f"{env_yaml}: {key}={actual_value}, expected {expected_value}"
        )

actual_reward_weights = {}
for term, term_cfg in cfg["rewards"].items():
    if term_cfg is None:
        continue
    try:
        weight = float(term_cfg["weight"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"{env_yaml}: missing/invalid rewards.{term}.weight") from exc
    if weight != 0.0:
        actual_reward_weights[term] = weight

missing_terms = sorted(expected_reward_weights.keys() - actual_reward_weights.keys())
unexpected_terms = sorted(actual_reward_weights.keys() - expected_reward_weights.keys())
if missing_terms or unexpected_terms:
    raise SystemExit(
        f"{env_yaml}: active non-zero reward set mismatch; "
        f"missing={missing_terms}, unexpected={unexpected_terms}"
    )

for term, expected_weight in expected_reward_weights.items():
    actual_weight = actual_reward_weights[term]
    if not math.isclose(actual_weight, expected_weight, rel_tol=0.0, abs_tol=1.0e-12):
        raise SystemExit(
            f"{env_yaml}: rewards.{term}.weight={actual_weight}, expected {expected_weight}"
        )

try:
    actual_walking_gate = cfg["rewards"]["track_EE_pb"]["params"]["use_walking_gate"]
except (KeyError, TypeError) as exc:
    raise SystemExit(
        f"{env_yaml}: missing rewards.track_EE_pb.params.use_walking_gate"
    ) from exc
if actual_walking_gate is not expected_walking_gate:
    raise SystemExit(
        f"{env_yaml}: PB use_walking_gate={actual_walking_gate!r}, "
        f"expected {expected_walking_gate!r}"
    )
PY
}

verify_only_controlled_differences() {
    "${PYTHON_BIN}" - "$@" <<'PY'
import copy
import sys

import yaml

paths = sys.argv[1:]
if not paths:
    raise SystemExit("expected at least one environment YAML file")

configs = []
for path in paths:
    with open(path, encoding="utf-8") as stream:
        # These files are generated locally by Isaac Lab and contain trusted
        # Python-specific YAML tags such as tuples and slices.
        configs.append(yaml.unsafe_load(stream))

allowed_difference_paths = (
    ("actions", "joint_pos", "scale"),
    ("rewards", "base_pitch_exp", "weight"),
    ("rewards", "base_target_heading_alignment", "weight"),
    ("rewards", "track_EE_orientation_exp", "weight"),
    ("rewards", "track_EE_pb", "params", "use_walking_gate"),
    # Per-process URDF-to-USD output directories prevent conversion races.
    ("scene", "robot", "spawn", "usd_dir"),
)

def without_allowed_differences(config):
    result = copy.deepcopy(config)
    for path in allowed_difference_paths:
        parent = result
        for key in path[:-1]:
            parent = parent[key]
        del parent[path[-1]]
    return result

baseline = without_allowed_differences(configs[0])
for path, config in zip(paths[1:], configs[1:]):
    if without_allowed_differences(config) != baseline:
        raise SystemExit(
            f"{path} differs from {paths[0]} outside the "
            "five controlled experiment fields and the per-process USD directory"
        )
PY
}

preflight() {
    local gpu_count
    local -a compute_pids=()

    check_training_source
    for gpu in "${ACTIVE_GPUS[@]}"; do
        validate_action_scale_override "${gpu}"
        validate_reward_overrides "${gpu}"
    done

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi is not available." >&2
        return 1
    fi
    gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l)"
    for gpu in "${ACTIVE_GPUS[@]}"; do
        if (( gpu >= gpu_count )); then
            echo "ERROR: Selected GPU ${gpu} is unavailable; ${gpu_count} GPUs were detected." >&2
            return 1
        fi
    done

    mapfile -t compute_pids < <(nvidia-smi -i "${selected_gpu_csv}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed -n 's/^[[:space:]]*\([0-9][0-9]*\).*$/\1/p' | sort -u)
    if (( ${#compute_pids[@]} > 0 )); then
        echo "ERROR: Selected GPUs ${selected_gpu_csv} already have compute processes; refusing to oversubscribe them." >&2
        ps -o pid=,user=,etime=,args= -p "$(IFS=,; echo "${compute_pids[*]}")" >&2 || true
        return 1
    fi
}

if ! "${PYTHON_BIN}" -c "import isaaclab" >/dev/null 2>&1; then
    echo "ERROR: ${PYTHON_BIN} cannot import isaaclab." >&2
    echo "Activate the training environment first:" >&2
    echo "  conda activate isaaclab_umi_on_tron" >&2
    exit 1
fi

if [[ "${check_config}" == true ]]; then
    check_training_source
    for gpu in "${ACTIVE_GPUS[@]}"; do
        validate_action_scale_override "${gpu}"
        validate_reward_overrides "${gpu}"
        echo "GPU ${gpu} configuration:"
        echo "  rewards:"
        print_reward_config "${gpu}" "    "
        echo "  action_scale: legs=${leg_action_scales[$gpu]}, J1-3=${j1_3_action_scales[$gpu]}, J4-6=${j4_6_action_scales[$gpu]}"
    done
    echo "All active non-zero rewards, action scales, and current-checkout import path verified."
    exit 0
fi

launch_stamp="${LAUNCH_STAMP:-$(date +%Y%m%d_%H%M%S)}"
session_name="reward_sweep_${launch_stamp}"

# Outside tmux, relaunch this script in a detached session. The absolute Python
# path preserves the currently activated Isaac Lab environment.
if [[ "${foreground}" == false && -z "${TMUX:-}" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "ERROR: tmux is not installed. Run with --foreground or install tmux." >&2
        exit 1
    fi

    preflight

    printf -v tmux_command \
        'exec env PYTHON_BIN=%q PYTHONPATH=%q WBC_LOG_ROOT=%q PYTORCH_CUDA_ALLOC_CONF=%q STARTUP_GRACE_SECONDS=%q LAUNCH_STAMP=%q bash %q --foreground --gpus %q' \
        "${PYTHON_BIN}" \
        "${PYTHONPATH}" \
        "${WBC_LOG_ROOT}" \
        "${PYTORCH_CUDA_ALLOC_CONF}" \
        "${STARTUP_GRACE_SECONDS}" \
        "${launch_stamp}" \
        "${SCRIPT_PATH}" \
        "${selected_gpu_csv}"

    tmux new-session -d -s "${session_name}" -c "${SCRIPT_DIR}" "${tmux_command}"
    echo "Preflight passed. Verifying all ${worker_count} workers for ${STARTUP_GRACE_SECONDS}s; do not launch again."
    sleep "$((STARTUP_GRACE_SECONDS + 3))"
    if ! tmux has-session -t "${session_name}" 2>/dev/null; then
        echo "ERROR: Training session exited during startup." >&2
        echo "Diagnostic logs, if created, were kept in:" >&2
        echo "  ${WBC_LOG_ROOT}/launcher_${launch_stamp}" >&2
        exit 1
    fi
    echo "${worker_count}-GPU reward sweep started in detached tmux session: ${session_name}"
    echo "Attach:  tmux attach -t ${session_name}"
    echo "Detach:  Ctrl-b, then d"
    echo "Status:  tmux has-session -t ${session_name} && echo running"
    echo "Stop:    tmux kill-session -t ${session_name}"
    exit 0
fi

if ! command -v flock >/dev/null 2>&1; then
    echo "ERROR: flock is required to prevent duplicate launches." >&2
    exit 1
fi

# Lock only the selected physical GPUs. This allows disjoint sweeps (for
# example, an existing run on GPUs 2,3 and a new run on GPUs 0,1) while still
# preventing two launchers from assigning the same GPU.
gpu_lock_fds=()
for gpu in "${ACTIVE_GPUS[@]}"; do
    lock_file="${TMPDIR:-/tmp}/launch_reward_sweep_gpu${gpu}.lock"
    exec {lock_fd}>"${lock_file}"
    if ! flock -n "${lock_fd}"; then
        echo "ERROR: Physical GPU ${gpu} is already reserved by another reward-sweep launcher." >&2
        exit 1
    fi
    gpu_lock_fds+=("${lock_fd}")
done
preflight

console_log_dir="${WBC_LOG_ROOT}/launcher_${launch_stamp}"
pids=()

if [[ -e "${console_log_dir}" ]] || find "${WBC_LOG_ROOT}" -mindepth 1 -maxdepth 1 -name "*_${launch_stamp}" -print -quit 2>/dev/null | grep -q .; then
    echo "ERROR: Launch stamp ${launch_stamp} already has logs; choose a new stamp." >&2
    exit 1
fi

launcher_valid=false
cleanup_failed_startup() {
    local status=$?
    if [[ "${launcher_valid}" == false ]]; then
        for pid in "${pids[@]}"; do
            kill "${pid}" 2>/dev/null || true
        done
        for pid in "${pids[@]}"; do
            wait "${pid}" 2>/dev/null || true
        done
        echo "Startup failed; diagnostic logs were preserved in ${console_log_dir}" >&2
    fi
    return "${status}"
}
trap cleanup_failed_startup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "${console_log_dir}"

cd "${SCRIPT_DIR}"

# Which GPU to use for each run. The CUDA_VISIBLE_DEVICES environment variable is set
for gpu in "${ACTIVE_GPUS[@]}"; do
    run_name="reward_sweep_gpu${gpu}_${launch_stamp}"
    console_log="${console_log_dir}/${run_name}.log"
    usd_dir="/tmp/IsaacLab/${launch_stamp}_gpu${gpu}"
    action_scale_override="$(build_action_scale_override "${gpu}")"
    reward_overrides=()
    build_reward_overrides "${gpu}" reward_overrides

    echo "Launching ${run_name} on physical GPU ${gpu}"
    echo "  rewards:"
    print_reward_config "${gpu}" "    "
    echo "  action_scale: legs=${leg_action_scales[$gpu]}, J1-3=${j1_3_action_scales[$gpu]}, J4-6=${j4_6_action_scales[$gpu]}"
    echo "  usd_dir=${usd_dir}"
    echo "  console=${console_log}"

    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        exec "${PYTHON_BIN}" ios_train.py \
            --headless \
            --device=cuda:0 \
            --task="${task}" \
            --run_name="${run_name}" \
            --seed=42 \
            --asset_usd_dir="${usd_dir}" \
            --num_envs="${num_envs}" \
            --max_iterations="${max_iterations}" \
            --logger=wandb \
            "${reward_overrides[@]}" \
            "${action_scale_override}" \
            "agent.wandb_run_name=${run_name}"
    ) >"${console_log}" 2>&1 &

    pids+=("$!")
    echo "  pid=${pids[-1]}"
done

echo "Waiting ${STARTUP_GRACE_SECONDS}s to confirm that all ${worker_count} workers survive startup."
deadline=$((SECONDS + STARTUP_GRACE_SECONDS))
while (( SECONDS < deadline )); do
    for index in "${!pids[@]}"; do
        pid="${pids[$index]}"
        state="$(ps -o stat= -p "${pid}" 2>/dev/null | cut -c1 || true)"
        if [[ -z "${state}" || "${state}" == "Z" ]]; then
            gpu="${ACTIVE_GPUS[$index]}"
            echo "ERROR: GPU ${gpu} worker (pid=${pid}) exited during startup." >&2
            exit 1
        fi
    done
    sleep 1
done

env_yamls=()
for gpu in "${ACTIVE_GPUS[@]}"; do
    run_name="reward_sweep_gpu${gpu}_${launch_stamp}"
    run_dir="$(find "${WBC_LOG_ROOT}" -mindepth 1 -maxdepth 1 -type d -name "*_${run_name}" -print -quit 2>/dev/null)"
    if [[ -z "${run_dir}" ]]; then
        echo "ERROR: GPU ${gpu} did not create its training run directory during startup." >&2
        exit 1
    fi
    env_yaml="${run_dir}/params/env.yaml"
    if [[ ! -f "${env_yaml}" ]]; then
        echo "ERROR: GPU ${gpu} did not write ${env_yaml} during startup." >&2
        exit 1
    fi
    if ! verify_experiment_yaml "${env_yaml}" "${gpu}" "/tmp/IsaacLab/${launch_stamp}_gpu${gpu}"; then
        echo "ERROR: GPU ${gpu} experiment settings did not reach the saved environment configuration." >&2
        exit 1
    fi
    env_yamls+=("${env_yaml}")
    echo "Verified GPU ${gpu} reward/action-scale settings in ${env_yaml}"
done
if ! verify_only_controlled_differences "${env_yamls[@]}"; then
    echo "ERROR: Non-experimental environment settings are not identical across GPUs." >&2
    exit 1
fi
echo "Verified that all non-experimental environment settings are identical."

launcher_valid=true
echo "All ${worker_count} jobs passed startup validation and are running."
echo "Follow one job with: tail -f ${console_log_dir}/reward_sweep_gpu${ACTIVE_GPUS[0]}_${launch_stamp}.log"

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

exit "${status}"
