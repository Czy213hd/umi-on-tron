#!/usr/bin/env bash

# Common-target evaluation for the three architecture baselines plus the new
# matched-additive reward model and the full privileged-state Oracle.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/miniforge3/envs/isaaclab_umi_on_tron/bin/python}"
export PYTHONPATH="${PROJECT_DIR}/source/ext_loco:${PROJECT_DIR}/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NUM_ENVS="${NUM_ENVS:-256}"
TARGET_DURATION="${TARGET_DURATION:-8}"
EVAL_SEED="${EVAL_SEED:-42}"
Z_MIN="${Z_MIN:-0.4}"
Z_MAX="${Z_MAX:-1.6}"
RPY_MIN="${RPY_MIN:--0.6}"
RPY_MAX="${RPY_MAX:-0.6}"
STAMP="${EVAL_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/evaluation/ablation_5model_${STAMP}}"

declare -a NAMES=(
    full_baseline
    no_latent
    dreamwaq_cenet
    matched_additive_reward
    privileged_oracle
)
declare -a VARIANTS=(
    transformer_gru
    no_latent
    cenet
    transformer_gru
    privileged_oracle
)
declare -a CHECKPOINTS=(
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-03_17-09-34_ablation_full_baseline_gpu0_20260803_170924/model_19999.pt"
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-03_17-09-33_ablation_no_latent_gpu1_20260803_170924/model_19999.pt"
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-03_17-09-33_ablation_dreamwaq_cenet_gpu2_20260803_170924/model_19999.pt"
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-05_13-26-43_ablation_matched_additive_gpu3_20260805_matched_v2/model_19999.pt"
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-05_22-22-55_ablation_privileged_oracle_gpu0_20260805_153000/model_19999.pt"
)

mkdir -p "${OUTPUT_DIR}/logs"

evaluate_one() {
    local index="$1"
    local gpu="$2"
    local name="${NAMES[$index]}"
    local checkpoint="${CHECKPOINTS[$index]}"
    local log="${OUTPUT_DIR}/logs/${name}.log"
    [[ -f "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}" >&2; return 1; }
    echo "Launching ${name} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_ablation.py" \
        --headless \
        --device cuda:0 \
        --checkpoint "${checkpoint}" \
        --variant "${VARIANTS[$index]}" \
        --model-name "${name}" \
        --output "${OUTPUT_DIR}/${name}.csv" \
        --num-envs "${NUM_ENVS}" \
        --num-targets 1 \
        --target-duration "${TARGET_DURATION}" \
        --seed "${EVAL_SEED}" \
        --x-min -0.3 \
        --x-max 0.8 \
        --z-min "${Z_MIN}" \
        --z-max "${Z_MAX}" \
        --rpy-min "${RPY_MIN}" \
        --rpy-max "${RPY_MAX}" \
        --asset-usd-dir "${OUTPUT_DIR}/.usd_cache/${name}" \
        >"${log}" 2>&1
}

# Four GPUs evaluate the first four models concurrently.  GPU0 then evaluates
# the fifth model, avoiding two simultaneous Isaac Sim processes on one GPU.
pids=()
for index in 0 1 2 3; do
    evaluate_one "${index}" "${index}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || { echo "One of the first four evaluations failed: ${OUTPUT_DIR}/logs" >&2; exit 1; }
evaluate_one 4 0

for name in "${NAMES[@]}"; do
    [[ -s "${OUTPUT_DIR}/${name}.csv" ]] || { echo "Missing ${name}.csv" >&2; exit 1; }
    [[ -s "${OUTPUT_DIR}/${name}.summary.json" ]] || { echo "Missing ${name}.summary.json" >&2; exit 1; }
done
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_ablation_eval.py" --input-dir "${OUTPUT_DIR}"
echo "Evaluation complete: ${OUTPUT_DIR}"
