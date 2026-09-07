#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/miniforge3/envs/isaaclab_umi_on_tron/bin/python}"
export PYTHONPATH="${PROJECT_DIR}/source/ext_loco:${PROJECT_DIR}/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NUM_ENVS="${NUM_ENVS:-128}"
NUM_TARGETS="${NUM_TARGETS:-1}"
TARGET_DURATION="${TARGET_DURATION:-8}"
EVAL_SEED="${EVAL_SEED:-42}"
X_MIN="${X_MIN:--0.3}"
X_MAX="${X_MAX:-0.8}"
Z_MIN="${Z_MIN:-0.4}"
Z_MAX="${Z_MAX:-1.7}"
RPY_MIN="${RPY_MIN:--0.3}"
RPY_MAX="${RPY_MAX:-0.3}"
STAMP="${EVAL_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/evaluation/ablation_${STAMP}}"

declare -a NAMES=(full_baseline no_latent dreamwaq_cenet additive_reward)
declare -a VARIANTS=(transformer_gru no_latent cenet transformer_gru)
declare -a CHECKPOINTS=(
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-03_17-09-34_ablation_full_baseline_gpu0_20260803_170924/model_19999.pt"
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-03_17-09-33_ablation_no_latent_gpu1_20260803_170924/model_19999.pt"
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-03_17-09-33_ablation_dreamwaq_cenet_gpu2_20260803_170924/model_19999.pt"
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-03_17-09-34_ablation_additive_reward_gpu3_20260803_170924/model_19999.pt"
)

mkdir -p "${OUTPUT_DIR}/logs"
pids=()
for gpu in 0 1 2 3; do
    checkpoint="${CHECKPOINTS[$gpu]}"
    [[ -f "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}" >&2; exit 1; }
    name="${NAMES[$gpu]}"
    log="${OUTPUT_DIR}/logs/${name}.log"
    echo "Launching ${name} on GPU ${gpu}"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        exec "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_ablation.py" \
            --headless \
            --device cuda:0 \
            --checkpoint "${checkpoint}" \
            --variant "${VARIANTS[$gpu]}" \
            --model-name "${name}" \
            --output "${OUTPUT_DIR}/${name}.csv" \
            --num-envs "${NUM_ENVS}" \
            --num-targets "${NUM_TARGETS}" \
            --target-duration "${TARGET_DURATION}" \
            --seed "${EVAL_SEED}" \
            --x-min "${X_MIN}" \
            --x-max "${X_MAX}" \
            --z-min "${Z_MIN}" \
            --z-max "${Z_MAX}" \
            --rpy-min "${RPY_MIN}" \
            --rpy-max "${RPY_MAX}" \
            --asset-usd-dir "${OUTPUT_DIR}/.usd_cache/gpu${gpu}"
    ) >"${log}" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done
if (( status != 0 )); then
    echo "At least one evaluation failed. Logs: ${OUTPUT_DIR}/logs" >&2
    exit "${status}"
fi
for name in "${NAMES[@]}"; do
    [[ -s "${OUTPUT_DIR}/${name}.csv" ]] || {
        echo "Evaluation exited without producing ${name}.csv. Logs: ${OUTPUT_DIR}/logs" >&2
        exit 1
    }
    [[ -s "${OUTPUT_DIR}/${name}.summary.json" ]] || {
        echo "Evaluation exited without producing ${name}.summary.json. Logs: ${OUTPUT_DIR}/logs" >&2
        exit 1
    }
done
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_ablation_eval.py" --input-dir "${OUTPUT_DIR}"
echo "Evaluation complete: ${OUTPUT_DIR}"
