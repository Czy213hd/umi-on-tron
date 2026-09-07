#!/usr/bin/env bash

# Evaluate the completed Transformer-only ablation with one seed per GPU.

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/miniforge3/envs/isaaclab_umi_on_tron/bin/python}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/logs/rsl_rl/2026-08-10_18-36-34_ablation_transformer_only_gpu1_20260810_183629/model_19999.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/evaluation/transformer_only_1000x3_20260812}"
NUM_TARGETS="${NUM_TARGETS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TARGET_DURATION="${TARGET_DURATION:-8}"
MAX_RETRIES="${MAX_RETRIES:-3}"
X_MIN="${X_MIN:--0.3}"
X_MAX="${X_MAX:-0.8}"
Z_MIN="${Z_MIN:-0.4}"
Z_MAX="${Z_MAX:-1.6}"
RPY_MIN="${RPY_MIN:--0.6}"
RPY_MAX="${RPY_MAX:-0.6}"

export PYTHONPATH="${PROJECT_DIR}/source/ext_loco:${PROJECT_DIR}/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${OUTPUT_DIR}"

run_seed() {
    local gpu="$1"
    local seed="$2"
    local seed_dir="${OUTPUT_DIR}/transformer_only/seed_${seed}"
    mkdir -p "${seed_dir}"

    for ((offset=0; offset<NUM_TARGETS; offset+=BATCH_SIZE)); do
        local count="${BATCH_SIZE}"
        (( offset + count <= NUM_TARGETS )) || count=$((NUM_TARGETS - offset))
        local csv_path="${seed_dir}/batch_${offset}.csv"
        [[ -s "${csv_path}" ]] && continue

        for ((attempt=1; attempt<=MAX_RETRIES; attempt+=1)); do
            local log_path="${seed_dir}/batch_${offset}.attempt_${attempt}.log"
            CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_ablation.py" \
                --headless --device cuda:0 \
                --checkpoint "${CHECKPOINT}" \
                --variant transformer_only --model-name transformer_only \
                --output "${csv_path}" --num-envs "${count}" --num-targets 1 \
                --target-population "${NUM_TARGETS}" --target-offset "${offset}" \
                --target-duration "${TARGET_DURATION}" --seed "${seed}" \
                --x-min "${X_MIN}" --x-max "${X_MAX}" \
                --z-min "${Z_MIN}" --z-max "${Z_MAX}" \
                --rpy-min "${RPY_MIN}" --rpy-max "${RPY_MAX}" \
                --asset-usd-dir "${OUTPUT_DIR}/.usd_cache/gpu${gpu}" \
                >"${log_path}" 2>&1
            local rc=$?
            echo "evaluator_exit_code=${rc}" >>"${log_path}"
            [[ -s "${csv_path}" ]] && break
        done

        if [[ ! -s "${csv_path}" ]]; then
            echo "FAILED seed=${seed} offset=${offset} after ${MAX_RETRIES} attempts" >&2
        fi
    done
}

pids=()
run_seed 0 42 >"${OUTPUT_DIR}/worker_gpu0_seed42.log" 2>&1 & pids+=("$!")
sleep 20
run_seed 2 43 >"${OUTPUT_DIR}/worker_gpu2_seed43.log" 2>&1 & pids+=("$!")
sleep 20
run_seed 3 44 >"${OUTPUT_DIR}/worker_gpu3_seed44.log" 2>&1 & pids+=("$!")

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done

for seed in 42 43 44; do
    rows=$(find "${OUTPUT_DIR}/transformer_only/seed_${seed}" -maxdepth 1 -name 'batch_*.csv' -type f -print0 \
        | xargs -0 -r awk 'FNR > 1 {n += 1} END {print n + 0}')
    echo "seed=${seed} completed_targets=${rows}/${NUM_TARGETS}"
    (( rows == NUM_TARGETS )) || status=1
done

exit "${status}"
