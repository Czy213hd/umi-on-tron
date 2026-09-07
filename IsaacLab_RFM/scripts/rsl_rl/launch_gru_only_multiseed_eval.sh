#!/usr/bin/env bash

# Evaluate the completed GRU-only ablation with one evaluation seed per GPU.
# Isaac Sim 4.5 cannot reliably initialize a second Vulkan instance while the
# temporal-ablation trainer is alive, so the launcher can wait for that PID.

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/miniforge3/envs/isaaclab_umi_on_tron/bin/python}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_DIR}/logs/rsl_rl/2026-08-10_18-36-34_ablation_gru_only_gpu0_20260810_183629/model_19999.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/evaluation/gru_only_1000x3_20260811}"
WAIT_PID="${WAIT_PID:-}"
NUM_TARGETS="${NUM_TARGETS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TARGET_DURATION="${TARGET_DURATION:-8}"
MAX_RETRIES="${MAX_RETRIES:-3}"

export PYTHONPATH="${PROJECT_DIR}/source/ext_loco:${PROJECT_DIR}/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${OUTPUT_DIR}"

if [[ -n "${WAIT_PID}" ]]; then
    echo "Waiting for active Isaac Sim training process ${WAIT_PID} to finish."
    while kill -0 "${WAIT_PID}" 2>/dev/null; do
        sleep 30
    done
    echo "Training process ${WAIT_PID} finished; starting GRU-only evaluation."
fi

run_seed() {
    local gpu="$1"
    local seed="$2"
    local seed_dir="${OUTPUT_DIR}/gru_only/seed_${seed}"
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
                --checkpoint "${CHECKPOINT}" --variant gru_only --model-name gru_only \
                --output "${csv_path}" --num-envs "${count}" --num-targets 1 \
                --target-population "${NUM_TARGETS}" --target-offset "${offset}" \
                --target-duration "${TARGET_DURATION}" --seed "${seed}" \
                --x-min -0.3 --x-max 0.8 --z-min 0.4 --z-max 1.6 \
                --rpy-min -0.6 --rpy-max 0.6 \
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
    rows=$(find "${OUTPUT_DIR}/gru_only/seed_${seed}" -maxdepth 1 -name 'batch_*.csv' -type f -print0 \
        | xargs -0 -r awk 'FNR > 1 {n += 1} END {print n + 0}')
    echo "seed=${seed} completed_targets=${rows}/${NUM_TARGETS}"
    (( rows == NUM_TARGETS )) || status=1
done

exit "${status}"
