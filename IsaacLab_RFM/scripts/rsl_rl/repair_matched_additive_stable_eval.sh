#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/miniforge3/envs/isaaclab_umi_on_tron/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/evaluation/stable_success_7model_1000x3_20260812}"
CHECKPOINT="${PROJECT_DIR}/logs/rsl_rl/2026-08-05_13-26-43_ablation_matched_additive_gpu3_20260805_matched_v2/model_19999.pt"
MAX_RETRIES="${MAX_RETRIES:-5}"

export PYTHONPATH="${PROJECT_DIR}/source/ext_loco:${PROJECT_DIR}/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TASKS=("42:160" "42:288" "42:784" "43:448" "44:176" "44:240" "44:544")

repair_batch() {
    local gpu="$1" seed="$2" offset="$3"
    local seed_dir="${OUTPUT_DIR}/matched_additive_reward/seed_${seed}"
    local part_dir="${seed_dir}/.parts_${offset}"
    local batch_csv="${seed_dir}/batch_${offset}.csv"
    local merged="${seed_dir}/batch_${offset}.repairing.csv"
    mkdir -p "${part_dir}"
    : >"${merged}"
    local i global part_csv attempt log rc
    for ((i=0; i<16; i+=1)); do
        global=$((offset+i))
        part_csv="${part_dir}/target_${global}.csv"
        if [[ ! -s "${part_csv}" ]]; then
            for ((attempt=1; attempt<=MAX_RETRIES; attempt+=1)); do
                log="${part_dir}/target_${global}.attempt_${attempt}.log"
                CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_ablation.py" \
                    --headless --device cuda:0 --checkpoint "${CHECKPOINT}" \
                    --variant transformer_gru --model-name matched_additive_reward \
                    --output "${part_csv}" --num-envs 1 --num-targets 1 \
                    --target-population 1000 --target-offset "${global}" \
                    --target-duration 8 --seed "${seed}" \
                    --x-min -0.3 --x-max 0.8 --z-min 0.4 --z-max 1.6 \
                    --rpy-min -0.6 --rpy-max 0.6 \
                    --asset-usd-dir "${OUTPUT_DIR}/.usd_cache/gpu${gpu}" >"${log}" 2>&1
                rc=$?
                echo "evaluator_exit_code=${rc}" >>"${log}"
                [[ -s "${part_csv}" ]] && break
            done
        fi
        if [[ ! -s "${part_csv}" ]]; then
            echo "FAILED seed=${seed} target=${global}" >&2
            return 1
        fi
        awk -F, -v OFS=, -v env_id="${i}" \
            'NR==1 {if (env_id==0) print; next} {$5=env_id; print}' \
            "${part_csv}" >>"${merged}"
    done
    mv -- "${merged}" "${batch_csv}"
    echo "REPAIRED seed=${seed} offset=${offset}"
}

run_gpu_queue() {
    local gpu="$1"
    local idx seed offset
    for ((idx=gpu; idx<${#TASKS[@]}; idx+=4)); do
        IFS=: read -r seed offset <<<"${TASKS[$idx]}"
        repair_batch "${gpu}" "${seed}" "${offset}" \
            >"${OUTPUT_DIR}/repair_seed${seed}_offset${offset}_gpu${gpu}.log" 2>&1 || return 1
    done
}

pids=()
for gpu in 0 1 2 3; do
    run_gpu_queue "${gpu}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
exit "${status}"
