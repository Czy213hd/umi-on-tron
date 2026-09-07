#!/usr/bin/env bash

# Seven unique checkpoints, three evaluation seeds, 1000 paired targets.
# BEEM and Transformer+GRU refer to the same full-baseline checkpoint.

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/miniforge3/envs/isaaclab_umi_on_tron/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/evaluation/stable_success_7model_1000x3_20260812}"
NUM_TARGETS="${NUM_TARGETS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TARGET_DURATION="${TARGET_DURATION:-8}"
MAX_RETRIES="${MAX_RETRIES:-3}"

export PYTHONPATH="${PROJECT_DIR}/source/ext_loco:${PROJECT_DIR}/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SEEDS=(42 43 44)
NAMES=(matched_additive_reward beem_transformer_gru no_latent dreamwaq_cenet gru_only transformer_only privileged_oracle)
VARIANTS=(transformer_gru transformer_gru no_latent cenet gru_only transformer_only privileged_oracle)
CHECKPOINTS=(
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-05_13-26-43_ablation_matched_additive_gpu3_20260805_matched_v2/model_19999.pt"
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-03_17-09-34_ablation_full_baseline_gpu0_20260803_170924/model_19999.pt"
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-03_17-09-33_ablation_no_latent_gpu1_20260803_170924/model_19999.pt"
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-03_17-09-33_ablation_dreamwaq_cenet_gpu2_20260803_170924/model_19999.pt"
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-10_18-36-34_ablation_gru_only_gpu0_20260810_183629/model_19999.pt"
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-10_18-36-34_ablation_transformer_only_gpu1_20260810_183629/model_19999.pt"
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-05_22-22-55_ablation_privileged_oracle_gpu0_20260805_153000/model_19999.pt"
)

mkdir -p "${OUTPUT_DIR}"

run_worker() {
    local gpu="$1"
    local task_id=0
    local model_index seed offset count name variant checkpoint seed_dir csv_path log_path rc attempt
    for model_index in "${!NAMES[@]}"; do
        name="${NAMES[$model_index]}"
        variant="${VARIANTS[$model_index]}"
        checkpoint="${CHECKPOINTS[$model_index]}"
        for seed in "${SEEDS[@]}"; do
            seed_dir="${OUTPUT_DIR}/${name}/seed_${seed}"
            mkdir -p "${seed_dir}"
            for ((offset=0; offset<NUM_TARGETS; offset+=BATCH_SIZE)); do
                if (( task_id % 4 != gpu )); then
                    ((task_id+=1))
                    continue
                fi
                count="${BATCH_SIZE}"
                (( offset + count <= NUM_TARGETS )) || count=$((NUM_TARGETS - offset))
                csv_path="${seed_dir}/batch_${offset}.csv"
                if [[ -s "${csv_path}" ]]; then
                    ((task_id+=1))
                    continue
                fi
                for ((attempt=1; attempt<=MAX_RETRIES; attempt+=1)); do
                    log_path="${seed_dir}/batch_${offset}.attempt_${attempt}.log"
                    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_ablation.py" \
                        --headless --device cuda:0 --checkpoint "${checkpoint}" \
                        --variant "${variant}" --model-name "${name}" --output "${csv_path}" \
                        --num-envs "${count}" --num-targets 1 --target-population "${NUM_TARGETS}" \
                        --target-offset "${offset}" --target-duration "${TARGET_DURATION}" --seed "${seed}" \
                        --x-min -0.3 --x-max 0.8 --z-min 0.4 --z-max 1.6 \
                        --rpy-min -0.6 --rpy-max 0.6 \
                        --asset-usd-dir "${OUTPUT_DIR}/.usd_cache/gpu${gpu}" \
                        >"${log_path}" 2>&1
                    rc=$?
                    echo "evaluator_exit_code=${rc}" >>"${log_path}"
                    [[ -s "${csv_path}" ]] && break
                done
                [[ -s "${csv_path}" ]] || echo "FAILED model=${name} seed=${seed} offset=${offset}" >&2
                ((task_id+=1))
            done
        done
    done
}

pids=()
for gpu in 0 1 2 3; do
    run_worker "${gpu}" >"${OUTPUT_DIR}/worker_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
done

for name in "${NAMES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        rows=$(find "${OUTPUT_DIR}/${name}/seed_${seed}" -maxdepth 1 -name 'batch_*.csv' -type f -print0 \
            | xargs -0 -r awk 'FNR > 1 {n += 1} END {print n + 0}')
        echo "model=${name} seed=${seed} completed_targets=${rows}/${NUM_TARGETS}"
        (( rows == NUM_TARGETS )) || status=1
    done
done

exit "${status}"
