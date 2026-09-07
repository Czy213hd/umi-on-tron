#!/usr/bin/env bash

# Five-model, 1000-target, three-evaluation-seed benchmark.  Small isolated
# batches avoid losing an entire seed if Isaac Sim exits during a large rollout.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/miniforge3/envs/isaaclab_umi_on_tron/bin/python}"
export PYTHONPATH="${PROJECT_DIR}/source/ext_loco:${PROJECT_DIR}/rsl_rl${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NUM_TARGETS="${NUM_TARGETS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TARGET_DURATION="${TARGET_DURATION:-8}"
MAX_RETRIES="${MAX_RETRIES:-3}"
STAMP="${EVAL_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/evaluation/ablation_1000x3_${STAMP}}"
SEEDS=(42 43 44)
NAMES=(full_baseline no_latent dreamwaq_cenet matched_additive_reward privileged_oracle)
VARIANTS=(transformer_gru no_latent cenet transformer_gru privileged_oracle)
CHECKPOINTS=(
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-03_17-09-34_ablation_full_baseline_gpu0_20260803_170924/model_19999.pt"
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-03_17-09-33_ablation_no_latent_gpu1_20260803_170924/model_19999.pt"
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-03_17-09-33_ablation_dreamwaq_cenet_gpu2_20260803_170924/model_19999.pt"
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-05_13-26-43_ablation_matched_additive_gpu3_20260805_matched_v2/model_19999.pt"
    "${PROJECT_DIR}/logs/rsl_rl/2026-08-05_22-22-55_ablation_privileged_oracle_gpu0_20260805_153000/model_19999.pt"
)

mkdir -p "${OUTPUT_DIR}"

run_worker() {
    local gpu="$1"
    local task_id=0
    for model_index in 0 1 2 3 4; do
        local name="${NAMES[$model_index]}"
        local variant="${VARIANTS[$model_index]}"
        local checkpoint="${CHECKPOINTS[$model_index]}"
        for seed in "${SEEDS[@]}"; do
            local seed_dir="${OUTPUT_DIR}/${name}/seed_${seed}"
            mkdir -p "${seed_dir}"
            for ((offset=0; offset<NUM_TARGETS; offset+=BATCH_SIZE)); do
                if (( task_id % 4 != gpu )); then
                    ((task_id+=1))
                    continue
                fi
                local count="${BATCH_SIZE}"
                (( offset + count <= NUM_TARGETS )) || count=$((NUM_TARGETS - offset))
                local csv_path="${seed_dir}/batch_${offset}.csv"
                if [[ -s "${csv_path}" ]]; then
                    ((task_id+=1))
                    continue
                fi
                local success=false
                local last_rc=0
                for ((attempt=1; attempt<=MAX_RETRIES; attempt+=1)); do
                    local log_path="${seed_dir}/batch_${offset}.attempt_${attempt}.log"
                    set +e
                    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_ablation.py" \
                        --headless --device cuda:0 \
                        --checkpoint "${checkpoint}" \
                        --variant "${variant}" \
                        --model-name "${name}" \
                        --output "${csv_path}" \
                        --num-envs "${count}" \
                        --num-targets 1 \
                        --target-population "${NUM_TARGETS}" \
                        --target-offset "${offset}" \
                        --target-duration "${TARGET_DURATION}" \
                        --seed "${seed}" \
                        --x-min -0.3 --x-max 0.8 \
                        --z-min 0.4 --z-max 1.6 \
                        --rpy-min -0.6 --rpy-max 0.6 \
                        --asset-usd-dir "${OUTPUT_DIR}/.usd_cache/gpu${gpu}" \
                        >"${log_path}" 2>&1
                    last_rc=$?
                    set -e
                    echo "evaluator_exit_code=${last_rc}" >>"${log_path}"
                    if [[ -s "${csv_path}" ]]; then
                        success=true
                        break
                    fi
                done
                [[ "${success}" == true ]] || {
                    # A single unstable simulated robot can terminate an entire
                    # vectorized PhysX rollout without a Python traceback.  Retry
                    # the affected vector batch as isolated one-environment runs,
                    # then reconstruct the original batch CSV.
                    echo "Vector batch failed (rc=${last_rc}); isolating ${name} seed=${seed} offset=${offset}" >&2
                    local part_dir="${seed_dir}/.parts_${offset}"
                    local merged_path="${csv_path}.tmp"
                    mkdir -p "${part_dir}"
                    rm -f -- "${merged_path}"
                    local part_failed=false
                    for ((i=0; i<count; i+=1)); do
                        local global_offset=$((offset + i))
                        local part_csv="${part_dir}/target_${global_offset}.csv"
                        local part_log="${part_dir}/target_${global_offset}.log"
                        local part_rc=0
                        if [[ ! -s "${part_csv}" ]]; then
                            for ((part_attempt=1; part_attempt<=MAX_RETRIES; part_attempt+=1)); do
                                set +e
                                CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" "${SCRIPT_DIR}/evaluate_ablation.py" \
                                    --headless --device cuda:0 --checkpoint "${checkpoint}" \
                                    --variant "${variant}" --model-name "${name}" \
                                    --output "${part_csv}" --num-envs 1 --num-targets 1 \
                                    --target-population "${NUM_TARGETS}" --target-offset "${global_offset}" \
                                    --target-duration "${TARGET_DURATION}" --seed "${seed}" \
                                    --x-min -0.3 --x-max 0.8 --z-min 0.4 --z-max 1.6 \
                                    --rpy-min -0.6 --rpy-max 0.6 \
                                    --asset-usd-dir "${OUTPUT_DIR}/.usd_cache/gpu${gpu}" \
                                    >"${part_log}.attempt_${part_attempt}" 2>&1
                                part_rc=$?
                                set -e
                                echo "evaluator_exit_code=${part_rc}" >>"${part_log}.attempt_${part_attempt}"
                                [[ -s "${part_csv}" ]] && break
                            done
                        fi
                        if [[ ! -s "${part_csv}" ]]; then
                            echo "ISOLATED TARGET FAILED rc=${part_rc}: ${name} seed=${seed} target=${global_offset}" >&2
                            part_failed=true
                            break
                        fi
                        awk -F, -v OFS=, -v env_id="${i}" \
                            'NR == 1 { if (env_id == 0) print; next } {$5=env_id; print}' \
                            "${part_csv}" >>"${merged_path}"
                    done
                    if [[ "${part_failed}" == false ]]; then
                        mv -- "${merged_path}" "${csv_path}"
                        success=true
                    else
                        rm -f -- "${merged_path}"
                    fi
                }
                [[ "${success}" == true ]] || { ((task_id+=1)); continue; }
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
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || { echo "At least one worker failed: ${OUTPUT_DIR}" >&2; exit 1; }

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_multiseed_ablation.py" \
    --input-dir "${OUTPUT_DIR}" --num-targets "${NUM_TARGETS}" --seeds "${SEEDS[@]}"
echo "Evaluation complete: ${OUTPUT_DIR}"
