#!/usr/bin/env bash
set -euo pipefail

# exp2: SFT with class-conditional sample weights + 8 epochs + 10% warmup
# Changes from exp1:
#   - sample_weight in data (normal=1.0, anomaly=0.6)
#   - 8 epochs (was 5)
#   - warmup_ratio=0.10 (was 0.03), via exp2 config
# Unchanged: lr=1e-5, grad_accum=8, max_grad_norm=1.0, cosine schedule

PROJECT_ROOT="/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3"
DATA_DIR="${PROJECT_ROOT}/data_utils"
WEIGHTED_DATA="${DATA_DIR}/sft_train.compact_trace_v2.weighted.jsonl"
WEIGHTED_CACHE="${DATA_DIR}/sft_train.compact_trace_v2.weighted.materialized_messages_v1.jsonl"

# Verify weighted data and cache exist
for f in "${WEIGHTED_DATA}" "${WEIGHTED_CACHE}"; do
    if [[ ! -e "${f}" ]]; then
        echo "[ERROR] Required file not found: ${f}"
        exit 1
    fi
done

echo "[exp2] Key changes: sample_weights(normal=1.0,anomaly=0.6), 8 epochs, warmup=0.10"
echo "[exp2] Data: ${WEIGHTED_DATA}"

EXP_NAME="${EXP_NAME:-exp2}" \
NUM_SFT_EPOCHS=8 \
SFT_PREPARED_FILE="${WEIGHTED_DATA}" \
SFT_MATERIALIZED_FILE="${WEIGHTED_CACHE}" \
SFT_CONFIG="${PROJECT_ROOT}/configs/sft/qwen3_vl_8b_full_train_exp2.yaml" \
bash "${PROJECT_ROOT}/scripts/run_full_pipeline.sh"
