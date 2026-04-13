#!/usr/bin/env bash
set -euo pipefail

# exp3: SFT with hard-normal augmented data + sample weights + 8 epochs + 10% warmup
# Changes from exp2:
#   - 600 training samples (was 480): +120 hard-normal trajectories
#   - Hard-normals have scene-specific queries and lower sufficiency scores
# Same as exp2: sample_weights, 8 epochs, warmup=0.10, lr=1e-5, cosine

PROJECT_ROOT="/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3"
DATA_DIR="${PROJECT_ROOT}/data_utils"
AUGMENTED_DATA="${DATA_DIR}/sft_train.compact_trace_v2.hard_normal_augmented.jsonl"

if [[ ! -f "${AUGMENTED_DATA}" ]]; then
    echo "[ERROR] Augmented data not found. Run: python3 scripts/generate_hard_normals.py"
    exit 1
fi

echo "[exp3] Key changes: +120 hard-normal trajectories (600 total), sample_weights, 8 epochs"

EXP_NAME="${EXP_NAME:-exp3}" \
NUM_SFT_EPOCHS=8 \
SFT_PREPARED_FILE="${AUGMENTED_DATA}" \
SFT_CONFIG="${PROJECT_ROOT}/configs/sft/qwen3_vl_8b_full_train_exp2.yaml" \
bash "${PROJECT_ROOT}/scripts/run_full_pipeline.sh"
