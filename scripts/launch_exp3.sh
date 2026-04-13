#!/usr/bin/env bash
set -euo pipefail

# exp3: SFT with hard-normal augmentation + mild weights + 16 key frames + 3 epochs
# Changes from exp1:
#   - MAX_NUM_KEY_FRAMES 8->16 (temporal resolution 2x, code change in tools.py)
#   - 600 samples (+120 hard-normal with scene-specific queries)
#   - Mild sample weights: normal=1.0, hard-normal=0.80, anomaly=0.85
#   - warmup_ratio=0.10 (cosine schedule)
#   - 3 epochs (user constraint)

PROJECT_ROOT="/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3"
DATA="${PROJECT_ROOT}/data_utils/sft_train.compact_trace_v2.exp3.jsonl"

if [[ ! -f "${DATA}" ]]; then
    echo "[ERROR] Exp3 data not found at ${DATA}"
    exit 1
fi

echo "[exp3] Changes: MAX_KEY_FRAMES=16, hard-normal+600 samples, mild weights, 3 epochs"
echo "[exp3] Data: ${DATA}"

EXP_NAME="${EXP_NAME:-exp3}" \
NUM_SFT_EPOCHS=3 \
SFT_PREPARED_FILE="${DATA}" \
SFT_CONFIG="${PROJECT_ROOT}/configs/sft/qwen3_vl_8b_full_train_exp3.yaml" \
bash "${PROJECT_ROOT}/scripts/run_full_pipeline.sh"
