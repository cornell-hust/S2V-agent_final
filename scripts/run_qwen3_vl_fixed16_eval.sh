#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_MODE="print"
CONFIG_PATH="${CONFIG_PATH:-${ROOT_DIR}/configs/fixed_baseline_eval/vllm_qwen3_vl_8b_fixed_16.yaml}"
MODEL_PATH="${MODEL_PATH:-/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/qwen3-vl-8b-Instruct}"
DATA_ROOT="${DATA_ROOT:-${SAVER_V3_DATA_ROOT:-/mnt/shared-storage-user/mineru2-shared/zengweijun}}"
DATA_PATH="${DATA_PATH:-data_utils/msad_saver_runtime_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/fixed_baseline_16/qwen3_vl_8b}"
INCLUDE_SPLITS="${INCLUDE_SPLITS:-test}"
FRAME_BUDGET="${FRAME_BUDGET:-16}"
MAX_RECORDS="${MAX_RECORDS:-0}"

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29731}"

EXTRA_OVERRIDES=()
EXTRA_OVERRIDE_COUNT=0

usage() {
  cat <<'USAGE'
Run Qwen3-VL-8B as a fixed-observation SEEK-Bench baseline and compute paper metrics.

Default protocol:
  - model: /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/qwen3-vl-8b-Instruct
  - data: data_utils/msad_saver_runtime_test.jsonl
  - visual budget: 16 full-video preview frames
  - inference: single-shot, no SEEK tools, deterministic decoding
  - metrics: metrics.json, semantic_metrics.json, table_metrics.json from saver_v3 evaluator

Usage:
  bash scripts/run_qwen3_vl_fixed16_eval.sh [--run] [options]

Options:
  --run                       Execute. Without this flag, only print the torchrun command.
  --config PATH               Baseline YAML config. Default: configs/fixed_baseline_eval/vllm_qwen3_vl_8b_fixed_16.yaml
  --model-path PATH           Qwen3-VL-8B model path.
  --data-path PATH            SEEK runtime test JSONL.
  --data-root PATH            Video/cache data root.
  --output-dir PATH           Output artifact directory.
  --include-splits SPLITS     Split filter passed to evaluator. Default: test
  --frame-budget N            Fixed visual frame/image budget. Default: 16
  --max-records N             Smoke-test cap. Default: 0 means all records.
  --nproc-per-node N          Number of local torchrun ranks. Default: 3
  --master-port PORT          torchrun master port. Default: 29731
  --override KEY=VALUE        Extra saver_v3 config override. May be repeated.
  -h, --help                  Show this help.

Examples:
  # Dry-run command preview
  bash scripts/run_qwen3_vl_fixed16_eval.sh

  # Full test split evaluation on the server
  conda activate qwen3-vl
  cd /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3
  bash scripts/run_qwen3_vl_fixed16_eval.sh --run

  # 20-sample smoke test
  bash scripts/run_qwen3_vl_fixed16_eval.sh --run \
    --max-records 20 \
    --output-dir artifacts/fixed_baseline_16/qwen3_vl_8b_smoke
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      RUN_MODE="run"
      shift
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --data-path)
      DATA_PATH="$2"
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --include-splits)
      INCLUDE_SPLITS="$2"
      shift 2
      ;;
    --frame-budget)
      FRAME_BUDGET="$2"
      shift 2
      ;;
    --max-records)
      MAX_RECORDS="$2"
      shift 2
      ;;
    --nproc-per-node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --master-port)
      MASTER_PORT="$2"
      shift 2
      ;;
    --override)
      EXTRA_OVERRIDES+=("$2")
      EXTRA_OVERRIDE_COUNT=$((EXTRA_OVERRIDE_COUNT + 1))
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${ROOT_DIR}/${CONFIG_PATH}"
fi

CMD_ARGS=()
if [[ "${RUN_MODE}" == "run" ]]; then
  CMD_ARGS+=(--run)
fi
CMD_ARGS+=(
  --override "base_model=${MODEL_PATH}"
  --override "client.num_preview_frames=${FRAME_BUDGET}"
  --override "client.max_total_images=${FRAME_BUDGET}"
  --override "io.data_path=${DATA_PATH}"
  --override "io.data_root=${DATA_ROOT}"
  --override "io.output_dir=${OUTPUT_DIR}"
  --override "io.include_splits=${INCLUDE_SPLITS}"
  --override "io.max_records=${MAX_RECORDS}"
)

if (( EXTRA_OVERRIDE_COUNT > 0 )); then
  for override in "${EXTRA_OVERRIDES[@]}"; do
    CMD_ARGS+=(--override "${override}")
  done
fi

export NNODES
export NPROC_PER_NODE
export NODE_RANK
export MASTER_ADDR
export MASTER_PORT
export BASELINE_CONFIG="${CONFIG_PATH}"

printf 'Qwen3-VL Fixed-%s SEEK metric evaluation\n' "${FRAME_BUDGET}"
printf '  model: %s\n' "${MODEL_PATH}"
printf '  data: %s\n' "${DATA_PATH}"
printf '  data_root: %s\n' "${DATA_ROOT}"
printf '  output_dir: %s\n' "${OUTPUT_DIR}"
printf '  nproc_per_node: %s\n' "${NPROC_PER_NODE}"
printf '\n'

(
  cd "${ROOT_DIR}"
  bash scripts/run_fixed_baseline_eval_vllm.sh "${CMD_ARGS[@]}"
)

if [[ "${RUN_MODE}" != "run" ]]; then
  exit 0
fi

TABLE_METRICS_PATH="${OUTPUT_DIR}/table_metrics.json"
METRICS_PATH="${OUTPUT_DIR}/metrics.json"
SEMANTIC_METRICS_PATH="${OUTPUT_DIR}/semantic_metrics.json"

if [[ "${OUTPUT_DIR}" != /* ]]; then
  TABLE_METRICS_PATH="${ROOT_DIR}/${TABLE_METRICS_PATH}"
  METRICS_PATH="${ROOT_DIR}/${METRICS_PATH}"
  SEMANTIC_METRICS_PATH="${ROOT_DIR}/${SEMANTIC_METRICS_PATH}"
fi

printf '\nEvaluation artifacts:\n'
printf '  table metrics: %s\n' "${TABLE_METRICS_PATH}"
printf '  full metrics:  %s\n' "${METRICS_PATH}"
printf '  semantic:      %s\n' "${SEMANTIC_METRICS_PATH}"

if [[ -f "${TABLE_METRICS_PATH}" ]]; then
  python3 - "${TABLE_METRICS_PATH}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
paper_metrics = payload.get("paper_metrics") or {}
print("\nPaper metrics:")
for key in (
    "Existence Acc.",
    "Temporal mIoU",
    "QA Accuracy",
    "Event-Chain F1",
    "Evidence F1@3",
    "FECV Sufficiency",
):
    print(f"  {key}: {paper_metrics.get(key)}")
PY
else
  printf '\nMissing table_metrics.json; check torchrun logs above.\n' >&2
  exit 1
fi
