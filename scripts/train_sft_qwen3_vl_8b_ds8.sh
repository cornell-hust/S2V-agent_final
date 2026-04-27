#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/lib/cache_env.sh"
setup_agenticvau_cache_env
RUN_MODE="print"
EXTRA_ARGS=()

for arg in "$@"; do
  if [[ "${arg}" == "--run" ]]; then
    RUN_MODE="run"
  else
    EXTRA_ARGS+=("${arg}")
  fi
done

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

CONFIG="${SFT_CONFIG:-${ROOT_DIR}/configs/sft/qwen3_vl_8b_full_train.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-${ROOT_DIR}/configs/model/qwen3_vl_8b_full.yaml}"
ATTENTION_CONFIG="${ATTENTION_CONFIG:-${ROOT_DIR}/configs/model/attention_fa3_only.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${ROOT_DIR}/configs/deepspeed/zero3_full_model.json}"

CMD=(
  deepspeed
  --num_nodes "${NNODES}"
  --num_gpus "${NPROC_PER_NODE}"
  --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
  --module saver_v3.cli.train_sft_ds
  --config "${CONFIG}"
  --model-config "${MODEL_CONFIG}"
  --attention-config "${ATTENTION_CONFIG}"
  --deepspeed-config "${DEEPSPEED_CONFIG}"
)

if (( ${#EXTRA_ARGS[@]} > 0 )); then
  CMD+=("${EXTRA_ARGS[@]}")
fi

printf 'Template SFT command:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

if [[ "${RUN_MODE}" != "run" ]]; then
  printf 'Dry run only. Pass --run to execute the 8-GPU DeepSpeed compact_trace_v5 SFT job.\n'
  exit 0
fi

python - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("saver_v3.cli.train_sft_ds") is None:
    sys.stderr.write("Missing module saver_v3.cli.train_sft_ds\n")
    raise SystemExit(1)
PY

exec "${CMD[@]}"
