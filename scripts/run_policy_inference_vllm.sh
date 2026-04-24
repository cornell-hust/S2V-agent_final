#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
MASTER_PORT="${MASTER_PORT:-29720}"
ROLLOUT_CONFIG="${ROLLOUT_CONFIG:-${ROOT_DIR}/configs/inference/vllm_qwen3_vl_8b_rollout.yaml}"

CMD=(
  torchrun
  --nnodes "${NNODES}"
  --nproc-per-node "${NPROC_PER_NODE}"
  --node-rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
  --module saver_v3.cli.run_policy_rollout_vllm
  --config "${ROLLOUT_CONFIG}"
)

if (( ${#EXTRA_ARGS[@]} > 0 )); then
  CMD+=("${EXTRA_ARGS[@]}")
fi

printf 'Policy-rollout command:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

if [[ "${RUN_MODE}" != "run" ]]; then
  printf 'Dry run only. Pass --run to execute the 8-GPU full-rollout vLLM inference job.\n'
  exit 0
fi

exec "${CMD[@]}"
