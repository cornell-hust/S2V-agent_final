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

CONFIG="${PREPARE_SFT_CONFIG:-${ROOT_DIR}/configs/prepare_sft/qwen3_vl_8b_prepare.yaml}"
CMD=(python -m saver_v3.cli.prepare_sft_manifest --config "${CONFIG}")

if (( ${#EXTRA_ARGS[@]} > 0 )); then
  CMD+=("${EXTRA_ARGS[@]}")
fi

printf 'Prepare-SFT-manifest command:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

if [[ "${RUN_MODE}" != "run" ]]; then
  printf 'Dry run only. Pass --run to execute compact_trace_v5 SFT preparation.\n'
  exit 0
fi

exec "${CMD[@]}"
