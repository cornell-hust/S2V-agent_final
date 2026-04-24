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
MASTER_PORT="${MASTER_PORT:-29730}"
BASELINE_CONFIG="${BASELINE_CONFIG:-${ROOT_DIR}/configs/fixed_baseline_eval/vllm_qwen3_vl_8b_fixed.yaml}"

CMD=(
  torchrun
  --nnodes "${NNODES}"
  --nproc-per-node "${NPROC_PER_NODE}"
  --node-rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
  --module saver_v3.cli.run_fixed_baseline_eval_vllm
  --config "${BASELINE_CONFIG}"
)

if (( ${#EXTRA_ARGS[@]} > 0 )); then
  CMD+=("${EXTRA_ARGS[@]}")
fi

printf 'Fixed baseline eval command (direct HF torchrun backend):\n'
printf '  %q' "${CMD[@]}"
printf '\n'

if [[ "${RUN_MODE}" != "run" ]]; then
  printf 'Dry run only. Pass --run to execute the multi-rank direct-HF fixed baseline eval job.\n'
  exit 0
fi

exec "${CMD[@]}"


# bash scripts/run_fixed_baseline_eval_vllm.sh --run \
#     --override 'io.data_path=data_utils/msad_saver_runtime_test.jsonl' \
#     --override 'io.data_root=/mnt/shared-storage-user/mineru2-shared/zengweijun' \
#     --override 'io.output_dir=artifacts/fixed_baseline_28/msad_test'


# • 会。脚本完成后，rank 0 会立刻合并各卡输出并计算评价指标。

#   指标和输出都在你指定的目录：

#   artifacts/fixed_baseline/msad_test/

#   关键文件：

#   - artifacts/fixed_baseline/msad_test/metrics.json：完整指标汇总，含 primary/secondary 指标和 paper_metrics
#   - artifacts/fixed_baseline/msad_test/semantic_metrics.json：QA Accuracy 等语义指标
#   - artifacts/fixed_baseline/msad_test/table_metrics.json：最适合直接填论文表格的 6 列格式，其中 fixed baseline 的 FECV Sufficiency 是 null，对应论文里写 —
#   - artifacts/fixed_baseline/msad_test/raw_predictions.merged.jsonl：模型原始输出
#   - artifacts/fixed_baseline/msad_test/normalized_predictions.merged.jsonl：解析后的 baseline 输出
#   - artifacts/fixed_baseline/msad_test/scored_predictions.merged.jsonl：给 scorer 使用的格式化输出

#   跑完后看表格指标：

#   cat artifacts/fixed_baseline/msad_test/table_metrics.json

#   或看完整指标：

#   cat artifacts/fixed_baseline/msad_test/metrics.json