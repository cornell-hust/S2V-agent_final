#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_MODE="print"
EXTRA_ARGS=()


resolve_cuda_runtime_lib_dirs() {
  local conda_prefix="${CONDA_PREFIX:-}"
  local py_ver=""
  local dirs=()
  if [[ -n "${conda_prefix}" ]]; then
    py_ver="$(python - <<'PY'
import sys
print(f"python{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
    dirs+=(
      "${conda_prefix}/lib/${py_ver}/site-packages/nvidia/cuda_runtime/lib"
      "${conda_prefix}/lib/${py_ver}/site-packages/nvidia/curand/lib"
    )
  fi
  local existing=()
  local d
  for d in "${dirs[@]}"; do
    if [[ -d "${d}" ]]; then
      existing+=("${d}")
    fi
  done
  printf '%s\n' "${existing[@]}"
}

link_fake_cuda_runtime_libs() {
  local conda_prefix="${CONDA_PREFIX:-}"
  [[ -n "${conda_prefix}" ]] || return 0
  local fake_cuda_lib64="${conda_prefix}/fake_cuda/lib64"
  mkdir -p "${fake_cuda_lib64}"
  local py_ver="$(python - <<'PY'
import sys
print(f"python{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  local libcudart_real="${conda_prefix}/lib/${py_ver}/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
  local libcurand_real="${conda_prefix}/lib/${py_ver}/site-packages/nvidia/curand/lib/libcurand.so.10"
  if [[ -f "${libcudart_real}" ]]; then
    ln -sfn "${libcudart_real}" "${fake_cuda_lib64}/libcudart.so.12"
    ln -sfn "${libcudart_real}" "${fake_cuda_lib64}/libcudart.so"
  fi
  if [[ -f "${libcurand_real}" ]]; then
    ln -sfn "${libcurand_real}" "${fake_cuda_lib64}/libcurand.so.10"
    ln -sfn "${libcurand_real}" "${fake_cuda_lib64}/libcurand.so"
  fi
}

export_rl_cuda_link_env() {
  link_fake_cuda_runtime_libs
  local lib_dirs=()
  mapfile -t lib_dirs < <(resolve_cuda_runtime_lib_dirs)
  if (( ${#lib_dirs[@]} > 0 )); then
    local joined=""
    local d
    for d in "${lib_dirs[@]}"; do
      if [[ -z "${d}" ]]; then
        continue
      fi
      if [[ -z "${joined}" ]]; then
        joined="${d}"
      else
        joined="${joined}:${d}"
      fi
    done
    export LD_LIBRARY_PATH="${joined}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
  if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/fake_cuda" ]]; then
    export CUDA_HOME="${CONDA_PREFIX}/fake_cuda"
  fi
}

prebuild_deepspeed_cpu_adam() {
  python - <<'PY'
from deepspeed.ops.op_builder import CPUAdamBuilder
CPUAdamBuilder().load(verbose=True)
PY
}

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

CONFIG="${RL_CONFIG:-${ROOT_DIR}/configs/rl/qwen3_vl_8b_grpo_train.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-${ROOT_DIR}/configs/model/qwen3_vl_8b_full.yaml}"
ATTENTION_CONFIG="${ATTENTION_CONFIG:-${ROOT_DIR}/configs/model/attention_fa3_only.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${ROOT_DIR}/configs/deepspeed/zero2_rl.json}"

CMD=(
  deepspeed
  --num_nodes "${NNODES}"
  --num_gpus "${NPROC_PER_NODE}"
  --node_rank "${NODE_RANK}"
  --master_addr "${MASTER_ADDR}"
  --master_port "${MASTER_PORT}"
  --module saver_v3.cli.train_rl_ds
  --config "${CONFIG}"
  --model-config "${MODEL_CONFIG}"
  --attention-config "${ATTENTION_CONFIG}"
  --deepspeed-config "${DEEPSPEED_CONFIG}"
)

if (( ${#EXTRA_ARGS[@]} > 0 )); then
  CMD+=("${EXTRA_ARGS[@]}")
fi

printf 'Template RL command:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

if [[ "${RUN_MODE}" != "run" ]]; then
  printf 'Dry run only. Pass --run after saver_v3.cli.train_rl_ds exists.\n'
  exit 0
fi

export_rl_cuda_link_env
prebuild_deepspeed_cpu_adam

python - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("saver_v3.cli.train_rl_ds") is None:
    sys.stderr.write("Missing module saver_v3.cli.train_rl_ds\n")
    raise SystemExit(1)
PY

exec "${CMD[@]}"
