#!/usr/bin/env bash

setup_agenticvau_cache_env() {
  local default_data_root="/mnt/shared-storage-user/mineru2-shared/zengweijun"
  local data_root="${SAVER_V3_DATA_ROOT:-${DATA_ROOT:-${default_data_root}}}"
  local cache_root="${AGENTICVAU_CACHE_ROOT:-${data_root}/cache/agenticvau}"

  export AGENTICVAU_CACHE_ROOT="${cache_root}"
  export TMPDIR="${AGENTICVAU_TMPDIR:-${cache_root}/tmp}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${cache_root}/xdg}"
  export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${cache_root}/vllm}"
  export VLLM_ASSETS_CACHE="${VLLM_ASSETS_CACHE:-${VLLM_CACHE_ROOT}/assets}"
  export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${cache_root}/torch_inductor}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${cache_root}/triton}"
  export HF_HOME="${HF_HOME:-${cache_root}/huggingface}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${cache_root}/pip}"
  export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${cache_root}/torch_extensions}"

  mkdir -p \
    "${AGENTICVAU_CACHE_ROOT}" \
    "${TMPDIR}" \
    "${XDG_CACHE_HOME}" \
    "${VLLM_CACHE_ROOT}" \
    "${VLLM_ASSETS_CACHE}" \
    "${TORCHINDUCTOR_CACHE_DIR}" \
    "${TRITON_CACHE_DIR}" \
    "${HF_HOME}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${PIP_CACHE_DIR}" \
    "${TORCH_EXTENSIONS_DIR}"

  printf '[INFO] AgenticVAU cache root: %s\n' "${AGENTICVAU_CACHE_ROOT}"
}
