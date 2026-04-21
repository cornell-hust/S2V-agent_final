#!/usr/bin/env bash
# run_full_pipeline.sh — Complete SFT → RL training pipeline for idea2_v3
# Stages: Setup → Data check → SFT (3 epochs w/ per-epoch eval) → RL → RL eval → Summary
set -euo pipefail

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

banner() {
  local color="${1}" msg="${2}"
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  printf "\n${color}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
  printf "  [%s] %s\n" "${ts}" "${msg}"
  printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n\n"
}

info()  { printf "${CYAN}[INFO]${RESET}  %s\n" "$*"; }
skip()  { printf "${YELLOW}[SKIP]${RESET}  %s\n" "$*"; }
ok()    { printf "${GREEN}[OK]${RESET}    %s\n" "$*"; }
err()   { printf "${RED}[ERROR]${RESET} %s\n" "$*" >&2; }

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


collect_sft_epoch_output_issues() {
  local epoch_dir="$1"
  local processor_sentinel_found=0
  local model_dir=""
  [[ -d "${epoch_dir}" ]] || { printf "%s\n" "missing directory"; return 0; }
  [[ -f "${epoch_dir}/sft_summary.json" ]] || printf "%s\n" "missing sft_summary.json"
  model_dir="$(resolve_sft_epoch_model_path "${epoch_dir}" 2>/dev/null || true)"
  if [[ -z "${model_dir}" ]]; then
    printf "%s\n" "missing authoritative model path"
    return 0
  fi
  [[ -f "${model_dir}/config.json" ]] || printf "%s\n" "missing config.json"
  for sentinel in preprocessor_config.json processor_config.json tokenizer_config.json; do
    if [[ -f "${model_dir}/${sentinel}" ]]; then
      processor_sentinel_found=1
      break
    fi
  done
  if (( ! processor_sentinel_found )); then
    printf "%s\n" "missing processor/tokenizer config"
  fi
}

resolve_sft_epoch_model_path() {
  local epoch_dir="$1"
  python3 - "${epoch_dir}" <<'PY'
import json
import sys
from pathlib import Path

epoch_dir = Path(sys.argv[1]).expanduser().resolve()
summary_path = epoch_dir / "sft_summary.json"
candidates = []
if summary_path.exists():
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if isinstance(payload, dict):
        for key in ("authoritative_model_path", "latest_checkpoint", "output_dir"):
            value = str(payload.get(key) or "").strip()
            if value:
                candidates.append(Path(value).expanduser().resolve())
candidates.append(epoch_dir)
for candidate in candidates:
    if candidate.exists():
        print(candidate)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

is_complete_sft_epoch_output() {
  local epoch_dir="$1"
  local issues=()
  mapfile -t issues < <(collect_sft_epoch_output_issues "${epoch_dir}")
  (( ${#issues[@]} == 0 ))
}

require_valid_sft_epoch_output() {
  local epoch_dir="$1"
  local label="${2:-SFT epoch output}"
  local issues=()
  mapfile -t issues < <(collect_sft_epoch_output_issues "${epoch_dir}")
  if (( ${#issues[@]} > 0 )); then
    err "${label} is not a valid HF checkpoint at ${epoch_dir}: ${issues[*]}"
    exit 1
  fi
}

collect_hf_checkpoint_dir_issues() {
  local model_dir="$1"
  local processor_sentinel_found=0
  [[ -d "${model_dir}" ]] || { printf "%s\n" "missing directory"; return 0; }
  [[ -f "${model_dir}/config.json" ]] || printf "%s\n" "missing config.json"
  if [[ -f "${model_dir}/adapter_config.json" ]]; then
    return 0
  fi
  for sentinel in preprocessor_config.json processor_config.json tokenizer_config.json; do
    if [[ -f "${model_dir}/${sentinel}" ]]; then
      processor_sentinel_found=1
      break
    fi
  done
  if (( ! processor_sentinel_found )); then
    printf "%s\n" "missing processor/tokenizer config"
  fi
}

is_valid_hf_checkpoint_dir() {
  local model_dir="$1"
  local issues=()
  mapfile -t issues < <(collect_hf_checkpoint_dir_issues "${model_dir}")
  (( ${#issues[@]} == 0 ))
}

resolve_rl_latest_checkpoint_path() {
  local rl_dir="$1"
  python3 - "${rl_dir}" <<'PY'
import json
import sys
from pathlib import Path

rl_dir = Path(sys.argv[1]).expanduser().resolve()
candidates = []
summary_path = rl_dir / "rl_summary.json"
if summary_path.exists():
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if isinstance(payload, dict):
        value = str(payload.get("latest_checkpoint") or "").strip()
        if value:
            candidates.append(Path(value).expanduser().resolve())
latest_txt = rl_dir / "latest_checkpoint.txt"
if latest_txt.exists():
    value = str(latest_txt.read_text(encoding="utf-8")).strip()
    if value:
        candidates.append(Path(value).expanduser().resolve())
for candidate in candidates:
    if candidate.exists():
        print(candidate)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

collect_rl_output_issues() {
  local rl_dir="$1"
  local latest_checkpoint=""
  [[ -d "${rl_dir}" ]] || { printf "%s\n" "missing directory"; return 0; }
  [[ -f "${rl_dir}/rl_summary.json" ]] || printf "%s\n" "missing rl_summary.json"
  [[ -f "${rl_dir}/latest_checkpoint.txt" ]] || printf "%s\n" "missing latest_checkpoint.txt"
  latest_checkpoint="$(resolve_rl_latest_checkpoint_path "${rl_dir}" 2>/dev/null || true)"
  if [[ -z "${latest_checkpoint}" ]]; then
    printf "%s\n" "missing latest RL checkpoint path"
    return 0
  fi
  local issues=()
  mapfile -t issues < <(collect_hf_checkpoint_dir_issues "${latest_checkpoint}")
  if (( ${#issues[@]} > 0 )); then
    local issue
    for issue in "${issues[@]}"; do
      printf "%s\n" "latest checkpoint invalid: ${issue}"
    done
  fi
}

is_complete_rl_output() {
  local rl_dir="$1"
  local issues=()
  mapfile -t issues < <(collect_rl_output_issues "${rl_dir}")
  (( ${#issues[@]} == 0 ))
}

require_valid_rl_output() {
  local rl_dir="$1"
  local label="${2:-RL output}"
  local issues=()
  mapfile -t issues < <(collect_rl_output_issues "${rl_dir}")
  if (( ${#issues[@]} > 0 )); then
    err "${label} is not a valid RL checkpoint root at ${rl_dir}: ${issues[*]}"
    exit 1
  fi
}

validate_materialized_cache() {
  local cache_path="$1"
  local expected_format="$2"
  local source_path="$3"
  local include_splits="$4"
  python3 - "$cache_path" "$expected_format" "$source_path" "$include_splits" <<'INNERPY'
from saver_v3.data.materialized_cache import ensure_materialized_cache_metadata
import sys
ensure_materialized_cache_metadata(
    sys.argv[1],
    expected_format=sys.argv[2],
    expected_source_path=sys.argv[3],
    expected_include_splits=sys.argv[4] or None,
    require_source=True,
)
INNERPY
}

build_materialized_cache() {
  local mode="$1"
  local input_path="$2"
  local output_path="$3"
  local include_splits="$4"
  shift 4
  python3 prepare_materialized_cache.py     --mode "$mode"     --input "$input_path"     --output "$output_path"     --include-splits "$include_splits"     --overwrite-existing     "$@"
}

resolve_materialized_cache() {
  local label="$1"
  local mode="$2"
  local input_path="$3"
  local output_path="$4"
  local expected_format="$5"
  local include_splits="$6"
  shift 6
  local rebuild=0
  if [[ -f "$output_path" && -f "${output_path}.meta.json" ]]; then
    if validate_materialized_cache "$output_path" "$expected_format" "$input_path" "$include_splits"; then
      skip "$label already valid at $output_path — skipping rebuild."
      return 0
    fi
    info "$label is stale or invalid at $output_path — rebuilding."
    rm -f "$output_path" "${output_path}.meta.json" "${output_path}.summary.json"
    rebuild=1
  fi
  if [[ ! -f "$output_path" ]]; then
    if (( rebuild == 0 )); then
      info "Building $label -> $output_path"
    fi
    build_materialized_cache "$mode" "$input_path" "$output_path" "$include_splits" "$@"
    validate_materialized_cache "$output_path" "$expected_format" "$input_path" "$include_splits"
    ok "$label ready: $output_path"
  fi
}


# ---------------------------------------------------------------------------
# Stage 0: Experiment name & directory setup
# ---------------------------------------------------------------------------
banner "${GREEN}" "Stage 0: Setup"

if [[ -n "${EXP_NAME:-}" ]]; then
  info "Using EXP_NAME from environment: ${EXP_NAME}"
else
  printf "${BOLD}Enter experiment name (e.g. exp8): ${RESET}"
  read -r EXP_NAME
  if [[ -z "${EXP_NAME}" ]]; then
    err "Experiment name cannot be empty."
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment variables)
# ---------------------------------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

DATA_ROOT="${DATA_ROOT:-/mnt/shared-storage-user/mineru2-shared/zengweijun}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data_utils}"
MODEL_PATH="${MODEL_PATH:-${DATA_ROOT}/Wmh/MLLMs/qwen3-vl-8b-Instruct}"
PROPOSAL_MODEL_PATH="${PROPOSAL_MODEL_PATH:-${DATA_ROOT}/Wmh/MLLMs/siglip}"
PROPOSAL_TORCH_DTYPE="${PROPOSAL_TORCH_DTYPE:-auto}"
RL_PROPOSAL_TORCH_DTYPE="${RL_PROPOSAL_TORCH_DTYPE:-float16}"
PROPOSAL_DEVICE="${PROPOSAL_DEVICE:-}"

SFT_CONFIG="${SFT_CONFIG:-${ROOT_DIR}/configs/sft/qwen3_vl_8b_full_train.yaml}"
PREPARE_SFT_CONFIG="${PREPARE_SFT_CONFIG:-${ROOT_DIR}/configs/prepare_sft/qwen3_vl_8b_prepare.yaml}"
RL_CONFIG="${RL_CONFIG:-${ROOT_DIR}/configs/rl/qwen3_vl_8b_grpo_train.yaml}"
ROLLOUT_CONFIG="${ROLLOUT_CONFIG:-${ROOT_DIR}/configs/rollout_eval/vllm_qwen3_vl_8b.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-${ROOT_DIR}/configs/model/qwen3_vl_8b_full.yaml}"
ATTENTION_CONFIG="${ATTENTION_CONFIG:-${ROOT_DIR}/configs/model/attention_fa3_only.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${ROOT_DIR}/configs/deepspeed/zero3_full_model.json}"
RL_DEEPSPEED_CONFIG="${RL_DEEPSPEED_CONFIG:-${ROOT_DIR}/configs/deepspeed/zero2_rl.json}"

NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
EVAL_MASTER_PORT="${EVAL_MASTER_PORT:-29710}"

BASELINE_NPROC_PER_NODE="${BASELINE_NPROC_PER_NODE:-8}"
BASELINE_GRADIENT_ACCUMULATION_STEPS="${BASELINE_GRADIENT_ACCUMULATION_STEPS:-8}"
DEFAULT_EQUIVALENT_GRAD_ACC="$(( (BASELINE_NPROC_PER_NODE * BASELINE_GRADIENT_ACCUMULATION_STEPS + NPROC_PER_NODE - 1) / NPROC_PER_NODE ))"
SFT_GRADIENT_ACCUMULATION_STEPS="${SFT_GRADIENT_ACCUMULATION_STEPS:-${DEFAULT_EQUIVALENT_GRAD_ACC}}"
RL_GRPO_NUM_GENERATIONS="${RL_GRPO_NUM_GENERATIONS:-4}"
RL_GRPO_ROLLOUT_COUNT="${RL_GRPO_ROLLOUT_COUNT:-12}"
RL_GRPO_PER_DEVICE_BATCH="${RL_GRPO_PER_DEVICE_BATCH:-1}"
# GRPO semantic correctness: effective_batch must equal k * num_generations * rollout_count
# so that each optimizer step consumes whole iterations and no group is split across steps.
# For 3 GPUs, pdbs=1, num_generations=4, rollout_count=12: target effective=48 → GA=16.
RL_OPTIMAL_GA="$(( (RL_GRPO_NUM_GENERATIONS * RL_GRPO_ROLLOUT_COUNT) / (NPROC_PER_NODE * RL_GRPO_PER_DEVICE_BATCH) ))"
RL_GRADIENT_ACCUMULATION_STEPS="${RL_GRADIENT_ACCUMULATION_STEPS:-${RL_OPTIMAL_GA}}"

NUM_SFT_EPOCHS="${NUM_SFT_EPOCHS:-3}"
PREPARE_SFT_OVERWRITE_EXISTING="${PREPARE_SFT_OVERWRITE_EXISTING:-0}"
SFT_PREPARED_FILE="${SFT_PREPARED_FILE:-${DATA_DIR}/sft_train.compact_trace_v2.jsonl}"
SFT_PREPARED_META_FILE="${SFT_PREPARED_META_FILE:-${SFT_PREPARED_FILE}.meta.json}"
TEACHER_PREPARED_FILE="${TEACHER_PREPARED_FILE:-${DATA_DIR}/sft_train.teacher_rollout_primary.compact_trace_v2.jsonl}"
TEACHER_PREPARED_META_FILE="${TEACHER_PREPARED_META_FILE:-${TEACHER_PREPARED_FILE}.meta.json}"
TEACHER_JUDGE_MODEL_PATH="${TEACHER_JUDGE_MODEL_PATH:-}"
TEACHER_JUDGE_INPUT_MODE="${TEACHER_JUDGE_INPUT_MODE:-auto}"
TEACHER_JUDGE_TORCH_DTYPE="${TEACHER_JUDGE_TORCH_DTYPE:-auto}"
TEACHER_JUDGE_DEVICE_MAP="${TEACHER_JUDGE_DEVICE_MAP:-auto}"
TEACHER_JUDGE_ATTN_IMPLEMENTATION="${TEACHER_JUDGE_ATTN_IMPLEMENTATION:-}"
TEACHER_JUDGE_MAX_NEW_TOKENS="${TEACHER_JUDGE_MAX_NEW_TOKENS:-384}"
TEACHER_JUDGE_MAX_IMAGES="${TEACHER_JUDGE_MAX_IMAGES:-8}"
TEACHER_JUDGE_TOPK_FRAMES_PER_VIEW="${TEACHER_JUDGE_TOPK_FRAMES_PER_VIEW:-4}"
TEACHER_JUDGE_FRAME_CACHE_MAX_CACHED_VIDEOS="${TEACHER_JUDGE_FRAME_CACHE_MAX_CACHED_VIDEOS:-64}"
TEACHER_JUDGE_BATCH_SIZE="${TEACHER_JUDGE_BATCH_SIZE:-1}"
TEACHER_JUDGE_OVERWRITE_EXISTING="${TEACHER_JUDGE_OVERWRITE_EXISTING:-0}"
RUNTIME_TRAIN_FILE="${RUNTIME_TRAIN_FILE:-${DATA_DIR}/msad_saver_runtime_train.jsonl}"
RUNTIME_EVAL_FILE="${RUNTIME_EVAL_FILE:-${DATA_DIR}/msad_saver_runtime_test.jsonl}"
TRAIN_INCLUDE_SPLIT="${TRAIN_INCLUDE_SPLIT:-train}"
EVAL_INCLUDE_SPLIT="${EVAL_INCLUDE_SPLIT:-test}"
SFT_MATERIALIZED_FILE="${SFT_MATERIALIZED_FILE:-${DATA_DIR}/sft_train.compact_trace_v2.materialized_messages_v1.jsonl}"
RUNTIME_TRAIN_ITEMS_FILE="${RUNTIME_TRAIN_ITEMS_FILE:-${DATA_DIR}/msad_saver_runtime_train.materialized_items_v1.jsonl}"
RUNTIME_EVAL_ITEMS_FILE="${RUNTIME_EVAL_ITEMS_FILE:-${DATA_DIR}/msad_saver_runtime_test.materialized_items_v1.jsonl}"

# Derived paths
ARTIFACTS_DIR="${ROOT_DIR}/artifacts/${EXP_NAME}"
SFT_DIR="${ARTIFACTS_DIR}/sft"
RL_DIR="${ARTIFACTS_DIR}/rl"
EVAL_DIR="${ARTIFACTS_DIR}/eval"
LOG_DIR="${ARTIFACTS_DIR}/logs"

# Create directory structure
mkdir -p "${SFT_DIR}" "${RL_DIR}" "${EVAL_DIR}" "${LOG_DIR}"
ok "Created artifact directories under ${ARTIFACTS_DIR}"

# Set up logging via tee — all subsequent output also goes to log file
LOG_FILE="${LOG_DIR}/pipeline_$(date '+%Y%m%d_%H%M%S').log"
# Reopen stdout/stderr through tee into the log file (subshell trick)
exec > >(tee -a "${LOG_FILE}") 2>&1
info "Logging to ${LOG_FILE}"

# Print resolved configuration
info "Configuration:"
printf "  ROOT_DIR           = %s\n" "${ROOT_DIR}"
printf "  DATA_DIR           = %s\n" "${DATA_DIR}"
printf "  EXP_NAME           = %s\n" "${EXP_NAME}"
printf "  ARTIFACTS_DIR      = %s\n" "${ARTIFACTS_DIR}"
printf "  NUM_SFT_EPOCHS     = %s\n" "${NUM_SFT_EPOCHS}"
printf "  NNODES             = %s\n" "${NNODES}"
printf "  NPROC_PER_NODE     = %s\n" "${NPROC_PER_NODE}"
printf "  EVAL_MASTER_PORT   = %s\n" "${EVAL_MASTER_PORT}"
printf "  SFT_GRAD_ACC       = %s\n" "${SFT_GRADIENT_ACCUMULATION_STEPS}"
printf "  RL_GRAD_ACC        = %s\n" "${RL_GRADIENT_ACCUMULATION_STEPS}"
printf "  SFT_CONFIG         = %s\n" "${SFT_CONFIG}"
printf "  PREPARE_SFT_CONFIG = %s\n" "${PREPARE_SFT_CONFIG}"
printf "  RL_CONFIG          = %s\n" "${RL_CONFIG}"
printf "  ROLLOUT_CONFIG     = %s\n" "${ROLLOUT_CONFIG}"
printf "  DEEPSPEED_CONFIG   = %s\n" "${DEEPSPEED_CONFIG}"
printf "  RL_DEEPSPEED_CONFIG= %s\n" "${RL_DEEPSPEED_CONFIG}"
printf "  RL_PROPOSAL_DTYPE  = %s\n" "${RL_PROPOSAL_TORCH_DTYPE}"
printf "  SFT_PREPARED_FILE      = %s\n" "${SFT_PREPARED_FILE}"
printf "  TEACHER_PREPARED_FILE  = %s\n" "${TEACHER_PREPARED_FILE}"
printf "  TEACHER_JUDGE_MODEL    = %s\n" "${TEACHER_JUDGE_MODEL_PATH:-disabled}"
printf "  RUNTIME_TRAIN_FILE     = %s\n" "${RUNTIME_TRAIN_FILE}"
printf "  RUNTIME_EVAL_FILE      = %s\n" "${RUNTIME_EVAL_FILE}"
printf "  SFT_MATERIALIZED_FILE  = %s\n" "${SFT_MATERIALIZED_FILE}"
printf "  RUNTIME_TRAIN_ITEMS    = %s\n" "${RUNTIME_TRAIN_ITEMS_FILE}"
printf "  RUNTIME_EVAL_ITEMS     = %s\n" "${RUNTIME_EVAL_ITEMS_FILE}"
printf "  PROPOSAL_MODEL         = %s\n" "${PROPOSAL_MODEL_PATH}"

# ---------------------------------------------------------------------------
# Stage 1: Data preprocessing check
# ---------------------------------------------------------------------------
banner "${GREEN}" "Stage 1: Data Preprocessing Check"

MISSING=0
for f in "${PREPARE_SFT_CONFIG}" "${SFT_CONFIG}" "${RL_CONFIG}" "${ROLLOUT_CONFIG}" "${MODEL_CONFIG}" "${ATTENTION_CONFIG}" "${DEEPSPEED_CONFIG}" "${RL_DEEPSPEED_CONFIG}" "${RUNTIME_TRAIN_FILE}" "${RUNTIME_EVAL_FILE}"; do
  if [[ ! -f "${f}" ]]; then
    err "Missing required file: ${f}"
    MISSING=1
  fi
done
if [[ -z "${PROPOSAL_MODEL_PATH}" || ! -e "${PROPOSAL_MODEL_PATH}" ]]; then
  err "Missing required proposal model path: ${PROPOSAL_MODEL_PATH:-'(empty)'}"
  MISSING=1
fi
if [[ -n "${TEACHER_JUDGE_MODEL_PATH}" && ! -e "${TEACHER_JUDGE_MODEL_PATH}" ]]; then
  err "Missing teacher judge model path: ${TEACHER_JUDGE_MODEL_PATH}"
  MISSING=1
fi
if (( MISSING )); then
  err "One or more required data/model paths are missing. Aborting."
  exit 1
fi

TRAIN_LINES="$(wc -l < "${RUNTIME_TRAIN_FILE}")"
EVAL_LINES="$(wc -l < "${RUNTIME_EVAL_FILE}")"
ok "RL train data     : ${RUNTIME_TRAIN_FILE} (${TRAIN_LINES} lines)"
ok "Evaluation data   : ${RUNTIME_EVAL_FILE} (${EVAL_LINES} lines)"

python3 - "${RUNTIME_TRAIN_FILE}" "${RUNTIME_EVAL_FILE}" "${DATA_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

runtime_train_path = Path(sys.argv[1]).resolve()
runtime_eval_path = Path(sys.argv[2]).resolve()
data_root = Path(sys.argv[3]).resolve()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no}: expected dict JSONL row")
            yield line_no, payload


def candidate_roots(base: Path, data_path: Path):
    return [
        base,
        base / "data",
        base / "datasets",
        base / "Wmh" / "datasets",
        base / "Wmh" / "datasets" / "MSDA",
        data_path.parent,
    ]


def resolve_video_path(raw_video_path: str, *, data_path: Path) -> Path:
    raw_path = Path(str(raw_video_path))
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path
    variants = [raw_path]
    if raw_path.parts and raw_path.parts[0] in {"data", "datasets"} and len(raw_path.parts) > 1:
        variants.append(Path(*raw_path.parts[1:]))
    for relative_path in variants:
        for root in candidate_roots(data_root, data_path):
            candidate = root / relative_path
            if candidate.exists():
                return candidate
    return candidate_roots(data_root, data_path)[0] / variants[0]


def validate_runtime(path: Path) -> None:
    missing = []
    for line_no, row in iter_jsonl(path):
        video_path = resolve_video_path(str(row.get("video_path") or ""), data_path=path)
        if not video_path.exists():
            missing.append(f"missing video: {path.name}:{line_no}:{video_path}")
            continue
        frame_cache = Path(str(video_path) + ".frame_cache")
        feature_cache = Path(str(video_path) + ".feature_cache")
        if not frame_cache.exists():
            missing.append(f"missing frame_cache: {path.name}:{line_no}:{frame_cache}")
        if not feature_cache.exists():
            missing.append(f"missing feature_cache: {path.name}:{line_no}:{feature_cache}")
        if len(missing) >= 8:
            break
    if missing:
        raise SystemExit("Preprocessing cache validation failed:\n" + "\n".join(missing))



validate_runtime(runtime_train_path)
validate_runtime(runtime_eval_path)
print("[OK] preprocessing cache validation passed for runtime train/eval manifests")
PY

validate_prepared_metadata() {
  local prepared_file="${1}"
  local label="${2}"
  local mode="${3}"
  python3 - "${prepared_file}" "${SFT_CONFIG}" "${MODEL_CONFIG}" "${ATTENTION_CONFIG}" "${DEEPSPEED_CONFIG}" "${TRAIN_INCLUDE_SPLIT}" "${RUNTIME_TRAIN_FILE}" "${SFT_PREPARED_FILE}" "${mode}" <<'PY'
import sys
from saver_v3.sft.runtime import SFTJobConfig, _saver_config_from_dict
from saver_v3.data.prepared_metadata import ensure_prepared_sft_metadata

prepared_file, sft_config, model_config, attention_config, deepspeed_config, include_splits, runtime_train_file, base_prepared_file, mode = sys.argv[1:10]
job = SFTJobConfig.from_files(
    config_path=sft_config,
    model_config_path=model_config,
    attention_config_path=attention_config,
    deepspeed_config_path=deepspeed_config or None,
    config_overrides=[
        f"data.prepared_data_path={prepared_file}",
        f"data.include_splits={include_splits}",
    ],
)
kwargs = dict(
    config=_saver_config_from_dict(job.saver_config),
    require_config_match=True,
)
if mode == "base":
    kwargs.update(
        expected_source_runtime_path=runtime_train_file,
        expected_source_runtime_include_splits=include_splits,
        require_source_runtime=True,
    )
elif mode == "teacher":
    kwargs.update(
        expected_source_runtime_path=runtime_train_file,
        expected_source_runtime_include_splits=include_splits,
        require_source_runtime=True,
        expected_source_prepared_path=base_prepared_file,
        expected_source_prepared_include_splits=include_splits,
        require_source_prepared=True,
        require_teacher_annotated=True,
        require_teacher_rollout_primary_materialized=True,
    )
ensure_prepared_sft_metadata(
    prepared_file,
    **kwargs,
)
PY
}

prepare_base_sft_manifest() {
  python3 -m saver_v3.cli.prepare_sft_manifest \
    --config "${PREPARE_SFT_CONFIG}" \
    --override "saver_config_source=${SFT_CONFIG}" \
    --override "io.input_data_path=${RUNTIME_TRAIN_FILE}" \
    --override "io.output_path=${SFT_PREPARED_FILE}" \
    --override "io.data_root=${DATA_ROOT}" \
    --override "io.include_splits=${TRAIN_INCLUDE_SPLIT}"
}

validate_prepared_metadata_or_die() {
  local prepared_file="${1}"
  local label="${2}"
  local mode="${3}"
  if validate_prepared_metadata "${prepared_file}" "${label}" "${mode}"; then
    ok "${label} metadata matches SFT preview/prompt/rollout_trace config: ${prepared_file}"
  else
    err "${label} metadata does not match current SFT config: ${prepared_file}. Regenerate it before continuing."
    exit 1
  fi
}

banner "${GREEN}" "Stage 1b: Base Prepared SFT Resolve"
BASE_PREPARE_REBUILD_REASON=""
if [[ "${PREPARE_SFT_OVERWRITE_EXISTING}" == "1" ]]; then
  BASE_PREPARE_REBUILD_REASON="PREPARE_SFT_OVERWRITE_EXISTING=1"
elif [[ ! -f "${SFT_PREPARED_FILE}" ]]; then
  BASE_PREPARE_REBUILD_REASON="missing base prepared file"
elif [[ ! -f "${SFT_PREPARED_META_FILE}" ]]; then
  BASE_PREPARE_REBUILD_REASON="missing base prepared metadata"
elif ! validate_prepared_metadata "${SFT_PREPARED_FILE}" "base SFT prepared" "base"; then
  BASE_PREPARE_REBUILD_REASON="base prepared metadata mismatch or stale source"
fi

if [[ -n "${BASE_PREPARE_REBUILD_REASON}" ]]; then
  info "Regenerating base prepared SFT: ${BASE_PREPARE_REBUILD_REASON}"
  prepare_base_sft_manifest
fi

validate_prepared_metadata_or_die "${SFT_PREPARED_FILE}" "base SFT prepared" "base"

EFFECTIVE_SFT_PREPARED_FILE="${SFT_PREPARED_FILE}"
EFFECTIVE_SFT_PREPARED_META_FILE="${SFT_PREPARED_META_FILE}"

if [[ -n "${TEACHER_JUDGE_MODEL_PATH}" ]]; then
  banner "${GREEN}" "Stage 1b: Teacher-Judge Prepared SFT Resolve"
  TEACHER_REBUILD_REASON=""
  if [[ "${TEACHER_JUDGE_OVERWRITE_EXISTING}" == "1" ]]; then
    TEACHER_REBUILD_REASON="TEACHER_JUDGE_OVERWRITE_EXISTING=1"
  elif [[ ! -f "${TEACHER_PREPARED_FILE}" ]]; then
    TEACHER_REBUILD_REASON="missing teacher prepared file"
  elif [[ ! -f "${TEACHER_PREPARED_META_FILE}" ]]; then
    TEACHER_REBUILD_REASON="missing teacher prepared metadata"
  elif ! validate_prepared_metadata "${TEACHER_PREPARED_FILE}" "teacher SFT prepared" "teacher"; then
    TEACHER_REBUILD_REASON="teacher prepared metadata mismatch"
  fi

  if [[ -n "${TEACHER_REBUILD_REASON}" ]]; then
    info "Regenerating teacher prepared SFT: ${TEACHER_REBUILD_REASON}"
    teacher_cmd=(
      python "${ROOT_DIR}/annotate_teacher_judge_sft.py"
      --input "${SFT_PREPARED_FILE}"
      --output "${TEACHER_PREPARED_FILE}"
      --model-path "${TEACHER_JUDGE_MODEL_PATH}"
      --include-splits "${TRAIN_INCLUDE_SPLIT}"
      --input-mode "${TEACHER_JUDGE_INPUT_MODE}"
      --torch-dtype "${TEACHER_JUDGE_TORCH_DTYPE}"
      --device-map "${TEACHER_JUDGE_DEVICE_MAP}"
      --max-new-tokens "${TEACHER_JUDGE_MAX_NEW_TOKENS}"
      --max-images "${TEACHER_JUDGE_MAX_IMAGES}"
      --topk-frames-per-view "${TEACHER_JUDGE_TOPK_FRAMES_PER_VIEW}"
      --frame-cache-max-cached-videos "${TEACHER_JUDGE_FRAME_CACHE_MAX_CACHED_VIDEOS}"
      --batch-size "${TEACHER_JUDGE_BATCH_SIZE}"
      --proposal-model-path "${PROPOSAL_MODEL_PATH}"
      --proposal-torch-dtype "${PROPOSAL_TORCH_DTYPE}"
    )
    if [[ -n "${PROPOSAL_DEVICE}" ]]; then
      teacher_cmd+=(--proposal-device "${PROPOSAL_DEVICE}")
    fi
    if [[ "${TEACHER_JUDGE_OVERWRITE_EXISTING}" == "1" ]]; then
      teacher_cmd+=(--overwrite-existing)
    fi
    if [[ -n "${TEACHER_JUDGE_ATTN_IMPLEMENTATION}" ]]; then
      teacher_cmd+=(--attn-implementation "${TEACHER_JUDGE_ATTN_IMPLEMENTATION}")
    fi
    "${teacher_cmd[@]}"
    validate_prepared_metadata_or_die "${TEACHER_PREPARED_FILE}" "teacher SFT prepared" "teacher"
  else
    ok "Reusing existing teacher prepared SFT: ${TEACHER_PREPARED_FILE}"
  fi
  EFFECTIVE_SFT_PREPARED_FILE="${TEACHER_PREPARED_FILE}"
  EFFECTIVE_SFT_PREPARED_META_FILE="${TEACHER_PREPARED_META_FILE}"
fi

EFFECTIVE_SFT_LINES="$(wc -l < "${EFFECTIVE_SFT_PREPARED_FILE}")"
ok "Effective SFT data : ${EFFECTIVE_SFT_PREPARED_FILE} (${EFFECTIVE_SFT_LINES} lines)"
ok "Effective SFT meta : ${EFFECTIVE_SFT_PREPARED_META_FILE}"

# ---------------------------------------------------------------------------
# Helper: print eval metrics from a metrics.json file (if it exists)
# ---------------------------------------------------------------------------
print_metrics() {
  local metrics_file="${1}"
  local label="${2}"
  if [[ -f "${metrics_file}" ]]; then
    info "Metrics for ${label}:"
    # Pretty-print key-value pairs from JSON without requiring jq
    python3 - "${metrics_file}" "${label}" <<'PY'
import json, sys
path, label = sys.argv[1], sys.argv[2]
with open(path) as fh:
    data = json.load(fh)
for k, v in data.items():
    if isinstance(v, float):
        print(f"    {k:40s} = {v:.4f}")
    else:
        print(f"    {k:40s} = {v}")
PY
  else
    info "No metrics.json found at ${metrics_file}"
  fi
}

# Accumulate summary rows: "label|key=value,key=value,..."
SUMMARY_ROWS=()

record_summary() {
  local label="${1}" metrics_file="${2}"
  if [[ -f "${metrics_file}" ]]; then
    local kvs
    kvs="$(python3 - "${metrics_file}" <<'PY'
import json, sys
with open(sys.argv[1]) as fh:
    data = json.load(fh)
pairs = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in data.items()]
print(",".join(pairs))
PY
    )"
    SUMMARY_ROWS+=("${label}|${kvs}")
  else
    SUMMARY_ROWS+=("${label}|N/A")
  fi
}

resolve_eval_summary_file() {
  local eval_dir="${1}"
  local metrics_file="${eval_dir}/metrics.json"
  local wrapper_file="${eval_dir}/rollout_eval_wrapper_summary.json"
  if [[ -f "${metrics_file}" ]]; then
    printf "%s\n" "${metrics_file}"
  else
    printf "%s\n" "${wrapper_file}"
  fi
}

# ---------------------------------------------------------------------------
# Stage 1c: Materialized Cache Resolve
# ---------------------------------------------------------------------------
banner "${GREEN}" "Stage 1c: Materialized Cache Resolve"

resolve_materialized_cache   "SFT materialized messages"   "sft"   "${EFFECTIVE_SFT_PREPARED_FILE}"   "${SFT_MATERIALIZED_FILE}"   "materialized_sft_messages_v1"   "${TRAIN_INCLUDE_SPLIT}"   --config "${SFT_CONFIG}"   --model-config "${MODEL_CONFIG}"   --proposal-model-path "${PROPOSAL_MODEL_PATH}"   --proposal-torch-dtype "${PROPOSAL_TORCH_DTYPE}"   --proposal-device "${PROPOSAL_DEVICE}"

resolve_materialized_cache   "Runtime train materialized items"   "runtime"   "${RUNTIME_TRAIN_FILE}"   "${RUNTIME_TRAIN_ITEMS_FILE}"   "materialized_runtime_items_v1"   "${TRAIN_INCLUDE_SPLIT}"   --config "${SFT_CONFIG}"   --data-root "${DATA_ROOT}"

resolve_materialized_cache   "Runtime eval materialized items"   "runtime"   "${RUNTIME_EVAL_FILE}"   "${RUNTIME_EVAL_ITEMS_FILE}"   "materialized_runtime_items_v1"   "${EVAL_INCLUDE_SPLIT}"   --config "${SFT_CONFIG}"   --data-root "${DATA_ROOT}"

# ---------------------------------------------------------------------------
# Stage 2: SFT Training — 5 epochs with per-epoch evaluation
# ---------------------------------------------------------------------------
banner "${GREEN}" "Stage 2: SFT Training (${NUM_SFT_EPOCHS} epochs)"

for (( epoch=1; epoch<=NUM_SFT_EPOCHS; epoch++ )); do
  EPOCH_TAG="$(printf '%03d' "${epoch}")"
  EPOCH_OUT="${SFT_DIR}/epoch_${EPOCH_TAG}"

  # ---- Resume-friendly skip ------------------------------------------------
  if is_complete_sft_epoch_output "${EPOCH_OUT}"; then
    skip "SFT epoch ${epoch} checkpoint already complete at ${EPOCH_OUT} — skipping training."
  else
    if [[ -d "${EPOCH_OUT}" ]]; then
      info "Removing stale/incomplete SFT epoch output at ${EPOCH_OUT} before retraining."
      rm -rf "${EPOCH_OUT}"
    fi

    banner "${GREEN}" "Stage 2.${epoch}: SFT Training — Epoch ${epoch}/${NUM_SFT_EPOCHS}"

    sft_cmd=(
      deepspeed
      --num_nodes "${NNODES}"
      --num_gpus "${NPROC_PER_NODE}"
      --module saver_v3.cli.train_sft_ds
      --config "${SFT_CONFIG}"
      --model-config "${MODEL_CONFIG}"
      --attention-config "${ATTENTION_CONFIG}"
      --deepspeed-config "${DEEPSPEED_CONFIG}"
      --override "data.prepared_data_path=${EFFECTIVE_SFT_PREPARED_FILE}"
      --override "data.include_splits=${TRAIN_INCLUDE_SPLIT}"
      --override "data.materialized_messages_path=${SFT_MATERIALIZED_FILE}"
      --override "data.require_materialized_cache=true"
      --override "proposal.model_path=${PROPOSAL_MODEL_PATH}"
      --override "proposal.torch_dtype=${PROPOSAL_TORCH_DTYPE}"
      --override "proposal.device=${PROPOSAL_DEVICE}"
      --override "optimization.epochs=1"
      --override "optimization.gradient_accumulation_steps=${SFT_GRADIENT_ACCUMULATION_STEPS}"
      --override "distributed.nnodes=${NNODES}"
      --override "distributed.nproc_per_node=${NPROC_PER_NODE}"
      --override "output_dir=${EPOCH_OUT}"
      --override "logging.log_dir=${LOG_DIR}"
      --override "logging.rollout_eval_output_dir=${EVAL_DIR}/sft_epoch_${EPOCH_TAG}"
    )
    if (( epoch > 1 )); then
      PREV_TAG="$(printf '%03d' "$(( epoch - 1 ))")"
      PREV_EPOCH_OUT="${SFT_DIR}/epoch_${PREV_TAG}"
      require_valid_sft_epoch_output "${PREV_EPOCH_OUT}" "base model for SFT epoch ${epoch}"
      PREV_EPOCH_MODEL_PATH="$(resolve_sft_epoch_model_path "${PREV_EPOCH_OUT}")"
      sft_cmd+=(--override "base_model=${PREV_EPOCH_MODEL_PATH}")
    fi

    info "Launching DeepSpeed SFT epoch ${epoch} with prepared data ${EFFECTIVE_SFT_PREPARED_FILE} ..."
    "${sft_cmd[@]}"
    require_valid_sft_epoch_output "${EPOCH_OUT}" "SFT epoch ${epoch} output"
    ok "SFT epoch ${epoch} training complete → ${EPOCH_OUT}"
  fi

  # ---- Per-epoch evaluation ------------------------------------------------
  EPOCH_EVAL_DIR="${EVAL_DIR}/sft_epoch_${EPOCH_TAG}"
  EPOCH_METRICS="${EPOCH_EVAL_DIR}/metrics.json"
  EPOCH_EVAL_DONE_FILE="${EPOCH_EVAL_DIR}/rollout_eval_wrapper_summary.json"

  if [[ -f "${EPOCH_EVAL_DONE_FILE}" ]]; then
    skip "Eval wrapper summary for SFT epoch ${epoch} already exists at ${EPOCH_EVAL_DONE_FILE} — skipping."
  else
    banner "${GREEN}" "Stage 2.${epoch}: Evaluating SFT Epoch ${epoch}"

    require_valid_sft_epoch_output "${EPOCH_OUT}" "SFT epoch ${epoch} checkpoint for rollout eval"
    EPOCH_MODEL_PATH="$(resolve_sft_epoch_model_path "${EPOCH_OUT}")"
    info "Launching torchrun eval for SFT epoch ${epoch} ..."
    torchrun \
      --nnodes          "${NNODES}" \
      --nproc-per-node  "${NPROC_PER_NODE}" \
      --master_port     "${EVAL_MASTER_PORT}" \
      --module saver_v3.cli.run_sft_rollout_eval_vllm \
      --config "${ROLLOUT_CONFIG}" \
      --override "base_model=${EPOCH_MODEL_PATH}" \
      --override "io.data_path=${RUNTIME_EVAL_FILE}" \
      --override "io.data_root=${DATA_ROOT}" \
      --override "io.materialized_items_path=${RUNTIME_EVAL_ITEMS_FILE}" \
      --override "io.require_materialized_cache=true" \
      --override "io.include_splits=${EVAL_INCLUDE_SPLIT}" \
      --override "io.output_dir=${EPOCH_EVAL_DIR}" \
      --override "proposal.model_path=${PROPOSAL_MODEL_PATH}" \
      --override "proposal.torch_dtype=${PROPOSAL_TORCH_DTYPE}" \
      --override "proposal.device=${PROPOSAL_DEVICE}" \
      --override "evaluation.epoch_index=${epoch}"

    ok "Eval for SFT epoch ${epoch} complete → ${EPOCH_EVAL_DIR}"
  fi
  EPOCH_SUMMARY_FILE="$(resolve_eval_summary_file "${EPOCH_EVAL_DIR}")"
  print_metrics "${EPOCH_SUMMARY_FILE}" "SFT epoch ${epoch}"
  record_summary "SFT epoch ${epoch}" "${EPOCH_SUMMARY_FILE}"
done

# ---------------------------------------------------------------------------
# Stage 3: RL Training (initialised from last SFT checkpoint)
# ---------------------------------------------------------------------------
banner "${GREEN}" "Stage 3: RL Training"

LAST_SFT_EPOCH_DIR="${SFT_DIR}/epoch_$(printf '%03d' "${NUM_SFT_EPOCHS}")"

if [[ ! -d "${LAST_SFT_EPOCH_DIR}" ]]; then
  err "Last SFT checkpoint not found at ${LAST_SFT_EPOCH_DIR}. Cannot start RL training."
  exit 1
fi
require_valid_sft_epoch_output "${LAST_SFT_EPOCH_DIR}" "last SFT checkpoint before RL"
LAST_SFT_CKPT="$(resolve_sft_epoch_model_path "${LAST_SFT_EPOCH_DIR}")"
info "Using SFT checkpoint: ${LAST_SFT_CKPT}"

if is_complete_rl_output "${RL_DIR}"; then
  RL_CKPT="$(resolve_rl_latest_checkpoint_path "${RL_DIR}")"
  skip "RL output already complete at ${RL_DIR}; latest checkpoint=${RL_CKPT} — skipping RL training."
else
  if [[ -d "${RL_DIR}" && -n "$(ls -A "${RL_DIR}" 2>/dev/null)" ]]; then
    info "Removing stale/incomplete RL output at ${RL_DIR} before retraining."
    rm -rf "${RL_DIR}"
    mkdir -p "${RL_DIR}"
  fi
  info "Launching DeepSpeed RL training ..."
  export_rl_cuda_link_env
  info "Prebuilding DeepSpeed CPUAdam extension ..."
  prebuild_deepspeed_cpu_adam
  deepspeed \
    --num_nodes  "${NNODES}" \
    --num_gpus   "${NPROC_PER_NODE}" \
    --module saver_v3.cli.train_rl_ds \
    --config "${RL_CONFIG}" \
    --model-config "${MODEL_CONFIG}" \
    --attention-config "${ATTENTION_CONFIG}" \
    --deepspeed-config "${RL_DEEPSPEED_CONFIG}" \
    --override "data.train_manifest=${RUNTIME_TRAIN_FILE}" \
    --override "data.eval_manifest=${RUNTIME_EVAL_FILE}" \
    --override "data.data_root=${DATA_ROOT}" \
    --override "data.eval_data_root=${DATA_ROOT}" \
    --override "data.include_splits=${TRAIN_INCLUDE_SPLIT}" \
    --override "data.eval_include_splits=${EVAL_INCLUDE_SPLIT}" \
    --override "data.materialized_train_items_path=${RUNTIME_TRAIN_ITEMS_FILE}" \
    --override "data.materialized_eval_items_path=${RUNTIME_EVAL_ITEMS_FILE}" \
    --override "data.require_materialized_runtime_cache=true" \
    --override "proposal.model_path=${PROPOSAL_MODEL_PATH}" \
    --override "proposal.torch_dtype=${RL_PROPOSAL_TORCH_DTYPE}" \
    --override "proposal.device=${PROPOSAL_DEVICE}" \
    --override "proposal.eval_model_path=${PROPOSAL_MODEL_PATH}" \
    --override "proposal.eval_torch_dtype=${RL_PROPOSAL_TORCH_DTYPE}" \
    --override "proposal.eval_device=${PROPOSAL_DEVICE}" \
    --override "optimization.gradient_accumulation_steps=${RL_GRADIENT_ACCUMULATION_STEPS}" \
    --override "distributed.nnodes=${NNODES}" \
    --override "distributed.nproc_per_node=${NPROC_PER_NODE}" \
    --override "policy_init_from=${LAST_SFT_CKPT}" \
    --override "output_dir=${RL_DIR}" \
    --override "logging.log_dir=${LOG_DIR}"

  ok "RL training complete → ${RL_DIR}"
  require_valid_rl_output "${RL_DIR}" "RL output after training"
fi

# ---------------------------------------------------------------------------
# Stage 4: RL Evaluation
# ---------------------------------------------------------------------------
banner "${GREEN}" "Stage 4: RL Evaluation"

require_valid_rl_output "${RL_DIR}" "RL checkpoint before eval"
RL_CKPT="$(resolve_rl_latest_checkpoint_path "${RL_DIR}")"
info "Using RL checkpoint for eval: ${RL_CKPT}"

RL_EVAL_DIR="${EVAL_DIR}/rl_final"
RL_METRICS="${RL_EVAL_DIR}/metrics.json"
RL_EVAL_DONE_FILE="${RL_EVAL_DIR}/rollout_eval_wrapper_summary.json"

if [[ -f "${RL_EVAL_DONE_FILE}" ]]; then
  skip "RL eval wrapper summary already exists at ${RL_EVAL_DONE_FILE} — skipping."
else
  info "Launching torchrun RL eval ..."
  torchrun \
    --nnodes          "${NNODES}" \
    --nproc-per-node  "${NPROC_PER_NODE}" \
    --master_port     "${EVAL_MASTER_PORT}" \
    --module saver_v3.cli.run_sft_rollout_eval_vllm \
      --config "${ROLLOUT_CONFIG}" \
      --override "base_model=${RL_CKPT}" \
      --override "io.data_path=${RUNTIME_EVAL_FILE}" \
      --override "io.data_root=${DATA_ROOT}" \
      --override "io.materialized_items_path=${RUNTIME_EVAL_ITEMS_FILE}" \
      --override "io.require_materialized_cache=true" \
      --override "io.include_splits=${EVAL_INCLUDE_SPLIT}" \
      --override "io.output_dir=${RL_EVAL_DIR}" \
      --override "proposal.model_path=${PROPOSAL_MODEL_PATH}" \
      --override "proposal.torch_dtype=${PROPOSAL_TORCH_DTYPE}" \
      --override "proposal.device=${PROPOSAL_DEVICE}" \
      --override "evaluation.epoch_index=0"

  ok "RL eval complete → ${RL_EVAL_DIR}"
fi

RL_SUMMARY_FILE="$(resolve_eval_summary_file "${RL_EVAL_DIR}")"
print_metrics "${RL_SUMMARY_FILE}" "RL final"
record_summary "RL final" "${RL_SUMMARY_FILE}"

# ---------------------------------------------------------------------------
# Stage 5: Summary table
# ---------------------------------------------------------------------------
banner "${CYAN}" "Stage 5: Results Summary — ${EXP_NAME}"

printf "${BOLD}%-20s  %s${RESET}\n" "Stage" "Metrics"
printf "%s\n" "$(printf '─%.0s' {1..80})"

for row in "${SUMMARY_ROWS[@]}"; do
  label="${row%%|*}"
  kvs="${row#*|}"
  # Print label, then each key=value on its own continuation line
  printf "${BOLD}%-20s${RESET}  " "${label}"
  if [[ "${kvs}" == "N/A" ]]; then
    printf "N/A\n"
  else
    first=1
    IFS=',' read -ra pairs <<< "${kvs}"
    for pair in "${pairs[@]}"; do
      if (( first )); then
        printf "%s\n" "${pair}"
        first=0
      else
        printf "%-22s%s\n" "" "${pair}"
      fi
    done
  fi
done

printf "%s\n" "$(printf '─%.0s' {1..80})"
banner "${GREEN}" "Pipeline complete for experiment: ${EXP_NAME}"
info "All artifacts in: ${ARTIFACTS_DIR}"
info "Full log at:      ${LOG_FILE}"
