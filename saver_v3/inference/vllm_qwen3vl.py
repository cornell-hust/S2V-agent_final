from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

from saver_v3.common import ensure_fa3_training_ready


@dataclass
class LocalRankVllmConfig:
    model_path: str
    gpu_memory_utilization: float = 0.9
    dtype: str = "bfloat16"
    max_model_len: int | None = None
    limit_mm_per_prompt: Dict[str, int] | None = None
    seed: int | None = None


def _current_local_rank() -> int:
    try:
        return int(os.environ.get("LOCAL_RANK", "0") or "0")
    except Exception:
        return 0


def _pin_local_rank_cuda_visible_devices() -> int:
    local_rank = _current_local_rank()
    current = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if not current or "," in current:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    return local_rank


def build_local_rank_vllm_engine(config: LocalRankVllmConfig) -> Any:
    ensure_fa3_training_ready(require_gpu=True, require_module=False)
    local_rank = _pin_local_rank_cuda_visible_devices()
    try:
        from vllm import LLM
    except Exception as exc:
        raise ImportError("vLLM runtime requires the vllm package.") from exc
    limit_mm_per_prompt = dict(config.limit_mm_per_prompt or {"image": 16, "video": 1})
    return LLM(
        model=config.model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=float(config.gpu_memory_utilization),
        dtype=str(config.dtype),
        max_model_len=config.max_model_len,
        distributed_executor_backend="external_launcher",
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        seed=(local_rank if config.seed is None else int(config.seed)),
        limit_mm_per_prompt=limit_mm_per_prompt,
    )
