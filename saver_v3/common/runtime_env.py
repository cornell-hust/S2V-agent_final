"""Runtime environment helpers for v3 training and inference."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _env_flag(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    normalized = str(value).strip().lower()
    if not normalized:
        return bool(default)
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


@dataclass(frozen=True)
class DistributedRuntime:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def shard_list(self, values: list[Any]) -> list[Any]:
        if self.world_size <= 1:
            return list(values)
        return [value for index, value in enumerate(values) if index % self.world_size == self.rank]


@dataclass(frozen=True)
class RuntimeEnv:
    runtime: DistributedRuntime
    hf_home: str = ""
    hf_token: str = ""
    data_root: str = ""
    output_root: str = ""
    attention_backend_override: str = "auto"
    disable_fa3: bool = False
    disable_fa2: bool = False
    cuda_visible_devices: str = ""


def distributed_runtime_from_env(env: Optional[Mapping[str, str]] = None) -> DistributedRuntime:
    env = env or os.environ
    world_size = max(1, _safe_int(env.get("WORLD_SIZE", 1), 1))
    rank = _safe_int(env.get("RANK", 0), 0)
    local_rank = _safe_int(env.get("LOCAL_RANK", rank), rank)
    if rank < 0 or rank >= world_size:
        rank = 0
    if local_rank < 0:
        local_rank = 0
    return DistributedRuntime(rank=rank, world_size=world_size, local_rank=local_rank)


def resolve_runtime_env(env: Optional[Mapping[str, str]] = None) -> RuntimeEnv:
    env = env or os.environ
    return RuntimeEnv(
        runtime=distributed_runtime_from_env(env),
        hf_home=str(env.get("HF_HOME", "") or "").strip(),
        hf_token=str(env.get("HF_TOKEN", "") or "").strip(),
        data_root=str(env.get("SAVER_V3_DATA_ROOT", "") or "").strip(),
        output_root=str(env.get("SAVER_V3_OUTPUT_ROOT", "") or "").strip(),
        attention_backend_override=str(env.get("SAVER_V3_ATTN_BACKEND", "auto") or "auto").strip().lower(),
        disable_fa3=_env_flag(env.get("SAVER_V3_DISABLE_FA3", "0")),
        disable_fa2=_env_flag(env.get("SAVER_V3_DISABLE_FA2", "0")),
        cuda_visible_devices=str(env.get("CUDA_VISIBLE_DEVICES", "") or "").strip(),
    )


def runtime_log(message: str, *, runtime: DistributedRuntime | None = None, main_process_only: bool = False) -> None:
    runtime = runtime or distributed_runtime_from_env()
    if main_process_only and not runtime.is_main_process:
        return
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [rank {runtime.rank}/{runtime.world_size}] {message}", flush=True)
