"""FA3-only attention backend guard for Qwen3-VL full-model training."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence


class AttentionBackendUnavailableError(RuntimeError):
    """Raised when the hard FA3 contract cannot be satisfied."""


@dataclass(frozen=True)
class AttentionBackendDecision:
    backend: str
    attn_implementation: str
    compute_capability_major: int
    forced_backend: str
    resolution_order: tuple[str, ...]
    reasons: tuple[str, ...]


def _env_flag(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _normalize_backend_name(value: object) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized in {"", "auto"}:
        return "auto"
    if normalized in {"fa3", "flash_attention_3", "flash-attention-3"}:
        return "fa3"
    return normalized


def _default_module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _detect_cuda_device_capabilities() -> list[tuple[int, int]]:
    try:
        import torch
    except Exception:
        return []
    try:
        if not torch.cuda.is_available():
            return []
        device_count = int(torch.cuda.device_count())
    except Exception:
        return []
    capabilities: list[tuple[int, int]] = []
    for device_index in range(max(0, device_count)):
        try:
            major, minor = torch.cuda.get_device_capability(device_index)
        except Exception:
            continue
        capabilities.append((int(major), int(minor)))
    return capabilities


def _fa3_module_available(module_available: Callable[[str], bool]) -> bool:
    return bool(module_available("flash_attn_interface") or module_available("flash_attn"))


def resolve_attention_backend(
    *,
    env: Optional[Mapping[str, str]] = None,
    cuda_device_capabilities: Optional[Sequence[tuple[int, int]]] = None,
    module_available: Optional[Callable[[str], bool]] = None,
    resolution_order: Optional[Sequence[str]] = None,
    require_gpu: bool = True,
    require_module: bool = True,
) -> AttentionBackendDecision:
    """Resolve the single supported backend.

    v3 intentionally fails closed: there is no FA2/SDPA fallback.  Tests and CLI
    import paths may call this with ``require_gpu=False`` or ``require_module=False``
    so config parsing can run on login nodes, but actual training launchers should
    keep both requirements enabled.
    """
    del resolution_order  # Kept for compatibility with older tests/callers.
    env = env or os.environ
    module_available = module_available or _default_module_available
    forced_backend = _normalize_backend_name(env.get("SAVER_V3_ATTN_BACKEND", "auto"))
    capabilities = list(cuda_device_capabilities) if cuda_device_capabilities is not None else _detect_cuda_device_capabilities()
    compute_capability_major = max((int(major) for major, _minor in capabilities), default=0)
    reasons: list[str] = ["v3 attention policy is FA3-only; fallback backends are disabled by design."]

    if forced_backend not in {"auto", "fa3"}:
        reasons.append(f"unsupported SAVER_V3_ATTN_BACKEND={forced_backend!r}")
        raise AttentionBackendUnavailableError(" ".join(reasons))
    if _env_flag(env.get("SAVER_V3_DISABLE_FA3", "0")):
        reasons.append("SAVER_V3_DISABLE_FA3 disables the only supported backend.")
        raise AttentionBackendUnavailableError(" ".join(reasons))

    if capabilities:
        unsupported = [capability for capability in capabilities if int(capability[0]) < 9]
        if unsupported:
            reasons.append(f"non-Hopper GPU capability detected: {unsupported}; FA3 requires compute capability >= 9.")
            raise AttentionBackendUnavailableError(" ".join(reasons))
        reasons.append(f"all visible CUDA devices satisfy Hopper requirement: {capabilities}.")
    elif require_gpu:
        reasons.append("no visible CUDA GPU; FA3 training requires Hopper GPUs.")
        raise AttentionBackendUnavailableError(" ".join(reasons))
    else:
        reasons.append("no CUDA GPU was visible during this non-strict check.")

    if _fa3_module_available(module_available):
        reasons.append("FA3-capable flash attention Python bindings are importable.")
    elif require_module:
        reasons.append("neither flash_attn_interface nor flash_attn is importable.")
        raise AttentionBackendUnavailableError(" ".join(reasons))
    else:
        reasons.append("FA3 module import was skipped or unavailable during this non-strict check.")

    return AttentionBackendDecision(
        backend="fa3",
        attn_implementation="flash_attention_3",
        compute_capability_major=compute_capability_major,
        forced_backend=forced_backend,
        resolution_order=("fa3",),
        reasons=tuple(reasons),
    )


def ensure_fa3_training_ready(
    *,
    require_gpu: bool = True,
    require_module: bool = True,
    env: Optional[Mapping[str, str]] = None,
    cuda_device_capabilities: Optional[Sequence[tuple[int, int]]] = None,
    module_available: Optional[Callable[[str], bool]] = None,
) -> AttentionBackendDecision:
    return resolve_attention_backend(
        env=env,
        cuda_device_capabilities=cuda_device_capabilities,
        module_available=module_available,
        require_gpu=require_gpu,
        require_module=require_module,
    )
