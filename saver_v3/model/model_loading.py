from __future__ import annotations

from typing import Any, Dict, Optional

import torch


_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "float": "float32",
    "fp64": "float64",
    "float64": "float64",
    "double": "float64",
}


def resolve_model_dtype(torch_dtype: Any) -> Any:
    if isinstance(torch_dtype, str):
        normalized = str(torch_dtype).strip()
        if not normalized or normalized == "auto":
            return "auto"
        normalized = normalized.removeprefix("torch.").strip().lower()
        normalized = _DTYPE_ALIASES.get(normalized, normalized)
        if not hasattr(torch, normalized):
            raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
        return getattr(torch, normalized)
    if torch_dtype is None:
        return "auto"
    return torch_dtype


def resolve_inference_model_dtype(torch_dtype: Any) -> Any:
    resolved = resolve_model_dtype(torch_dtype)
    if resolved != "auto":
        return resolved
    try:
        if torch.cuda.is_available():
            try:
                if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                    return torch.bfloat16
            except Exception:
                pass
            return torch.float16
    except Exception:
        pass
    return "auto"


def build_hf_model_init_kwargs(
    *,
    torch_dtype: Any = "auto",
    attn_implementation: Optional[str] = None,
    device_map: Any = None,
    prefer_inference_dtype: bool = False,
) -> Dict[str, Any]:
    resolved_dtype = (
        resolve_inference_model_dtype(torch_dtype)
        if bool(prefer_inference_dtype)
        else resolve_model_dtype(torch_dtype)
    )
    model_init_kwargs: Dict[str, Any] = {
        "dtype": resolved_dtype,
    }
    if device_map is not None:
        model_init_kwargs["device_map"] = device_map
    if attn_implementation:
        model_init_kwargs["attn_implementation"] = attn_implementation
    return model_init_kwargs


def infer_model_floating_dtype(model: Any) -> Optional[torch.dtype]:
    for parameter in getattr(model, "parameters", lambda: [])():
        dtype = getattr(parameter, "dtype", None)
        if isinstance(dtype, torch.dtype) and torch.empty((), dtype=dtype).is_floating_point():
            return dtype
    for buffer in getattr(model, "buffers", lambda: [])():
        dtype = getattr(buffer, "dtype", None)
        if isinstance(dtype, torch.dtype) and torch.empty((), dtype=dtype).is_floating_point():
            return dtype
    model_dtype = getattr(model, "dtype", None)
    if isinstance(model_dtype, torch.dtype) and torch.empty((), dtype=model_dtype).is_floating_point():
        return model_dtype
    config = getattr(model, "config", None)
    config_dtype = getattr(config, "torch_dtype", None)
    if isinstance(config_dtype, torch.dtype) and torch.empty((), dtype=config_dtype).is_floating_point():
        return config_dtype
    return None


def ensure_flash_attention_supported_dtype(
    model: Any,
    *,
    attn_implementation: Optional[str],
) -> None:
    implementation = str(attn_implementation or "").strip().lower()
    if not implementation.startswith("flash_attention"):
        return
    model_dtype = infer_model_floating_dtype(model)
    if model_dtype == torch.float32:
        raise ValueError(
            "Flash Attention requires float16 or bfloat16 model weights, but the loaded model resolved to float32."
        )
