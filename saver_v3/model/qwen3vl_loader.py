from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from saver_v3.common.fa3_guard import ensure_fa3_training_ready
from saver_v3.model.trainability import assert_full_model_trainable


def _resolve_torch_dtype(value: str | None) -> Any:
    normalized = str(value or "bfloat16").strip().lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": "auto",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {value}")
    return mapping[normalized]


def load_qwen3vl_full_model(
    model_path: str | Path,
    *,
    torch_dtype: str = "bfloat16",
    gradient_checkpointing: bool = True,
    attn_implementation: str = "flash_attention_3",
) -> Any:
    if str(attn_implementation).strip() != "flash_attention_3":
        raise ValueError("idea2_v3 locks attn_implementation to flash_attention_3")
    ensure_fa3_training_ready(require_gpu=False)
    try:
        from transformers import Qwen3VLForConditionalGeneration
    except Exception as exc:
        raise ImportError("Qwen3-VL model loading requires a transformers build with Qwen3-VL support.") from exc
    resolved_dtype = _resolve_torch_dtype(torch_dtype)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=resolved_dtype,
        attn_implementation="flash_attention_3",
    )
    for parameter in model.parameters():
        parameter.requires_grad = True
    if bool(gradient_checkpointing):
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
    assert_full_model_trainable(model)
    return model
