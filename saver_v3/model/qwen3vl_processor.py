from __future__ import annotations

from pathlib import Path
from typing import Any

from saver_v3.model.qwen_policy import load_auto_processor_with_compat


def load_qwen3vl_processor(model_path: str | Path) -> Any:
    processor = load_auto_processor_with_compat(str(model_path), trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if getattr(processor, "pad_token", None) is None and getattr(processor, "eos_token", None) is not None:
        processor.pad_token = processor.eos_token
    if tokenizer is not None and getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer is not None:
        try:
            tokenizer.padding_side = "left"
        except Exception:
            pass
    try:
        processor.padding_side = "left"
    except Exception:
        pass
    return processor
