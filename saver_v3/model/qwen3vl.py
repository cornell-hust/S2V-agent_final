"""Self-contained Qwen3-VL loader and processor helpers for v3 training."""

from __future__ import annotations

import importlib
from typing import Any, Callable, Mapping, Optional, Sequence

from saver_v3.common.fa3_guard import resolve_attention_backend


DEFAULT_QWEN3_VL_8B_INSTRUCT_MODEL = "/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/qwen3-vl-8b-Instruct"

_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float32": "float32",
    "float": "float32",
}


def _require_transformers() -> Any:
    try:
        return importlib.import_module("transformers")
    except Exception as exc:
        raise ImportError(
            "Qwen3-VL helpers require the `transformers` package with Qwen3-VL support."
        ) from exc


def _resolve_model_class(transformers_module: Any) -> Any:
    for attribute_name in (
        "Qwen3VLForConditionalGeneration",
        "AutoModelForVision2Seq",
        "AutoModelForImageTextToText",
    ):
        model_class = getattr(transformers_module, attribute_name, None)
        if model_class is not None:
            return model_class
    raise ImportError(
        "The installed transformers build does not expose a Qwen3-VL compatible model class."
    )


def _resolve_torch_dtype(torch_dtype: Any) -> Any:
    if torch_dtype is None:
        return "auto"
    if not isinstance(torch_dtype, str):
        return torch_dtype
    normalized = str(torch_dtype).strip().lower()
    if not normalized or normalized == "auto":
        return "auto"
    normalized = normalized.removeprefix("torch.")
    normalized = _DTYPE_ALIASES.get(normalized, normalized)
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:
        raise ImportError("Resolving explicit torch dtypes requires the `torch` package.") from exc
    if not hasattr(torch, normalized):
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    return getattr(torch, normalized)


def configure_qwen3vl_processor(processor: Any) -> Any:
    tokenizer = getattr(processor, "tokenizer", None)
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


def load_qwen3vl_processor(
    model_name_or_path: str = DEFAULT_QWEN3_VL_8B_INSTRUCT_MODEL,
    *,
    trust_remote_code: bool = True,
    **kwargs: Any,
) -> Any:
    transformers_module = _require_transformers()
    auto_processor = getattr(transformers_module, "AutoProcessor", None)
    if auto_processor is None:
        raise ImportError("The installed transformers build does not expose AutoProcessor.")
    processor = auto_processor.from_pretrained(
        str(model_name_or_path),
        trust_remote_code=bool(trust_remote_code),
        **kwargs,
    )
    return configure_qwen3vl_processor(processor)


def load_qwen3vl_model(
    model_name_or_path: str = DEFAULT_QWEN3_VL_8B_INSTRUCT_MODEL,
    *,
    torch_dtype: Any = "bfloat16",
    trust_remote_code: bool = True,
    device_map: Any = None,
    env: Optional[Mapping[str, str]] = None,
    cuda_device_capabilities: Optional[Sequence[tuple[int, int]]] = None,
    module_available: Optional[Callable[[str], bool]] = None,
    require_gpu: bool = True,
    require_module: bool = True,
    **kwargs: Any,
) -> Any:
    transformers_module = _require_transformers()
    model_class = _resolve_model_class(transformers_module)
    attention = resolve_attention_backend(
        env=env,
        cuda_device_capabilities=cuda_device_capabilities,
        module_available=module_available,
        require_gpu=require_gpu,
        require_module=require_module,
    )

    model_init_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(trust_remote_code),
        "attn_implementation": attention.attn_implementation,
    }
    resolved_dtype = _resolve_torch_dtype(torch_dtype)
    if resolved_dtype != "auto":
        model_init_kwargs["torch_dtype"] = resolved_dtype
    if device_map is not None:
        model_init_kwargs["device_map"] = device_map
    model_init_kwargs.update(kwargs)
    return model_class.from_pretrained(str(model_name_or_path), **model_init_kwargs)


def prepare_qwen3vl_for_full_training(model: Any, *, gradient_checkpointing: bool = True) -> Any:
    if hasattr(model, "train"):
        model.train()
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
    config = getattr(model, "config", None)
    if config is not None:
        try:
            config.use_cache = False
        except Exception:
            pass
    for parameter in list(getattr(model, "parameters", lambda: [])()):
        try:
            parameter.requires_grad_(True)
        except Exception:
            try:
                parameter.requires_grad = True
            except Exception:
                continue
    return model


def extract_vision_inputs(messages: Sequence[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    images: list[Any] = []
    videos: list[Any] = []
    for message in list(messages or []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image" and "image" in item:
                images.append(item["image"])
            elif item.get("type") == "video" and "video" in item:
                videos.append(item["video"])
    return images, videos


def build_qwen3vl_inputs(
    processor: Any,
    messages: Sequence[dict[str, Any]],
    *,
    add_generation_prompt: bool = True,
    return_tensors: str = "pt",
) -> Any:
    try:
        return processor.apply_chat_template(
            list(messages or []),
            tokenize=True,
            add_generation_prompt=bool(add_generation_prompt),
            return_dict=True,
            return_tensors=return_tensors,
        )
    except (TypeError, ValueError):
        prompt_text = processor.apply_chat_template(
            list(messages or []),
            tokenize=False,
            add_generation_prompt=bool(add_generation_prompt),
        )
        images, videos = extract_vision_inputs(messages)
        processor_kwargs: dict[str, Any] = {
            "text": prompt_text,
            "padding": True,
            "return_tensors": return_tensors,
        }
        if images:
            processor_kwargs["images"] = images
        if videos:
            processor_kwargs["videos"] = videos
        return processor(**processor_kwargs)
