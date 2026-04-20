from __future__ import annotations

import copy
import json
import math
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from saver_v3.common.message_budget import (
    apply_message_budget,
    drop_oldest_history_turn,
    summarize_visual_budget,
)
from saver_v3.common.runtime import distributed_runtime_from_env, runtime_log
from saver_v3.model.model_loading import build_hf_model_init_kwargs, ensure_flash_attention_supported_dtype
from saver_v3.core.self_verification import build_policy_self_verification_payload


DEFAULT_MODEL_PATH = os.environ.get(
    "SAVER_QWEN_MODEL_PATH",
    "/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/qwen3-vl-8b-Instruct",
)
_TIMESTAMP_ONLY_RE = re.compile(r"^\s*\d+(?:\.\d+)?s\s*$")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_ANSWER_BLOCK_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL)
_VERIFY_COMPACT_KEYS = {
    "verification_decision",
    "recommended_action",
    "sufficiency_score",
    "necessity_score",
    "finalize_readiness_score",
    "counterfactual_faithfulness",
    "selected_window_ids",
    "selected_evidence_moment_ids",
}
_STRUCTURED_STOP_STRINGS = ("</tool_call>", "</answer>")


def _configure_qwen_processor(processor: Any) -> Any:
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


def load_auto_processor_with_compat(
    model_path: str | Path,
    *,
    trust_remote_code: Optional[bool] = None,
    **kwargs: Any,
) -> Any:
    try:
        from transformers import AutoProcessor
    except Exception as exc:
        raise ImportError("Loading the Qwen processor requires the `transformers` package.") from exc

    load_kwargs: Dict[str, Any] = dict(kwargs)
    if trust_remote_code is not None:
        load_kwargs["trust_remote_code"] = bool(trust_remote_code)
    try:
        return AutoProcessor.from_pretrained(
            str(model_path),
            fix_mistral_regex=True,
            **load_kwargs,
        )
    except TypeError as exc:
        if "fix_mistral_regex" not in str(exc):
            raise
        return AutoProcessor.from_pretrained(str(model_path), **load_kwargs)


def _build_generation_kwargs(
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    repetition_penalty: Optional[float],
    stopping_criteria: Any = None,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
    }
    if do_sample:
        if temperature is not None:
            kwargs["temperature"] = temperature
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
    if repetition_penalty is not None:
        kwargs["repetition_penalty"] = repetition_penalty
    if stopping_criteria is not None:
        kwargs["stopping_criteria"] = stopping_criteria
    return kwargs


def _build_structured_stopping_criteria(processor: Any) -> Any:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "encode"):
        return None
    stop_token_sequences: List[List[int]] = []
    for stop_string in _STRUCTURED_STOP_STRINGS:
        try:
            token_ids = tokenizer.encode(stop_string, add_special_tokens=False)
        except Exception:
            token_ids = []
        token_ids = [int(token_id) for token_id in list(token_ids or [])]
        if token_ids:
            stop_token_sequences.append(token_ids)
    if not stop_token_sequences:
        return None

    def _should_stop(input_ids) -> bool:
        if input_ids is None or getattr(input_ids, "ndim", 0) != 2:
            return False
        for row in input_ids:
            row_ids = [int(token_id) for token_id in row.tolist()]
            matched = False
            for stop_ids in stop_token_sequences:
                if len(row_ids) >= len(stop_ids) and row_ids[-len(stop_ids) :] == stop_ids:
                    matched = True
                    break
            if not matched:
                return False
        return True

    try:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _StructuredStopCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return _should_stop(input_ids)

        return StoppingCriteriaList([_StructuredStopCriteria()])
    except Exception:
        class _FallbackStructuredStopCriteria:
            def __call__(self, input_ids, scores=None, **kwargs):
                return _should_stop(input_ids)

        return [_FallbackStructuredStopCriteria()]


def _trim_to_first_structured_block(output_text: str) -> str:
    text = str(output_text or "").strip()
    if not text:
        return text
    think_match = _THINK_BLOCK_RE.search(text)
    block_matches = [match for match in (_TOOL_CALL_BLOCK_RE.search(text), _ANSWER_BLOCK_RE.search(text)) if match]
    if not block_matches:
        return text
    chosen = min(block_matches, key=lambda match: match.start())
    prefix = ""
    if think_match is not None and think_match.start() <= chosen.start():
        prefix = think_match.group(0).strip()
    block_text = chosen.group(0).strip()
    if prefix:
        return f"{prefix}{block_text}"
    return block_text


def _compact_verify_tool_call(output_text: str) -> str:
    trimmed = _trim_to_first_structured_block(output_text)
    tool_match = _TOOL_CALL_BLOCK_RE.search(trimmed)
    if tool_match is None:
        return trimmed
    try:
        function_payload = json.loads(
            tool_match.group(0)[len("<tool_call>") : -len("</tool_call>")].strip()
        )
    except Exception:
        return trimmed
    if not isinstance(function_payload, dict):
        return trimmed
    if str(function_payload.get("name") or "") != "verify_hypothesis":
        return trimmed
    arguments = function_payload.get("arguments")
    if not isinstance(arguments, dict):
        return trimmed
    if not any(key in arguments for key in _VERIFY_COMPACT_KEYS):
        return trimmed
    try:
        compact_arguments = build_policy_self_verification_payload(
            arguments,
            include_query=False,
            include_rationale=False,
        )
    except Exception:
        return trimmed
    compact_function_payload = {
        "name": "verify_hypothesis",
        "arguments": compact_arguments,
    }
    compact_block = (
        "<tool_call>"
        + json.dumps(compact_function_payload, ensure_ascii=False, separators=(",", ":"))
        + "</tool_call>"
    )
    think_match = _THINK_BLOCK_RE.search(trimmed)
    if think_match is not None and think_match.start() == 0:
        return f"{think_match.group(0).strip()}{compact_block}"
    return compact_block


def _to_pil_image(image: Any) -> Any:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.clamp(0, 255).to(torch.uint8)
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(3, 1, 1)
            array = tensor.permute(1, 2, 0).numpy()
            return Image.fromarray(array, mode="RGB")
    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 3 and array.shape[-1] in (1, 3):
            array = np.clip(array, 0, 255).astype(np.uint8)
            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
            return Image.fromarray(array, mode="RGB")
    return image


def _resize_image_for_budget(
    image: Any,
    *,
    max_image_side: int = 0,
    max_image_pixels: int = 0,
) -> Any:
    pil_image = _to_pil_image(image)
    if not hasattr(pil_image, "size"):
        return pil_image

    width, height = pil_image.size
    if width <= 0 or height <= 0:
        return pil_image

    scale = 1.0
    if int(max_image_side) > 0:
        current_max_side = max(width, height)
        if current_max_side > int(max_image_side):
            scale = min(scale, float(max_image_side) / float(current_max_side))
    if int(max_image_pixels) > 0:
        current_pixels = width * height
        if current_pixels > int(max_image_pixels):
            scale = min(scale, math.sqrt(float(max_image_pixels) / float(current_pixels)))

    if scale >= 0.999:
        return pil_image

    resized_width = max(28, int(round(width * scale)))
    resized_height = max(28, int(round(height * scale)))
    return pil_image.resize((resized_width, resized_height))


def _is_image_item(item: Any) -> bool:
    return isinstance(item, dict) and item.get("type") == "image" and ("image" in item or "image_ref" in item)


def _is_timestamp_text_item(item: Any) -> bool:
    if not isinstance(item, dict) or item.get("type") != "text":
        return False
    return bool(_TIMESTAMP_ONLY_RE.match(str(item.get("text") or "").strip()))


def _paired_multimodal_removal_indices(content: List[Dict[str, Any]], content_index: int) -> set[int]:
    removals = {int(content_index)}
    timestamp_index = int(content_index) - 1
    if 0 <= timestamp_index < len(content) and _is_timestamp_text_item(content[timestamp_index]):
        removals.add(timestamp_index)
    return removals


def _prune_stale_text_history(
    messages: List[Dict[str, Any]],
    *,
    keep_recent_text_messages: int = 0,
) -> List[Dict[str, Any]]:
    if int(keep_recent_text_messages) <= 0:
        return messages

    prepared = copy.deepcopy(messages)
    prefix_end = 0
    while prefix_end < len(prepared) and prepared[prefix_end].get("role") in {"system", "user"}:
        prefix_end += 1

    preserved_prefix = prepared[:prefix_end]
    history = prepared[prefix_end:]
    if len(history) <= int(keep_recent_text_messages):
        return preserved_prefix + history
    return preserved_prefix + history[-int(keep_recent_text_messages) :]


def _has_multimodal_content(messages: List[Dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            item_type = item.get("type")
            if item_type == "image" and ("image" in item or "image_ref" in item):
                return True
            if item_type == "video" and ("video" in item or "video_ref" in item):
                return True
    return False


def _drop_oldest_multimodal_item(messages: List[Dict[str, Any]]) -> bool:
    for message_index, message in enumerate(messages):
        content = list(message.get("content", []))
        for content_index, item in enumerate(content):
            item_type = item.get("type")
            if item_type not in {"image", "video"}:
                continue
            if item_type == "image" and "image" not in item and "image_ref" not in item:
                continue
            if item_type == "video" and "video" not in item and "video_ref" not in item:
                continue
            for removal_index in sorted(
                _paired_multimodal_removal_indices(content, content_index),
                reverse=True,
            ):
                del content[removal_index]
            if content:
                messages[message_index]["content"] = content
            elif message.get("role") not in {"system", "user"}:
                del messages[message_index]
            else:
                messages[message_index]["content"] = []
            return True
    return False


def _drop_oldest_history_message(messages: List[Dict[str, Any]]) -> bool:
    return drop_oldest_history_turn(messages)


def _count_model_input_tokens(encoded_inputs: Dict[str, Any]) -> int:
    input_ids = encoded_inputs["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        return int(input_ids.shape[-1])
    if input_ids and isinstance(input_ids[0], list):
        return int(len(input_ids[0]))
    return int(len(input_ids))


def _model_input_padded_width(encoded_inputs: Dict[str, Any]) -> int:
    input_ids = encoded_inputs["input_ids"] if isinstance(encoded_inputs, dict) else encoded_inputs.input_ids
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim == 1:
            return int(input_ids.shape[0])
        return int(input_ids.shape[-1])
    if isinstance(input_ids, list):
        if input_ids and isinstance(input_ids[0], list):
            return int(max(len(row) for row in input_ids))
        return int(len(input_ids))
    raise ValueError("Encoded inputs must expose tensor or list-based input_ids.")


def _prune_messages_to_max_total_images(
    messages: List[Dict[str, Any]],
    *,
    max_total_images: int = 0,
) -> List[Dict[str, Any]]:
    if int(max_total_images) <= 0:
        return messages

    image_positions: List[Tuple[int, int]] = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for content_index, item in enumerate(content):
            if _is_image_item(item):
                image_positions.append((message_index, content_index))

    overflow = len(image_positions) - int(max_total_images)
    if overflow <= 0:
        return messages

    prepared: List[Dict[str, Any]] = []
    for message in messages:
        copied_message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            copied_message["content"] = list(content)
        prepared.append(copied_message)

    for message_index, content_index in image_positions[:overflow]:
        content = prepared[message_index].get("content")
        if not isinstance(content, list):
            continue
        if 0 <= content_index < len(content):
            content[content_index] = None
        timestamp_index = content_index - 1
        if 0 <= timestamp_index < len(content) and _is_timestamp_text_item(content[timestamp_index]):
            content[timestamp_index] = None

    for message in prepared:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        message["content"] = [item for item in content if item is not None]
    return prepared


def _clone_prepared_messages_view(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prepared: List[Dict[str, Any]] = []
    for message in messages:
        copied_message = dict(message)
        content = message.get("content")
        if isinstance(content, list):
            copied_message["content"] = [
                dict(item) if isinstance(item, dict) else item
                for item in content
            ]
        prepared.append(copied_message)
    return prepared


def _signature_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return tuple(sorted((str(key), _signature_scalar(subvalue)) for key, subvalue in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_signature_scalar(item) for item in value)
    return ("id", id(value))


def _content_item_signature(item: Any) -> Any:
    if not isinstance(item, dict):
        return _signature_scalar(item)
    item_type = str(item.get("type") or "")
    if item_type == "text":
        return ("text", str(item.get("text") or ""))
    if item_type == "image":
        if "image_ref" in item:
            return ("image_ref", _signature_scalar(item.get("image_ref")))
        return ("image", id(item.get("image")))
    if item_type == "video":
        if "video_ref" in item:
            return ("video_ref", _signature_scalar(item.get("video_ref")))
        video = item.get("video")
        if isinstance(video, (list, tuple)):
            return ("video", tuple(id(frame) for frame in video))
        return ("video", id(video))
    return tuple(sorted((str(key), _signature_scalar(value)) for key, value in item.items()))


def _message_signature(message: Dict[str, Any]) -> Any:
    content = message.get("content")
    if isinstance(content, list):
        content_signature = tuple(_content_item_signature(item) for item in content)
    else:
        content_signature = _signature_scalar(content)
    return (
        str(message.get("role") or ""),
        str(message.get("name") or ""),
        content_signature,
    )


class QwenGenerationPolicy:
    """Single-turn Qwen generation policy for SAVER rollouts."""

    def __init__(
        self,
        *,
        model: Any = None,
        processor: Any,
        max_new_tokens: int = 512,
        max_total_images: int = 0,
        max_tool_message_frames: int = 0,
        max_total_video_frames: int = 0,
        max_seq_length: int = 0,
        keep_recent_tool_image_messages: int = 0,
        keep_recent_text_messages: int = 0,
        max_image_side: int = 0,
        max_image_pixels: int = 0,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        use_generation_cache: bool = False,
    ):
        self.model = model
        self.processor = processor
        self.max_new_tokens = int(max_new_tokens)
        self.max_total_images = int(max_total_images)
        self.max_tool_message_frames = int(max_tool_message_frames)
        self.max_total_video_frames = int(max_total_video_frames)
        self.max_seq_length = int(max_seq_length)
        self.keep_recent_tool_image_messages = int(keep_recent_tool_image_messages)
        self.keep_recent_text_messages = int(keep_recent_text_messages)
        self.max_image_side = int(max_image_side)
        self.max_image_pixels = int(max_image_pixels)
        self.do_sample = bool(do_sample)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.use_generation_cache = bool(use_generation_cache)
        self.structured_stopping_criteria = _build_structured_stopping_criteria(processor)
        self._prepared_messages_cache_source_id: Optional[int] = None
        self._prepared_messages_cache_len: int = 0
        self._prepared_messages_cache: List[Dict[str, Any]] = []
        self._prepared_messages_cache_signatures: List[Any] = []

    @classmethod
    def from_components(
        cls,
        *,
        model: Any,
        processor: Any,
        max_new_tokens: int = 512,
        max_total_images: int = 0,
        max_tool_message_frames: int = 0,
        max_total_video_frames: int = 0,
        max_seq_length: int = 0,
        keep_recent_tool_image_messages: int = 0,
        keep_recent_text_messages: int = 0,
        max_image_side: int = 0,
        max_image_pixels: int = 0,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        use_generation_cache: bool = False,
    ) -> "QwenGenerationPolicy":
        return cls(
            model=model,
            processor=processor,
            max_new_tokens=max_new_tokens,
            max_total_images=max_total_images,
            max_tool_message_frames=max_tool_message_frames,
            max_total_video_frames=max_total_video_frames,
            max_seq_length=max_seq_length,
            keep_recent_tool_image_messages=keep_recent_tool_image_messages,
            keep_recent_text_messages=keep_recent_text_messages,
            max_image_side=max_image_side,
            max_image_pixels=max_image_pixels,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            use_generation_cache=use_generation_cache,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        attn_implementation: Optional[str] = None,
        max_new_tokens: int = 512,
        max_total_images: int = 0,
        max_tool_message_frames: int = 0,
        max_total_video_frames: int = 0,
        max_seq_length: int = 0,
        keep_recent_tool_image_messages: int = 0,
        keep_recent_text_messages: int = 0,
        max_image_side: int = 0,
        max_image_pixels: int = 0,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        use_generation_cache: bool = False,
    ) -> "QwenGenerationPolicy":
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except Exception as exc:
            raise ImportError(
                "QwenGenerationPolicy requires a recent transformers build with Qwen3-VL support. "
                "Install it with `pip install git+https://github.com/huggingface/transformers accelerate`."
            ) from exc

        model_init_kwargs = build_hf_model_init_kwargs(
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
        )

        resolved_model_path = Path(model_path)
        adapter_config_path = resolved_model_path / "adapter_config.json"
        processor_path = resolve_generation_processor_path(resolved_model_path)
        if adapter_config_path.exists():
            try:
                from peft import PeftConfig, PeftModel
            except Exception as exc:
                raise ImportError("Loading LoRA adapter checkpoints requires `peft` to be installed.") from exc
            peft_config = PeftConfig.from_pretrained(str(resolved_model_path))
            base_model_path = str(peft_config.base_model_name_or_path)
            model = Qwen3VLForConditionalGeneration.from_pretrained(base_model_path, **model_init_kwargs)
            model = PeftModel.from_pretrained(model, str(resolved_model_path))
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(str(resolved_model_path), **model_init_kwargs)
        ensure_flash_attention_supported_dtype(
            model,
            attn_implementation=attn_implementation,
        )
        model.eval()
        processor = _configure_qwen_processor(load_auto_processor_with_compat(processor_path))
        return cls(
            model=model,
            processor=processor,
            max_new_tokens=max_new_tokens,
            max_total_images=max_total_images,
            max_tool_message_frames=max_tool_message_frames,
            max_total_video_frames=max_total_video_frames,
            max_seq_length=max_seq_length,
            keep_recent_tool_image_messages=keep_recent_tool_image_messages,
            keep_recent_text_messages=keep_recent_text_messages,
            max_image_side=max_image_side,
            max_image_pixels=max_image_pixels,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            use_generation_cache=use_generation_cache,
        )

    def prepare_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self._prepared_messages_cache_source_id is not None and self._prepared_messages_cache_source_id != id(messages):
            self._prepared_messages_cache = []
            self._prepared_messages_cache_signatures = []
            self._prepared_messages_cache_len = 0
        current_signatures = [_message_signature(message) for message in messages]
        cached_prefix = 0
        max_prefix = min(len(current_signatures), len(self._prepared_messages_cache_signatures))
        while cached_prefix < max_prefix:
            if current_signatures[cached_prefix] != self._prepared_messages_cache_signatures[cached_prefix]:
                break
            cached_prefix += 1

        if cached_prefix < len(self._prepared_messages_cache):
            self._prepared_messages_cache = self._prepared_messages_cache[:cached_prefix]
        if len(messages) > cached_prefix:
            self._prepared_messages_cache.extend(self._prepare_message_slice(messages[cached_prefix:]))

        self._prepared_messages_cache_source_id = id(messages)
        self._prepared_messages_cache_len = len(messages)
        self._prepared_messages_cache_signatures = current_signatures
        prepared_messages: List[Dict[str, Any]] = _clone_prepared_messages_view(self._prepared_messages_cache)
        before_budget = summarize_visual_budget(prepared_messages)
        budgeted_messages = apply_message_budget(
            prepared_messages,
            keep_recent_text_messages=self.keep_recent_text_messages,
            keep_recent_tool_image_messages=self.keep_recent_tool_image_messages,
            max_total_images=self.max_total_images,
            max_tool_message_frames=self.max_tool_message_frames,
            max_total_video_frames=self.max_total_video_frames,
            copy_messages=False,
        )
        after_budget = summarize_visual_budget(budgeted_messages)
        if (
            any(
                int(value) > 0
                for value in (
                    self.keep_recent_text_messages,
                    self.keep_recent_tool_image_messages,
                    self.max_total_images,
                    self.max_tool_message_frames,
                    self.max_total_video_frames,
                )
            )
            and before_budget != after_budget
        ):
            runtime_log(
                "rollout visual budget debug: "
                f"before={before_budget} after={after_budget} "
                f"keep_recent_text_messages={int(self.keep_recent_text_messages) or 'all'} "
                f"keep_recent_tool_image_messages={int(self.keep_recent_tool_image_messages) or 'all'} "
                f"max_total_images={int(self.max_total_images) or 'all'} "
                f"max_tool_message_frames={int(self.max_tool_message_frames) or 'all'} "
                f"max_total_video_frames={int(self.max_total_video_frames) or 'all'}",
                runtime=distributed_runtime_from_env(),
                main_process_only=True,
            )
        return budgeted_messages

    def _prepare_message_slice(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prepared = copy.deepcopy(messages)
        for message in prepared:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                item_type = item.get("type")
                if item_type == "image" and "image" in item:
                    item["image"] = _resize_image_for_budget(
                        item["image"],
                        max_image_side=self.max_image_side,
                        max_image_pixels=self.max_image_pixels,
                    )
                elif item_type == "video" and "video" in item:
                    item["video"] = self._prepare_video_payload(item["video"])
        return prepared

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        multimodal_cache: Dict[str, Any],
        state: Any,
        step_index: int,
    ) -> str:
        return _compact_verify_tool_call(self.generate_from_messages(messages))

    def generate_from_messages_batch(
        self,
        messages_batch: List[List[Dict[str, Any]]],
    ) -> List[str]:
        if self.model is None:
            raise RuntimeError("QwenGenerationPolicy requires a loaded HF model for local generation.")
        if not messages_batch:
            return []
        prepared_messages_batch = [self.prepare_messages(messages) for messages in messages_batch]
        inputs = self._build_inputs_batch(prepared_messages_batch)
        inputs = self._move_to_model_device(inputs)
        generation_kwargs = self._generation_kwargs()
        with torch.inference_mode():
            with self._temporary_generation_cache():
                output_ids = self.model.generate(**inputs, **generation_kwargs)
            generated_ids_trimmed = self._trim_generated_ids(inputs, output_ids)
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        del inputs
        del output_ids
        del generated_ids_trimmed
        return [_trim_to_first_structured_block(str(text)) for text in output_texts]

    def generate_from_messages(self, messages: List[Dict[str, Any]]) -> str:
        if self.model is None:
            raise RuntimeError("QwenGenerationPolicy requires a loaded HF model for local generation.")
        prepared_messages = self.prepare_messages(messages)
        inputs = self._build_inputs(prepared_messages)
        inputs = self._move_to_model_device(inputs)
        generation_kwargs = self._generation_kwargs()
        with torch.inference_mode():
            with self._temporary_generation_cache():
                output_ids = self.model.generate(**inputs, **generation_kwargs)
            generated_ids_trimmed = self._trim_generated_ids(inputs, output_ids)
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        del inputs
        del output_ids
        del generated_ids_trimmed
        return _trim_to_first_structured_block(output_text[0])

    def _trim_generated_ids(self, inputs: Any, output_ids: Any) -> List[Any]:
        prompt_width = _model_input_padded_width(inputs)
        return [out_ids[prompt_width:] for out_ids in output_ids]

    def _build_inputs_exact(self, prepared_messages: List[Dict[str, Any]]) -> Any:
        try:
            return self.processor.apply_chat_template(
                prepared_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            prompt_text = self.processor.apply_chat_template(
                prepared_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = self._extract_vision_inputs(prepared_messages)
            processor_kwargs: Dict[str, Any] = {
                "text": prompt_text,
                "padding": True,
                "return_tensors": "pt",
            }
            if image_inputs:
                processor_kwargs["images"] = image_inputs
            if video_inputs:
                processor_kwargs["videos"] = video_inputs
            return self.processor(**processor_kwargs)

    def _build_inputs_with_truncation(
        self,
        prepared_messages: List[Dict[str, Any]],
        *,
        max_length: int,
        truncation_side: str = "left",
    ) -> Any:
        prompt_text = self.processor.apply_chat_template(
            prepared_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = self._extract_vision_inputs(prepared_messages)
        processor_kwargs: Dict[str, Any] = {
            "text": prompt_text,
            "padding": True,
            "return_tensors": "pt",
            "truncation": True,
            "max_length": int(max_length),
        }
        if image_inputs:
            processor_kwargs["images"] = image_inputs
        if video_inputs:
            processor_kwargs["videos"] = video_inputs
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "truncation_side"):
            return self.processor(**processor_kwargs)
        original_side = tokenizer.truncation_side
        tokenizer.truncation_side = str(truncation_side or "left")
        try:
            return self.processor(**processor_kwargs)
        finally:
            tokenizer.truncation_side = original_side

    def _build_inputs_batch_exact(
        self,
        prepared_messages_batch: List[List[Dict[str, Any]]],
    ) -> Any:
        try:
            return self.processor.apply_chat_template(
                prepared_messages_batch,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
        except TypeError:
            prompt_text_batch = self.processor.apply_chat_template(
                prepared_messages_batch,
                tokenize=False,
                add_generation_prompt=True,
            )
            return self._build_inputs_batch_from_prompt_texts(prompt_text_batch, prepared_messages_batch)

    def _build_inputs_batch_from_prompt_texts(
        self,
        prompt_text_batch: Any,
        prepared_messages_batch: List[List[Dict[str, Any]]],
    ) -> Any:
        image_inputs, video_inputs, video_kwargs = self._extract_vision_inputs_batch(prepared_messages_batch)
        processor_kwargs: Dict[str, Any] = {
            "text": prompt_text_batch,
            "padding": True,
            "return_tensors": "pt",
        }
        if image_inputs:
            processor_kwargs["images"] = image_inputs
        if video_inputs:
            processor_kwargs["videos"] = video_inputs
        if video_kwargs:
            processor_kwargs.update(video_kwargs)
        return self.processor(**processor_kwargs)

    def _build_inputs_batch(self, prepared_messages_batch: List[List[Dict[str, Any]]]) -> Any:
        if self.max_seq_length <= 0:
            return self._build_inputs_batch_exact(prepared_messages_batch)

        max_length = int(self.max_seq_length)
        fitted_messages_batch: List[List[Dict[str, Any]]] = []
        prompt_text_batch: List[str] = []
        any_text_truncated = False
        for prepared_messages in prepared_messages_batch:
            fitted_messages, exact_fit = self._fit_prepared_messages_to_max_length(
                prepared_messages,
                max_length=max_length,
            )
            fitted_messages_batch.append(fitted_messages)
            if exact_fit:
                prompt_text_batch.append(
                    self.processor.apply_chat_template(
                        fitted_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                continue
            any_text_truncated = True
            prompt_text_batch.append(
                self._build_prompt_text_with_truncation(
                    fitted_messages,
                    max_length=max_length,
                    truncation_side="left",
                )
            )
        if not any_text_truncated:
            return self._build_inputs_batch_exact(fitted_messages_batch)
        return self._build_inputs_batch_from_prompt_texts(prompt_text_batch, fitted_messages_batch)

    def _fit_prepared_messages_to_max_length(
        self,
        prepared_messages: List[Dict[str, Any]],
        *,
        max_length: int,
    ) -> Tuple[List[Dict[str, Any]], bool]:
        fitted_messages = copy.deepcopy(prepared_messages)
        for _ in range(512):
            full_inputs = self._build_inputs_exact(fitted_messages)
            if _count_model_input_tokens(full_inputs) <= max_length:
                return fitted_messages, True

            if _drop_oldest_multimodal_item(fitted_messages):
                continue
            if _drop_oldest_history_message(fitted_messages):
                continue

            if _has_multimodal_content(fitted_messages):
                raise ValueError(
                    f"Unable to fit rollout prompt within max_seq_length={max_length}. "
                    "Increase the sequence budget or reduce retained multimodal context."
                )
            return fitted_messages, False

        raise RuntimeError("Exceeded pruning attempts while fitting a rollout prompt to the sequence budget.")

    def _build_prompt_text_with_truncation(
        self,
        prepared_messages: List[Dict[str, Any]],
        *,
        max_length: int,
        truncation_side: str = "left",
    ) -> str:
        prompt_text = self.processor.apply_chat_template(
            prepared_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None or not hasattr(tokenizer, "encode") or not hasattr(tokenizer, "decode"):
            raise ValueError("Unable to text-truncate rollout prompt because the processor tokenizer is unavailable.")

        token_ids = list(tokenizer.encode(prompt_text, add_special_tokens=False) or [])
        if len(token_ids) > int(max_length):
            if str(truncation_side or "left") == "left":
                token_ids = token_ids[-int(max_length) :]
            else:
                token_ids = token_ids[: int(max_length)]
        try:
            truncated_prompt = tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            truncated_prompt = tokenizer.decode(token_ids)

        for _ in range(4):
            roundtrip_ids = list(tokenizer.encode(truncated_prompt, add_special_tokens=False) or [])
            if len(roundtrip_ids) <= int(max_length):
                return truncated_prompt
            if str(truncation_side or "left") == "left":
                roundtrip_ids = roundtrip_ids[-int(max_length) :]
            else:
                roundtrip_ids = roundtrip_ids[: int(max_length)]
            try:
                truncated_prompt = tokenizer.decode(
                    roundtrip_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            except TypeError:
                truncated_prompt = tokenizer.decode(roundtrip_ids)
        return truncated_prompt

    def _build_inputs(self, prepared_messages: List[Dict[str, Any]]) -> Any:
        if self.max_seq_length <= 0:
            return self._build_inputs_exact(prepared_messages)

        max_length = int(self.max_seq_length)
        fitted_messages, exact_fit = self._fit_prepared_messages_to_max_length(
            prepared_messages,
            max_length=max_length,
        )
        if exact_fit:
            return self._build_inputs_exact(fitted_messages)
        return self._build_inputs_with_truncation(
            fitted_messages,
            max_length=max_length,
            truncation_side="left",
        )

    def _move_to_model_device(self, inputs: Any) -> Any:
        if not hasattr(inputs, "to"):
            return inputs
        device = getattr(self.model, "device", None)
        if device is None:
            return inputs
        return inputs.to(device)

    def _generation_kwargs(self) -> Dict[str, Any]:
        kwargs = _build_generation_kwargs(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            stopping_criteria=self.structured_stopping_criteria,
        )
        kwargs["use_cache"] = bool(self.use_generation_cache)
        return kwargs

    @contextmanager
    def _temporary_generation_cache(self):
        if not self.use_generation_cache:
            yield
            return
        tracked_values: List[Tuple[Any, str, Any]] = []
        for owner, attr_name in (
            (getattr(self.model, "config", None), "use_cache"),
            (getattr(self.model, "generation_config", None), "use_cache"),
            (getattr(getattr(self.model, "config", None), "text_config", None), "use_cache"),
            (getattr(getattr(self.model, "config", None), "language_config", None), "use_cache"),
        ):
            if owner is None or not hasattr(owner, attr_name):
                continue
            tracked_values.append((owner, attr_name, getattr(owner, attr_name)))
            try:
                setattr(owner, attr_name, True)
            except Exception:
                continue
        try:
            yield
        finally:
            for owner, attr_name, previous_value in reversed(tracked_values):
                try:
                    setattr(owner, attr_name, previous_value)
                except Exception:
                    continue

    def _extract_vision_inputs(
        self,
        prepared_messages: List[Dict[str, Any]],
    ) -> Tuple[List[Any], List[Any]]:
        image_inputs: List[Any] = []
        video_inputs: List[Any] = []
        for message in prepared_messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                item_type = item.get("type")
                if item_type == "image" and "image" in item:
                    image_inputs.append(item["image"])
                elif item_type == "video" and "video" in item:
                    video_inputs.append(item["video"])
        return image_inputs, video_inputs

    def _extract_vision_inputs_batch(
        self,
        prepared_messages_batch: List[List[Dict[str, Any]]],
    ) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs, video_kwargs = process_vision_info(
                prepared_messages_batch,
                return_video_kwargs=True,
            )
            return list(image_inputs or []), list(video_inputs or []), dict(video_kwargs or {})
        except Exception:
            image_inputs: List[Any] = []
            video_inputs: List[Any] = []
            for prepared_messages in prepared_messages_batch:
                sample_images, sample_videos = self._extract_vision_inputs(prepared_messages)
                image_inputs.extend(sample_images)
                video_inputs.extend(sample_videos)
            return image_inputs, video_inputs, {}

    def _prepare_video_payload(self, video: Any) -> Any:
        if isinstance(video, list):
            return [
                _resize_image_for_budget(
                    frame,
                    max_image_side=self.max_image_side,
                    max_image_pixels=self.max_image_pixels,
                )
                for frame in video
            ]
        if isinstance(video, tuple):
            return [
                _resize_image_for_budget(
                    frame,
                    max_image_side=self.max_image_side,
                    max_image_pixels=self.max_image_pixels,
                )
                for frame in video
            ]
        if isinstance(video, torch.Tensor) and video.ndim == 4:
            return [
                _resize_image_for_budget(
                    frame,
                    max_image_side=self.max_image_side,
                    max_image_pixels=self.max_image_pixels,
                )
                for frame in video
            ]
        return video


_GENERATION_PROCESSOR_FILENAMES = (
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
)


def resolve_generation_processor_path(model_path: str | Path) -> str:
    resolved_model_path = Path(model_path)
    adapter_config_path = resolved_model_path / "adapter_config.json"
    if not adapter_config_path.exists():
        return str(resolved_model_path)
    if any((resolved_model_path / filename).exists() for filename in _GENERATION_PROCESSOR_FILENAMES):
        return str(resolved_model_path)
    try:
        payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    except Exception:
        return str(resolved_model_path)
    base_model_path = str(payload.get("base_model_name_or_path") or "").strip()
    return base_model_path or str(resolved_model_path)


def load_generation_processor_for_checkpoint(model_path: str | Path) -> Any:
    processor_path = resolve_generation_processor_path(model_path)
    return _configure_qwen_processor(load_auto_processor_with_compat(processor_path))
