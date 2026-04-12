from __future__ import annotations

import copy
import gc
import hashlib
import inspect
import json
import math
import random
import re
import sys
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from split_utils import parse_include_splits

from convert_to_saver_agent import build_finalize_case_payload
from saver_v3.core.adapter import TimeSearchRolloutAdapter
from saver_v3.data.dataset import SaverRecordItemBuilder
from saver_v3.core.environment import SaverEnvironmentState
from saver_v3.common.experiment_logging import append_jsonl, write_json
from saver_v3.metrics.evaluation import RolloutEvaluationConfig, run_rollout_evaluation
from saver_v3.common.message_budget import apply_message_budget, drop_oldest_history_turn
from saver_v3.model.model_loading import build_hf_model_init_kwargs, ensure_flash_attention_supported_dtype
from saver_v3.data.prepared_metadata import PREPARED_SFT_FORMAT
from saver_v3.core.proposal import SiglipFeatureEncoder
from saver_v3.model.qwen_policy import _resize_image_for_budget, _to_pil_image
from saver_v3.common.runtime import (
    distributed_barrier,
    distributed_runtime_from_env,
    log_timestamp,
    resolve_inference_device_map,
    runtime_log,
    should_log_progress,
)
from saver_v3.core.tool_registry import execute_tool_call
from saver_v3.data.training_data import (
    _apply_oracle_verifier_feedback,
    _assistant_tool_response,
    _merge_verify_arguments_with_oracle_feedback,
)


DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

SFT_TENSOR_CACHE_SCHEMA_VERSION = "saver_agent.sft_tensor_cache.v3"
STRICT_SFT_PRETOKENIZED_SCHEMA_VERSION = 1
SFT_EPOCH_RESUME_DIRNAME = "epoch_resume"
_TIMESTAMP_ONLY_RE = re.compile(r"^\s*\d+(?:\.\d+)?s\s*$")


@dataclass
class BatchBuildResult:
    batch: Optional[Dict[str, Any]]
    cached_plan: Optional[Dict[str, Any]]
    completion_token_count: int
    drop_reason: Optional[str]
    budgeting_attempted: bool
    is_episode_feature: bool = False


@dataclass
class BudgetingStats:
    num_batch_builder_attempts_after_budgeting: int = 0
    num_zero_response_batches: int = 0
    num_dropped_zero_response_episodes: int = 0
    num_invalid_completion_specs: int = 0

    def record(self, result: BatchBuildResult) -> None:
        if bool(result.budgeting_attempted):
            self.num_batch_builder_attempts_after_budgeting += 1
        drop_reason = str(result.drop_reason or "")
        if drop_reason == "truncated_completion_after_budgeting":
            self.num_invalid_completion_specs += 1
            return
        if drop_reason == "zero_response_after_budgeting":
            self.num_zero_response_batches += 1
            if bool(result.is_episode_feature):
                self.num_dropped_zero_response_episodes += 1

    def merge(self, other: "BudgetingStats") -> None:
        self.num_batch_builder_attempts_after_budgeting += int(other.num_batch_builder_attempts_after_budgeting)
        self.num_zero_response_batches += int(other.num_zero_response_batches)
        self.num_dropped_zero_response_episodes += int(other.num_dropped_zero_response_episodes)
        self.num_invalid_completion_specs += int(other.num_invalid_completion_specs)

    def as_dict(self) -> Dict[str, Any]:
        attempts = int(self.num_batch_builder_attempts_after_budgeting)
        zero_batches = int(self.num_zero_response_batches)
        invalid_specs = int(self.num_invalid_completion_specs)
        dropped_total = zero_batches + invalid_specs
        return {
            "num_zero_response_batches": zero_batches,
            "num_dropped_zero_response_episodes": int(self.num_dropped_zero_response_episodes),
            "num_invalid_completion_specs": invalid_specs,
            "drop_rate_after_budgeting": (
                float(dropped_total) / float(attempts) if attempts > 0 else 0.0
            ),
        }


def _format_budgeting_stats(prefix: str, stats: BudgetingStats) -> str:
    metrics = stats.as_dict()
    return (
        f"{prefix}: "
        f"num_zero_response_batches={int(metrics['num_zero_response_batches'])} "
        f"num_dropped_zero_response_episodes={int(metrics['num_dropped_zero_response_episodes'])} "
        f"num_invalid_completion_specs={int(metrics['num_invalid_completion_specs'])} "
        f"drop_rate_after_budgeting={float(metrics['drop_rate_after_budgeting']):.6f}"
    )


def _frame_cache_path(video_path: Path) -> Path:
    return Path(str(video_path) + ".frame_cache")


def _feature_cache_path(video_path: Path) -> Path:
    return Path(str(video_path) + ".feature_cache")


def _print_cache_warning(message: str) -> None:
    print(f"[cache-warning] {message}", flush=True)


def _iter_image_ref_video_paths(messages: Sequence[Dict[str, Any]]) -> Iterable[Path]:
    for message in messages:
        for item in message.get("content", []):
            if item.get("type") != "image":
                continue
            image_ref = item.get("image_ref") or {}
            video_path = str(image_ref.get("video_path") or "")
            if video_path:
                yield Path(video_path)


def _is_compact_trace_sft_row(example: Dict[str, Any]) -> bool:
    return (
        isinstance(example, dict)
        and str(example.get("prepared_format") or "").strip() == str(PREPARED_SFT_FORMAT)
        and isinstance(example.get("oracle_trajectory"), list)
        and str(example.get("video_path") or "").strip() != ""
    )


def _iter_prepared_video_paths(examples: Sequence[Dict[str, Any]]) -> Iterable[Path]:
    seen_video_paths: set[str] = set()
    for example in examples:
        if _is_compact_trace_sft_row(example):
            video_path = Path(str(example.get("video_path") or ""))
            key = str(video_path)
            if key and key not in seen_video_paths:
                seen_video_paths.add(key)
                yield video_path
            continue
        for video_path in _iter_image_ref_video_paths(example.get("messages", [])):
            key = str(video_path)
            if key in seen_video_paths:
                continue
            seen_video_paths.add(key)
            yield video_path


def summarize_example_frame_cache_status(
    examples: Sequence[Dict[str, Any]],
    *,
    max_examples: int = 5,
) -> Dict[str, Any]:
    referenced_video_paths = list(_iter_prepared_video_paths(examples))

    num_cached_videos = 0
    num_missing_frame_cache = 0
    num_missing_video_files = 0
    missing_examples: List[Dict[str, str]] = []
    for video_path in referenced_video_paths:
        cache_path = _frame_cache_path(video_path)
        if cache_path.exists():
            num_cached_videos += 1
        else:
            num_missing_frame_cache += 1
            if len(missing_examples) < max(0, int(max_examples)):
                missing_examples.append(
                    {
                        "video_path": str(video_path),
                        "cache_path": str(cache_path),
                    }
                )
        if not video_path.exists():
            num_missing_video_files += 1

    return {
        "num_examples": len(examples),
        "num_referenced_videos": len(referenced_video_paths),
        "num_cached_videos": num_cached_videos,
        "num_missing_frame_cache": num_missing_frame_cache,
        "num_missing_video_files": num_missing_video_files,
        "missing_examples": missing_examples,
    }


def format_example_frame_cache_status(
    summary: Dict[str, Any],
    *,
    prefix: str = "prepared frame cache",
) -> str:
    message = (
        f"{prefix}: cached={int(summary.get('num_cached_videos', 0))}/"
        f"{int(summary.get('num_referenced_videos', 0))} "
        f"missing_frame_cache={int(summary.get('num_missing_frame_cache', 0))} "
        f"missing_video_files={int(summary.get('num_missing_video_files', 0))}"
    )
    missing_examples = list(summary.get("missing_examples") or [])
    if missing_examples:
        preview = "; ".join(
            f"cache_path={item.get('cache_path') or ''} video_path={item.get('video_path') or ''}"
            for item in missing_examples
        )
        message += f" missing_examples=[{preview}]"
    return message


def summarize_example_feature_cache_status(
    examples: Sequence[Dict[str, Any]],
    *,
    max_examples: int = 5,
) -> Dict[str, Any]:
    referenced_video_paths = list(_iter_prepared_video_paths(examples))

    num_cached_videos = 0
    num_missing_feature_cache = 0
    missing_examples: List[Dict[str, str]] = []
    for video_path in referenced_video_paths:
        cache_path = _feature_cache_path(video_path)
        if cache_path.exists():
            num_cached_videos += 1
        else:
            num_missing_feature_cache += 1
            if len(missing_examples) < max(0, int(max_examples)):
                missing_examples.append(
                    {
                        "video_path": str(video_path),
                        "cache_path": str(cache_path),
                    }
                )
    return {
        "num_examples": len(examples),
        "num_referenced_videos": len(referenced_video_paths),
        "num_cached_videos": num_cached_videos,
        "num_missing_feature_cache": num_missing_feature_cache,
        "missing_examples": missing_examples,
    }


def format_example_feature_cache_status(
    summary: Dict[str, Any],
    *,
    prefix: str = "prepared feature cache",
) -> str:
    message = (
        f"{prefix}: cached={int(summary.get('num_cached_videos', 0))}/"
        f"{int(summary.get('num_referenced_videos', 0))} "
        f"missing_feature_cache={int(summary.get('num_missing_feature_cache', 0))}"
    )
    missing_examples = list(summary.get("missing_examples") or [])
    if missing_examples:
        preview = "; ".join(
            f"cache_path={item.get('cache_path') or ''} video_path={item.get('video_path') or ''}"
            for item in missing_examples
        )
        message += f" missing_examples=[{preview}]"
    return message

def _hash_json_payload(payload: Any) -> str:
    encoded = json.dumps(
        _strip_private_fields_for_cache_key(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None or not hasattr(obj, "to_dict"):
        return {}
    try:
        payload = obj.to_dict()
    except Exception:
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def build_processor_signature_payload(processor: Any) -> Dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", None)
    image_processor = getattr(processor, "image_processor", None)

    tokenizer_payload: Dict[str, Any] = {
        "class_name": type(tokenizer).__name__ if tokenizer is not None else "",
        "init_kwargs": _safe_to_dict(tokenizer) or _strip_private_fields_for_cache_key(
            getattr(tokenizer, "init_kwargs", {}) or {}
        ),
        "special_tokens_map": _strip_private_fields_for_cache_key(
            getattr(tokenizer, "special_tokens_map", {}) or {}
        ),
        "chat_template": str(
            getattr(tokenizer, "chat_template", None)
            or getattr(processor, "chat_template", None)
            or ""
        ),
    }
    if tokenizer is not None:
        try:
            added_vocab = tokenizer.get_added_vocab()
        except Exception:
            added_vocab = {}
        tokenizer_payload["added_vocab"] = _strip_private_fields_for_cache_key(added_vocab)
        try:
            vocab = tokenizer.get_vocab()
        except Exception:
            vocab = {}
        tokenizer_payload["vocab_size"] = int(len(vocab) or getattr(tokenizer, "vocab_size", 0) or 0)
        tokenizer_payload["vocab_digest"] = _hash_json_payload(vocab) if vocab else ""

    image_processor_payload = {
        "class_name": type(image_processor).__name__ if image_processor is not None else "",
        "config": _safe_to_dict(image_processor),
    }
    processor_payload = {
        "class_name": type(processor).__name__,
        "config": _safe_to_dict(processor),
        "tokenizer": tokenizer_payload,
        "image_processor": image_processor_payload,
    }
    return _strip_private_fields_for_cache_key(processor_payload)


def build_processor_signature(processor: Any) -> str:
    return _hash_json_payload(build_processor_signature_payload(processor))


def load_processor_signature_from_model_path(model_path: str | Path) -> str:
    try:
        from transformers import AutoProcessor
    except Exception as exc:
        raise ImportError("Computing processor signatures requires the `transformers` package.") from exc
    processor = AutoProcessor.from_pretrained(str(model_path))
    return build_processor_signature(processor)


def build_processor_signature_summary(processor: Any) -> Dict[str, Any]:
    payload = build_processor_signature_payload(processor)
    tokenizer_payload = dict(payload.get("tokenizer") or {})
    image_processor_payload = dict(payload.get("image_processor") or {})
    return {
        "processor_class": str(payload.get("class_name") or ""),
        "tokenizer_class": str(tokenizer_payload.get("class_name") or ""),
        "image_processor_class": str(image_processor_payload.get("class_name") or ""),
        "vocab_size": int(tokenizer_payload.get("vocab_size") or 0),
        "signature": build_processor_signature(processor),
    }


def _strip_private_fields_for_cache_key(payload: Any) -> Any:
    if payload is None or isinstance(payload, (str, int, float, bool)):
        return payload
    if isinstance(payload, dict):
        return {
            str(key): _strip_private_fields_for_cache_key(value)
            for key, value in payload.items()
            if not str(key).startswith("_")
        }
    if isinstance(payload, list):
        return [_strip_private_fields_for_cache_key(value) for value in payload]
    if isinstance(payload, tuple):
        return [_strip_private_fields_for_cache_key(value) for value in payload]
    if isinstance(payload, (set, frozenset)):
        normalized_items = [_strip_private_fields_for_cache_key(value) for value in payload]
        return sorted(
            normalized_items,
            key=lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, bytes):
        try:
            return payload.decode("utf-8")
        except Exception:
            return {"__type__": "bytes", "hex": payload.hex()}
    if isinstance(payload, torch.Tensor):
        return {
            "__type__": "torch.Tensor",
            "shape": list(payload.shape),
            "dtype": str(payload.dtype),
        }
    if hasattr(payload, "item") and callable(getattr(payload, "item")):
        try:
            return _strip_private_fields_for_cache_key(payload.item())
        except Exception:
            pass
    if hasattr(payload, "to_dict") and callable(getattr(payload, "to_dict")):
        try:
            return _strip_private_fields_for_cache_key(payload.to_dict())
        except Exception:
            pass
    if hasattr(payload, "__getstate__") and callable(getattr(payload, "__getstate__")):
        try:
            state = payload.__getstate__()
            if state is not None:
                return {
                    "__type__": type(payload).__name__,
                    "state": _strip_private_fields_for_cache_key(state),
                }
        except Exception:
            pass
    if hasattr(payload, "__dict__"):
        try:
            return {
                "__type__": type(payload).__name__,
                "attrs": _strip_private_fields_for_cache_key(vars(payload)),
            }
        except Exception:
            pass
    if hasattr(payload, "size") and hasattr(payload, "mode"):
        try:
            width, height = payload.size
            return {
                "__type__": type(payload).__name__,
                "size": [int(width), int(height)],
                "mode": str(payload.mode),
            }
        except Exception:
            return {"__type__": type(payload).__name__}
    return {"__type__": type(payload).__name__, "repr": repr(payload)}


def build_sft_tensor_cache_key(example: Dict[str, Any]) -> str:
    normalized_payload = _strip_private_fields_for_cache_key(example)
    encoded = json.dumps(
        normalized_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_sft_tensor_cache_config(
    *,
    processor_signature: str,
    max_image_side: int = 0,
    max_image_pixels: int = 0,
    keep_recent_tool_image_messages: int = 0,
    max_total_images: int = 0,
    max_seq_length: int = 0,
    keep_recent_text_messages: int = 0,
) -> Dict[str, Any]:
    return {
        "processor_signature": str(processor_signature or ""),
        "completion_schema_version": 2,
        "max_image_side": int(max_image_side),
        "max_image_pixels": int(max_image_pixels),
        "keep_recent_tool_image_messages": int(keep_recent_tool_image_messages),
        "max_total_images": int(max_total_images),
        "max_seq_length": int(max_seq_length),
        "keep_recent_text_messages": int(keep_recent_text_messages),
    }


def build_sft_tensor_cache_metadata(
    *,
    model_path: str | Path,
    processor_signature: str,
    processor_signature_summary: Optional[Dict[str, Any]] = None,
    max_image_side: int = 0,
    max_image_pixels: int = 0,
    keep_recent_tool_image_messages: int = 0,
    max_total_images: int = 0,
    max_seq_length: int = 0,
    keep_recent_text_messages: int = 0,
    prepared_data_path: str | Path = "",
    num_examples: int = 0,
) -> Dict[str, Any]:
    return {
        "schema_version": SFT_TENSOR_CACHE_SCHEMA_VERSION,
        "cache_config": normalize_sft_tensor_cache_config(
            processor_signature=processor_signature,
            max_image_side=max_image_side,
            max_image_pixels=max_image_pixels,
            keep_recent_tool_image_messages=keep_recent_tool_image_messages,
            max_total_images=max_total_images,
            max_seq_length=max_seq_length,
            keep_recent_text_messages=keep_recent_text_messages,
        ),
        "model_path": str(model_path),
        "processor_signature_summary": dict(processor_signature_summary or {}),
        "prepared_data_path": str(prepared_data_path) if prepared_data_path else "",
        "num_examples": int(num_examples),
    }


def resolve_sft_tensor_cache_config_from_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    actual_config = dict(metadata.get("cache_config") or {})
    if not str(actual_config.get("processor_signature") or "").strip():
        raise ValueError("tensor cache metadata is missing cache_config.processor_signature")
    return actual_config


def sft_tensor_cache_entry_path(cache_dir: str | Path, cache_key: str) -> Path:
    normalized_key = str(cache_key)
    prefix = normalized_key[:2] if len(normalized_key) >= 2 else "xx"
    return Path(cache_dir) / "entries" / prefix / f"{normalized_key}.pt"

def _unwrap_model(model: Any) -> Any:
    return getattr(model, "module", model)


def _resize_image_for_training(
    image: Any,
    *,
    max_image_side: int = 0,
    max_image_pixels: int = 0,
) -> Any:
    return _resize_image_for_budget(
        image,
        max_image_side=max_image_side,
        max_image_pixels=max_image_pixels,
    )


def _prune_stale_tool_images(
    messages: List[Dict[str, Any]],
    *,
    keep_recent_tool_image_messages: int = 0,
) -> List[Dict[str, Any]]:
    if int(keep_recent_tool_image_messages) <= 0:
        return copy.deepcopy(messages)

    prepared = copy.deepcopy(messages)
    image_tool_message_indices = [
        message_index
        for message_index, message in enumerate(prepared)
        if message.get("role") == "tool"
        and any(item.get("type") == "image" and "image" in item for item in message.get("content", []))
    ]
    keep_indices = set(image_tool_message_indices[-int(keep_recent_tool_image_messages) :])

    for message_index in image_tool_message_indices:
        if message_index in keep_indices:
            continue
        content = prepared[message_index].get("content", [])
        removals: set[int] = set()
        for content_index, item in enumerate(content):
            item_type = item.get("type")
            if item_type not in {"image", "video"}:
                continue
            removals.update(_paired_multimodal_removal_indices(content, content_index))
        prepared[message_index]["content"] = [
            item for idx, item in enumerate(content) if idx not in removals
        ]
    return prepared


def _prune_stale_text_history(
    messages: List[Dict[str, Any]],
    *,
    keep_recent_text_messages: int = 0,
) -> List[Dict[str, Any]]:
    if int(keep_recent_text_messages) <= 0:
        return copy.deepcopy(messages)

    prepared = copy.deepcopy(messages)
    prefix_end = 0
    while prefix_end < len(prepared) and prepared[prefix_end].get("role") in {"system", "user"}:
        prefix_end += 1

    preserved_prefix = prepared[:prefix_end]
    history = prepared[prefix_end:]
    if len(history) <= int(keep_recent_text_messages):
        return preserved_prefix + history
    return preserved_prefix + history[-int(keep_recent_text_messages) :]


def _cap_total_images(
    messages: List[Dict[str, Any]],
    *,
    max_total_images: int = 0,
) -> List[Dict[str, Any]]:
    if int(max_total_images) <= 0:
        return messages

    image_positions: List[Tuple[int, int]] = []
    for message_index, message in enumerate(messages):
        for content_index, item in enumerate(message.get("content", [])):
            if item.get("type") == "image" and ("image" in item or "image_ref" in item):
                image_positions.append((message_index, content_index))

    overflow = len(image_positions) - int(max_total_images)
    if overflow <= 0:
        return messages

    image_positions.sort()
    removals_by_message: Dict[int, set[int]] = {}
    for message_index, content_index in image_positions[:overflow]:
        content = list(messages[message_index].get("content", []))
        removals_by_message.setdefault(message_index, set()).update(
            _paired_multimodal_removal_indices(content, content_index)
        )

    for message_index, removals in removals_by_message.items():
        content = list(messages[message_index].get("content", []))
        messages[message_index]["content"] = [
            item for idx, item in enumerate(content) if idx not in removals
        ]
    return messages


def _prepare_messages(
    messages: List[Dict[str, Any]],
    *,
    max_image_side: int = 0,
    max_image_pixels: int = 0,
    keep_recent_tool_image_messages: int = 0,
    max_total_images: int = 0,
    keep_recent_text_messages: int = 0,
) -> List[Dict[str, Any]]:
    prepared = apply_message_budget(
        messages,
        keep_recent_text_messages=keep_recent_text_messages,
        keep_recent_tool_image_messages=keep_recent_tool_image_messages,
        max_total_images=max_total_images,
    )
    for message in prepared:
        for item in message.get("content", []):
            if item.get("type") == "image" and "image" in item:
                item["image"] = _resize_image_for_training(
                    item["image"],
                    max_image_side=max_image_side,
                    max_image_pixels=max_image_pixels,
                )
            elif item.get("type") == "video" and "video" in item:
                video = item["video"]
                if isinstance(video, (list, tuple)):
                    item["video"] = [
                        _resize_image_for_training(
                            frame,
                            max_image_side=max_image_side,
                            max_image_pixels=max_image_pixels,
                        )
                        for frame in video
                    ]
                elif isinstance(video, torch.Tensor) and video.ndim == 4:
                    item["video"] = [
                        _resize_image_for_training(
                            frame,
                            max_image_side=max_image_side,
                            max_image_pixels=max_image_pixels,
                        )
                        for frame in video
                    ]
    return prepared


def _extract_vision_inputs(messages: List[Dict[str, Any]]) -> Tuple[List[Any], List[Any]]:
    image_inputs: List[Any] = []
    video_inputs: List[Any] = []
    for message in messages:
        for item in message.get("content", []):
            if item.get("type") == "image" and "image" in item:
                image_inputs.append(item["image"])
            elif item.get("type") == "video" and "video" in item:
                video_inputs.append(item["video"])
    return image_inputs, video_inputs


def _build_chat_text(processor: Any, messages: List[Dict[str, Any]], *, add_generation_prompt: bool) -> str:
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    except TypeError:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def _tokenize_chat(
    processor: Any,
    text: str,
    messages: List[Dict[str, Any]],
    *,
    max_length: int = 0,
    truncation_side: str = "left",
) -> Dict[str, torch.Tensor]:
    image_inputs, video_inputs = _extract_vision_inputs(messages)
    has_multimodal_inputs = bool(image_inputs or video_inputs)
    processor_kwargs: Dict[str, Any] = {
        "text": text,
        "padding": False,
        "return_tensors": "pt",
    }
    if int(max_length) > 0 and not has_multimodal_inputs:
        processor_kwargs["truncation"] = True
        processor_kwargs["max_length"] = int(max_length)
    if image_inputs:
        processor_kwargs["images"] = image_inputs
    if video_inputs:
        processor_kwargs["videos"] = video_inputs
    tokenizer = getattr(processor, "tokenizer", None)
    if (
        tokenizer is None
        or int(max_length) <= 0
        or not hasattr(tokenizer, "truncation_side")
        or has_multimodal_inputs
    ):
        return processor(**processor_kwargs)
    original_side = tokenizer.truncation_side
    tokenizer.truncation_side = str(truncation_side or "left")
    try:
        return processor(**processor_kwargs)
    finally:
        tokenizer.truncation_side = original_side


def _count_text_tokens(processor: Any, text: str) -> int:
    tokenizer = getattr(processor, "tokenizer", processor)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    input_ids = encoded["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        return int(input_ids.shape[-1])
    if input_ids and isinstance(input_ids[0], list):
        return int(len(input_ids[0]))
    return int(len(input_ids))


def _count_model_input_tokens(encoded_inputs: Dict[str, torch.Tensor]) -> int:
    input_ids = encoded_inputs["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        return int(input_ids.shape[-1])
    if input_ids and isinstance(input_ids[0], list):
        return int(len(input_ids[0]))
    return int(len(input_ids))


def _model_input_sequence_lengths(encoded_inputs: Dict[str, Any]) -> torch.Tensor:
    attention_mask = encoded_inputs.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        if attention_mask.ndim == 1:
            return attention_mask.to(dtype=torch.long).view(1, -1).sum(dim=-1)
        return attention_mask.to(dtype=torch.long).sum(dim=-1)
    input_ids = encoded_inputs.get("input_ids")
    if isinstance(input_ids, torch.Tensor):
        if input_ids.ndim == 1:
            return torch.tensor([int(input_ids.shape[0])], dtype=torch.long)
        return torch.full((int(input_ids.shape[0]),), int(input_ids.shape[-1]), dtype=torch.long)
    if isinstance(input_ids, list):
        if input_ids and isinstance(input_ids[0], list):
            return torch.tensor([len(row) for row in input_ids], dtype=torch.long)
        return torch.tensor([len(input_ids)], dtype=torch.long)
    raise ValueError("Encoded inputs must expose tensor or list-based input_ids/attention_mask.")


def _single_example_model_input_length(
    processor: Any,
    text: str,
    messages: List[Dict[str, Any]],
) -> int:
    encoded = _tokenize_chat(
        processor,
        text,
        messages,
        max_length=0,
        truncation_side="left",
    )
    lengths = _model_input_sequence_lengths(encoded).view(-1)
    if lengths.numel() != 1:
        raise ValueError("Single-example token span resolution expects exactly one encoded example.")
    return int(lengths[0].item())


def _resolve_retained_prompt_completion_lengths(
    *,
    processor: Any,
    prompt_messages: List[Dict[str, Any]],
    full_messages: List[Dict[str, Any]],
    prompt_text: str,
    full_text: str,
    retained_full_inputs: Dict[str, Any],
) -> Tuple[int, int, int]:
    prompt_inputs_exact = _tokenize_chat(
        processor,
        prompt_text,
        prompt_messages,
        max_length=0,
        truncation_side="left",
    )
    full_inputs_exact = _tokenize_chat(
        processor,
        full_text,
        full_messages,
        max_length=0,
        truncation_side="left",
    )
    prompt_lengths = _model_input_sequence_lengths(prompt_inputs_exact).view(-1)
    full_lengths = _model_input_sequence_lengths(full_inputs_exact).view(-1)
    retained_lengths = _model_input_sequence_lengths(retained_full_inputs).view(-1)
    if prompt_lengths.numel() != 1 or full_lengths.numel() != 1 or retained_lengths.numel() != 1:
        raise ValueError("RL completion-native batching expects single-example tokenized inputs.")

    prompt_total = int(prompt_lengths[0].item())
    full_total = int(full_lengths[0].item())
    retained_total = int(retained_lengths[0].item())
    if full_total < prompt_total:
        raise ValueError(
            f"Full sequence token length must be >= prompt token length, got full={full_total} prompt={prompt_total}."
        )
    if retained_total > full_total:
        raise ValueError(
            f"Retained sequence token length must be <= full token length, got retained={retained_total} full={full_total}."
        )

    completion_total = max(0, full_total - prompt_total)
    left_offset = max(0, full_total - retained_total)
    retained_prompt = max(0, min(retained_total, prompt_total - left_offset))
    retained_completion = max(0, retained_total - retained_prompt)
    return retained_prompt, retained_completion, completion_total


def _has_multimodal_content(messages: List[Dict[str, Any]]) -> bool:
    for message in messages:
        for item in message.get("content", []):
            item_type = item.get("type")
            if item_type == "image" and ("image" in item or "image_ref" in item):
                return True
            if item_type == "video" and "video" in item:
                return True
    return False


def _drop_oldest_multimodal_item(messages: List[Dict[str, Any]]) -> bool:
    for message_index, message in enumerate(messages):
        content = list(message.get("content", []))
        for content_index, item in enumerate(content):
            item_type = item.get("type")
            if item_type == "image" and ("image" in item or "image_ref" in item):
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
            if item_type == "video" and "video" in item:
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


def _is_timestamp_text_item(item: Dict[str, Any]) -> bool:
    if not isinstance(item, dict) or item.get("type") != "text":
        return False
    return bool(_TIMESTAMP_ONLY_RE.match(str(item.get("text") or "").strip()))


def _paired_multimodal_removal_indices(content: List[Dict[str, Any]], content_index: int) -> set[int]:
    removals = {int(content_index)}
    timestamp_index = int(content_index) - 1
    if 0 <= timestamp_index < len(content) and _is_timestamp_text_item(content[timestamp_index]):
        removals.add(timestamp_index)
    return removals


def _drop_oldest_history_message(messages: List[Dict[str, Any]]) -> bool:
    return drop_oldest_history_turn(messages)


def _fit_messages_to_budget(
    processor: Any,
    prompt_messages: List[Dict[str, Any]],
    *,
    target_response: str,
    max_seq_length: int = 0,
    truncation_side: str = "left",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str, str, Dict[str, torch.Tensor]]:
    fitted_prompt_messages = copy.deepcopy(prompt_messages)
    target_message = {"role": "assistant", "content": [{"type": "text", "text": target_response}]}
    max_length = int(max_seq_length)

    for _ in range(512):
        full_messages = fitted_prompt_messages + [target_message]
        prompt_text = _build_chat_text(processor, fitted_prompt_messages, add_generation_prompt=True)
        full_text = _build_chat_text(processor, full_messages, add_generation_prompt=False)
        has_multimodal = _has_multimodal_content(full_messages)
        full_inputs = _tokenize_chat(
            processor,
            full_text,
            full_messages,
            max_length=max_length if max_length > 0 and not has_multimodal else 0,
            truncation_side=truncation_side,
        )
        if max_length <= 0 or _count_model_input_tokens(full_inputs) <= max_length:
            return fitted_prompt_messages, full_messages, prompt_text, full_text, full_inputs

        if _drop_oldest_multimodal_item(fitted_prompt_messages):
            continue
        if _drop_oldest_history_message(fitted_prompt_messages):
            continue

        if has_multimodal:
            raise ValueError(
                f"Unable to fit multimodal example within max_seq_length={max_length}. "
                "Increase the sequence budget or reduce retained multimodal context."
            )
        return fitted_prompt_messages, full_messages, prompt_text, full_text, full_inputs

    raise RuntimeError("Exceeded pruning attempts while fitting a multimodal example to the sequence budget.")


def _tag_messages_for_cache(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tagged_messages: List[Dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        tagged_message = dict(message)
        tagged_message["_cache_message_index"] = int(message_index)
        tagged_content: List[Dict[str, Any]] = []
        for content_index, item in enumerate(message.get("content", [])):
            tagged_item = dict(item)
            tagged_item["_cache_content_index"] = int(content_index)
            tagged_content.append(tagged_item)
        tagged_message["content"] = tagged_content
        tagged_messages.append(tagged_message)
    return tagged_messages


def _capture_message_plan(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for message in messages:
        message_index = message.get("_cache_message_index")
        if message_index is None:
            continue
        plan.append(
            {
                "message_index": int(message_index),
                "content_indices": [
                    int(item["_cache_content_index"])
                    for item in message.get("content", [])
                    if item.get("_cache_content_index") is not None
                ],
            }
        )
    return plan


def _apply_cached_message_plan(
    original_messages: List[Dict[str, Any]],
    plan: List[Dict[str, Any]],
    *,
    max_image_side: int = 0,
    max_image_pixels: int = 0,
) -> List[Dict[str, Any]]:
    rebuilt_messages: List[Dict[str, Any]] = []
    for entry in plan:
        message_index = int(entry.get("message_index", -1))
        if not 0 <= message_index < len(original_messages):
            continue
        source_message = original_messages[message_index]
        rebuilt_message = {
            key: copy.deepcopy(value)
            for key, value in source_message.items()
            if key != "content"
        }
        rebuilt_content: List[Dict[str, Any]] = []
        source_content = list(source_message.get("content", []))
        for content_index in entry.get("content_indices", []):
            if not 0 <= int(content_index) < len(source_content):
                continue
            item = copy.deepcopy(source_content[int(content_index)])
            if item.get("type") == "image" and "image" in item:
                item["image"] = _resize_image_for_training(
                    item["image"],
                    max_image_side=max_image_side,
                    max_image_pixels=max_image_pixels,
                )
            elif item.get("type") == "video" and "video" in item:
                video = item["video"]
                if isinstance(video, (list, tuple)):
                    item["video"] = [
                        _resize_image_for_training(
                            frame,
                            max_image_side=max_image_side,
                            max_image_pixels=max_image_pixels,
                        )
                        for frame in video
                    ]
                elif isinstance(video, torch.Tensor) and video.ndim == 4:
                    item["video"] = [
                        _resize_image_for_training(
                            frame,
                            max_image_side=max_image_side,
                            max_image_pixels=max_image_pixels,
                        )
                        for frame in video
                    ]
            rebuilt_content.append(item)
        rebuilt_message["content"] = rebuilt_content
        rebuilt_messages.append(rebuilt_message)
    return rebuilt_messages


def _build_response_labels(
    input_ids: torch.Tensor,
    *,
    response_token_count: int,
) -> torch.Tensor:
    labels = input_ids.clone()
    labels.fill_(-100)
    retained_response_tokens = min(max(int(response_token_count), 0), int(labels.shape[-1]))
    if retained_response_tokens <= 0:
        return labels
    labels[:, -retained_response_tokens:] = input_ids[:, -retained_response_tokens:]
    return labels


def _build_completion_mask_from_suffix_length(
    input_ids: torch.Tensor,
    *,
    completion_token_count: int,
) -> torch.Tensor:
    completion_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    retained_completion_tokens = min(max(int(completion_token_count), 0), int(input_ids.shape[-1]))
    if retained_completion_tokens <= 0:
        return completion_mask
    completion_mask[:, -retained_completion_tokens:] = True
    return completion_mask


def _build_labels_from_completion_mask(
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    labels = input_ids.clone()
    labels.fill_(-100)
    labels[completion_mask] = input_ids[completion_mask]
    return labels


def _normalize_batch_build_result(
    result: Any,
    *,
    is_episode_feature: bool,
) -> BatchBuildResult:
    if isinstance(result, BatchBuildResult):
        result.is_episode_feature = bool(is_episode_feature)
        return result
    if isinstance(result, tuple) and len(result) == 2:
        batch, cached_plan = result
        completion_token_count = 0
        if isinstance(batch, dict):
            labels = batch.get("labels")
            if isinstance(labels, torch.Tensor):
                completion_token_count = int(labels.ne(-100).sum().item())
        return BatchBuildResult(
            batch=batch,
            cached_plan=cached_plan,
            completion_token_count=completion_token_count,
            drop_reason="zero_response_after_budgeting" if completion_token_count <= 0 else None,
            budgeting_attempted=True,
            is_episode_feature=bool(is_episode_feature),
        )
    raise TypeError(f"Unsupported batch build result type: {type(result)!r}")


def _is_valid_pretokenized_sft_payload(payload: Dict[str, Any]) -> bool:
    if int(payload.get("_sft_pretokenized_schema_version", payload.get("sft_pretokenized_schema_version") or 0)) != int(
        STRICT_SFT_PRETOKENIZED_SCHEMA_VERSION
    ):
        return False
    input_ids = payload.get("input_ids")
    attention_mask = payload.get("attention_mask")
    labels = payload.get("labels")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor) or not isinstance(labels, torch.Tensor):
        return False
    if input_ids.ndim != 2 or attention_mask.ndim != 2 or labels.ndim != 2:
        return False
    if input_ids.shape != attention_mask.shape or input_ids.shape != labels.shape:
        return False
    if not bool(torch.any(labels.ne(-100))):
        return False
    forbidden_keys = {"completion_mask", "completion_token_count", "sample_weight", "advantage", "token_advantages"}
    if any(key in payload for key in forbidden_keys):
        return False
    return True

def _load_prepared_jsonl_rows(
    path: str | Path,
    *,
    include_splits: Optional[str | Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    include_split_set = set(parse_include_splits(include_splits) or [])
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Invalid prepared SFT row in {path}: expected dict")
            if include_split_set and str(row.get("split") or "") not in include_split_set:
                continue
            rows.append(row)
    return rows

def materialize_example_for_training(
    example: Dict[str, Any],
    *,
    resolver: Optional["_FrameReferenceResolver"] = None,
) -> Dict[str, Any]:
    prepared_example = copy.deepcopy(example)
    prepared_example.setdefault("_feature_cache_key", build_sft_tensor_cache_key(example))
    active_resolver = resolver or _FrameReferenceResolver()
    prepared_example["messages"] = active_resolver.materialize_messages(prepared_example["messages"])
    return prepared_example


def materialize_example_messages(
    example: Dict[str, Any],
    *,
    resolver: Optional["_FrameReferenceResolver"] = None,
) -> Dict[str, Any]:
    return materialize_example_for_training(example, resolver=resolver)


def _is_episode_feature(feature: Dict[str, Any]) -> bool:
    return isinstance(feature.get("messages"), list) and isinstance(feature.get("assistant_supervision"), list)


def _fit_episode_messages_to_budget(
    processor: Any,
    messages: List[Dict[str, Any]],
    *,
    max_seq_length: int = 0,
    truncation_side: str = "left",
) -> Tuple[List[Dict[str, Any]], str, Dict[str, torch.Tensor]]:
    fitted_messages = copy.deepcopy(messages)
    max_length = int(max_seq_length)

    for _ in range(512):
        full_text = _build_chat_text(processor, fitted_messages, add_generation_prompt=False)
        has_multimodal = _has_multimodal_content(fitted_messages)
        full_inputs = _tokenize_chat(
            processor,
            full_text,
            fitted_messages,
            max_length=max_length if max_length > 0 and not has_multimodal else 0,
            truncation_side=truncation_side,
        )
        if max_length <= 0 or _count_model_input_tokens(full_inputs) <= max_length:
            return fitted_messages, full_text, full_inputs

        if _drop_oldest_multimodal_item(fitted_messages):
            continue
        if _drop_oldest_history_message(fitted_messages):
            continue

        if has_multimodal:
            raise ValueError(
                f"Unable to fit episode-format multimodal example within max_seq_length={max_length}. "
                "Increase the sequence budget or reduce retained multimodal context."
            )
        return fitted_messages, full_text, full_inputs

    raise RuntimeError("Exceeded pruning attempts while fitting an episode-format example to the sequence budget.")


def _build_episode_labels_and_token_advantages(
    *,
    processor: Any,
    messages: List[Dict[str, Any]],
    full_text: str,
    full_inputs: Dict[str, torch.Tensor],
    assistant_supervision: Sequence[Dict[str, Any]],
    base_advantage: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = full_inputs["input_ids"]
    labels = input_ids.clone()
    labels.fill_(-100)
    token_advantages = input_ids.new_zeros(input_ids.shape, dtype=torch.float32)

    if not assistant_supervision:
        return labels, token_advantages

    retained_token_count = int(input_ids.shape[-1])
    total_token_count = _single_example_model_input_length(
        processor,
        full_text,
        messages,
    )
    left_offset = max(0, total_token_count - retained_token_count)
    position_by_original_index = {
        int(message.get("_cache_message_index")): idx
        for idx, message in enumerate(messages)
        if message.get("_cache_message_index") is not None
    }
    prefix_token_count_cache: Dict[int, int] = {}
    span_token_count_cache: Dict[int, int] = {}

    for entry in assistant_supervision:
        try:
            original_message_index = int(entry.get("assistant_message_index"))
        except Exception:
            continue
        message_position = position_by_original_index.get(original_message_index)
        if message_position is None:
            continue
        message = messages[message_position]
        if str(message.get("role") or "") != "assistant":
            continue

        prefix_messages = messages[:message_position]
        span_messages = messages[: message_position + 1]
        prefix_text = _build_chat_text(
            processor,
            prefix_messages,
            add_generation_prompt=True,
        )
        span_text = _build_chat_text(
            processor,
            span_messages,
            add_generation_prompt=False,
        )
        if message_position not in prefix_token_count_cache:
            prefix_token_count_cache[message_position] = _single_example_model_input_length(
                processor,
                prefix_text,
                prefix_messages,
            )
        if message_position not in span_token_count_cache:
            span_token_count_cache[message_position] = _single_example_model_input_length(
                processor,
                span_text,
                span_messages,
            )
        span_start = max(0, prefix_token_count_cache[message_position] - left_offset)
        span_end = min(retained_token_count, span_token_count_cache[message_position] - left_offset)
        if span_end <= span_start:
            continue

        labels[:, span_start:span_end] = input_ids[:, span_start:span_end]
        loss_weight = float(entry.get("loss_weight", 1.0) or 1.0)
        token_advantages[:, span_start:span_end] = float(base_advantage) * loss_weight

    return labels, token_advantages


def _build_episode_batch_from_feature(
    processor: Any,
    feature: Dict[str, Any],
    *,
    max_image_side: int = 0,
    max_image_pixels: int = 0,
    keep_recent_tool_image_messages: int = 0,
    max_total_images: int = 0,
    max_seq_length: int = 0,
    keep_recent_text_messages: int = 0,
) -> BatchBuildResult:
    tagged_messages = _tag_messages_for_cache(feature["messages"])
    prepared_messages = _prepare_messages(
        tagged_messages,
        max_image_side=max_image_side,
        max_image_pixels=max_image_pixels,
        keep_recent_tool_image_messages=keep_recent_tool_image_messages,
        max_total_images=max_total_images,
        keep_recent_text_messages=keep_recent_text_messages,
    )
    fitted_messages, full_text, full_inputs = _fit_episode_messages_to_budget(
        processor,
        prepared_messages,
        max_seq_length=max_seq_length,
        truncation_side="left",
    )

    base_advantage = float(feature.get("advantage", feature.get("sample_weight", 1.0)) or 0.0)
    labels, token_advantages = _build_episode_labels_and_token_advantages(
        processor=processor,
        messages=fitted_messages,
        full_text=full_text,
        full_inputs=full_inputs,
        assistant_supervision=list(feature.get("assistant_supervision") or []),
        base_advantage=base_advantage,
    )

    batch = dict(full_inputs)
    batch["labels"] = labels
    batch["completion_mask"] = labels.ne(-100)
    completion_token_count = int(batch["completion_mask"].sum().item())
    batch["completion_token_count"] = torch.tensor([completion_token_count], dtype=torch.long)
    batch["sample_weight"] = torch.tensor([float(feature.get("sample_weight", 1.0))], dtype=torch.float32)
    if "advantage" in feature:
        batch["advantage"] = torch.tensor([float(feature.get("advantage", 0.0))], dtype=torch.float32)
    if completion_token_count > 0:
        batch["token_advantages"] = token_advantages
        return BatchBuildResult(
            batch=batch,
            cached_plan=None,
            completion_token_count=completion_token_count,
            drop_reason=None,
            budgeting_attempted=True,
            is_episode_feature=True,
        )
    return BatchBuildResult(
        batch=None,
        cached_plan=None,
        completion_token_count=0,
        drop_reason="zero_response_after_budgeting",
        budgeting_attempted=True,
        is_episode_feature=True,
    )


def _extract_assistant_text(message: Dict[str, Any]) -> str:
    texts: List[str] = []
    for item in list(message.get("content") or []):
        if item.get("type") == "text":
            texts.append(str(item.get("text") or ""))
    return "".join(texts).strip()


def _coerce_rl_prompt_completion_feature(
    feature: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], str]:
    prompt_messages = feature.get("prompt_messages")
    completion_text = str(feature.get("completion_text") or "").strip()
    if isinstance(prompt_messages, list) and completion_text:
        return copy.deepcopy(prompt_messages), completion_text

    messages = copy.deepcopy(list(feature.get("messages") or []))
    if not messages:
        raise ValueError("RL completion-native batching requires prompt_messages or legacy messages.")

    target_response = str(feature.get("target_response") or "").strip()
    last_message = messages[-1]
    last_role = str(last_message.get("role") or "")
    last_assistant_text = _extract_assistant_text(last_message) if last_role == "assistant" else ""
    if not completion_text:
        completion_text = target_response or last_assistant_text
    if not completion_text:
        raise ValueError("RL completion-native batching requires a non-empty completion_text/target_response.")

    if last_role == "assistant" and last_assistant_text == completion_text:
        messages = messages[:-1]
    return messages, completion_text


def _build_rl_completion_episode_spec_from_feature(
    processor: Any,
    feature: Dict[str, Any],
    *,
    max_image_side: int = 0,
    max_image_pixels: int = 0,
    keep_recent_tool_image_messages: int = 0,
    max_total_images: int = 0,
    max_seq_length: int = 0,
    keep_recent_text_messages: int = 0,
) -> BatchBuildResult:
    prompt_messages_raw, completion_text = _coerce_rl_prompt_completion_feature(feature)
    tagged_messages = _tag_messages_for_cache(prompt_messages_raw)
    prompt_messages = _prepare_messages(
        tagged_messages,
        max_image_side=max_image_side,
        max_image_pixels=max_image_pixels,
        keep_recent_tool_image_messages=keep_recent_tool_image_messages,
        max_total_images=max_total_images,
        keep_recent_text_messages=keep_recent_text_messages,
    )
    prompt_messages, full_messages, prompt_text, full_text, full_inputs = _fit_messages_to_budget(
        processor,
        prompt_messages,
        target_response=completion_text,
        max_seq_length=max_seq_length,
        truncation_side="left",
    )

    retained_prompt_tokens, retained_completion_tokens, completion_token_count_total = (
        _resolve_retained_prompt_completion_lengths(
            processor=processor,
            prompt_messages=prompt_messages,
            full_messages=full_messages,
            prompt_text=prompt_text,
            full_text=full_text,
            retained_full_inputs=full_inputs,
        )
    )

    if retained_completion_tokens <= 0:
        return BatchBuildResult(
            batch=None,
            cached_plan=None,
            completion_token_count=0,
            drop_reason="zero_response_after_budgeting",
            budgeting_attempted=True,
            is_episode_feature=True,
        )
    if retained_prompt_tokens <= 0:
        return BatchBuildResult(
            batch=None,
            cached_plan=None,
            completion_token_count=int(retained_completion_tokens),
            drop_reason="zero_prompt_after_budgeting",
            budgeting_attempted=True,
            is_episode_feature=True,
        )
    if retained_completion_tokens < completion_token_count_total:
        return BatchBuildResult(
            batch=None,
            cached_plan=None,
            completion_token_count=int(retained_completion_tokens),
            drop_reason="truncated_completion_after_budgeting",
            budgeting_attempted=True,
            is_episode_feature=True,
        )

    input_ids = full_inputs["input_ids"]
    attention_mask = full_inputs["attention_mask"]
    prompt_ids = input_ids[:, :retained_prompt_tokens].clone()
    prompt_mask = attention_mask[:, :retained_prompt_tokens].clone()
    completion_ids = input_ids[:, retained_prompt_tokens:].clone()
    completion_mask = attention_mask[:, retained_prompt_tokens:].clone()

    episode_spec: Dict[str, Any] = {
        "prompt_ids": prompt_ids.detach().cpu(),
        "prompt_mask": prompt_mask.detach().cpu(),
        "completion_ids": completion_ids.detach().cpu(),
        "completion_mask": completion_mask.detach().cpu(),
        "prompt_token_count": torch.tensor([int(retained_prompt_tokens)], dtype=torch.long),
        "completion_token_count": torch.tensor([int(retained_completion_tokens)], dtype=torch.long),
        "sample_weight": torch.tensor([float(feature.get("sample_weight", 1.0))], dtype=torch.float32),
        "advantage": torch.tensor(
            [float(feature.get("advantage", feature.get("sample_weight", 1.0)) or 0.0)],
            dtype=torch.float32,
        ),
    }
    for key, value in full_inputs.items():
        if key in {"input_ids", "attention_mask"}:
            continue
        if isinstance(value, torch.Tensor):
            episode_spec[key] = value.detach().cpu()
        else:
            episode_spec[key] = copy.deepcopy(value)
    return BatchBuildResult(
        batch=episode_spec,
        cached_plan=None,
        completion_token_count=int(retained_completion_tokens),
        drop_reason=None,
        budgeting_attempted=True,
        is_episode_feature=True,
    )


def _build_batch_from_feature(
    processor: Any,
    feature: Dict[str, Any],
    *,
    max_image_side: int = 0,
    max_image_pixels: int = 0,
    keep_recent_tool_image_messages: int = 0,
    max_total_images: int = 0,
    max_seq_length: int = 0,
    keep_recent_text_messages: int = 0,
    cached_plan: Optional[Dict[str, Any]] = None,
) -> BatchBuildResult:
    if _is_episode_feature(feature):
        return _build_episode_batch_from_feature(
            processor,
            feature,
            max_image_side=max_image_side,
            max_image_pixels=max_image_pixels,
            keep_recent_tool_image_messages=keep_recent_tool_image_messages,
            max_total_images=max_total_images,
            max_seq_length=max_seq_length,
            keep_recent_text_messages=keep_recent_text_messages,
        )
    if cached_plan is not None:
        prompt_messages = _apply_cached_message_plan(
            feature["messages"],
            list(cached_plan.get("message_plan") or []),
            max_image_side=max_image_side,
            max_image_pixels=max_image_pixels,
        )
        target_message = {"role": "assistant", "content": [{"type": "text", "text": feature["target_response"]}]}
        full_messages = prompt_messages + [target_message]
        full_text = str(cached_plan.get("full_text") or "")
        full_inputs = _tokenize_chat(
            processor,
            full_text,
            full_messages,
            max_length=max_seq_length,
            truncation_side="left",
        )
        completion_token_count_total = int(cached_plan.get("completion_token_count_total") or 0)
        next_cached_plan = None
    else:
        tagged_messages = _tag_messages_for_cache(feature["messages"])
        prompt_messages = _prepare_messages(
            tagged_messages,
            max_image_side=max_image_side,
            max_image_pixels=max_image_pixels,
            keep_recent_tool_image_messages=keep_recent_tool_image_messages,
            max_total_images=max_total_images,
            keep_recent_text_messages=keep_recent_text_messages,
        )
        prompt_messages, full_messages, prompt_text, full_text, full_inputs = _fit_messages_to_budget(
            processor,
            prompt_messages,
            target_response=feature["target_response"],
            max_seq_length=max_seq_length,
            truncation_side="left",
        )
        completion_token_count_total = max(
            0,
            _count_text_tokens(processor, full_text) - _count_text_tokens(processor, prompt_text),
        )
        next_cached_plan = {
            "message_plan": _capture_message_plan(prompt_messages),
            "full_text": full_text,
            "completion_token_count_total": int(completion_token_count_total),
        }

    completion_mask = _build_completion_mask_from_suffix_length(
        full_inputs["input_ids"],
        completion_token_count=completion_token_count_total,
    )
    labels = _build_labels_from_completion_mask(full_inputs["input_ids"], completion_mask)

    batch = dict(full_inputs)
    batch["labels"] = labels
    batch["completion_mask"] = completion_mask
    batch["sample_weight"] = torch.tensor([float(feature.get("sample_weight", 1.0))], dtype=torch.float32)
    if "advantage" in feature:
        batch["advantage"] = torch.tensor([float(feature.get("advantage", 0.0))], dtype=torch.float32)
    response_positions = labels.ne(-100)
    response_token_count = int(response_positions.sum().item())
    batch["completion_token_count"] = torch.tensor([response_token_count], dtype=torch.long)
    if response_token_count <= 0:
        return BatchBuildResult(
            batch=None,
            cached_plan=next_cached_plan,
            completion_token_count=0,
            drop_reason="zero_response_after_budgeting",
            budgeting_attempted=True,
            is_episode_feature=False,
        )
    if response_token_count > 0:
        token_advantages = _build_token_advantages_for_feature(
            processor=processor,
            feature=feature,
            response_token_count=response_token_count,
        )
        full_token_advantages = labels.new_zeros(labels.shape, dtype=torch.float32)
        full_token_advantages[response_positions] = torch.tensor(token_advantages, dtype=torch.float32)
        batch["token_advantages"] = full_token_advantages
    return BatchBuildResult(
        batch=batch,
        cached_plan=next_cached_plan,
        completion_token_count=response_token_count,
        drop_reason=None,
        budgeting_attempted=True,
        is_episode_feature=False,
    )

class _FrameReferenceResolver:
    def __init__(self, *, max_cached_videos: int = 2, allow_raw_video_fallback: bool = False):
        self.max_cached_videos = max(0, int(max_cached_videos))
        self.allow_raw_video_fallback = bool(allow_raw_video_fallback)
        self._frame_cache_tensors: "OrderedDict[str, Optional[torch.Tensor]]" = OrderedDict()
        self._frame_cache_status: "OrderedDict[str, str]" = OrderedDict()
        self._decord_readers: "OrderedDict[str, Any]" = OrderedDict()
        self._logged_raw_frame_fallbacks: set[tuple[str, str, str]] = set()
        self._stats: Dict[str, int] = {
            "frame_cache_hits": 0,
            "raw_frame_fallbacks": 0,
            "missing_frame_cache": 0,
            "missing_sampled_frame_index": 0,
            "sampled_frame_index_out_of_range": 0,
            "cache_load_error": 0,
            "cache_invalid": 0,
        }

    def snapshot_stats(self) -> Dict[str, int]:
        return {key: int(value) for key, value in self._stats.items()}

    def materialize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        materialized = copy.deepcopy(messages)
        for message in materialized:
            for item in message.get("content", []):
                image_ref = item.pop("image_ref", None)
                if item.get("type") != "image" or image_ref is None:
                    continue
                item["image"] = self._resolve_image_ref(image_ref)
        return materialized

    def _resolve_image_ref(self, image_ref: Dict[str, Any]) -> torch.Tensor:
        video_path = Path(str(image_ref.get("video_path") or ""))
        if not str(video_path):
            raise ValueError("image_ref is missing video_path")

        sampled_frame_index = image_ref.get("sampled_frame_index")
        cache_tensor, cache_status = self._load_frame_cache_tensor(video_path)
        if cache_tensor is not None and sampled_frame_index is not None:
            index = int(sampled_frame_index)
            if 0 <= index < int(cache_tensor.shape[0]):
                self._stats["frame_cache_hits"] += 1
                return cache_tensor[index]

        fallback_reason = "missing_frame_cache"
        if cache_tensor is not None and sampled_frame_index is None:
            fallback_reason = "missing_sampled_frame_index"
        elif cache_tensor is not None and sampled_frame_index is not None:
            fallback_reason = f"sampled_frame_index_out_of_range:{int(sampled_frame_index)}"
        elif cache_status != "missing":
            fallback_reason = f"cache_status:{cache_status}"
        self._stats["raw_frame_fallbacks"] += 1
        if fallback_reason == "missing_frame_cache":
            self._stats["missing_frame_cache"] += 1
        elif fallback_reason == "missing_sampled_frame_index":
            self._stats["missing_sampled_frame_index"] += 1
        elif fallback_reason.startswith("sampled_frame_index_out_of_range:"):
            self._stats["sampled_frame_index_out_of_range"] += 1
        if not self.allow_raw_video_fallback:
            raise ValueError(
                "Prepared multimodal training data requires prebuilt frame_cache and exact sampled frame references. "
                f"video_path={video_path} cache_status={cache_status} fallback_reason={fallback_reason} "
                f"cache_path={_frame_cache_path(video_path)}"
            )
        self._warn_raw_frame_fallback(video_path=video_path, cache_status=cache_status, fallback_reason=fallback_reason)

        raw_frame_index = image_ref.get("raw_frame_index")
        if raw_frame_index is None:
            raw_frame_index = sampled_frame_index
        if raw_frame_index is None and image_ref.get("timestamp_sec") is not None:
            raw_frame_index = self._resolve_frame_index_from_timestamp(video_path, float(image_ref["timestamp_sec"]))
        if raw_frame_index is None:
            raise ValueError(f"Cannot resolve image_ref for {video_path}")
        return self._load_raw_video_frame(video_path, int(raw_frame_index))

    def _load_frame_cache_tensor(self, video_path: Path) -> tuple[Optional[torch.Tensor], str]:
        key = str(video_path)
        if key in self._frame_cache_tensors:
            self._touch_cache_key(key)
            return self._frame_cache_tensors[key], self._frame_cache_status.get(key, "missing")

        cache_path = _frame_cache_path(video_path)
        if not cache_path.exists():
            self._store_frame_cache_entry(key, None, "missing")
            return None, "missing"

        try:
            cache = torch.load(cache_path, map_location="cpu")
        except Exception:
            self._stats["cache_load_error"] += 1
            self._store_frame_cache_entry(key, None, "load_error")
            return None, "load_error"

        frame_tensor = cache.get("frame_tensor")
        resolved_tensor = frame_tensor if isinstance(frame_tensor, torch.Tensor) else None
        resolved_status = "loaded" if resolved_tensor is not None else "invalid"
        if resolved_tensor is None:
            self._stats["cache_invalid"] += 1
        self._store_frame_cache_entry(key, resolved_tensor, resolved_status)
        return resolved_tensor, resolved_status

    def _warn_raw_frame_fallback(self, *, video_path: Path, cache_status: str, fallback_reason: str) -> None:
        warning_key = (str(video_path), str(cache_status), str(fallback_reason))
        if warning_key in self._logged_raw_frame_fallbacks:
            return
        self._logged_raw_frame_fallbacks.add(warning_key)
        _print_cache_warning(
            f"video_path={video_path} cache_status={cache_status} fallback_reason={fallback_reason} "
            f"cache_path={_frame_cache_path(video_path)} falling back to raw frame materialization."
        )

    def _resolve_frame_index_from_timestamp(self, video_path: Path, timestamp_sec: float) -> int:
        try:
            reader = self._get_decord_reader(video_path)
            fps = float(reader.get_avg_fps() or 0.0)
            if fps > 0:
                return max(0, min(int(round(timestamp_sec * fps)), len(reader) - 1))
        except Exception:
            pass

        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("Resolving image_ref by timestamp requires decord or cv2.") from exc

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video for timestamp resolution: {video_path}")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            capture.release()
        if fps <= 0:
            raise RuntimeError(f"Failed to read fps for video: {video_path}")
        if frame_count <= 0:
            return max(0, int(round(timestamp_sec * fps)))
        return max(0, min(int(round(timestamp_sec * fps)), frame_count - 1))

    def _load_raw_video_frame(self, video_path: Path, frame_index: int) -> torch.Tensor:
        try:
            reader = self._get_decord_reader(video_path)
            frame = reader[int(frame_index)].asnumpy()
            return torch.from_numpy(frame).permute(2, 0, 1).contiguous()
        except Exception:
            return self._load_raw_video_frame_with_cv2(video_path, frame_index)

    def _get_decord_reader(self, video_path: Path):
        key = str(video_path)
        if key in self._decord_readers:
            self._touch_cache_key(key)
            return self._decord_readers[key]
        from decord import VideoReader, cpu

        reader = VideoReader(str(video_path), ctx=cpu(0))
        self._decord_readers[key] = reader
        self._touch_cache_key(key)
        self._prune_cached_videos()
        return reader

    def _store_frame_cache_entry(self, key: str, tensor: Optional[torch.Tensor], status: str) -> None:
        self._frame_cache_tensors[key] = tensor
        self._frame_cache_status[key] = str(status)
        self._touch_cache_key(key)
        self._prune_cached_videos()

    def _touch_cache_key(self, key: str) -> None:
        if key in self._frame_cache_tensors:
            self._frame_cache_tensors.move_to_end(key)
        if key in self._frame_cache_status:
            self._frame_cache_status.move_to_end(key)
        if key in self._decord_readers:
            self._decord_readers.move_to_end(key)

    def _prune_cached_videos(self) -> None:
        if self.max_cached_videos <= 0:
            keys_to_drop = list(self._frame_cache_tensors.keys())
            keys_to_drop.extend([key for key in self._decord_readers.keys() if key not in self._frame_cache_tensors])
            for key in list(dict.fromkeys(keys_to_drop)):
                self._evict_cache_key(key)
            return
        while len(self._frame_cache_tensors) > self.max_cached_videos:
            oldest_key = next(iter(self._frame_cache_tensors))
            self._evict_cache_key(oldest_key)
        while len(self._decord_readers) > self.max_cached_videos:
            oldest_key = next(iter(self._decord_readers))
            self._evict_cache_key(oldest_key)

    def _evict_cache_key(self, key: str) -> None:
        self._frame_cache_tensors.pop(key, None)
        self._frame_cache_status.pop(key, None)
        self._decord_readers.pop(key, None)

    @staticmethod
    def _load_raw_video_frame_with_cv2(video_path: Path, frame_index: int) -> torch.Tensor:
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError("Materializing image_ref requires decord or cv2.") from exc

        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video for frame materialization: {video_path}")
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_index} from {video_path}")
        frame = frame[:, :, ::-1].copy()
        return torch.from_numpy(frame).permute(2, 0, 1).contiguous()


class SingleExampleMultimodalCollator:
    """Keep batching conservative for multimodal Qwen training.

    This collator expects `per_device_train_batch_size=1`. Users can recover
    throughput with gradient accumulation and multi-GPU launch.
    """

    def __init__(
        self,
        processor: Any,
        *,
        max_image_side: int = 0,
        max_image_pixels: int = 0,
        keep_recent_tool_image_messages: int = 0,
        max_total_images: int = 0,
        max_seq_length: int = 0,
        keep_recent_text_messages: int = 0,
        feature_plan_cache_size: int = 4096,
    ):
        self.processor = processor
        self.max_image_side = int(max_image_side)
        self.max_image_pixels = int(max_image_pixels)
        self.keep_recent_tool_image_messages = int(keep_recent_tool_image_messages)
        self.max_total_images = int(max_total_images)
        self.max_seq_length = int(max_seq_length)
        self.keep_recent_text_messages = int(keep_recent_text_messages)
        self.feature_plan_cache_size = max(0, int(feature_plan_cache_size))
        self._feature_plan_cache: "OrderedDict[Tuple[str, str], Dict[str, Any]]" = OrderedDict()
        self._budgeting_stats = BudgetingStats()

    def _feature_cache_key(self, feature: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        if _is_episode_feature(feature) and "target_response" not in feature:
            return None
        raw_key = feature.get("_feature_cache_key")
        if raw_key is None or str(raw_key) == "":
            return None
        return str(raw_key), str(feature.get("target_response") or "")

    def _get_cached_plan(self, feature: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cache_key = self._feature_cache_key(feature)
        if cache_key is None or self.feature_plan_cache_size <= 0:
            return None
        cached = self._feature_plan_cache.get(cache_key)
        if cached is None:
            return None
        self._feature_plan_cache.move_to_end(cache_key)
        return cached

    def _store_cached_plan(self, feature: Dict[str, Any], payload: Dict[str, Any]) -> None:
        cache_key = self._feature_cache_key(feature)
        if cache_key is None or self.feature_plan_cache_size <= 0:
            return
        self._feature_plan_cache[cache_key] = dict(payload)
        self._feature_plan_cache.move_to_end(cache_key)
        while len(self._feature_plan_cache) > self.feature_plan_cache_size:
            self._feature_plan_cache.popitem(last=False)

    def get_budget_drop_metrics(self) -> Dict[str, Any]:
        return self._budgeting_stats.as_dict()

    def get_budgeting_stats(self) -> BudgetingStats:
        stats = BudgetingStats()
        stats.merge(self._budgeting_stats)
        return stats

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(features) != 1:
            raise ValueError(
                "SingleExampleMultimodalCollator currently requires per_device_train_batch_size=1 "
                "for stable multimodal padding."
            )
        feature = features[0]
        if bool(feature.get("_pretokenized", False)):
            return {
                key: value
                for key, value in feature.items()
                if not str(key).startswith("_")
            }

        feature = copy.deepcopy(feature)
        cached_plan = self._get_cached_plan(feature)
        result = _normalize_batch_build_result(
            _build_batch_from_feature(
                self.processor,
                feature,
                max_image_side=self.max_image_side,
                max_image_pixels=self.max_image_pixels,
                keep_recent_tool_image_messages=self.keep_recent_tool_image_messages,
                max_total_images=self.max_total_images,
                max_seq_length=self.max_seq_length,
                keep_recent_text_messages=self.keep_recent_text_messages,
                cached_plan=cached_plan,
            ),
            is_episode_feature=_is_episode_feature(feature),
        )
        self._budgeting_stats.record(result)
        if result.cached_plan is not None:
            self._store_cached_plan(feature, result.cached_plan)
        if result.batch is None:
            raise ValueError(
                f"Zero-response batch was not filtered before collation: reason={result.drop_reason or 'unknown'}"
            )
        return result.batch


class WeightedExampleDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: Sequence[Dict[str, Any]],
    ):
        self.examples = list(examples)
        self._frame_reference_resolver = _FrameReferenceResolver()
        self._example_cache_keys = [build_sft_tensor_cache_key(example) for example in self.examples]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cache_key = self._example_cache_keys[idx]
        example = materialize_example_for_training(
            self.examples[idx],
            resolver=self._frame_reference_resolver,
        )
        example["_feature_cache_key"] = cache_key
        return example


def _compact_trace_row_to_runtime_record(row: Dict[str, Any]) -> Dict[str, Any]:
    if not _is_compact_trace_sft_row(row):
        raise ValueError(
            "SFT training now requires compact_trace_v2 prepared rows. "
            f"Got legacy row keys={sorted(list(row.keys()))[:12]}"
        )
    runtime_record = {
        "video_id": row.get("video_id"),
        "split": row.get("split"),
        "video_path": row.get("video_path"),
        "video_meta": copy.deepcopy(row.get("video_meta") or {}),
        "scene": copy.deepcopy(row.get("scene") or {}),
        "agent_task": copy.deepcopy(row.get("agent_task") or {}),
        "structured_target": copy.deepcopy(row.get("structured_target") or {}),
        "tool_io": copy.deepcopy(row.get("tool_io") or {}),
        "label": copy.deepcopy(row.get("label") or {}),
        "temporal": copy.deepcopy(row.get("temporal") or {}),
        "evidence": copy.deepcopy(row.get("evidence") or {}),
        "language": copy.deepcopy(row.get("language") or {}),
        "qa_pairs": copy.deepcopy(row.get("qa_pairs") or []),
        "proposal_supervision": copy.deepcopy(row.get("proposal_supervision") or {}),
        "oracle_sft": {
            "trajectory": copy.deepcopy(list(row.get("oracle_trajectory") or [])),
            "final_decision": copy.deepcopy(row.get("oracle_final_decision") or {}),
        },
    }
    return runtime_record


def compact_trace_row_to_runtime_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return _compact_trace_row_to_runtime_record(row)


def _build_compact_trace_step_response(
    step: Dict[str, Any],
    *,
    record: Dict[str, Any],
    state: SaverEnvironmentState,
) -> tuple[str, Dict[str, Any], str]:
    tool_name = str(step.get("tool") or "").strip()
    arguments = copy.deepcopy(step.get("arguments") or {})
    if not tool_name:
        raise ValueError("Compact trace step is missing tool name.")
    response_arguments = copy.deepcopy(arguments)
    if tool_name == "verify_hypothesis":
        response_arguments, _ = _merge_verify_arguments_with_oracle_feedback(
            arguments,
            step.get("oracle_verifier_feedback"),
        )
    elif tool_name == "finalize_case":
        response_arguments = build_finalize_case_payload(arguments)
    response_text = _assistant_tool_response(
        tool_name,
        response_arguments,
        record=record,
        state=state,
    )
    return response_text, response_arguments, tool_name


def replay_compact_trace_messages(
    row: Dict[str, Any],
    *,
    config: Any = None,
    stop_before_step_index: int | None = None,
    proposal_runtime: Any = None,
    strict_feature_guided_proposal: bool = False,
) -> List[Dict[str, Any]]:
    record = _compact_trace_row_to_runtime_record(row)
    record_builder = SaverRecordItemBuilder(
        config=copy.deepcopy(config) if config is not None else None,
        require_frame_cache=True,
        require_feature_cache=True,
        proposal_runtime=proposal_runtime,
        strict_feature_guided_proposal=strict_feature_guided_proposal,
    )
    item = record_builder.build_item(record)
    adapter = TimeSearchRolloutAdapter(config=config)
    state = SaverEnvironmentState()
    messages = adapter.build_initial_messages(item)
    multimodal_cache = item["multimodal_cache"]
    for step_index, step in enumerate(list(row.get("oracle_trajectory") or []), start=1):
        if stop_before_step_index is not None and int(step_index) >= int(stop_before_step_index):
            break
        response_text, response_arguments, tool_name = _build_compact_trace_step_response(
            step,
            record=record,
            state=state,
        )
        assistant_message = adapter.build_assistant_message(response_text)
        messages.append(assistant_message)
        tool_message, state = execute_tool_call(
            {"function": {"name": tool_name, "arguments": response_arguments}},
            multimodal_cache,
            state,
        )
        tool_message = _apply_oracle_verifier_feedback(tool_message, step=step)
        messages.append(adapter.adapt_tool_observation(tool_message, multimodal_cache))
    return messages


def expand_compact_trace_row_to_step_rows(
    row: Dict[str, Any],
    *,
    config: Any = None,
    load_frame_cache: bool = True,
    load_feature_cache: bool = True,
    proposal_runtime: Any = None,
    strict_feature_guided_proposal: bool = False,
) -> List[Dict[str, Any]]:
    record = _compact_trace_row_to_runtime_record(row)
    record_builder = SaverRecordItemBuilder(
        config=copy.deepcopy(config) if config is not None else None,
        require_frame_cache=bool(load_frame_cache),
        require_feature_cache=bool(load_feature_cache),
        load_frame_cache=load_frame_cache,
        load_feature_cache=load_feature_cache,
        proposal_runtime=proposal_runtime,
        strict_feature_guided_proposal=strict_feature_guided_proposal,
    )
    item = record_builder.build_item(record)
    adapter = TimeSearchRolloutAdapter(config=config)
    state = SaverEnvironmentState()
    messages = adapter.build_initial_messages(item)
    multimodal_cache = item["multimodal_cache"]
    step_rows: List[Dict[str, Any]] = []
    for step_index, step in enumerate(list(row.get("oracle_trajectory") or []), start=1):
        response_text, response_arguments, tool_name = _build_compact_trace_step_response(
            step,
            record=record,
            state=state,
        )
        step_rows.append(
            {
                "prepared_format": str(PREPARED_SFT_FORMAT),
                "video_id": row.get("video_id"),
                "split": row.get("split"),
                "source": row.get("source"),
                "step_index": int(step_index),
                "messages": copy.deepcopy(messages),
                "target_response": response_text,
                "target_action": "tool_call",
                "tool_name": tool_name,
                "_compact_trace_step_index": int(step_index),
            }
        )
        assistant_message = adapter.build_assistant_message(response_text)
        messages.append(assistant_message)
        tool_message, state = execute_tool_call(
            {"function": {"name": tool_name, "arguments": response_arguments}},
            multimodal_cache,
            state,
        )
        tool_message = _apply_oracle_verifier_feedback(tool_message, step=step)
        messages.append(adapter.adapt_tool_observation(tool_message, multimodal_cache))
    return step_rows


def _strip_private_message_fields(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned_messages: List[Dict[str, Any]] = []
    for message in messages:
        cleaned_message = {
            key: copy.deepcopy(value)
            for key, value in message.items()
            if not str(key).startswith("_") and key != "content"
        }
        cleaned_content: List[Dict[str, Any]] = []
        for item in list(message.get("content") or []):
            if not isinstance(item, dict):
                cleaned_content.append(copy.deepcopy(item))
                continue
            cleaned_content.append(
                {
                    key: copy.deepcopy(value)
                    for key, value in item.items()
                    if not str(key).startswith("_")
                }
            )
        cleaned_message["content"] = cleaned_content
        cleaned_messages.append(cleaned_message)
    return cleaned_messages


def _assistant_region_mask(
    input_ids: torch.Tensor,
    *,
    processor: Any,
) -> torch.Tensor:
    tokenizer = getattr(processor, "tokenizer", processor)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        raise ValueError("Plain SFT assistant masking requires tokenizer.convert_tokens_to_ids(...).")
    im_start_id = convert("<|im_start|>")
    assistant_id = convert("assistant")
    if im_start_id is None or assistant_id is None:
        raise ValueError("Failed to resolve Qwen chat template token ids for assistant masking.")
    seq = input_ids if input_ids.ndim == 2 else input_ids.view(1, -1)
    is_im_start = seq == int(im_start_id)
    region_id = is_im_start.int().cumsum(dim=1)
    next_is_assistant = torch.zeros_like(seq, dtype=torch.bool)
    next_is_assistant[:, :-1] = is_im_start[:, :-1] & (seq[:, 1:] == int(assistant_id))
    region_flag = torch.zeros_like(region_id)
    region_flag = region_flag.scatter_add(
        dim=1,
        index=region_id,
        src=next_is_assistant.int().to(region_flag.dtype),
    )
    return region_flag.gather(dim=1, index=region_id).to(dtype=torch.bool)


def _build_plain_sft_labels(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    processor: Any,
) -> torch.Tensor:
    assistant_mask = _assistant_region_mask(input_ids, processor=processor)
    labels = input_ids.clone()
    labels[~assistant_mask] = -100
    labels[attention_mask == 0] = -100
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is not None:
        labels[input_ids == int(pad_token_id)] = -100
    image_token = getattr(processor, "image_token", None)
    video_token = getattr(processor, "video_token", None)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is not None:
        for special_token in (image_token, video_token):
            if not special_token:
                continue
            token_id = convert(special_token)
            if token_id is not None:
                labels[input_ids == int(token_id)] = -100
    return labels


def _sft_pad_and_concat(
    tensors: Sequence[torch.Tensor],
    *,
    pad_value: Any,
    pad_side: str = "right",
) -> torch.Tensor:
    if not tensors:
        raise ValueError("Cannot merge an empty tensor list.")
    max_seq_len = max(int(tensor.shape[-1]) for tensor in tensors)
    padded_tensors: List[torch.Tensor] = []
    for tensor in tensors:
        pad_width = max_seq_len - int(tensor.shape[-1])
        if pad_width <= 0:
            padded_tensors.append(tensor)
            continue
        pad_shape = list(tensor.shape)
        pad_shape[-1] = int(pad_width)
        pad_tensor = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
        if str(pad_side) == "left":
            padded_tensors.append(torch.cat([pad_tensor, tensor], dim=-1))
        else:
            padded_tensors.append(torch.cat([tensor, pad_tensor], dim=-1))
    return torch.cat(padded_tensors, dim=0)


def _tokenize_chat_batch(
    processor: Any,
    texts: Sequence[str],
    messages_batch: Sequence[List[Dict[str, Any]]],
    *,
    max_length: int = 0,
    truncation_side: str = "left",
) -> Dict[str, torch.Tensor]:
    if len(texts) != len(messages_batch):
        raise ValueError("Batch tokenization expects matching text/message batch lengths.")
    image_inputs: List[Any] = []
    video_inputs: List[Any] = []
    has_multimodal_inputs = False
    for messages in messages_batch:
        example_images, example_videos = _extract_vision_inputs(messages)
        if example_images:
            image_inputs.extend(example_images)
            has_multimodal_inputs = True
        if example_videos:
            video_inputs.extend(example_videos)
            has_multimodal_inputs = True

    processor_kwargs: Dict[str, Any] = {
        "text": list(texts),
        "padding": True,
        "return_tensors": "pt",
    }
    if int(max_length) > 0 and not has_multimodal_inputs:
        processor_kwargs["truncation"] = True
        processor_kwargs["max_length"] = int(max_length)
    if image_inputs:
        processor_kwargs["images"] = image_inputs
    if video_inputs:
        processor_kwargs["videos"] = video_inputs

    tokenizer = getattr(processor, "tokenizer", None)
    if (
        tokenizer is None
        or int(max_length) <= 0
        or not hasattr(tokenizer, "truncation_side")
        or has_multimodal_inputs
    ):
        return processor(**processor_kwargs)
    original_side = tokenizer.truncation_side
    tokenizer.truncation_side = str(truncation_side or "left")
    try:
        return processor(**processor_kwargs)
    finally:
        tokenizer.truncation_side = original_side


def _fit_batch_episode_messages_to_budget(
    processor: Any,
    messages_batch: Sequence[List[Dict[str, Any]]],
    *,
    max_seq_length: int = 0,
    truncation_side: str = "left",
) -> Tuple[List[List[Dict[str, Any]]], List[str]]:
    fitted_batches = [copy.deepcopy(messages) for messages in messages_batch]
    full_texts = [
        _build_chat_text(processor, messages, add_generation_prompt=False)
        for messages in fitted_batches
    ]
    max_length = int(max_seq_length)
    if max_length <= 0:
        return fitted_batches, full_texts

    pending_indices = [
        index
        for index, messages in enumerate(fitted_batches)
        if _has_multimodal_content(messages)
    ]
    for _ in range(512):
        if not pending_indices:
            return fitted_batches, full_texts
        current_messages = [fitted_batches[index] for index in pending_indices]
        current_texts = [
            _build_chat_text(processor, messages, add_generation_prompt=False)
            for messages in current_messages
        ]
        current_inputs = _tokenize_chat_batch(
            processor,
            current_texts,
            current_messages,
            max_length=0,
            truncation_side=truncation_side,
        )
        current_lengths = _model_input_sequence_lengths(current_inputs).view(-1)
        next_pending: List[int] = []
        for local_index, global_index in enumerate(pending_indices):
            full_texts[global_index] = current_texts[local_index]
            if int(current_lengths[local_index].item()) <= max_length:
                continue
            if _drop_oldest_multimodal_item(fitted_batches[global_index]):
                next_pending.append(global_index)
                continue
            if _drop_oldest_history_message(fitted_batches[global_index]):
                next_pending.append(global_index)
                continue
            raise ValueError(
                f"Unable to fit episode-format multimodal example within max_seq_length={max_length}. "
                "Increase the sequence budget or reduce retained multimodal context."
            )
        pending_indices = next_pending
    raise RuntimeError("Exceeded pruning attempts while fitting an episode-format batch to the sequence budget.")


def _truncate_text_only_batch_to_budget(
    batch_inputs: Dict[str, Any],
    *,
    processor: Any,
    multimodal_flags: Sequence[bool],
    max_seq_length: int = 0,
) -> Dict[str, Any]:
    max_length = int(max_seq_length)
    if max_length <= 0:
        return batch_inputs
    input_ids = batch_inputs.get("input_ids")
    attention_mask = batch_inputs.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(attention_mask, torch.Tensor):
        return batch_inputs
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        return batch_inputs
    batch_size = int(input_ids.shape[0])
    if len(multimodal_flags) != batch_size:
        raise ValueError("Text-only truncation expects one multimodal flag per batch row.")

    lengths = attention_mask.to(dtype=torch.long).sum(dim=-1).tolist()
    if all(bool(multimodal_flags[index]) or int(lengths[index]) <= max_length for index in range(batch_size)):
        return batch_inputs

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or getattr(tokenizer, "eos_token_id", 0) or 0)
    sequence_pad_values = {
        "input_ids": pad_token_id,
        "attention_mask": 0,
        "token_type_ids": 0,
        "position_ids": 0,
    }
    truncated_batch = dict(batch_inputs)
    for key, pad_value in sequence_pad_values.items():
        tensor = batch_inputs.get(key)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or int(tensor.shape[0]) != batch_size:
            continue
        rows: List[torch.Tensor] = []
        for index in range(batch_size):
            sequence_length = int(lengths[index])
            keep_length = sequence_length
            if not bool(multimodal_flags[index]) and sequence_length > max_length:
                keep_length = max_length
            if sequence_length <= 0:
                rows.append(tensor[index : index + 1, :0])
                continue
            valid_positions = attention_mask[index].nonzero(as_tuple=True)[0]
            if len(valid_positions) <= 0:
                rows.append(tensor[index : index + 1, :0])
                continue
            content_start = int(valid_positions[0].item())
            content_end = int(valid_positions[-1].item()) + 1
            content = tensor[index : index + 1, content_start:content_end]
            rows.append(content[:, -keep_length:])
        truncated_batch[key] = _sft_pad_and_concat(rows, pad_value=pad_value, pad_side="left")
    return truncated_batch


class LazyVideoSFTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: Sequence[Dict[str, Any]],
        *,
        config: Any = None,
        proposal_runtime: Any = None,
        strict_feature_guided_proposal: bool = False,
    ):
        self.examples = list(examples)
        self.config = copy.deepcopy(config) if config is not None else None
        self.proposal_runtime = proposal_runtime
        self.strict_feature_guided_proposal = bool(strict_feature_guided_proposal)

    def __len__(self) -> int:
        return len(self.examples)

    def _replay_messages(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        return replay_compact_trace_messages(
            row,
            config=self.config,
            proposal_runtime=self.proposal_runtime,
            strict_feature_guided_proposal=self.strict_feature_guided_proposal,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.examples[idx]
        messages = self._replay_messages(row)
        return {
            "messages": messages,
            "video_id": row.get("video_id"),
            "split": row.get("split"),
        }


class BatchEpisodeSFTCollator:
    def __init__(
        self,
        processor: Any,
        *,
        max_image_side: int = 0,
        max_image_pixels: int = 0,
        keep_recent_tool_image_messages: int = 0,
        max_total_images: int = 0,
        max_seq_length: int = 0,
        keep_recent_text_messages: int = 0,
    ):
        self.processor = processor
        self.max_image_side = int(max_image_side)
        self.max_image_pixels = int(max_image_pixels)
        self.keep_recent_tool_image_messages = int(keep_recent_tool_image_messages)
        self.max_total_images = int(max_total_images)
        self.max_seq_length = int(max_seq_length)
        self.keep_recent_text_messages = int(keep_recent_text_messages)

    def _prepare_feature_messages(self, feature: Dict[str, Any]) -> List[Dict[str, Any]]:
        tagged_messages = _tag_messages_for_cache(list(feature.get("messages") or []))
        return _prepare_messages(
            tagged_messages,
            max_image_side=self.max_image_side,
            max_image_pixels=self.max_image_pixels,
            keep_recent_tool_image_messages=self.keep_recent_tool_image_messages,
            max_total_images=self.max_total_images,
            keep_recent_text_messages=self.keep_recent_text_messages,
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("BatchEpisodeSFTCollator requires at least one feature.")
        prepared_messages = [self._prepare_feature_messages(feature) for feature in features]
        fitted_messages, full_texts = _fit_batch_episode_messages_to_budget(
            self.processor,
            prepared_messages,
            max_seq_length=self.max_seq_length,
            truncation_side="left",
        )
        multimodal_flags = [_has_multimodal_content(messages) for messages in fitted_messages]
        batch = _tokenize_chat_batch(
            self.processor,
            full_texts,
            fitted_messages,
            max_length=0,
            truncation_side="left",
        )
        batch = _truncate_text_only_batch_to_budget(
            batch,
            processor=self.processor,
            multimodal_flags=multimodal_flags,
            max_seq_length=self.max_seq_length,
        )
        labels = _build_plain_sft_labels(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            processor=self.processor,
        )
        zero_supervision_rows = [
            str(features[index].get("video_id") or f"index_{index}")
            for index in range(labels.shape[0])
            if not bool(torch.any(labels[index].ne(-100)))
        ]
        if zero_supervision_rows:
            raise ValueError(
                "Plain SFT batch has zero assistant supervision after compact-trace replay. "
                f"video_ids={zero_supervision_rows}"
            )
        result = {
            key: value
            for key, value in dict(batch).items()
            if isinstance(value, torch.Tensor)
        }
        result["labels"] = labels
        return result


def filter_examples_after_budgeting(
    *,
    processor: Any,
    examples: Sequence[Dict[str, Any]],
    max_image_side: int = 0,
    max_image_pixels: int = 0,
    keep_recent_tool_image_messages: int = 0,
    max_total_images: int = 0,
    max_seq_length: int = 0,
    keep_recent_text_messages: int = 0,
) -> Tuple[List[Dict[str, Any]], BudgetingStats]:
    resolver = _FrameReferenceResolver()
    kept_examples: List[Dict[str, Any]] = []
    stats = BudgetingStats()
    for example in examples:
        materialized_example = materialize_example_for_training(example, resolver=resolver)
        result = _normalize_batch_build_result(
            _build_batch_from_feature(
                processor,
                materialized_example,
                max_image_side=max_image_side,
                max_image_pixels=max_image_pixels,
                keep_recent_tool_image_messages=keep_recent_tool_image_messages,
                max_total_images=max_total_images,
                max_seq_length=max_seq_length,
                keep_recent_text_messages=keep_recent_text_messages,
                cached_plan=None,
            ),
            is_episode_feature=_is_episode_feature(materialized_example),
        )
        stats.record(result)
        if result.batch is not None:
            kept_examples.append(example)
    return kept_examples, stats


def validate_prepared_examples(
    examples: Sequence[Dict[str, Any]],
    *,
    materialize_images: bool = False,
    max_materialized_examples: int = 0,
    progress_every: int = 0,
) -> Dict[str, Any]:
    runtime = distributed_runtime_from_env()
    summary: Dict[str, Any] = {
        "num_examples": len(examples),
        "num_examples_with_image_refs": 0,
        "num_image_refs": 0,
        "num_inline_images": 0,
        "materialized_examples": 0,
        "num_errors": 0,
        "errors": [],
    }
    error_examples = set()
    total_examples = len(examples)

    for idx, example in enumerate(examples):
        prefix = f"example[{idx}]"
        if _is_compact_trace_sft_row(example):
            if not str(example.get("video_path") or "").strip():
                summary["errors"].append(f"{prefix}: compact_trace_v2 row is missing video_path")
                error_examples.add(idx)
            if not isinstance(example.get("oracle_trajectory"), list):
                summary["errors"].append(f"{prefix}: compact_trace_v2 row is missing oracle_trajectory")
                error_examples.add(idx)
            completed = idx + 1
            if should_log_progress(completed, total_examples, int(progress_every)):
                runtime_log(
                    (
                        "Prepared data validation progress: "
                        f"examples={completed}/{total_examples} "
                        f"errors={len(summary['errors'])}"
                    ),
                    runtime=runtime,
                    main_process_only=True,
                )
            continue

        messages = example.get("messages")
        if not isinstance(messages, list) or not messages:
            summary["errors"].append(f"{prefix}: missing non-empty messages list")
            error_examples.add(idx)
            continue

        if _is_episode_feature(example):
            assistant_supervision = example.get("assistant_supervision")
            if not isinstance(assistant_supervision, list):
                summary["errors"].append(f"{prefix}: assistant_supervision is not a list")
                error_examples.add(idx)
            else:
                for supervision_idx, entry in enumerate(assistant_supervision):
                    try:
                        assistant_message_index = int(entry.get("assistant_message_index"))
                    except Exception:
                        summary["errors"].append(
                            f"{prefix}: assistant_supervision[{supervision_idx}] is missing a valid assistant_message_index"
                        )
                        error_examples.add(idx)
                        continue
                    if not 0 <= assistant_message_index < len(messages):
                        summary["errors"].append(
                            f"{prefix}: assistant_supervision[{supervision_idx}] references message index {assistant_message_index} "
                            f"outside [0, {len(messages) - 1}]"
                        )
                        error_examples.add(idx)
                        continue
                    if str(messages[assistant_message_index].get("role") or "") != "assistant":
                        summary["errors"].append(
                            f"{prefix}: assistant_supervision[{supervision_idx}] does not point to an assistant message"
                        )
                        error_examples.add(idx)
        else:
            target_response = example.get("target_response")
            if not isinstance(target_response, str) or not target_response.strip():
                summary["errors"].append(f"{prefix}: missing non-empty target_response")
                error_examples.add(idx)

        example_image_ref_count = 0
        for message_idx, message in enumerate(messages):
            content = message.get("content")
            if not isinstance(content, list):
                summary["errors"].append(f"{prefix}: message[{message_idx}] content is not a list")
                error_examples.add(idx)
                continue
            for content_idx, item in enumerate(content):
                if item.get("type") != "image":
                    continue
                item_prefix = f"{prefix}: message[{message_idx}].content[{content_idx}]"
                if "image_ref" in item:
                    example_image_ref_count += 1
                    summary["num_image_refs"] += 1
                    image_ref = item.get("image_ref") or {}
                    if not str(image_ref.get("video_path") or "").strip():
                        summary["errors"].append(f"{item_prefix}: image_ref.video_path is missing")
                        error_examples.add(idx)
                    if all(
                        image_ref.get(key) is None
                        for key in ("sampled_frame_index", "raw_frame_index", "timestamp_sec")
                    ):
                        summary["errors"].append(
                            f"{item_prefix}: image_ref needs sampled_frame_index, raw_frame_index, or timestamp_sec"
                        )
                        error_examples.add(idx)
                elif "image" in item:
                    summary["num_inline_images"] += 1
                else:
                    summary["errors"].append(f"{item_prefix}: image item is missing both image and image_ref")
                    error_examples.add(idx)
        if example_image_ref_count > 0:
            summary["num_examples_with_image_refs"] += 1
        completed = idx + 1
        if should_log_progress(completed, total_examples, int(progress_every)):
            runtime_log(
                (
                    "Prepared data validation progress: "
                    f"examples={completed}/{total_examples} "
                    f"image_refs={summary['num_image_refs']} errors={len(summary['errors'])}"
                ),
                runtime=runtime,
                main_process_only=True,
            )

    if materialize_images:
        inspect_count = len(examples) if max_materialized_examples <= 0 else min(len(examples), int(max_materialized_examples))
        dataset = WeightedExampleDataset(list(examples[:inspect_count]))
        for idx in range(inspect_count):
            try:
                dataset[idx]
            except Exception as exc:
                summary["errors"].append(f"example[{idx}]: failed to materialize image_ref payloads: {exc}")
                error_examples.add(idx)
            completed = idx + 1
            if should_log_progress(completed, inspect_count, int(progress_every)):
                runtime_log(
                    (
                        "Prepared data materialization progress: "
                        f"examples={completed}/{inspect_count} errors={len(summary['errors'])}"
                    ),
                    runtime=runtime,
                    main_process_only=True,
                )
        summary["materialized_examples"] = inspect_count

    summary["num_errors"] = len(summary["errors"])
    summary["num_invalid_examples"] = len(error_examples)
    return summary


def _shift_logits_and_labels(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    response_mask = shift_labels.ne(-100)
    return shift_logits, shift_labels, response_mask


def compute_masked_response_token_log_probs(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    shift_logits, shift_labels, response_mask = _shift_logits_and_labels(logits, labels)
    if not torch.any(response_mask):
        return shift_logits.new_zeros(shift_labels.shape), response_mask

    log_probs = F.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.masked_fill(~response_mask, 0)
    token_log_probs = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(~response_mask, 0.0)
    return token_log_probs, response_mask


def compute_masked_response_nll(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    shift_logits, shift_labels, response_mask = _shift_logits_and_labels(logits, labels)
    if not torch.any(response_mask):
        return logits.new_zeros(())

    log_probs = F.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.masked_fill(~response_mask, 0)
    token_nll = -torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    masked_nll = token_nll.masked_select(response_mask)
    return masked_nll.mean() if masked_nll.numel() else logits.new_zeros(())


def compute_weighted_masked_response_nll(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    token_weights: torch.Tensor,
) -> torch.Tensor:
    shift_logits, shift_labels, response_mask = _shift_logits_and_labels(logits, labels)
    if not torch.any(response_mask):
        return logits.new_zeros(())

    if token_weights.shape != labels.shape:
        raise ValueError(
            f"token_weights must match labels shape for weighted SFT: got {tuple(token_weights.shape)} vs {tuple(labels.shape)}"
        )

    log_probs = F.log_softmax(shift_logits, dim=-1)
    safe_labels = shift_labels.masked_fill(~response_mask, 0)
    target_log_probs = torch.gather(log_probs, dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    token_nll = -target_log_probs

    shifted_weights = token_weights[..., 1:].to(token_nll.device, dtype=torch.float32)
    masked_nll = token_nll.masked_select(response_mask)
    masked_target_log_probs = target_log_probs.masked_select(response_mask)
    masked_weights = shifted_weights.masked_select(response_mask)
    if masked_nll.numel() == 0:
        return logits.new_zeros(())

    positive_weights = masked_weights.clamp_min(0.0)
    negative_weights = (-masked_weights).clamp_min(0.0)
    safe_negative_log_probs = masked_target_log_probs.clamp(max=-1e-6)
    suppression_loss = -torch.log(-torch.expm1(safe_negative_log_probs))
    weighted_loss = (masked_nll * positive_weights).sum() + (suppression_loss * negative_weights).sum()
    normalizer = positive_weights.sum() + negative_weights.sum()
    if float(normalizer.detach().item()) <= 1e-8:
        return logits.new_zeros(())
    return weighted_loss / normalizer


def compute_signed_weighted_masked_response_loss(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    sample_weights = sample_weights.to(logits.device, dtype=torch.float32).view(-1)
    token_log_probs, response_mask = compute_masked_response_token_log_probs(logits=logits, labels=labels)
    if token_log_probs.shape[0] != sample_weights.shape[0]:
        raise ValueError(
            "sample_weights batch shape must match logits batch shape for weighted SFT: "
            f"got {tuple(sample_weights.shape)} vs {tuple(token_log_probs.shape[:1])}"
        )
    token_counts = response_mask.sum(dim=-1)
    valid_examples = token_counts > 0
    if not torch.any(valid_examples):
        return logits.new_zeros(())

    average_log_probs = logits.new_zeros((token_log_probs.shape[0],), dtype=torch.float32)
    average_log_probs[valid_examples] = (
        token_log_probs.sum(dim=-1)[valid_examples] / token_counts[valid_examples].to(torch.float32)
    )
    effective_weights = sample_weights.clone()
    effective_weights[~valid_examples] = 0.0
    positive_weights = effective_weights.clamp_min(0.0)
    negative_weights = (-effective_weights).clamp_min(0.0)
    response_nll = -average_log_probs
    safe_negative_log_probs = average_log_probs.clamp(max=-1e-6)
    suppression_loss = -torch.log(-torch.expm1(safe_negative_log_probs))
    weighted_loss = (response_nll * positive_weights).sum() + (suppression_loss * negative_weights).sum()
    normalizer = positive_weights.sum() + negative_weights.sum()
    if float(normalizer.detach().item()) <= 1e-8:
        return logits.new_zeros(())
    return weighted_loss / normalizer


def compute_masked_response_log_probs(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    batch_size = int(logits.shape[0]) if logits.ndim > 0 else 1
    token_log_probs, response_mask = compute_masked_response_token_log_probs(logits=logits, labels=labels)
    if not torch.any(response_mask):
        return logits.new_zeros((batch_size,))
    token_counts = response_mask.sum(dim=-1).clamp(min=1)
    return token_log_probs.sum(dim=-1) / token_counts


def compute_completion_only_token_log_probs_from_ids(
    *,
    model: Any,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    multimodal_inputs: Optional[Dict[str, Any]] = None,
    temperature: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if prompt_ids.ndim != 2 or prompt_mask.ndim != 2 or completion_ids.ndim != 2 or completion_mask.ndim != 2:
        raise ValueError("completion-only token log-prob computation expects rank-2 prompt/completion tensors.")

    response_mask = completion_mask.to(dtype=torch.bool)
    empty_token_log_probs = completion_ids.new_zeros(completion_ids.shape, dtype=torch.float32)
    if not torch.any(response_mask):
        return empty_token_log_probs, response_mask
    prompt_rows_with_context = prompt_mask.to(dtype=torch.bool).any(dim=-1)
    if not bool(torch.all(prompt_rows_with_context)):
        raise ValueError(
            "completion-only forward requires at least one retained prompt token in every batch row."
        )

    if not _model_supports_logits_to_keep(model):
        raise RuntimeError(
            "completion-only forward requires the current model forward() to accept logits_to_keep."
        )

    input_ids = torch.cat([prompt_ids, completion_ids], dim=-1)
    attention_mask = torch.cat([prompt_mask, completion_mask.to(dtype=prompt_mask.dtype)], dim=-1)
    model_inputs = dict(multimodal_inputs or {})
    model_inputs.update(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )
    logits_to_keep = int(completion_ids.shape[-1]) + 1
    outputs = model(**model_inputs, logits_to_keep=logits_to_keep)
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("completion-only forward expected model outputs to expose `.logits`.")
    if int(logits.shape[-2]) < logits_to_keep:
        raise RuntimeError(
            "completion-only forward expected the model to honor logits_to_keep="
            f"{logits_to_keep}, but received logits with seq_len={int(logits.shape[-2])}."
        )
    logits = logits[:, -logits_to_keep:, :]
    shift_logits = logits[:, :-1, :]
    if int(shift_logits.shape[-2]) != int(completion_ids.shape[-1]):
        raise RuntimeError("completion-only forward produced logits that do not align with completion_ids.")

    temperature_value = 1.0
    if temperature is not None:
        try:
            parsed_temperature = float(temperature)
        except Exception:
            parsed_temperature = 1.0
        if parsed_temperature > 0.0:
            temperature_value = parsed_temperature
    shift_logits = shift_logits / float(temperature_value)
    log_probs = F.log_softmax(shift_logits, dim=-1)
    safe_completion_ids = completion_ids.masked_fill(~response_mask, 0)
    token_log_probs = torch.gather(log_probs, dim=-1, index=safe_completion_ids.unsqueeze(-1)).squeeze(-1)
    token_log_probs = token_log_probs.masked_fill(~response_mask, 0.0)
    return token_log_probs, response_mask


def compute_completion_only_log_probs_from_ids(
    *,
    model: Any,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    multimodal_inputs: Optional[Dict[str, Any]] = None,
    temperature: Optional[float] = None,
) -> torch.Tensor:
    token_log_probs, response_mask = compute_completion_only_token_log_probs_from_ids(
        model=model,
        prompt_ids=prompt_ids,
        prompt_mask=prompt_mask,
        completion_ids=completion_ids,
        completion_mask=completion_mask,
        multimodal_inputs=multimodal_inputs,
        temperature=temperature,
    )
    if not torch.any(response_mask):
        return token_log_probs.new_zeros((int(completion_ids.shape[0]),), dtype=torch.float32)
    token_counts = response_mask.sum(dim=-1).clamp(min=1)
    return token_log_probs.sum(dim=-1) / token_counts.to(torch.float32)


def _model_supports_logits_to_keep(model: Any) -> bool:
    forward = getattr(model, "forward", None)
    if not callable(forward):
        return False
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == "logits_to_keep":
            return True
    return False


def _response_suffix_plan(labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if labels.ndim != 2:
        raise ValueError(f"labels must be rank-2 for completion-only forward, got shape={tuple(labels.shape)}")
    shifted_response_mask = labels[..., 1:].ne(-100)
    response_token_counts = shifted_response_mask.sum(dim=-1)
    shifted_seq_len = int(shifted_response_mask.shape[-1])
    for row_index in range(int(shifted_response_mask.shape[0])):
        count = int(response_token_counts[row_index].item())
        if count <= 0:
            continue
        expected_mask = torch.zeros_like(shifted_response_mask[row_index], dtype=torch.bool)
        expected_mask[-count:] = True
        if not torch.equal(shifted_response_mask[row_index], expected_mask):
            raise ValueError(
                "completion-only forward requires response labels to form a contiguous suffix in every batch row."
            )
    return shifted_response_mask, response_token_counts, int(response_token_counts.max().item())


def compute_completion_only_response_token_log_probs(
    *,
    model: Any,
    model_inputs: Dict[str, Any],
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    full_response_mask, response_token_counts, max_response_tokens = _response_suffix_plan(labels)
    batch_size = int(labels.shape[0])
    shifted_seq_len = int(labels.shape[-1]) - 1
    empty_token_log_probs = labels.new_zeros((batch_size, shifted_seq_len), dtype=torch.float32)
    if max_response_tokens <= 0:
        return empty_token_log_probs, full_response_mask

    if not _model_supports_logits_to_keep(model):
        raise RuntimeError(
            "completion-only forward requires the current model forward() to accept logits_to_keep."
        )

    logits_to_keep = min(int(labels.shape[-1]), max_response_tokens + 1)
    outputs = model(**model_inputs, logits_to_keep=logits_to_keep)
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise RuntimeError("completion-only forward expected model outputs to expose `.logits`.")
    if int(logits.shape[-2]) != logits_to_keep:
        raise RuntimeError(
            "completion-only forward expected the model to honor logits_to_keep="
            f"{logits_to_keep}, but received logits with seq_len={int(logits.shape[-2])}."
        )

    cropped_labels = labels[:, -logits_to_keep:]
    cropped_token_log_probs, cropped_response_mask = compute_masked_response_token_log_probs(
        logits=logits,
        labels=cropped_labels,
    )
    token_log_probs = logits.new_zeros((batch_size, shifted_seq_len), dtype=torch.float32)
    token_log_probs[:, -cropped_token_log_probs.shape[-1] :] = cropped_token_log_probs.to(torch.float32)
    reconstructed_response_mask = torch.zeros(
        (batch_size, shifted_seq_len),
        device=token_log_probs.device,
        dtype=torch.bool,
    )
    reconstructed_response_mask[:, -cropped_response_mask.shape[-1] :] = cropped_response_mask.to(
        device=reconstructed_response_mask.device
    )
    expected_response_mask = full_response_mask.to(device=token_log_probs.device)
    if not torch.equal(reconstructed_response_mask, expected_response_mask):
        raise RuntimeError("completion-only forward produced a response mask that does not match the full labels.")
    return token_log_probs, expected_response_mask


def compute_completion_only_response_log_probs(
    *,
    model: Any,
    model_inputs: Dict[str, Any],
    labels: torch.Tensor,
) -> torch.Tensor:
    token_log_probs, response_mask = compute_completion_only_response_token_log_probs(
        model=model,
        model_inputs=model_inputs,
        labels=labels,
    )
    if not torch.any(response_mask):
        return token_log_probs.new_zeros((int(labels.shape[0]),), dtype=torch.float32)
    token_counts = response_mask.sum(dim=-1).clamp(min=1)
    return token_log_probs.sum(dim=-1) / token_counts.to(torch.float32)


def _zero_loss_from_model(model: Any) -> torch.Tensor:
    try:
        zero_loss = None
        first_parameter = None
        for parameter in model.parameters():
            if first_parameter is None:
                first_parameter = parameter
            if bool(getattr(parameter, "requires_grad", False)):
                contribution = parameter.reshape(-1)[:1].sum() * 0.0
                zero_loss = contribution if zero_loss is None else zero_loss + contribution
        if zero_loss is not None:
            return zero_loss
        if first_parameter is not None:
            return first_parameter.sum() * 0.0
    except StopIteration:
        pass
    except Exception:
        pass
    try:
        first_buffer = next(model.buffers())
        return first_buffer.new_zeros(())
    except Exception:
        return torch.tensor(0.0)


def compute_masked_sampled_token_kl(
    *,
    policy_token_log_probs: torch.Tensor,
    reference_token_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    if not torch.any(response_mask):
        return policy_token_log_probs.new_zeros(())
    masked_policy = policy_token_log_probs.masked_select(response_mask)
    masked_reference = reference_token_log_probs.to(policy_token_log_probs.device).masked_select(response_mask)
    if masked_policy.numel() == 0:
        return policy_token_log_probs.new_zeros(())
    delta = masked_reference - masked_policy
    token_kl = torch.exp(delta) - delta - 1.0
    return token_kl.mean()


def compute_masked_forward_kl(
    *,
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    policy_shift_logits, shift_labels, response_mask = _shift_logits_and_labels(policy_logits, labels)
    reference_shift_logits = reference_logits[..., :-1, :].contiguous()
    if not torch.any(response_mask):
        return policy_logits.new_zeros(())

    policy_log_probs = F.log_softmax(policy_shift_logits, dim=-1)
    reference_log_probs = F.log_softmax(reference_shift_logits, dim=-1)
    reference_probs = reference_log_probs.exp()
    token_kl = torch.sum(reference_probs * (reference_log_probs - policy_log_probs), dim=-1)
    masked_kl = token_kl.masked_select(response_mask)
    return masked_kl.mean() if masked_kl.numel() else policy_logits.new_zeros(())


def compute_grpo_surrogate_loss(
    *,
    policy_log_probs: torch.Tensor,
    old_policy_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor:
    policy_log_probs = policy_log_probs.view(-1)
    old_policy_log_probs = old_policy_log_probs.to(policy_log_probs.device).view(-1)
    advantages = advantages.to(policy_log_probs.device).view(-1)
    if (
        policy_log_probs.numel() != old_policy_log_probs.numel()
        or policy_log_probs.numel() != advantages.numel()
    ):
        raise ValueError("policy_log_probs, old_policy_log_probs, and advantages must have matching lengths.")

    ratios = torch.exp(policy_log_probs - old_policy_log_probs)
    clipped_ratios = torch.clamp(ratios, 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon))
    surrogate_unclipped = ratios * advantages
    surrogate_clipped = clipped_ratios * advantages
    return -torch.minimum(surrogate_unclipped, surrogate_clipped).mean()


def _assign_weight_range(char_weights: List[float], start: int, end: int, weight: float) -> None:
    start = max(0, int(start))
    end = min(len(char_weights), int(end))
    for idx in range(start, end):
        char_weights[idx] = float(weight)


def _find_tag_bounds(text: str, tag_name: str) -> Optional[Tuple[int, int, int, int]]:
    open_tag = f"<{tag_name}>"
    close_tag = f"</{tag_name}>"
    open_idx = text.find(open_tag)
    close_idx = text.find(close_tag)
    if open_idx < 0 or close_idx < 0 or close_idx < open_idx:
        return None
    content_start = open_idx + len(open_tag)
    return open_idx, content_start, close_idx, close_idx + len(close_tag)


def _annotate_json_like_span_weights(
    text: str,
    char_weights: List[float],
    *,
    base_offset: int = 0,
) -> None:
    critical_keys = {
        "existence",
        "category",
        "severity",
        "anomaly_interval_sec",
        "precursor_interval_sec",
        "earliest_actionable_sec",
        "earliest_alert_sec",
        "decision",
        "alert_sec",
        "start_sec",
        "end_sec",
        "window_id",
        "evidence_id",
        "evidence_moment_ids",
        "counterfactual_type",
        "name",
        "arguments",
    }
    punctuation_weight = 0.2
    key_weight = 0.8
    value_weight = 1.15
    critical_value_weight = 1.35
    string_weight = 1.0

    idx = 0
    expecting_key = False
    current_key = ""
    stack: List[str] = []
    while idx < len(text):
        ch = text[idx]
        global_idx = base_offset + idx
        if ch in "{[":
            _assign_weight_range(char_weights, global_idx, global_idx + 1, punctuation_weight)
            stack.append(ch)
            expecting_key = ch == "{"
            idx += 1
            continue
        if ch in "}]":
            _assign_weight_range(char_weights, global_idx, global_idx + 1, punctuation_weight)
            if stack:
                stack.pop()
            expecting_key = bool(stack and stack[-1] == "{")
            current_key = ""
            idx += 1
            continue
        if ch in ",:":
            _assign_weight_range(char_weights, global_idx, global_idx + 1, punctuation_weight)
            if ch == ",":
                expecting_key = bool(stack and stack[-1] == "{")
                current_key = ""
            else:
                expecting_key = False
            idx += 1
            continue
        if ch.isspace():
            _assign_weight_range(char_weights, global_idx, global_idx + 1, punctuation_weight)
            idx += 1
            continue
        if ch == '"':
            end_idx = idx + 1
            escaped = False
            while end_idx < len(text):
                current = text[end_idx]
                if current == '"' and not escaped:
                    end_idx += 1
                    break
                escaped = current == "\\" and not escaped
                if current != "\\":
                    escaped = False
                end_idx += 1
            literal = text[idx + 1 : max(idx + 1, end_idx - 1)]
            if expecting_key:
                _assign_weight_range(char_weights, global_idx, base_offset + end_idx, key_weight)
                current_key = literal
            else:
                is_critical = current_key in critical_keys
                _assign_weight_range(
                    char_weights,
                    global_idx,
                    base_offset + end_idx,
                    critical_value_weight if is_critical else value_weight,
                )
            idx = end_idx
            continue

        end_idx = idx + 1
        while end_idx < len(text) and text[end_idx] not in ',:{}[] \t\r\n':
            end_idx += 1
        token = text[idx:end_idx]
        is_value = not expecting_key
        weight = string_weight
        if is_value:
            weight = critical_value_weight if current_key in critical_keys else value_weight
        _assign_weight_range(char_weights, global_idx, base_offset + end_idx, weight)
        idx = end_idx


def _build_response_char_weights(
    *,
    response_text: str,
    target_action: Optional[str] = None,
    tool_name: Optional[str] = None,
) -> List[float]:
    if not response_text:
        return []
    char_weights = [1.0 for _ in response_text]

    for tag_name in ("think", "tool_call", "answer"):
        bounds = _find_tag_bounds(response_text, tag_name)
        if bounds is None:
            continue
        open_start, content_start, content_end, close_end = bounds
        _assign_weight_range(char_weights, open_start, content_start, 0.15)
        _assign_weight_range(char_weights, content_end, close_end, 0.15)
        if tag_name == "think":
            _assign_weight_range(char_weights, content_start, content_end, 0.35)
        else:
            _assign_weight_range(char_weights, content_start, content_end, 1.0)
            _annotate_json_like_span_weights(
                response_text[content_start:content_end],
                char_weights,
                base_offset=content_start,
            )

    if str(target_action or "") == "answer" and "<answer>" not in response_text:
        _annotate_json_like_span_weights(response_text, char_weights, base_offset=0)
    elif str(target_action or "") == "tool_call" and str(tool_name or "") and "<tool_call>" not in response_text:
        _annotate_json_like_span_weights(response_text, char_weights, base_offset=0)
    return char_weights


def _apply_focus_term_weights(
    response_text: str,
    char_weights: List[float],
    *,
    focus_terms: Sequence[str],
    boost_weight: float,
) -> None:
    lowered_text = response_text.lower()
    for term in focus_terms:
        lowered_term = str(term or "").strip().lower()
        if not lowered_term:
            continue
        start = 0
        while True:
            index = lowered_text.find(lowered_term, start)
            if index < 0:
                break
            _assign_weight_range(char_weights, index, index + len(lowered_term), boost_weight)
            start = index + len(lowered_term)


def _build_component_response_char_weights(
    *,
    response_text: str,
    component_name: str,
    target_action: Optional[str] = None,
    tool_name: Optional[str] = None,
) -> List[float]:
    char_weights = _build_response_char_weights(
        response_text=response_text,
        target_action=target_action,
        tool_name=tool_name,
    )
    if not char_weights or component_name == "global":
        return char_weights

    focus_terms_by_component = {
        "search_local": [
            "scan_timeline",
            "seek_evidence",
            "start_sec",
            "end_sec",
            "query",
            "purpose",
            "num_frames",
            "top_k",
            "window",
        ],
        "evidence_local": [
            "candidate_window_ids",
            "candidate_evidence_ids",
            "candidate_evidence_moment_ids",
            "selected_window_ids",
            "selected_evidence_ids",
            "evidence_moment_ids",
            "window_id",
            "evidence_id",
        ],
        "query_local": [
            "query",
            "purpose",
            "description",
            "scene_anchor",
        ],
        "stage_local": [
            "covered_stages",
            "missing_required_stages",
            "stage_selected_moment_ids",
            "selected_evidence_moment_ids",
            "precursor",
            "trigger",
            "confirmation",
            "aftermath",
            "verification_decision",
            "recommended_action",
            "finalize_readiness_score",
            "continue_search",
            "finalize",
        ],
        "teacher_local": [
            "verification_decision",
            "recommended_action",
            "sufficiency_score",
            "necessity_score",
            "finalize_readiness_score",
            "counterfactual_faithfulness",
            "selected_window_ids",
            "selected_evidence_ids",
            "selected_evidence_moment_ids",
            "rationale",
            "continue_search",
            "revise_claim",
            "refine_evidence",
            "finalize",
        ],
    }
    _apply_focus_term_weights(
        response_text,
        char_weights,
        focus_terms=focus_terms_by_component.get(component_name, []),
        boost_weight=2.5,
    )
    return char_weights


def _token_weights_from_char_weights(
    char_weights: Sequence[float],
    offsets: Sequence[Tuple[int, int]],
) -> List[float]:
    token_weights: List[float] = []
    for start, end in offsets:
        start = max(0, int(start))
        end = min(len(char_weights), int(end))
        if end <= start:
            token_weights.append(1.0)
            continue
        token_weights.append(sum(float(value) for value in char_weights[start:end]) / float(end - start))
    return token_weights


def build_token_advantages_from_offsets(
    *,
    response_text: str,
    offsets: Sequence[Tuple[int, int]],
    base_advantage: float,
    target_action: Optional[str] = None,
    tool_name: Optional[str] = None,
    advantage_components: Optional[Dict[str, float]] = None,
    turn_component_weights: Optional[Dict[str, float]] = None,
) -> List[float]:
    if not offsets:
        return []
    component_pairs: List[Tuple[str, float]] = []
    if advantage_components:
        for component_name in ("global", "search_local", "evidence_local", "teacher_local", "query_local", "stage_local"):
            component_advantage = float(advantage_components.get(component_name, 0.0) or 0.0)
            component_weight = 1.0
            if turn_component_weights is not None:
                component_weight = float(turn_component_weights.get(component_name, 0.0) or 0.0)
            scaled_advantage = component_advantage * component_weight
            if abs(scaled_advantage) > 1e-8:
                component_pairs.append((component_name, scaled_advantage))

    if component_pairs:
        combined_advantages = [0.0 for _ in offsets]
        for component_name, component_advantage in component_pairs:
            char_weights = _build_component_response_char_weights(
                response_text=response_text,
                component_name=component_name,
                target_action=target_action,
                tool_name=tool_name,
            )
            if not char_weights:
                char_weights = [1.0 for _ in response_text]
            token_weights = _token_weights_from_char_weights(char_weights, offsets)
            mean_weight = sum(token_weights) / float(len(token_weights)) if token_weights else 1.0
            if mean_weight <= 1e-8:
                mean_weight = 1.0
            for index, weight in enumerate(token_weights):
                combined_advantages[index] += float(component_advantage) * (weight / mean_weight)
        return combined_advantages

    char_weights = _build_response_char_weights(
        response_text=response_text,
        target_action=target_action,
        tool_name=tool_name,
    )
    if not char_weights:
        return [float(base_advantage) for _ in offsets]

    token_weights = _token_weights_from_char_weights(char_weights, offsets)
    mean_weight = sum(token_weights) / float(len(token_weights)) if token_weights else 1.0
    if mean_weight <= 1e-8:
        mean_weight = 1.0
    return [float(base_advantage) * (weight / mean_weight) for weight in token_weights]


def _extract_response_offsets(processor: Any, response_text: str) -> List[Tuple[int, int]]:
    tokenizer = getattr(processor, "tokenizer", None) or processor
    try:
        encoded = tokenizer(
            response_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping") if isinstance(encoded, dict) else None
        if offsets is None:
            return []
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        return [tuple(map(int, offset)) for offset in offsets]
    except Exception:
        return []


def _align_token_advantages(
    token_advantages: Sequence[float],
    *,
    response_token_count: int,
    base_advantage: float,
) -> List[float]:
    if response_token_count <= 0:
        return []
    values = [float(value) for value in token_advantages]
    if not values:
        return [float(base_advantage) for _ in range(response_token_count)]
    if len(values) == response_token_count:
        return values
    if len(values) < response_token_count:
        pad_value = values[-1] if values else float(base_advantage)
        return values + [pad_value for _ in range(response_token_count - len(values))]
    return values[:response_token_count]


def _build_token_advantages_for_feature(
    *,
    processor: Any,
    feature: Dict[str, Any],
    response_token_count: int,
) -> List[float]:
    base_advantage = float(feature.get("advantage", feature.get("sample_weight", 1.0)) or 0.0)
    response_text = str(feature.get("target_response") or "")
    if not response_text:
        return [base_advantage for _ in range(response_token_count)]
    offsets = _extract_response_offsets(processor, response_text)
    token_advantages = build_token_advantages_from_offsets(
        response_text=response_text,
        offsets=offsets,
        base_advantage=base_advantage,
        target_action=feature.get("target_action"),
        tool_name=feature.get("tool_name"),
        advantage_components=feature.get("advantage_components"),
        turn_component_weights=feature.get("turn_component_weights"),
    )
    return _align_token_advantages(
        token_advantages,
        response_token_count=response_token_count,
        base_advantage=base_advantage,
    )


def load_qwen_model_and_processor(
    model_path: str | Path,
    *,
    torch_dtype: str = "auto",
    attn_implementation: Optional[str] = None,
    gradient_checkpointing: bool = False,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[Sequence[str]] = None,
) -> Tuple[Any, Any]:
    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    except Exception as exc:
        raise ImportError(
            "Training requires a recent transformers build with Qwen3-VL support. "
            "Install it with `pip install git+https://github.com/huggingface/transformers accelerate`."
        ) from exc

    model_init_kwargs = build_hf_model_init_kwargs(
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )

    resolved_model_path = Path(model_path)
    adapter_config_path = resolved_model_path / "adapter_config.json"
    base_model_path = str(resolved_model_path)
    processor_path = str(resolved_model_path)
    loaded_from_adapter_checkpoint = bool(adapter_config_path.exists())
    if loaded_from_adapter_checkpoint:
        try:
            from peft import PeftConfig, PeftModel
        except Exception as exc:
            raise ImportError("Loading LoRA adapter checkpoints requires `peft` to be installed.") from exc
        peft_config = PeftConfig.from_pretrained(str(resolved_model_path))
        base_model_path = str(peft_config.base_model_name_or_path)
        model = Qwen3VLForConditionalGeneration.from_pretrained(base_model_path, **model_init_kwargs)
        try:
            model = PeftModel.from_pretrained(model, str(resolved_model_path), is_trainable=bool(use_lora))
        except TypeError:
            model = PeftModel.from_pretrained(model, str(resolved_model_path))
        if not any((resolved_model_path / filename).exists() for filename in ("preprocessor_config.json", "processor_config.json", "tokenizer_config.json", "tokenizer.json")):
            processor_path = base_model_path
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(str(resolved_model_path), **model_init_kwargs)
    ensure_flash_attention_supported_dtype(
        model,
        attn_implementation=attn_implementation,
    )

    processor = AutoProcessor.from_pretrained(processor_path)
    if hasattr(model.config, "use_cache"):
        # KV cache is useful for autoregressive decoding, not for teacher-forced SFT.
        model.config.use_cache = False
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, "use_cache"):
        generation_config.use_cache = False
    for nested_config_name in ("text_config", "language_config"):
        nested_config = getattr(model.config, nested_config_name, None)
        if nested_config is not None and hasattr(nested_config, "use_cache"):
            nested_config.use_cache = False
    if gradient_checkpointing:
        gradient_checkpointing_kwargs = {"use_reentrant": False}
        try:
            signature = inspect.signature(model.gradient_checkpointing_enable)
        except (TypeError, ValueError):
            signature = None
        if signature is not None and "gradient_checkpointing_kwargs" in signature.parameters:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        else:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            try:
                model.enable_input_require_grads()
            except Exception:
                pass

    if use_lora and not loaded_from_adapter_checkpoint:
        try:
            from peft import LoraConfig, get_peft_model
        except Exception as exc:
            raise ImportError("LoRA training requires `peft` to be installed.") from exc
        peft_config = LoraConfig(
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=list(lora_target_modules or DEFAULT_LORA_TARGET_MODULES),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    return model, processor


def _iter_model_use_cache_owners(model: Any):
    seen_owner_ids = set()
    model_objects = [model]
    unwrapped_model = _unwrap_model(model)
    if unwrapped_model is not model:
        model_objects.append(unwrapped_model)
    for model_object in model_objects:
        config = getattr(model_object, "config", None)
        for owner in (
            config,
            getattr(model_object, "generation_config", None),
            getattr(config, "text_config", None) if config is not None else None,
            getattr(config, "language_config", None) if config is not None else None,
        ):
            if owner is None or not hasattr(owner, "use_cache"):
                continue
            owner_id = id(owner)
            if owner_id in seen_owner_ids:
                continue
            seen_owner_ids.add(owner_id)
            yield owner


def _set_model_use_cache(model: Any, enabled: bool) -> None:
    for owner in _iter_model_use_cache_owners(model):
        try:
            setattr(owner, "use_cache", bool(enabled))
        except Exception:
            continue


@contextmanager
def _temporary_model_use_cache(model: Any, enabled: bool):
    tracked_values: List[Tuple[Any, Any]] = []
    for owner in _iter_model_use_cache_owners(model):
        tracked_values.append((owner, getattr(owner, "use_cache")))
        try:
            setattr(owner, "use_cache", bool(enabled))
        except Exception:
            continue
    try:
        yield
    finally:
        for owner, previous_value in tracked_values:
            try:
                setattr(owner, "use_cache", previous_value)
            except Exception:
                continue


def create_trainer(
    *,
    model: Any,
    processor: Any,
    train_dataset: torch.utils.data.Dataset,
    output_dir: str | Path,
    learning_rate: float,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    logging_steps: int,
    save_steps: int,
    save_total_limit: int,
    warmup_ratio: float,
    weight_decay: float,
    max_grad_norm: float,
    bf16: bool,
    fp16: bool,
    max_image_side: int = 0,
    max_image_pixels: int = 0,
    keep_recent_tool_image_messages: int = 0,
    max_total_images: int = 0,
    max_seq_length: int = 0,
    keep_recent_text_messages: int = 0,
    dataloader_num_workers: int = 0,
    dataloader_prefetch_factor: int = 0,
    dataloader_persistent_workers: bool = False,
    training_objective: str = "weighted_sft",
    old_policy_model: Optional[Any] = None,
    ppo_clip_epsilon: float = 0.2,
    reference_model: Optional[Any] = None,
    kl_beta: float = 0.0,
    callbacks: Optional[Sequence[Any]] = None,
) -> Any:
    try:
        from transformers import Trainer, TrainingArguments
    except Exception as exc:
        raise ImportError("Training requires the `transformers` package.") from exc

    class WeightedLossTrainer(Trainer):
        def __init__(
            self,
            *trainer_args,
            training_objective: str = "weighted_sft",
            old_policy_model: Optional[Any] = None,
            ppo_clip_epsilon: float = 0.2,
            reference_model: Optional[Any] = None,
            kl_beta: float = 0.0,
            **trainer_kwargs,
        ):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.training_objective = str(training_objective)
            self.old_policy_model = old_policy_model
            self.ppo_clip_epsilon = float(ppo_clip_epsilon)
            self.reference_model = reference_model
            self.kl_beta = float(kl_beta)
            self._old_policy_model_device = None
            self._reference_model_device = None
            if self.old_policy_model is not None:
                self.old_policy_model.eval()
                for parameter in self.old_policy_model.parameters():
                    parameter.requires_grad_(False)
            if self.reference_model is not None:
                self.reference_model.eval()
                for parameter in self.reference_model.parameters():
                    parameter.requires_grad_(False)

        def _ensure_aux_model_device(self, aux_model: Any, current_device: Any, device_attr_name: str) -> None:
            if aux_model is None:
                return
            if getattr(self, device_attr_name) != current_device:
                aux_model.to(current_device)
                aux_model.eval()
                setattr(self, device_attr_name, current_device)

        def _ensure_old_policy_model_device(self, model: Any) -> None:
            if self.old_policy_model is None:
                return
            try:
                target_device = next(model.parameters()).device
            except StopIteration:
                return
            self._ensure_aux_model_device(self.old_policy_model, target_device, "_old_policy_model_device")

        def _ensure_reference_model_device(self, model: Any) -> None:
            if self.reference_model is None:
                return
            try:
                target_device = next(model.parameters()).device
            except StopIteration:
                return
            self._ensure_aux_model_device(self.reference_model, target_device, "_reference_model_device")

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            sample_weight = inputs.pop("sample_weight", None)
            advantage = inputs.pop("advantage", None)
            token_advantages = inputs.pop("token_advantages", None)
            inputs.pop("completion_mask", None)
            inputs.pop("completion_token_count", None)
            labels = inputs.get("labels")
            if labels is None:
                outputs = model(**inputs)
                loss = outputs.loss
            else:
                outputs = None
                policy_token_log_probs = None
                response_mask = None
                has_response_tokens = bool(torch.any(labels.ne(-100)))
                if self.training_objective == "grpo":
                    if not has_response_tokens:
                        loss = _zero_loss_from_model(model)
                        if outputs is None:
                            outputs = SimpleNamespace()
                        if sample_weight is not None:
                            outputs.sample_weight = sample_weight
                        if advantage is not None:
                            outputs.advantage = advantage
                        if token_advantages is not None:
                            outputs.token_advantages = token_advantages
                        return (loss, outputs) if return_outputs else loss
                    effective_advantage = advantage if advantage is not None else sample_weight
                    if effective_advantage is None:
                        raise ValueError("GRPO training requires `advantage` or `sample_weight` in each batch.")
                    if self.old_policy_model is None:
                        raise ValueError("GRPO training requires a frozen old_policy_model.")
                    model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
                    needs_token_log_probs = bool(token_advantages is not None) or (
                        self.reference_model is not None and self.kl_beta > 0.0
                    )
                    if needs_token_log_probs:
                        policy_token_log_probs, response_mask = compute_completion_only_response_token_log_probs(
                            model=model,
                            model_inputs=model_inputs,
                            labels=labels,
                        )
                    if token_advantages is not None and response_mask is not None:
                        valid_policy_tokens = torch.any(response_mask)
                        if not valid_policy_tokens:
                            policy_log_probs = policy_token_log_probs.new_zeros(
                                (policy_token_log_probs.shape[0],),
                                dtype=torch.float32,
                            )
                        else:
                            token_counts = response_mask.sum(dim=-1).clamp(min=1)
                            policy_log_probs = policy_token_log_probs.sum(dim=-1) / token_counts
                    else:
                        policy_log_probs = compute_completion_only_response_log_probs(
                            model=model,
                            model_inputs=model_inputs,
                            labels=labels,
                        )
                    self._ensure_old_policy_model_device(model)
                    old_policy_inputs = model_inputs
                    with torch.inference_mode():
                        if token_advantages is not None and response_mask is not None:
                            old_policy_token_log_probs, _ = compute_completion_only_response_token_log_probs(
                                model=self.old_policy_model,
                                model_inputs=old_policy_inputs,
                                labels=labels,
                            )
                        else:
                            old_policy_token_log_probs = None
                    if token_advantages is not None and response_mask is not None:
                        if old_policy_token_log_probs is None:
                            raise RuntimeError("completion-only GRPO expected old policy token log-probs.")
                        shifted_token_advantages = token_advantages[..., 1:].to(policy_token_log_probs.device)
                        masked_policy_log_probs = policy_token_log_probs.masked_select(response_mask)
                        masked_old_policy_log_probs = old_policy_token_log_probs.masked_select(response_mask)
                        masked_token_advantages = shifted_token_advantages.masked_select(response_mask)
                        loss = compute_grpo_surrogate_loss(
                            policy_log_probs=masked_policy_log_probs,
                            old_policy_log_probs=masked_old_policy_log_probs.detach(),
                            advantages=masked_token_advantages,
                            clip_epsilon=self.ppo_clip_epsilon,
                        )
                    else:
                        with torch.inference_mode():
                            old_policy_log_probs = compute_completion_only_response_log_probs(
                                model=self.old_policy_model,
                                model_inputs=old_policy_inputs,
                                labels=labels,
                            )
                        loss = compute_grpo_surrogate_loss(
                            policy_log_probs=policy_log_probs,
                            old_policy_log_probs=old_policy_log_probs.detach(),
                            advantages=effective_advantage,
                            clip_epsilon=self.ppo_clip_epsilon,
                        )
                else:
                    outputs = model(**inputs)
                    # The collator already masks prompt tokens with -100, so the model's
                    # native loss is the same objective without an extra full-vocab log_softmax.
                    if token_advantages is not None:
                        nll_loss = compute_weighted_masked_response_nll(
                            logits=outputs.logits,
                            labels=labels,
                            token_weights=token_advantages,
                        )
                        loss = nll_loss
                    else:
                        if sample_weight is not None:
                            loss = compute_signed_weighted_masked_response_loss(
                                logits=outputs.logits,
                                labels=labels,
                                sample_weights=sample_weight,
                            )
                        else:
                            loss = getattr(outputs, "loss", None)
                            if loss is None:
                                loss = compute_masked_response_nll(logits=outputs.logits, labels=labels)

                if self.reference_model is not None and self.kl_beta > 0.0:
                    if policy_token_log_probs is None or response_mask is None:
                        policy_token_log_probs, response_mask = compute_completion_only_response_token_log_probs(
                            model=model,
                            model_inputs={key: value for key, value in inputs.items() if key != "labels"},
                            labels=labels,
                        )
                    self._ensure_reference_model_device(model)
                    reference_inputs = {key: value for key, value in inputs.items() if key != "labels"}
                    with torch.inference_mode():
                        reference_token_log_probs, _ = compute_completion_only_response_token_log_probs(
                            model=self.reference_model,
                            model_inputs=reference_inputs,
                            labels=labels,
                        )
                    kl_loss = compute_masked_sampled_token_kl(
                        policy_token_log_probs=policy_token_log_probs,
                        reference_token_log_probs=reference_token_log_probs,
                        response_mask=response_mask,
                    )
                    loss = loss + loss.new_tensor(self.kl_beta) * kl_loss

            if outputs is None:
                outputs = SimpleNamespace()
            if sample_weight is not None:
                outputs.sample_weight = sample_weight
            if advantage is not None:
                outputs.advantage = advantage
            if token_advantages is not None:
                outputs.token_advantages = token_advantages
            return (loss, outputs) if return_outputs else loss

    training_args_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": float(learning_rate),
        "num_train_epochs": float(num_train_epochs),
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "logging_steps": int(logging_steps),
        "lr_scheduler_type": str(lr_scheduler_type),
        "seed": int(seed),
        "lr_scheduler_type": str(lr_scheduler_type),
        "seed": int(seed),
        "save_steps": int(save_steps),
        "save_total_limit": int(save_total_limit),
        "warmup_ratio": float(warmup_ratio),
        "weight_decay": float(weight_decay),
        "max_grad_norm": float(max_grad_norm),
        "bf16": bool(bf16),
        "fp16": bool(fp16),
        "remove_unused_columns": False,
        "report_to": list(report_to or []),
        "disable_tqdm": True,
        "dataloader_num_workers": max(0, int(dataloader_num_workers)),
        "dataloader_persistent_workers": bool(dataloader_persistent_workers) and int(dataloader_num_workers) > 0,
    }
    if int(dataloader_num_workers) > 0 and int(dataloader_prefetch_factor) > 0:
        training_args_kwargs["dataloader_prefetch_factor"] = int(dataloader_prefetch_factor)
    args = TrainingArguments(**training_args_kwargs)
    trainer = WeightedLossTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=SingleExampleMultimodalCollator(
            processor,
            max_image_side=max_image_side,
            max_image_pixels=max_image_pixels,
            keep_recent_tool_image_messages=keep_recent_tool_image_messages,
            max_total_images=max_total_images,
            max_seq_length=max_seq_length,
            keep_recent_text_messages=keep_recent_text_messages,
        ),
        training_objective=training_objective,
        old_policy_model=old_policy_model,
        ppo_clip_epsilon=ppo_clip_epsilon,
        reference_model=reference_model,
        kl_beta=kl_beta,
        callbacks=list(callbacks or []),
    )
    trainer.add_callback(_build_epoch_progress_callback(trainer=trainer))
    return trainer


def _build_epoch_progress_callback(*, trainer: Any):
    try:
        from transformers import TrainerCallback
    except Exception as exc:
        raise ImportError("Epoch progress callbacks require the `transformers` package.") from exc
    try:
        from tqdm.auto import tqdm
    except Exception as exc:
        raise ImportError("Epoch progress callbacks require `tqdm` to be installed.") from exc

    class EpochProgressCallback(TrainerCallback):
        def __init__(self):
            self.trainer = trainer
            self.runtime = distributed_runtime_from_env()
            self.progress_bar = None
            self.current_epoch_index = 0
            self.epoch_start_global_step = 0
            self.last_epoch_step = 0
            self.steps_per_epoch = 0
            self.display_total_epochs = 0
            self.current_epoch_total = 0
            self.use_live_tqdm = False
            self.progress_log_every = 1
            self.last_logged_epoch_step = 0

        def _supports_live_tqdm(self) -> bool:
            stderr = getattr(sys, "stderr", None)
            isatty = getattr(stderr, "isatty", None)
            if not callable(isatty):
                return False
            try:
                return bool(isatty())
            except Exception:
                return False

        def _log_epoch_progress(self, *, epoch_step: int, epoch_total: int, force: bool = False) -> None:
            epoch_step = max(0, int(epoch_step))
            epoch_total = max(1, int(epoch_total))
            if epoch_step <= self.last_logged_epoch_step and not force:
                return
            if not force and not should_log_progress(epoch_step, epoch_total, int(self.progress_log_every)):
                return
            percent = 100.0 * float(epoch_step) / float(epoch_total)
            runtime_log(
                (
                    f"SFT epoch progress: epoch={self.current_epoch_index}/{self.display_total_epochs} "
                    f"step={epoch_step}/{epoch_total} progress={percent:.1f}%"
                ),
                runtime=self.runtime,
                main_process_only=True,
            )
            self.last_logged_epoch_step = epoch_step

        def _close_progress_bar(self) -> None:
            if self.progress_bar is None:
                return
            # Force-clear the tqdm line before later eval logs start printing.
            self.progress_bar.clear()
            self.progress_bar.close()
            self.progress_bar = None
            print("", flush=True)

        def on_train_begin(self, args, state, control, **kwargs):
            if not self.runtime.is_main_process:
                return control
            train_dataloader = self.trainer.get_train_dataloader()
            dataloader_steps = len(train_dataloader) if hasattr(train_dataloader, "__len__") else 0
            accumulation = max(1, int(args.gradient_accumulation_steps))
            self.steps_per_epoch = max(1, int(math.ceil(float(dataloader_steps) / float(accumulation))))
            self.display_total_epochs = max(1, int(math.ceil(float(args.num_train_epochs))))
            self.use_live_tqdm = self._supports_live_tqdm()
            self.progress_log_every = max(1, int(math.ceil(float(self.steps_per_epoch) / 100.0)))
            return control

        def on_epoch_begin(self, args, state, control, **kwargs):
            if not self.runtime.is_main_process:
                return control
            self.current_epoch_index += 1
            self.epoch_start_global_step = int(state.global_step or 0)
            self.last_epoch_step = 0
            self.last_logged_epoch_step = 0
            remaining_steps = max(0, int(state.max_steps or 0) - self.epoch_start_global_step)
            epoch_total = max(1, min(int(self.steps_per_epoch or 1), remaining_steps or int(self.steps_per_epoch or 1)))
            self.current_epoch_total = int(epoch_total)
            if self.use_live_tqdm:
                if self.progress_bar is not None:
                    self._close_progress_bar()
                self.progress_bar = tqdm(
                    total=epoch_total,
                    desc=f"Epoch {self.current_epoch_index}/{self.display_total_epochs}",
                    leave=False,
                    dynamic_ncols=True,
                )
            else:
                runtime_log(
                    (
                        f"SFT epoch progress start: epoch={self.current_epoch_index}/{self.display_total_epochs} "
                        f"steps={epoch_total}"
                    ),
                    runtime=self.runtime,
                    main_process_only=True,
                )
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if not self.runtime.is_main_process:
                return control
            current_epoch_step = max(0, int(state.global_step or 0) - self.epoch_start_global_step)
            delta = current_epoch_step - self.last_epoch_step
            if delta > 0:
                if self.progress_bar is not None:
                    remaining = max(0, int(self.progress_bar.total or 0) - int(self.progress_bar.n or 0))
                    self.progress_bar.update(min(delta, remaining))
                else:
                    epoch_total = max(1, int(self.current_epoch_total or self.steps_per_epoch or 1))
                    self._log_epoch_progress(epoch_step=current_epoch_step, epoch_total=epoch_total)
                self.last_epoch_step = current_epoch_step
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            if not self.runtime.is_main_process:
                return control
            epoch_total = max(1, int(self.current_epoch_total or self.steps_per_epoch or 1))
            if self.progress_bar is not None:
                remaining = max(0, int(self.progress_bar.total or 0) - int(self.progress_bar.n or 0))
                if remaining > 0:
                    self.progress_bar.update(remaining)
                self._close_progress_bar()
            else:
                self._log_epoch_progress(epoch_step=epoch_total, epoch_total=epoch_total, force=True)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            self._close_progress_bar()
            return control

    return EpochProgressCallback()


def _build_rollout_eval_callback(
    *,
    processor: Any,
    rollout_eval_config: RolloutEvaluationConfig,
    rollout_eval_output_dir: str | Path = "",
    policy_factory: Any = None,
):
    try:
        from transformers import TrainerCallback
    except Exception as exc:
        raise ImportError("Rollout evaluation callbacks require the `transformers` package.") from exc

    class RolloutEvalCallback(TrainerCallback):
        def __init__(self):
            self.processor = processor
            self.rollout_eval_config = rollout_eval_config
            self.rollout_eval_output_dir = str(rollout_eval_output_dir or "").strip()
            self.runtime = distributed_runtime_from_env()

        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            if model is None:
                return control
            eval_model = _unwrap_model(model)
            was_training = bool(getattr(eval_model, "training", False))
            if hasattr(eval_model, "eval"):
                eval_model.eval()
            try:
                epoch_index = max(1, int(round(float(state.epoch or 0.0))))
                resume_dir = save_sft_epoch_resume_checkpoint(
                    model=eval_model,
                    processor=self.processor,
                    output_dir=args.output_dir,
                    epoch_index=epoch_index,
                    optimizer=kwargs.get("optimizer"),
                    lr_scheduler=kwargs.get("lr_scheduler"),
                    state=state,
                    runtime=self.runtime,
                )
                runtime_log(
                    f"saved epoch-resume checkpoint for rollout-eval recovery: {resume_dir}",
                    runtime=self.runtime,
                    main_process_only=True,
                )
                if bool(self.rollout_eval_config.inline_rollout_eval):
                    runtime_log(
                        (
                            "running epoch-end rollout eval inline before the next epoch; "
                            f"resume checkpoint remains available at {resume_dir}"
                        ),
                        runtime=self.runtime,
                        main_process_only=True,
                    )
                    cleanup_fn = None
                    if callable(policy_factory):
                        policy_factory_result = policy_factory(
                            eval_model=eval_model,
                            processor=self.processor,
                            rollout_eval_config=self.rollout_eval_config,
                            state=state,
                            runtime=self.runtime,
                        )
                        if isinstance(policy_factory_result, tuple) and len(policy_factory_result) == 2:
                            policy, cleanup_fn = policy_factory_result
                        else:
                            policy = policy_factory_result
                    else:
                        from saver_v3.model.qwen_policy import QwenGenerationPolicy

                        policy = QwenGenerationPolicy.from_components(
                            model=eval_model,
                            processor=self.processor,
                            max_new_tokens=int(self.rollout_eval_config.policy_max_new_tokens),
                            max_total_images=int(self.rollout_eval_config.max_total_images),
                            max_seq_length=int(self.rollout_eval_config.max_seq_length),
                            keep_recent_tool_image_messages=int(
                                getattr(self.rollout_eval_config, "keep_recent_tool_image_messages", 0)
                            ),
                            keep_recent_text_messages=int(self.rollout_eval_config.keep_recent_text_messages),
                            max_image_side=int(self.rollout_eval_config.max_image_side),
                            max_image_pixels=int(self.rollout_eval_config.max_image_pixels),
                            do_sample=False,
                            use_generation_cache=bool(self.rollout_eval_config.use_generation_cache),
                        )
                    try:
                        run_rollout_eval_with_policy(
                            policy,
                            rollout_eval_config=self.rollout_eval_config,
                            output_dir=args.output_dir,
                            rollout_eval_output_dir=self.rollout_eval_output_dir,
                            epoch_index=epoch_index,
                            epoch_value=float(state.epoch or epoch_index),
                            runtime=self.runtime,
                        )
                    finally:
                        if callable(cleanup_fn):
                            cleanup_fn()
                else:
                    runtime_log(
                        (
                            "deferred epoch-end rollout eval to an external process; "
                            f"resume checkpoint is ready at {resume_dir}"
                        ),
                        runtime=self.runtime,
                        main_process_only=True,
                    )
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if was_training and hasattr(eval_model, "train"):
                    eval_model.train()
            return control

    return RolloutEvalCallback()


def _resolve_training_proposal_device(explicit_device: str | None, *, runtime: Any) -> str:
    if str(explicit_device or "").strip():
        return str(explicit_device)
    try:
        import torch
    except Exception:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    try:
        visible_cuda_devices = int(torch.cuda.device_count())
    except Exception:
        visible_cuda_devices = 0
    if visible_cuda_devices <= 0:
        return "cpu"
    local_rank = int(getattr(runtime, "local_rank", 0) or 0)
    if 0 <= local_rank < visible_cuda_devices:
        return f"cuda:{local_rank}"
    return "cuda:0"


def _load_training_proposal_runtime(
    *,
    proposal_model_path: str | Path = "",
    proposal_torch_dtype: str = "auto",
    proposal_device: str = "",
    runtime: Any,
) -> Any:
    if not str(proposal_model_path or "").strip():
        return None
    resolved_device = _resolve_training_proposal_device(proposal_device, runtime=runtime)
    runtime_log(
        f"loading SFT proposal model from {proposal_model_path} on device={resolved_device}",
        runtime=runtime,
        main_process_only=True,
    )
    return SiglipFeatureEncoder.from_pretrained(
        str(proposal_model_path),
        torch_dtype=proposal_torch_dtype,
        device=resolved_device,
    )


def _prepared_rows_require_feature_guided_proposal(rows: Sequence[Dict[str, Any]]) -> bool:
    for row in rows:
        for step in list(row.get("oracle_trajectory") or []):
            if str(step.get("tool") or "").strip() == "seek_evidence":
                return True
    return False


def sft_epoch_resume_dir(output_dir: str | Path, epoch_index: int) -> Path:
    return Path(output_dir) / SFT_EPOCH_RESUME_DIRNAME / f"epoch_{int(epoch_index):03d}"


def _write_rollout_eval_record(
    *,
    output_dir: str | Path,
    eval_output_dir: str | Path = "",
    metrics: Dict[str, Any],
    epoch_value: float,
    runtime: Any,
) -> None:
    if not runtime.is_main_process:
        return
    output_path = Path(output_dir)
    explicit_eval_output_dir = str(eval_output_dir or "").strip()
    write_to_legacy_output_dir = not explicit_eval_output_dir or Path(explicit_eval_output_dir) == output_path
    record_output_dir = output_path if write_to_legacy_output_dir else Path(explicit_eval_output_dir)
    record = {"epoch": float(epoch_value), **metrics}
    append_jsonl(record_output_dir / "rollout_eval_metrics.jsonl", record)
    summary_line = _format_rollout_eval_summary_line(epoch_value=float(epoch_value), metrics=metrics)
    _append_rollout_eval_summary_line(record_output_dir / "rollout_eval_summary.log", summary_line)
    if write_to_legacy_output_dir:
        append_jsonl(output_path / "logs" / "rollout_eval_metrics.jsonl", record)
        _append_rollout_eval_summary_line(output_path / "logs" / "rollout_eval_summary.log", summary_line)
    runtime_log(
        f"rollout eval headline: {_format_rollout_eval_console_line(epoch_value=float(epoch_value), metrics=metrics)}",
        runtime=runtime,
        main_process_only=True,
    )
    runtime_log(
        f"epoch {float(epoch_value):.2f} rollout eval (full): {json.dumps(metrics, ensure_ascii=False)}",
        runtime=runtime,
        main_process_only=True,
    )


def _format_rollout_eval_summary_line(*, epoch_value: float, metrics: Dict[str, Any]) -> str:
    return f"[{log_timestamp()}] {_format_rollout_eval_console_line(epoch_value=epoch_value, metrics=metrics)}"


def _format_rollout_eval_console_line(*, epoch_value: float, metrics: Dict[str, Any]) -> str:
    ordered_keys = (
        "num_records",
        "anomaly_span_recall_at_0_3",
        "anomaly_span_recall_at_0_5",
        "anomaly_span_recall_at_0_7",
        "temporal_miou",
        "existence_accuracy",
        "category_macro_f1",
        "temporal_r1_at_0_3",
        "temporal_r1_at_0_5",
        "temporal_r1_at_0_7",
        "temporal_map_avg",
        "precursor_miou",
        "evidence_f1_at_3",
        "event_chain_f1",
        "event_chain_finalize_rate",
        "protocol_compliance_rate",
        "mean_num_turns",
    )
    parts = [f"epoch={float(epoch_value):.2f}"]
    for key in ordered_keys:
        if key not in metrics:
            continue
        value = metrics.get(key)
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, int):
            rendered = str(int(value))
        elif isinstance(value, float):
            rendered = f"{float(value):.4f}"
        else:
            rendered = str(value)
        parts.append(f"{key}={rendered}")
    semantic_metrics = metrics.get("semantic_metrics")
    if isinstance(semantic_metrics, dict):
        qa_relaxed_summary = semantic_metrics.get("qa_relaxed_summary")
        qa_judge_summary = semantic_metrics.get("qa_judge_summary")
        bertscore_summary = semantic_metrics.get("bertscore_summary")
        rouge_summary = semantic_metrics.get("rouge_summary")
        qa_relaxed_overall = qa_relaxed_summary.get("overall") if isinstance(qa_relaxed_summary, dict) else None
        qa_judge_overall = qa_judge_summary.get("overall") if isinstance(qa_judge_summary, dict) else None
        bertscore_f1 = bertscore_summary.get("f1") if isinstance(bertscore_summary, dict) and bool(bertscore_summary.get("available", True)) else None
        rouge_l_f1 = rouge_summary.get("rouge_l_f1") if isinstance(rouge_summary, dict) else None
        if isinstance(qa_relaxed_overall, (int, float)):
            parts.append(f"qa_relaxed={float(qa_relaxed_overall):.4f}")
        if isinstance(qa_judge_overall, (int, float)) and bool((qa_judge_summary or {}).get("configured")):
            parts.append(f"qa_judge={float(qa_judge_overall):.4f}")
        if isinstance(bertscore_f1, (int, float)):
            parts.append(f"bertscore_f1={float(bertscore_f1):.4f}")
        if isinstance(rouge_l_f1, (int, float)):
            parts.append(f"rouge_l_f1={float(rouge_l_f1):.4f}")
    return " ".join(parts)


def _append_rollout_eval_summary_line(path: str | Path, line: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(str(line).rstrip("\n") + "\n")


def run_rollout_eval_with_policy(
    policy: Any,
    *,
    rollout_eval_config: RolloutEvaluationConfig,
    output_dir: str | Path,
    rollout_eval_output_dir: str | Path = "",
    epoch_index: int,
    epoch_value: float | None = None,
    runtime: Any = None,
) -> Optional[Dict[str, Any]]:
    runtime = runtime or distributed_runtime_from_env()
    resolved_rollout_eval_output_dir = str(rollout_eval_output_dir or "").strip() or str(output_dir)
    metrics = run_rollout_evaluation(
        policy,
        eval_config=rollout_eval_config,
        output_dir=resolved_rollout_eval_output_dir,
        epoch_index=int(epoch_index),
        runtime=runtime,
    )
    if metrics is not None:
        _write_rollout_eval_record(
            output_dir=output_dir,
            eval_output_dir=resolved_rollout_eval_output_dir,
            metrics=metrics,
            epoch_value=float(epoch_value if epoch_value is not None else epoch_index),
            runtime=runtime,
        )
    return metrics


def _save_epoch_resume_rng_state(
    output_dir: str | Path,
    *,
    runtime: Any,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng_states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        if getattr(runtime, "world_size", 1) > 1:
            rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
        else:
            rng_states["cuda"] = torch.cuda.random.get_rng_state()
    rng_filename = "rng_state.pth" if getattr(runtime, "world_size", 1) <= 1 else f"rng_state_{int(runtime.rank)}.pth"
    torch.save(rng_states, output_dir / rng_filename)


def save_sft_epoch_resume_checkpoint(
    *,
    model: Any,
    processor: Any,
    output_dir: str | Path,
    epoch_index: int,
    optimizer: Any = None,
    lr_scheduler: Any = None,
    state: Any = None,
    runtime: Any = None,
) -> Path:
    runtime = runtime or distributed_runtime_from_env()
    checkpoint_dir = sft_epoch_resume_dir(output_dir, epoch_index)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_to_save = _unwrap_model(model)
    if runtime.is_main_process:
        with _temporary_model_use_cache(model_to_save, enabled=True):
            model_to_save.save_pretrained(str(checkpoint_dir))
        if hasattr(processor, "save_pretrained"):
            processor.save_pretrained(str(checkpoint_dir))
        if optimizer is not None:
            torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
        if lr_scheduler is not None:
            torch.save(lr_scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        if state is not None and hasattr(state, "save_to_json"):
            state.save_to_json(str(checkpoint_dir / "trainer_state.json"))
        write_json(
            checkpoint_dir / "resume_metadata.json",
            {
                "epoch_index": int(epoch_index),
                "checkpoint_kind": "epoch_end_before_rollout_eval",
                "world_size": int(getattr(runtime, "world_size", 1)),
                "global_step": int(getattr(state, "global_step", 0) or 0) if state is not None else 0,
                "epoch": float(getattr(state, "epoch", float(epoch_index)) or float(epoch_index)),
            },
        )
    _save_epoch_resume_rng_state(checkpoint_dir, runtime=runtime)
    runtime_log(
        (
            f"epoch-resume checkpoint ready at {checkpoint_dir}; "
            "waiting for all distributed ranks to finish the RL epoch save"
        ),
        runtime=runtime,
        main_process_only=True,
    )
    runtime_log(
        f"entering RL epoch-save barrier at {checkpoint_dir}",
        runtime=runtime,
        main_process_only=False,
    )
    distributed_barrier(runtime)
    runtime_log(
        f"passed RL epoch-save barrier at {checkpoint_dir}",
        runtime=runtime,
        main_process_only=False,
    )
    return checkpoint_dir


def run_rollout_eval_from_checkpoint(
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    rollout_eval_output_dir: str | Path = "",
    rollout_eval_config: RolloutEvaluationConfig,
    epoch_index: int,
    model_path: str | Path = "",
    torch_dtype: str = "auto",
    attn_implementation: Optional[str] = None,
    runtime: Any = None,
    policy_factory: Any = None,
) -> Optional[Dict[str, Any]]:
    from saver_v3.model.qwen_policy import QwenGenerationPolicy

    runtime = runtime or distributed_runtime_from_env()
    resolved_checkpoint_path = Path(checkpoint_path)
    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(f"resume checkpoint does not exist: {resolved_checkpoint_path}")

    runtime_log(
        (
            f"loading rollout-eval recovery checkpoint from {resolved_checkpoint_path} "
            f"(base_model_hint={model_path or resolved_checkpoint_path})"
        ),
        runtime=runtime,
        main_process_only=True,
    )
    resolved_device_map = resolve_inference_device_map("auto", runtime=runtime)
    runtime_log(
        f"rollout-eval recovery policy device_map={resolved_device_map}",
        runtime=runtime,
        main_process_only=True,
    )
    cleanup_fn = None
    if callable(policy_factory):
        policy_factory_result = policy_factory(
            checkpoint_path=resolved_checkpoint_path,
            model_path=model_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            rollout_eval_config=rollout_eval_config,
            runtime=runtime,
        )
        if isinstance(policy_factory_result, tuple) and len(policy_factory_result) == 2:
            policy, cleanup_fn = policy_factory_result
        else:
            policy = policy_factory_result
    else:
        policy = QwenGenerationPolicy.from_pretrained(
            resolved_checkpoint_path,
            torch_dtype=torch_dtype,
            device_map=resolved_device_map,
            attn_implementation=attn_implementation,
            max_new_tokens=int(rollout_eval_config.policy_max_new_tokens),
            max_total_images=int(rollout_eval_config.max_total_images),
            max_seq_length=int(rollout_eval_config.max_seq_length),
            keep_recent_tool_image_messages=int(getattr(rollout_eval_config, "keep_recent_tool_image_messages", 0)),
            keep_recent_text_messages=int(rollout_eval_config.keep_recent_text_messages),
            max_image_side=int(rollout_eval_config.max_image_side),
            max_image_pixels=int(rollout_eval_config.max_image_pixels),
            do_sample=False,
            use_generation_cache=bool(rollout_eval_config.use_generation_cache),
        )
    try:
        return run_rollout_eval_with_policy(
            policy,
            rollout_eval_config=rollout_eval_config,
            output_dir=output_dir,
            rollout_eval_output_dir=rollout_eval_output_dir,
            epoch_index=int(epoch_index),
            epoch_value=float(epoch_index),
            runtime=runtime,
        )
    finally:
        if callable(cleanup_fn):
            cleanup_fn()
        else:
            del policy
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_standard_sft(
    *,
    prepared_data_path: str | Path,
    include_splits: Optional[str | Sequence[str]] = None,
    model_path: str | Path,
    output_dir: str | Path,
    log_dir: str | Path = "",
    rollout_eval_output_dir: str | Path = "",
    resume_from_checkpoint: str | Path = "",
    torch_dtype: str = "auto",
    attn_implementation: Optional[str] = None,
    gradient_checkpointing: bool = False,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[Sequence[str]] = None,
    learning_rate: float = 1e-5,
    num_train_epochs: float = 1.0,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    logging_steps: int = 10,
    save_steps: int = 100,
    save_total_limit: int = 2,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    bf16: bool = True,
    fp16: bool = False,
    max_image_side: int = 0,
    max_image_pixels: int = 0,
    keep_recent_tool_image_messages: int = 0,
    max_total_images: int = 0,
    max_seq_length: int = 0,
    keep_recent_text_messages: int = 0,
    dataloader_num_workers: int = 0,
    dataloader_prefetch_factor: int = 0,
    dataloader_persistent_workers: bool = False,
    lr_scheduler_type: str = "cosine",
    report_to: Optional[Sequence[str]] = None,
    seed: int = 42,
    ddp_find_unused_parameters: Optional[bool] = False,
    deepspeed: str | Path = "",
    saver_config: Any = None,
    rollout_eval_config: RolloutEvaluationConfig | None = None,
    rollout_eval_policy_factory: Any = None,
    proposal_model_path: str | Path = "",
    proposal_torch_dtype: str = "auto",
    proposal_device: str = "",
) -> Dict[str, Any]:
    try:
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:
        raise ImportError("Standard SFT training requires the `trl` package.") from exc

    runtime = distributed_runtime_from_env()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_log_dir = Path(str(log_dir).strip()) if str(log_dir or "").strip() else output_dir / "logs"
    resolved_rollout_eval_output_dir = (
        Path(str(rollout_eval_output_dir).strip()) if str(rollout_eval_output_dir or "").strip() else output_dir
    )
    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    resolved_rollout_eval_output_dir.mkdir(parents=True, exist_ok=True)

    runtime_log(
        (
            f"SFT setup: prepared_data={prepared_data_path} "
            f"output_dir={output_dir} model_path={model_path}"
        ),
        runtime=runtime,
        main_process_only=True,
    )
    runtime_log(
        (
            "SFT memory controls: "
            f"max_image_side={int(max_image_side) or 'off'} "
            f"max_image_pixels={int(max_image_pixels) or 'off'} "
            f"keep_recent_tool_image_messages={int(keep_recent_tool_image_messages) or 'all'} "
            f"max_total_images={int(max_total_images) or 'all'} "
            f"max_seq_length={int(max_seq_length) or 'off'} "
            f"keep_recent_text_messages={int(keep_recent_text_messages) or 'all'}"
        ),
        runtime=runtime,
        main_process_only=True,
    )
    runtime_log(
        (
            "SFT dataloader controls: "
            f"num_workers={max(0, int(dataloader_num_workers))} "
            f"prefetch_factor={int(dataloader_prefetch_factor) if int(dataloader_num_workers) > 0 and int(dataloader_prefetch_factor) > 0 else 'off'} "
            f"persistent_workers={bool(dataloader_persistent_workers) and int(dataloader_num_workers) > 0}"
        ),
        runtime=runtime,
        main_process_only=True,
    )
    runtime_log(f"loading policy model from {model_path}", runtime=runtime, main_process_only=True)
    model, processor = load_qwen_model_and_processor(
        model_path,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        gradient_checkpointing=gradient_checkpointing,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
    )
    examples = _load_prepared_jsonl_rows(
        prepared_data_path,
        include_splits=include_splits,
    )
    if not examples:
        raise ValueError(f"No prepared SFT examples were loaded from {prepared_data_path}")
    if not all(_is_compact_trace_sft_row(example) for example in examples):
        raise ValueError(
            "Prepared SFT training now only accepts compact_trace_v2 rows. "
            "Regenerate the prepared JSONL with the new lazy video-level format instead of legacy step/episode rows."
        )
    frame_cache_summary = summarize_example_frame_cache_status(examples)
    feature_cache_summary = summarize_example_feature_cache_status(examples)
    runtime_log(
        format_example_frame_cache_status(frame_cache_summary, prefix="training frame cache"),
        runtime=runtime,
        main_process_only=True,
    )
    runtime_log(
        format_example_feature_cache_status(feature_cache_summary, prefix="training feature cache"),
        runtime=runtime,
        main_process_only=True,
    )
    if int(frame_cache_summary.get("num_missing_frame_cache", 0)) > 0:
        raise ValueError(
            "Prepared SFT training requires frame_cache for every referenced video. "
            f"{format_example_frame_cache_status(frame_cache_summary)}"
        )
    if int(feature_cache_summary.get("num_missing_feature_cache", 0)) > 0:
        raise ValueError(
            "Prepared SFT training requires feature_cache for every referenced video. "
            f"{format_example_feature_cache_status(feature_cache_summary)}"
        )
    strict_feature_guided_proposal = _prepared_rows_require_feature_guided_proposal(examples)
    if strict_feature_guided_proposal and not str(proposal_model_path or "").strip():
        raise ValueError(
            "Prepared SFT training requires proposal_model_path because compact_trace replay includes seek_evidence."
        )
    proposal_runtime = (
        _load_training_proposal_runtime(
            proposal_model_path=proposal_model_path,
            proposal_torch_dtype=proposal_torch_dtype,
            proposal_device=proposal_device,
            runtime=runtime,
        )
        if strict_feature_guided_proposal
        else None
    )
    train_dataset = LazyVideoSFTDataset(
        examples,
        config=saver_config,
        proposal_runtime=proposal_runtime,
        strict_feature_guided_proposal=strict_feature_guided_proposal,
    )
    data_collator: Any = BatchEpisodeSFTCollator(
        processor,
        max_image_side=max_image_side,
        max_image_pixels=max_image_pixels,
        keep_recent_tool_image_messages=keep_recent_tool_image_messages,
        max_total_images=max_total_images,
        max_seq_length=max_seq_length,
        keep_recent_text_messages=keep_recent_text_messages,
    )
    callbacks = []
    if rollout_eval_config is not None:
        callbacks.append(
            _build_rollout_eval_callback(
                processor=processor,
                rollout_eval_config=rollout_eval_config,
                rollout_eval_output_dir=resolved_rollout_eval_output_dir,
                policy_factory=rollout_eval_policy_factory,
            )
        )

    config_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": float(learning_rate),
        "num_train_epochs": float(num_train_epochs),
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "logging_steps": int(logging_steps),
        "lr_scheduler_type": str(lr_scheduler_type),
        "seed": int(seed),
        "save_steps": int(save_steps),
        "save_total_limit": int(save_total_limit),
        "warmup_ratio": float(warmup_ratio),
        "weight_decay": float(weight_decay),
        "max_grad_norm": float(max_grad_norm),
        "bf16": bool(bf16),
        "fp16": bool(fp16),
        "remove_unused_columns": False,
        "report_to": list(report_to or []),
        "disable_tqdm": True,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "max_length": int(max_seq_length) if int(max_seq_length) > 0 else None,
        "packing": False,
        "dataloader_num_workers": max(0, int(dataloader_num_workers)),
        "dataloader_persistent_workers": bool(dataloader_persistent_workers) and int(dataloader_num_workers) > 0,
        "ddp_find_unused_parameters": (False if ddp_find_unused_parameters is None else bool(ddp_find_unused_parameters)),
        "gradient_checkpointing": bool(gradient_checkpointing),
    }
    if int(dataloader_num_workers) > 0 and int(dataloader_prefetch_factor) > 0:
        config_kwargs["dataloader_prefetch_factor"] = int(dataloader_prefetch_factor)
    if str(deepspeed or "").strip():
        config_kwargs["deepspeed"] = str(deepspeed)
    if bool(gradient_checkpointing):
        config_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    args = SFTConfig(**config_kwargs)
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=processor,
        callbacks=list(callbacks or []),
    )
    trainer.add_callback(_build_epoch_progress_callback(trainer=trainer))

    if runtime.is_main_process:
        write_json(
            resolved_log_dir / "run_standard_sft_config.json",
            {
                "prepared_data_path": str(prepared_data_path),
                "include_splits": list(parse_include_splits(include_splits) or []),
                "model_path": str(model_path),
                "output_dir": str(output_dir),
                "log_dir": str(resolved_log_dir),
                "rollout_eval_output_dir": str(resolved_rollout_eval_output_dir),
                "resume_from_checkpoint": str(resume_from_checkpoint or ""),
                "num_examples": len(train_dataset),
                "learning_rate": float(learning_rate),
                "num_train_epochs": float(num_train_epochs),
                "per_device_train_batch_size": int(per_device_train_batch_size),
                "gradient_accumulation_steps": int(gradient_accumulation_steps),
                "gradient_checkpointing": bool(gradient_checkpointing),
                "use_lora": bool(use_lora),
                "max_image_side": int(max_image_side),
                "max_image_pixels": int(max_image_pixels),
                "keep_recent_tool_image_messages": int(keep_recent_tool_image_messages),
                "max_total_images": int(max_total_images),
                "max_seq_length": int(max_seq_length),
                "keep_recent_text_messages": int(keep_recent_text_messages),
                "dataloader_num_workers": int(dataloader_num_workers),
                "dataloader_prefetch_factor": int(dataloader_prefetch_factor),
                "dataloader_persistent_workers": bool(dataloader_persistent_workers),
                "lr_scheduler_type": str(lr_scheduler_type),
                "report_to": list(report_to or []),
                "seed": int(seed),
                "ddp_find_unused_parameters": (False if ddp_find_unused_parameters is None else bool(ddp_find_unused_parameters)),
                "deepspeed": str(deepspeed or ""),
                "rollout_eval_enabled": rollout_eval_config is not None,
            },
        )

    resolved_resume_from_checkpoint = str(resume_from_checkpoint) if resume_from_checkpoint else None
    if resolved_resume_from_checkpoint:
        runtime_log(
            f"resuming SFTTrainer.train() from {resolved_resume_from_checkpoint}",
            runtime=runtime,
            main_process_only=True,
        )
    runtime_log("starting SFTTrainer.train()", runtime=runtime, main_process_only=True)
    train_result = trainer.train(resume_from_checkpoint=resolved_resume_from_checkpoint)

    if trainer.is_world_process_zero():
        trainer_model = getattr(trainer, "model", model)
        with _temporary_model_use_cache(trainer_model, enabled=True):
            trainer.save_model(str(output_dir))
        _set_model_use_cache(trainer_model, True)
        if hasattr(processor, "save_pretrained"):
            processor.save_pretrained(str(output_dir))
        write_json(
            resolved_log_dir / "trainer_log_history.json",
            {
                "log_history": list(getattr(trainer.state, "log_history", []) or []),
                "global_step": int(getattr(trainer.state, "global_step", 0) or 0),
                "epoch": float(getattr(trainer.state, "epoch", 0.0) or 0.0),
                "max_steps": int(getattr(trainer.state, "max_steps", 0) or 0),
                "num_train_epochs": int(getattr(trainer.state, "num_train_epochs", 0) or 0),
            },
        )
    distributed_barrier(runtime)
    result = {
        "prepared_data_path": str(prepared_data_path),
        "num_examples": len(train_dataset),
        "output_dir": str(output_dir),
        "log_dir": str(resolved_log_dir),
        "rollout_eval_output_dir": str(resolved_rollout_eval_output_dir),
        "train_loss": float(getattr(train_result, "training_loss", 0.0)),
    }
    if trainer.is_world_process_zero():
        write_json(resolved_log_dir / "run_standard_sft_result.json", result)
    return result
