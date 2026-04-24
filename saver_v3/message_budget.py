from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Set, Tuple

from saver_v3.core.initial_observation import (
    error_on_initial_scan_seq_prune,
    is_initial_global_scan_message,
    protect_initial_global_scan_message,
)

try:
    import torch
except Exception:  # pragma: no cover - torch is available in training runtimes
    torch = None  # type: ignore[assignment]


_TIMESTAMP_ONLY_RE = re.compile(r"^\s*\d+(?:\.\d+)?s\s*$")


def is_image_item(item: Any) -> bool:
    return isinstance(item, dict) and item.get("type") == "image" and ("image" in item or "image_ref" in item)


def is_video_item(item: Any) -> bool:
    return isinstance(item, dict) and item.get("type") == "video" and ("video" in item or "video_ref" in item)


def is_image_or_video_item(item: Any) -> bool:
    return is_image_item(item) or is_video_item(item)


def is_timestamp_text_item(item: Any) -> bool:
    if not isinstance(item, dict) or item.get("type") != "text":
        return False
    return bool(_TIMESTAMP_ONLY_RE.match(str(item.get("text") or "").strip()))


def paired_multimodal_removal_indices(content: List[Dict[str, Any]], content_index: int) -> set[int]:
    removals = {int(content_index)}
    timestamp_index = int(content_index) - 1
    if 0 <= timestamp_index < len(content) and is_timestamp_text_item(content[timestamp_index]):
        removals.add(timestamp_index)
    return removals


def _history_prefix_end(messages: List[Dict[str, Any]]) -> int:
    prefix_end = 0
    while prefix_end < len(messages) and messages[prefix_end].get("role") in {"system", "user"}:
        prefix_end += 1
    return prefix_end


def _history_turn_spans(messages: List[Dict[str, Any]]) -> Tuple[int, List[Tuple[int, int]]]:
    prefix_end = _history_prefix_end(messages)
    spans: List[Tuple[int, int]] = []
    index = prefix_end
    while index < len(messages):
        start = index
        role = str(messages[index].get("role") or "")
        index += 1
        if role == "assistant":
            while index < len(messages) and str(messages[index].get("role") or "") == "tool":
                index += 1
        elif role == "tool":
            while index < len(messages) and str(messages[index].get("role") or "") == "tool":
                index += 1
        spans.append((start, index))
    return prefix_end, spans


def _span_contains_protected_initial_scan(messages: List[Dict[str, Any]], *, start: int, end: int) -> bool:
    for message in messages[start:end]:
        if is_initial_global_scan_message(message) and error_on_initial_scan_seq_prune(message):
            return True
    return False


def _protected_scan_timeline_image_positions(messages: List[Dict[str, Any]]) -> Set[Tuple[int, int]]:
    protected_positions: Set[Tuple[int, int]] = set()
    for message_index, message in enumerate(messages):
        if not (is_initial_global_scan_message(message) and protect_initial_global_scan_message(message)):
            continue
        content = list(message.get("content", []))
        protected_positions.update(
            {
                (int(message_index), int(content_index))
                for content_index, item in enumerate(content)
                if is_image_item(item)
            }
        )
    return protected_positions


def _content_frame_count(item: Dict[str, Any]) -> int:
    if is_image_item(item):
        return 1
    if not is_video_item(item):
        return 0
    if "video_ref" in item and isinstance(item.get("video_ref"), dict):
        video_ref = dict(item.get("video_ref") or {})
        for key in ("nframes", "num_frames", "max_frames"):
            value = video_ref.get(key)
            if value is None:
                continue
            try:
                return max(1, int(value))
            except Exception:
                continue
        return 1
    video = item.get("video")
    if isinstance(video, (list, tuple)):
        return max(1, len(video))
    if torch is not None and isinstance(video, torch.Tensor):
        if video.ndim <= 0:
            return 1
        return max(1, int(video.shape[0]))
    if isinstance(video, dict):
        for key in ("nframes", "num_frames", "max_frames"):
            value = video.get(key)
            if value is None:
                continue
            try:
                return max(1, int(value))
            except Exception:
                continue
    for key in ("nframes", "num_frames", "max_frames"):
        value = item.get(key)
        if value is None:
            continue
        try:
            return max(1, int(value))
        except Exception:
            continue
    return 1


def _uniform_keep_positions(total_count: int, keep_count: int) -> Set[int]:
    total_count = max(0, int(total_count))
    keep_count = max(0, min(int(keep_count), total_count))
    if keep_count >= total_count:
        return set(range(total_count))
    if keep_count <= 0:
        return set()
    if keep_count == 1:
        return {total_count - 1}
    selected: Set[int] = set()
    for slot in range(keep_count):
        position = int(round(slot * (total_count - 1) / float(keep_count - 1)))
        selected.add(max(0, min(total_count - 1, position)))
    while len(selected) < keep_count:
        for position in range(total_count - 1, -1, -1):
            selected.add(position)
            if len(selected) >= keep_count:
                break
    return selected


def _trim_video_item_frames(item: Dict[str, Any], keep_subindices: List[int]) -> Dict[str, Any]:
    trimmed = dict(item)
    keep_subindices = sorted({int(index) for index in keep_subindices if int(index) >= 0})
    if not keep_subindices:
        return trimmed
    if "video_ref" in trimmed and isinstance(trimmed.get("video_ref"), dict):
        video_ref = dict(trimmed.get("video_ref") or {})
        kept_count = len(keep_subindices)
        video_ref["nframes"] = int(kept_count)
        max_frames_value = video_ref.get("max_frames")
        if max_frames_value is not None:
            try:
                video_ref["max_frames"] = min(int(max_frames_value), int(kept_count))
            except Exception:
                video_ref["max_frames"] = int(kept_count)
        trimmed["video_ref"] = video_ref
        return trimmed
    video = trimmed.get("video")
    if isinstance(video, list):
        trimmed["video"] = [video[index] for index in keep_subindices if 0 <= index < len(video)]
        return trimmed
    if isinstance(video, tuple):
        trimmed["video"] = tuple(video[index] for index in keep_subindices if 0 <= index < len(video))
        return trimmed
    if torch is not None and isinstance(video, torch.Tensor) and video.ndim > 0:
        index_tensor = torch.tensor(
            [index for index in keep_subindices if 0 <= index < int(video.shape[0])],
            dtype=torch.long,
            device=video.device,
        )
        trimmed["video"] = video.index_select(0, index_tensor)
        return trimmed
    if isinstance(video, dict):
        kept_count = len(keep_subindices)
        copied = dict(video)
        copied["nframes"] = int(kept_count)
        max_frames_value = copied.get("max_frames")
        if max_frames_value is not None:
            try:
                copied["max_frames"] = min(int(max_frames_value), int(kept_count))
            except Exception:
                copied["max_frames"] = int(kept_count)
        trimmed["video"] = copied
        return trimmed
    kept_count = len(keep_subindices)
    trimmed["nframes"] = int(kept_count)
    if "max_frames" in trimmed:
        try:
            trimmed["max_frames"] = min(int(trimmed.get("max_frames") or kept_count), int(kept_count))
        except Exception:
            trimmed["max_frames"] = int(kept_count)
    return trimmed


def _collect_frame_units(messages: List[Dict[str, Any]]) -> List[Tuple[int, int, int]]:
    frame_units: List[Tuple[int, int, int]] = []
    for message_index, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for content_index, item in enumerate(content):
            if not isinstance(item, dict):
                continue
            frame_count = _content_frame_count(item)
            for subindex in range(frame_count):
                frame_units.append((int(message_index), int(content_index), int(subindex)))
    return frame_units


def cap_tool_message_frames(
    messages: List[Dict[str, Any]],
    *,
    max_tool_message_frames: int = 0,
    copy_messages: bool = True,
) -> List[Dict[str, Any]]:
    if copy_messages:
        prepared = copy.deepcopy(messages)
    else:
        prepared = messages
    max_tool_message_frames = int(max_tool_message_frames)
    if max_tool_message_frames <= 0:
        return prepared

    for message_index, message in enumerate(prepared):
        if str(message.get("role") or "") != "tool":
            continue
        if is_initial_global_scan_message(message) and protect_initial_global_scan_message(message):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        frame_units = [
            (content_index, subindex)
            for content_index, item in enumerate(content)
            if isinstance(item, dict)
            for subindex in range(_content_frame_count(item))
        ]
        if len(frame_units) <= max_tool_message_frames:
            continue
        keep_unit_positions = _uniform_keep_positions(len(frame_units), max_tool_message_frames)
        keep_subindices_by_content: Dict[int, List[int]] = {}
        for unit_position, (content_index, subindex) in enumerate(frame_units):
            if unit_position in keep_unit_positions:
                keep_subindices_by_content.setdefault(int(content_index), []).append(int(subindex))
        removals: Set[int] = set()
        updated_content = list(content)
        for content_index, item in enumerate(content):
            if not isinstance(item, dict):
                continue
            frame_count = _content_frame_count(item)
            if frame_count <= 0:
                continue
            kept_subindices = keep_subindices_by_content.get(int(content_index), [])
            if not kept_subindices:
                removals.update(paired_multimodal_removal_indices(content, content_index))
                continue
            if is_video_item(item) and len(kept_subindices) < frame_count:
                updated_content[content_index] = _trim_video_item_frames(item, kept_subindices)
        if removals:
            updated_content = [item for idx, item in enumerate(updated_content) if idx not in removals]
        prepared[message_index]["content"] = updated_content
    return prepared


def cap_total_video_frames(
    messages: List[Dict[str, Any]],
    *,
    max_total_video_frames: int = 0,
    copy_messages: bool = True,
) -> List[Dict[str, Any]]:
    if copy_messages:
        prepared = copy.deepcopy(messages)
    else:
        prepared = messages
    max_total_video_frames = int(max_total_video_frames)
    if max_total_video_frames <= 0:
        return prepared

    frame_units = _collect_frame_units(prepared)
    if len(frame_units) <= max_total_video_frames:
        return prepared

    keep_unit_positions = set(range(len(frame_units) - max_total_video_frames, len(frame_units)))
    protected_unit_positions = [
        position
        for position, (message_index, _, _) in enumerate(frame_units)
        if is_initial_global_scan_message(prepared[int(message_index)])
        and protect_initial_global_scan_message(prepared[int(message_index)])
    ]
    if protected_unit_positions:
        keep_unit_positions.update(protected_unit_positions)
        overflow = len(keep_unit_positions) - max_total_video_frames
        if overflow > 0:
            removable_positions = [
                position
                for position in sorted(keep_unit_positions)
                if position not in protected_unit_positions
            ]
            for position in removable_positions[:overflow]:
                keep_unit_positions.discard(position)

    keep_subindices_by_item: Dict[Tuple[int, int], List[int]] = {}
    for position, (message_index, content_index, subindex) in enumerate(frame_units):
        if position in keep_unit_positions:
            keep_subindices_by_item.setdefault((int(message_index), int(content_index)), []).append(int(subindex))

    for message_index, message in enumerate(prepared):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        removals: Set[int] = set()
        updated_content = list(content)
        for content_index, item in enumerate(content):
            if not isinstance(item, dict):
                continue
            frame_count = _content_frame_count(item)
            if frame_count <= 0:
                continue
            kept_subindices = keep_subindices_by_item.get((int(message_index), int(content_index)), [])
            if not kept_subindices:
                removals.update(paired_multimodal_removal_indices(content, content_index))
                continue
            if is_video_item(item) and len(kept_subindices) < frame_count:
                updated_content[content_index] = _trim_video_item_frames(item, kept_subindices)
        if removals:
            updated_content = [item for idx, item in enumerate(updated_content) if idx not in removals]
        prepared[message_index]["content"] = updated_content
    return prepared


def summarize_visual_budget(messages: List[Dict[str, Any]]) -> Dict[str, int]:
    summary = {
        "message_count": 0,
        "tool_message_count": 0,
        "image_item_count": 0,
        "video_item_count": 0,
        "total_frame_count": 0,
        "user_frame_count": 0,
        "tool_frame_count": 0,
    }
    for message in list(messages or []):
        if not isinstance(message, dict):
            continue
        summary["message_count"] += 1
        role = str(message.get("role") or "")
        if role == "tool":
            summary["tool_message_count"] += 1
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            frame_count = _content_frame_count(item)
            if frame_count <= 0:
                continue
            if is_image_item(item):
                summary["image_item_count"] += 1
            elif is_video_item(item):
                summary["video_item_count"] += 1
            summary["total_frame_count"] += int(frame_count)
            if role == "tool":
                summary["tool_frame_count"] += int(frame_count)
            else:
                summary["user_frame_count"] += int(frame_count)
    return summary


def drop_oldest_history_turn(messages: List[Dict[str, Any]]) -> bool:
    _, spans = _history_turn_spans(messages)
    for start, end in spans:
        if _span_contains_protected_initial_scan(messages, start=start, end=end):
            continue
        del messages[start:end]
        return True
    return False


def prune_stale_text_history(
    messages: List[Dict[str, Any]],
    *,
    keep_recent_text_messages: int = 0,
    copy_messages: bool = True,
) -> List[Dict[str, Any]]:
    if copy_messages:
        prepared = copy.deepcopy(messages)
    else:
        prepared = messages
    if int(keep_recent_text_messages) <= 0:
        return prepared

    prefix_end, spans = _history_turn_spans(prepared)
    if len(spans) <= int(keep_recent_text_messages):
        return prepared
    keep_span_indices = {
        span_index
        for span_index, (start, end) in enumerate(spans)
        if _span_contains_protected_initial_scan(prepared, start=start, end=end)
    }
    keep_span_indices.update(range(max(0, len(spans) - int(keep_recent_text_messages)), len(spans)))

    retained_messages = prepared[:prefix_end]
    for span_index, (start, end) in enumerate(spans):
        if span_index not in keep_span_indices:
            continue
        retained_messages.extend(prepared[start:end])
    return retained_messages


def prune_stale_tool_images(
    messages: List[Dict[str, Any]],
    *,
    keep_recent_tool_image_messages: int = 0,
    copy_messages: bool = True,
) -> List[Dict[str, Any]]:
    if copy_messages:
        prepared = copy.deepcopy(messages)
    else:
        prepared = messages
    if int(keep_recent_tool_image_messages) <= 0:
        return prepared
    # Do not hard-drop whole older tool messages here. Let later frame/image
    # budget stages prune the oldest visual items incrementally instead.
    return prepared


def cap_total_images(
    messages: List[Dict[str, Any]],
    *,
    max_total_images: int = 0,
    copy_messages: bool = True,
) -> List[Dict[str, Any]]:
    if copy_messages:
        prepared = copy.deepcopy(messages)
    else:
        prepared = messages
    if int(max_total_images) <= 0:
        return prepared

    image_positions: List[Tuple[int, int]] = []
    for message_index, message in enumerate(prepared):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for content_index, item in enumerate(content):
            if is_image_item(item):
                image_positions.append((message_index, content_index))

    protected_positions = _protected_scan_timeline_image_positions(prepared)
    removable_positions = [position for position in image_positions if position not in protected_positions]
    overflow = min(len(removable_positions), len(image_positions) - int(max_total_images))
    if overflow <= 0:
        return prepared

    removals_by_message: Dict[int, set[int]] = {}
    for message_index, content_index in removable_positions[:overflow]:
        content = list(prepared[message_index].get("content", []))
        removals_by_message.setdefault(message_index, set()).update(
            paired_multimodal_removal_indices(content, content_index)
        )

    for message_index, removals in removals_by_message.items():
        content = list(prepared[message_index].get("content", []))
        prepared[message_index]["content"] = [
            item for idx, item in enumerate(content) if idx not in removals
        ]
    return prepared


def apply_message_budget(
    messages: List[Dict[str, Any]],
    *,
    keep_recent_text_messages: int = 0,
    keep_recent_tool_image_messages: int = 0,
    max_total_images: int = 0,
    max_tool_message_frames: int = 0,
    max_total_video_frames: int = 0,
    copy_messages: bool = True,
) -> List[Dict[str, Any]]:
    prepared = prune_stale_text_history(
        messages,
        keep_recent_text_messages=keep_recent_text_messages,
        copy_messages=copy_messages,
    )
    prepared = prune_stale_tool_images(
        prepared,
        keep_recent_tool_image_messages=keep_recent_tool_image_messages,
        copy_messages=False,
    )
    prepared = cap_total_images(
        prepared,
        max_total_images=max_total_images,
        copy_messages=False,
    )
    prepared = cap_tool_message_frames(
        prepared,
        max_tool_message_frames=max_tool_message_frames,
        copy_messages=False,
    )
    prepared = cap_total_video_frames(
        prepared,
        max_total_video_frames=max_total_video_frames,
        copy_messages=False,
    )
    return prepared
