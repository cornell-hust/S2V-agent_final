from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Tuple


_TIMESTAMP_ONLY_RE = re.compile(r"^\s*\d+(?:\.\d+)?s\s*$")


def is_image_item(item: Any) -> bool:
    return isinstance(item, dict) and item.get("type") == "image" and ("image" in item or "image_ref" in item)


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


def drop_oldest_history_turn(messages: List[Dict[str, Any]]) -> bool:
    _, spans = _history_turn_spans(messages)
    if not spans:
        return False
    start, end = spans[0]
    del messages[start:end]
    return True


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
    keep_start = int(spans[-int(keep_recent_text_messages)][0])
    return prepared[:prefix_end] + prepared[keep_start:]


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

    image_tool_message_indices = [
        message_index
        for message_index, message in enumerate(prepared)
        if message.get("role") == "tool"
        and any(
            item.get("type") in {"image", "video"}
            and (
                "image" in item
                or "image_ref" in item
                or "video" in item
                or "video_ref" in item
            )
            for item in list(message.get("content", []))
            if isinstance(item, dict)
        )
    ]
    keep_budget = int(keep_recent_tool_image_messages)
    keep_indices: set[int] = set()
    pinned_scan_index = None
    for message_index in reversed(image_tool_message_indices):
        message = prepared[message_index]
        if str(message.get("name") or "").strip() == "scan_timeline":
            pinned_scan_index = int(message_index)
            break
    if keep_budget > 0 and pinned_scan_index is not None:
        keep_indices.add(int(pinned_scan_index))
        keep_budget -= 1
    for message_index in reversed(image_tool_message_indices):
        if keep_budget <= 0:
            break
        if int(message_index) in keep_indices:
            continue
        keep_indices.add(int(message_index))
        keep_budget -= 1

    for message_index in image_tool_message_indices:
        if message_index in keep_indices:
            continue
        content = list(prepared[message_index].get("content", []))
        removals: set[int] = set()
        for content_index, item in enumerate(content):
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type not in {"image", "video"}:
                continue
            removals.update(paired_multimodal_removal_indices(content, content_index))
        prepared[message_index]["content"] = [
            item for idx, item in enumerate(content) if idx not in removals
        ]
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

    overflow = len(image_positions) - int(max_total_images)
    if overflow <= 0:
        return prepared

    removals_by_message: Dict[int, set[int]] = {}
    for message_index, content_index in image_positions[:overflow]:
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
    return prepared
