"""Build lightweight SAVER rollout items from compact-trace rows."""

from __future__ import annotations

import copy
from typing import Any, Iterable, List

from saver_v3.data.prepared_schema import validate_compact_trace_row
from saver_v3.core.prompts import build_system_prompt, build_user_prompt
from saver_v3.core.tool_registry import DEFAULT_TOOL_NAMES, get_tool_schemas


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clean_allowed_tools(tool_io: dict[str, Any]) -> list[str]:
    raw_tools = tool_io.get("allowed_tools") or list(DEFAULT_TOOL_NAMES)
    cleaned: list[str] = []
    for tool_name in list(raw_tools or []):
        text = str(tool_name or "").strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned or list(DEFAULT_TOOL_NAMES)


def _build_tool_io(row: dict[str, Any]) -> dict[str, Any]:
    tool_io = copy.deepcopy(row.get("tool_io") or {})
    allowed_tools = _clean_allowed_tools(tool_io)
    tool_schemas = get_tool_schemas(
        finalize_case_schema=copy.deepcopy(tool_io.get("finalize_case_schema") or {}),
        allowed_tools=allowed_tools,
    )
    function_schemas = [copy.deepcopy(tool.get("function") or {}) for tool in tool_schemas]

    augmented_finalize_schema = copy.deepcopy(tool_io.get("finalize_case_schema") or {})
    for schema in function_schemas:
        if schema.get("name") == "finalize_case":
            augmented_finalize_schema = copy.deepcopy(schema.get("parameters") or {})
            break

    tool_io["allowed_tools"] = allowed_tools
    tool_io["tool_schemas"] = copy.deepcopy(tool_schemas)
    tool_io["function_schemas"] = function_schemas
    tool_io["finalize_case_schema"] = augmented_finalize_schema
    return tool_io


def _default_task_prompt(row: dict[str, Any]) -> str:
    prompt = str(((row.get("agent_task") or {}).get("task_prompt") or "")).strip()
    return prompt or "Determine whether the video contains an actionable anomaly, gather evidence, verify the hypothesis, and finalize the case."


def _resolve_duration(row: dict[str, Any]) -> float:
    video_meta = row.get("video_meta") or {}
    temporal = row.get("temporal") or {}
    duration = _coerce_float(video_meta.get("duration_sec"), 0.0)
    if duration > 0:
        return duration
    for key in ("duration_sec", "end_sec"):
        duration = _coerce_float(temporal.get(key), 0.0)
        if duration > 0:
            return duration
    return 0.0


def _resolve_fps(row: dict[str, Any]) -> float:
    video_meta = row.get("video_meta") or {}
    fps = _coerce_float(video_meta.get("fps"), 1.0)
    return fps if fps > 0 else 1.0


def _build_multimodal_cache(row: dict[str, Any], *, tool_io: dict[str, Any]) -> dict[str, Any]:
    video_meta = copy.deepcopy(row.get("video_meta") or {})
    duration = _resolve_duration(row)
    fps = _resolve_fps(row)
    frame_indices = [int(index) for index in list(row.get("frame_indices") or [])]
    return {
        "video": None,
        "embedding": None,
        "fps": float(fps),
        "duration": float(duration),
        "question": _default_task_prompt(row),
        "structured_target": copy.deepcopy(row.get("structured_target") or {}),
        "tool_io": copy.deepcopy(tool_io),
        "video_path": str(row.get("video_path") or ""),
        "video_meta": video_meta,
        "frame_indices": frame_indices,
        "preview_frames": None,
        "preview_timestamps": [],
        "preview_frame_indices": [],
        "proposal_runtime": None,
        "strict_feature_guided_proposal": False,
    }


def _build_user_content(row: dict[str, Any], user_prompt: str) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    video_path = str(row.get("video_path") or "").strip()
    if video_path:
        content.append({"type": "video", "video": video_path})
    content.append({"type": "text", "text": user_prompt})
    return content


def _build_messages(row: dict[str, Any], *, tool_io: dict[str, Any]) -> list[dict[str, Any]]:
    system_prompt = build_system_prompt(tool_io.get("function_schemas") or tool_io.get("tool_schemas") or [])
    user_prompt = build_user_prompt(row, preview_available=False)
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": _build_user_content(row, user_prompt)},
    ]


def build_runtime_item_from_compact_trace_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a v2-rollout-compatible item from a validated compact-trace row.

    The builder intentionally avoids video decoding, feature caches, torch, and GPU-only
    dependencies. Tool execution can still sample textual timestamps from duration/fps.
    """
    normalized = validate_compact_trace_row(row)
    tool_io = _build_tool_io(normalized)
    item = copy.deepcopy(normalized)
    item["video"] = str(normalized.get("video_path") or "")
    item["multimodal_cache"] = _build_multimodal_cache(normalized, tool_io=tool_io)
    item["messages"] = _build_messages(normalized, tool_io=tool_io)
    return item


def build_runtime_items_from_compact_trace_rows(rows: Iterable[dict[str, Any]]) -> List[dict[str, Any]]:
    return [build_runtime_item_from_compact_trace_row(row) for row in rows]
