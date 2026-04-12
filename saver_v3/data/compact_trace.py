"""Compact-trace replay helpers that stay independent from the heavier rollout stack."""

from __future__ import annotations

import copy
import json
from typing import Any, Optional

from saver_v3.data.prepared_schema import validate_prepared_row


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def compact_trace_row_to_runtime_record(row: dict[str, Any]) -> dict[str, Any]:
    normalized = validate_prepared_row(row)
    return {
        "video_id": normalized.get("video_id"),
        "video_path": normalized.get("video_path"),
        "split": normalized.get("split"),
        "source": normalized.get("source"),
        "video_meta": copy.deepcopy(normalized.get("video_meta") or {}),
        "scene": copy.deepcopy(normalized.get("scene") or {}),
        "agent_task": copy.deepcopy(normalized.get("agent_task") or {}),
        "structured_target": copy.deepcopy(normalized.get("structured_target") or {}),
        "tool_io": copy.deepcopy(normalized.get("tool_io") or {}),
        "label": copy.deepcopy(normalized.get("label") or {}),
        "temporal": copy.deepcopy(normalized.get("temporal") or {}),
        "evidence": copy.deepcopy(normalized.get("evidence") or {}),
        "language": copy.deepcopy(normalized.get("language") or {}),
        "qa_pairs": copy.deepcopy(normalized.get("qa_pairs") or []),
        "proposal_supervision": copy.deepcopy(normalized.get("proposal_supervision") or {}),
        "oracle_sft": {
            "trajectory": copy.deepcopy(normalized.get("oracle_trajectory") or []),
            "final_decision": copy.deepcopy(normalized.get("oracle_final_decision") or {}),
        },
    }


def _default_task_prompt(row: dict[str, Any]) -> str:
    agent_task = row.get("agent_task") or {}
    prompt = str(agent_task.get("task_prompt") or "").strip()
    return prompt or "Inspect the clip and gather evidence before finalizing the case."


def _initial_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are SAVER. Use tools before finalizing the case."}],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _default_task_prompt(row)},
                {"type": "video", "video": row.get("video_path")},
            ],
        },
    ]


def build_compact_trace_step_response(step: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
    tool_name = str(step.get("tool") or "").strip()
    if not tool_name:
        raise ValueError("Compact-trace step is missing tool name.")
    arguments = copy.deepcopy(step.get("arguments") or {})
    think_text = f"I should call {tool_name} based on the evidence collected so far."
    response_text = f"<think>{think_text}</think><tool_call>{_json_dumps({'name': tool_name, 'arguments': arguments})}</tool_call>"
    return response_text, arguments, tool_name


def _tool_observation_message(tool_name: str, observation: Any) -> dict[str, Any]:
    if isinstance(observation, str):
        text = observation
    else:
        text = json.dumps(observation if observation is not None else {}, ensure_ascii=False, sort_keys=True)
    return {
        "role": "tool",
        "name": tool_name,
        "content": [{"type": "text", "text": text}],
    }


def replay_compact_trace_messages(
    row: dict[str, Any],
    *,
    stop_before_step_index: Optional[int] = None,
    include_tool_observations: bool = True,
) -> list[dict[str, Any]]:
    normalized = validate_prepared_row(row)
    messages = _initial_messages(normalized)
    for step_index, step in enumerate(list(normalized.get("oracle_trajectory") or []), start=1):
        if stop_before_step_index is not None and int(step_index) >= int(stop_before_step_index):
            break
        response_text, _arguments, tool_name = build_compact_trace_step_response(step)
        messages.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})
        if include_tool_observations:
            messages.append(_tool_observation_message(tool_name, step.get("observation") or {}))
    return copy.deepcopy(messages)
