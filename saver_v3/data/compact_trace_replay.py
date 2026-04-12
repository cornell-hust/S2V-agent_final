from __future__ import annotations

import warnings as _warnings
_warnings.warn(
    "saver_v3.data.compact_trace_replay generates SYNTHETIC tool observations. "
    "For training, use saver_v3.sft.training which executes real tool calls. "
    "This module is for debug/preview only.",
    DeprecationWarning,
    stacklevel=2,
)

import copy
import json
from typing import Any, Dict, Iterable, List, Sequence

from .episode_views import EpisodeMessagesExample


def _json_compact(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _tool_call_block(tool_name: str, arguments: Dict[str, Any]) -> str:
    payload = {"name": str(tool_name), "arguments": dict(arguments or {})}
    return f"<tool_call>{_json_compact(payload)}</tool_call>"


def _answer_block(payload: Dict[str, Any]) -> str:
    return f"<answer>{_json_compact(payload)}</answer>"


def _build_default_answer_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    final_decision = dict(row.get("oracle_final_decision") or row.get("structured_target") or {})
    existence = str(final_decision.get("existence") or final_decision.get("decision") or "normal")
    category = str(final_decision.get("category") or "normal")
    summary = str(final_decision.get("summary") or "")
    if not summary:
        if existence == "anomaly":
            summary = f"Detected anomaly category: {category}."
        else:
            summary = "No actionable anomaly detected."
    return {
        "decision": {"existence": existence, "category": category},
        "summary": summary,
        "event_chain_summary": dict(final_decision.get("event_chain_summary") or {}),
        "qa_focus_answers": dict(final_decision.get("qa_focus_answers") or {}),
    }


def _observation_for_step(step: Dict[str, Any], row: Dict[str, Any]) -> str:
    tool_name = str(step.get("tool") or "")
    arguments = dict(step.get("arguments") or {})
    if tool_name == "scan_timeline":
        return (
            f"Scanned interval [{arguments.get('start_sec', 0.0)}, {arguments.get('end_sec', 0.0)}] seconds. "
            "Use the returned overview to decide whether to continue searching, verify, or finalize."
        )
    if tool_name == "seek_evidence":
        query = str(arguments.get("query") or "").strip()
        role = str(arguments.get("role") or "").strip()
        return (
            f"Searched for {role or 'evidence'} using query: {query or '(empty)'}. "
            "Review the evidence window before the next action."
        )
    if tool_name == "verify_hypothesis":
        feedback = dict(step.get("oracle_verifier_feedback") or {})
        return json.dumps(feedback, ensure_ascii=False, separators=(",", ":")) if feedback else "Verification completed."
    if tool_name == "finalize_case":
        return json.dumps({"status": "case_finalized", "note": "Final decision recorded."})
    return f"Tool {tool_name} executed."


def build_initial_messages(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    task_prompt = str(((row.get("agent_task") or {}).get("task_prompt") or "")).strip()
    if not task_prompt:
        task_prompt = "Determine whether the video contains an actionable anomaly, gather evidence, verify the hypothesis, and finalize the case."
    video_path = str(row.get("video_path") or "").strip()
    tool_schemas = list(((row.get("tool_io") or {}).get("allowed_tools") or []))
    system_text = (
        "You are the SAVER policy. Use exactly one structured block per response. "
        "Emit <tool_call>{...}</tool_call> for tool use and <answer>{...}</answer> only after the rollout is complete. "
        f"Available tools: {', '.join(tool_schemas) if tool_schemas else 'scan_timeline, seek_evidence, verify_hypothesis, finalize_case'}."
    )
    user_content: List[Dict[str, Any]] = []
    if video_path:
        user_content.append({"type": "video", "video": video_path})
    user_content.append({"type": "text", "text": task_prompt})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user", "content": user_content},
    ]


def expand_compact_trace_row_to_sft_steps(
    row: Dict[str, Any],
    *,
    include_terminal_answer: bool = False,
) -> List[EpisodeMessagesExample]:
    messages = build_initial_messages(row)
    step_examples: List[EpisodeMessagesExample] = []
    for step_index, step in enumerate(list(row.get("oracle_trajectory") or []), start=1):
        tool_name = str(step.get("tool") or "").strip()
        arguments = dict(step.get("arguments") or {})
        target_response = _tool_call_block(tool_name, arguments)
        candidate_messages = copy.deepcopy(messages)
        candidate_messages.append({"role": "assistant", "content": [{"type": "text", "text": target_response}]})
        step_examples.append(
            EpisodeMessagesExample(
                messages=candidate_messages,
                video_id=str(row.get("video_id") or ""),
                split=str(row.get("split") or ""),
                target_action="tool_call",
                tool_name=tool_name,
                target_response=target_response,
                metadata={"step_index": step_index},
            )
        )
        messages = copy.deepcopy(candidate_messages)
        messages.append(
            {
                "role": "tool",
                "name": tool_name,
                "content": [{"type": "text", "text": _observation_for_step(step, row)}],
            }
        )
    if include_terminal_answer:
        answer_payload = _build_default_answer_payload(row)
        target_response = _answer_block(answer_payload)
        candidate_messages = copy.deepcopy(messages)
        candidate_messages.append({"role": "assistant", "content": [{"type": "text", "text": target_response}]})
        step_examples.append(
            EpisodeMessagesExample(
                messages=candidate_messages,
                video_id=str(row.get("video_id") or ""),
                split=str(row.get("split") or ""),
                target_action="answer",
                tool_name="",
                target_response=target_response,
                metadata={"step_index": len(step_examples) + 1},
            )
        )
    return step_examples


class CompactTraceStepSFTDataset:
    def __init__(self, rows: Sequence[Dict[str, Any]], *, include_terminal_answer: bool = False):
        self.rows = list(rows)
        self.include_terminal_answer = bool(include_terminal_answer)
        self.examples: List[EpisodeMessagesExample] = []
        for row in self.rows:
            self.examples.extend(
                expand_compact_trace_row_to_sft_steps(row, include_terminal_answer=self.include_terminal_answer)
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[int(index)]
        return {
            "messages": copy.deepcopy(example.messages),
            "video_id": example.video_id,
            "split": example.split,
            "target_action": example.target_action,
            "tool_name": example.tool_name,
            "target_response": example.target_response,
            "metadata": copy.deepcopy(example.metadata),
        }
