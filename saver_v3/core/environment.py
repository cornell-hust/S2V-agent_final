from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from saver_v3.core.schema import SaverEnvironmentState
from saver_v3.core.tool_registry import execute_tool_call


INVALID_TOOL_CALL_PROMPT = (
    "The previous response is invalid. Retry immediately with exactly one of these formats: "
    '<tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":4.0}}</tool_call> '
    'or <tool_call>{"name":"verify_hypothesis","arguments":{"verification_mode":"final_check","claim":{"existence":"anomaly","category":"assault"},"selected_window_ids":["w0001"],"verification_decision":"sufficient","recommended_action":"finalize","sufficiency_score":0.8,"necessity_score":0.8,"finalize_readiness_score":0.8,"counterfactual_faithfulness":0.5}}</tool_call> '
    "Do not output the terminal answer unless the latest observation explicitly requested it. "
    "Do not describe the intended tool call in plain English. Do not output bare tool names."
)
INVALID_ANSWER_PROMPT = (
    "The previous <answer> payload is invalid. Retry immediately with exactly one structured JSON object "
    'inside <answer></answer>, for example <answer>{"decision":{"existence":"normal","category":"normal"},"summary":"No anomaly found.","event_chain_summary":{"precursor":"","trigger":"","confirmation":""},"qa_focus_answers":{"existence":"No. No anomaly is visible in this video.","category":"The video is normal.","temporal":"There is no anomaly interval in this video."}}</answer>. '
    "Do not output plain text inside <answer>. Do not describe the answer in prose."
)
OPTIONAL_FINALIZE_SEMANTIC_FIELDS = (
    "summary",
    "rationale",
    "event_chain_summary",
    "qa_focus_answers",
)


def cleanup_llm_response(response_str: str) -> str:
    tagged_blocks = list(re.finditer(r"<(tool_call|answer)>(.*?)</\1>", response_str, re.DOTALL))
    if tagged_blocks:
        return tagged_blocks[-1].group(0)
    if "<think>" in response_str:
        response_str = "<think>" + response_str.split("<think>")[-1]
    return response_str


def _finalize_retry_prompt(multimodal_cache: Dict[str, Any] | None = None) -> str:
    schema = ((multimodal_cache or {}).get("tool_io") or {}).get("finalize_case_schema") or {}
    required = [str(field_name) for field_name in list(schema.get("required") or []) if str(field_name).strip()]
    required_text = ", ".join(required) if required else "the required finalize_case fields"
    return (
        "The previous finalize_case call was invalid. Retry immediately with exactly one "
        '<tool_call>{"name":"finalize_case","arguments":{...}}</tool_call>. '
        f"finalize_case requires: {required_text}. "
        "Prioritize a compact finalize_case that closes cleanly. "
        "If optional summary, rationale, event_chain_summary, or qa_focus_answers would make the tool call too long, omit them. "
        "Do not output <answer> yet. Do not describe the intended tool call in plain English."
    )


def _verify_retry_prompt(state: SaverEnvironmentState | None = None) -> str:
    evidence_window_ids = [
        str(entry.get("window_id")).strip()
        for entry in (state.evidence_ledger if state is not None else [])
        if str(entry.get("window_id") or "").strip()
    ]
    selected_window_ids = evidence_window_ids[:1] or ["w0001"]
    example_arguments = {
        "verification_mode": "final_check",
        "selected_window_ids": selected_window_ids,
        "verification_decision": "sufficient",
        "recommended_action": "finalize",
        "sufficiency_score": 0.8,
        "necessity_score": 0.8,
        "finalize_readiness_score": 0.8,
        "counterfactual_faithfulness": 0.5,
    }
    return (
        "The previous verify_hypothesis call was invalid. Retry immediately with exactly one "
        f'<tool_call>{{"name":"verify_hypothesis","arguments":{json.dumps(example_arguments, ensure_ascii=False, separators=(",", ":"))}}}</tool_call>. '
        "selected_window_ids must reference currently available evidence windows. "
        "Do not output <answer> yet. Do not describe the intended tool call in plain English."
    )


def invalid_tool_call_message(
    *,
    tool_name: str | None = None,
    multimodal_cache: Dict[str, Any] | None = None,
    state: SaverEnvironmentState | None = None,
) -> Dict[str, Any]:
    if tool_name == "finalize_case":
        prompt_text = _finalize_retry_prompt(multimodal_cache)
    elif tool_name == "verify_hypothesis":
        prompt_text = _verify_retry_prompt(state)
    else:
        prompt_text = INVALID_TOOL_CALL_PROMPT
    return {
        "role": "tool",
        "name": "parse_error",
        "content": [{"type": "text", "text": prompt_text}],
    }


def invalid_answer_message() -> Dict[str, Any]:
    return {
        "role": "tool",
        "name": "parse_error",
        "content": [{"type": "text", "text": INVALID_ANSWER_PROMPT}],
    }


def _parse_sec_value(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text.endswith("s"):
        text = text[:-1].strip()
    if ":" in text:
        parts = text.split(":")
        try:
            total = 0.0
            for part in parts:
                total = total * 60.0 + float(part)
            return total
        except Exception:
            return value
    try:
        return float(text)
    except Exception:
        return value


def _normalize_tool_arguments(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(arguments)
    if "start_time" in normalized and "start_sec" not in normalized:
        normalized["start_sec"] = _parse_sec_value(normalized.pop("start_time"))
    if "end_time" in normalized and "end_sec" not in normalized:
        normalized["end_sec"] = _parse_sec_value(normalized.pop("end_time"))
    if "start_sec" in normalized:
        normalized["start_sec"] = _parse_sec_value(normalized["start_sec"])
    if "end_sec" in normalized:
        normalized["end_sec"] = _parse_sec_value(normalized["end_sec"])
    if "earliest_actionable_sec" in normalized:
        normalized["earliest_actionable_sec"] = _parse_sec_value(normalized["earliest_actionable_sec"])
    if "earliest_alert_sec" in normalized:
        normalized["earliest_alert_sec"] = _parse_sec_value(normalized["earliest_alert_sec"])
    return normalized


def _parse_tool_payload(content: str) -> Dict[str, Any] | None:
    parsed: Dict[str, Any] | None = None
    try:
        candidate = json.loads(content)
    except Exception:
        candidate = None

    if candidate is None:
        repaired_content = _repair_json_object_text(content)
        if repaired_content is not None and repaired_content != content:
            try:
                candidate = json.loads(repaired_content)
            except Exception:
                candidate = None

    if isinstance(candidate, dict):
        if "name" in candidate:
            parsed = {"name": candidate.get("name"), "arguments": candidate.get("arguments") or {}}
        elif len(candidate) == 1:
            tool_name, arguments = next(iter(candidate.items()))
            if isinstance(arguments, dict):
                parsed = {"name": tool_name, "arguments": arguments}

    if parsed is None:
        match = re.fullmatch(r"\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(\{.*\})\s*\}", content, re.DOTALL)
        if match:
            tool_name = match.group(1)
            try:
                arguments = json.loads(match.group(2))
            except Exception:
                arguments = None
            if isinstance(arguments, dict):
                parsed = {"name": tool_name, "arguments": arguments}

    if parsed is None:
        return None

    name = str(parsed.get("name") or "").strip()
    arguments = parsed.get("arguments") or {}
    if not name or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": _normalize_tool_arguments(name, arguments)}


def _json_brace_balance(text: str) -> int:
    balance = 0
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            balance += 1
        elif char == "}":
            balance -= 1
    return balance


def _repair_json_object_text(text: str) -> str | None:
    candidate = str(text or "").strip()
    if not candidate.startswith("{"):
        return None

    repaired = candidate
    balance = _json_brace_balance(repaired)
    while balance < 0 and repaired.endswith("}"):
        repaired = repaired[:-1].rstrip()
        balance += 1
    if balance > 0:
        repaired = repaired + ("}" * balance)
    return repaired if repaired else None


def _parse_answer_payload(content: str) -> Dict[str, Any] | None:
    try:
        payload = json.loads(content)
    except Exception:
        repaired_content = _repair_json_object_text(content)
        if repaired_content is None or repaired_content == content:
            return None
        try:
            payload = json.loads(repaired_content)
        except Exception:
            return None
    if not isinstance(payload, dict):
        return None
    return payload


def _repair_truncated_finalize_tool_payload(content: str) -> Dict[str, Any] | None:
    candidate = str(content or "").strip()
    if not candidate.startswith("{") or "finalize_case" not in candidate or '"arguments"' not in candidate:
        return None
    cut_positions: List[int] = []
    for field_name in OPTIONAL_FINALIZE_SEMANTIC_FIELDS:
        for marker in (f',"{field_name}":', f', "{field_name}":'):
            index = candidate.find(marker)
            if index >= 0:
                cut_positions.append(index)
    for cut_index in sorted(set(cut_positions)):
        truncated = candidate[:cut_index].rstrip()
        repaired_candidate = _repair_json_object_text(truncated + "}}")
        if not repaired_candidate:
            continue
        try:
            parsed = json.loads(repaired_candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_open_structured_fragment(prediction: str) -> Tuple[str | None, str]:
    latest_tag = None
    latest_index = -1
    for tag_name in ("tool_call", "answer"):
        open_tag = f"<{tag_name}>"
        index = prediction.rfind(open_tag)
        if index > latest_index:
            latest_index = index
            latest_tag = tag_name
    if latest_tag is None or latest_index < 0:
        return None, ""
    content_start = latest_index + len(f"<{latest_tag}>")
    return latest_tag, prediction[content_start:].strip()


def parse_actions_and_contents(predictions: List[Any]) -> Tuple[List[str | None], List[Any]]:
    actions: List[str | None] = []
    contents: List[Any] = []
    for prediction in predictions:
        stripped_prediction = re.sub(r"<think>.*?</think>", "", prediction, flags=re.DOTALL)
        matches = re.findall(r"<(tool_call|answer)>(.*?)</\1>", stripped_prediction, re.DOTALL)
        if not matches:
            matches = re.findall(r"<(tool_call|answer)>(.*?)</\1>", prediction, re.DOTALL)
        match = matches[-1] if matches else None
        if not match:
            fallback_action, fallback_content = _extract_open_structured_fragment(prediction)
            if fallback_action is None:
                actions.append(None)
                contents.append("")
                continue
            match = (fallback_action, fallback_content)
        action = match[0]
        content = match[1].strip()
        if action == "tool_call":
            func = _parse_tool_payload(content)
            if func is None:
                repaired_payload = _repair_truncated_finalize_tool_payload(content)
                if isinstance(repaired_payload, dict):
                    repaired_name = str(repaired_payload.get("name") or "").strip()
                    repaired_arguments = repaired_payload.get("arguments") or {}
                    if repaired_name and isinstance(repaired_arguments, dict):
                        func = {
                            "name": repaired_name,
                            "arguments": _normalize_tool_arguments(repaired_name, repaired_arguments),
                        }
            if func is None:
                actions.append(None)
                contents.append("")
            else:
                actions.append("tool_call")
                contents.append({"type": "function", "function": func})
        else:
            payload = _parse_answer_payload(content)
            if payload is None:
                actions.append("invalid_answer")
                contents.append(content)
            else:
                actions.append("answer")
                contents.append(content)
    return actions, contents


class SaverVideoInteraction:
    """Step-level environment wrapper for SAVER-style tool use."""

    def execute_predictions(
        self,
        predictions: List[str],
        multimodal_cache_batch: List[Dict[str, Any]],
        states: List[SaverEnvironmentState],
        active_mask: List[bool],
    ) -> Tuple[List[Any], List[int], List[int], List[int], List[SaverEnvironmentState]]:
        cleaned = [cleanup_llm_response(prediction) for prediction in predictions]
        actions, contents = parse_actions_and_contents(cleaned)

        next_obs: List[Any] = []
        dones: List[int] = []
        valid_actions: List[int] = []
        is_search: List[int] = []
        next_states: List[SaverEnvironmentState] = []

        for action, content, multimodal_cache, state, active in zip(
            actions, contents, multimodal_cache_batch, states, active_mask
        ):
            if not active:
                next_obs.append(None)
                dones.append(1)
                valid_actions.append(0)
                is_search.append(0)
                next_states.append(state)
                continue

            if action == "answer":
                next_obs.append(None)
                dones.append(1)
                valid_actions.append(1)
                is_search.append(0)
                next_states.append(state)
                continue

            if action == "invalid_answer":
                next_obs.append(invalid_answer_message())
                dones.append(0)
                valid_actions.append(0)
                is_search.append(0)
                next_states.append(state)
                continue

            if action == "tool_call":
                try:
                    message, next_state = execute_tool_call(content, multimodal_cache, state)
                    next_obs.append(message)
                    dones.append(0)
                    valid_actions.append(1)
                    is_search.append(1)
                    next_states.append(next_state)
                except ValueError:
                    next_obs.append(
                        invalid_tool_call_message(
                            tool_name=content["function"]["name"],
                            multimodal_cache=multimodal_cache,
                            state=state,
                        )
                    )
                    dones.append(0)
                    valid_actions.append(0)
                    is_search.append(0)
                    next_states.append(state)
                continue

            next_obs.append(invalid_tool_call_message(multimodal_cache=multimodal_cache))
            dones.append(0)
            valid_actions.append(0)
            is_search.append(0)
            next_states.append(state)

        return next_obs, dones, valid_actions, is_search, next_states
