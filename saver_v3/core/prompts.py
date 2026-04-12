from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, Iterable, List, Sequence

from saver_v3.data.config import PromptConfig
from saver_v3.core.protocol_guidance import build_counterfactual_type_legend


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_public_case_id(record: Dict[str, Any]) -> str:
    raw_identifier = (
        str(record.get("video_id") or "").strip()
        or str(record.get("video_path") or "").strip()
        or str(record.get("file_name") or "").strip()
    )
    if not raw_identifier:
        return "case_unknown"
    digest = hashlib.sha1(raw_identifier.encode("utf-8")).hexdigest()[:10]
    return f"case_{digest}"


def _coerce_function_schemas(tool_schemas_or_names: Sequence[Any]) -> List[Dict[str, Any]]:
    function_schemas: List[Dict[str, Any]] = []
    for item in list(tool_schemas_or_names or []):
        if isinstance(item, dict) and isinstance(item.get("function"), dict):
            function_schemas.append(copy.deepcopy(item["function"]))
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            function_schemas.append(copy.deepcopy(item))
    return function_schemas


def _extract_text_from_tool_prompt_message(message: Any) -> str:
    if hasattr(message, "model_dump"):
        message = message.model_dump()
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return "\n".join(part for part in text_parts if part.strip())
    return ""


def _build_qwen_agent_tool_use_prompt(function_schemas: Sequence[Dict[str, Any]]) -> str:
    try:
        from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import NousFnCallPrompt
    except Exception:
        return ""
    try:
        messages = NousFnCallPrompt().preprocess_fncall_messages(
            messages=[],
            functions=list(function_schemas),
            lang=None,
        )
    except Exception:
        return ""
    if not messages:
        return ""
    return _extract_text_from_tool_prompt_message(messages[0]).strip()


def _build_fallback_tool_use_prompt(function_schemas: Sequence[Dict[str, Any]]) -> str:
    serialized = [
        _json_dumps({"type": "function", "function": schema})
        for schema in function_schemas
    ]
    if not serialized:
        return ""
    return (
        "Tool function schemas:\n"
        + "\n".join(serialized)
    )


def build_tool_use_prompt(tool_schemas_or_names: Sequence[Any]) -> str:
    function_schemas = _coerce_function_schemas(tool_schemas_or_names)
    if function_schemas:
        qwen_agent_prompt = _build_qwen_agent_tool_use_prompt(function_schemas)
        if qwen_agent_prompt:
            return qwen_agent_prompt
        return _build_fallback_tool_use_prompt(function_schemas)
    tool_names = [str(item) for item in list(tool_schemas_or_names or []) if str(item).strip()]
    if not tool_names:
        return ""
    return "Allowed tools: " + ", ".join(tool_names)


def _extract_finalize_case_required_fields(function_schemas: Sequence[Dict[str, Any]]) -> List[str]:
    for schema in function_schemas:
        if str(schema.get("name") or "") != "finalize_case":
            continue
        parameters = schema.get("parameters") or {}
        required = list(parameters.get("required") or [])
        return [str(field_name) for field_name in required if str(field_name).strip()]
    return []


def _finalize_case_has_counterfactual_enum(function_schemas: Sequence[Dict[str, Any]]) -> bool:
    for schema in function_schemas:
        if str(schema.get("name") or "") != "finalize_case":
            continue
        parameters = schema.get("parameters") or {}
        properties = parameters.get("properties") or {}
        counterfactual_schema = properties.get("counterfactual_type") or {}
        if list(counterfactual_schema.get("enum") or []):
            return True
    return False


def build_system_prompt(tool_schemas_or_names: Sequence[Any]) -> str:
    tool_contract = (
        'Valid tool format example: <think>inspect</think><tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":4.0,"num_frames":4}}</tool_call>. '
        "Do not describe the tool call in plain English. "
        "Do not output bare tool names. "
        "Do not invent tool names or argument keys."
    )
    function_schemas = _coerce_function_schemas(tool_schemas_or_names)
    tool_use_prompt = build_tool_use_prompt(function_schemas or tool_schemas_or_names)
    finalize_required_fields = _extract_finalize_case_required_fields(function_schemas)
    finalize_contract = ""
    if finalize_required_fields:
        finalize_contract = (
            " When you call finalize_case, include every required field exactly once: "
            + ", ".join(finalize_required_fields)
            + "."
        )
    counterfactual_contract = ""
    if _finalize_case_has_counterfactual_enum(function_schemas):
        counterfactual_contract = (
            " For counterfactual_type, use one allowed enum value such as "
            + build_counterfactual_type_legend(include_descriptions=False)
            + "."
        )
    return (
        "You are SAVER, an active video anomaly agent. "
        "Think carefully before each action. "
        "Use tools to inspect evidence, verify the current hypothesis, "
        "and only finalize after you have enough support. "
        "The main rollout ends immediately after finalize_case returns a valid structured decision. "
        "Prioritize a compact valid finalize_case with the core decision fields. "
        "Optional summary, rationale, event_chain_summary, and qa_focus_answers may be included only when they stay compact; omit them rather than risk truncation. "
        "Do not output <answer> during the main rollout. "
        "For anomaly cases, try to complete the event chain from precursor to trigger to confirmation whenever those stages are required. "
        "When you call seek_evidence, make query a concrete visual event description rather than a category word. "
        "Prefer short grounded phrases such as actions, interactions, or scene-specific evidence, for example 'person raises a fist toward another person in the aisle', not just 'assault'. "
        "Use the search anchor in tool observations to decide what stage to search next; do not stuff long protocol explanations into the query itself. "
        "verify_hypothesis must report selected_window_ids plus verification_decision, recommended_action, "
        "sufficiency_score, necessity_score, finalize_readiness_score, counterfactual_faithfulness, covered_stages, missing_required_stages, and stage_selected_moment_ids. "
        "If you use a tool, respond as <think>...</think><tool_call>{...}</tool_call>. "
        "Do not skip finalize_case when the protocol requires a structured final decision. "
        f"{finalize_contract}"
        f"{counterfactual_contract}"
        f"{tool_contract}\n"
        f"{tool_use_prompt}"
    )


def build_user_prompt(
    record: Dict,
    *,
    preview_available: bool = False,
    prompt_config: PromptConfig | None = None,
) -> str:
    prompt_config = prompt_config or PromptConfig()
    scene = record.get("scene", {}).get("scenario", "unknown")
    duration = record.get("video_meta", {}).get("duration_sec", "unknown")
    task_prompt = record.get("agent_task", {}).get("task_prompt", "")
    criteria = record.get("agent_task", {}).get("success_criteria", [])
    criteria_text = "\n".join(f"- {item}" for item in criteria) or "- none provided"
    public_case_id = build_public_case_id(record)
    prompt = prompt_config.initial_user_template.format(
        video_id=public_case_id,
        public_case_id=public_case_id,
        raw_video_id=public_case_id,
        scene=scene,
        duration=duration,
        task_prompt=task_prompt,
        criteria_text=criteria_text,
    )
    if preview_available and prompt_config.preview_instruction:
        prompt += f"\n{prompt_config.preview_instruction}"
    return prompt


def build_tool_response_prompt(
    timestamps: Iterable[str] | str,
    *,
    question: str = "",
    duration: float | None = None,
    prompt_config: PromptConfig | None = None,
) -> str:
    prompt_config = prompt_config or PromptConfig()
    if isinstance(timestamps, str):
        timestamps_text = timestamps
    else:
        timestamps_text = ", ".join(timestamps)
    prompt = prompt_config.tool_response_template.format(timestamps=timestamps_text)
    prompt += (
        'Respond with exactly one <tool_call>{"name":"...","arguments":{...}}</tool_call>.\n'
        "Do not output <answer> during the main rollout.\n"
        "Do not describe the tool call in plain English.\n"
    )
    return prompt
