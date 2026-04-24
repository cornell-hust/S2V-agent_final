from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence


EVENT_CHAIN_STAGES: List[str] = ["precursor", "trigger", "confirmation"]
EVENT_CHAIN_ROLE_TO_STAGE: Dict[str, str] = {
    "precursor": "precursor",
    "trigger": "trigger",
    "peak": "trigger",
    "peak_action": "trigger",
    "confirmation": "confirmation",
    "aftermath": "confirmation",
}


COUNTERFACTUAL_TYPE_SPECS: List[Dict[str, str]] = [
]

COUNTERFACTUAL_TYPE_VALUES: List[str] = [spec["value"] for spec in COUNTERFACTUAL_TYPE_SPECS]
COUNTERFACTUAL_TYPE_DESCRIPTION = (
    "Choose one compact counterfactual enum that best describes the minimal change preventing the event."
)


def event_chain_stage_for_role(role: Any) -> str | None:
    return EVENT_CHAIN_ROLE_TO_STAGE.get(str(role or "").strip().lower())


def normalize_event_chain_stages(values: Iterable[Any] | None) -> List[str]:
    requested = {
        str(value).strip().lower()
        for value in list(values or [])
        if str(value).strip()
    }
    return [stage for stage in EVENT_CHAIN_STAGES if stage in requested]


def normalize_stage_selected_moment_ids(payload: Any) -> Dict[str, List[str]]:
    if not isinstance(payload, dict):
        return {}
    normalized: Dict[str, List[str]] = {}
    for stage in EVENT_CHAIN_STAGES:
        values = payload.get(stage)
        if values is None:
            continue
        if isinstance(values, (list, tuple, set)):
            moment_ids = []
            seen = set()
            for value in values:
                moment_id = str(value).strip()
                if not moment_id or moment_id in seen:
                    continue
                seen.add(moment_id)
                moment_ids.append(moment_id)
        else:
            moment_id = str(values).strip()
            moment_ids = [moment_id] if moment_id else []
        if moment_ids:
            normalized[stage] = moment_ids
    return normalized


def build_stage_selected_moment_ids_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            stage: {
                "type": "array",
                "items": {"type": "string"},
            }
            for stage in EVENT_CHAIN_STAGES
        },
        "additionalProperties": False,
    }


def _format_seconds(value: Any) -> str:
    try:
        text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    except Exception:
        return str(value)
    if "." not in text:
        text += ".0"
    return text


def build_counterfactual_type_schema() -> Dict[str, Any]:
    raise RuntimeError(
        "counterfactual_type has been removed from the active v5 contract. "
        "Do not build or request a counterfactual_type schema."
    )


def summarize_evidence_ledger(
    evidence_ledger: Sequence[Dict[str, Any]] | None,
    *,
    max_items: int = 6,
) -> str:
    entries = list(evidence_ledger or [])
    if not entries:
        return "Evidence ledger so far: none."
    recent_entries = entries[-max(1, int(max_items)) :]
    parts: List[str] = []
    for entry in recent_entries:
        window_id = str(entry.get("window_id") or "").strip() or "unknown_window"
        role = str(entry.get("role") or entry.get("kind") or "evidence").strip()
        start_sec = _format_seconds(entry.get("start_sec", 0.0))
        end_sec = _format_seconds(entry.get("end_sec", entry.get("start_sec", 0.0)))
        parts.append(f"{window_id}({role},{start_sec}-{end_sec}s)")
    prefix = "Evidence ledger so far: "
    if len(entries) > len(recent_entries):
        prefix += "...; "
    return prefix + "; ".join(parts) + "."


def _selected_ids_sentence(label: str, values: Iterable[Any]) -> str:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    if not cleaned:
        return ""
    return f"{label}: {', '.join(cleaned)}."


def build_counterfactual_type_legend(*, include_descriptions: bool = True) -> str:
    del include_descriptions
    raise RuntimeError(
        "counterfactual_type has been removed from the active v5 contract."
    )


def build_finalize_scaffold(
    *,
    verification_payload: Dict[str, Any] | None,
    finalize_schema: Dict[str, Any] | None,
) -> str:
    verification_payload = verification_payload or {}
    finalize_schema = finalize_schema or {}
    required_fields = [str(field).strip() for field in list(finalize_schema.get("required") or []) if str(field).strip()]
    scaffold_parts = [
        (
            "Sufficient evidence. "
            "Call finalize_case next using only searched evidence. "
            "Prioritize a compact finalize_case with the verified decision fields. "
            "Optional summary, rationale, event_chain_summary, and qa_focus_answers may be omitted if they would make the tool call too long."
        )
    ]
    if required_fields:
        scaffold_parts.append("Required finalize_case fields: " + ", ".join(required_fields) + ".")
    selected_windows_sentence = _selected_ids_sentence(
        "Selected windows",
        verification_payload.get("selected_window_ids") or [],
    )
    if selected_windows_sentence:
        scaffold_parts.append(selected_windows_sentence)
    selected_moments_sentence = _selected_ids_sentence(
        "Selected evidence moments",
        verification_payload.get("selected_evidence_moment_ids") or [],
    )
    if selected_moments_sentence:
        scaffold_parts.append(selected_moments_sentence)
    covered_stages = normalize_event_chain_stages(verification_payload.get("covered_stages"))
    if covered_stages:
        scaffold_parts.append("Covered event-chain stages: " + ", ".join(covered_stages) + ".")
    missing_required_stages = normalize_event_chain_stages(verification_payload.get("missing_required_stages"))
    if missing_required_stages:
        scaffold_parts.append("Missing required stages should be empty before finalizing; current payload reports: " + ", ".join(missing_required_stages) + ".")
    stage_selected_moment_ids = normalize_stage_selected_moment_ids(verification_payload.get("stage_selected_moment_ids"))
    if stage_selected_moment_ids:
        stage_parts = []
        for stage in EVENT_CHAIN_STAGES:
            moment_ids = stage_selected_moment_ids.get(stage) or []
            if moment_ids:
                stage_parts.append(f"{stage}={','.join(moment_ids)}")
        if stage_parts:
            scaffold_parts.append("Stage-to-moment mapping: " + "; ".join(stage_parts) + ".")
    return " ".join(part for part in scaffold_parts if part)
