from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from saver_v3.core.categories import canonicalize_category_payload, canonicalize_saver_category


SEMANTIC_EVENT_CHAIN_STAGES = ("precursor", "trigger", "confirmation")
_ROLE_TO_STAGE = {
    "precursor": "precursor",
    "trigger": "trigger",
    "peak_action": "trigger",
    "confirmation": "confirmation",
    "aftermath": "confirmation",
}
_QA_FOCUS_KEYS = ("existence", "category", "temporal")
FINALIZE_CASE_SEMANTIC_FIELD_KEYS = ("summary", "rationale", "event_chain_summary", "qa_focus_answers")
FINALIZE_CASE_EVENT_CHAIN_SCHEMA = {
    "type": "object",
    "properties": {
        stage: {"type": "string"}
        for stage in SEMANTIC_EVENT_CHAIN_STAGES
    },
}
FINALIZE_CASE_QA_FOCUS_SCHEMA = {
    "type": "object",
    "properties": {
        key: {"type": "string"}
        for key in _QA_FOCUS_KEYS
    },
}
FINALIZE_CASE_SEMANTIC_PROPERTIES = {
    "summary": {"type": "string"},
    "rationale": {"type": "string"},
    "event_chain_summary": FINALIZE_CASE_EVENT_CHAIN_SCHEMA,
    "qa_focus_answers": FINALIZE_CASE_QA_FOCUS_SCHEMA,
}


def build_replay_decision_scaffold() -> Dict[str, Any]:
    return {
        # Shape-only replay scaffold: do not leak target labels into verifier prompts.
        "existence": "<anomaly_or_normal>",
        "category": "<canonical_category>",
        "anomaly_interval_sec": [0.0, 1.0],
    }


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _clean_text(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    return text


def _ensure_sentence(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    if cleaned[-1] in ".!?":
        return cleaned
    return cleaned + "."


def _semantic_fields_present(payload: Dict[str, Any]) -> bool:
    if _clean_text(payload.get("summary")):
        return True
    if _clean_text(payload.get("rationale")):
        return True
    event_chain_summary = dict(payload.get("event_chain_summary") or {})
    if any(_clean_text(event_chain_summary.get(stage)) for stage in SEMANTIC_EVENT_CHAIN_STAGES):
        return True
    qa_focus_answers = dict(payload.get("qa_focus_answers") or {})
    return any(_clean_text(qa_focus_answers.get(key)) for key in _QA_FOCUS_KEYS)


def _moment_lookup(evidence_moments: Sequence[Dict[str, Any]] | None) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for moment in list(evidence_moments or []):
        moment_id = str(moment.get("moment_id") or "").strip()
        if moment_id:
            lookup[moment_id] = dict(moment)
    return lookup


def _stage_to_moment_ids(structured_target: Dict[str, Any]) -> Dict[str, List[str]]:
    explicit = (
        ((structured_target.get("event_chain_target") or {}).get("stage_to_moment_ids"))
        or structured_target.get("stage_selected_moment_ids")
        or {}
    )
    stage_to_ids: Dict[str, List[str]] = {}
    for stage in SEMANTIC_EVENT_CHAIN_STAGES:
        values = explicit.get(stage) or []
        stage_to_ids[stage] = [str(value).strip() for value in values if str(value).strip()]
    return stage_to_ids


def _fallback_stage_descriptions(
    stage: str,
    *,
    structured_target: Dict[str, Any],
    evidence_moments: Sequence[Dict[str, Any]] | None,
) -> List[str]:
    wanted_stage = str(stage or "").strip().lower()
    descriptions: List[str] = []
    for moment in list(evidence_moments or []):
        role = str(moment.get("role") or "").strip().lower()
        if _ROLE_TO_STAGE.get(role) != wanted_stage:
            continue
        description = _clean_text(moment.get("description"))
        if description:
            descriptions.append(description)
    if descriptions:
        return descriptions

    role_to_window = {
        str(entry.get("role") or "").strip().lower(): entry
        for entry in list(structured_target.get("evidence_windows_sec") or [])
    }
    wanted_roles = [role for role, mapped_stage in _ROLE_TO_STAGE.items() if mapped_stage == wanted_stage]
    for role in wanted_roles:
        entry = role_to_window.get(role)
        if not isinstance(entry, dict):
            continue
        window_sec = list(entry.get("window_sec") or [])
        if len(window_sec) >= 2:
            descriptions.append(f"{wanted_stage} evidence around {float(window_sec[0]):.1f}s-{float(window_sec[1]):.1f}s")
            break
    return descriptions


def build_event_chain_summary(
    *,
    structured_target: Dict[str, Any] | None,
    evidence_moments: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, str]:
    structured_target = copy.deepcopy(structured_target or {})
    stage_to_ids = _stage_to_moment_ids(structured_target)
    moment_lookup = _moment_lookup(evidence_moments)
    summary: Dict[str, str] = {}
    for stage in SEMANTIC_EVENT_CHAIN_STAGES:
        descriptions: List[str] = []
        for moment_id in stage_to_ids.get(stage) or []:
            description = _clean_text((moment_lookup.get(moment_id) or {}).get("description"))
            if description:
                descriptions.append(description)
        if not descriptions:
            descriptions = _fallback_stage_descriptions(
                stage,
                structured_target=structured_target,
                evidence_moments=evidence_moments,
            )
        deduped: List[str] = []
        seen = set()
        for description in descriptions:
            normalized = description.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped.append(description)
        if not deduped:
            summary[stage] = ""
            continue
        if len(deduped) == 1:
            summary[stage] = _ensure_sentence(deduped[0])
        else:
            summary[stage] = _ensure_sentence(" Then ".join(deduped))
    return summary


def _qa_pairs_to_map(qa_pairs: Sequence[Dict[str, Any]] | None) -> Dict[str, str]:
    answer_map: Dict[str, str] = {}
    for pair in list(qa_pairs or []):
        qa_type = str(pair.get("type") or "").strip().lower()
        if qa_type in _QA_FOCUS_KEYS and qa_type not in answer_map:
            answer_map[qa_type] = _ensure_sentence(pair.get("answer"))
    return answer_map


def _fallback_temporal_answer(structured_target: Dict[str, Any]) -> str:
    existence = str(structured_target.get("existence") or "").strip().lower()
    if existence == "normal":
        return "There is no anomaly interval in this video."
    anomaly_interval = list(structured_target.get("anomaly_interval_sec") or [])
    if len(anomaly_interval) >= 2:
        return f"The anomaly occurs from {float(anomaly_interval[0]):.3f}s to {float(anomaly_interval[1]):.3f}s."
    return "The anomaly timing is uncertain."


def build_qa_focus_answers(
    *,
    structured_target: Dict[str, Any] | None,
    qa_pairs: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, str]:
    structured_target = copy.deepcopy(structured_target or {})
    answer_map = _qa_pairs_to_map(qa_pairs)
    existence = str(structured_target.get("existence") or "").strip().lower()
    category = str(
        canonicalize_saver_category(
            structured_target.get("category"),
            existence=existence,
        )
        or structured_target.get("category")
        or ""
    ).strip()
    if "existence" not in answer_map:
        if existence == "normal":
            answer_map["existence"] = "No. No anomaly is visible in this video."
        elif category:
            article = "an" if category[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
            answer_map["existence"] = f"Yes, there is {article} {category} occurring."
        else:
            answer_map["existence"] = "Yes, there is an anomaly occurring."
    if "category" not in answer_map:
        if existence == "normal":
            answer_map["category"] = "The video is normal."
        elif category:
            answer_map["category"] = f"The anomaly is {category}."
        else:
            answer_map["category"] = "The anomaly category is uncertain."
    if "temporal" not in answer_map:
        answer_map["temporal"] = _fallback_temporal_answer(structured_target)
    return {key: _ensure_sentence(answer_map.get(key, "")) for key in _QA_FOCUS_KEYS}


def extract_decision_from_semantic_answer(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("decision"), dict):
        return canonicalize_category_payload(copy.deepcopy(payload["decision"]))
    if "existence" in payload or "category" in payload:
        return canonicalize_category_payload(copy.deepcopy(payload))
    return None


def normalize_semantic_answer_payload(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    decision = extract_decision_from_semantic_answer(payload)
    if decision is None:
        return None
    event_chain_summary = dict(payload.get("event_chain_summary") or {})
    qa_focus_answers = dict(payload.get("qa_focus_answers") or {})
    return {
        "decision": decision,
        "summary": _ensure_sentence(payload.get("summary")),
        "rationale": _ensure_sentence(payload.get("rationale")),
        "event_chain_summary": {
            stage: _ensure_sentence(event_chain_summary.get(stage))
            for stage in SEMANTIC_EVENT_CHAIN_STAGES
        },
        "qa_focus_answers": {
            key: _ensure_sentence(qa_focus_answers.get(key))
            for key in _QA_FOCUS_KEYS
        },
    }


def augment_finalize_case_schema(schema: Dict[str, Any] | None) -> Dict[str, Any]:
    normalized_schema = copy.deepcopy(schema or {"type": "object", "properties": {}})
    normalized_schema["type"] = "object"
    properties = dict(normalized_schema.get("properties") or {})
    for field_name, field_schema in FINALIZE_CASE_SEMANTIC_PROPERTIES.items():
        properties.setdefault(field_name, copy.deepcopy(field_schema))
    normalized_schema["properties"] = properties
    return normalized_schema


def split_finalize_case_payload(payload: Any) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    if not isinstance(payload, dict):
        return {}, None
    normalized_payload = copy.deepcopy(payload)
    summary = normalized_payload.pop("summary", None)
    rationale = normalized_payload.pop("rationale", None)
    event_chain_summary = normalized_payload.pop("event_chain_summary", None)
    qa_focus_answers = normalized_payload.pop("qa_focus_answers", None)
    decision = canonicalize_category_payload(normalized_payload)
    semantic_candidate = {
        "decision": decision,
        "summary": summary,
        "rationale": rationale,
        "event_chain_summary": dict(event_chain_summary or {}),
        "qa_focus_answers": dict(qa_focus_answers or {}),
    }
    if not _semantic_fields_present(semantic_candidate):
        return decision, None
    return decision, normalize_semantic_answer_payload(semantic_candidate)


def build_semantic_answer_payload(
    *,
    structured_target: Dict[str, Any] | None,
    qa_pairs: Sequence[Dict[str, Any]] | None = None,
    evidence_moments: Sequence[Dict[str, Any]] | None = None,
    finalized_case: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    structured_target = canonicalize_category_payload(copy.deepcopy(structured_target or {}))
    merged_decision = copy.deepcopy(structured_target)
    if isinstance(finalized_case, dict):
        merged_decision.update(copy.deepcopy(canonicalize_category_payload(finalized_case)))
    merged_decision = canonicalize_category_payload(merged_decision)
    payload = {
        "decision": merged_decision,
        "summary": _ensure_sentence(
            merged_decision.get("summary")
            or structured_target.get("summary")
            or ""
        ),
        "rationale": _ensure_sentence(
            merged_decision.get("rationale")
            or structured_target.get("rationale")
            or ""
        ),
        "event_chain_summary": build_event_chain_summary(
            structured_target=structured_target,
            evidence_moments=evidence_moments,
        ),
        "qa_focus_answers": build_qa_focus_answers(
            structured_target=merged_decision,
            qa_pairs=qa_pairs,
        ),
    }
    return normalize_semantic_answer_payload(payload) or payload


def render_semantic_answer_response(payload: Dict[str, Any], *, think_text: str = "") -> str:
    if not think_text:
        summary = _clean_text(payload.get("summary"))
        if summary:
            think_text = f"The structured decision is finalized, so I can summarize the case: {summary}"
        else:
            think_text = "The structured decision is finalized, so I can now provide the terminal answer."
    return f"<think>{_clean_text(think_text)}</think><answer>{_json_dumps(payload)}</answer>"


def build_semantic_answer_scaffold(*, finalized_case: Dict[str, Any] | None = None) -> str:
    del finalized_case
    scaffold = {
        "decision": build_replay_decision_scaffold(),
        "summary": "one concise case summary",
        "rationale": "brief evidence-grounded rationale",
        "event_chain_summary": {
            "precursor": "optional precursor evidence summary",
            "trigger": "optional trigger evidence summary",
            "confirmation": "optional confirmation or aftermath summary",
        },
        "qa_focus_answers": {
            "existence": "direct answer to whether an anomaly exists",
            "category": "direct answer to which anomaly category applies",
            "temporal": "direct answer to when the anomaly occurs",
        },
    }
    return _json_dumps(scaffold)


def semantic_answer_to_text(payload: Dict[str, Any] | None) -> Optional[str]:
    normalized = normalize_semantic_answer_payload(payload)
    if normalized is None:
        return None
    return json.dumps(normalized, ensure_ascii=False)


def normalize_public_semantic_replay_payload(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    event_chain_summary = dict(payload.get("event_chain_summary") or {})
    qa_focus_answers = dict(payload.get("qa_focus_answers") or {})
    normalized = {
        "summary": _ensure_sentence(payload.get("summary")),
        "rationale": _ensure_sentence(payload.get("rationale")),
        "event_chain_summary": {
            stage: _ensure_sentence(event_chain_summary.get(stage))
            for stage in SEMANTIC_EVENT_CHAIN_STAGES
        },
        "qa_focus_answers": {
            key: _ensure_sentence(qa_focus_answers.get(key))
            for key in _QA_FOCUS_KEYS
        },
    }
    if not any(
        str(normalized.get(key) or "").strip()
        for key in ("summary", "rationale")
    ) and not any(
        str(value or "").strip()
        for value in list(normalized["event_chain_summary"].values()) + list(normalized["qa_focus_answers"].values())
    ):
        return None
    return normalized


def build_public_semantic_replay_payload(
    *,
    structured_target: Dict[str, Any] | None,
    qa_pairs: Sequence[Dict[str, Any]] | None = None,
    evidence_moments: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    structured_target = canonicalize_category_payload(copy.deepcopy(structured_target or {}))
    payload = {
        "summary": _ensure_sentence(structured_target.get("summary") or ""),
        "rationale": _ensure_sentence(structured_target.get("rationale") or ""),
        "event_chain_summary": build_event_chain_summary(
            structured_target=structured_target,
            evidence_moments=evidence_moments,
        ),
        "qa_focus_answers": build_qa_focus_answers(
            structured_target=structured_target,
            qa_pairs=qa_pairs,
        ),
    }
    return normalize_public_semantic_replay_payload(payload) or payload


def build_public_semantic_replay_scaffold() -> str:
    scaffold = {
        "summary": "one concise case summary",
        "rationale": "brief evidence-grounded rationale",
        "event_chain_summary": {
            "precursor": "optional precursor evidence summary",
            "trigger": "optional trigger evidence summary",
            "confirmation": "optional confirmation or aftermath summary",
        },
        "qa_focus_answers": {
            "existence": "direct answer to whether an anomaly exists",
            "category": "direct answer to which anomaly category applies",
            "temporal": "direct answer to when the anomaly occurs",
        },
    }
    return _json_dumps(scaffold)


def merge_public_semantic_replay_with_decision(
    payload: Any,
    *,
    finalized_case: Dict[str, Any] | None,
) -> Optional[Dict[str, Any]]:
    normalized_payload = normalize_public_semantic_replay_payload(payload)
    if normalized_payload is None:
        return None
    decision = canonicalize_category_payload(copy.deepcopy(finalized_case or {}))
    if not isinstance(decision, dict) or not decision:
        return None
    merged_payload = {
        "decision": decision,
        "summary": normalized_payload.get("summary"),
        "rationale": normalized_payload.get("rationale"),
        "event_chain_summary": normalized_payload.get("event_chain_summary"),
        "qa_focus_answers": normalized_payload.get("qa_focus_answers"),
    }
    return normalize_semantic_answer_payload(merged_payload)


def public_semantic_replay_to_text(payload: Dict[str, Any] | None) -> Optional[str]:
    normalized = normalize_public_semantic_replay_payload(payload)
    if normalized is None:
        return None
    return json.dumps(normalized, ensure_ascii=False)


def normalize_text_match(value: Any) -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"[^\w\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
