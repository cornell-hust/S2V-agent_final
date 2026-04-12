from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from saver_v3.core.categories import CANONICAL_POLICY_CATEGORIES
from saver_v3.core.proposal import normalize_query_text
from saver_v3.core.protocol_guidance import (
    EVENT_CHAIN_STAGES,
    event_chain_stage_for_role,
    normalize_event_chain_stages,
    normalize_stage_selected_moment_ids,
)


def _clamp_signed(value: Any) -> float:
    try:
        return max(-1.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _has_interval(interval: Any) -> bool:
    return isinstance(interval, (list, tuple)) and len(interval) == 2


def infer_required_stages_from_target(
    target: Dict[str, Any] | None,
    *,
    tool_io: Dict[str, Any] | None = None,
) -> List[str]:
    payload = dict(target or {}) if isinstance(target, dict) else {}
    required = normalize_event_chain_stages(((payload.get("event_chain_target") or {}).get("required_stages") or []))
    if required:
        return required

    covered = normalize_event_chain_stages(payload.get("covered_stages") or [])
    if covered:
        return covered

    stage_selected_moment_ids = normalize_stage_selected_moment_ids(payload.get("stage_selected_moment_ids"))
    if stage_selected_moment_ids:
        return [stage for stage in EVENT_CHAIN_STAGES if stage in stage_selected_moment_ids]

    inferred_from_tool_io: List[str] = []
    for entry in list((tool_io or {}).get("oracle_windows_sec") or []):
        stage = event_chain_stage_for_role(entry.get("role"))
        if stage:
            inferred_from_tool_io.append(stage)
    inferred_from_tool_io = normalize_event_chain_stages(inferred_from_tool_io)
    if inferred_from_tool_io:
        return inferred_from_tool_io

    if str(payload.get("existence") or "").strip().lower() != "anomaly":
        return []

    inferred: List[str] = []
    if _has_interval(payload.get("precursor_interval_sec")):
        inferred.append("precursor")
    if _has_interval(payload.get("anomaly_interval_sec")):
        inferred.append("trigger")
    return normalize_event_chain_stages(inferred or ["trigger"])


def _normalize_stage_annotation(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    stage_selected_moment_ids = normalize_stage_selected_moment_ids(payload.get("stage_selected_moment_ids"))
    covered_stages = normalize_event_chain_stages(payload.get("covered_stages") or stage_selected_moment_ids.keys())
    missing_required_stages = normalize_event_chain_stages(payload.get("missing_required_stages"))
    recommended_action = str(payload.get("recommended_action") or "").strip().lower()
    verification_decision = str(
        payload.get("verification_decision")
        or payload.get("self_verification_decision")
        or ""
    ).strip().lower()
    claim = payload.get("claim") or {}
    existence = str(claim.get("existence") or payload.get("existence") or "").strip().lower()
    if not (covered_stages or missing_required_stages or stage_selected_moment_ids or recommended_action or verification_decision):
        return {}
    return {
        "covered_stages": covered_stages,
        "missing_required_stages": missing_required_stages,
        "stage_selected_moment_ids": stage_selected_moment_ids,
        "recommended_action": recommended_action,
        "verification_decision": verification_decision,
        "existence": existence,
        "has_stage_signal": bool(covered_stages or missing_required_stages or stage_selected_moment_ids),
    }


def extract_stage_annotation_from_turn(turn: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = dict(turn or {})
    candidates = [
        payload,
        ((payload.get("new_verifications") or [])[-1] if list(payload.get("new_verifications") or []) else None),
        ((((payload.get("state_after") or {}).get("verification_records")) or [])[-1] if (((payload.get("state_after") or {}).get("verification_records")) or []) else None),
        (((payload.get("parsed_tool_call") or {}).get("arguments")) or None),
        payload.get("new_finalized_case"),
        (payload.get("state_after") or {}).get("finalized_case"),
        payload.get("latest_claim_after"),
    ]
    fallback: Dict[str, Any] = {}
    for candidate in candidates:
        normalized = _normalize_stage_annotation(candidate)
        if normalized.get("has_stage_signal"):
            return normalized
        if normalized and not fallback:
            fallback = normalized
    return fallback


def extract_stage_annotation_from_record(record: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = dict(record or {})
    fallback: Dict[str, Any] = {}
    for turn in reversed(list(payload.get("turns") or [])):
        normalized = extract_stage_annotation_from_turn(turn)
        if normalized.get("has_stage_signal"):
            return normalized
        if normalized and not fallback:
            fallback = normalized

    state = payload.get("state") or {}
    candidates = [
        ((state.get("verification_records") or [])[-1] if list(state.get("verification_records") or []) else None),
        state.get("finalized_case"),
        payload.get("final_answer"),
        state.get("last_claim"),
    ]
    for candidate in candidates:
        normalized = _normalize_stage_annotation(candidate)
        if normalized.get("has_stage_signal"):
            return normalized
        if normalized and not fallback:
            fallback = normalized
    return fallback


def compute_stage_f1(required_stages: Iterable[Any], predicted_stages: Iterable[Any]) -> float:
    required = set(normalize_event_chain_stages(required_stages))
    predicted = set(normalize_event_chain_stages(predicted_stages))
    if not required:
        return 0.0
    tp = len(required & predicted)
    precision = tp / float(len(predicted)) if predicted else 0.0
    recall = tp / float(len(required)) if required else 0.0
    denom = precision + recall
    return (2.0 * precision * recall / denom) if denom > 0 else 0.0


def has_complete_event_chain(required_stages: Iterable[Any], annotation: Dict[str, Any] | None) -> bool:
    required = set(normalize_event_chain_stages(required_stages))
    normalized = dict(annotation or {})
    covered = set(normalize_event_chain_stages(normalized.get("covered_stages") or []))
    missing = set(normalize_event_chain_stages(normalized.get("missing_required_stages") or []))
    if not required:
        return False
    return required.issubset(covered) and not (missing & required)


def compute_event_chain_score(
    required_stages: Iterable[Any],
    annotation: Dict[str, Any] | None,
    *,
    terminal: bool = False,
) -> float:
    required = normalize_event_chain_stages(required_stages)
    normalized = dict(annotation or {})
    if not required or not normalized:
        return 0.0

    required_set = set(required)
    covered = set(normalize_event_chain_stages(normalized.get("covered_stages") or []))
    missing = set(normalize_event_chain_stages(normalized.get("missing_required_stages") or []))
    stage_selected_moment_ids = normalize_stage_selected_moment_ids(normalized.get("stage_selected_moment_ids"))
    coverage_ratio = len(required_set & covered) / float(len(required_set))
    missing_ratio = len(required_set & missing) / float(len(required_set))
    grounded_ratio = (
        sum(1 for stage in required if list(stage_selected_moment_ids.get(stage) or [])) / float(len(required_set))
    )
    recommended_action = str(normalized.get("recommended_action") or "").strip().lower()
    verification_decision = str(normalized.get("verification_decision") or "").strip().lower()

    score = (
        0.60 * (2.0 * coverage_ratio - 1.0)
        + 0.20 * (2.0 * grounded_ratio - 1.0)
        - 0.35 * missing_ratio
    )
    if recommended_action == "finalize" or terminal:
        score += 0.35 if has_complete_event_chain(required, normalized) else -0.60
    elif recommended_action in {"continue_search", "refine_evidence", "revise_claim"}:
        score += 0.20 if not has_complete_event_chain(required, normalized) else -0.20

    if verification_decision == "sufficient" and not has_complete_event_chain(required, normalized):
        score -= 0.30
    if verification_decision in {"insufficient", "redundant"} and not has_complete_event_chain(required, normalized):
        score += 0.10
    return _clamp_signed(score)


def infer_query_target_stage(turn: Dict[str, Any] | None) -> Optional[str]:
    payload = dict(turn or {})
    if str(payload.get("tool_name") or "") != "seek_evidence":
        return None

    arguments = ((payload.get("parsed_tool_call") or {}).get("arguments")) or {}
    role_stage = event_chain_stage_for_role(payload.get("role"))
    if role_stage:
        return role_stage
    role_stage = event_chain_stage_for_role(arguments.get("role"))
    if role_stage:
        return role_stage

    query_source = str(payload.get("proposal_query_source") or payload.get("query_source") or "").strip().lower()
    if "precursor" in query_source:
        return "precursor"
    if "confirmation" in query_source or "aftermath" in query_source:
        return "confirmation"
    if "trigger" in query_source or "peak" in query_source:
        return "trigger"

    query_candidates = [
        str(payload.get("proposal_query_normalized") or "").strip(),
        str(payload.get("proposal_query_raw") or "").strip(),
        str(arguments.get("query") or "").strip(),
    ]
    query_text = " ".join(normalize_query_text(value) for value in query_candidates if value).strip()
    if not query_text:
        return None

    keyword_groups = {
        "precursor": ["precursor", "lead up", "lead-up", "early cue", "setup", "before the attack", "before attack"],
        "confirmation": ["confirmation", "aftermath", "confirm", "after the event", "after the attack", "outcome"],
        "trigger": ["trigger", "peak action", "peak", "main action", "strongest visible anomalous action"],
    }
    for stage, keywords in keyword_groups.items():
        if any(keyword in query_text for keyword in keywords):
            return stage
    return None


def extract_query_text(turn: Dict[str, Any] | None) -> str:
    payload = dict(turn or {})
    arguments = ((payload.get("parsed_tool_call") or {}).get("arguments")) or {}
    query_candidates = [
        str(payload.get("proposal_query_normalized") or "").strip(),
        str(payload.get("proposal_query_raw") or "").strip(),
        str(arguments.get("query") or "").strip(),
    ]
    for value in query_candidates:
        normalized = normalize_query_text(value)
        if normalized:
            return normalized
    return ""


def observed_stages_for_search_turn(turn: Dict[str, Any] | None) -> List[str]:
    payload = dict(turn or {})
    observed: List[str] = []
    for entry in list((payload.get("state_delta") or {}).get("new_evidence_windows") or []):
        stage = event_chain_stage_for_role(entry.get("role"))
        if stage:
            observed.append(stage)
    if observed:
        return normalize_event_chain_stages(observed)

    state_after = payload.get("state_after") or {}
    active_window_ids = {str(value) for value in list(state_after.get("active_evidence_window_ids") or []) if str(value)}
    for entry in list(state_after.get("evidence_ledger") or []):
        if active_window_ids and str(entry.get("window_id") or "") not in active_window_ids:
            continue
        stage = event_chain_stage_for_role(entry.get("role"))
        if stage:
            observed.append(stage)
    return normalize_event_chain_stages(observed)


def compute_query_stage_alignment_score(turn: Dict[str, Any] | None) -> float:
    target_stage = infer_query_target_stage(turn)
    observed_stages = observed_stages_for_search_turn(turn)
    if not target_stage or not observed_stages:
        return 0.0
    return 1.0 if target_stage in set(observed_stages) else -1.0


def is_degenerate_query(turn: Dict[str, Any] | None) -> bool:
    query_text = extract_query_text(turn)
    if not query_text:
        return True
    generic_queries = {
        "anomaly",
        "event",
        "incident",
        "evidence",
        "normal",
        *{normalize_query_text(category) for category in CANONICAL_POLICY_CATEGORIES},
    }
    if query_text in generic_queries:
        return True
    if len(query_text.split()) <= 1:
        return True
    return False


def compute_query_non_degenerate_score(turn: Dict[str, Any] | None) -> float:
    return -1.0 if is_degenerate_query(turn) else 1.0


def compute_query_retrieval_effectiveness_score(turn: Dict[str, Any] | None) -> float:
    payload = dict(turn or {})
    if str(payload.get("tool_name") or "") != "seek_evidence":
        return 0.0
    observed_stages = observed_stages_for_search_turn(payload)
    if observed_stages:
        target_stage = infer_query_target_stage(payload)
        if not target_stage:
            return 1.0
        return 1.0 if target_stage in set(observed_stages) else -1.0

    state_delta = dict(payload.get("state_delta") or {})
    if list(state_delta.get("new_visited_windows") or []):
        return 0.0
    return -1.0


def compute_query_alignment_score(turn: Dict[str, Any] | None) -> float:
    retrieval_effectiveness = compute_query_retrieval_effectiveness_score(turn)
    stage_alignment = compute_query_stage_alignment_score(turn)
    non_degenerate = compute_query_non_degenerate_score(turn)
    return _clamp_signed(0.6 * retrieval_effectiveness + 0.3 * stage_alignment + 0.1 * non_degenerate)
