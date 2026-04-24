from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from saver_v3.core.categories import (
    CANONICAL_POLICY_CATEGORIES,
    canonicalize_saver_category,
    normalize_existence,
)
from saver_v3.core.protocol_guidance import (
    EVENT_CHAIN_STAGES,
    build_stage_selected_moment_ids_schema,
    normalize_event_chain_stages,
    normalize_stage_selected_moment_ids,
)

_REMOVED_CLAIM_FIELDS = (
    "severity",
    "counterfactual_type",
    "counterfactual_faithfulness",
    "precursor_interval_sec",
    "earliest_actionable_sec",
)

SELF_VERIFICATION_DECISIONS = {"insufficient", "sufficient", "misaligned", "redundant"}
SELF_VERIFICATION_NEXT_TOOLS = {"seek_evidence", "finalize_case"}
SELF_VERIFICATION_MODES = {
    "final_check",
    "stage_check",
}
PUBLIC_SELF_VERIFICATION_MODES = SELF_VERIFICATION_MODES
LEGACY_VERIFICATION_MODE_ALIASES = {
    "normal_check": "final_check",
    "declare_normal": "final_check",
}
PRIMARY_STATUS_TO_DECISION = {
    "complete": "sufficient",
    "incomplete": "insufficient",
    "misaligned": "misaligned",
    "redundant": "redundant",
}
DECISION_TO_PRIMARY_STATUS = {
    "sufficient": "complete",
    "insufficient": "incomplete",
    "misaligned": "misaligned",
    "redundant": "redundant",
}
POLICY_SELF_VERIFICATION_CLAIM_KEYS = ("existence", "category")
POLICY_SELF_VERIFICATION_REQUIRED_FIELDS = (
    "verification_mode",
    "selected_window_ids",
    "verification_decision",
    "next_tool",
    "sufficiency_score",
    "necessity_score",
    "finalize_readiness_score",
)


def _compact_payload_object(payload: Dict[str, Any], *, keys: tuple[str, ...]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        compact[key] = value
    return compact


def _drop_removed_claim_fields(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    normalized = dict(payload or {})
    for field_name in _REMOVED_CLAIM_FIELDS:
        normalized.pop(field_name, None)
    return normalized


def _normalize_self_verification_claim_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized = _drop_removed_claim_fields(dict(payload or {}))
    existence = normalize_existence(normalized.get("existence"))
    if existence:
        normalized["existence"] = existence
    else:
        normalized.pop("existence", None)
    if "category" in normalized:
        category = str(
            canonicalize_saver_category(
                normalized.get("category"),
                existence=normalized.get("existence"),
            )
            or ""
        ).strip()
        if category:
            normalized["category"] = category
        else:
            normalized.pop("category", None)
    return normalized


def coerce_self_verification_claim_payload(
    payload: Any,
    *,
    fallback_claim: Optional[Dict[str, Any]] = None,
    require_category_for_anomaly: bool = False,
) -> Dict[str, Any]:
    del require_category_for_anomaly
    candidate = payload
    if isinstance(candidate, str):
        text = candidate.strip()
        if text:
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            candidate = parsed if isinstance(parsed, dict) else None
        else:
            candidate = None
    if not isinstance(candidate, dict):
        candidate = fallback_claim if isinstance(fallback_claim, dict) else {}
    return _normalize_self_verification_claim_fields(dict(candidate or {}))


def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        result: List[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                result.append(text)
        return result
    text = str(value).strip()
    return [text] if text else []


def _coerce_score(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return float(default)


def _normalize_decision(payload: Dict[str, Any]) -> str:
    decision = str(payload.get("verification_decision") or "").strip().lower()
    if decision in SELF_VERIFICATION_DECISIONS:
        return decision
    primary_status = str(payload.get("primary_status") or "").strip().lower()
    return PRIMARY_STATUS_TO_DECISION.get(primary_status, "insufficient")


def _normalize_next_tool(payload: Dict[str, Any], decision: str) -> str:
    if "recommended_action" in payload and "next_tool" not in payload:
        raise ValueError(
            "verify_hypothesis self-verification payload uses legacy field `recommended_action`. "
            "Regenerate the payload with `next_tool`."
        )
    next_tool = str(payload.get("next_tool") or "").strip().lower()
    if next_tool:
        if next_tool not in SELF_VERIFICATION_NEXT_TOOLS:
            raise ValueError(
                "verify_hypothesis self-verification payload has invalid `next_tool`: "
                f"{next_tool}. Expected one of {sorted(SELF_VERIFICATION_NEXT_TOOLS)}."
            )
        return next_tool
    if decision == "sufficient":
        return "finalize_case"
    return "seek_evidence"


def normalize_self_verification_mode(
    value: Any,
    *,
    default: str = "stage_check",
    public_only: bool = False,
) -> str:
    text = str(value or "").strip().lower()
    if text in LEGACY_VERIFICATION_MODE_ALIASES:
        return LEGACY_VERIFICATION_MODE_ALIASES[text]
    if text in {"full_keep_drop", "reward_only", "search_step_check"}:
        raise ValueError(
            f"Legacy verification_mode `{text}` is no longer supported in the active v5 contract. "
            "Use `stage_check` or `final_check`."
        )
    if text in SELF_VERIFICATION_MODES:
        return text
    fallback = str(default or "stage_check").strip().lower() or "stage_check"
    if fallback in LEGACY_VERIFICATION_MODE_ALIASES:
        fallback = LEGACY_VERIFICATION_MODE_ALIASES[fallback]
    return fallback


def _derive_failure_reasons(primary_status: str) -> List[str]:
    reasons: List[str] = []
    if primary_status == "incomplete":
        reasons.append("selected_evidence_not_sufficient")
    elif primary_status == "misaligned":
        reasons.append("selected_evidence_not_aligned_with_claim")
    elif primary_status == "redundant":
        reasons.append("selected_evidence_not_necessary_enough")
    return reasons


def parse_self_verification_payload(
    payload: Dict[str, Any],
    *,
    fallback_claim: Optional[Dict[str, Any]] = None,
    verification_mode: str = "stage_check",
) -> Dict[str, Any]:
    payload = dict(payload or {})
    claim = coerce_self_verification_claim_payload(
        payload.get("claim"),
        fallback_claim=fallback_claim,
        require_category_for_anomaly=True,
    )
    verification_mode_normalized = normalize_self_verification_mode(
        payload.get("verification_mode") or verification_mode,
        default=verification_mode,
        public_only=True,
    )

    decision = _normalize_decision(payload)
    primary_status = DECISION_TO_PRIMARY_STATUS.get(decision, "incomplete")
    next_tool = _normalize_next_tool(payload, decision)

    derived_scores_in = dict(payload.get("derived_scores") or {})
    sufficiency = _coerce_score(payload.get("sufficiency_score", derived_scores_in.get("sufficiency", 0.0)))
    necessity = _coerce_score(payload.get("necessity_score", derived_scores_in.get("necessity", 0.0)))
    finalize_readiness = _coerce_score(
        payload.get("finalize_readiness_score", derived_scores_in.get("finalize_readiness", 0.0))
    )
    consistency = _coerce_score(
        derived_scores_in.get("consistency", 1.0 - abs(sufficiency - necessity) if (sufficiency or necessity) else 0.0)
    )
    derived_scores = {
        "sufficiency": round(sufficiency, 6),
        "necessity": round(necessity, 6),
        "consistency": round(consistency, 6),
        "finalize_readiness": round(finalize_readiness, 6),
    }

    selected_window_ids = _coerce_string_list(
        payload.get("selected_window_ids")
        or payload.get("verified_window_ids")
        or payload.get("best_effort_window_ids")
    )
    candidate_window_ids = _coerce_string_list(payload.get("candidate_window_ids"))
    selected_evidence_ids = _coerce_string_list(
        payload.get("selected_evidence_ids")
        or payload.get("candidate_evidence_ids")
        or payload.get("evidence_ids")
    )
    selected_evidence_moment_ids = _coerce_string_list(
        payload.get("selected_evidence_moment_ids")
        or payload.get("candidate_evidence_moment_ids")
        or payload.get("evidence_moment_ids")
    )
    best_effort_window_ids = list(selected_window_ids or candidate_window_ids)
    failure_reasons = _coerce_string_list(payload.get("failure_reasons")) or _derive_failure_reasons(primary_status)
    stage_selected_moment_ids = normalize_stage_selected_moment_ids(payload.get("stage_selected_moment_ids"))
    covered_stages = normalize_event_chain_stages(
        payload.get("covered_stages") or stage_selected_moment_ids.keys()
    )
    missing_required_stages = normalize_event_chain_stages(payload.get("missing_required_stages"))

    parsed = {
        "verification_mode": verification_mode_normalized,
        "claim": claim,
        "query": str(payload.get("query") or ""),
        "candidate_window_ids": candidate_window_ids,
        "candidate_evidence_ids": _coerce_string_list(payload.get("candidate_evidence_ids")),
        "candidate_evidence_moment_ids": _coerce_string_list(
            payload.get("candidate_evidence_moment_ids") or payload.get("evidence_moment_ids")
        ),
        "selected_window_ids": selected_window_ids,
        "selected_evidence_ids": selected_evidence_ids,
        "selected_evidence_moment_ids": selected_evidence_moment_ids,
        "verification_decision": decision,
        "next_tool": next_tool,
        "rationale": str(payload.get("rationale") or payload.get("explanation") or ""),
        "primary_status": primary_status,
        "derived_scores": derived_scores,
        "verified_window_ids": list(selected_window_ids),
        "best_effort_window_ids": best_effort_window_ids,
        "failure_reasons": failure_reasons,
        "covered_stages": covered_stages,
        "missing_required_stages": missing_required_stages,
        "stage_selected_moment_ids": stage_selected_moment_ids,
        "verifier_backend": "self_report",
        "self_verification_scores": dict(derived_scores),
        "self_verification_selected_window_ids": list(selected_window_ids),
        "self_verification_confidence": round(max(sufficiency, necessity, finalize_readiness), 6),
    }
    return parsed


def validate_policy_self_verification_payload(
    payload: Dict[str, Any],
    *,
    fallback_claim: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized = dict(payload or {})
    if "claim" in normalized:
        normalized["claim"] = coerce_self_verification_claim_payload(
            normalized.get("claim"),
            fallback_claim=fallback_claim,
            require_category_for_anomaly=True,
        )
    if "recommended_action" in normalized and "next_tool" not in normalized:
        raise ValueError(
            "verify_hypothesis self-verification payload uses legacy field `recommended_action`. "
            "Regenerate the payload with `next_tool`."
        )
    missing = [
        field_name
        for field_name in POLICY_SELF_VERIFICATION_REQUIRED_FIELDS
        if field_name not in normalized
    ]
    if missing:
        raise ValueError(
            "verify_hypothesis self-verification payload is missing required field(s): "
            + ", ".join(missing)
            + "."
        )
    if not _coerce_string_list(normalized.get("selected_window_ids")):
        raise ValueError(
            "verify_hypothesis self-verification payload must include at least one selected_window_id."
        )
    return normalized


def build_policy_self_verification_payload(
    payload: Dict[str, Any],
    *,
    include_query: bool = True,
    include_rationale: bool = True,
) -> Dict[str, Any]:
    source = dict(payload or {})
    decision = _normalize_decision(source)
    next_tool = _normalize_next_tool(source, decision)
    compact: Dict[str, Any] = {
        "verification_mode": normalize_self_verification_mode(
            source.get("verification_mode"),
            default="stage_check",
            public_only=True,
        ),
        "verification_decision": decision,
        "next_tool": next_tool,
        "sufficiency_score": round(_coerce_score(source.get("sufficiency_score")), 6),
        "necessity_score": round(_coerce_score(source.get("necessity_score")), 6),
        "finalize_readiness_score": round(
            _coerce_score(source.get("finalize_readiness_score"))
        ),
    }

    claim = coerce_self_verification_claim_payload(
        source.get("claim"),
        require_category_for_anomaly=True,
    )
    if claim:
        compact_claim = _compact_payload_object(
            claim,
            keys=POLICY_SELF_VERIFICATION_CLAIM_KEYS,
        )
        if compact_claim:
            compact["claim"] = compact_claim

    query = str(source.get("query") or "").strip()
    if include_query and query:
        compact["query"] = query

    selected_window_ids = _coerce_string_list(
        source.get("selected_window_ids")
        or source.get("verified_window_ids")
        or source.get("best_effort_window_ids")
    )
    if selected_window_ids:
        compact["selected_window_ids"] = selected_window_ids

    selected_evidence_moment_ids = _coerce_string_list(
        source.get("selected_evidence_moment_ids")
        or source.get("evidence_moment_ids")
        or source.get("candidate_evidence_moment_ids")
    )
    if selected_evidence_moment_ids:
        compact["selected_evidence_moment_ids"] = selected_evidence_moment_ids

    covered_stages = normalize_event_chain_stages(source.get("covered_stages"))
    if covered_stages or "covered_stages" in source:
        compact["covered_stages"] = covered_stages

    missing_required_stages = normalize_event_chain_stages(source.get("missing_required_stages"))
    if missing_required_stages or "missing_required_stages" in source:
        compact["missing_required_stages"] = missing_required_stages

    stage_selected_moment_ids = normalize_stage_selected_moment_ids(source.get("stage_selected_moment_ids"))
    if stage_selected_moment_ids or "stage_selected_moment_ids" in source:
        compact["stage_selected_moment_ids"] = stage_selected_moment_ids

    rationale = str(source.get("rationale") or source.get("explanation") or "").strip()
    if include_rationale and rationale:
        compact["rationale"] = rationale

    return compact


def build_self_verification_tool_schema() -> Dict[str, Any]:
    claim_schema = {
        "type": "object",
        "properties": {
            "existence": {"type": "string", "enum": ["normal", "anomaly"]},
            "category": {"type": "string", "enum": list(CANONICAL_POLICY_CATEGORIES)},
        },
    }
    return {
        "type": "object",
        "properties": {
            "verification_mode": {
                "type": "string",
                "enum": sorted(PUBLIC_SELF_VERIFICATION_MODES),
            },
            "selected_window_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "selected_evidence_moment_ids": {"type": "array", "items": {"type": "string"}},
            "covered_stages": {"type": "array", "items": {"type": "string", "enum": list(EVENT_CHAIN_STAGES)}},
            "missing_required_stages": {"type": "array", "items": {"type": "string", "enum": list(EVENT_CHAIN_STAGES)}},
            "stage_selected_moment_ids": build_stage_selected_moment_ids_schema(),
            "claim": claim_schema,
            "query": {"type": "string"},
            "verification_decision": {"type": "string", "enum": sorted(SELF_VERIFICATION_DECISIONS)},
            "next_tool": {"type": "string", "enum": sorted(SELF_VERIFICATION_NEXT_TOOLS)},
            "sufficiency_score": {"type": "number"},
            "necessity_score": {"type": "number"},
            "finalize_readiness_score": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "required": list(POLICY_SELF_VERIFICATION_REQUIRED_FIELDS),
    }
