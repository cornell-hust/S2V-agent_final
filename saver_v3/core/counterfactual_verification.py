from __future__ import annotations

import copy
import json
import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from saver_v3.core.adapter import TimeSearchRolloutAdapter
from saver_v3.core.categories import canonicalize_saver_category, normalize_existence
from saver_v3.core.environment import (
    _json_brace_balance,
    _repair_json_object_text,
    cleanup_llm_response,
    parse_actions_and_contents,
)
from saver_v3.core.proposal import compose_scene_anchored_query, normalize_description_query_phrases
from saver_v3.core.protocol_guidance import normalize_stage_selected_moment_ids, summarize_evidence_ledger
from saver_v3.core.semantic_answer import (
    build_replay_decision_scaffold,
    build_semantic_answer_payload,
    build_semantic_answer_scaffold,
    extract_decision_from_semantic_answer,
    normalize_semantic_answer_payload,
    normalize_text_match,
    semantic_answer_to_text,
)


STAGE_ORDER = ("precursor", "trigger", "confirmation")
ROLE_TO_STAGE = {
    "precursor": "precursor",
    "evidence": "precursor",
    "trigger": "trigger",
    "peak": "trigger",
    "peak_action": "trigger",
    "confirmation": "confirmation",
    "aftermath": "confirmation",
}
CORE_DECISION_FIELDS = ("existence", "category", "temporal")
STAGE_TEXT_THRESHOLD = 0.3
BRANCH_ORDER = (
    "full_selected",
    "minimal_subset",
    "drop_precursor",
    "drop_trigger",
    "drop_confirmation",
    "hard_negative_swap",
)
COUNTERFACTUAL_BRANCH_PROFILES = ("full", "offline_full", "online_core", "online_core")
FINALIZE_READINESS_THRESHOLD = 0.75

logger = logging.getLogger(__name__)


class CounterfactualReplayProtocolError(RuntimeError):
    def __init__(
        self,
        *,
        video_id: str,
        generation_id: str,
        branch_name: str,
        reason: str,
        parse_mode: str,
        response_preview: str,
    ) -> None:
        self.video_id = str(video_id or "")
        self.generation_id = str(generation_id or "")
        self.branch_name = str(branch_name or "")
        self.reason = str(reason or "")
        self.parse_mode = str(parse_mode or "")
        self.response_preview = str(response_preview or "")
        super().__init__(
            "Counterfactual replay protocol violation: "
            f"video_id={self.video_id} generation_id={self.generation_id} "
            f"branch_name={self.branch_name} reason={self.reason} "
            f"parse_mode={self.parse_mode} response_preview={self.response_preview}"
        )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_counterfactual_branch_profile(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"", "full"}:
        return "full"
    if normalized == "offline_full":
        return "offline_full"
    if normalized == "online_core":
        return "online_core"
    raise ValueError(
        f"Unsupported counterfactual branch profile: {value!r}. "
        f"Expected one of {COUNTERFACTUAL_BRANCH_PROFILES}."
    )


def _dedupe_window_ids(values: Sequence[Any] | None) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for value in list(values or []):
        text = str(value).strip()
        if not text or text in seen:
            continue
        deduped.append(text)
        seen.add(text)
    return deduped


def infer_counterfactual_window_ids(rollout: Dict[str, Any]) -> List[str]:
    verification_record = _latest_verification_record(rollout)
    authoritative_window_ids = _selected_window_ids_from_verification_record(verification_record)
    if authoritative_window_ids:
        return authoritative_window_ids
    verify_turn = _latest_verify_turn_with_windows(rollout)
    return _selected_window_ids_from_verify_turn(verify_turn)


def _latest_verify_turn_with_windows(rollout: Dict[str, Any]) -> Dict[str, Any]:
    for turn in reversed(list(rollout.get("turns") or [])):
        if str(turn.get("tool_name") or "") != "verify_hypothesis":
            continue
        return dict(turn)
    return {}


def _latest_verification_record(rollout: Dict[str, Any]) -> Dict[str, Any]:
    state = dict(rollout.get("state") or {})
    verification_records = list(state.get("verification_records") or [])
    if not verification_records:
        return {}
    latest = verification_records[-1]
    return dict(latest) if isinstance(latest, dict) else {}


def _latest_evidence_anchor_selected_window_ids(rollout: Dict[str, Any]) -> List[str]:
    for turn in reversed(list(rollout.get("turns") or [])):
        tags = list(turn.get("counterfactual_anchor_tags") or [])
        tool_name = str(turn.get("tool_name") or "").strip()
        if "evidence_anchor" not in tags and tool_name not in {"verify_hypothesis", "finalize_case"}:
            continue
        selected_window_ids_after = _dedupe_window_ids(turn.get("selected_window_ids_after") or [])
        if selected_window_ids_after:
            return selected_window_ids_after
    return []


def _selected_window_ids_from_verification_record(record: Dict[str, Any]) -> List[str]:
    if not isinstance(record, dict):
        return []
    for key in (
        "selected_window_ids",
        "selected_window_ids_after",
        "verifier_verified_window_ids",
        "self_verification_selected_window_ids",
        "window_ids",
    ):
        window_ids = _dedupe_window_ids(record.get(key) or [])
        if window_ids:
            return window_ids
    return []


def _selected_window_ids_from_verify_turn(turn: Dict[str, Any]) -> List[str]:
    if not isinstance(turn, dict):
        return []
    for key in (
        "verifier_verified_window_ids",
        "self_verification_selected_window_ids",
        "selected_window_ids_after",
        "selected_window_ids",
    ):
        window_ids = _dedupe_window_ids(turn.get(key) or [])
        if window_ids:
            return window_ids
    return []


def resolve_selected_window_ids_for_fecv(rollout: Dict[str, Any]) -> Dict[str, Any]:
    state = dict(rollout.get("state") or {})
    verify_turn = _latest_verify_turn_with_windows(rollout)
    verification_record = _latest_verification_record(rollout)
    active_evidence_window_ids = _dedupe_window_ids(state.get("active_evidence_window_ids") or [])
    evidence_anchor_window_ids = _latest_evidence_anchor_selected_window_ids(rollout)
    verification_record_selected_window_ids = _selected_window_ids_from_verification_record(verification_record)
    verification_record_verified_window_ids = _dedupe_window_ids(
        verification_record.get("verified_window_ids") or []
    )
    verification_record_best_effort_window_ids = _dedupe_window_ids(
        verification_record.get("best_effort_window_ids") or []
    )
    latest_verify_turn_selected_window_ids = _selected_window_ids_from_verify_turn(verify_turn)
    verify_turn_verified_window_ids = _dedupe_window_ids(verify_turn.get("verifier_verified_window_ids") or [])
    verify_turn_self_verification_selected_window_ids = _dedupe_window_ids(
        verify_turn.get("self_verification_selected_window_ids") or []
    )
    authoritative_window_ids = (
        verification_record_verified_window_ids
        or verification_record_selected_window_ids
        or latest_verify_turn_selected_window_ids
        or verify_turn_verified_window_ids
        or verify_turn_self_verification_selected_window_ids
    )
    source_conflict_detected = bool(
        authoritative_window_ids
        and (
            (evidence_anchor_window_ids and evidence_anchor_window_ids != authoritative_window_ids)
            or (active_evidence_window_ids and active_evidence_window_ids != authoritative_window_ids)
        )
    )

    candidates: List[Tuple[str, List[str], bool]] = [
        ("verification_record_verified_window_ids", verification_record_verified_window_ids, True),
        ("verification_record_selected_window_ids", verification_record_selected_window_ids, True),
        ("latest_verify_turn_selected_window_ids", latest_verify_turn_selected_window_ids, True),
        ("verify_turn_verified_window_ids", verify_turn_verified_window_ids, False),
        (
            "verify_turn_self_verification_selected_window_ids",
            verify_turn_self_verification_selected_window_ids,
            False,
        ),
        ("verification_record_best_effort_window_ids", verification_record_best_effort_window_ids, True),
        ("evidence_anchor_selected_window_ids_after", evidence_anchor_window_ids, True),
        ("active_evidence_window_ids", active_evidence_window_ids, False),
    ]
    discarded_sources = {
        source_name: {"window_ids": list(window_ids), "used": False}
        for source_name, window_ids, _ in candidates
    }
    compatibility_discarded_sources = {
        "verification_record": {
            "window_ids": list(verification_record_selected_window_ids or verification_record_verified_window_ids),
            "used": False,
        },
        "latest_verify_turn": {
            "window_ids": list(
                latest_verify_turn_selected_window_ids
                or verify_turn_verified_window_ids
                or verify_turn_self_verification_selected_window_ids
            ),
            "used": False,
        },
        "evidence_anchor": {"window_ids": list(evidence_anchor_window_ids), "used": False},
        "active_evidence": {"window_ids": list(active_evidence_window_ids), "used": False},
    }
    for source_name, window_ids, recovered_from_trace in candidates:
        if not window_ids:
            continue
        discarded_sources[source_name]["used"] = True
        if source_name.startswith("verification_record_"):
            compatibility_discarded_sources["verification_record"]["used"] = True
        elif source_name.startswith("latest_verify_turn_") or source_name.startswith("verify_turn_"):
            compatibility_discarded_sources["latest_verify_turn"]["used"] = True
        elif source_name == "evidence_anchor_selected_window_ids_after":
            compatibility_discarded_sources["evidence_anchor"]["used"] = True
        elif source_name == "active_evidence_window_ids":
            compatibility_discarded_sources["active_evidence"]["used"] = True
        return {
            "selected_window_ids": list(window_ids),
            "window_ids": list(window_ids),
            "selected_window_ids_effective": list(window_ids),
            "source": str(source_name),
            "selected_window_ids_source": str(source_name),
            "selection_resolution_source": str(source_name),
            "selection_unavailable_reason": "",
            "available": True,
            "source_conflict_detected": source_conflict_detected,
            "discarded_sources": compatibility_discarded_sources,
            "discarded_sources_detailed": discarded_sources,
            "candidates": candidates,
            "verify_turn": verify_turn,
            "verification_record": verification_record,
            "verification_record_selected_window_ids": verification_record_selected_window_ids,
            "verification_record_verified_window_ids": verification_record_verified_window_ids,
            "verification_record_best_effort_window_ids": verification_record_best_effort_window_ids,
            "latest_verify_turn_selected_window_ids": latest_verify_turn_selected_window_ids,
            "verify_turn_verified_window_ids": verify_turn_verified_window_ids,
            "verify_turn_self_verification_selected_window_ids": verify_turn_self_verification_selected_window_ids,
            "evidence_anchor_selected_window_ids": evidence_anchor_window_ids,
            "active_evidence_window_ids": active_evidence_window_ids,
            "recovered_from_trace": bool(recovered_from_trace),
        }
    return {
        "selected_window_ids": [],
        "window_ids": [],
        "selected_window_ids_effective": [],
        "source": "recovery_failed",
        "selected_window_ids_source": "recovery_failed",
        "selection_resolution_source": "recovery_failed",
        "selection_unavailable_reason": "missing_selected_windows",
        "available": False,
        "source_conflict_detected": False,
        "discarded_sources": compatibility_discarded_sources,
        "discarded_sources_detailed": discarded_sources,
        "candidates": candidates,
        "verify_turn": verify_turn,
        "verification_record": verification_record,
        "verification_record_selected_window_ids": verification_record_selected_window_ids,
        "verification_record_verified_window_ids": verification_record_verified_window_ids,
        "verification_record_best_effort_window_ids": verification_record_best_effort_window_ids,
        "latest_verify_turn_selected_window_ids": latest_verify_turn_selected_window_ids,
        "verify_turn_verified_window_ids": verify_turn_verified_window_ids,
        "verify_turn_self_verification_selected_window_ids": verify_turn_self_verification_selected_window_ids,
        "evidence_anchor_selected_window_ids": evidence_anchor_window_ids,
        "active_evidence_window_ids": active_evidence_window_ids,
        "recovered_from_trace": False,
    }


def _resolve_selected_window_ids_for_normal_skip(rollout: Dict[str, Any]) -> Dict[str, Any]:
    resolution = resolve_selected_window_ids_for_fecv(rollout)
    selected_window_ids = list(resolution.get("selected_window_ids") or [])
    selection_source = str(resolution.get("selection_resolution_source") or "").strip()
    if selected_window_ids:
        return {
            "selected_window_ids": selected_window_ids,
            "selection_resolution_source": selection_source or "unknown",
            "recovered_from_trace": bool(resolution.get("recovered_from_trace", False)),
        }
    return {
        "selected_window_ids": [],
        "selection_resolution_source": "normal_sample_skipped",
        "recovered_from_trace": False,
    }


def derive_counterfactual_stage_requirements(
    structured_target: Dict[str, Any] | None,
    *,
    evidence_moments: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, List[str]]:
    structured_target = dict(structured_target or {})
    existence = str(structured_target.get("existence") or "").strip().lower()
    if existence != "anomaly":
        return {
            "decision_required_stages": [],
            "finalize_required_stages": [],
            "narrative_required_stages": [],
        }

    has_confirmation = False
    chain_target = dict(structured_target.get("event_chain_target") or {})
    stage_to_moment_ids = dict(chain_target.get("stage_to_moment_ids") or {})
    confirmation_ids = [str(value).strip() for value in list(stage_to_moment_ids.get("confirmation") or []) if str(value).strip()]
    if confirmation_ids:
        has_confirmation = True
    else:
        for moment in list(evidence_moments or []):
            stage = _role_stage(moment.get("role"))
            if stage == "confirmation":
                has_confirmation = True
                break

    finalize_required = ["trigger", "confirmation"] if has_confirmation else ["trigger"]
    return {
        "decision_required_stages": ["trigger"],
        "finalize_required_stages": finalize_required,
        "narrative_required_stages": ["precursor"],
    }


def _interval_iou(pred_interval: Sequence[float] | None, ref_interval: Sequence[float] | None) -> float:
    if not pred_interval or not ref_interval or len(pred_interval) != 2 or len(ref_interval) != 2:
        return 0.0
    pred_start, pred_end = float(pred_interval[0]), float(pred_interval[1])
    ref_start, ref_end = float(ref_interval[0]), float(ref_interval[1])
    if pred_end < pred_start:
        pred_start, pred_end = pred_end, pred_start
    if ref_end < ref_start:
        ref_start, ref_end = ref_end, ref_start
    intersection = max(0.0, min(pred_end, ref_end) - max(pred_start, ref_start))
    union = max(pred_end, ref_end) - min(pred_start, ref_start)
    if union <= 0:
        return 0.0
    return intersection / union


def _normalize_decision_payload(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = dict(payload or {})
    normalized: Dict[str, Any] = {}
    normalized["existence"] = normalize_existence(payload.get("existence"))
    normalized["category"] = canonicalize_saver_category(
        payload.get("category"),
        existence=normalized["existence"] or None,
    ) or str(payload.get("category") or "").strip().lower()
    interval = payload.get("anomaly_interval_sec")
    normalized["anomaly_interval_sec"] = list(interval) if isinstance(interval, (list, tuple)) and len(interval) == 2 else None
    return normalized


def _extract_final_decision_payload(rollout: Dict[str, Any]) -> Dict[str, Any]:
    state = dict(rollout.get("state") or {})
    for candidate in (rollout.get("final_answer"), state.get("finalized_case")):
        if isinstance(candidate, dict) and candidate:
            return _normalize_decision_payload(candidate)
    return {}


def _reference_stage_moments(reference_record: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    evidence_moments = list(((reference_record.get("evidence") or {}).get("evidence_moments") or []))
    by_id = {
        str(moment.get("moment_id") or "").strip(): dict(moment)
        for moment in evidence_moments
        if str(moment.get("moment_id") or "").strip()
    }
    grouped: Dict[str, List[Dict[str, Any]]] = {stage: [] for stage in STAGE_ORDER}
    chain_target = dict((reference_record.get("structured_target") or {}).get("event_chain_target") or {})
    stage_to_moment_ids = dict(chain_target.get("stage_to_moment_ids") or {})
    for stage in STAGE_ORDER:
        for moment_id in list(stage_to_moment_ids.get(stage) or []):
            moment = by_id.get(str(moment_id).strip())
            if moment is not None:
                grouped[stage].append(moment)
    for moment in evidence_moments:
        stage = _role_stage(moment.get("role"))
        if stage and all(str(existing.get("moment_id") or "") != str(moment.get("moment_id") or "") for existing in grouped[stage]):
            grouped[stage].append(dict(moment))
    return grouped


def _record_interval(entry: Dict[str, Any]) -> List[float] | None:
    start_sec = entry.get("start_sec")
    end_sec = entry.get("end_sec")
    if start_sec is None or end_sec is None:
        return None
    try:
        start_val = float(start_sec)
        end_val = float(end_sec)
    except Exception:
        return None
    if end_val < start_val:
        start_val, end_val = end_val, start_val
    return [start_val, end_val]


def _moment_interval(moment: Dict[str, Any]) -> List[float] | None:
    try:
        start_val = float(moment.get("start_sec"))
        end_val = float(moment.get("end_sec"))
    except Exception:
        return None
    if end_val < start_val:
        start_val, end_val = end_val, start_val
    return [start_val, end_val]


def _records_support_against_moments(records: Sequence[Dict[str, Any]], moments: Sequence[Dict[str, Any]]) -> float:
    max_score = 0.0
    for record in list(records or []):
        record_interval = _record_interval(record)
        if record_interval is None:
            continue
        for moment in list(moments or []):
            score = _interval_iou(record_interval, _moment_interval(moment))
            if score > max_score:
                max_score = float(score)
    return round(float(max_score), 6)


def _structured_decision_scores(
    *,
    rollout: Dict[str, Any],
    target: Dict[str, Any],
) -> Dict[str, float]:
    prediction = _extract_final_decision_payload(rollout)
    scores: Dict[str, float] = {}
    target_existence = normalize_existence(target.get("existence"))
    prediction_existence = normalize_existence(prediction.get("existence"))
    if prediction_existence and target_existence:
        scores["existence"] = 1.0 if prediction_existence == target_existence else 0.0
    else:
        scores["existence"] = 0.0
    target_category = canonicalize_saver_category(target.get("category"), existence=target_existence or None) or str(target.get("category") or "").strip().lower()
    prediction_category = canonicalize_saver_category(prediction.get("category"), existence=prediction_existence or None) or str(prediction.get("category") or "").strip().lower()
    scores["category"] = 1.0 if target_category and prediction_category == target_category else 0.0
    target_interval = target.get("anomaly_interval_sec")
    prediction_interval = prediction.get("anomaly_interval_sec")
    if target_existence == "anomaly" and isinstance(target_interval, (list, tuple)) and len(target_interval) == 2:
        scores["temporal"] = round(float(_interval_iou(prediction_interval, target_interval)), 6)
    return scores


def _average(values: Sequence[float]) -> float:
    values = [float(value) for value in list(values or [])]
    if not values:
        return 0.0
    return round(sum(values) / float(len(values)), 6)


def _required_stage_list(target: Dict[str, Any], stage_requirements: Dict[str, List[str]]) -> List[str]:
    chain_target = dict(target.get("event_chain_target") or {})
    required = [str(stage).strip().lower() for stage in list(chain_target.get("required_stages") or []) if str(stage).strip()]
    if required:
        return [stage for stage in STAGE_ORDER if stage in required]
    fallback = [str(stage).strip().lower() for stage in list(stage_requirements.get("finalize_required_stages") or []) if str(stage).strip()]
    return [stage for stage in STAGE_ORDER if stage in fallback]


def _build_normal_skip_profile(
    *,
    item: Dict[str, Any],
    rollout: Dict[str, Any],
) -> Dict[str, Any]:
    selection_resolution = _resolve_selected_window_ids_for_normal_skip(rollout)
    selected_window_ids = list(selection_resolution.get("selected_window_ids") or [])
    selected_records = _selected_window_records(rollout, selected_window_ids)
    stage_by_moment_id = _stage_lookup_from_target(
        ((item.get("multimodal_cache") or {}).get("structured_target") or item.get("structured_target")),
        reference_record=item.get("record") or item,
    )
    selected_by_stage = _stage_window_ids(selected_records, stage_by_moment_id=stage_by_moment_id)
    profile_source = "normal_skip_v1"
    return {
        "counterfactual_branches": {},
        "counterfactual_profile": {
            "summary": {
                "decision_sufficiency": False,
                "minimal_subset_sufficiency": False,
                "negative_specificity_pass": False,
                "stage_necessity": {},
            },
            "branch_field_matrix": {},
            "branch_delta_matrix": {},
            "stage_packages": {
                "selected_window_ids": list(selected_window_ids),
                "selected_by_stage": copy.deepcopy(selected_by_stage),
                "minimal_subset_window_ids": [],
                "hard_negative_window_ids": [],
            },
            "selection_metadata": {
                "normalized_branch_profile": profile_source,
                "selected_window_count": len(selected_window_ids),
                "selected_record_count": len(selected_records),
                "selection_resolution_source": str(
                    selection_resolution.get("selection_resolution_source") or "normal_sample_skipped"
                ),
                "recovered_from_trace": bool(selection_resolution.get("recovered_from_trace", False)),
                "selected_by_stage": copy.deepcopy(selected_by_stage),
                "stage_requirements": {
                    "decision_required_stages": [],
                    "finalize_required_stages": [],
                    "narrative_required_stages": [],
                },
                "stage_queries": {},
                "minimal_subset_trace": [],
                "full_selected_available": False,
                "full_selected_parse_mode": "skipped_normal",
                "full_selected_unavailable_reason": "normal_sample_skipped",
                "full_selected_window_ids": [],
                "hard_negative_reason": "normal_sample_skipped",
            },
            "counterfactual_profile_source": profile_source,
            "counterfactual_branch_profile": profile_source,
        },
        "counterfactual_profile_source": profile_source,
        "counterfactual_branch_profile": profile_source,
    }


def _tokenize(text: Any) -> List[str]:
    normalized = normalize_text_match(text)
    return [token for token in normalized.split(" ") if token]


def _lcs_length(pred_tokens: Sequence[str], ref_tokens: Sequence[str]) -> int:
    if not pred_tokens or not ref_tokens:
        return 0
    previous = [0 for _ in range(len(ref_tokens) + 1)]
    for pred_token in pred_tokens:
        current = [0]
        for ref_idx, ref_token in enumerate(ref_tokens, start=1):
            if pred_token == ref_token:
                current.append(previous[ref_idx - 1] + 1)
            else:
                current.append(max(current[-1], previous[ref_idx]))
        previous = current
    return int(previous[-1])


def _rouge_l_f1(prediction: Any, reference: Any) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if not pred_tokens or not ref_tokens or lcs <= 0:
        return 0.0
    precision = float(lcs) / float(len(pred_tokens))
    recall = float(lcs) / float(len(ref_tokens))
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _window_record_lookup(rollout: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    state = rollout.get("state") or {}
    for source_key in ("visited_windows", "evidence_ledger"):
        for entry in list(state.get(source_key) or []):
            window_id = str(entry.get("window_id") or "").strip()
            if not window_id or window_id in lookup:
                continue
            lookup[window_id] = copy.deepcopy(entry)
    return lookup


def _selected_window_records(rollout: Dict[str, Any], window_ids: Sequence[str]) -> List[Dict[str, Any]]:
    lookup = _window_record_lookup(rollout)
    return [copy.deepcopy(lookup[window_id]) for window_id in _dedupe_window_ids(window_ids) if window_id in lookup]


def _role_stage(role: Any) -> str:
    return ROLE_TO_STAGE.get(str(role or "").strip().lower(), "")


def _stage_lookup_from_target(
    target: Dict[str, Any] | None,
    *,
    reference_record: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    normalized_target = dict(target or {})
    chain_target = dict(normalized_target.get("event_chain_target") or {})
    stage_to_moment_ids = normalize_stage_selected_moment_ids(
        chain_target.get("stage_to_moment_ids") or normalized_target.get("stage_selected_moment_ids")
    )
    for stage, moment_ids in stage_to_moment_ids.items():
        for moment_id in moment_ids:
            text = str(moment_id).strip()
            if text:
                lookup[text] = stage
    if not isinstance(reference_record, dict):
        return lookup
    evidence_moments = list(((reference_record.get("evidence") or {}).get("evidence_moments") or []))
    for moment in evidence_moments:
        moment_id = str(moment.get("moment_id") or "").strip()
        stage = _role_stage(moment.get("role"))
        if moment_id and stage and moment_id not in lookup:
            lookup[moment_id] = stage
    return lookup


def _stage_for_entry(
    entry: Dict[str, Any],
    *,
    stage_by_moment_id: Dict[str, str] | None = None,
) -> str:
    stage = _role_stage(entry.get("role"))
    if stage:
        return stage
    moment_id = str(entry.get("moment_id") or "").strip()
    if moment_id:
        return str((stage_by_moment_id or {}).get(moment_id) or "")
    return ""


def _stage_window_ids(
    records: Sequence[Dict[str, Any]],
    *,
    stage_by_moment_id: Dict[str, str] | None = None,
) -> Dict[str, List[str]]:
    grouped = {stage: [] for stage in STAGE_ORDER}
    for entry in list(records or []):
        stage = _stage_for_entry(entry, stage_by_moment_id=stage_by_moment_id)
        if not stage:
            continue
        window_id = str(entry.get("window_id") or "").strip()
        if window_id and window_id not in grouped[stage]:
            grouped[stage].append(window_id)
    return grouped


def _derive_scene_context(item: Dict[str, Any], reference_record: Dict[str, Any]) -> str:
    multimodal_cache = dict(item.get("multimodal_cache") or {})
    for candidate in (
        multimodal_cache.get("scene_context"),
        multimodal_cache.get("scene"),
        (reference_record.get("scene") or {}).get("scenario") if isinstance(reference_record.get("scene"), dict) else reference_record.get("scene"),
        (reference_record.get("scene") or {}).get("place") if isinstance(reference_record.get("scene"), dict) else "",
    ):
        if isinstance(candidate, dict):
            for nested in candidate.values():
                if str(nested or "").strip():
                    return str(nested).strip()
        text = str(candidate or "").strip()
        if text:
            return text
    return ""


def _build_stage_query_map(
    *,
    item: Dict[str, Any],
    reference_record: Dict[str, Any],
) -> Dict[str, str]:
    scene_context = _derive_scene_context(item, reference_record)
    stage_queries: Dict[str, str] = {}
    for moment in list(((reference_record.get("evidence") or {}).get("evidence_moments") or [])):
        stage = _role_stage(moment.get("role"))
        description = str(moment.get("description") or "").strip()
        if not stage or not description or stage in stage_queries:
            continue
        stage_queries[stage] = compose_scene_anchored_query(description, scene_context) or description
    return stage_queries


def _window_semantic_text(entry: Dict[str, Any]) -> str:
    pieces = [
        str(entry.get("description") or "").strip(),
        str(entry.get("query") or "").strip(),
        str(entry.get("moment_description") or "").strip(),
        str(entry.get("role") or "").strip(),
    ]
    return " ".join(piece for piece in pieces if piece)


def _choose_minimal_subset_window_ids(
    selected_records: Sequence[Dict[str, Any]],
    *,
    stage_requirements: Dict[str, List[str]],
    stage_by_moment_id: Dict[str, str] | None = None,
) -> List[str]:
    selected_records = list(selected_records or [])
    if not selected_records:
        return []
    by_stage = _stage_window_ids(selected_records, stage_by_moment_id=stage_by_moment_id)
    wanted_stages = []
    for key in ("decision_required_stages", "finalize_required_stages"):
        for stage in list(stage_requirements.get(key) or []):
            if stage not in wanted_stages:
                wanted_stages.append(stage)
    minimal: List[str] = []
    for stage in wanted_stages:
        if by_stage.get(stage):
            minimal.append(str(by_stage[stage][0]))
    if not minimal:
        return [str(selected_records[0].get("window_id"))]
    return _dedupe_window_ids(minimal)


def _branch_stage_drop_window_ids(
    selected_window_ids: Sequence[str],
    selected_records: Sequence[Dict[str, Any]],
    *,
    stage: str,
    stage_by_moment_id: Dict[str, str] | None = None,
) -> Tuple[List[str], bool]:
    stage_window_ids = set(
        _stage_window_ids(selected_records, stage_by_moment_id=stage_by_moment_id).get(stage) or []
    )
    if not stage_window_ids:
        return [], False
    kept = [window_id for window_id in _dedupe_window_ids(selected_window_ids) if window_id not in stage_window_ids]
    return kept, True


def _midpoint(entry: Dict[str, Any]) -> float:
    return 0.5 * (_safe_float(entry.get("start_sec")) + _safe_float(entry.get("end_sec")))


def _build_hard_negative_swap_window_ids(
    rollout: Dict[str, Any],
    *,
    selected_window_ids: Sequence[str],
    selected_records: Sequence[Dict[str, Any]],
    stage_queries: Dict[str, str] | None = None,
    stage_by_moment_id: Dict[str, str] | None = None,
) -> Tuple[List[str], bool, str]:
    selected_window_ids = _dedupe_window_ids(selected_window_ids)
    if not selected_window_ids:
        return [], False, "no_selected_windows"
    stage_groups = _stage_window_ids(selected_records, stage_by_moment_id=stage_by_moment_id)
    key_stage = ""
    for stage in ("trigger", "confirmation", "precursor"):
        if stage_groups.get(stage):
            key_stage = stage
            break
    if not key_stage:
        return [], False, "no_stage_specific_window_to_swap"
    stage_to_drop = set(stage_groups.get(key_stage) or [])
    lookup = _window_record_lookup(rollout)
    target_mid = _safe_float(sum(_midpoint(lookup[window_id]) for window_id in stage_to_drop) / max(len(stage_to_drop), 1))
    query_text = str((stage_queries or {}).get(key_stage) or "").strip()
    query_tokens = {
        str(entry.get("text") or "").strip().lower()
        for entry in normalize_description_query_phrases(query_text)
        if str(entry.get("text") or "").strip()
    }
    candidate_entries: List[Tuple[float, float, str]] = []
    for window_id, entry in lookup.items():
        if window_id in selected_window_ids:
            continue
        if _stage_for_entry(entry) == key_stage:
            continue
        entry_tokens = set(_tokenize(_window_semantic_text(entry)))
        overlap = len(query_tokens & entry_tokens) / float(max(len(query_tokens), 1)) if query_tokens else 0.0
        midpoint_distance = abs(_midpoint(entry) - target_mid)
        candidate_entries.append((-overlap, midpoint_distance, window_id))
    candidate_entries.sort(key=lambda item: (item[0], item[1], item[2]))
    if len(candidate_entries) < len(stage_to_drop):
        return [], False, "insufficient_negative_windows"
    replacement_ids = [window_id for _, _, window_id in candidate_entries[: len(stage_to_drop)]]
    swapped = [window_id for window_id in selected_window_ids if window_id not in stage_to_drop] + replacement_ids
    return _dedupe_window_ids(swapped), True, ""


def _resolve_counterfactual_branch_order(
    branch_profile: str,
    *,
    stage_requirements: Dict[str, List[str]],
) -> List[str]:
    del stage_requirements
    normalized = _normalize_counterfactual_branch_profile(branch_profile)
    if normalized == "online_core":
        return [
            "full_selected",
            "minimal_subset",
            "drop_trigger",
        ]
    return list(BRANCH_ORDER)


def _branch_uses_compact_decision_only(
    *,
    normalized_branch_profile: str,
    branch_name: str,
) -> bool:
    normalized_branch_name = str(branch_name or "").strip().lower()
    if str(normalized_branch_profile or "").strip().lower() != "online_core":
        return False
    return normalized_branch_name in {"full_selected", "minimal_subset", "drop_trigger"}


def _build_compact_semantic_scaffold(
    *,
    target: Dict[str, Any],
    branch_name: str,
) -> str:
    del branch_name
    del target
    scaffold = {
        "decision": build_replay_decision_scaffold(),
        "covered_stages": [],
        "missing_required_stages": [],
        "stage_selected_moment_ids": {},
        "event_chain_summary": {
            "precursor": "",
            "trigger": "",
            "confirmation": "",
        },
    }
    return json.dumps(scaffold, ensure_ascii=False, separators=(",", ":"))


def _build_reference_payload(reference_record: Dict[str, Any]) -> Dict[str, Any]:
    return build_semantic_answer_payload(
        structured_target=reference_record.get("structured_target") or {},
        qa_pairs=reference_record.get("qa_pairs") or [],
        evidence_moments=((reference_record.get("evidence") or {}).get("evidence_moments") or []),
        finalized_case=reference_record.get("structured_target") or {},
    )


def _compare_existence(prediction_payload: Dict[str, Any] | None, reference_payload: Dict[str, Any]) -> Tuple[float, bool]:
    prediction = str(((prediction_payload or {}).get("decision") or {}).get("existence") or "").strip().lower()
    reference = str((reference_payload.get("decision") or {}).get("existence") or "").strip().lower()
    supported = bool(prediction and prediction == reference)
    return (1.0 if supported else 0.0), supported


def _compare_category(prediction_payload: Dict[str, Any] | None, reference_payload: Dict[str, Any]) -> Tuple[float, bool]:
    pred_decision = (prediction_payload or {}).get("decision") or {}
    ref_decision = reference_payload.get("decision") or {}
    prediction = canonicalize_saver_category(pred_decision.get("category"), existence=pred_decision.get("existence"))
    reference = canonicalize_saver_category(ref_decision.get("category"), existence=ref_decision.get("existence"))
    supported = bool(reference) and prediction == reference
    return (1.0 if supported else 0.0), supported


def _compare_temporal(prediction_payload: Dict[str, Any] | None, reference_payload: Dict[str, Any]) -> Tuple[float, bool]:
    pred_decision = (prediction_payload or {}).get("decision") or {}
    ref_decision = reference_payload.get("decision") or {}
    prediction_interval = pred_decision.get("anomaly_interval_sec")
    reference_interval = ref_decision.get("anomaly_interval_sec")
    if isinstance(prediction_interval, list) and isinstance(reference_interval, list):
        score = _interval_iou(prediction_interval, reference_interval)
        return score, bool(score >= 0.5)
    pred_text = str(((prediction_payload or {}).get("qa_focus_answers") or {}).get("temporal") or "").strip()
    ref_text = str((reference_payload.get("qa_focus_answers") or {}).get("temporal") or "").strip()
    supported = bool(ref_text) and normalize_text_match(pred_text) == normalize_text_match(ref_text)
    return (1.0 if supported else 0.0), supported


def _compare_stage_text(
    prediction_payload: Dict[str, Any] | None,
    reference_payload: Dict[str, Any],
    *,
    stage: str,
) -> Tuple[float, bool]:
    pred_text = str(((prediction_payload or {}).get("event_chain_summary") or {}).get(stage) or "").strip()
    ref_text = str((reference_payload.get("event_chain_summary") or {}).get(stage) or "").strip()
    if not ref_text:
        return 0.0, not pred_text
    score = _rouge_l_f1(pred_text, ref_text)
    return score, bool(score >= STAGE_TEXT_THRESHOLD)


def _field_support(
    prediction_payload: Dict[str, Any] | None,
    reference_payload: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    support: Dict[str, Dict[str, Any]] = {}
    for field_name, compare_fn in (
        ("existence", _compare_existence),
        ("category", _compare_category),
        ("temporal", _compare_temporal),
    ):
        score, supported = compare_fn(prediction_payload, reference_payload)
        support[field_name] = {"score": round(float(score), 6), "supported": bool(supported)}
    for stage in STAGE_ORDER:
        score, supported = _compare_stage_text(prediction_payload, reference_payload, stage=stage)
        support[stage] = {"score": round(float(score), 6), "supported": bool(supported)}
    return support


def _core_decision_supported(
    target: Dict[str, Any],
    field_support: Dict[str, Dict[str, Any]],
) -> bool:
    existence = str(target.get("existence") or "").strip().lower()
    if not bool(field_support.get("existence", {}).get("supported")):
        return False
    if not bool(field_support.get("category", {}).get("supported")):
        return False
    if existence == "anomaly" and target.get("anomaly_interval_sec") is not None:
        if not bool(field_support.get("temporal", {}).get("supported")):
            return False
    return True


def _branch_finalize_readiness(
    *,
    branch_window_ids: Sequence[str],
    rollout: Dict[str, Any],
    field_support: Dict[str, Dict[str, Any]],
    stage_requirements: Dict[str, List[str]],
    target: Dict[str, Any],
) -> Dict[str, Any]:
    records = _selected_window_records(rollout, branch_window_ids)
    stage_by_moment_id = _stage_lookup_from_target(target)
    supported_stages = [
        stage
        for stage, values in _stage_window_ids(records, stage_by_moment_id=stage_by_moment_id).items()
        if values
    ]
    finalize_required = list(stage_requirements.get("finalize_required_stages") or [])
    missing_required_stages = [stage for stage in finalize_required if stage not in supported_stages]
    coverage_ratio = (
        float(len(finalize_required) - len(missing_required_stages)) / float(len(finalize_required))
        if finalize_required
        else 1.0
    )
    core_supported = _core_decision_supported(target, field_support)
    readiness_score = coverage_ratio * (1.0 if core_supported else 0.4)
    return {
        "supported_stages": supported_stages,
        "missing_required_stages": missing_required_stages,
        "finalize_readiness_score": round(float(readiness_score), 6),
    }


def _branch_supports_decision(
    branch: Dict[str, Any],
    *,
    target: Dict[str, Any],
) -> bool:
    field_support = dict(branch.get("field_support") or {})
    branch_verification = dict(branch.get("branch_verification") or {})
    readiness_score = _safe_float(branch_verification.get("finalize_readiness_score"), 0.0)
    if not _core_decision_supported(target, field_support):
        return False
    if str(target.get("existence") or "").strip().lower() == "anomaly":
        return readiness_score >= FINALIZE_READINESS_THRESHOLD
    return True


def _branch_field_matrix_entry(
    branch: Dict[str, Any],
    *,
    target: Dict[str, Any],
) -> Dict[str, Any]:
    branch_verification = dict(branch.get("branch_verification") or {})
    fields = copy.deepcopy(branch.get("field_support") or {})
    readiness_score = _safe_float(branch_verification.get("finalize_readiness_score"), 0.0)
    fields["finalize_readiness"] = {
        "score": round(float(readiness_score), 6),
        "supported": bool(readiness_score >= FINALIZE_READINESS_THRESHOLD),
    }
    return {
        "available": bool(branch.get("available")),
        "window_ids": list(branch.get("window_ids") or []),
        "fields": fields,
        "core_decision_supported": bool(_core_decision_supported(target, dict(branch.get("field_support") or {}))),
        "supported_stages": list(branch_verification.get("supported_stages") or []),
        "missing_required_stages": list(branch_verification.get("missing_required_stages") or []),
    }


def _field_score_from_matrix(branch_matrix_entry: Dict[str, Any], field_name: str) -> float:
    return _safe_float(((branch_matrix_entry.get("fields") or {}).get(field_name) or {}).get("score"), 0.0)


def _build_branch_delta_matrix(branch_field_matrix: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    full_entry = dict(branch_field_matrix.get("full_selected") or {})
    full_fields = dict(full_entry.get("fields") or {})
    all_field_names = set(full_fields)
    for entry in branch_field_matrix.values():
        all_field_names.update((entry.get("fields") or {}).keys())
    delta_matrix: Dict[str, Dict[str, Any]] = {}
    for branch_name, entry in branch_field_matrix.items():
        fields_delta: Dict[str, float] = {}
        for field_name in sorted(all_field_names):
            full_score = _safe_float((full_fields.get(field_name) or {}).get("score"), 0.0)
            branch_score = _field_score_from_matrix(entry, field_name)
            fields_delta[field_name] = round(max(0.0, full_score - branch_score), 6)
        delta_matrix[branch_name] = {
            "available": bool(entry.get("available")),
            "window_ids": list(entry.get("window_ids") or []),
            "fields": fields_delta,
            "core_decision_drop": round(
                max(
                    0.0,
                    float(bool(full_entry.get("core_decision_supported")))
                    - float(bool(entry.get("core_decision_supported"))),
                ),
                6,
            ),
        }
    return delta_matrix


def _derive_stage_necessity_from_deltas(
    *,
    branch_field_matrix: Dict[str, Dict[str, Any]],
    branch_delta_matrix: Dict[str, Dict[str, Any]],
) -> Dict[str, str]:
    precursor_delta = dict((branch_delta_matrix.get("drop_precursor") or {}).get("fields") or {})
    trigger_delta = dict((branch_delta_matrix.get("drop_trigger") or {}).get("fields") or {})
    confirmation_delta = dict((branch_delta_matrix.get("drop_confirmation") or {}).get("fields") or {})

    trigger_decision_drop = max(
        _safe_float(trigger_delta.get("existence"), 0.0),
        _safe_float(trigger_delta.get("category"), 0.0),
        _safe_float(trigger_delta.get("temporal"), 0.0),
    )
    precursor_decision_drop = max(
        _safe_float(precursor_delta.get("existence"), 0.0),
        _safe_float(precursor_delta.get("category"), 0.0),
        _safe_float(precursor_delta.get("temporal"), 0.0),
    )
    confirmation_decision_drop = max(
        _safe_float(confirmation_delta.get("existence"), 0.0),
        _safe_float(confirmation_delta.get("category"), 0.0),
        _safe_float(confirmation_delta.get("temporal"), 0.0),
    )

    precursor_available = bool((branch_field_matrix.get("drop_precursor") or {}).get("available"))
    trigger_available = bool((branch_field_matrix.get("drop_trigger") or {}).get("available"))
    confirmation_available = bool((branch_field_matrix.get("drop_confirmation") or {}).get("available"))

    if not precursor_available:
        precursor_label = "not_observed"
    elif precursor_decision_drop >= 0.5:
        precursor_label = "decision_critical"
    elif _safe_float(precursor_delta.get("precursor"), 0.0) >= STAGE_TEXT_THRESHOLD:
        precursor_label = "narrative_only"
    else:
        precursor_label = "optional"

    if not trigger_available:
        trigger_label = "not_observed"
    elif trigger_decision_drop >= 0.5:
        trigger_label = "decision_critical"
    else:
        trigger_label = "non_critical"

    if not confirmation_available:
        confirmation_label = "not_observed"
    elif (
        _safe_float(confirmation_delta.get("confirmation"), 0.0) >= STAGE_TEXT_THRESHOLD
        or _safe_float(confirmation_delta.get("finalize_readiness"), 0.0) >= 0.25
        or confirmation_decision_drop >= 0.5
    ):
        confirmation_label = "finalize_critical"
    else:
        confirmation_label = "optional"

    return {
        "precursor": precursor_label,
        "trigger": trigger_label,
        "confirmation": confirmation_label,
    }


def _branch_pass(
    branch_name: str,
    *,
    target: Dict[str, Any],
    field_support: Dict[str, Dict[str, Any]],
    branch_verification: Dict[str, Any],
    stage_requirements: Dict[str, List[str]],
) -> bool:
    core_supported = _core_decision_supported(target, field_support)
    if branch_name in {"full_selected", "minimal_subset"}:
        return core_supported
    if branch_name == "drop_precursor":
        return (
            not bool(field_support.get("precursor", {}).get("supported"))
            and core_supported
        )
    if branch_name == "drop_trigger":
        return not core_supported
    if branch_name == "drop_confirmation":
        confirmation_required = "confirmation" in (stage_requirements.get("finalize_required_stages") or [])
        confirmation_drops = not bool(field_support.get("confirmation", {}).get("supported"))
        readiness_low = _safe_float(branch_verification.get("finalize_readiness_score"), 0.0) < 0.75
        return confirmation_drops and (readiness_low or (confirmation_required and not core_supported))
    if branch_name == "hard_negative_swap":
        return not core_supported
    return False


def _build_counterfactual_messages(
    item: Dict[str, Any],
    *,
    rollout: Dict[str, Any],
    branch_name: str,
    window_ids: Sequence[str],
    target: Dict[str, Any],
    max_images: int = 12,
    compact_decision_only: bool = False,
) -> List[Dict[str, Any]]:
    selected_records = _selected_window_records(rollout, window_ids)
    multimodal_cache = dict(item.get("multimodal_cache") or {})
    question = str(multimodal_cache.get("question") or "Determine the final structured anomaly decision from the evidence.")
    if compact_decision_only:
        scaffold = _build_compact_semantic_scaffold(target=target, branch_name=branch_name)
        output_instruction = (
            f"Return exactly one <answer></answer> JSON in this compact shape: {scaffold}\n"
            "Keep only decision, covered_stages, missing_required_stages, stage_selected_moment_ids, and event_chain_summary. "
            "Populate event_chain_summary for stages supported by this branch and leave unsupported stages as empty strings. "
            "Do not add summary, rationale, or qa_focus_answers."
        )
    else:
        scaffold = build_semantic_answer_scaffold(finalized_case=target)
        output_instruction = f"Return exactly one <answer></answer> JSON in this shape: {scaffold}"
    user_content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Use only the provided evidence frames and timestamps. "
                "Do not search, do not call tools, and do not assume evidence outside this package.\n"
                f"Counterfactual branch: {branch_name}\n"
                f"Task: {question}\n"
                f"Selected evidence window ids: {json.dumps(list(window_ids), ensure_ascii=False)}\n"
                f"Evidence ledger summary: {summarize_evidence_ledger(selected_records)}\n"
                f"{output_instruction}"
            ),
        }
    ]

    video = multimodal_cache.get("video")
    image_count = 0
    for record in selected_records:
        frame_indices = list(record.get("selected_frame_indices") or [])
        timestamps = list(record.get("selected_timestamps") or [])
        for frame_index, timestamp in zip(frame_indices, timestamps):
            if image_count >= int(max_images):
                break
            if video is None:
                continue
            try:
                image = video[int(frame_index)]
            except Exception:
                continue
            user_content.append({"type": "text", "text": f"{float(timestamp):.3f}s"})
            user_content.append(
                {
                    "type": "image",
                    "image": image,
                    "sampled_frame_index": int(frame_index),
                    "timestamp_sec": float(timestamp),
                }
            )
            image_count += 1
        if image_count >= int(max_images):
            break

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a SAVER counterfactual verifier. "
                        "You must not call tools. "
                        "Answer only from the provided visual evidence."
                    ),
                }
            ],
        },
        {"role": "user", "content": user_content},
    ]


def _run_counterfactual_branch_replay(policy: Any, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    result = _run_counterfactual_branch_replay_batch(
        policy,
        [
            {
                "messages": list(messages or []),
                "target": {},
                "compact_decision_only": False,
                "branch_name": "counterfactual_replay",
            }
        ],
    )
    return dict(result[0]) if result else _unavailable_branch_payload(window_ids=[], reason="counterfactual_replay_missing")


def _counterfactual_answer_retry_prompt(
    *,
    reason: str,
    target: Dict[str, Any],
    compact_decision_only: bool,
) -> str:
    normalized_reason = str(reason or "").strip() or "invalid_action_format"
    reason_prefix = {
        "missing_answer_tag": "The previous response did not contain any <answer></answer> block.",
        "invalid_answer_json": "The previous <answer> block did not contain valid JSON.",
        "normalized_payload_empty": (
            "The previous <answer> block did not contain a usable semantic decision payload."
        ),
        "tool_call_not_allowed": "Counterfactual verification replay cannot call tools.",
        "invalid_action_format": "The previous response did not follow the required structured format.",
    }.get(normalized_reason, "The previous response violated the counterfactual replay protocol.")
    if compact_decision_only:
        scaffold = _build_compact_semantic_scaffold(
            target=target,
            branch_name="counterfactual_replay",
        )
        shape_instruction = (
            f"Retry immediately with exactly one <answer></answer> JSON block in this compact shape: {scaffold}. "
            "Keep only decision and event_chain_summary. "
            "Leave unsupported stages in event_chain_summary as empty strings. "
            "Do not add summary, rationale, or qa_focus_answers."
        )
    else:
        scaffold = build_semantic_answer_scaffold(finalized_case=target)
        shape_instruction = (
            f"Retry immediately with exactly one <answer></answer> JSON block in this shape: {scaffold}. "
            "Do not output plain text outside <answer>."
        )
    return (
        f"{reason_prefix} {shape_instruction} "
        "Do not output <tool_call>. Do not explain the answer in prose."
    )


def _counterfactual_parse_error_message(
    *,
    reason: str,
    target: Dict[str, Any],
    compact_decision_only: bool,
) -> Dict[str, Any]:
    return {
        "role": "tool",
        "name": "parse_error",
        "content": [
            {
                "type": "text",
                "text": _counterfactual_answer_retry_prompt(
                    reason=reason,
                    target=target,
                    compact_decision_only=compact_decision_only,
                ),
            }
        ],
    }


def _counterfactual_failure_preview(response_text: str, *, limit: int = 200) -> str:
    text = " ".join(str(response_text or "").split())
    if len(text) <= int(limit):
        return text
    return text[: max(0, int(limit) - 3)] + "..."


def _classify_counterfactual_branch_replay_response(response_text: str) -> Dict[str, Any]:
    cleaned_response = cleanup_llm_response(str(response_text or ""))
    actions, contents = parse_actions_and_contents([cleaned_response])
    action = actions[0]
    if action == "tool_call":
        return {
            "response_text": cleaned_response,
            "semantic_answer": None,
            "semantic_answer_text": None,
            "final_answer": None,
            "available": False,
            "unavailable_reason": "tool_call_not_allowed",
            "parsed_action": "tool_call",
            "failure_preview": _counterfactual_failure_preview(cleaned_response),
            "raw_parsed_content": contents[0],
        }
    parsed = _parse_counterfactual_branch_replay_response(cleaned_response)
    parsed["parsed_action"] = str(action or "")
    parsed["failure_preview"] = (
        _counterfactual_failure_preview(cleaned_response)
        if not bool(parsed.get("available"))
        else ""
    )
    return parsed


def _generate_counterfactual_branch_replay_text_batch(
    policy: Any,
    messages_batch: Sequence[List[Dict[str, Any]]],
) -> List[str]:
    message_list = [list(messages or []) for messages in list(messages_batch or [])]
    if not message_list:
        return []
    batch_generate = getattr(policy, "generate_from_messages_batch", None)
    if callable(batch_generate):
        response_texts = list(batch_generate(message_list) or [])
    else:
        single_generate = getattr(policy, "generate_from_messages", None)
        if not callable(single_generate):
            raise AttributeError(
                "Counterfactual verification batch replay requires a policy with either "
                "generate_from_messages_batch(messages_batch) or generate_from_messages(messages)."
            )
        response_texts = [single_generate(messages) for messages in message_list]
    if len(response_texts) != len(message_list):
        raise ValueError(
            "Counterfactual verification batch replay returned an unexpected number of responses: "
            f"{len(response_texts)} vs {len(message_list)}"
        )
    return [str(response_text or "") for response_text in response_texts]


def _counterfactual_replay_result(
    *,
    response_text: str,
    payload: Optional[Dict[str, Any]],
    parse_mode: str,
    unavailable_reason: str,
) -> Dict[str, Any]:
    final_answer = extract_decision_from_semantic_answer(payload) if payload else None
    return {
        "response_text": response_text,
        "semantic_answer": payload,
        "semantic_answer_text": semantic_answer_to_text(payload),
        "final_answer": final_answer,
        "available": bool(payload is not None),
        "unavailable_reason": "" if payload is not None else str(unavailable_reason or ""),
        "parse_mode": str(parse_mode or ""),
    }


def _extract_balanced_json_object(text: str, *, start_index: int) -> Optional[str]:
    if start_index < 0 or start_index >= len(text) or text[start_index] != "{":
        return None
    balance = 0
    in_string = False
    escaped = False
    for index in range(start_index, len(text)):
        char = text[index]
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
            continue
        if char == "{":
            balance += 1
        elif char == "}":
            balance -= 1
            if balance == 0:
                return text[start_index : index + 1]
    return None


def _extract_balanced_json_array(text: str, *, start_index: int) -> Optional[str]:
    if start_index < 0 or start_index >= len(text) or text[start_index] != "[":
        return None
    balance = 0
    in_string = False
    escaped = False
    for index in range(start_index, len(text)):
        char = text[index]
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
            continue
        if char == "[":
            balance += 1
        elif char == "]":
            balance -= 1
            if balance == 0:
                return text[start_index : index + 1]
    return None


def _extract_json_value_snippet(text: str, *, key: str) -> Optional[str]:
    match = re.search(rf'"{re.escape(str(key))}"\s*:\s*', text)
    if match is None:
        return None
    index = match.end()
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text):
        return None
    char = text[index]
    if char == "{":
        return _extract_balanced_json_object(text, start_index=index)
    if char == "[":
        return _extract_balanced_json_array(text, start_index=index)
    if char == '"':
        escaped = False
        for end in range(index + 1, len(text)):
            current = text[end]
            if escaped:
                escaped = False
                continue
            if current == "\\":
                escaped = True
                continue
            if current == '"':
                return text[index : end + 1]
        return None
    end = index
    while end < len(text) and text[end] not in ",}":
        end += 1
    snippet = text[index:end].strip()
    return snippet or None


def _salvage_compact_decision_payload(payload_text: str) -> Optional[Dict[str, Any]]:
    marker = '"decision"'
    marker_index = payload_text.find(marker)
    if marker_index < 0:
        return None
    colon_index = payload_text.find(":", marker_index + len(marker))
    if colon_index < 0:
        return None
    brace_index = payload_text.find("{", colon_index + 1)
    if brace_index < 0:
        return None
    decision_object_text = _extract_balanced_json_object(payload_text, start_index=brace_index)
    candidate_texts: List[str] = []
    if decision_object_text:
        candidate_texts.append(decision_object_text)
    else:
        fallback_markers = (
            '},"summary":',
            '}, "summary":',
            '},"rationale":',
            '}, "rationale":',
            '},"event_chain_summary":',
            '}, "event_chain_summary":',
            '},"qa_focus_answers":',
            '}, "qa_focus_answers":',
            '},"required_stages":',
            '}, "required_stages":',
            '},"available_stages":',
            '}, "available_stages":',
            '},"stage_to_moment_ids":',
            '}, "stage_to_moment_ids":',
        )
        marker_positions = [
            payload_text.find(fallback_marker, brace_index + 1)
            for fallback_marker in fallback_markers
        ]
        marker_positions = [position for position in marker_positions if position > brace_index]
        if marker_positions:
            cutoff = min(marker_positions) + 1
            candidate_texts.append(payload_text[brace_index:cutoff])
    for candidate_text in candidate_texts:
        try:
            parsed = json.loads('{"decision":' + candidate_text + "}")
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _salvage_compact_decision_fields(payload_text: str) -> Optional[Dict[str, Any]]:
    decision: Dict[str, Any] = {}
    for key in (
        "existence",
        "category",
        "counterfactual_type",
        "severity",
        "hard_normal",
        "anomaly_interval_sec",
        "evidence_moment_ids",
    ):
        snippet = _extract_json_value_snippet(payload_text, key=key)
        if snippet is None:
            continue
        try:
            decision[key] = json.loads(snippet)
        except Exception:
            continue
    if not str(decision.get("existence") or "").strip():
        return None
    if not str(decision.get("category") or "").strip():
        return None
    return {"decision": decision}


def _parse_counterfactual_payload_text(
    payload_text: str,
    *,
    response_text: str,
    parse_mode: str,
    invalid_json_reason: str,
    compact_decision_only: bool = False,
) -> Dict[str, Any]:
    try:
        parsed = json.loads(payload_text)
    except Exception:
        repaired_payload_text = _repair_json_object_text(payload_text)
        if repaired_payload_text is None or repaired_payload_text == payload_text:
            repaired_payload_text = None
        try:
            parsed = json.loads(repaired_payload_text) if repaired_payload_text is not None else None
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            parse_mode = f"{parse_mode}_repaired"
        elif compact_decision_only:
            salvage_source = repaired_payload_text or payload_text
            parsed = _salvage_compact_decision_payload(salvage_source)
            if parsed is not None:
                parse_mode = f"{parse_mode}_decision_extracted"
            else:
                parsed = _salvage_compact_decision_fields(salvage_source)
                if parsed is not None:
                    parse_mode = f"{parse_mode}_decision_fields_extracted"
                else:
                    return _counterfactual_replay_result(
                        response_text=response_text,
                        payload=None,
                        parse_mode=parse_mode,
                        unavailable_reason=invalid_json_reason,
                    )
        else:
            return _counterfactual_replay_result(
                response_text=response_text,
                payload=None,
                parse_mode=parse_mode,
                unavailable_reason=invalid_json_reason,
            )
    if not isinstance(parsed, dict):
        return _counterfactual_replay_result(
            response_text=response_text,
            payload=None,
            parse_mode=parse_mode,
            unavailable_reason=invalid_json_reason,
        )
    payload = normalize_semantic_answer_payload(parsed)
    if payload is None:
        return _counterfactual_replay_result(
            response_text=response_text,
            payload=None,
            parse_mode=parse_mode,
            unavailable_reason="normalized_payload_empty",
        )
    return _counterfactual_replay_result(
        response_text=response_text,
        payload=payload,
        parse_mode=parse_mode,
        unavailable_reason="",
    )


def _parse_counterfactual_branch_replay_response(
    response_text: str,
    *,
    compact_decision_only: bool = False,
) -> Dict[str, Any]:
    cleaned_response = cleanup_llm_response(str(response_text or ""))
    actions, contents = parse_actions_and_contents([cleaned_response])
    action = actions[0]
    content = contents[0]
    if action == "tool_call":
        return _counterfactual_replay_result(
            response_text=cleaned_response,
            payload=None,
            parse_mode="tool_call",
            unavailable_reason="tool_call_not_allowed",
        )
    if action == "answer":
        return _parse_counterfactual_payload_text(
            str(content or ""),
            response_text=cleaned_response,
            parse_mode="answer_tag",
            invalid_json_reason="invalid_answer_json",
            compact_decision_only=bool(compact_decision_only),
        )
    if action == "invalid_answer":
        return _counterfactual_replay_result(
            response_text=cleaned_response,
            payload=None,
            parse_mode="answer_tag",
            unavailable_reason="invalid_answer_json",
        )
    if str(cleaned_response or "").lstrip().startswith("{"):
        return _parse_counterfactual_payload_text(
            cleaned_response,
            response_text=cleaned_response,
            parse_mode="bare_json",
            invalid_json_reason="invalid_bare_json",
            compact_decision_only=bool(compact_decision_only),
        )
    return _counterfactual_replay_result(
        response_text=cleaned_response,
        payload=None,
        parse_mode="failed",
        unavailable_reason="missing_answer_tag",
    )


def _raise_counterfactual_replay_protocol_error(
    *,
    request: Dict[str, Any],
    result: Dict[str, Any],
) -> None:
    rollout = dict(request.get("rollout") or {})
    item = dict(request.get("item") or {})
    branch_name = str(request.get("branch_name") or "")
    reason = str(result.get("unavailable_reason") or "")
    parse_mode = str(result.get("parse_mode") or "")
    response_text = str(result.get("response_text") or "")
    response_preview = _counterfactual_failure_preview(response_text)
    response_suffix = response_text[-200:] if response_text else ""
    repaired_candidate = _repair_json_object_text(response_text) if response_text.lstrip().startswith("{") else None
    repaired_candidate_suffix = repaired_candidate[-200:] if repaired_candidate else ""
    compact_decision_only = bool(request.get("compact_decision_only"))
    window_ids = list(_dedupe_window_ids(request.get("window_ids") or []))
    response_char_count = len(response_text)
    brace_balance = _json_brace_balance(response_text) if response_text.lstrip().startswith("{") else 0
    ends_with_brace = response_text.rstrip().endswith("}")
    has_answer_tag = "<answer>" in response_text and "</answer>" in response_text
    logger.error(
        "fecv replay protocol error: video_id=%s generation_id=%s branch_name=%s reason=%s parse_mode=%s response_preview=%s",
        str(rollout.get("video_id") or item.get("video_id") or ""),
        str(rollout.get("generation_id") or ""),
        branch_name,
        reason,
        parse_mode,
        response_preview,
        )
    if branch_name in {"minimal_subset", "full_selected"}:
        logger.error(
            "fecv %s protocol detail: video_id=%s generation_id=%s compact_decision_only=%s "
            "window_ids=%s stage_requirements=%s response_char_count=%s brace_balance=%s "
            "ends_with_brace=%s has_answer_tag=%s repaired_candidate_present=%s repaired_candidate_changed=%s "
            "response_suffix=%s repaired_candidate_suffix=%s",
            branch_name,
            str(rollout.get("video_id") or item.get("video_id") or ""),
            str(rollout.get("generation_id") or ""),
            compact_decision_only,
            window_ids,
            dict(request.get("stage_requirements") or {}),
            response_char_count,
            brace_balance,
            ends_with_brace,
            has_answer_tag,
            repaired_candidate is not None,
            repaired_candidate not in {None, response_text},
            response_suffix,
            repaired_candidate_suffix,
        )
    raise CounterfactualReplayProtocolError(
        video_id=str(rollout.get("video_id") or item.get("video_id") or ""),
        generation_id=str(rollout.get("generation_id") or ""),
        branch_name=branch_name,
        reason=reason,
        parse_mode=parse_mode,
        response_preview=response_preview,
    )


def _run_counterfactual_branch_replay_batch(
    policy: Any,
    requests: Sequence[Dict[str, Any]],
    *,
    raise_on_protocol_error: bool = True,
) -> List[Dict[str, Any]]:
    request_list = [dict(request or {}) for request in list(requests or [])]
    if not request_list:
        return []
    response_texts = _generate_counterfactual_branch_replay_text_batch(
        policy,
        [list(request.get("messages") or []) for request in request_list],
    )
    results: List[Dict[str, Any]] = []
    for request, response_text in zip(request_list, response_texts):
        result = _parse_counterfactual_branch_replay_response(
            response_text,
            compact_decision_only=bool(request.get("compact_decision_only")),
        )
        if not bool(result.get("available")):
            if raise_on_protocol_error:
                _raise_counterfactual_replay_protocol_error(request=request, result=result)
            result = dict(result)
            result["failure_preview"] = _counterfactual_failure_preview(str(result.get("response_text") or ""))
        results.append(result)
    return results


def _unavailable_branch_payload(
    *,
    window_ids: Sequence[str],
    reason: str,
) -> Dict[str, Any]:
    return {
        "available": False,
        "window_ids": list(_dedupe_window_ids(window_ids)),
        "response_text": None,
        "semantic_answer": None,
        "semantic_answer_text": None,
        "final_answer": None,
        "field_support": {},
        "branch_verification": {
            "supported_stages": [],
            "missing_required_stages": [],
            "finalize_readiness_score": 0.0,
        },
        "branch_pass": False,
        "unavailable_reason": str(reason),
    }


def _evaluate_branch_window_ids(
    policy: Any,
    *,
    item: Dict[str, Any],
    rollout: Dict[str, Any],
    reference_payload: Dict[str, Any],
    target: Dict[str, Any],
    stage_requirements: Dict[str, List[str]],
    branch_name: str,
    window_ids: Sequence[str],
    max_images: int,
    compact_decision_only: bool = False,
    raise_on_protocol_error: bool = True,
) -> Dict[str, Any]:
    normalized_window_ids = list(_dedupe_window_ids(window_ids))
    if not normalized_window_ids:
        return _unavailable_branch_payload(window_ids=normalized_window_ids, reason="no_branch_windows")
    messages = _build_counterfactual_messages(
        item,
        rollout=rollout,
        branch_name=branch_name,
        window_ids=normalized_window_ids,
        target=target,
        max_images=int(max_images),
        compact_decision_only=compact_decision_only,
    )
    replay_results = _run_counterfactual_branch_replay_batch(
        policy,
        [
            {
                "messages": messages,
                "target": dict(target or {}),
                "compact_decision_only": bool(compact_decision_only),
                "branch_name": str(branch_name),
                "rollout": dict(rollout or {}),
                "item": dict(item or {}),
            }
        ],
        raise_on_protocol_error=raise_on_protocol_error,
    )
    replay = dict(replay_results[0]) if replay_results else _unavailable_branch_payload(
        window_ids=normalized_window_ids,
        reason="counterfactual_replay_missing",
    )
    field_support = _field_support(replay.get("semantic_answer"), reference_payload)
    branch_verification = _branch_finalize_readiness(
        branch_window_ids=normalized_window_ids,
        rollout=rollout,
        field_support=field_support,
        stage_requirements=stage_requirements,
        target=target,
    )
    return {
        "available": bool(replay.get("available")),
        "window_ids": normalized_window_ids,
        "response_text": replay.get("response_text"),
        "semantic_answer": replay.get("semantic_answer"),
        "semantic_answer_text": replay.get("semantic_answer_text"),
        "final_answer": replay.get("final_answer"),
        "field_support": field_support,
        "branch_verification": branch_verification,
        "branch_pass": False,
        "unavailable_reason": str(replay.get("unavailable_reason") or ""),
        "parse_mode": str(replay.get("parse_mode") or ""),
    }


def _build_branch_evaluation_request(
    *,
    item: Dict[str, Any],
    rollout: Dict[str, Any],
    reference_payload: Dict[str, Any],
    target: Dict[str, Any],
    stage_requirements: Dict[str, List[str]],
    branch_name: str,
    window_ids: Sequence[str],
    max_images: int,
    compact_decision_only: bool = False,
    raise_on_protocol_error: bool = True,
) -> Dict[str, Any]:
    normalized_window_ids = list(_dedupe_window_ids(window_ids))
    return {
        "item": item,
        "rollout": rollout,
        "reference_payload": reference_payload,
        "target": target,
        "stage_requirements": stage_requirements,
        "branch_name": str(branch_name),
        "window_ids": normalized_window_ids,
        "compact_decision_only": bool(compact_decision_only),
        "raise_on_protocol_error": bool(raise_on_protocol_error),
        "messages": _build_counterfactual_messages(
            item,
            rollout=rollout,
            branch_name=branch_name,
            window_ids=normalized_window_ids,
            target=target,
            max_images=int(max_images),
            compact_decision_only=compact_decision_only,
        ),
    }


def _finalize_branch_evaluation_request(
    request: Dict[str, Any],
    replay: Dict[str, Any],
) -> Dict[str, Any]:
    normalized_window_ids = list(_dedupe_window_ids(request.get("window_ids") or []))
    reference_payload = dict(request.get("reference_payload") or {})
    target = dict(request.get("target") or {})
    stage_requirements = dict(request.get("stage_requirements") or {})
    rollout = dict(request.get("rollout") or {})
    field_support = _field_support(replay.get("semantic_answer"), reference_payload)
    branch_verification = _branch_finalize_readiness(
        branch_window_ids=normalized_window_ids,
        rollout=rollout,
        field_support=field_support,
        stage_requirements=stage_requirements,
        target=target,
    )
    return {
        "available": bool(replay.get("available")),
        "window_ids": normalized_window_ids,
        "response_text": replay.get("response_text"),
        "semantic_answer": replay.get("semantic_answer"),
        "semantic_answer_text": replay.get("semantic_answer_text"),
        "final_answer": replay.get("final_answer"),
        "field_support": field_support,
        "branch_verification": branch_verification,
        "branch_pass": False,
        "unavailable_reason": str(replay.get("unavailable_reason") or ""),
        "parse_mode": str(replay.get("parse_mode") or ""),
    }


def _evaluate_branch_requests_batch(
    policy: Any,
    requests: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    request_list = list(requests or [])
    if not request_list:
        return []
    strict_mode = all(bool(dict(request).get("raise_on_protocol_error", True)) for request in request_list)
    replays = _run_counterfactual_branch_replay_batch(
        policy,
        request_list,
        raise_on_protocol_error=bool(strict_mode),
    )
    return [
        _finalize_branch_evaluation_request(request, replay)
        for request, replay in zip(request_list, replays)
    ]


def _select_replay_guided_minimal_subset(
    policy: Any,
    *,
    item: Dict[str, Any],
    rollout: Dict[str, Any],
    reference_payload: Dict[str, Any],
    target: Dict[str, Any],
    stage_requirements: Dict[str, List[str]],
    selected_window_ids: Sequence[str],
    max_images: int,
    compact_decision_only: bool = False,
) -> Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]]:
    current_ids = list(_dedupe_window_ids(selected_window_ids))
    trace: List[Dict[str, Any]] = []
    if not current_ids:
        unavailable = _unavailable_branch_payload(window_ids=current_ids, reason="no_selected_windows")
        return current_ids, unavailable, trace

    current_branch = _evaluate_branch_window_ids(
        policy,
        item=item,
        rollout=rollout,
        reference_payload=reference_payload,
        target=target,
        stage_requirements=stage_requirements,
        branch_name="minimal_subset",
        window_ids=current_ids,
        max_images=max_images,
        compact_decision_only=compact_decision_only,
    )
    improved = True
    while improved and len(current_ids) > 1:
        improved = False
        best_candidate_ids: Optional[List[str]] = None
        best_candidate_branch: Optional[Dict[str, Any]] = None
        best_objective = -1.0
        for removable_window_id in list(current_ids):
            candidate_ids = [window_id for window_id in current_ids if window_id != removable_window_id]
            candidate_branch = _evaluate_branch_window_ids(
                policy,
                item=item,
                rollout=rollout,
                reference_payload=reference_payload,
                target=target,
                stage_requirements=stage_requirements,
                branch_name="minimal_subset",
                window_ids=candidate_ids,
                max_images=max_images,
                compact_decision_only=compact_decision_only,
            )
            decision_supported = _branch_supports_decision(candidate_branch, target=target)
            objective = (
                _safe_float(((candidate_branch.get("branch_verification") or {}).get("finalize_readiness_score")), 0.0)
                + sum(
                    _safe_float(((candidate_branch.get("field_support") or {}).get(field_name) or {}).get("score"), 0.0)
                    for field_name in CORE_DECISION_FIELDS
                )
            )
            trace.append(
                {
                    "removed_window_id": removable_window_id,
                    "candidate_window_ids": candidate_ids,
                    "decision_supported": bool(decision_supported),
                    "objective": round(float(objective), 6),
                }
            )
            if not decision_supported:
                continue
            if objective > best_objective:
                best_objective = float(objective)
                best_candidate_ids = candidate_ids
                best_candidate_branch = candidate_branch
        if best_candidate_ids is not None and best_candidate_branch is not None:
            current_ids = list(best_candidate_ids)
            current_branch = best_candidate_branch
            improved = True
    current_branch["window_ids"] = list(current_ids)
    return current_ids, current_branch, trace


def run_counterfactual_verification_batch(
    policy: Any,
    *,
    batch_inputs: Sequence[Dict[str, Any]],
    max_images: int = 12,
    branch_profile: str = "full",
) -> List[Dict[str, Any]]:
    batch_input_list = list(batch_inputs or [])
    if not batch_input_list:
        return []
    normalized_branch_profile = _normalize_counterfactual_branch_profile(branch_profile)

    outputs: List[Optional[Dict[str, Any]]] = [None for _ in batch_input_list]
    entries: List[Dict[str, Any]] = []
    for batch_input_index, batch_input in enumerate(batch_input_list):
        item = dict(batch_input.get("item") or {})
        rollout = dict(batch_input.get("rollout") or {})
        reference_record = dict(batch_input.get("reference_record") or item)
        reference_payload = _build_reference_payload(reference_record)
        target = dict(reference_record.get("structured_target") or {})
        if str(target.get("existence") or "").strip().lower() != "anomaly":
            outputs[int(batch_input_index)] = (
                _build_normal_skip_profile(
                    item=item,
                    rollout=rollout,
                )
            )
            continue
        evidence_moments = ((reference_record.get("evidence") or {}).get("evidence_moments") or [])
        stage_requirements = derive_counterfactual_stage_requirements(target, evidence_moments=evidence_moments)
        active_branch_order = _resolve_counterfactual_branch_order(branch_profile, stage_requirements=stage_requirements)
        normalized_branch_profile = _normalize_counterfactual_branch_profile(branch_profile)
        compact_decision_only = False
        stage_queries = _build_stage_query_map(item=item, reference_record=reference_record)
        stage_by_moment_id = _stage_lookup_from_target(target, reference_record=reference_record)

        selection_resolution = resolve_selected_window_ids_for_fecv(rollout)
        selected_window_ids = list(selection_resolution.get("selected_window_ids") or [])
        selection_resolution_source = str(selection_resolution.get("selection_resolution_source") or "")
        recovered_from_trace = bool(selection_resolution.get("recovered_from_trace", False))
        selected_records = _selected_window_records(rollout, selected_window_ids)
        selected_by_stage = _stage_window_ids(selected_records, stage_by_moment_id=stage_by_moment_id)
        if not selected_window_ids:
            logger.info(
                "fecv selected-window debug: video_id=%s generation_id=%s branch_profile=%s "
                "selected_window_count=0 selection_resolution_source=%s recovered_from_trace=%s "
                "selected_by_stage=%s turn_count=%s decision_turn_indices=%s",
                str(rollout.get("video_id") or item.get("video_id") or ""),
                str(rollout.get("generation_id") or ""),
                normalized_branch_profile,
                selection_resolution_source,
                recovered_from_trace,
                selected_by_stage,
                len(rollout.get("turns") or []),
                list(rollout.get("decision_turn_indices") or []),
            )
        elif len(selected_records) != len(selected_window_ids):
            logger.info(
                "fecv selected-window debug: video_id=%s generation_id=%s branch_profile=%s "
                "selected_window_ids=%s selected_window_count=%s selected_record_count=%s "
                "selection_resolution_source=%s recovered_from_trace=%s selected_by_stage=%s",
                str(rollout.get("video_id") or item.get("video_id") or ""),
                str(rollout.get("generation_id") or ""),
                normalized_branch_profile,
                list(selected_window_ids),
                len(selected_window_ids),
                len(selected_records),
                selection_resolution_source,
                recovered_from_trace,
                selected_by_stage,
            )
        drop_precursor_ids, drop_precursor_available = _branch_stage_drop_window_ids(
            selected_window_ids,
            selected_records,
            stage="precursor",
            stage_by_moment_id=stage_by_moment_id,
        )
        drop_trigger_ids, drop_trigger_available = _branch_stage_drop_window_ids(
            selected_window_ids,
            selected_records,
            stage="trigger",
            stage_by_moment_id=stage_by_moment_id,
        )
        drop_confirmation_ids, drop_confirmation_available = _branch_stage_drop_window_ids(
            selected_window_ids,
            selected_records,
            stage="confirmation",
            stage_by_moment_id=stage_by_moment_id,
        )
        hard_negative_ids, hard_negative_available, hard_negative_reason = _build_hard_negative_swap_window_ids(
            rollout,
            selected_window_ids=selected_window_ids,
            selected_records=selected_records,
            stage_queries=stage_queries,
            stage_by_moment_id=stage_by_moment_id,
        )
        entries.append(
            {
                "item": item,
                "rollout": rollout,
                "reference_record": reference_record,
                "reference_payload": reference_payload,
                "target": target,
                "stage_requirements": stage_requirements,
                "active_branch_order": active_branch_order,
                "normalized_branch_profile": normalized_branch_profile,
                "stage_queries": stage_queries,
                "selected_window_ids": list(selected_window_ids),
                "selection_resolution_source": selection_resolution_source,
                "recovered_from_trace": recovered_from_trace,
                "selected_records": selected_records,
                "selected_by_stage": copy.deepcopy(selected_by_stage),
                "branch_specs": {
                    "drop_precursor": {
                        "window_ids": list(drop_precursor_ids),
                        "available": bool(drop_precursor_available),
                        "reason": "no_precursor_window",
                    },
                    "drop_trigger": {
                        "window_ids": list(drop_trigger_ids),
                        "available": bool(drop_trigger_available),
                        "reason": "no_trigger_window",
                    },
                    "drop_confirmation": {
                        "window_ids": list(drop_confirmation_ids),
                        "available": bool(drop_confirmation_available),
                        "reason": "no_confirmation_window",
                    },
                    "hard_negative_swap": {
                        "window_ids": list(hard_negative_ids),
                        "available": bool(hard_negative_available),
                        "reason": hard_negative_reason or "negative_swap_unavailable",
                    },
                },
                "branches": {},
                "minimal_subset_trace": [],
                "minimal_subset_ids": list(selected_window_ids),
                "hard_negative_ids": list(hard_negative_ids),
                "batch_input_index": int(batch_input_index),
            }
        )

    if not entries:
        return [dict(entry or {}) for entry in outputs if entry is not None]
    if policy is None:
        raise ValueError(
            "run_counterfactual_verification_batch requires a non-null policy unless branch_profile='online_core' or the target sample is normal."
        )

    full_requests: List[Dict[str, Any]] = []
    full_request_indices: List[int] = []
    for entry_index, entry in enumerate(entries):
        if entry["selected_window_ids"]:
            full_requests.append(
                _build_branch_evaluation_request(
                    item=entry["item"],
                    rollout=entry["rollout"],
                    reference_payload=entry["reference_payload"],
                    target=entry["target"],
                    stage_requirements=entry["stage_requirements"],
                    branch_name="full_selected",
                    window_ids=entry["selected_window_ids"],
                    max_images=int(max_images),
                    compact_decision_only=_branch_uses_compact_decision_only(
                        normalized_branch_profile=str(entry["normalized_branch_profile"] or ""),
                        branch_name="full_selected",
                    ),
                    raise_on_protocol_error=True,
                )
            )
            full_request_indices.append(int(entry_index))
        else:
            entry["branches"]["full_selected"] = _unavailable_branch_payload(
                window_ids=[],
                reason="contract_violation_empty_selection",
            )
    for entry_index, branch in zip(full_request_indices, _evaluate_branch_requests_batch(policy, full_requests)):
        entries[entry_index]["branches"]["full_selected"] = branch
        if not bool((branch or {}).get("available")):
            entry = entries[entry_index]
            logger.info(
                "fecv full-selected branch debug: video_id=%s generation_id=%s branch_profile=%s "
                "selected_window_count=%s selected_window_ids=%s unavailable_reason=%s "
                "parse_mode=%s selection_resolution_source=%s recovered_from_trace=%s "
                "full_selected_window_ids=%s failure_preview=%s",
                str((entry.get("rollout") or {}).get("video_id") or (entry.get("item") or {}).get("video_id") or ""),
                str((entry.get("rollout") or {}).get("generation_id") or ""),
                str(entry.get("normalized_branch_profile") or ""),
                len(entry.get("selected_window_ids") or []),
                list(entry.get("selected_window_ids") or []),
                str((branch or {}).get("unavailable_reason") or ""),
                str((branch or {}).get("parse_mode") or ""),
                str(entry.get("selection_resolution_source") or ""),
                bool(entry.get("recovered_from_trace", False)),
                list((branch or {}).get("window_ids") or []),
                _counterfactual_failure_preview(str((branch or {}).get("response_text") or "")),
            )

    if any("minimal_subset" in list(entry.get("active_branch_order") or []) for entry in entries):
        minimal_requests: List[Dict[str, Any]] = []
        minimal_request_indices: List[int] = []
        for entry_index, entry in enumerate(entries):
            if "minimal_subset" not in list(entry.get("active_branch_order") or []):
                continue
            current_ids = list(entry["selected_window_ids"])
            if not current_ids:
                entry["minimal_subset_ids"] = []
                entry["branches"]["minimal_subset"] = _unavailable_branch_payload(
                    window_ids=[],
                    reason="no_selected_windows",
                )
                continue
            minimal_requests.append(
                _build_branch_evaluation_request(
                    item=entry["item"],
                    rollout=entry["rollout"],
                    reference_payload=entry["reference_payload"],
                    target=entry["target"],
                    stage_requirements=entry["stage_requirements"],
                    branch_name="minimal_subset",
                    window_ids=current_ids,
                    max_images=int(max_images),
                    compact_decision_only=_branch_uses_compact_decision_only(
                        normalized_branch_profile=str(entry["normalized_branch_profile"] or ""),
                        branch_name="minimal_subset",
                    ),
                    raise_on_protocol_error=False,
                )
            )
            minimal_request_indices.append(int(entry_index))
        for entry_index, branch in zip(minimal_request_indices, _evaluate_branch_requests_batch(policy, minimal_requests)):
            entries[entry_index]["branches"]["minimal_subset"] = branch

        improved = True
        while improved:
            candidate_requests: List[Dict[str, Any]] = []
            candidate_meta: List[Tuple[int, str, List[str]]] = []
            for entry_index, entry in enumerate(entries):
                if "minimal_subset" not in list(entry.get("active_branch_order") or []):
                    continue
                current_ids = list((entry["branches"].get("minimal_subset") or {}).get("window_ids") or [])
                if len(current_ids) <= 1:
                    continue
                for removable_window_id in list(current_ids):
                    candidate_ids = [window_id for window_id in current_ids if window_id != removable_window_id]
                    candidate_requests.append(
                        _build_branch_evaluation_request(
                            item=entry["item"],
                            rollout=entry["rollout"],
                            reference_payload=entry["reference_payload"],
                            target=entry["target"],
                            stage_requirements=entry["stage_requirements"],
                            branch_name="minimal_subset",
                            window_ids=candidate_ids,
                            max_images=int(max_images),
                            compact_decision_only=_branch_uses_compact_decision_only(
                                normalized_branch_profile=str(entry["normalized_branch_profile"] or ""),
                                branch_name="minimal_subset",
                            ),
                            raise_on_protocol_error=False,
                        )
                    )
                    candidate_meta.append((int(entry_index), str(removable_window_id), list(candidate_ids)))
            candidate_branches = _evaluate_branch_requests_batch(policy, candidate_requests)
            best_updates: Dict[int, Tuple[List[str], Dict[str, Any], float]] = {}
            improved = False
            for (entry_index, removable_window_id, candidate_ids), candidate_branch in zip(candidate_meta, candidate_branches):
                decision_supported = _branch_supports_decision(candidate_branch, target=entries[entry_index]["target"])
                objective = (
                    _safe_float(((candidate_branch.get("branch_verification") or {}).get("finalize_readiness_score")), 0.0)
                    + sum(
                        _safe_float(((candidate_branch.get("field_support") or {}).get(field_name) or {}).get("score"), 0.0)
                        for field_name in CORE_DECISION_FIELDS
                    )
                )
                entries[entry_index]["minimal_subset_trace"].append(
                    {
                        "removed_window_id": removable_window_id,
                        "candidate_window_ids": list(candidate_ids),
                        "decision_supported": bool(decision_supported),
                        "objective": round(float(objective), 6),
                    }
                )
                if not decision_supported:
                    continue
                previous = best_updates.get(entry_index)
                if previous is None or float(objective) > float(previous[2]):
                    best_updates[entry_index] = (list(candidate_ids), candidate_branch, float(objective))
            for entry_index, (candidate_ids, candidate_branch, _objective) in best_updates.items():
                entries[entry_index]["branches"]["minimal_subset"] = candidate_branch
                entries[entry_index]["minimal_subset_ids"] = list(candidate_ids)
                improved = True

    branch_requests: List[Dict[str, Any]] = []
    branch_request_meta: List[Tuple[int, str]] = []
    for entry_index, entry in enumerate(entries):
        for branch_name in entry["active_branch_order"]:
            if branch_name in {"full_selected", "minimal_subset"}:
                continue
            spec = dict(entry["branch_specs"].get(branch_name) or {})
            if not bool(spec.get("available")):
                entry["branches"][branch_name] = _unavailable_branch_payload(
                    window_ids=spec.get("window_ids") or [],
                    reason=str(spec.get("reason") or ""),
                )
                continue
            branch_requests.append(
                _build_branch_evaluation_request(
                    item=entry["item"],
                    rollout=entry["rollout"],
                    reference_payload=entry["reference_payload"],
                    target=entry["target"],
                    stage_requirements=entry["stage_requirements"],
                    branch_name=branch_name,
                    window_ids=spec.get("window_ids") or [],
                    max_images=int(max_images),
                    compact_decision_only=_branch_uses_compact_decision_only(
                        normalized_branch_profile=str(entry["normalized_branch_profile"] or ""),
                        branch_name=str(branch_name),
                    ),
                    raise_on_protocol_error=False,
                )
            )
            branch_request_meta.append((int(entry_index), str(branch_name)))
    for (entry_index, branch_name), branch in zip(branch_request_meta, _evaluate_branch_requests_batch(policy, branch_requests)):
        entries[entry_index]["branches"][branch_name] = branch

    for entry in entries:
        branches = dict(entry["branches"] or {})
        active_branch_order = list(entry["active_branch_order"] or [])
        target = dict(entry["target"] or {})
        stage_requirements = dict(entry["stage_requirements"] or {})
        for branch_name in active_branch_order:
            if branch_name not in branches:
                continue
            branch_payload = dict(branches[branch_name] or {})
            branch_payload["branch_pass"] = bool(
                _branch_pass(
                    branch_name,
                    target=target,
                    field_support=dict(branch_payload.get("field_support") or {}),
                    branch_verification=dict(branch_payload.get("branch_verification") or {}),
                    stage_requirements=stage_requirements,
                )
            )
            branches[branch_name] = branch_payload
        branch_field_matrix = {
            branch_name: _branch_field_matrix_entry(branches[branch_name], target=target)
            for branch_name in active_branch_order
            if branch_name in branches
        }
        branch_delta_matrix = _build_branch_delta_matrix(branch_field_matrix)
        full_selected_branch = dict(branches.get("full_selected") or {})
        # Continuous delta scores for the redesigned R_fecv (used by reward.py)
        _hn_delta_fields = dict((branch_delta_matrix.get("hard_negative_swap") or {}).get("fields") or {})
        _dt_delta_fields = dict((branch_delta_matrix.get("drop_trigger") or {}).get("fields") or {})
        summary = {
            # Legacy boolean gates (kept for backward compat with metrics/evaluation)
            "decision_sufficiency": bool(_branch_supports_decision(full_selected_branch, target=target)),
            "minimal_subset_sufficiency": bool(
                _branch_supports_decision(dict(branches.get("minimal_subset") or {}), target=target)
            ),
            "negative_specificity_pass": bool(
                _safe_float(_hn_delta_fields.get("existence"), 0.0) >= 0.5
                or _safe_float(_hn_delta_fields.get("category"), 0.0) >= 0.5
            ),
            "stage_necessity": _derive_stage_necessity_from_deltas(
                branch_field_matrix=branch_field_matrix,
                branch_delta_matrix=branch_delta_matrix,
            ),
            # Continuous scores for gradient-friendly R_fecv
            "trigger_necessity_delta": max(
                _safe_float(_dt_delta_fields.get("existence"), 0.0),
                _safe_float(_dt_delta_fields.get("category"), 0.0),
            ),
            "negative_resistance_delta": max(
                _safe_float(_hn_delta_fields.get("existence"), 0.0),
                _safe_float(_hn_delta_fields.get("category"), 0.0),
            ),
        }
        profile = {
            "summary": summary,
            "branch_field_matrix": branch_field_matrix,
            "branch_delta_matrix": branch_delta_matrix,
            "stage_packages": {
                "selected_window_ids": list(entry["selected_window_ids"]),
                "selected_by_stage": copy.deepcopy(entry["selected_by_stage"]),
                "minimal_subset_window_ids": list((branches.get("minimal_subset") or {}).get("window_ids") or []),
                "hard_negative_window_ids": list(entry["hard_negative_ids"]),
            },
            "selection_metadata": {
                "normalized_branch_profile": entry["normalized_branch_profile"],
                "selected_window_count": len(entry["selected_window_ids"]),
                "selected_record_count": len(entry["selected_records"]),
                "selection_resolution_source": str(entry.get("selection_resolution_source") or ""),
                "recovered_from_trace": bool(entry.get("recovered_from_trace", False)),
                "selected_by_stage": copy.deepcopy(entry["selected_by_stage"]),
                "stage_requirements": copy.deepcopy(stage_requirements),
                "stage_queries": copy.deepcopy(entry["stage_queries"]),
                "minimal_subset_trace": list(entry["minimal_subset_trace"]),
                "full_selected_available": bool(full_selected_branch.get("available")),
                "full_selected_parse_mode": str(full_selected_branch.get("parse_mode") or ""),
                "full_selected_unavailable_reason": str(full_selected_branch.get("unavailable_reason") or ""),
                "full_selected_window_ids": list(full_selected_branch.get("window_ids") or []),
                "hard_negative_reason": str(
                    ((entry["branch_specs"].get("hard_negative_swap") or {}).get("reason")) or ""
                ),
            },
        }
        outputs[int(entry["batch_input_index"])] = {
            "counterfactual_branches": branches,
            "counterfactual_profile": profile,
            "counterfactual_profile_source": entry["normalized_branch_profile"],
            "counterfactual_branch_profile": entry["normalized_branch_profile"],
        }
    return [dict(entry or {}) for entry in outputs if entry is not None]


def run_counterfactual_verification(
    policy: Any,
    *,
    item: Dict[str, Any],
    rollout: Dict[str, Any],
    reference_record: Dict[str, Any],
    max_images: int = 12,
    branch_profile: str = "full",
) -> Dict[str, Any]:
    results = run_counterfactual_verification_batch(
        policy,
        batch_inputs=[
            {
                "item": item,
                "rollout": rollout,
                "reference_record": reference_record,
            }
        ],
        max_images=int(max_images),
        branch_profile=branch_profile,
    )
    return dict(results[0] if results else {})
