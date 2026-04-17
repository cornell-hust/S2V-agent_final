from __future__ import annotations

import copy
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from saver_v3.core.adapter import TimeSearchRolloutAdapter
from saver_v3.core.categories import canonicalize_saver_category
from saver_v3.core.proposal import compose_scene_anchored_query, normalize_description_query_phrases
from saver_v3.core.protocol_guidance import summarize_evidence_ledger
from saver_v3.core.semantic_answer import (
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
    "trigger": "trigger",
    "peak_action": "trigger",
    "confirmation": "confirmation",
    "aftermath": "confirmation",
}
CORE_DECISION_FIELDS = ("existence", "category", "temporal", "counterfactual_type")
STAGE_TEXT_THRESHOLD = 0.3
BRANCH_ORDER = (
    "full_selected",
    "minimal_subset",
    "drop_precursor",
    "drop_trigger",
    "drop_confirmation",
    "hard_negative_swap",
)
COUNTERFACTUAL_BRANCH_PROFILES = ("full", "offline_full", "online_core", "structured_oracle_v1")
FINALIZE_READINESS_THRESHOLD = 0.75


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
    if normalized == "structured_oracle_v1":
        return "structured_oracle_v1"
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
    state = dict(rollout.get("state") or {})
    active_window_ids = _dedupe_window_ids(state.get("active_evidence_window_ids") or [])
    if active_window_ids:
        return active_window_ids

    for turn in reversed(list(rollout.get("turns") or [])):
        if str(turn.get("tool_name") or "") != "verify_hypothesis":
            continue
        verified = _dedupe_window_ids(turn.get("verifier_verified_window_ids") or [])
        if verified:
            return verified
        selected = _dedupe_window_ids(turn.get("self_verification_selected_window_ids") or [])
        if selected:
            return selected
    return []


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
            stage = ROLE_TO_STAGE.get(str(moment.get("role") or "").strip().lower())
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
    existence = str(payload.get("existence") or "").strip().lower()
    normalized["existence"] = "anomaly" if existence == "anomaly" else "normal"
    normalized["category"] = canonicalize_saver_category(
        payload.get("category"),
        existence=normalized["existence"],
    ) or str(payload.get("category") or "").strip().lower()
    interval = payload.get("anomaly_interval_sec")
    normalized["anomaly_interval_sec"] = list(interval) if isinstance(interval, (list, tuple)) and len(interval) == 2 else None
    normalized["counterfactual_type"] = str(payload.get("counterfactual_type") or "").strip().lower() or "none"
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
        stage = ROLE_TO_STAGE.get(str(moment.get("role") or "").strip().lower(), "")
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
    target_existence = str(target.get("existence") or "").strip().lower()
    scores["existence"] = 1.0 if prediction.get("existence") == ("anomaly" if target_existence == "anomaly" else "normal") else 0.0
    target_category = canonicalize_saver_category(target.get("category"), existence=target_existence) or str(target.get("category") or "").strip().lower()
    prediction_category = canonicalize_saver_category(prediction.get("category"), existence=prediction.get("existence")) or str(prediction.get("category") or "").strip().lower()
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


def _structured_oracle_profile_entry(
    *,
    rollout: Dict[str, Any],
    reference_record: Dict[str, Any],
    include_counterfactual_type: bool = True,
) -> Dict[str, Any]:
    target = dict(reference_record.get("structured_target") or {})
    evidence_moments = ((reference_record.get("evidence") or {}).get("evidence_moments") or [])
    stage_requirements = derive_counterfactual_stage_requirements(target, evidence_moments=evidence_moments)
    selected_window_ids = infer_counterfactual_window_ids(rollout)
    selected_records = _selected_window_records(rollout, selected_window_ids)
    stage_moments = _reference_stage_moments(reference_record)
    decision_scores = _structured_decision_scores(rollout=rollout, target=target)
    decision_score = _average(list(decision_scores.values()))

    stage_support_scores = {
        stage: _records_support_against_moments(selected_records, stage_moments.get(stage) or [])
        for stage in STAGE_ORDER
    }
    target_existence = str(target.get("existence") or "").strip().lower()
    trigger_support_full = float(stage_support_scores.get("trigger", 0.0))
    if target_existence == "anomaly":
        selected_support = round(float(decision_score) * float(trigger_support_full), 6)
    else:
        selected_support = float(decision_score)

    required_stages = _required_stage_list(target, stage_requirements)
    if target_existence != "anomaly":
        required_stage_coverage = 1.0
    else:
        required_stage_coverage = _average([stage_support_scores.get(stage, 0.0) for stage in required_stages])

    trigger_aligned_window_ids: List[str] = []
    for record in selected_records:
        best_stage = ""
        best_score = 0.0
        for stage, moments in stage_moments.items():
            score = _records_support_against_moments([record], moments)
            if score > best_score:
                best_score = score
                best_stage = stage
        if best_stage == "trigger" and best_score >= 0.3:
            window_id = str(record.get("window_id") or "").strip()
            if window_id:
                trigger_aligned_window_ids.append(window_id)

    remaining_records = [
        record for record in selected_records
        if str(record.get("window_id") or "").strip() not in set(trigger_aligned_window_ids)
    ]
    trigger_support_drop = _records_support_against_moments(remaining_records, stage_moments.get("trigger") or [])
    if target_existence == "anomaly":
        drop_trigger_necessity = round(max(0.0, float(trigger_support_full) - float(trigger_support_drop)), 6)
    else:
        drop_trigger_necessity = 0.0

    full_selected_fields: Dict[str, Dict[str, Any]] = {
        "existence": {"score": float(decision_scores.get("existence", 0.0)), "supported": bool(decision_scores.get("existence", 0.0) >= 1.0)},
        "category": {"score": float(decision_scores.get("category", 0.0)), "supported": bool(decision_scores.get("category", 0.0) >= 1.0)},
        "temporal": {"score": float(decision_scores.get("temporal", 0.0)), "supported": bool(decision_scores.get("temporal", 0.0) >= 0.5)},
        "trigger": {"score": float(stage_support_scores.get("trigger", 0.0)), "supported": bool(stage_support_scores.get("trigger", 0.0) >= 0.3)},
        "precursor": {"score": float(stage_support_scores.get("precursor", 0.0)), "supported": bool(stage_support_scores.get("precursor", 0.0) >= 0.3)},
        "confirmation": {"score": float(stage_support_scores.get("confirmation", 0.0)), "supported": bool(stage_support_scores.get("confirmation", 0.0) >= 0.3)},
        "finalize_readiness": {"score": float(required_stage_coverage), "supported": bool(required_stage_coverage >= FINALIZE_READINESS_THRESHOLD)},
    }
    drop_trigger_fields: Dict[str, Dict[str, Any]] = {
        "existence": {"score": max(0.0, float(full_selected_fields["existence"]["score"]) - float(drop_trigger_necessity)), "supported": False},
        "category": {"score": max(0.0, float(full_selected_fields["category"]["score"]) - float(drop_trigger_necessity)), "supported": False},
        "temporal": {"score": max(0.0, float(full_selected_fields["temporal"]["score"]) - float(drop_trigger_necessity)), "supported": False},
        "trigger": {"score": float(trigger_support_drop), "supported": bool(trigger_support_drop >= 0.3)},
        "precursor": {"score": float(stage_support_scores.get("precursor", 0.0)), "supported": bool(stage_support_scores.get("precursor", 0.0) >= 0.3)},
        "confirmation": {"score": float(stage_support_scores.get("confirmation", 0.0)), "supported": bool(stage_support_scores.get("confirmation", 0.0) >= 0.3)},
        "finalize_readiness": {"score": float(required_stage_coverage), "supported": bool(required_stage_coverage >= FINALIZE_READINESS_THRESHOLD)},
    }
    branch_field_matrix = {
        "full_selected": {
            "available": bool(selected_window_ids),
            "window_ids": list(selected_window_ids),
            "fields": full_selected_fields,
            "core_decision_supported": bool(selected_support >= 0.5),
            "supported_stages": [stage for stage, score in stage_support_scores.items() if score >= 0.3],
            "missing_required_stages": [stage for stage in required_stages if stage_support_scores.get(stage, 0.0) < 0.3],
        },
        "drop_trigger": {
            "available": bool(trigger_aligned_window_ids),
            "window_ids": [
                str(record.get("window_id") or "").strip()
                for record in remaining_records
                if str(record.get("window_id") or "").strip()
            ],
            "fields": drop_trigger_fields,
            "core_decision_supported": False,
            "supported_stages": [stage for stage, score in stage_support_scores.items() if stage != "trigger" and score >= 0.3],
            "missing_required_stages": [stage for stage in required_stages if (stage == "trigger" or stage_support_scores.get(stage, 0.0) < 0.3)],
        },
    }
    branch_delta_matrix = _build_branch_delta_matrix(branch_field_matrix)
    summary = {
        "decision_sufficiency": bool(selected_support >= 0.5),
        "minimal_subset_sufficiency": False,
        "negative_specificity_pass": False,
        "stage_necessity": {
            "trigger": (
                "decision_critical"
                if drop_trigger_necessity >= 0.3
                else ("non_critical" if stage_moments.get("trigger") else "not_observed")
            )
        },
        "oracle_selected_support_score": float(selected_support),
        "oracle_required_stage_coverage_score": float(required_stage_coverage),
        "oracle_drop_trigger_necessity_score": float(drop_trigger_necessity),
    }
    if include_counterfactual_type:
        summary["counterfactual_type_supported"] = True
    return {
        "counterfactual_branches": {},
        "counterfactual_profile": {
            "summary": summary,
            "branch_field_matrix": branch_field_matrix,
            "branch_delta_matrix": branch_delta_matrix,
            "stage_packages": {
                "selected_window_ids": list(selected_window_ids),
                "selected_by_stage": _stage_window_ids(selected_records),
                "trigger_aligned_window_ids": list(trigger_aligned_window_ids),
                "required_stages": list(required_stages),
            },
            "selection_metadata": {
                "normalized_branch_profile": "structured_oracle_v1",
                "stage_requirements": copy.deepcopy(stage_requirements),
            },
            "counterfactual_profile_source": "structured_oracle_v1",
            "counterfactual_branch_profile": "structured_oracle_v1",
        },
        "counterfactual_profile_source": "structured_oracle_v1",
        "counterfactual_branch_profile": "structured_oracle_v1",
    }


def _run_structured_oracle_verification_batch(
    batch_inputs: Sequence[Dict[str, Any]],
    *,
    include_counterfactual_type: bool = True,
) -> List[Dict[str, Any]]:
    return [
        _structured_oracle_profile_entry(
            rollout=dict((batch_input.get("rollout") or {})),
            reference_record=dict((batch_input.get("reference_record") or batch_input.get("item") or {})),
            include_counterfactual_type=include_counterfactual_type,
        )
        for batch_input in list(batch_inputs or [])
    ]


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


def _stage_for_entry(entry: Dict[str, Any]) -> str:
    return ROLE_TO_STAGE.get(str(entry.get("role") or "").strip().lower(), "")


def _stage_window_ids(records: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    grouped = {stage: [] for stage in STAGE_ORDER}
    for entry in list(records or []):
        stage = _stage_for_entry(entry)
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
        stage = ROLE_TO_STAGE.get(str(moment.get("role") or "").strip().lower())
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
) -> List[str]:
    selected_records = list(selected_records or [])
    if not selected_records:
        return []
    by_stage = _stage_window_ids(selected_records)
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
) -> Tuple[List[str], bool]:
    stage_window_ids = set(_stage_window_ids(selected_records).get(stage) or [])
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
) -> Tuple[List[str], bool, str]:
    selected_window_ids = _dedupe_window_ids(selected_window_ids)
    if not selected_window_ids:
        return [], False, "no_selected_windows"
    stage_groups = _stage_window_ids(selected_records)
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
            "drop_trigger",
        ]
    return list(BRANCH_ORDER)


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


def _compare_counterfactual_type(prediction_payload: Dict[str, Any] | None, reference_payload: Dict[str, Any]) -> Tuple[float, bool]:
    pred_decision = (prediction_payload or {}).get("decision") or {}
    ref_decision = reference_payload.get("decision") or {}
    reference = str(ref_decision.get("counterfactual_type") or "").strip().lower()
    prediction = str(pred_decision.get("counterfactual_type") or "").strip().lower()
    if not reference or reference == "none":
        return 1.0, True
    supported = bool(prediction) and prediction == reference
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
        ("counterfactual_type", _compare_counterfactual_type),
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
    target_counterfactual = str(target.get("counterfactual_type") or "").strip().lower()
    if target_counterfactual and target_counterfactual != "none":
        if not bool(field_support.get("counterfactual_type", {}).get("supported")):
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
    supported_stages = [stage for stage, values in _stage_window_ids(records).items() if values]
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
        _safe_float(trigger_delta.get("counterfactual_type"), 0.0),
    )
    precursor_decision_drop = max(
        _safe_float(precursor_delta.get("existence"), 0.0),
        _safe_float(precursor_delta.get("category"), 0.0),
        _safe_float(precursor_delta.get("temporal"), 0.0),
        _safe_float(precursor_delta.get("counterfactual_type"), 0.0),
    )
    confirmation_decision_drop = max(
        _safe_float(confirmation_delta.get("existence"), 0.0),
        _safe_float(confirmation_delta.get("category"), 0.0),
        _safe_float(confirmation_delta.get("temporal"), 0.0),
        _safe_float(confirmation_delta.get("counterfactual_type"), 0.0),
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
        scaffold = json.dumps(
            {"decision": dict(target or {"existence": "normal", "category": "normal"})},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        output_instruction = (
            f"Return exactly one <answer></answer> JSON in this compact shape: {scaffold}\n"
            "Do not add summary, rationale, event_chain_summary, or qa_focus_answers."
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
    if not hasattr(policy, "generate_from_messages"):
        raise AttributeError("Counterfactual verification requires a policy with generate_from_messages(messages).")
    response_text = str(policy.generate_from_messages(messages) or "")
    return _parse_counterfactual_branch_replay_response(response_text)


def _parse_counterfactual_branch_replay_response(response_text: str) -> Dict[str, Any]:
    answer_text = TimeSearchRolloutAdapter.parse_answer_text(response_text)
    payload: Optional[Dict[str, Any]] = None
    if answer_text:
        try:
            parsed = json.loads(answer_text)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            payload = normalize_semantic_answer_payload(parsed)
    final_answer = extract_decision_from_semantic_answer(payload) if payload else None
    return {
        "response_text": response_text,
        "semantic_answer": payload,
        "semantic_answer_text": semantic_answer_to_text(payload),
        "final_answer": final_answer,
        "available": bool(payload is not None),
    }


def _run_counterfactual_branch_replay_batch(
    policy: Any,
    messages_batch: Sequence[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    message_list = list(messages_batch or [])
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
    return [
        _parse_counterfactual_branch_replay_response(str(response_text or ""))
        for response_text in response_texts
    ]


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
    replay = _run_counterfactual_branch_replay(policy, messages)
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
        "unavailable_reason": "",
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
        "unavailable_reason": "",
    }


def _evaluate_branch_requests_batch(
    policy: Any,
    requests: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    request_list = list(requests or [])
    if not request_list:
        return []
    replays = _run_counterfactual_branch_replay_batch(
        policy,
        [dict(request).get("messages") or [] for request in request_list],
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
    include_counterfactual_type: bool = True,
) -> List[Dict[str, Any]]:
    batch_input_list = list(batch_inputs or [])
    if not batch_input_list:
        return []
    normalized_branch_profile = _normalize_counterfactual_branch_profile(branch_profile)
    if normalized_branch_profile == "structured_oracle_v1":
        return _run_structured_oracle_verification_batch(
            batch_input_list,
            include_counterfactual_type=include_counterfactual_type,
        )
    if policy is None:
        raise ValueError(
            "run_counterfactual_verification_batch requires a non-null policy unless branch_profile='structured_oracle_v1'."
        )

    entries: List[Dict[str, Any]] = []
    for batch_input in batch_input_list:
        item = dict(batch_input.get("item") or {})
        rollout = dict(batch_input.get("rollout") or {})
        reference_record = dict(batch_input.get("reference_record") or item)
        reference_payload = _build_reference_payload(reference_record)
        target = dict(reference_record.get("structured_target") or {})
        evidence_moments = ((reference_record.get("evidence") or {}).get("evidence_moments") or [])
        stage_requirements = derive_counterfactual_stage_requirements(target, evidence_moments=evidence_moments)
        active_branch_order = _resolve_counterfactual_branch_order(branch_profile, stage_requirements=stage_requirements)
        normalized_branch_profile = _normalize_counterfactual_branch_profile(branch_profile)
        compact_decision_only = normalized_branch_profile == "online_core"
        stage_queries = _build_stage_query_map(item=item, reference_record=reference_record)

        selected_window_ids = infer_counterfactual_window_ids(rollout)
        selected_records = _selected_window_records(rollout, selected_window_ids)
        drop_precursor_ids, drop_precursor_available = _branch_stage_drop_window_ids(
            selected_window_ids,
            selected_records,
            stage="precursor",
        )
        drop_trigger_ids, drop_trigger_available = _branch_stage_drop_window_ids(
            selected_window_ids,
            selected_records,
            stage="trigger",
        )
        drop_confirmation_ids, drop_confirmation_available = _branch_stage_drop_window_ids(
            selected_window_ids,
            selected_records,
            stage="confirmation",
        )
        hard_negative_ids, hard_negative_available, hard_negative_reason = _build_hard_negative_swap_window_ids(
            rollout,
            selected_window_ids=selected_window_ids,
            selected_records=selected_records,
            stage_queries=stage_queries,
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
                "compact_decision_only": compact_decision_only,
                "stage_queries": stage_queries,
                "selected_window_ids": list(selected_window_ids),
                "selected_records": selected_records,
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
            }
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
                    compact_decision_only=bool(entry["compact_decision_only"]),
                )
            )
            full_request_indices.append(int(entry_index))
        else:
            entry["branches"]["full_selected"] = _unavailable_branch_payload(
                window_ids=[],
                reason="no_selected_windows",
            )
    for entry_index, branch in zip(full_request_indices, _evaluate_branch_requests_batch(policy, full_requests)):
        entries[entry_index]["branches"]["full_selected"] = branch

    minimal_requests: List[Dict[str, Any]] = []
    minimal_request_indices: List[int] = []
    for entry_index, entry in enumerate(entries):
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
                compact_decision_only=bool(entry["compact_decision_only"]),
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
                        compact_decision_only=bool(entry["compact_decision_only"]),
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
                    compact_decision_only=bool(entry["compact_decision_only"]),
                )
            )
            branch_request_meta.append((int(entry_index), str(branch_name)))
    for (entry_index, branch_name), branch in zip(branch_request_meta, _evaluate_branch_requests_batch(policy, branch_requests)):
        entries[entry_index]["branches"][branch_name] = branch

    outputs: List[Dict[str, Any]] = []
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
        full_support = dict(full_selected_branch.get("field_support") or {})
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
            "counterfactual_type_supported": bool(full_support.get("counterfactual_type", {}).get("supported", False)),
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
                "selected_by_stage": _stage_window_ids(entry["selected_records"]),
                "minimal_subset_window_ids": list((branches.get("minimal_subset") or {}).get("window_ids") or []),
                "hard_negative_window_ids": list(entry["hard_negative_ids"]),
            },
            "selection_metadata": {
                "normalized_branch_profile": entry["normalized_branch_profile"],
                "stage_requirements": copy.deepcopy(stage_requirements),
                "stage_queries": copy.deepcopy(entry["stage_queries"]),
                "minimal_subset_trace": list(entry["minimal_subset_trace"]),
                "hard_negative_reason": str(
                    ((entry["branch_specs"].get("hard_negative_swap") or {}).get("reason")) or ""
                ),
            },
        }
        outputs.append(
            {
                "counterfactual_branches": branches,
                "counterfactual_profile": profile,
                "counterfactual_profile_source": entry["normalized_branch_profile"],
                "counterfactual_branch_profile": entry["normalized_branch_profile"],
            }
        )
    return outputs


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
