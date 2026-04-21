from __future__ import annotations

import copy
import logging
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from saver_v3.core.categories import canonicalize_saver_category, normalize_existence
from saver_v3.core.event_chain import (
    compute_event_chain_score,
    compute_query_alignment_score,
    extract_stage_annotation_from_record,
    extract_stage_annotation_from_turn,
    infer_required_stages_from_target,
    observed_stages_for_search_turn,
)
from saver_v3.core.llm_judge import OpenAICompatibleLlmJudge
from saver_v3.core.protocol_guidance import normalize_event_chain_stages
from saver_v3.core.semantic_answer import (
    SEMANTIC_EVENT_CHAIN_STAGES,
    build_public_semantic_replay_payload,
    normalize_semantic_answer_payload,
    normalize_text_match,
)


DEFAULT_RL_REWARD_VERSION = "timesearch_v3"
DEFAULT_COMPONENT_WEIGHTS = {
    "fecv_decision_sufficiency_reward": 1.0,
    "fecv_specificity_reward": 1.0,
    "protocol_finalize_reward": 1.0,
    "counterfactual_sufficiency_reward": 1.0,
    "stage_necessity_reward": 0.0,
    "query_alignment_reward": 0.0,
    "efficiency_reward": 0.4,
}
TIMESARCH_V1_COMPONENT_WEIGHTS = {
    "accuracy_reward": 1.0,
    "fecv_evidence_faithfulness_reward": 0.75,
    "protocol_finalize_reward": 0.25,
}

logger = logging.getLogger(__name__)
NORMAL_SKIP_PROFILE = "normal_skip_v1"
EASY_NORMAL_SAMPLE_LOSS_MULTIPLIER = 0.20
NORMAL_EVIDENCE_TOOL_NAMES = frozenset({
    "scan_timeline",
    "seek_evidence",
    "verify_hypothesis",
})
TIMESARCH_V2_COMPONENT_WEIGHTS = {
    "accuracy_reward": 1.0,
    "fecv_evidence_faithfulness_reward": 0.35,
    "protocol_finalize_reward": 0.1,
}
TIMESARCH_V3_COMPONENT_WEIGHTS = {
    "accuracy_reward": 1.0,
    "fecv_evidence_faithfulness_reward": 0.35,
    "protocol_finalize_reward": 0.05,
}
LEGACY_COMPONENT_ALIASES = {
    "evidence_support_reward": "fecv_decision_sufficiency_reward",
    "counterfactual_sufficiency_reward": "fecv_decision_sufficiency_reward",
    "stage_reward": "fecv_specificity_reward",
}
TIMESARCH_COMPONENT_ALIASES = {
    "evidence_support_reward": "fecv_evidence_faithfulness_reward",
    "counterfactual_sufficiency_reward": "fecv_evidence_faithfulness_reward",
    "fecv_decision_sufficiency_reward": "fecv_evidence_faithfulness_reward",
    "fecv_specificity_reward": "fecv_evidence_faithfulness_reward",
}

PRIMARY_STATUS_REWARD = {
    "complete": 1.0,
    "redundant": 0.35,
    "incomplete": -0.35,
    "misaligned": -1.0,
}

_FRAME_INTERVAL_RE = re.compile(
    r"frame(?:s)?\s*(\d+(?:\.\d+)?)\s*(?:to|and|-)\s*(?:frame(?:s)?\s*)?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_SECOND_INTERVAL_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\s*(?:to|and|-)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)",
    re.IGNORECASE,
)
_COUNTERFACTUAL_STAGE_TEXT_THRESHOLD = 0.3
_OPEN_ENDED_QUESTION_TYPES = (
    "trigger_evidence",
    "normal_reason",
    "summary",
    "rationale",
) + tuple(f"event_chain_summary.{stage}" for stage in SEMANTIC_EVENT_CHAIN_STAGES)

# V3 (2026-04-21): metric-aligned semantic-quality subset. Included per paper:
#   - trigger_evidence: what visible evidence makes anomaly actionable
#   - summary: overall case narrative
#   - event_chain_summary.{precursor,trigger,confirmation}: stage-wise reasoning
# Excluded: normal_reason, rationale (no primary metric in paper).
# Each qa_type is scored via OpenAICompatibleLlmJudge.score() — if the remote
# judge service is not configured, falls back to normalized-exact + token-F1.
# Sub-rewards contribute independently (one score each, weight 1/N) per Fix B.
_OPEN_ENDED_QUESTION_TYPES_V3: tuple[str, ...] = (
    "trigger_evidence",
    "summary",
) + tuple(f"event_chain_summary.{stage}" for stage in SEMANTIC_EVENT_CHAIN_STAGES)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _normalize_reward_version(value: Any) -> str:
    normalized = str(value or "legacy").strip().lower()
    if normalized in {"legacy", "timesearch_v1", "timesearch_v2", "timesearch_v3"}:
        return normalized
    raise ValueError(f"Unsupported reward version: {value!r}")


def _normalize_component_weights(
    weights: Optional[Dict[str, float]],
    *,
    reward_version: str = "legacy",
) -> Dict[str, float]:
    normalized_reward_version = _normalize_reward_version(reward_version)
    if normalized_reward_version in {"timesearch_v1", "timesearch_v2", "timesearch_v3"}:
        if normalized_reward_version == "timesearch_v3":
            merged = dict(TIMESARCH_V3_COMPONENT_WEIGHTS)
        elif normalized_reward_version == "timesearch_v2":
            merged = dict(TIMESARCH_V2_COMPONENT_WEIGHTS)
        else:
            merged = dict(TIMESARCH_V1_COMPONENT_WEIGHTS)
        aliases = TIMESARCH_COMPONENT_ALIASES
    else:
        merged = dict(DEFAULT_COMPONENT_WEIGHTS)
        aliases = LEGACY_COMPONENT_ALIASES
    for key, value in (weights or {}).items():
        text = aliases.get(str(key).strip(), str(key).strip())
        if text in merged:
            merged[text] = float(value)
    return merged


def resolve_reward_component_weights(
    *,
    reward_version: str = DEFAULT_RL_REWARD_VERSION,
    reward_config: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    resolved_reward_version = reward_version
    if isinstance(reward_config, dict) and str(reward_config.get("reward_version") or "").strip():
        resolved_reward_version = str(reward_config.get("reward_version"))
    return _normalize_component_weights(
        weights or dict((reward_config or {}).get("weights") or {}),
        reward_version=_normalize_reward_version(resolved_reward_version),
    )


def build_open_ended_reward_judge(
    *,
    reward_config: Optional[Dict[str, Any]] = None,
) -> OpenAICompatibleLlmJudge:
    del reward_config
    return OpenAICompatibleLlmJudge()


def build_timesearch_reward_funcs(
    *,
    reward_config: Optional[Dict[str, Any]] = None,
    llm_judge: Optional[OpenAICompatibleLlmJudge] = None,
    reward_version: str = "timesearch_v3",
) -> List[Any]:
    del reward_config
    judge = llm_judge or OpenAICompatibleLlmJudge()

    def accuracy_reward(*, rollout_traces: Sequence[Dict[str, Any]], **_: Any) -> List[float]:
        traces = list(rollout_traces or [])
        if not traces:
            return []

        # Phase A: per-rollout collect (question, reference, prediction) triples
        # for every open-ended sub-reward (skip-empty gating already applied).
        queries_per_rollout: List[Dict[str, Tuple[str, str, str]]] = [
            _collect_semantic_queries(trace, reward_version=reward_version)
            for trace in traces
        ]

        # Phase B: flatten across rollouts and batch-judge in a single call.
        # Single rank, single engine.chat() -> continuous batching wins.
        flat_queries: List[Tuple[str, str, str]] = []
        index_map: List[Tuple[int, str]] = []  # (rollout_idx, qa_type)
        for i, qmap in enumerate(queries_per_rollout):
            for qa_type, triple in qmap.items():
                flat_queries.append(triple)
                index_map.append((i, qa_type))
        flat_scores: List[float] = judge.score_batch(flat_queries) if flat_queries else []

        # Fan out scores back to per-rollout override dicts.
        overrides: List[Dict[str, float]] = [{} for _ in traces]
        for (i, qa_type), score in zip(index_map, flat_scores):
            overrides[i][qa_type] = float(score)

        # Phase C: per-rollout final breakdown with semantic scores pre-supplied.
        return [
            float(_compute_accuracy_breakdown(
                trace,
                llm_judge=judge,
                reward_version=reward_version,
                semantic_override=overrides[i],
            )["accuracy_reward"])
            for i, trace in enumerate(traces)
        ]

    def fecv_evidence_faithfulness_reward(
        *,
        rollout_traces: Sequence[Dict[str, Any]],
        **_: Any,
    ) -> List[float]:
        values: List[float] = []
        for rollout_trace in list(rollout_traces or []):
            profile = _extract_counterfactual_profile(rollout_trace)
            target = _infer_target(rollout_trace)
            values.append(float(_timesearch_fecv_reward(profile, target=target, rollout_trace=rollout_trace)))
        return values

    def protocol_finalize_reward(*, rollout_traces: Sequence[Dict[str, Any]], **_: Any) -> List[float]:
        return [
            float(_protocol_finalize_reward(rollout_trace, _latest_verifier_turn(rollout_trace)))
            for rollout_trace in list(rollout_traces or [])
        ]

    accuracy_reward.__name__ = "accuracy_reward"
    fecv_evidence_faithfulness_reward.__name__ = "fecv_evidence_faithfulness_reward"
    protocol_finalize_reward.__name__ = "protocol_finalize_reward"
    return [
        accuracy_reward,
        fecv_evidence_faithfulness_reward,
        protocol_finalize_reward,
    ]


def _normalize_existence(value: Any) -> str:
    return normalize_existence(value)


def _normalize_counterfactual_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text or "none"


def _normalize_decision_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    normalized = copy.deepcopy(payload)
    if "existence" in normalized:
        normalized["existence"] = _normalize_existence(normalized.get("existence"))
    if "category" in normalized:
        normalized["category"] = canonicalize_saver_category(
            normalized.get("category"),
            existence=normalized.get("existence"),
        ) or str(normalized.get("category") or "").strip().lower()
    if "counterfactual_type" in normalized:
        normalized["counterfactual_type"] = _normalize_counterfactual_type(normalized.get("counterfactual_type"))
    return normalized


def _infer_target(rollout_trace: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("scoring_target", "structured_target", "target"):
        payload = rollout_trace.get(key)
        if isinstance(payload, dict) and payload:
            return _normalize_decision_payload(payload)
    return {}


def _infer_final_prediction(rollout_trace: Dict[str, Any]) -> Dict[str, Any]:
    state = rollout_trace.get("state") or {}
    for payload in (rollout_trace.get("final_answer"), state.get("finalized_case")):
        if isinstance(payload, dict) and payload:
            return _normalize_decision_payload(payload)
    semantic_payload = _infer_semantic_payload(rollout_trace)
    decision = dict((semantic_payload or {}).get("decision") or {})
    if decision:
        return _normalize_decision_payload(decision)
    return {}


def _reward_uses_counterfactual_type(reward_version: str) -> bool:
    return _normalize_reward_version(reward_version) != "timesearch_v3"


def _decision_matches(
    prediction: Dict[str, Any],
    target: Dict[str, Any],
    *,
    use_counterfactual_type: bool = False,
    use_severity: bool = False,
) -> bool:
    if not prediction or not target:
        return False
    if _normalize_existence(prediction.get("existence")) != _normalize_existence(target.get("existence")):
        return False
    target_existence = _normalize_existence(target.get("existence"))
    if target_existence == "anomaly":
        pred_category = canonicalize_saver_category(prediction.get("category"), existence="anomaly")
        target_category = canonicalize_saver_category(target.get("category"), existence="anomaly")
        if target_category and pred_category != target_category:
            return False
    if use_counterfactual_type:
        target_counterfactual_type = _normalize_counterfactual_type(target.get("counterfactual_type"))
        if target_counterfactual_type != "none":
            if _normalize_counterfactual_type(prediction.get("counterfactual_type")) != target_counterfactual_type:
                return False
    if use_severity and target.get("severity") is not None and prediction.get("severity") is not None:
        if int(round(_safe_float(prediction.get("severity")))) != int(round(_safe_float(target.get("severity")))):
            return False
    return True


def _latest_verifier_turn(rollout_trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for turn in reversed(list(rollout_trace.get("turns") or [])):
        if str(turn.get("tool_name") or "") == "verify_hypothesis":
            copied = dict(turn)
            copied["_verifier_source"] = "online_turn"
            copied["_uses_reference_conditioned_verifier"] = False
            return copied
    offline_verifier = rollout_trace.get("offline_verifier")
    if isinstance(offline_verifier, dict):
        uses_reference_conditioned_verifier = bool(offline_verifier.get("reference_conditioned", True))
        return {
            "tool_name": "verify_hypothesis",
            "verifier_primary_status": offline_verifier.get("primary_status"),
            "verifier_recommended_action": offline_verifier.get("recommended_action"),
            "verifier_derived_scores": offline_verifier.get("derived_scores") or {},
            "covered_stages": offline_verifier.get("covered_stages") or [],
            "missing_required_stages": offline_verifier.get("missing_required_stages") or [],
            "stage_selected_moment_ids": offline_verifier.get("stage_selected_moment_ids") or {},
            "_verifier_source": (
                "offline_reference_conditioned" if uses_reference_conditioned_verifier else "offline_unconditioned"
            ),
            "_uses_reference_conditioned_verifier": uses_reference_conditioned_verifier,
        }
    return None


def _normal_evidence_tool_turn_count(rollout_trace: Dict[str, Any]) -> int:
    turns = list(rollout_trace.get("turns") or [])
    return sum(
        1
        for turn in turns
        if str(turn.get("tool_name") or "").strip() in NORMAL_EVIDENCE_TOOL_NAMES
    )


def _normal_search_tool_counts(rollout_trace: Dict[str, Any]) -> Tuple[int, int, int]:
    turns = list(rollout_trace.get("turns") or [])
    num_scan = sum(1 for turn in turns if str(turn.get("tool_name") or "").strip() == "scan_timeline")
    num_seek = sum(1 for turn in turns if str(turn.get("tool_name") or "").strip() == "seek_evidence")
    num_verify = sum(1 for turn in turns if str(turn.get("tool_name") or "").strip() == "verify_hypothesis")
    return int(num_scan), int(num_seek), int(num_verify)


def _normal_search_restraint_score(rollout_trace: Dict[str, Any]) -> float:
    num_scan, num_seek, num_verify = _normal_search_tool_counts(rollout_trace)
    score = 1.0
    score -= 0.05 * max(0, int(num_scan) - 1)
    score -= 0.10 * int(num_seek)
    score -= 0.10 * max(0, int(num_verify) - 1)
    return round(_clamp(score), 6)


def _normal_window_restraint_score(
    *,
    selected_window_count: int,
    selected_by_stage: Dict[str, Any],
) -> float:
    normalized_selected_by_stage = dict(selected_by_stage or {})
    score = 1.0
    score -= 0.15 * max(0, int(selected_window_count))
    if list(normalized_selected_by_stage.get("trigger") or []):
        score -= 0.25
    if list(normalized_selected_by_stage.get("precursor") or []):
        score -= 0.10
    if list(normalized_selected_by_stage.get("confirmation") or []):
        score -= 0.10
    return round(_clamp(score), 6)


def _normal_verification_consistency_score(rollout_trace: Dict[str, Any]) -> float:
    verifier_turn = _latest_verifier_turn(rollout_trace)
    if verifier_turn is None:
        return 1.0
    failure_reasons = list(verifier_turn.get("verifier_failure_reasons") or verifier_turn.get("failure_reasons") or [])
    invalid_selected_ids = list(verifier_turn.get("invalid_selected_window_ids") or [])
    recommended_action = str(
        verifier_turn.get("verifier_recommended_action")
        or verifier_turn.get("recommended_action")
        or ""
    ).strip().lower()
    if not failure_reasons and not invalid_selected_ids:
        if recommended_action == "finalize":
            return 1.0
        if recommended_action not in {"continue_search", "revise_claim", "refine_evidence"}:
            return 0.6
    return 0.2


def _normal_continuous_verifier_score(rollout_trace: Dict[str, Any]) -> float:
    details = _normal_continuous_verifier_score_details(rollout_trace)
    return round(float(details.get("score_after_action") or 0.0), 6)


def _normal_continuous_verifier_score_details(rollout_trace: Dict[str, Any]) -> Dict[str, Any]:
    verifier_turn = _latest_verifier_turn(rollout_trace)
    if verifier_turn is None:
        return {
            "primary_status": "unknown",
            "recommended_action": "unknown",
            "base_status_score": 0.5,
            "action_offset": 0.0,
            "score_before_action": 0.5,
            "score_after_action": 0.5,
        }
    primary_status = str(
        verifier_turn.get("verifier_primary_status")
        or verifier_turn.get("primary_status")
        or ""
    ).strip().lower()
    recommended_action = str(
        verifier_turn.get("verifier_recommended_action")
        or verifier_turn.get("recommended_action")
        or ""
    ).strip().lower()
    status_scores = {
        "complete": 1.0,
        "redundant": 0.75,
        "incomplete": 0.35,
        "misaligned": 0.0,
        "unknown": 0.5,
    }
    normalized_status = primary_status or "unknown"
    normalized_action = recommended_action or "unknown"
    base_status_score = float(status_scores.get(normalized_status, 0.5))
    action_offset = 0.0
    score_after_action = _clamp(base_status_score)
    return {
        "primary_status": normalized_status,
        "recommended_action": normalized_action,
        "base_status_score": round(float(base_status_score), 6),
        "action_offset": round(float(action_offset), 6),
        "score_before_action": round(float(base_status_score), 6),
        "score_after_action": round(float(score_after_action), 6),
    }


def _normal_case_type(
    rollout_trace: Dict[str, Any],
    *,
    target: Dict[str, Any],
    selected_window_count: int,
    use_counterfactual_type: bool = True,
) -> str:
    final_prediction = _infer_final_prediction(rollout_trace)
    if not _decision_matches(
        final_prediction,
        target,
        use_counterfactual_type=use_counterfactual_type,
    ):
        return "incorrect_normal"
    verifier_turn = _latest_verifier_turn(rollout_trace)
    num_scan, num_seek, num_verify = _normal_search_tool_counts(rollout_trace)
    if (
        int(num_scan) <= 1
        and int(num_seek) == 0
        and int(num_verify) == 0
        and int(selected_window_count) == 0
        and verifier_turn is None
    ):
        return "easy_normal"
    return "suspicious_normal"


def _normal_provenance_source_bucket(selection_resolution_source: str) -> str:
    normalized_source = str(selection_resolution_source or "").strip().lower()
    if normalized_source in {
        "verification_record_verified_window_ids",
        "verification_record_selected_window_ids",
        "latest_verify_turn_selected_window_ids",
        "verify_turn_verified_window_ids",
        "verify_turn_self_verification_selected_window_ids",
        "verification_record",
        "latest_verify_turn",
    }:
        return "verified"
    if normalized_source in {
        "verification_record_best_effort_window_ids",
        "evidence_anchor_selected_window_ids_after",
        "evidence_anchor",
    }:
        return "best_effort_or_evidence_anchor"
    return "active_evidence_or_unknown_or_normal_sample_skipped"


def _normal_provenance_score(selection_resolution_source: str) -> float:
    bucket = _normal_provenance_source_bucket(selection_resolution_source)
    if bucket == "verified":
        return 1.0
    if bucket == "best_effort_or_evidence_anchor":
        return 0.7
    return 0.4


def _normal_query_alignment_score(rollout_trace: Dict[str, Any]) -> float:
    turns = list(rollout_trace.get("turns") or [])
    if not any(str(turn.get("tool_name") or "").strip() == "seek_evidence" for turn in turns):
        return 1.0
    return round(_clamp((_query_alignment_reward(rollout_trace) + 1.0) / 2.0), 6)


def _window_duration_seconds(entry: Dict[str, Any]) -> float:
    start_sec = entry.get("start_sec")
    end_sec = entry.get("end_sec")
    if start_sec is not None and end_sec is not None:
        duration = _safe_float(end_sec) - _safe_float(start_sec)
        if duration > 0.0:
            return float(duration)
    timestamps = [_safe_float(value, 0.0) for value in list(entry.get("selected_timestamps") or [])]
    if len(timestamps) >= 2:
        duration = max(timestamps) - min(timestamps)
        if duration > 0.0:
            return float(duration)
    return 0.0


def _normal_selected_duration_ratio(
    rollout_trace: Dict[str, Any],
    *,
    selected_window_ids: Sequence[str] | None,
) -> float:
    selected_ids = {str(value).strip() for value in list(selected_window_ids or []) if str(value).strip()}
    if not selected_ids:
        return 0.0
    state = dict(rollout_trace.get("state") or {})
    evidence_ledger = [dict(entry) for entry in list(state.get("evidence_ledger") or []) if isinstance(entry, dict)]
    selected_records = [
        entry
        for entry in evidence_ledger
        if str(entry.get("window_id") or "").strip() in selected_ids
    ]
    selected_duration = sum(_window_duration_seconds(entry) for entry in selected_records)
    total_duration = sum(_window_duration_seconds(entry) for entry in evidence_ledger)
    if total_duration > 0.0:
        return round(_clamp(selected_duration / total_duration), 6)
    total_window_count = max(
        len({
            str(entry.get("window_id") or "").strip()
            for entry in evidence_ledger
            if str(entry.get("window_id") or "").strip()
        }),
        len(selected_ids),
    )
    if total_window_count <= 0:
        return 0.0
    return round(_clamp(float(len(selected_ids)) / float(total_window_count)), 6)


def _verifier_trace_score_from_payload(
    derived_scores: Dict[str, Any],
    *,
    default: float = 0.5,
) -> float:
    if not isinstance(derived_scores, dict) or not derived_scores:
        return round(_clamp(default), 6)
    sufficiency = _safe_float(
        derived_scores.get("sufficiency_score", derived_scores.get("sufficiency")),
        default,
    )
    necessity = _safe_float(
        derived_scores.get("necessity_score", derived_scores.get("necessity")),
        default,
    )
    finalize_readiness = _safe_float(
        derived_scores.get("finalize_readiness_score", derived_scores.get("finalize_readiness")),
        default,
    )
    counterfactual_faithfulness = _safe_float(
        derived_scores.get("counterfactual_faithfulness", derived_scores.get("counterfactual_faithfulness_score")),
        default,
    )
    return round(
        _clamp(
            (
                float(sufficiency)
                + float(necessity)
                + float(finalize_readiness)
                + float(counterfactual_faithfulness)
            ) / 4.0
        ),
        6,
    )


def _normal_verifier_trace_score(
    rollout_trace: Dict[str, Any],
    *,
    continuous_verifier_score: float,
    verification_consistency_score: float,
) -> float:
    verifier_turn = _latest_verifier_turn(rollout_trace)
    derived_scores = dict((verifier_turn or {}).get("verifier_derived_scores") or {})
    if derived_scores:
        return _verifier_trace_score_from_payload(
            derived_scores,
            default=continuous_verifier_score,
        )
    return round(
        _clamp(0.70 * float(continuous_verifier_score) + 0.30 * float(verification_consistency_score)),
        6,
    )


def _normal_skip_restraint_reward(
    rollout_trace: Dict[str, Any],
    *,
    target: Dict[str, Any],
    selected_window_count: int,
    selected_window_ids: Optional[Sequence[str]] = None,
    selected_by_stage: Optional[Dict[str, Any]] = None,
    selection_resolution_source: str = "",
    use_counterfactual_type: bool = True,
) -> Dict[str, Any]:
    evidence_tool_turn_count = _normal_evidence_tool_turn_count(rollout_trace)
    final_prediction = _infer_final_prediction(rollout_trace)
    case_type = _normal_case_type(
        rollout_trace,
        target=target,
        selected_window_count=selected_window_count,
        use_counterfactual_type=use_counterfactual_type,
    )
    provenance_score = _normal_provenance_score(selection_resolution_source)
    provenance_source_bucket = _normal_provenance_source_bucket(selection_resolution_source)
    window_restraint_score = _normal_window_restraint_score(
        selected_window_count=selected_window_count,
        selected_by_stage=dict(selected_by_stage or {}),
    )
    continuous_verifier_details = _normal_continuous_verifier_score_details(rollout_trace)
    continuous_verifier_score = float(continuous_verifier_details.get("score_after_action") or 0.0)
    verification_consistency_score = _normal_verification_consistency_score(rollout_trace)
    query_alignment_score = _normal_query_alignment_score(rollout_trace)
    selected_duration_ratio = _normal_selected_duration_ratio(
        rollout_trace,
        selected_window_ids=selected_window_ids,
    )
    verifier_trace_score_normal = _normal_verifier_trace_score(
        rollout_trace,
        continuous_verifier_score=continuous_verifier_score,
        verification_consistency_score=verification_consistency_score,
    )
    grounded_local_score = _clamp(
        0.35 * window_restraint_score
        + 0.20 * provenance_score
        + 0.25 * (1.0 - selected_duration_ratio)
        + 0.20 * verifier_trace_score_normal
    )
    easy_normal_sample_loss_multiplier = (
        float(EASY_NORMAL_SAMPLE_LOSS_MULTIPLIER) if case_type == "easy_normal" else 1.0
    )
    if not _decision_matches(
        final_prediction,
        target,
        use_counterfactual_type=use_counterfactual_type,
    ):
        return {
            "normal_reward_mode": "restraint_v5",
            "normal_case_type": case_type,
            "easy_normal_sample_loss_multiplier": round(float(easy_normal_sample_loss_multiplier), 6),
            "normal_evidence_tool_turn_count": int(evidence_tool_turn_count),
            "normal_search_restraint_score": 0.0,
            "normal_window_restraint_score": 0.0,
            "normal_verification_consistency_score": 0.0,
            "normal_query_alignment_score": 0.0,
            "normal_continuous_verifier_score": 0.0,
            "normal_verifier_primary_status": str(continuous_verifier_details.get("primary_status") or "unknown"),
            "normal_verifier_recommended_action": str(
                continuous_verifier_details.get("recommended_action") or "unknown"
            ),
            "normal_verifier_base_status_score": round(
                float(continuous_verifier_details.get("base_status_score") or 0.0),
                6,
            ),
            "normal_verifier_action_offset": round(
                float(continuous_verifier_details.get("action_offset") or 0.0),
                6,
            ),
            "normal_continuous_verifier_score_before_action": round(
                float(continuous_verifier_details.get("score_before_action") or 0.0),
                6,
            ),
            "normal_continuous_verifier_score_after_action": 0.0,
            "normal_verifier_trace_score": 0.0,
            "normal_selected_duration_ratio": 0.0,
            "normal_grounded_local_mode": "grounded_local_score_v2",
            "normal_grounded_local_score": 0.0,
            "normal_provenance_score": 0.0,
            "normal_provenance_source_bucket": provenance_source_bucket,
            "normal_restraint_reward": 0.0,
        }
    search_restraint_score = _normal_search_restraint_score(rollout_trace)
    if case_type == "easy_normal":
        reward = _clamp(
            0.55 * search_restraint_score
            + 0.25 * window_restraint_score
            + 0.20 * verifier_trace_score_normal
        )
    else:
        reward = _clamp(
            0.35 * search_restraint_score
            + 0.25 * grounded_local_score
            + 0.20 * query_alignment_score
            + 0.20 * verifier_trace_score_normal
        )
    return {
        "normal_reward_mode": "restraint_v5",
        "normal_case_type": case_type,
        "easy_normal_sample_loss_multiplier": round(float(easy_normal_sample_loss_multiplier), 6),
        "normal_evidence_tool_turn_count": int(evidence_tool_turn_count),
        "normal_search_restraint_score": round(float(search_restraint_score), 6),
        "normal_window_restraint_score": round(float(window_restraint_score), 6),
        "normal_verification_consistency_score": round(float(verification_consistency_score), 6),
        "normal_query_alignment_score": round(float(query_alignment_score), 6),
        "normal_continuous_verifier_score": round(float(continuous_verifier_score), 6),
        "normal_verifier_primary_status": str(continuous_verifier_details.get("primary_status") or "unknown"),
        "normal_verifier_recommended_action": str(
            continuous_verifier_details.get("recommended_action") or "unknown"
        ),
        "normal_verifier_base_status_score": round(
            float(continuous_verifier_details.get("base_status_score") or 0.0),
            6,
        ),
        "normal_verifier_action_offset": round(
            float(continuous_verifier_details.get("action_offset") or 0.0),
            6,
        ),
        "normal_continuous_verifier_score_before_action": round(
            float(continuous_verifier_details.get("score_before_action") or 0.0),
            6,
        ),
        "normal_continuous_verifier_score_after_action": round(float(continuous_verifier_score), 6),
        "normal_verifier_trace_score": round(float(verifier_trace_score_normal), 6),
        "normal_selected_duration_ratio": round(float(selected_duration_ratio), 6),
        "normal_grounded_local_mode": "grounded_local_score_v2",
        "normal_grounded_local_score": round(float(grounded_local_score), 6),
        "normal_provenance_score": round(float(provenance_score), 6),
        "normal_provenance_source_bucket": provenance_source_bucket,
        "normal_restraint_reward": round(float(reward), 6),
    }


def _protocol_finalize_reward(rollout_trace: Dict[str, Any], verifier_turn: Optional[Dict[str, Any]]) -> float:
    turns = list(rollout_trace.get("turns") or [])
    state = rollout_trace.get("state") or {}
    final_answer = rollout_trace.get("final_answer")
    verify_steps = [
        int(turn.get("step_index") or 0)
        for turn in turns
        if str(turn.get("tool_name") or "") == "verify_hypothesis"
    ]
    finalize_steps = [
        int(turn.get("step_index") or 0)
        for turn in turns
        if str(turn.get("tool_name") or "") == "finalize_case"
    ]
    answer_steps = [
        int(turn.get("step_index") or 0)
        for turn in turns
        if str(turn.get("action") or "") == "answer"
    ]
    has_finalize_artifact = bool(finalize_steps) or isinstance(state.get("finalized_case"), dict)
    has_answer = isinstance(final_answer, dict) or (not answer_steps and has_finalize_artifact)
    if not has_finalize_artifact or not has_answer:
        return -1.0
    if verify_steps:
        first_verify = min(verify_steps)
        first_finalize = min(finalize_steps) if finalize_steps else first_verify + 1
        if first_finalize <= first_verify:
            return -1.0
    if verifier_turn is not None and str(verifier_turn.get("verifier_recommended_action") or "") == "finalize":
        return 1.0 if has_finalize_artifact else -1.0
    return 0.75


def _extract_counterfactual_profile(rollout_trace: Dict[str, Any]) -> Dict[str, Any]:
    profile = rollout_trace.get("counterfactual_profile")
    if isinstance(profile, dict) and profile:
        if "summary" in profile:
            return copy.deepcopy(profile)
        summary = {
            "decision_sufficiency": bool(profile.get("decision_sufficiency")),
            "minimal_subset_sufficiency": bool(profile.get("minimal_subset_sufficiency")),
            "negative_specificity_pass": bool(profile.get("negative_specificity_pass")),
            "stage_necessity": dict(profile.get("stage_necessity") or {}),
        }
        return {
            "summary": summary,
            "branch_field_matrix": dict(profile.get("branch_field_matrix") or {}),
            "branch_delta_matrix": dict(profile.get("branch_delta_matrix") or {}),
            "stage_packages": dict(profile.get("stage_packages") or {}),
            "selection_metadata": dict(profile.get("selection_metadata") or {}),
        }
    return {}


def _counterfactual_summary(profile: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(profile, dict):
        return {}
    summary = profile.get("summary")
    if isinstance(summary, dict):
        return summary
    return {}


def _counterfactual_branch_delta(profile: Dict[str, Any], branch_name: str, field_name: str) -> float:
    matrix = dict(profile.get("branch_delta_matrix") or {})
    return _safe_float((((matrix.get(branch_name) or {}).get("fields") or {}).get(field_name)), 0.0)


def _target_requires_counterfactual_type(target: Dict[str, Any], *, reward_version: str) -> bool:
    if not _reward_uses_counterfactual_type(reward_version):
        return False
    return _normalize_counterfactual_type(target.get("counterfactual_type")) != "none"


def _fecv_decision_sufficiency_reward(profile: Dict[str, Any]) -> float:
    if not profile:
        return 0.0
    summary = _counterfactual_summary(profile)
    if not (((profile.get("branch_field_matrix") or {}).get("full_selected") or {}).get("available")):
        return 0.0
    return 1.0 if bool(summary.get("decision_sufficiency")) else -1.0


def _fecv_specificity_reward(
    profile: Dict[str, Any],
    *,
    target: Dict[str, Any],
    reward_version: str = "legacy",
) -> float:
    if not profile:
        return 0.0
    summary = _counterfactual_summary(profile)
    profile_source = str(
        profile.get("counterfactual_profile_source") or profile.get("counterfactual_branch_profile") or ""
    ).strip().lower()
    selection_metadata = dict(profile.get("selection_metadata") or {})
    normalized_branch_profile = str(
        selection_metadata.get("normalized_branch_profile") or profile_source
    ).strip().lower()
    terms: list[float] = []
    minimal_subset_allowed = normalized_branch_profile not in {"", "online_core"}
    if minimal_subset_allowed and bool(((profile.get("branch_field_matrix") or {}).get("minimal_subset") or {}).get("available")):
        terms.append(1.0 if bool(summary.get("minimal_subset_sufficiency")) else -1.0)
    negative_available = any(
        bool(((profile.get("branch_field_matrix") or {}).get(branch_name) or {}).get("available"))
        for branch_name in ("hard_negative_swap", "drop_trigger", "drop_confirmation", "drop_precursor")
    )
    if negative_available:
        negative_drop = max(
            _counterfactual_branch_delta(profile, "hard_negative_swap", "existence"),
            _counterfactual_branch_delta(profile, "hard_negative_swap", "category"),
        )
        if negative_drop <= 0.0:
            negative_drop = max(
                _counterfactual_branch_delta(profile, "drop_trigger", "existence"),
                _counterfactual_branch_delta(profile, "drop_trigger", "category"),
            )
        negative_score = min(1.0, max(negative_drop, 0.0))
        terms.append(negative_score if bool(summary.get("negative_specificity_pass")) else -1.0)
    if _target_requires_counterfactual_type(target, reward_version=reward_version):
        terms.append(1.0 if bool(summary.get("counterfactual_type_supported")) else -1.0)
    if not terms:
        return 0.0
    return max(-1.0, min(1.0, sum(terms) / float(len(terms))))


def _query_alignment_reward(rollout_trace: Dict[str, Any]) -> float:
    scores = [
        compute_query_alignment_score(turn)
        for turn in list(rollout_trace.get("turns") or [])
        if str(turn.get("tool_name") or "") == "seek_evidence"
    ]
    if not scores:
        return 0.0
    return max(-1.0, min(1.0, sum(float(score) for score in scores) / float(len(scores))))


def _stage_progress_reward(rollout_trace: Dict[str, Any]) -> float:
    target = _infer_target(rollout_trace)
    required_stages = infer_required_stages_from_target(target)
    if not required_stages:
        return 0.0
    required_set = set(normalize_event_chain_stages(required_stages))
    discovered: set[str] = set()
    progress = 0.0
    for turn in list(rollout_trace.get("turns") or []):
        annotation = extract_stage_annotation_from_turn(turn)
        covered = set(normalize_event_chain_stages((annotation or {}).get("covered_stages") or []))
        if not covered and str(turn.get("tool_name") or "") == "seek_evidence":
            covered = set(observed_stages_for_search_turn(turn))
        new_stages = (covered & required_set) - discovered
        if new_stages:
            progress += float(len(new_stages)) / float(len(required_set))
            discovered.update(new_stages)
    if discovered == required_set:
        progress += 0.15
    return max(-1.0, min(1.0, progress))


def _stage_reward(rollout_trace: Dict[str, Any], verifier_turn: Optional[Dict[str, Any]]) -> float:
    target = _infer_target(rollout_trace)
    required_stages = infer_required_stages_from_target(target)
    if not required_stages:
        return 0.0
    annotation = extract_stage_annotation_from_turn(verifier_turn) if verifier_turn else {}
    if not annotation:
        annotation = extract_stage_annotation_from_record(rollout_trace)
    event_chain_score = compute_event_chain_score(
        required_stages,
        annotation,
        terminal=bool(_infer_final_prediction(rollout_trace)),
    )
    progress_score = _stage_progress_reward(rollout_trace)
    return max(-1.0, min(1.0, 0.5 * float(event_chain_score) + 0.5 * float(progress_score)))


def _stage_necessity_reward(
    rollout_trace: Dict[str, Any],
    *,
    verifier_turn: Optional[Dict[str, Any]],
    profile: Dict[str, Any],
) -> float:
    stage_necessity = _counterfactual_summary(profile).get("stage_necessity")
    if isinstance(stage_necessity, dict) and stage_necessity:
        label_scores = {
            "narrative_only": 0.5,
            "decision_critical": 1.0,
            "finalize_critical": 1.0,
            "optional": 0.0,
            "non_critical": -0.5,
            "not_observed": 0.0,
        }
        values = [
            float(label_scores.get(str(value).strip().lower(), 0.0))
            for value in stage_necessity.values()
        ]
        if values:
            return max(-1.0, min(1.0, sum(values) / float(len(values))))
    return _stage_reward(rollout_trace, verifier_turn)


def _score_rollout_trace_legacy(
    rollout_trace: Dict[str, Any],
    *,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    normalized_weights = _normalize_component_weights(weights, reward_version="legacy")
    verifier_turn = _latest_verifier_turn(rollout_trace)
    target = _infer_target(rollout_trace)
    final_prediction = _infer_final_prediction(rollout_trace)
    profile = _extract_counterfactual_profile(rollout_trace)
    profile_summary = _counterfactual_summary(profile)

    final_decision_correct = 1.0 if _decision_matches(
        final_prediction,
        target,
        use_counterfactual_type=True,
        use_severity=True,
    ) else 0.0
    decision_sufficiency = 1.0 if bool(profile_summary.get("decision_sufficiency")) else 0.0
    minimal_subset_sufficiency = 1.0 if bool(profile_summary.get("minimal_subset_sufficiency")) else 0.0
    negative_specificity_pass = 1.0 if bool(profile_summary.get("negative_specificity_pass")) else 0.0
    counterfactual_type_supported = 1.0 if bool(profile_summary.get("counterfactual_type_supported")) else 0.0
    fecv_full_selected_available = bool(((profile.get("branch_field_matrix") or {}).get("full_selected") or {}).get("available"))
    fecv_grounded_decision = 1.0 if final_decision_correct > 0.0 and decision_sufficiency > 0.0 else 0.0

    components = {
        "fecv_decision_sufficiency_reward": float(_fecv_decision_sufficiency_reward(profile)),
        "fecv_specificity_reward": float(_fecv_specificity_reward(profile, target=target)),
        "protocol_finalize_reward": float(_protocol_finalize_reward(rollout_trace, verifier_turn)),
    }

    total_reward = 0.0
    for key, value in components.items():
        total_reward += float(normalized_weights.get(key, 0.0)) * float(value)

    return {
        "reward_version": "legacy",
        "total_reward": round(total_reward, 6),
        "components": {key: round(float(value), 6) for key, value in components.items()},
        "weights": dict(normalized_weights),
        "final_decision_correct": float(final_decision_correct),
        "counterfactual_sufficiency_reward": round(float(components["fecv_decision_sufficiency_reward"]), 6),
        "stage_necessity_reward": round(float(_stage_necessity_reward(rollout_trace, verifier_turn=verifier_turn, profile=profile)), 6),
        "query_alignment_reward": round(float(_query_alignment_reward(rollout_trace)), 6),
        "fecv_full_selected_available": bool(fecv_full_selected_available),
        "fecv_grounded_decision": float(fecv_grounded_decision),
        "fecv_decision_sufficiency": float(decision_sufficiency),
        "fecv_minimal_subset_sufficiency": float(minimal_subset_sufficiency),
        "fecv_negative_specificity_pass": float(negative_specificity_pass),
        "fecv_counterfactual_type_supported": float(counterfactual_type_supported),
        "counterfactual_decision_sufficiency": float(decision_sufficiency),
        "counterfactual_minimal_subset_sufficiency": float(minimal_subset_sufficiency),
        "negative_specificity_pass": float(negative_specificity_pass),
        "counterfactual_type_supported": float(counterfactual_type_supported),
        "latest_verifier_turn_present": verifier_turn is not None,
        "verifier_source": str(verifier_turn.get("_verifier_source")) if verifier_turn is not None else "none",
        "uses_reference_conditioned_verifier": bool(
            verifier_turn.get("_uses_reference_conditioned_verifier") if verifier_turn is not None else False
        ),
    }


def _infer_scoring_qa_pairs(rollout_trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("scoring_qa_pairs", "qa_pairs"):
        value = rollout_trace.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _infer_scoring_evidence_moments(rollout_trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("scoring_evidence_moments",):
        value = rollout_trace.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
    evidence = rollout_trace.get("evidence")
    if isinstance(evidence, dict):
        moments = evidence.get("evidence_moments")
        if isinstance(moments, list):
            return [dict(item) for item in moments if isinstance(item, dict)]
    return []


def _infer_semantic_payload(rollout_trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    semantic_payload = normalize_semantic_answer_payload(rollout_trace.get("semantic_answer"))
    if semantic_payload is not None:
        return semantic_payload
    state = rollout_trace.get("state") or {}
    final_prediction: Dict[str, Any] = {}
    for payload in (rollout_trace.get("final_answer"), state.get("finalized_case")):
        if isinstance(payload, dict) and payload:
            final_prediction = _normalize_decision_payload(payload)
            break
    if not final_prediction:
        return None
    fallback_payload = {
        "decision": final_prediction,
        "summary": "",
        "rationale": "",
        "event_chain_summary": {stage: "" for stage in SEMANTIC_EVENT_CHAIN_STAGES},
        "qa_focus_answers": {},
    }
    return normalize_semantic_answer_payload(fallback_payload)


def _normalize_severity(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return ""
    return str(int(round(_safe_float(value)))).strip()


def _reference_semantic_payload(
    *,
    structured_target: Dict[str, Any],
    qa_pairs: Sequence[Dict[str, Any]],
    evidence_moments: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return build_public_semantic_replay_payload(
        structured_target=structured_target,
        qa_pairs=qa_pairs,
        evidence_moments=evidence_moments,
    )


def _qa_pairs_by_type(qa_pairs: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    pairs: Dict[str, Dict[str, Any]] = {}
    for pair in qa_pairs:
        qa_type = str(pair.get("type") or "").strip().lower()
        if qa_type and qa_type not in pairs:
            pairs[qa_type] = dict(pair)
    return pairs


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = [token for token in normalize_text_match(prediction).split(" ") if token]
    ref_tokens = [token for token in normalize_text_match(reference).split(" ") if token]
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_counts: Dict[str, int] = {}
    ref_counts: Dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    overlap = 0
    for token, count in pred_counts.items():
        overlap += min(count, ref_counts.get(token, 0))
    if overlap <= 0:
        return 0.0
    precision = float(overlap) / float(len(pred_tokens))
    recall = float(overlap) / float(len(ref_tokens))
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


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
    if union <= 0.0:
        return 0.0
    return intersection / union


def _parse_interval_from_text(text: Any, *, fps: float = 0.0) -> Optional[List[float]]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return None
    second_match = _SECOND_INTERVAL_RE.search(raw_text)
    if second_match is not None:
        start_sec = _safe_float(second_match.group(1))
        end_sec = _safe_float(second_match.group(2))
        return [float(start_sec), float(end_sec)]
    frame_match = _FRAME_INTERVAL_RE.search(raw_text)
    if frame_match is not None and float(fps) > 0.0:
        start_frame = _safe_float(frame_match.group(1))
        end_frame = _safe_float(frame_match.group(2))
        return [float(start_frame) / float(fps), float(end_frame) / float(fps)]
    return None


def _prediction_interval_from_semantic_payload(
    semantic_payload: Optional[Dict[str, Any]],
    *,
    decision_field: str,
    qa_key: str,
    fps: float,
) -> Optional[List[float]]:
    decision = dict((semantic_payload or {}).get("decision") or {})
    interval = decision.get(decision_field)
    if isinstance(interval, list) and len(interval) >= 2:
        return [float(interval[0]), float(interval[1])]
    qa_focus_answers = dict((semantic_payload or {}).get("qa_focus_answers") or {})
    return _parse_interval_from_text(qa_focus_answers.get(qa_key), fps=fps)


def _open_ended_target_map(
    *,
    structured_target: Dict[str, Any],
    qa_pairs: Sequence[Dict[str, Any]],
    evidence_moments: Sequence[Dict[str, Any]],
) -> Dict[str, Tuple[str, str]]:
    semantic_target = _reference_semantic_payload(
        structured_target=structured_target,
        qa_pairs=qa_pairs,
        evidence_moments=evidence_moments,
    )
    pair_map = _qa_pairs_by_type(qa_pairs)
    targets: Dict[str, Tuple[str, str]] = {}
    summary = str(semantic_target.get("summary") or "").strip()
    rationale = str(semantic_target.get("rationale") or "").strip()
    if summary:
        targets["summary"] = ("Summarize the case.", summary)
    if rationale:
        targets["rationale"] = ("Explain the evidence-grounded rationale.", rationale)
    normal_reason_pair = pair_map.get("normal_reason")
    normal_reason_target = str((normal_reason_pair or {}).get("answer") or summary or rationale or "").strip()
    if normal_reason_target:
        question = str((normal_reason_pair or {}).get("question") or "Why is the video normal?").strip()
        targets["normal_reason"] = (question, normal_reason_target)
    trigger_pair = pair_map.get("trigger_evidence")
    trigger_target = str(
        (trigger_pair or {}).get("answer")
        or ((semantic_target.get("event_chain_summary") or {}).get("trigger"))
        or rationale
        or ""
    ).strip()
    if trigger_target:
        question = str((trigger_pair or {}).get("question") or "What visible evidence first makes the anomaly actionable?").strip()
        targets["trigger_evidence"] = (question, trigger_target)
    event_chain_summary = dict(semantic_target.get("event_chain_summary") or {})
    for stage in SEMANTIC_EVENT_CHAIN_STAGES:
        text = str(event_chain_summary.get(stage) or "").strip()
        if text:
            targets[f"event_chain_summary.{stage}"] = (
                f"Summarize the {stage} evidence.",
                text,
            )
    return targets


def _open_ended_prediction_map(semantic_payload: Optional[Dict[str, Any]]) -> Dict[str, str]:
    event_chain_summary = dict((semantic_payload or {}).get("event_chain_summary") or {})
    summary = str((semantic_payload or {}).get("summary") or "").strip()
    rationale = str((semantic_payload or {}).get("rationale") or "").strip()
    predictions: Dict[str, str] = {}
    if summary:
        predictions["summary"] = summary
    if rationale:
        predictions["rationale"] = rationale
    normal_reason = summary or rationale
    if normal_reason:
        predictions["normal_reason"] = normal_reason
    trigger_evidence = str(event_chain_summary.get("trigger") or rationale or summary or "").strip()
    if trigger_evidence:
        predictions["trigger_evidence"] = trigger_evidence
    for stage in SEMANTIC_EVENT_CHAIN_STAGES:
        text = str(event_chain_summary.get(stage) or "").strip()
        if text:
            predictions[f"event_chain_summary.{stage}"] = text
    return predictions


def _collect_semantic_queries(
    rollout_trace: Dict[str, Any],
    *,
    reward_version: str = "timesearch_v3",
) -> Dict[str, Tuple[str, str, str]]:
    """Collect (question, reference, prediction) triples for every measurable
    open-ended sub-reward of this rollout, WITHOUT calling any LLM judge.

    Mirrors the skip-empty / target-absent gating in _compute_accuracy_breakdown
    so callers can batch-score many rollouts in one engine.chat() invocation,
    then feed the resulting scores back via `semantic_override=` keyword.
    """
    structured_target = _infer_target(rollout_trace)
    semantic_payload = _infer_semantic_payload(rollout_trace)
    qa_pairs = _infer_scoring_qa_pairs(rollout_trace)
    evidence_moments = _infer_scoring_evidence_moments(rollout_trace)
    open_targets = _open_ended_target_map(
        structured_target=structured_target,
        qa_pairs=qa_pairs,
        evidence_moments=evidence_moments,
    )
    open_predictions = _open_ended_prediction_map(semantic_payload)
    _is_v3 = reward_version == "timesearch_v3"
    _open_ended_types = _OPEN_ENDED_QUESTION_TYPES_V3 if _is_v3 else _OPEN_ENDED_QUESTION_TYPES
    out: Dict[str, Tuple[str, str, str]] = {}
    for qa_type in _open_ended_types:
        target_entry = open_targets.get(qa_type)
        if target_entry is None:
            continue
        question, reference = target_entry
        prediction = str(open_predictions.get(qa_type) or "").strip()
        if not prediction:
            continue
        out[qa_type] = (str(question or ""), str(reference or ""), prediction)
    return out


def _compute_accuracy_breakdown(
    rollout_trace: Dict[str, Any],
    *,
    llm_judge: Optional[OpenAICompatibleLlmJudge] = None,
    reward_version: str = "timesearch_v3",
    semantic_override: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    structured_target = _infer_target(rollout_trace)
    semantic_payload = _infer_semantic_payload(rollout_trace)
    final_prediction = dict((semantic_payload or {}).get("decision") or {}) or _infer_final_prediction(rollout_trace)
    qa_pairs = _infer_scoring_qa_pairs(rollout_trace)
    evidence_moments = _infer_scoring_evidence_moments(rollout_trace)
    fps = _safe_float(((rollout_trace.get("video_meta") or {}).get("fps")), 0.0)
    if fps <= 0.0:
        fps = _safe_float((((rollout_trace.get("multimodal_cache") or {}).get("fps"))), 0.0)

    family_scores: Dict[str, List[float]] = {
        "multiple_choice": [],
        "grounding": [],
        "open_ended": [],
    }
    type_scores: Dict[str, float] = {}
    # Fix B (2026-04-21): each measurable sub-reward (existence / category / temporal /
    # open-ended / etc.) contributes one independent score with equal weight 1/N.
    # A sub-reward is only appended when it can actually be computed (prediction and
    # target both present). Missing fields are skipped, never zero-padded — so one
    # unparseable field cannot drag its siblings down. `family_scores` is kept below
    # for the diagnostic `accuracy_by_family` output consumers, but the authoritative
    # `accuracy_reward` is the flat mean of `all_scores`.
    all_scores: List[float] = []
    _is_v3 = reward_version == "timesearch_v3"

    if structured_target:
        prediction_existence = _normalize_existence(final_prediction.get("existence"))
        target_existence = _normalize_existence(structured_target.get("existence"))
        if prediction_existence and target_existence:
            existence_score = 1.0 if prediction_existence == target_existence else 0.0
            all_scores.append(existence_score)
            family_scores["multiple_choice"].append(existence_score)
            type_scores["existence"] = existence_score
        else:
            # Prediction or target existence could not be resolved — skip this
            # sub-reward entirely (do NOT penalize with 0), only expose a zero
            # for diagnostic visibility.
            type_scores["existence"] = 0.0

        target_category = canonicalize_saver_category(structured_target.get("category"), existence=target_existence)
        prediction_category_raw = final_prediction.get("category")
        if target_category and prediction_category_raw is not None and str(prediction_category_raw).strip():
            prediction_category = canonicalize_saver_category(
                prediction_category_raw,
                existence=prediction_existence or None,
            )
            category_score = 1.0 if prediction_category == target_category else 0.0
            all_scores.append(category_score)
            family_scores["multiple_choice"].append(category_score)
            type_scores["category"] = category_score
        elif target_category:
            # Target expects a category but prediction omitted it — skip this
            # sub-reward (missing ≠ wrong); record diagnostic zero.
            type_scores["category"] = 0.0

        if structured_target.get("severity") is not None:
            severity_score = 1.0 if _normalize_severity(final_prediction.get("severity")) == _normalize_severity(structured_target.get("severity")) else 0.0
            if not _is_v3:  # v3: no primary metric for severity
                all_scores.append(severity_score)
                family_scores["multiple_choice"].append(severity_score)
            type_scores["severity"] = severity_score

        target_counterfactual_type = _normalize_counterfactual_type(structured_target.get("counterfactual_type"))
        if target_counterfactual_type != "none":
            prediction_counterfactual_type = _normalize_counterfactual_type(final_prediction.get("counterfactual_type"))
            counterfactual_score = 1.0 if prediction_counterfactual_type == target_counterfactual_type else 0.0
            if not _is_v3:  # v3: no primary metric for counterfactual_type
                all_scores.append(counterfactual_score)
                family_scores["multiple_choice"].append(counterfactual_score)
            type_scores["counterfactual"] = counterfactual_score

        target_anomaly_interval = structured_target.get("anomaly_interval_sec")
        if isinstance(target_anomaly_interval, list) and len(target_anomaly_interval) >= 2:
            prediction_anomaly_interval = _prediction_interval_from_semantic_payload(
                semantic_payload,
                decision_field="anomaly_interval_sec",
                qa_key="temporal",
                fps=fps,
            )
            if prediction_anomaly_interval is not None:
                temporal_score = _interval_iou(prediction_anomaly_interval, target_anomaly_interval)
                all_scores.append(temporal_score)
                family_scores["grounding"].append(temporal_score)
                type_scores["temporal"] = temporal_score
            else:
                # Prediction omitted temporal interval — skip sub-reward
                # (align with existence/category skip semantics).
                type_scores["temporal"] = 0.0

        target_precursor_interval = structured_target.get("precursor_interval_sec")
        if isinstance(target_precursor_interval, list) and len(target_precursor_interval) >= 2:
            prediction_precursor_interval = _prediction_interval_from_semantic_payload(
                semantic_payload,
                decision_field="precursor_interval_sec",
                qa_key="precursor_temporal",
                fps=fps,
            )
            if prediction_precursor_interval is not None:
                precursor_score = _interval_iou(prediction_precursor_interval, target_precursor_interval)
                if not _is_v3:  # v3: no primary metric for precursor temporal
                    all_scores.append(precursor_score)
                    family_scores["grounding"].append(precursor_score)
                type_scores["precursor_temporal"] = precursor_score
            else:
                type_scores["precursor_temporal"] = 0.0

    judge = llm_judge or OpenAICompatibleLlmJudge()
    open_targets = _open_ended_target_map(
        structured_target=structured_target,
        qa_pairs=qa_pairs,
        evidence_moments=evidence_moments,
    )
    open_predictions = _open_ended_prediction_map(semantic_payload)
    _open_ended_types = _OPEN_ENDED_QUESTION_TYPES_V3 if _is_v3 else _OPEN_ENDED_QUESTION_TYPES
    # Callers (e.g. the batched accuracy_reward closure) may pre-compute
    # semantic judge scores in one engine.chat() and feed them in via
    # `semantic_override`. When a qa_type score is in the override dict, we
    # skip the per-item judge.score() call but still apply skip-empty gating.
    _override_scores: Dict[str, float] = dict(semantic_override or {})
    for qa_type in _open_ended_types:
        target_entry = open_targets.get(qa_type)
        if target_entry is None:
            # No reference answer available — skip this sub-reward entirely.
            continue
        question, reference = target_entry
        prediction = str(open_predictions.get(qa_type) or "").strip()
        if not prediction:
            # Model did not emit this semantic field — skip (Fix B principle:
            # unmeasurable sub-reward is skipped, not zero-padded).
            type_scores[qa_type] = 0.0  # diagnostic only
            continue
        if qa_type in _override_scores:
            score = float(_override_scores[qa_type])
        else:
            score = judge.score(question=question, reference=reference, prediction=prediction)
        all_scores.append(score)
        family_scores["open_ended"].append(score)
        type_scores[qa_type] = score

    # Fix B (2026-04-21): flat independent mean of all measurable sub-rewards.
    # Each sub-reward (existence, category, temporal, open-ended QAs for non-v3)
    # contributes 1/N weight. Missing/unmeasurable fields are skipped, not zeroed.
    accuracy_reward = sum(all_scores) / float(len(all_scores)) if all_scores else 0.0
    # `accuracy_by_family` is retained for diagnostic consumers (reward_summary
    # output + tests). It reflects within-family means but is NOT used to compute
    # `accuracy_reward` any more.
    accuracy_by_family = {
        family: (sum(scores) / float(len(scores)) if scores else 0.0)
        for family, scores in family_scores.items()
    }
    # ACC_PROBE_START 2026-04-21 Fighting_6 accuracy-reward-0 investigation (Fix B)
    try:
        _vid = (
            (rollout_trace.get("video_meta") or {}).get("video_id")
            or rollout_trace.get("video_id")
            or (structured_target or {}).get("video_id")
            or "unknown"
        )
        _state = rollout_trace.get("state") or {}
        print(
            f"ACC_PROBE video_id={_vid} "
            f"accuracy_reward={float(accuracy_reward):.4f} "
            f"all_scores={all_scores} n={len(all_scores)} "
            f"fp_empty={not bool(final_prediction)} "
            f"fp_existence={final_prediction.get('existence') if final_prediction else None!r} "
            f"fp_category={final_prediction.get('category') if final_prediction else None!r} "
            f"fp_anomaly_interval={final_prediction.get('anomaly_interval_sec') if final_prediction else None!r} "
            f"target_existence={(structured_target or {}).get('existence')!r} "
            f"target_category={(structured_target or {}).get('category')!r} "
            f"target_anomaly_interval={(structured_target or {}).get('anomaly_interval_sec')!r} "
            f"sp_none={semantic_payload is None} "
            f"sp_has_decision={bool((semantic_payload or {}).get('decision')) if semantic_payload else False} "
            f"has_final_answer={bool(rollout_trace.get('final_answer'))} "
            f"has_finalized_case={bool(_state.get('finalized_case'))} "
            f"has_semantic_answer={bool(rollout_trace.get('semantic_answer'))} "
            f"fs_lens={{'mc':{len(family_scores['multiple_choice'])},'gr':{len(family_scores['grounding'])},'oe':{len(family_scores['open_ended'])}}} "
            f"family_scores={family_scores} "
            f"type_scores={type_scores}",
            flush=True,
        )
    except Exception as _probe_exc:
        print(f"ACC_PROBE error: {_probe_exc}", flush=True)
    # ACC_PROBE_END
    return {
        "accuracy_reward": round(float(accuracy_reward), 6),
        "accuracy_by_family": {
            family: round(float(value), 6)
            for family, value in accuracy_by_family.items()
        },
        "accuracy_by_type": {
            qa_type: round(float(score), 6)
            for qa_type, score in type_scores.items()
        },
        "accuracy_question_count": sum(len(scores) for scores in family_scores.values()),
    }


def _timesearch_selected_support_score(
    profile: Dict[str, Any],
    *,
    target: Dict[str, Any],
    reward_version: str = "timesearch_v3",
) -> float:
    """Continuous evidence support score -- NO hard boolean gate.

    Returns a smooth [0,1] score reflecting how well the selected evidence
    supports the decision. Unlike the previous version, this does NOT require
    all decision fields to pass a binary threshold; instead, each field
    contributes its raw score, providing gradient signal even when the model
    is partially correct (e.g., temporal IoU = 0.3 contributes 0.3, not 0).
    """
    full_selected = dict(((profile.get("branch_field_matrix") or {}).get("full_selected") or {}))
    if not bool(full_selected.get("available")):
        return 0.0
    fields = dict(full_selected.get("fields") or {})
    decision_keys = ["existence", "category"]
    if str(target.get("existence") or "").strip().lower() == "anomaly" and target.get("anomaly_interval_sec") is not None:
        decision_keys.append("temporal")
    if _target_requires_counterfactual_type(target, reward_version=reward_version):
        decision_keys.append("counterfactual_type")
    # Continuous scores -- no boolean gating. Each field contributes its raw score.
    decision_scores = [_safe_float((fields.get(key) or {}).get("score"), 0.0) for key in decision_keys if key in fields]
    decision_field_support = (
        sum(decision_scores) / float(len(decision_scores))
        if decision_scores
        else 0.0
    )
    required_stages = normalize_event_chain_stages(infer_required_stages_from_target(target))
    stage_scores = [_safe_float((fields.get(stage) or {}).get("score"), 0.0) for stage in required_stages if stage in fields]
    stage_text_support = sum(stage_scores) / float(len(stage_scores)) if stage_scores else 0.0
    return max(0.0, min(1.0, 0.7 * decision_field_support + 0.3 * stage_text_support))


def _timesearch_selected_support_score_v2(
    profile: Dict[str, Any],
    *,
    target: Dict[str, Any],
    reward_version: str = "timesearch_v3",
) -> float:
    full_selected = dict(((profile.get("branch_field_matrix") or {}).get("full_selected") or {}))
    if not bool(full_selected.get("available")):
        return 0.0
    fields = dict(full_selected.get("fields") or {})
    decision_keys = ["existence", "category"]
    if str(target.get("existence") or "").strip().lower() == "anomaly" and target.get("anomaly_interval_sec") is not None:
        decision_keys.append("temporal")
    if _target_requires_counterfactual_type(target, reward_version=reward_version):
        decision_keys.append("counterfactual_type")
    decision_scores = [_safe_float((fields.get(key) or {}).get("score"), 0.0) for key in decision_keys]
    decision_field_support = (
        sum(decision_scores) / float(len(decision_scores))
        if decision_scores
        else 0.0
    )
    required_stages = normalize_event_chain_stages(infer_required_stages_from_target(target))
    if not required_stages:
        return round(_clamp(decision_field_support), 6)
    stage_scores = [_safe_float((fields.get(stage) or {}).get("score"), 0.0) for stage in required_stages]
    stage_text_support = sum(stage_scores) / float(len(stage_scores)) if stage_scores else 0.0
    return round(_clamp(0.75 * decision_field_support + 0.25 * stage_text_support), 6)


def _timesearch_trigger_necessity_score_v2(
    profile: Dict[str, Any],
    *,
    target: Dict[str, Any],
    reward_version: str = "timesearch_v3",
) -> float:
    branch_delta_matrix = dict(profile.get("branch_delta_matrix") or {})
    drop_trigger_delta = dict((branch_delta_matrix.get("drop_trigger") or {}).get("fields") or {})
    if not drop_trigger_delta:
        return 0.0
    required_keys = ["existence", "category"]
    if str(target.get("existence") or "").strip().lower() == "anomaly" and target.get("anomaly_interval_sec") is not None:
        required_keys.append("temporal")
    if _target_requires_counterfactual_type(target, reward_version=reward_version):
        required_keys.append("counterfactual_type")
    delta_scores = [_safe_float(drop_trigger_delta.get(key), 0.0) for key in required_keys]
    if not delta_scores:
        return 0.0
    return round(_clamp(max(delta_scores)), 6)


def _timesearch_stage_coverage_score(
    profile: Dict[str, Any],
    *,
    target: Dict[str, Any],
    rollout_trace: Optional[Dict[str, Any]] = None,
) -> float:
    required_stages = normalize_event_chain_stages(infer_required_stages_from_target(target))
    if not required_stages:
        return 1.0
    branch_field_matrix = dict(profile.get("branch_field_matrix") or {})
    full_selected = dict(branch_field_matrix.get("full_selected") or {})
    supported_stages = set(normalize_event_chain_stages(full_selected.get("supported_stages") or []))
    missing_required_stages = set(normalize_event_chain_stages(full_selected.get("missing_required_stages") or []))
    if not supported_stages and rollout_trace is not None:
        verifier_turn = _latest_verifier_turn(rollout_trace)
        supported_stages = set(normalize_event_chain_stages((verifier_turn or {}).get("covered_stages") or []))
        missing_required_stages = set(
            normalize_event_chain_stages((verifier_turn or {}).get("missing_required_stages") or [])
        )
    if supported_stages or missing_required_stages:
        covered_count = len(set(required_stages) - missing_required_stages)
        if supported_stages:
            covered_count = max(covered_count, len(set(required_stages) & supported_stages))
        return round(_clamp(float(covered_count) / float(len(required_stages))), 6)
    fields = dict(full_selected.get("fields") or {})
    covered_count = sum(
        1
        for stage in required_stages
        if _safe_float((fields.get(stage) or {}).get("score"), 0.0) >= _COUNTERFACTUAL_STAGE_TEXT_THRESHOLD
    )
    return round(_clamp(float(covered_count) / float(len(required_stages))), 6)


def _timesearch_verifier_trace_score(
    profile: Dict[str, Any],
    *,
    rollout_trace: Optional[Dict[str, Any]] = None,
) -> float:
    if rollout_trace is not None:
        verifier_turn = _latest_verifier_turn(rollout_trace)
        if verifier_turn is not None:
            derived_scores = dict(verifier_turn.get("verifier_derived_scores") or {})
            return _verifier_trace_score_from_payload(derived_scores, default=0.5)
    branch_field_matrix = dict(profile.get("branch_field_matrix") or {})
    full_selected = dict(branch_field_matrix.get("full_selected") or {})
    if bool(full_selected.get("available")):
        finalize_readiness = _safe_float(
            (((full_selected.get("fields") or {}).get("finalize_readiness") or {}).get("score")),
            0.5,
        )
        return round(_clamp(finalize_readiness), 6)
    return 0.0


def _timesearch_fecv_diagnostics(
    profile: Dict[str, Any],
    *,
    target: Dict[str, Any],
    rollout_trace: Optional[Dict[str, Any]] = None,
    reward_version: str = "timesearch_v3",
) -> Dict[str, Any]:
    use_counterfactual_type = _reward_uses_counterfactual_type(reward_version)
    branch_field_matrix = dict(profile.get("branch_field_matrix") or {})
    branch_delta_matrix = dict(profile.get("branch_delta_matrix") or {})
    profile_source = str(
        profile.get("counterfactual_profile_source") or profile.get("counterfactual_branch_profile") or ""
    ).strip().lower()
    selection_metadata = dict(profile.get("selection_metadata") or {})
    normalized_branch_profile = str(
        selection_metadata.get("normalized_branch_profile") or profile_source
    ).strip().lower()
    summary = _counterfactual_summary(profile)
    stage_packages = dict(profile.get("stage_packages") or {})
    selected_window_ids = list(stage_packages.get("selected_window_ids") or [])
    selected_by_stage = dict(stage_packages.get("selected_by_stage") or {})
    full_selected_window_ids = list(
        selection_metadata.get("full_selected_window_ids")
        or ((branch_field_matrix.get("full_selected") or {}).get("window_ids") or [])
    )
    full_selected_unavailable_reason = str(
        selection_metadata.get("full_selected_unavailable_reason")
        or ((profile.get("counterfactual_branches") or {}).get("full_selected") or {}).get("unavailable_reason")
        or ""
    )
    selected_window_count = int(selection_metadata.get("selected_window_count") or len(selected_window_ids))
    selected_record_count = int(selection_metadata.get("selected_record_count") or selected_window_count)
    hard_negative_reason = str(selection_metadata.get("hard_negative_reason") or "")
    full_selected_parse_mode = str(selection_metadata.get("full_selected_parse_mode") or "")
    selection_resolution_source = str(selection_metadata.get("selection_resolution_source") or "")
    recovered_from_trace = bool(selection_metadata.get("recovered_from_trace", False))
    normal_reward_mode = ""
    normal_case_type = ""
    easy_normal_sample_loss_multiplier = 1.0
    normal_evidence_tool_turn_count = 0
    normal_search_restraint_score = 0.0
    normal_window_restraint_score = 0.0
    normal_verification_consistency_score = 0.0
    normal_query_alignment_score = 0.0
    normal_continuous_verifier_score = 0.0
    normal_verifier_primary_status = "unknown"
    normal_verifier_recommended_action = "unknown"
    normal_verifier_base_status_score = 0.0
    normal_verifier_action_offset = 0.0
    normal_continuous_verifier_score_before_action = 0.0
    normal_continuous_verifier_score_after_action = 0.0
    normal_verifier_trace_score = 0.0
    normal_selected_duration_ratio = 0.0
    normal_grounded_local_mode = ""
    normal_grounded_local_score = 0.0
    normal_provenance_score = 0.0
    normal_provenance_source_bucket = ""
    normal_restraint_reward = 0.0

    if normalized_branch_profile == NORMAL_SKIP_PROFILE:
        if rollout_trace is not None:
            normal_reward = _normal_skip_restraint_reward(
                rollout_trace,
                target=target,
                selected_window_count=selected_window_count,
                selected_window_ids=selected_window_ids,
                selected_by_stage=selected_by_stage,
                selection_resolution_source=selection_resolution_source,
                use_counterfactual_type=use_counterfactual_type,
            )
            normal_reward_mode = str(normal_reward.get("normal_reward_mode") or "")
            normal_case_type = str(normal_reward.get("normal_case_type") or "")
            easy_normal_sample_loss_multiplier = float(
                normal_reward.get("easy_normal_sample_loss_multiplier") or 1.0
            )
            normal_evidence_tool_turn_count = int(normal_reward.get("normal_evidence_tool_turn_count") or 0)
            normal_search_restraint_score = float(normal_reward.get("normal_search_restraint_score") or 0.0)
            normal_window_restraint_score = float(normal_reward.get("normal_window_restraint_score") or 0.0)
            normal_verification_consistency_score = float(
                normal_reward.get("normal_verification_consistency_score") or 0.0
            )
            normal_query_alignment_score = float(normal_reward.get("normal_query_alignment_score") or 0.0)
            normal_continuous_verifier_score = float(
                normal_reward.get("normal_continuous_verifier_score") or 0.0
            )
            normal_verifier_primary_status = str(normal_reward.get("normal_verifier_primary_status") or "unknown")
            normal_verifier_recommended_action = str(
                normal_reward.get("normal_verifier_recommended_action") or "unknown"
            )
            normal_verifier_base_status_score = float(
                normal_reward.get("normal_verifier_base_status_score") or 0.0
            )
            normal_verifier_action_offset = float(normal_reward.get("normal_verifier_action_offset") or 0.0)
            normal_continuous_verifier_score_before_action = float(
                normal_reward.get("normal_continuous_verifier_score_before_action") or 0.0
            )
            normal_continuous_verifier_score_after_action = float(
                normal_reward.get("normal_continuous_verifier_score_after_action") or 0.0
            )
            normal_verifier_trace_score = float(normal_reward.get("normal_verifier_trace_score") or 0.0)
            normal_selected_duration_ratio = float(normal_reward.get("normal_selected_duration_ratio") or 0.0)
            normal_grounded_local_mode = str(normal_reward.get("normal_grounded_local_mode") or "")
            normal_grounded_local_score = float(normal_reward.get("normal_grounded_local_score") or 0.0)
            normal_provenance_score = float(normal_reward.get("normal_provenance_score") or 0.0)
            normal_provenance_source_bucket = str(
                normal_reward.get("normal_provenance_source_bucket") or ""
            )
            normal_restraint_reward = float(normal_reward.get("normal_restraint_reward") or 0.0)
        return {
            "branch_profile": normalized_branch_profile,
            "profile_source": profile_source,
            "full_selected_available": False,
            "decision_field_scores": {},
            "required_stage_scores": {},
            "selected_support_score": 0.0,
            "trigger_necessity_delta": 0.0,
            "negative_resistance_delta": 0.0,
            "minimal_subset_parsimony_bonus": 0.0,
            "oracle_required_stage_coverage_score": 0.0,
            "evidence_faithfulness_reward": round(float(normal_restraint_reward), 6),
            "selected_window_count": selected_window_count,
            "selected_record_count": selected_record_count,
            "selected_window_ids": selected_window_ids,
            "selected_by_stage": selected_by_stage,
            "full_selected_window_ids": full_selected_window_ids,
            "full_selected_unavailable_reason": full_selected_unavailable_reason,
            "hard_negative_reason": hard_negative_reason,
            "full_selected_parse_mode": full_selected_parse_mode,
            "selection_resolution_source": selection_resolution_source,
            "recovered_from_trace": recovered_from_trace,
            "normal_reward_mode": normal_reward_mode,
            "normal_case_type": normal_case_type,
            "easy_normal_sample_loss_multiplier": round(float(easy_normal_sample_loss_multiplier), 6),
            "normal_evidence_tool_turn_count": int(normal_evidence_tool_turn_count),
            "normal_search_restraint_score": round(float(normal_search_restraint_score), 6),
            "normal_window_restraint_score": round(float(normal_window_restraint_score), 6),
            "normal_verification_consistency_score": round(float(normal_verification_consistency_score), 6),
            "normal_query_alignment_score": round(float(normal_query_alignment_score), 6),
            "normal_continuous_verifier_score": round(float(normal_continuous_verifier_score), 6),
            "normal_verifier_primary_status": normal_verifier_primary_status,
            "normal_verifier_recommended_action": normal_verifier_recommended_action,
            "normal_verifier_base_status_score": round(float(normal_verifier_base_status_score), 6),
            "normal_verifier_action_offset": round(float(normal_verifier_action_offset), 6),
            "normal_continuous_verifier_score_before_action": round(
                float(normal_continuous_verifier_score_before_action),
                6,
            ),
            "normal_continuous_verifier_score_after_action": round(
                float(normal_continuous_verifier_score_after_action),
                6,
            ),
            "normal_verifier_trace_score": round(float(normal_verifier_trace_score), 6),
            "normal_selected_duration_ratio": round(float(normal_selected_duration_ratio), 6),
            "normal_grounded_local_mode": normal_grounded_local_mode,
            "normal_grounded_local_score": round(float(normal_grounded_local_score), 6),
            "normal_provenance_score": round(float(normal_provenance_score), 6),
            "normal_provenance_source_bucket": normal_provenance_source_bucket,
            "normal_restraint_reward": round(float(normal_restraint_reward), 6),
            "selected_support_mode": "",
            "trigger_necessity_mode": "",
            "verifier_trace_score": 0.0,
            "stage_coverage_score": 0.0,
        }

    if normalized_branch_profile == "structured_oracle_v1":
        selected_support = _safe_float(summary.get("oracle_selected_support_score"), 0.0)
        required_stage_coverage = _safe_float(summary.get("oracle_required_stage_coverage_score"), 0.0)
        trigger_necessity = _safe_float(summary.get("oracle_drop_trigger_necessity_score"), 0.0)
        reward = round(max(0.0, min(1.0, 0.6 * selected_support + 0.25 * required_stage_coverage + 0.15 * trigger_necessity)), 6)
        return {
            "branch_profile": normalized_branch_profile,
            "profile_source": profile_source,
            "full_selected_available": bool(((branch_field_matrix.get("full_selected") or {}).get("available"))),
            "decision_field_scores": {},
            "required_stage_scores": {},
            "selected_support_score": round(float(selected_support), 6),
            "trigger_necessity_delta": round(float(trigger_necessity), 6),
            "negative_resistance_delta": 0.0,
            "minimal_subset_parsimony_bonus": 0.0,
            "oracle_required_stage_coverage_score": round(float(required_stage_coverage), 6),
            "evidence_faithfulness_reward": reward,
            "selected_window_count": selected_window_count,
            "selected_record_count": selected_record_count,
            "selected_window_ids": selected_window_ids,
            "selected_by_stage": selected_by_stage,
            "full_selected_window_ids": full_selected_window_ids,
            "full_selected_unavailable_reason": full_selected_unavailable_reason,
            "hard_negative_reason": hard_negative_reason,
            "full_selected_parse_mode": full_selected_parse_mode,
            "selection_resolution_source": selection_resolution_source,
            "recovered_from_trace": recovered_from_trace,
            "normal_reward_mode": normal_reward_mode,
            "normal_case_type": normal_case_type,
            "easy_normal_sample_loss_multiplier": round(float(easy_normal_sample_loss_multiplier), 6),
            "normal_evidence_tool_turn_count": int(normal_evidence_tool_turn_count),
            "normal_search_restraint_score": round(float(normal_search_restraint_score), 6),
            "normal_window_restraint_score": round(float(normal_window_restraint_score), 6),
            "normal_verification_consistency_score": round(float(normal_verification_consistency_score), 6),
            "normal_query_alignment_score": round(float(normal_query_alignment_score), 6),
            "normal_continuous_verifier_score": round(float(normal_continuous_verifier_score), 6),
            "normal_verifier_primary_status": normal_verifier_primary_status,
            "normal_verifier_recommended_action": normal_verifier_recommended_action,
            "normal_verifier_base_status_score": round(float(normal_verifier_base_status_score), 6),
            "normal_verifier_action_offset": round(float(normal_verifier_action_offset), 6),
            "normal_continuous_verifier_score_before_action": round(
                float(normal_continuous_verifier_score_before_action),
                6,
            ),
            "normal_continuous_verifier_score_after_action": round(
                float(normal_continuous_verifier_score_after_action),
                6,
            ),
            "normal_verifier_trace_score": round(float(normal_verifier_trace_score), 6),
            "normal_selected_duration_ratio": round(float(normal_selected_duration_ratio), 6),
            "normal_grounded_local_mode": normal_grounded_local_mode,
            "normal_grounded_local_score": round(float(normal_grounded_local_score), 6),
            "normal_provenance_score": round(float(normal_provenance_score), 6),
            "normal_provenance_source_bucket": normal_provenance_source_bucket,
            "normal_restraint_reward": round(float(normal_restraint_reward), 6),
            "selected_support_mode": "",
            "trigger_necessity_mode": "",
            "verifier_trace_score": 0.0,
            "stage_coverage_score": 0.0,
        }

    full_selected = dict(branch_field_matrix.get("full_selected") or {})
    fields = dict(full_selected.get("fields") or {})
    decision_keys = ["existence", "category"]
    if str(target.get("existence") or "").strip().lower() == "anomaly" and target.get("anomaly_interval_sec") is not None:
        decision_keys.append("temporal")
    if _target_requires_counterfactual_type(target, reward_version=reward_version):
        decision_keys.append("counterfactual_type")
    decision_field_scores = {
        key: round(float(_safe_float((fields.get(key) or {}).get("score"), 0.0)), 6)
        for key in decision_keys
        if key in fields
    }
    required_stages = normalize_event_chain_stages(infer_required_stages_from_target(target))
    required_stage_scores = {
        stage: round(float(_safe_float((fields.get(stage) or {}).get("score"), 0.0)), 6)
        for stage in required_stages
        if stage in fields
    }
    selected_support = _timesearch_selected_support_score(
        profile,
        target=target,
        reward_version=reward_version,
    )
    selected_support_v2 = _timesearch_selected_support_score_v2(
        profile,
        target=target,
        reward_version=reward_version,
    )
    drop_trigger_delta = dict((branch_delta_matrix.get("drop_trigger") or {}).get("fields") or {})
    trigger_necessity_v1 = max(
        _safe_float(drop_trigger_delta.get("existence"), 0.0),
        _safe_float(drop_trigger_delta.get("category"), 0.0),
    ) if drop_trigger_delta else 0.0
    trigger_necessity_v2 = _timesearch_trigger_necessity_score_v2(
        profile,
        target=target,
        reward_version=reward_version,
    )
    stage_coverage_score = _timesearch_stage_coverage_score(
        profile,
        target=target,
        rollout_trace=rollout_trace,
    )
    verifier_trace_score = _timesearch_verifier_trace_score(
        profile,
        rollout_trace=rollout_trace,
    )

    hard_neg_delta = dict((branch_delta_matrix.get("hard_negative_swap") or {}).get("fields") or {})
    negative_resistance = max(
        _safe_float(hard_neg_delta.get("existence"), 0.0),
        _safe_float(hard_neg_delta.get("category"), 0.0),
    ) if hard_neg_delta else 0.0
    full_ids = ((branch_field_matrix.get("full_selected") or {}).get("window_ids") or [])
    minimal_ids = ((branch_field_matrix.get("minimal_subset") or {}).get("window_ids") or [])
    parsimony_bonus = 1.0 - (float(len(minimal_ids)) / float(len(full_ids))) if full_ids and minimal_ids else 0.0

    online_core = normalized_branch_profile == "online_core"
    trigger_necessity = trigger_necessity_v2 if online_core else trigger_necessity_v1
    if online_core:
        reward = (
            0.40 * selected_support_v2
            + 0.20 * trigger_necessity
            + 0.15 * verifier_trace_score
            + 0.15 * stage_coverage_score
            + 0.10 * parsimony_bonus
        )
    else:
        reward = 0.5 * selected_support + 0.2 * trigger_necessity + 0.2 * negative_resistance + 0.1 * parsimony_bonus
    reward = round(max(0.0, min(1.0, reward)), 6)

    return {
        "branch_profile": normalized_branch_profile,
        "profile_source": profile_source,
        "full_selected_available": bool(full_selected.get("available")),
        "decision_field_scores": decision_field_scores,
        "required_stage_scores": required_stage_scores,
        "selected_support_score": round(float(selected_support_v2 if online_core else selected_support), 6),
        "trigger_necessity_delta": round(float(trigger_necessity), 6),
        "negative_resistance_delta": round(float(negative_resistance), 6),
        "minimal_subset_parsimony_bonus": round(float(parsimony_bonus), 6),
        "oracle_required_stage_coverage_score": round(float(stage_coverage_score if online_core else 0.0), 6),
        "evidence_faithfulness_reward": reward,
        "selected_window_count": selected_window_count,
        "selected_record_count": selected_record_count,
        "selected_window_ids": selected_window_ids,
        "selected_by_stage": selected_by_stage,
        "full_selected_window_ids": full_selected_window_ids,
        "full_selected_unavailable_reason": full_selected_unavailable_reason,
        "hard_negative_reason": hard_negative_reason,
        "full_selected_parse_mode": full_selected_parse_mode,
        "selection_resolution_source": selection_resolution_source,
        "recovered_from_trace": recovered_from_trace,
        "normal_reward_mode": normal_reward_mode,
        "normal_case_type": normal_case_type,
        "easy_normal_sample_loss_multiplier": round(float(easy_normal_sample_loss_multiplier), 6),
        "normal_evidence_tool_turn_count": int(normal_evidence_tool_turn_count),
        "normal_search_restraint_score": round(float(normal_search_restraint_score), 6),
        "normal_window_restraint_score": round(float(normal_window_restraint_score), 6),
        "normal_verification_consistency_score": round(float(normal_verification_consistency_score), 6),
        "normal_query_alignment_score": round(float(normal_query_alignment_score), 6),
        "normal_continuous_verifier_score": round(float(normal_continuous_verifier_score), 6),
        "normal_verifier_primary_status": normal_verifier_primary_status,
        "normal_verifier_recommended_action": normal_verifier_recommended_action,
        "normal_verifier_base_status_score": round(float(normal_verifier_base_status_score), 6),
        "normal_verifier_action_offset": round(float(normal_verifier_action_offset), 6),
        "normal_continuous_verifier_score_before_action": round(
            float(normal_continuous_verifier_score_before_action),
            6,
        ),
        "normal_continuous_verifier_score_after_action": round(
            float(normal_continuous_verifier_score_after_action),
            6,
        ),
        "normal_verifier_trace_score": round(float(normal_verifier_trace_score), 6),
        "normal_selected_duration_ratio": round(float(normal_selected_duration_ratio), 6),
        "normal_grounded_local_mode": normal_grounded_local_mode,
        "normal_grounded_local_score": round(float(normal_grounded_local_score), 6),
        "normal_provenance_score": round(float(normal_provenance_score), 6),
        "normal_provenance_source_bucket": normal_provenance_source_bucket,
        "normal_restraint_reward": round(float(normal_restraint_reward), 6),
        "selected_support_mode": "selected_support_v2" if online_core else "selected_support_v1",
        "trigger_necessity_mode": "trigger_necessity_v2" if online_core else "trigger_necessity_v1",
        "verifier_trace_score": round(float(verifier_trace_score if online_core else 0.0), 6),
        "stage_coverage_score": round(float(stage_coverage_score if online_core else 0.0), 6),
    }


def _timesearch_fecv_reward(
    profile: Dict[str, Any],
    *,
    target: Dict[str, Any],
    rollout_trace: Optional[Dict[str, Any]] = None,
    reward_version: str = "timesearch_v3",
) -> float:
    """Continuous FECV reward.

    Online profiles keep hard_negative_swap disabled, but now include the
    replay-guided minimal_subset parsimony term alongside full_selected and
    drop_trigger to preserve the same local closed loop.
    """
    if not profile:
        return 0.0
    diagnostics = _timesearch_fecv_diagnostics(
        profile,
        target=target,
        rollout_trace=rollout_trace,
        reward_version=reward_version,
    )
    return float(diagnostics.get("evidence_faithfulness_reward") or 0.0)


def _protocol_finalize_reward_timesearch(
    rollout_trace: Dict[str, Any],
    *,
    verifier_turn: Optional[Dict[str, Any]],
    target: Dict[str, Any],
    fecv_diagnostics: Dict[str, Any],
) -> float:
    if (
        str(target.get("existence") or "").strip().lower() == "anomaly"
        and str(fecv_diagnostics.get("full_selected_unavailable_reason") or "") == "contract_violation_empty_selection"
    ):
        return -1.0
    return float(_protocol_finalize_reward(rollout_trace, verifier_turn))


def _score_rollout_trace_timesearch(
    rollout_trace: Dict[str, Any],
    *,
    reward_version: str = "timesearch_v1",
    weights: Optional[Dict[str, float]] = None,
    reward_config: Optional[Dict[str, Any]] = None,
    llm_judge: Optional[OpenAICompatibleLlmJudge] = None,
) -> Dict[str, Any]:
    normalized_reward_version = _normalize_reward_version(reward_version)
    normalized_weights = _normalize_component_weights(
        weights or dict((reward_config or {}).get("weights") or {}),
        reward_version=normalized_reward_version,
    )
    verifier_turn = _latest_verifier_turn(rollout_trace)
    target = _infer_target(rollout_trace)
    final_prediction = _infer_final_prediction(rollout_trace)
    profile = _extract_counterfactual_profile(rollout_trace)
    profile_summary = _counterfactual_summary(profile)
    accuracy = _compute_accuracy_breakdown(
        rollout_trace,
        llm_judge=llm_judge or OpenAICompatibleLlmJudge(),
        reward_version=normalized_reward_version,
    )
    use_counterfactual_type = _reward_uses_counterfactual_type(normalized_reward_version)

    final_decision_correct = 1.0 if _decision_matches(
        final_prediction,
        target,
        use_counterfactual_type=use_counterfactual_type,
    ) else 0.0
    decision_sufficiency = 1.0 if bool(profile_summary.get("decision_sufficiency")) else 0.0
    minimal_subset_sufficiency = 1.0 if bool(profile_summary.get("minimal_subset_sufficiency")) else 0.0
    negative_specificity_pass = 1.0 if bool(profile_summary.get("negative_specificity_pass")) else 0.0
    fecv_full_selected_available = bool(((profile.get("branch_field_matrix") or {}).get("full_selected") or {}).get("available"))
    fecv_grounded_decision = 1.0 if final_decision_correct > 0.0 and decision_sufficiency > 0.0 else 0.0
    legacy_fecv_decision_reward = float(_fecv_decision_sufficiency_reward(profile))
    legacy_fecv_specificity_reward = float(
        _fecv_specificity_reward(
            profile,
            target=target,
            reward_version=normalized_reward_version,
        )
    )
    fecv_diagnostics = _timesearch_fecv_diagnostics(
        profile,
        target=target,
        rollout_trace=rollout_trace,
        reward_version=normalized_reward_version,
    )
    if not bool(fecv_diagnostics.get("full_selected_available")):
        logger.info(
            "fecv reward availability debug: video_id=%s generation_id=%s branch_profile=%s profile_source=%s "
            "full_selected_available=%s reason=%s parse_mode=%s selected_window_count=%s selected_record_count=%s "
            "selected_window_ids=%s full_selected_window_ids=%s selected_by_stage=%s hard_negative_reason=%s "
            "selection_resolution_source=%s recovered_from_trace=%s "
            "normal_reward_mode=%s normal_case_type=%s easy_normal_sample_loss_multiplier=%s "
            "normal_evidence_tool_turn_count=%s "
            "normal_search_restraint_score=%s normal_window_restraint_score=%s "
            "normal_verification_consistency_score=%s normal_query_alignment_score=%s "
            "normal_continuous_verifier_score=%s normal_verifier_primary_status=%s "
            "normal_verifier_recommended_action=%s normal_verifier_base_status_score=%s "
            "normal_verifier_action_offset=%s normal_continuous_verifier_score_before_action=%s "
            "normal_continuous_verifier_score_after_action=%s normal_grounded_local_score=%s "
            "normal_provenance_score=%s normal_provenance_source_bucket=%s normal_restraint_reward=%s",
            str(rollout_trace.get("video_id") or ""),
            str(rollout_trace.get("generation_id") or ""),
            str(fecv_diagnostics.get("branch_profile") or ""),
            str(fecv_diagnostics.get("profile_source") or ""),
            bool(fecv_diagnostics.get("full_selected_available")),
            str(fecv_diagnostics.get("full_selected_unavailable_reason") or ""),
            str(fecv_diagnostics.get("full_selected_parse_mode") or ""),
            int(fecv_diagnostics.get("selected_window_count") or 0),
            int(fecv_diagnostics.get("selected_record_count") or 0),
            list(fecv_diagnostics.get("selected_window_ids") or []),
            list(fecv_diagnostics.get("full_selected_window_ids") or []),
            dict(fecv_diagnostics.get("selected_by_stage") or {}),
            str(fecv_diagnostics.get("hard_negative_reason") or ""),
            str(fecv_diagnostics.get("selection_resolution_source") or ""),
            bool(fecv_diagnostics.get("recovered_from_trace")),
            str(fecv_diagnostics.get("normal_reward_mode") or ""),
            str(fecv_diagnostics.get("normal_case_type") or ""),
            round(float(fecv_diagnostics.get("easy_normal_sample_loss_multiplier") or 1.0), 6),
            int(fecv_diagnostics.get("normal_evidence_tool_turn_count") or 0),
            round(float(fecv_diagnostics.get("normal_search_restraint_score") or 0.0), 6),
            round(float(fecv_diagnostics.get("normal_window_restraint_score") or 0.0), 6),
            round(float(fecv_diagnostics.get("normal_verification_consistency_score") or 0.0), 6),
            round(float(fecv_diagnostics.get("normal_query_alignment_score") or 0.0), 6),
            round(float(fecv_diagnostics.get("normal_continuous_verifier_score") or 0.0), 6),
            str(fecv_diagnostics.get("normal_verifier_primary_status") or "unknown"),
            str(fecv_diagnostics.get("normal_verifier_recommended_action") or "unknown"),
            round(float(fecv_diagnostics.get("normal_verifier_base_status_score") or 0.0), 6),
            round(float(fecv_diagnostics.get("normal_verifier_action_offset") or 0.0), 6),
            round(float(fecv_diagnostics.get("normal_continuous_verifier_score_before_action") or 0.0), 6),
            round(float(fecv_diagnostics.get("normal_continuous_verifier_score_after_action") or 0.0), 6),
            round(float(fecv_diagnostics.get("normal_grounded_local_score") or 0.0), 6),
            round(float(fecv_diagnostics.get("normal_provenance_score") or 0.0), 6),
            str(fecv_diagnostics.get("normal_provenance_source_bucket") or ""),
            round(float(fecv_diagnostics.get("normal_restraint_reward") or 0.0), 6),
        )

    protocol_finalize_reward = _protocol_finalize_reward_timesearch(
        rollout_trace,
        verifier_turn=verifier_turn,
        target=target,
        fecv_diagnostics=fecv_diagnostics,
    )
    components = {
        "accuracy_reward": float(accuracy["accuracy_reward"]),
        "fecv_evidence_faithfulness_reward": float(fecv_diagnostics.get("evidence_faithfulness_reward") or 0.0),
        "protocol_finalize_reward": float(protocol_finalize_reward),
        "fecv_decision_sufficiency_reward": float(legacy_fecv_decision_reward),
        "fecv_specificity_reward": float(legacy_fecv_specificity_reward),
    }
    weighted_components = {
        key: round(float(normalized_weights.get(key, 0.0)) * float(value), 6)
        for key, value in components.items()
    }

    total_reward = 0.0
    for key, weight in normalized_weights.items():
        total_reward += float(weight) * float(components.get(key, 0.0))

    return {
        "reward_version": str(normalized_reward_version),
        "total_reward": round(float(total_reward), 6),
        "components": {key: round(float(value), 6) for key, value in components.items()},
        "weighted_components": dict(weighted_components),
        "weights": dict(normalized_weights),
        "final_decision_correct": float(final_decision_correct),
        "accuracy_reward": float(accuracy["accuracy_reward"]),
        "accuracy_by_family": dict(accuracy["accuracy_by_family"]),
        "accuracy_by_type": dict(accuracy["accuracy_by_type"]),
        "accuracy_question_count": int(accuracy["accuracy_question_count"]),
        "counterfactual_sufficiency_reward": round(float(legacy_fecv_decision_reward), 6),
        "stage_necessity_reward": round(float(_stage_necessity_reward(rollout_trace, verifier_turn=verifier_turn, profile=profile)), 6),
        "query_alignment_reward": round(float(_query_alignment_reward(rollout_trace)), 6),
        "fecv_branch_profile": str(fecv_diagnostics.get("branch_profile") or ""),
        "fecv_profile_source": str(fecv_diagnostics.get("profile_source") or ""),
        "fecv_full_selected_available": bool(fecv_full_selected_available),
        "fecv_full_selected_parse_mode": str(fecv_diagnostics.get("full_selected_parse_mode") or ""),
        "fecv_full_selected_unavailable_reason": str(
            fecv_diagnostics.get("full_selected_unavailable_reason") or ""
        ),
        "fecv_selected_window_count": int(fecv_diagnostics.get("selected_window_count") or 0),
        "fecv_selected_record_count": int(fecv_diagnostics.get("selected_record_count") or 0),
        "fecv_selected_window_ids": list(fecv_diagnostics.get("selected_window_ids") or []),
        "fecv_full_selected_window_ids": list(fecv_diagnostics.get("full_selected_window_ids") or []),
        "fecv_selected_by_stage": dict(fecv_diagnostics.get("selected_by_stage") or {}),
        "fecv_hard_negative_reason": str(fecv_diagnostics.get("hard_negative_reason") or ""),
        "fecv_selection_resolution_source": str(fecv_diagnostics.get("selection_resolution_source") or ""),
        "fecv_recovered_from_trace": bool(fecv_diagnostics.get("recovered_from_trace")),
        "fecv_selected_support_score": round(float(fecv_diagnostics.get("selected_support_score") or 0.0), 6),
        "fecv_selected_support_mode": str(fecv_diagnostics.get("selected_support_mode") or ""),
        "fecv_trigger_necessity_delta": round(float(fecv_diagnostics.get("trigger_necessity_delta") or 0.0), 6),
        "fecv_trigger_necessity_mode": str(fecv_diagnostics.get("trigger_necessity_mode") or ""),
        "fecv_negative_resistance_delta": round(float(fecv_diagnostics.get("negative_resistance_delta") or 0.0), 6),
        "fecv_verifier_trace_score": round(float(fecv_diagnostics.get("verifier_trace_score") or 0.0), 6),
        "fecv_stage_coverage_score": round(float(fecv_diagnostics.get("stage_coverage_score") or 0.0), 6),
        "fecv_minimal_subset_parsimony_bonus": round(
            float(fecv_diagnostics.get("minimal_subset_parsimony_bonus") or 0.0),
            6,
        ),
        "fecv_oracle_required_stage_coverage_score": round(
            float(fecv_diagnostics.get("oracle_required_stage_coverage_score") or 0.0),
            6,
        ),
        "fecv_normal_reward_mode": str(fecv_diagnostics.get("normal_reward_mode") or ""),
        "fecv_normal_case_type": str(fecv_diagnostics.get("normal_case_type") or ""),
        "fecv_easy_normal_sample_loss_multiplier": round(
            float(fecv_diagnostics.get("easy_normal_sample_loss_multiplier") or 1.0),
            6,
        ),
        "fecv_normal_evidence_tool_turn_count": int(
            fecv_diagnostics.get("normal_evidence_tool_turn_count") or 0
        ),
        "fecv_normal_search_restraint_score": round(
            float(fecv_diagnostics.get("normal_search_restraint_score") or 0.0),
            6,
        ),
        "fecv_normal_window_restraint_score": round(
            float(fecv_diagnostics.get("normal_window_restraint_score") or 0.0),
            6,
        ),
        "fecv_normal_verification_consistency_score": round(
            float(fecv_diagnostics.get("normal_verification_consistency_score") or 0.0),
            6,
        ),
        "fecv_normal_query_alignment_score": round(
            float(fecv_diagnostics.get("normal_query_alignment_score") or 0.0),
            6,
        ),
        "fecv_normal_continuous_verifier_score": round(
            float(fecv_diagnostics.get("normal_continuous_verifier_score") or 0.0),
            6,
        ),
        "fecv_normal_verifier_primary_status": str(
            fecv_diagnostics.get("normal_verifier_primary_status") or "unknown"
        ),
        "fecv_normal_verifier_recommended_action": str(
            fecv_diagnostics.get("normal_verifier_recommended_action") or "unknown"
        ),
        "fecv_normal_verifier_base_status_score": round(
            float(fecv_diagnostics.get("normal_verifier_base_status_score") or 0.0),
            6,
        ),
        "fecv_normal_verifier_action_offset": round(
            float(fecv_diagnostics.get("normal_verifier_action_offset") or 0.0),
            6,
        ),
        "fecv_normal_continuous_verifier_score_before_action": round(
            float(fecv_diagnostics.get("normal_continuous_verifier_score_before_action") or 0.0),
            6,
        ),
        "fecv_normal_continuous_verifier_score_after_action": round(
            float(fecv_diagnostics.get("normal_continuous_verifier_score_after_action") or 0.0),
            6,
        ),
        "fecv_normal_verifier_trace_score": round(
            float(fecv_diagnostics.get("normal_verifier_trace_score") or 0.0),
            6,
        ),
        "fecv_normal_selected_duration_ratio": round(
            float(fecv_diagnostics.get("normal_selected_duration_ratio") or 0.0),
            6,
        ),
        "fecv_normal_grounded_local_mode": str(
            fecv_diagnostics.get("normal_grounded_local_mode") or ""
        ),
        "fecv_normal_grounded_local_score": round(
            float(fecv_diagnostics.get("normal_grounded_local_score") or 0.0),
            6,
        ),
        "fecv_normal_provenance_score": round(
            float(fecv_diagnostics.get("normal_provenance_score") or 0.0),
            6,
        ),
        "fecv_normal_provenance_source_bucket": str(
            fecv_diagnostics.get("normal_provenance_source_bucket") or ""
        ),
        "fecv_normal_restraint_reward": round(
            float(fecv_diagnostics.get("normal_restraint_reward") or 0.0),
            6,
        ),
        "fecv_decision_field_scores": dict(fecv_diagnostics.get("decision_field_scores") or {}),
        "fecv_required_stage_scores": dict(fecv_diagnostics.get("required_stage_scores") or {}),
        "fecv_grounded_decision": float(fecv_grounded_decision),
        "fecv_decision_sufficiency": float(decision_sufficiency),
        "fecv_minimal_subset_sufficiency": float(minimal_subset_sufficiency),
        "fecv_negative_specificity_pass": float(negative_specificity_pass),
        "counterfactual_decision_sufficiency": float(decision_sufficiency),
        "counterfactual_minimal_subset_sufficiency": float(minimal_subset_sufficiency),
        "negative_specificity_pass": float(negative_specificity_pass),
        "latest_verifier_turn_present": verifier_turn is not None,
        "verifier_source": str(verifier_turn.get("_verifier_source")) if verifier_turn is not None else "none",
        "uses_reference_conditioned_verifier": bool(
            verifier_turn.get("_uses_reference_conditioned_verifier") if verifier_turn is not None else False
        ),
    }


def score_rollout_trace(
    rollout_trace: Dict[str, Any],
    *,
    weights: Optional[Dict[str, float]] = None,
    reward_version: str = "legacy",
    reward_config: Optional[Dict[str, Any]] = None,
    llm_judge: Optional[OpenAICompatibleLlmJudge] = None,
) -> Dict[str, Any]:
    resolved_reward_version = reward_version
    if isinstance(reward_config, dict) and str(reward_config.get("reward_version") or "").strip():
        resolved_reward_version = str(reward_config.get("reward_version"))
    resolved_reward_version = _normalize_reward_version(resolved_reward_version)
    if resolved_reward_version in {"timesearch_v1", "timesearch_v2", "timesearch_v3"}:
        return _score_rollout_trace_timesearch(
            rollout_trace,
            reward_version=resolved_reward_version,
            weights=weights,
            reward_config=reward_config,
            llm_judge=llm_judge,
        )
    return _score_rollout_trace_legacy(
        rollout_trace,
        weights=weights,
    )
