from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from saver_v3.core.categories import canonicalize_saver_category, normalize_existence
from saver_v3.core.event_chain import (
    compute_event_chain_score,
    compute_query_alignment_score,
    extract_stage_annotation_from_record,
    infer_required_stages_from_target,
)
from saver_v3.core.llm_judge import OpenAICompatibleLlmJudge
from saver_v3.core.semantic_answer import (
    SEMANTIC_EVENT_CHAIN_STAGES,
    build_public_semantic_replay_payload,
    normalize_semantic_answer_payload,
)
from saver_v3.data.runtime_contract import REMOVED_RUNTIME_FIELDS


DEFAULT_RL_REWARD_VERSION = "timesearch_v4"
DEFAULT_COMPONENT_WEIGHTS = {
    "accuracy_reward": 1.0,
    "protocol_finalize_reward": 0.10,
    "stage_necessity_reward": 0.15,
    "query_alignment_reward": 0.10,
    "efficiency_reward": 0.05,
    "anomaly_false_normal_penalty": 1.25,
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
_OPEN_ENDED_QUESTION_TYPES_V4: tuple[str, ...] = (
    "trigger_evidence",
    "summary",
) + tuple(f"event_chain_summary.{stage}" for stage in SEMANTIC_EVENT_CHAIN_STAGES)
SUPPORTED_REWARD_VERSIONS = {DEFAULT_RL_REWARD_VERSION}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(float(lower), min(float(upper), float(value)))


def _normalize_reward_version(value: Any) -> str:
    normalized = str(value or DEFAULT_RL_REWARD_VERSION).strip().lower()
    if normalized not in SUPPORTED_REWARD_VERSIONS:
        raise ValueError(
            f"Unsupported reward version: {value!r}. Active RL now supports only {DEFAULT_RL_REWARD_VERSION}."
        )
    return normalized


def resolve_reward_component_weights(
    *,
    reward_version: str = DEFAULT_RL_REWARD_VERSION,
    reward_config: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    resolved_reward_version = reward_version
    if isinstance(reward_config, dict) and str(reward_config.get("reward_version") or "").strip():
        resolved_reward_version = str(reward_config.get("reward_version"))
    _normalize_reward_version(resolved_reward_version)
    merged = dict(DEFAULT_COMPONENT_WEIGHTS)
    for key, value in dict(weights or {}).items():
        if key in merged:
            merged[str(key)] = float(value)
    for key, value in dict((reward_config or {}).get("weights") or {}).items():
        if key in merged:
            merged[str(key)] = float(value)
    return merged


def build_open_ended_reward_judge(
    *,
    reward_config: Optional[Dict[str, Any]] = None,
) -> OpenAICompatibleLlmJudge:
    del reward_config
    return OpenAICompatibleLlmJudge()


def _collect_semantic_queries(
    rollout_trace: Dict[str, Any],
    *,
    reward_version: str = DEFAULT_RL_REWARD_VERSION,
) -> Dict[str, Tuple[str, str, str]]:
    _normalize_reward_version(reward_version)
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
    queries: Dict[str, Tuple[str, str, str]] = {}
    for qa_type in _OPEN_ENDED_QUESTION_TYPES_V4:
        target_entry = open_targets.get(qa_type)
        if target_entry is None:
            continue
        question, reference = target_entry
        prediction = str(open_predictions.get(qa_type) or "").strip()
        if not prediction:
            continue
        queries[qa_type] = (str(question or ""), str(reference or ""), prediction)
    return queries


def _normalize_interval(interval: Sequence[float] | None) -> Optional[Tuple[float, float]]:
    if not interval or len(interval) != 2:
        return None
    start_sec = _safe_float(interval[0], 0.0)
    end_sec = _safe_float(interval[1], 0.0)
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec
    return start_sec, end_sec


def _interval_iou(interval_a: Sequence[float] | None, interval_b: Sequence[float] | None) -> float:
    a = _normalize_interval(interval_a)
    b = _normalize_interval(interval_b)
    if a is None or b is None:
        return 0.0
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    if overlap <= 0.0:
        return 0.0
    union = max(a[1] - a[0], 0.0) + max(b[1] - b[0], 0.0) - overlap
    if union <= 0.0:
        return 0.0
    return max(0.0, min(1.0, overlap / union))


def _normalize_decision_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    normalized = copy.deepcopy(payload)
    if "existence" in normalized:
        normalized["existence"] = normalize_existence(normalized.get("existence"))
    if "category" in normalized:
        normalized["category"] = canonicalize_saver_category(
            normalized.get("category"),
            existence=normalized.get("existence"),
        ) or str(normalized.get("category") or "").strip().lower()
    for field_name in REMOVED_RUNTIME_FIELDS:
        normalized.pop(str(field_name), None)
    return normalized


def _infer_target(rollout_trace: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("scoring_target", "structured_target", "target"):
        payload = rollout_trace.get(key)
        if isinstance(payload, dict) and payload:
            return _normalize_decision_payload(payload)
    return {}


def _infer_semantic_payload(rollout_trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    semantic_payload = normalize_semantic_answer_payload(rollout_trace.get("semantic_answer"))
    if semantic_payload is not None:
        return semantic_payload
    final_answer = rollout_trace.get("final_answer")
    if isinstance(final_answer, dict) and "decision" in final_answer:
        return normalize_semantic_answer_payload(final_answer)
    state = dict(rollout_trace.get("state") or {})
    semantic_answer = state.get("finalized_semantic_answer")
    if isinstance(semantic_answer, dict):
        return normalize_semantic_answer_payload(semantic_answer)
    final_prediction: Dict[str, Any] = {}
    for payload in (final_answer, state.get("finalized_case")):
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
    return None


def _infer_final_prediction(rollout_trace: Dict[str, Any]) -> Dict[str, Any]:
    state = rollout_trace.get("state") or {}
    for payload in (rollout_trace.get("final_answer"), state.get("finalized_case")):
        if isinstance(payload, dict) and payload:
            if "decision" in payload:
                normalized = normalize_semantic_answer_payload(payload)
                decision = dict((normalized or {}).get("decision") or {})
                if decision:
                    return _normalize_decision_payload(decision)
            return _normalize_decision_payload(payload)
    semantic_payload = _infer_semantic_payload(rollout_trace)
    decision = dict((semantic_payload or {}).get("decision") or {})
    return _normalize_decision_payload(decision)


def _decision_matches(prediction: Dict[str, Any], target: Dict[str, Any]) -> bool:
    if not prediction or not target:
        return False
    if normalize_existence(prediction.get("existence")) != normalize_existence(target.get("existence")):
        return False
    target_existence = normalize_existence(target.get("existence"))
    if target_existence == "anomaly":
        pred_category = canonicalize_saver_category(prediction.get("category"), existence="anomaly")
        target_category = canonicalize_saver_category(target.get("category"), existence="anomaly")
        if target_category and pred_category != target_category:
            return False
    return True


def _latest_verifier_turn(rollout_trace: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for turn in reversed(list(rollout_trace.get("turns") or [])):
        if str(turn.get("tool_name") or "") == "verify_hypothesis":
            copied = dict(turn)
            copied["_verifier_source"] = "online_turn"
            return copied
    offline_verifier = rollout_trace.get("offline_verifier")
    if isinstance(offline_verifier, dict):
        return {
            "tool_name": "verify_hypothesis",
            "verifier_primary_status": offline_verifier.get("primary_status"),
            "verifier_next_tool": offline_verifier.get("next_tool"),
            "verifier_derived_scores": offline_verifier.get("derived_scores") or {},
            "covered_stages": offline_verifier.get("covered_stages") or [],
            "missing_required_stages": offline_verifier.get("missing_required_stages") or [],
            "stage_selected_moment_ids": offline_verifier.get("stage_selected_moment_ids") or {},
            "_verifier_source": "offline_verifier",
        }
    return None


def _infer_scoring_qa_pairs(rollout_trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("scoring_qa_pairs", "qa_pairs"):
        value = rollout_trace.get(key)
        if isinstance(value, list):
            return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _infer_scoring_evidence_moments(rollout_trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = rollout_trace.get("scoring_evidence_moments")
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    evidence = rollout_trace.get("evidence")
    if isinstance(evidence, dict):
        moments = evidence.get("evidence_moments")
        if isinstance(moments, list):
            return [dict(item) for item in moments if isinstance(item, dict)]
    return []


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


def _parse_interval_from_text(text: Any, *, fps: float = 0.0) -> Optional[List[float]]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return None
    second_match = _SECOND_INTERVAL_RE.search(raw_text)
    if second_match is not None:
        return [_safe_float(second_match.group(1)), _safe_float(second_match.group(2))]
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
    trigger_pair = pair_map.get("trigger_evidence")
    trigger_target = str(
        (trigger_pair or {}).get("answer")
        or ((semantic_target.get("event_chain_summary") or {}).get("trigger"))
        or rationale
        or summary
        or ""
    ).strip()
    if trigger_target:
        question = str(
            (trigger_pair or {}).get("question")
            or "What visible evidence first makes the anomaly actionable?"
        ).strip()
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
    trigger_evidence = str(event_chain_summary.get("trigger") or rationale or summary or "").strip()
    if trigger_evidence:
        predictions["trigger_evidence"] = trigger_evidence
    for stage in SEMANTIC_EVENT_CHAIN_STAGES:
        text = str(event_chain_summary.get(stage) or "").strip()
        if text:
            predictions[f"event_chain_summary.{stage}"] = text
    return predictions


def _normal_verification_consistency_score(rollout_trace: Dict[str, Any]) -> float:
    verifier_turn = _latest_verifier_turn(rollout_trace)
    if verifier_turn is None:
        return 1.0
    failure_reasons = list(verifier_turn.get("verifier_failure_reasons") or verifier_turn.get("failure_reasons") or [])
    invalid_selected_ids = list(verifier_turn.get("invalid_selected_window_ids") or [])
    next_tool = str(
        verifier_turn.get("verifier_next_tool")
        or verifier_turn.get("next_tool")
        or ""
    ).strip().lower()
    if not failure_reasons and not invalid_selected_ids:
        if next_tool == "finalize_case":
            return 1.0
        if next_tool not in {"seek_evidence"}:
            return 0.6
    return 0.2


def _normal_continuous_verifier_score_details(rollout_trace: Dict[str, Any]) -> Dict[str, Any]:
    verifier_turn = _latest_verifier_turn(rollout_trace)
    if verifier_turn is None:
        return {
            "primary_status": "unknown",
            "next_tool": "unknown",
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
    next_tool = str(
        verifier_turn.get("verifier_next_tool")
        or verifier_turn.get("next_tool")
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
    normalized_action = next_tool or "unknown"
    base_status_score = float(status_scores.get(normalized_status, 0.5))
    action_offset = 0.0
    score_after_action = _clamp(base_status_score)
    return {
        "primary_status": normalized_status,
        "next_tool": normalized_action,
        "base_status_score": round(float(base_status_score), 6),
        "action_offset": round(float(action_offset), 6),
        "score_before_action": round(float(base_status_score), 6),
        "score_after_action": round(float(score_after_action), 6),
    }


def _normal_case_type(rollout_trace: Dict[str, Any], *, target: Dict[str, Any]) -> str:
    if normalize_existence(target.get("existence")) != "normal":
        return "non_normal"
    turns = list(rollout_trace.get("turns") or [])
    num_seek = sum(1 for turn in turns if str(turn.get("tool_name") or "").strip() == "seek_evidence")
    num_verify = sum(1 for turn in turns if str(turn.get("tool_name") or "").strip() == "verify_hypothesis")
    return "easy_normal" if num_seek == 0 and num_verify <= 1 else "hard_normal"


def _anomaly_false_normal_penalty(*, prediction: Dict[str, Any], target: Dict[str, Any]) -> float:
    if normalize_existence(target.get("existence")) != "anomaly":
        return 0.0
    if normalize_existence(prediction.get("existence")) != "normal":
        return 0.0
    return -1.0


def _protocol_finalize_reward(rollout_trace: Dict[str, Any], verifier_turn: Optional[Dict[str, Any]]) -> float:
    turns = list(rollout_trace.get("turns") or [])
    state = dict(rollout_trace.get("state") or {})
    has_finalize_artifact = isinstance(state.get("finalized_case"), dict) or any(
        str(turn.get("tool_name") or "") == "finalize_case" for turn in turns
    )
    if not has_finalize_artifact:
        return -1.0
    if verifier_turn is None:
        return 0.2
    next_tool = str(verifier_turn.get("verifier_next_tool") or verifier_turn.get("next_tool") or "").strip().lower()
    return 1.0 if next_tool == "finalize_case" else 0.2


def _stage_necessity_reward(rollout_trace: Dict[str, Any], *, target: Dict[str, Any]) -> float:
    required_stages = infer_required_stages_from_target(target)
    if not required_stages:
        return 0.0
    annotation = extract_stage_annotation_from_record(rollout_trace)
    terminal = isinstance((rollout_trace.get("state") or {}).get("finalized_case"), dict) or isinstance(
        rollout_trace.get("final_answer"),
        dict,
    )
    return float(compute_event_chain_score(required_stages, annotation, terminal=terminal))


def _query_alignment_reward(rollout_trace: Dict[str, Any]) -> float:
    scores = [
        float(compute_query_alignment_score(turn))
        for turn in list(rollout_trace.get("turns") or [])
        if str(turn.get("tool_name") or "").strip() == "seek_evidence"
    ]
    if not scores:
        return 0.0
    return round(sum(scores) / float(len(scores)), 6)


def _efficiency_reward(rollout_trace: Dict[str, Any]) -> float:
    turns = list(rollout_trace.get("turns") or [])
    if not turns:
        return 0.0
    return round(max(-1.0, 1.0 - 0.08 * max(0, len(turns) - 4)), 6)


def _compute_accuracy_breakdown(
    rollout_trace: Dict[str, Any],
    *,
    llm_judge: Optional[OpenAICompatibleLlmJudge] = None,
    reward_version: str = DEFAULT_RL_REWARD_VERSION,
    semantic_override: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    _normalize_reward_version(reward_version)
    target = _infer_target(rollout_trace)
    semantic_payload = _infer_semantic_payload(rollout_trace)
    prediction = dict((semantic_payload or {}).get("decision") or {}) or _infer_final_prediction(rollout_trace)
    qa_pairs = _infer_scoring_qa_pairs(rollout_trace)
    evidence_moments = _infer_scoring_evidence_moments(rollout_trace)
    fps = _safe_float(((rollout_trace.get("video_meta") or {}).get("fps")), 0.0)
    if fps <= 0.0:
        fps = _safe_float((((rollout_trace.get("multimodal_cache") or {}).get("fps"))), 0.0)
    type_scores: Dict[str, float] = {}
    family_scores: Dict[str, List[float]] = {
        "classification": [],
        "temporal": [],
        "open_ended": [],
    }
    if not target or not prediction:
        return {
            "accuracy_reward": 0.0,
            "accuracy_by_family": {},
            "accuracy_by_type": {},
            "accuracy_question_count": 0,
        }
    all_scores: List[float] = []

    existence_score = 1.0 if normalize_existence(prediction.get("existence")) == normalize_existence(target.get("existence")) else 0.0
    type_scores["existence"] = existence_score
    family_scores["classification"].append(existence_score)
    all_scores.append(existence_score)

    if normalize_existence(target.get("existence")) == "anomaly":
        target_category = canonicalize_saver_category(target.get("category"), existence="anomaly")
        prediction_category = canonicalize_saver_category(prediction.get("category"), existence="anomaly")
        if target_category:
            category_score = 1.0 if prediction_category == target_category else 0.0
            type_scores["category"] = category_score
            family_scores["classification"].append(category_score)
            all_scores.append(category_score)
        if target.get("anomaly_interval_sec") is not None:
            temporal_interval = _prediction_interval_from_semantic_payload(
                semantic_payload,
                decision_field="anomaly_interval_sec",
                qa_key="temporal",
                fps=fps,
            ) or prediction.get("anomaly_interval_sec")
            temporal_score = 1.0 if _interval_iou(
                temporal_interval,
                target.get("anomaly_interval_sec"),
            ) >= 0.3 else 0.0
            type_scores["temporal"] = temporal_score
            family_scores["temporal"].append(temporal_score)
            all_scores.append(temporal_score)

    judge = llm_judge or OpenAICompatibleLlmJudge()
    open_targets = _open_ended_target_map(
        structured_target=target,
        qa_pairs=qa_pairs,
        evidence_moments=evidence_moments,
    )
    open_predictions = _open_ended_prediction_map(semantic_payload)
    override_scores = dict(semantic_override or {})
    for qa_type in _OPEN_ENDED_QUESTION_TYPES_V4:
        target_entry = open_targets.get(qa_type)
        if target_entry is None:
            continue
        question, reference = target_entry
        prediction_text = str(open_predictions.get(qa_type) or "").strip()
        if not prediction_text:
            score = 0.0
            type_scores[qa_type] = score
            family_scores["open_ended"].append(score)
            all_scores.append(score)
            continue
        if qa_type in override_scores:
            score = float(override_scores[qa_type])
        else:
            score = float(judge.score(question=question, reference=reference, prediction=prediction_text))
        type_scores[qa_type] = round(float(score), 6)
        family_scores["open_ended"].append(score)
        all_scores.append(score)

    accuracy_reward = sum(all_scores) / float(len(all_scores)) if all_scores else 0.0
    return {
        "accuracy_reward": round(float(accuracy_reward), 6),
        "accuracy_by_family": {
            family: round(float(sum(scores) / float(len(scores))), 6) if scores else 0.0
            for family, scores in family_scores.items()
        },
        "accuracy_by_type": {key: round(float(value), 6) for key, value in type_scores.items()},
        "accuracy_question_count": int(len(all_scores)),
    }


def build_timesearch_reward_funcs(
    *,
    reward_config: Optional[Dict[str, Any]] = None,
    llm_judge: Optional[OpenAICompatibleLlmJudge] = None,
    reward_version: str = DEFAULT_RL_REWARD_VERSION,
) -> List[Any]:
    del reward_config
    _normalize_reward_version(reward_version)
    judge = llm_judge or OpenAICompatibleLlmJudge()

    def accuracy_reward(*, rollout_traces: Sequence[Dict[str, Any]], **_: Any) -> List[float]:
        traces = list(rollout_traces or [])
        if not traces:
            return []
        queries_per_rollout = [
            _collect_semantic_queries(trace, reward_version=reward_version)
            for trace in traces
        ]
        flat_queries: List[Tuple[str, str, str]] = []
        index_map: List[Tuple[int, str]] = []
        for trace_index, query_map in enumerate(queries_per_rollout):
            for qa_type, triple in query_map.items():
                flat_queries.append(triple)
                index_map.append((trace_index, qa_type))
        flat_scores = judge.score_batch(flat_queries) if flat_queries else []
        overrides: List[Dict[str, float]] = [{} for _ in traces]
        for (trace_index, qa_type), score in zip(index_map, flat_scores):
            overrides[trace_index][qa_type] = float(score)
        return [
            float(
                _compute_accuracy_breakdown(
                    trace,
                    llm_judge=judge,
                    reward_version=reward_version,
                    semantic_override=overrides[index],
                )["accuracy_reward"]
            )
            for index, trace in enumerate(traces)
        ]

    def protocol_finalize_reward(*, rollout_traces: Sequence[Dict[str, Any]], **_: Any) -> List[float]:
        return [float(_protocol_finalize_reward(trace, _latest_verifier_turn(trace))) for trace in list(rollout_traces or [])]

    def stage_necessity_reward(*, rollout_traces: Sequence[Dict[str, Any]], **_: Any) -> List[float]:
        return [float(_stage_necessity_reward(trace, target=_infer_target(trace))) for trace in list(rollout_traces or [])]

    def query_alignment_reward(*, rollout_traces: Sequence[Dict[str, Any]], **_: Any) -> List[float]:
        return [float(_query_alignment_reward(trace)) for trace in list(rollout_traces or [])]

    def efficiency_reward(*, rollout_traces: Sequence[Dict[str, Any]], **_: Any) -> List[float]:
        return [float(_efficiency_reward(trace)) for trace in list(rollout_traces or [])]

    def anomaly_false_normal_penalty(*, rollout_traces: Sequence[Dict[str, Any]], **_: Any) -> List[float]:
        return [
            float(
                _anomaly_false_normal_penalty(
                    prediction=_infer_final_prediction(trace),
                    target=_infer_target(trace),
                )
            )
            for trace in list(rollout_traces or [])
        ]

    accuracy_reward.__name__ = "accuracy_reward"
    protocol_finalize_reward.__name__ = "protocol_finalize_reward"
    stage_necessity_reward.__name__ = "stage_necessity_reward"
    query_alignment_reward.__name__ = "query_alignment_reward"
    efficiency_reward.__name__ = "efficiency_reward"
    anomaly_false_normal_penalty.__name__ = "anomaly_false_normal_penalty"
    return [
        accuracy_reward,
        protocol_finalize_reward,
        stage_necessity_reward,
        query_alignment_reward,
        efficiency_reward,
        anomaly_false_normal_penalty,
    ]


def _score_rollout_trace_timesearch(
    rollout_trace: Dict[str, Any],
    *,
    reward_version: str = DEFAULT_RL_REWARD_VERSION,
    weights: Optional[Dict[str, float]] = None,
    reward_config: Optional[Dict[str, Any]] = None,
    llm_judge: Optional[OpenAICompatibleLlmJudge] = None,
) -> Dict[str, Any]:
    normalized_reward_version = _normalize_reward_version(reward_version)
    normalized_weights = resolve_reward_component_weights(
        reward_version=normalized_reward_version,
        reward_config=reward_config,
        weights=weights,
    )
    verifier_turn = _latest_verifier_turn(rollout_trace)
    target = _infer_target(rollout_trace)
    final_prediction = _infer_final_prediction(rollout_trace)
    accuracy = _compute_accuracy_breakdown(
        rollout_trace,
        llm_judge=llm_judge,
        reward_version=normalized_reward_version,
    )
    normal_case_type = _normal_case_type(rollout_trace, target=target)
    easy_normal_sample_loss_multiplier = 0.20 if normal_case_type == "easy_normal" else 1.0

    components = {
        "accuracy_reward": float(accuracy["accuracy_reward"]),
        "protocol_finalize_reward": float(_protocol_finalize_reward(rollout_trace, verifier_turn)),
        "stage_necessity_reward": float(_stage_necessity_reward(rollout_trace, target=target)),
        "query_alignment_reward": float(_query_alignment_reward(rollout_trace)),
        "efficiency_reward": float(_efficiency_reward(rollout_trace)),
        "anomaly_false_normal_penalty": float(
            _anomaly_false_normal_penalty(prediction=final_prediction, target=target)
        ),
    }
    weighted_components = {
        key: round(float(normalized_weights.get(key, 0.0)) * float(value), 6)
        for key, value in components.items()
    }
    total_reward = sum(float(weighted_components.get(key, 0.0)) for key in weighted_components)

    return {
        "reward_version": str(normalized_reward_version),
        "total_reward": round(float(total_reward), 6),
        "components": {key: round(float(value), 6) for key, value in components.items()},
        "weighted_components": dict(weighted_components),
        "weights": dict(normalized_weights),
        "final_decision_correct": 1.0 if _decision_matches(final_prediction, target) else 0.0,
        "accuracy_reward": float(accuracy["accuracy_reward"]),
        "accuracy_by_family": dict(accuracy["accuracy_by_family"]),
        "accuracy_by_type": dict(accuracy["accuracy_by_type"]),
        "accuracy_question_count": int(accuracy["accuracy_question_count"]),
        "normal_case_type": str(normal_case_type),
        "easy_normal_sample_loss_multiplier": round(float(easy_normal_sample_loss_multiplier), 6),
        "normal_verification_consistency_score": round(float(_normal_verification_consistency_score(rollout_trace)), 6),
        "normal_continuous_verifier_score": round(float(_normal_continuous_verifier_score_details(rollout_trace)["score_after_action"]), 6),
        "normal_verifier_primary_status": str(_normal_continuous_verifier_score_details(rollout_trace)["primary_status"]),
        "normal_verifier_next_tool": str(_normal_continuous_verifier_score_details(rollout_trace)["next_tool"]),
        "normal_verifier_base_status_score": round(float(_normal_continuous_verifier_score_details(rollout_trace)["base_status_score"]), 6),
        "normal_verifier_action_offset": round(float(_normal_continuous_verifier_score_details(rollout_trace)["action_offset"]), 6),
        "normal_continuous_verifier_score_before_action": round(float(_normal_continuous_verifier_score_details(rollout_trace)["score_before_action"]), 6),
        "normal_continuous_verifier_score_after_action": round(float(_normal_continuous_verifier_score_details(rollout_trace)["score_after_action"]), 6),
    }


def score_rollout_trace(
    rollout_trace: Dict[str, Any],
    *,
    reward_version: str = DEFAULT_RL_REWARD_VERSION,
    weights: Optional[Dict[str, float]] = None,
    reward_config: Optional[Dict[str, Any]] = None,
    llm_judge: Optional[OpenAICompatibleLlmJudge] = None,
) -> Dict[str, Any]:
    return _score_rollout_trace_timesearch(
        rollout_trace,
        reward_version=reward_version,
        weights=weights,
        reward_config=reward_config,
        llm_judge=llm_judge,
    )
