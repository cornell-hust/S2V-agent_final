from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from saver_v3.core.categories import canonicalize_saver_category
from saver_v3.core.event_chain import (
    compute_stage_f1,
    extract_stage_annotation_from_record,
    infer_required_stages_from_target,
    has_complete_event_chain,
)
from saver_v3.metrics.score_summary import extract_verifier_statuses, summarize_scored_rollouts
from saver_v3.metrics.temporal_grounding_metrics import (
    TEMPORAL_GROUNDING_THRESHOLDS,
    compute_temporal_grounding_summary,
    interval_iou,
)


PAPER_MAIN_METRIC_KEYS = (
    "anomaly_span_recall_at_0_3",
    "anomaly_span_recall_at_0_5",
    "anomaly_span_recall_at_0_7",
    "temporal_miou",
    "existence_accuracy",
    "category_macro_f1",
    "temporal_r1_at_0_3",
    "temporal_r1_at_0_5",
    "temporal_r1_at_0_7",
    "temporal_map_avg",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _threshold_key(threshold: float) -> str:
    return str(float(threshold)).replace(".", "_")


def _single_interval_recall_summary(
    interval_ious: Sequence[float],
    *,
    thresholds: Sequence[float] = TEMPORAL_GROUNDING_THRESHOLDS,
    metric_prefix: str = "anomaly_span_recall",
) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    values = [max(0.0, min(1.0, _safe_float(value, 0.0))) for value in interval_ious]
    for threshold in thresholds:
        key = _threshold_key(float(threshold))
        summary[f"{metric_prefix}_at_{key}"] = (
            sum(1.0 for value in values if value >= float(threshold)) / len(values)
            if values
            else 0.0
        )
    return summary


def _order_metric_summary_for_reporting(summary: Dict[str, Any]) -> Dict[str, Any]:
    ordered: Dict[str, Any] = {}
    for key in ("num_records", *PAPER_MAIN_METRIC_KEYS):
        if key in summary:
            ordered[key] = summary[key]
    for key, value in summary.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def _normalize_signature_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_signature_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_signature_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_signature_value(item) for item in value)
    if isinstance(value, float):
        return round(float(value), 4)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    return value


def _canonical_search_signature_from_turn(turn: Dict[str, Any]) -> Optional[str]:
    tool_name = str(turn.get("tool_name") or "")
    if tool_name not in {"scan_timeline", "seek_evidence"}:
        return None
    parsed_tool_call = dict(turn.get("parsed_tool_call") or {})
    arguments = parsed_tool_call.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}
    return str(
        {
            "name": tool_name,
            "arguments": _normalize_signature_value(arguments),
        }
    )


def _search_turn_has_progress(turn: Dict[str, Any]) -> bool:
    if str(turn.get("tool_name") or "") not in {"scan_timeline", "seek_evidence"}:
        return False
    if list(turn.get("new_evidence_ids") or []):
        return True
    state_delta = dict(turn.get("state_delta") or {})
    if list(state_delta.get("new_evidence_windows") or []):
        return True
    if list(state_delta.get("new_visited_windows") or []):
        return True
    return False


def _normalize_interval(interval: Sequence[float] | None) -> Optional[Tuple[float, float]]:
    if not interval or len(interval) != 2:
        return None
    start_sec = _safe_float(interval[0], 0.0)
    end_sec = _safe_float(interval[1], 0.0)
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec
    return start_sec, end_sec


def _interval_length(interval: Sequence[float] | None) -> float:
    normalized = _normalize_interval(interval)
    if normalized is None:
        return 0.0
    return max(0.0, normalized[1] - normalized[0])


def _interval_overlap(interval_a: Sequence[float] | None, interval_b: Sequence[float] | None) -> float:
    a = _normalize_interval(interval_a)
    b = _normalize_interval(interval_b)
    if a is None or b is None:
        return 0.0
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _merge_intervals(intervals: Iterable[Sequence[float]]) -> List[Tuple[float, float]]:
    normalized = sorted(
        (value for value in (_normalize_interval(interval) for interval in intervals) if value is not None),
        key=lambda item: item[0],
    )
    if not normalized:
        return []
    merged: List[Tuple[float, float]] = [normalized[0]]
    for start_sec, end_sec in normalized[1:]:
        prev_start, prev_end = merged[-1]
        if start_sec <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end_sec))
        else:
            merged.append((start_sec, end_sec))
    return merged


def _union_length(intervals: Iterable[Sequence[float]]) -> float:
    return sum(end_sec - start_sec for start_sec, end_sec in _merge_intervals(intervals))


def _binary_average_precision(targets: Sequence[int], scores: Sequence[float]) -> float:
    if not targets or not scores or len(targets) != len(scores):
        return 0.0
    total_positives = sum(1 for value in targets if int(value) > 0)
    if total_positives <= 0:
        return 0.0
    ranked_indices = sorted(range(len(targets)), key=lambda idx: float(scores[idx]), reverse=True)
    precision_sum = 0.0
    true_positives = 0
    for rank, idx in enumerate(ranked_indices, start=1):
        if int(targets[idx]) > 0:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / total_positives


def _macro_f1(gt_labels: Sequence[str], pred_labels: Sequence[str]) -> float:
    classes = sorted({label for label in gt_labels if label and label != "normal"})
    if not classes or len(gt_labels) != len(pred_labels):
        return 0.0
    per_class_f1: List[float] = []
    for label in classes:
        tp = fp = fn = 0
        for gt_label, pred_label in zip(gt_labels, pred_labels):
            if gt_label == label and pred_label == label:
                tp += 1
            elif gt_label != label and pred_label == label:
                fp += 1
            elif gt_label == label and pred_label != label:
                fn += 1
        denom = 2 * tp + fp + fn
        per_class_f1.append((2 * tp / denom) if denom > 0 else 0.0)
    return _mean(per_class_f1)


def _extract_reference_target(record: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(record.get("structured_target"), dict):
        return dict(record["structured_target"])
    label = dict(record.get("label") or {})
    temporal = dict(record.get("temporal") or {})
    return {
        "existence": "anomaly" if label.get("is_anomaly") else "normal",
        "category": label.get("category"),
        "severity": label.get("severity"),
        "hard_normal": label.get("hard_normal"),
        "anomaly_interval_sec": temporal.get("anomaly_interval_sec"),
        "precursor_interval_sec": temporal.get("precursor_interval_sec"),
        "earliest_actionable_sec": temporal.get("earliest_actionable_sec", temporal.get("earliest_alert_sec")),
        "counterfactual_type": (record.get("counterfactual") or {}).get("type", "none"),
    }


def _extract_reference_evidence_windows(record: Dict[str, Any]) -> List[Tuple[float, float]]:
    tool_io = dict(record.get("tool_io") or {})
    raw_windows = tool_io.get("oracle_windows_sec") or []
    intervals: List[Tuple[float, float]] = []
    for entry in raw_windows:
        interval = entry.get("window") or entry.get("window_sec")
        normalized = _normalize_interval(interval)
        if normalized is not None:
            intervals.append(normalized)
    if intervals:
        return intervals
    target = _extract_reference_target(record)
    for entry in target.get("evidence_windows_sec") or []:
        interval = entry.get("window") or entry.get("window_sec")
        normalized = _normalize_interval(interval)
        if normalized is not None:
            intervals.append(normalized)
    return intervals


def _predicted_existence_score(record: Dict[str, Any]) -> float:
    claim = _infer_claim_from_rollout(record)
    if claim:
        return 1.0 if str(claim.get("existence") or "").lower() == "anomaly" else 0.0
    return 0.0


def _normalize_existence(value: Any) -> str:
    text = str(value or "").strip().lower()
    return "anomaly" if text == "anomaly" else "normal"


def _normalize_category(value: Any) -> str:
    text = canonicalize_saver_category(value)
    return text or "unknown"


def _normalize_counterfactual_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text or "none"


def _infer_claim_from_rollout(record: Dict[str, Any]) -> Dict[str, Any]:
    state = record.get("state") or {}
    finalized_case = state.get("finalized_case")
    if isinstance(finalized_case, dict):
        return dict(finalized_case)
    final_answer = record.get("final_answer")
    if isinstance(final_answer, dict):
        return dict(final_answer)
    return {}


def _extract_counterfactual_profile(record: Dict[str, Any]) -> Dict[str, Any]:
    profile = record.get("counterfactual_profile")
    if isinstance(profile, dict) and profile:
        if "summary" in profile:
            return dict(profile)
        return {
            "summary": {
                "decision_sufficiency": bool(profile.get("decision_sufficiency")),
                "minimal_subset_sufficiency": bool(profile.get("minimal_subset_sufficiency")),
                "negative_specificity_pass": bool(profile.get("negative_specificity_pass")),
                "counterfactual_type_supported": bool(profile.get("counterfactual_type_supported")),
                "stage_necessity": dict(profile.get("stage_necessity") or {}),
            },
            "branch_field_matrix": dict(profile.get("branch_field_matrix") or {}),
            "branch_delta_matrix": dict(profile.get("branch_delta_matrix") or {}),
            "stage_packages": dict(profile.get("stage_packages") or {}),
            "selection_metadata": dict(profile.get("selection_metadata") or {}),
        }
    return {}


def _counterfactual_summary(profile: Dict[str, Any]) -> Dict[str, Any]:
    summary = profile.get("summary")
    if isinstance(summary, dict):
        return summary
    return {}


def _counterfactual_branch_delta(profile: Dict[str, Any], branch_name: str, field_name: str) -> float:
    matrix = dict(profile.get("branch_delta_matrix") or {})
    return _safe_float((((matrix.get(branch_name) or {}).get("fields") or {}).get(field_name)), 0.0)


def _known_evidence_window_ids(record: Dict[str, Any]) -> set[str]:
    state = record.get("state") or {}
    return {
        str(entry.get("window_id")).strip()
        for entry in (state.get("evidence_ledger") or [])
        if str(entry.get("window_id") or "").strip()
    }


def _filter_known_window_ids(window_ids: Sequence[str], *, known_window_ids: set[str]) -> List[str]:
    filtered: List[str] = []
    seen = set()
    for value in window_ids or []:
        window_id = str(value).strip()
        if not window_id or window_id not in known_window_ids or window_id in seen:
            continue
        filtered.append(window_id)
        seen.add(window_id)
    return filtered


def _latest_verification_turn(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for turn in reversed(record.get("turns") or []):
        if str(turn.get("tool_name") or "") == "verify_hypothesis":
            return turn
    return None


def _latest_verification_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    verification_records = ((record.get("state") or {}).get("verification_records") or [])
    if verification_records:
        return dict(verification_records[-1] or {})
    return None


def _infer_candidate_window_ids(record: Dict[str, Any]) -> Tuple[List[str], bool]:
    known_window_ids = _known_evidence_window_ids(record)
    latest_verify_turn = _latest_verification_turn(record)
    if latest_verify_turn is not None:
        raw_window_ids = (
            latest_verify_turn.get("verifier_verified_window_ids")
            or latest_verify_turn.get("verifier_best_effort_window_ids")
            or latest_verify_turn.get("self_verification_selected_window_ids")
            or []
        )
        selection_observed = bool(
            raw_window_ids
            or latest_verify_turn.get("verification_parse_mode")
            or latest_verify_turn.get("invalid_selected_window_ids")
            or latest_verify_turn.get("selection_resolution_source")
            or latest_verify_turn.get("verifier_failure_reasons")
        )
        if selection_observed:
            return _filter_known_window_ids(raw_window_ids, known_window_ids=known_window_ids), True

    state = record.get("state") or {}
    active_window_ids = list(state.get("active_evidence_window_ids") or [])
    if active_window_ids:
        return _filter_known_window_ids(active_window_ids, known_window_ids=known_window_ids), True

    latest_verification_record = _latest_verification_record(record)
    if latest_verification_record is not None:
        raw_window_ids = (
            latest_verification_record.get("verified_window_ids")
            or latest_verification_record.get("best_effort_window_ids")
            or latest_verification_record.get("selected_window_ids")
            or []
        )
        selection_observed = bool(
            raw_window_ids
            or latest_verification_record.get("verification_parse_mode")
            or latest_verification_record.get("invalid_selected_window_ids")
            or latest_verification_record.get("selection_resolution_source")
            or latest_verification_record.get("failure_reasons")
        )
        if selection_observed:
            return _filter_known_window_ids(raw_window_ids, known_window_ids=known_window_ids), True

    return [], False


def _resolve_state_windows(record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    state = record.get("state") or {}
    entries = list(state.get("evidence_ledger") or [])
    resolved: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        window_id = entry.get("window_id")
        if window_id and window_id not in resolved:
            resolved[str(window_id)] = dict(entry)
    return resolved


def _first_matching_turn_index(
    turns: Sequence[Dict[str, Any]],
    *,
    tool_name: Optional[str] = None,
    action: Optional[str] = None,
) -> Optional[int]:
    for turn in turns:
        if tool_name is not None and turn.get("tool_name") == tool_name:
            return _safe_int(turn.get("step_index"), 0)
        if action is not None and turn.get("action") == action:
            return _safe_int(turn.get("step_index"), 0)
    return None


def _protocol_compliance_flag(record: Dict[str, Any]) -> float:
    turns = list(record.get("turns") or [])
    state = record.get("state") or {}

    finalize_turn_index = _first_matching_turn_index(turns, tool_name="finalize_case")
    answer_turn_index = _first_matching_turn_index(turns, action="answer")

    has_finalize_artifact = finalize_turn_index is not None or isinstance(state.get("finalized_case"), dict)
    has_valid_explicit_answer = answer_turn_index is not None and isinstance(record.get("final_answer"), dict)
    has_implicit_terminal_answer = answer_turn_index is None and has_finalize_artifact
    has_answer_artifact = has_valid_explicit_answer or has_implicit_terminal_answer

    if not has_finalize_artifact or not has_answer_artifact:
        return 0.0
    if finalize_turn_index is not None and answer_turn_index is not None:
        return 1.0 if finalize_turn_index < answer_turn_index else 0.0
    return 1.0


def _select_predicted_evidence_windows(record: Dict[str, Any], *, top_k: int) -> List[Tuple[float, float]]:
    if top_k <= 0:
        return []
    return _rank_predicted_evidence_windows(record)[:top_k]


def _rank_predicted_evidence_windows(record: Dict[str, Any]) -> List[Tuple[float, float]]:
    candidate_ids, selection_observed = _infer_candidate_window_ids(record)
    by_window_id = _resolve_state_windows(record)
    selected: List[Tuple[float, float]] = []
    seen = set()
    for window_id in candidate_ids:
        window = by_window_id.get(str(window_id))
        if window is None:
            continue
        normalized = _normalize_interval((window.get("start_sec"), window.get("end_sec")))
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(normalized)
    if selected or selection_observed:
        return selected
    for window in by_window_id.values():
        normalized = _normalize_interval((window.get("start_sec"), window.get("end_sec")))
        if normalized is None or normalized in seen:
            continue
        seen.add(normalized)
        selected.append(normalized)
    return selected


def _evidence_match_counts(
    predicted_windows: Sequence[Tuple[float, float]],
    reference_windows: Sequence[Tuple[float, float]],
    *,
    iou_threshold: float,
) -> Tuple[int, int, int]:
    matched_reference = set()
    matched_predicted = 0
    for pred_index, predicted in enumerate(predicted_windows):
        best_idx = None
        best_iou = 0.0
        for ref_index, reference in enumerate(reference_windows):
            if ref_index in matched_reference:
                continue
            iou = interval_iou(predicted, reference)
            if iou >= float(iou_threshold) and iou > best_iou:
                best_iou = iou
                best_idx = ref_index
        if best_idx is not None:
            matched_reference.add(best_idx)
            matched_predicted += 1
    return matched_predicted, len(predicted_windows), len(reference_windows)


def _verify_health_summary(scored_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    verify_parse_mode_counts = Counter()
    total_verify_turns = 0
    invalid_selected_turns = 0
    unresolved_selection_turns = 0
    verify_invalid_turns = 0

    for record in scored_records:
        for turn in record.get("turns") or []:
            if str(turn.get("tool_name") or "") != "verify_hypothesis":
                continue
            total_verify_turns += 1
            parse_mode = str(turn.get("verification_parse_mode") or "unknown")
            verify_parse_mode_counts[parse_mode] += 1
            if list(turn.get("invalid_selected_window_ids") or []):
                invalid_selected_turns += 1
            failure_reasons = {str(value) for value in (turn.get("verifier_failure_reasons") or []) if str(value).strip()}
            if (
                "selected_evidence_not_resolved_to_known_windows" in failure_reasons
                or str(turn.get("selection_resolution_source") or "") == "unresolved"
            ):
                unresolved_selection_turns += 1
            if not bool(turn.get("valid_action", turn.get("action") in {"tool_call", "answer"})):
                verify_invalid_turns += 1

    return {
        "verify_parse_mode_counts": {str(key): int(value) for key, value in sorted(verify_parse_mode_counts.items())},
        "invalid_selected_window_rate": _safe_rate(invalid_selected_turns, total_verify_turns),
        "unresolved_selection_rate": _safe_rate(unresolved_selection_turns, total_verify_turns),
        "verify_invalid_turn_rate": _safe_rate(verify_invalid_turns, total_verify_turns),
    }


def summarize_saver_metrics(
    scored_records: Sequence[Dict[str, Any]],
    *,
    reference_data: Any,
    evidence_top_k: int = 3,
    evidence_iou_threshold: float = 0.3,
    include_diagnostic_summary: bool = False,
) -> Dict[str, Any]:
    existence_targets: List[int] = []
    existence_predictions: List[int] = []
    anomaly_gt_categories: List[str] = []
    anomaly_pred_categories: List[str] = []
    temporal_ious: List[float] = []
    temporal_grounding_samples: List[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]] = []
    precursor_ious: List[float] = []
    evidence_precisions: List[float] = []
    evidence_recalls: List[float] = []
    evidence_f1s: List[float] = []
    counterfactual_hits: List[float] = []
    event_chain_f1_values: List[float] = []
    event_chain_finalize_flags: List[float] = []
    severity_errors: List[float] = []
    inspected_clip_ratios: List[float] = []
    total_turns: List[float] = []
    tool_validity_values: List[float] = []
    formal_turn_validity_values: List[float] = []
    protocol_flags: List[float] = []
    null_final_answer_flags: List[float] = []
    finalized_case_flags: List[float] = []
    max_turn_termination_flags: List[float] = []
    fecv_decision_sufficiency_flags: List[float] = []
    fecv_minimal_subset_flags: List[float] = []
    fecv_negative_specificity_flags: List[float] = []
    fecv_trigger_drop_effects: List[float] = []
    fecv_confirmation_drop_effects: List[float] = []
    fecv_precursor_drop_effects: List[float] = []
    fecv_finalize_readiness_drops: List[float] = []
    verify_coverage_flags: List[float] = []
    primary_status_counter = Counter()
    verify_finalize_followthrough_numerator = 0
    verify_finalize_followthrough_denominator = 0

    for record in scored_records:
        video_id = record.get("video_id")
        if not video_id or video_id not in reference_data.by_video_id:
            continue
        reference_record = reference_data.by_video_id[str(video_id)]
        target = _extract_reference_target(reference_record)
        claim = _infer_claim_from_rollout(record)
        gt_existence = _normalize_existence(target.get("existence"))
        pred_existence = _normalize_existence(claim.get("existence"))

        existence_targets.append(1 if gt_existence == "anomaly" else 0)
        existence_predictions.append(1 if pred_existence == "anomaly" else 0)

        if gt_existence == "anomaly":
            anomaly_gt_categories.append(_normalize_category(target.get("category")))
            anomaly_pred_categories.append(
                _normalize_category(claim.get("category")) if pred_existence == "anomaly" else "normal"
            )
            temporal_ious.append(interval_iou(claim.get("anomaly_interval_sec"), target.get("anomaly_interval_sec")))
            if _normalize_interval(target.get("precursor_interval_sec")) is not None:
                precursor_ious.append(
                    interval_iou(claim.get("precursor_interval_sec"), target.get("precursor_interval_sec"))
                )
            gt_counterfactual_type = _normalize_counterfactual_type(target.get("counterfactual_type"))
            if gt_counterfactual_type != "none":
                counterfactual_hits.append(
                    1.0 if _normalize_counterfactual_type(claim.get("counterfactual_type")) == gt_counterfactual_type else 0.0
                )
            required_stages = infer_required_stages_from_target(
                target,
                tool_io=dict(reference_record.get("tool_io") or {}),
            )
            if required_stages:
                predicted_stage_annotation = extract_stage_annotation_from_record(record)
                event_chain_f1_values.append(
                    compute_stage_f1(required_stages, predicted_stage_annotation.get("covered_stages") or [])
                )
                event_chain_finalize_flags.append(
                    1.0 if has_complete_event_chain(required_stages, predicted_stage_annotation) else 0.0
                )
            gt_severity = target.get("severity")
            pred_severity = claim.get("severity")
            if gt_severity is not None and pred_severity is not None:
                severity_errors.append(abs(_safe_float(pred_severity) - _safe_float(gt_severity)))

            gt_evidence_windows = _extract_reference_evidence_windows(reference_record)
            if gt_evidence_windows:
                ranked_pred_evidence_windows = _rank_predicted_evidence_windows(record)
                temporal_grounding_samples.append((ranked_pred_evidence_windows, gt_evidence_windows))
                pred_evidence_windows = ranked_pred_evidence_windows[: max(0, int(evidence_top_k))]
                matched, pred_count, gt_count = _evidence_match_counts(
                    pred_evidence_windows,
                    gt_evidence_windows,
                    iou_threshold=evidence_iou_threshold,
                )
                precision = matched / pred_count if pred_count > 0 else 0.0
                recall = matched / gt_count if gt_count > 0 else 0.0
                denom = precision + recall
                f1 = (2 * precision * recall / denom) if denom > 0 else 0.0
                evidence_precisions.append(precision)
                evidence_recalls.append(recall)
                evidence_f1s.append(f1)

        duration = _safe_float((reference_record.get("video_meta") or {}).get("duration_sec"), 0.0)
        state = record.get("state") or {}
        visited_intervals = [
            (entry.get("start_sec"), entry.get("end_sec"))
            for entry in list(state.get("visited_windows") or [])
            if _normalize_interval((entry.get("start_sec"), entry.get("end_sec"))) is not None
        ]
        inspected_length = _union_length(visited_intervals)
        inspected_clip_ratios.append(inspected_length / duration if duration > 0 else 0.0)
        total_turns.append(_safe_float(record.get("num_turns"), 0.0))
        null_final_answer_flags.append(1.0 if not isinstance(record.get("final_answer"), dict) else 0.0)
        finalized_case_flags.append(1.0 if isinstance(state.get("finalized_case"), dict) else 0.0)
        max_turn_termination_flags.append(1.0 if str(record.get("terminated_reason") or "") == "max_turns" else 0.0)
        counterfactual_profile = _extract_counterfactual_profile(record)
        counterfactual_summary = _counterfactual_summary(counterfactual_profile)
        fecv_decision_sufficiency_flags.append(
            1.0 if bool(counterfactual_summary.get("decision_sufficiency")) else 0.0
        )
        fecv_minimal_subset_flags.append(
            1.0 if bool(counterfactual_summary.get("minimal_subset_sufficiency")) else 0.0
        )
        fecv_negative_specificity_flags.append(
            1.0 if bool(counterfactual_summary.get("negative_specificity_pass")) else 0.0
        )
        fecv_trigger_drop_effects.append(
            max(
                _counterfactual_branch_delta(counterfactual_profile, "drop_trigger", "existence"),
                _counterfactual_branch_delta(counterfactual_profile, "drop_trigger", "category"),
                _counterfactual_branch_delta(counterfactual_profile, "drop_trigger", "temporal"),
            )
        )
        fecv_confirmation_drop_effects.append(
            max(
                _counterfactual_branch_delta(counterfactual_profile, "drop_confirmation", "confirmation"),
                _counterfactual_branch_delta(counterfactual_profile, "drop_confirmation", "finalize_readiness"),
            )
        )
        fecv_precursor_drop_effects.append(
            max(
                _counterfactual_branch_delta(counterfactual_profile, "drop_precursor", "precursor"),
                _counterfactual_branch_delta(counterfactual_profile, "drop_precursor", "finalize_readiness"),
            )
        )
        fecv_finalize_readiness_drops.append(
            max(
                _counterfactual_branch_delta(counterfactual_profile, "drop_precursor", "finalize_readiness"),
                _counterfactual_branch_delta(counterfactual_profile, "drop_trigger", "finalize_readiness"),
                _counterfactual_branch_delta(counterfactual_profile, "drop_confirmation", "finalize_readiness"),
            )
        )

        turns = list(record.get("turns") or [])
        invalid_attempts = list(record.get("invalid_attempts") or [])
        if turns:
            formal_turn_validity_values.append(
                sum(
                    1.0
                    for turn in turns
                    if bool(turn.get("valid_action", turn.get("action") in {"tool_call", "answer"}))
                )
                / len(turns)
            )
        else:
            formal_turn_validity_values.append(0.0)

        attempts = turns + invalid_attempts
        if attempts:
            tool_validity_values.append(
                sum(
                    1.0
                    for attempt in attempts
                    if bool(attempt.get("valid_action", attempt.get("action") in {"tool_call", "answer"}))
                )
                / len(attempts)
            )
        else:
            tool_validity_values.append(0.0)

        protocol_flags.append(_protocol_compliance_flag(record))
        verify_turns = [turn for turn in turns if str(turn.get("tool_name") or "") == "verify_hypothesis"]
        verify_coverage_flags.append(1.0 if verify_turns else 0.0)
        finalize_recommend_steps = [
            _safe_int(turn.get("step_index"), 0)
            for turn in verify_turns
            if str(turn.get("verifier_recommended_action") or "") == "finalize"
        ]
        if finalize_recommend_steps:
            verify_finalize_followthrough_denominator += 1
            earliest_finalize_recommend_step = min(finalize_recommend_steps)
            has_followthrough = isinstance(state.get("finalized_case"), dict) or any(
                str(turn.get("tool_name") or "") == "finalize_case"
                and _safe_int(turn.get("step_index"), 0) > earliest_finalize_recommend_step
                for turn in turns
            )
            if has_followthrough:
                verify_finalize_followthrough_numerator += 1

        primary_status, _ = extract_verifier_statuses(record)
        primary_status_counter[str(primary_status or "unknown")] += 1

    num_records = len(existence_targets)
    primary_ratios = {
        key: (primary_status_counter.get(key, 0) / num_records if num_records else 0.0)
        for key in ["complete", "incomplete", "redundant", "misaligned", "unknown"]
    }
    verify_health = _verify_health_summary(scored_records)
    summary = {
        "num_records": num_records,
        "existence_accuracy": (
            sum(1.0 for gt, pred in zip(existence_targets, existence_predictions) if gt == pred) / num_records
            if num_records
            else 0.0
        ),
        "category_macro_f1": _macro_f1(anomaly_gt_categories, anomaly_pred_categories),
        **compute_temporal_grounding_summary(temporal_grounding_samples),
        **_single_interval_recall_summary(temporal_ious),
        "temporal_miou": _mean(temporal_ious),
        "precursor_miou": _mean(precursor_ious),
        "evidence_precision_at_3": _mean(evidence_precisions),
        "evidence_recall_at_3": _mean(evidence_recalls),
        "evidence_f1_at_3": _mean(evidence_f1s),
        "fecv_decision_sufficiency_rate": _mean(fecv_decision_sufficiency_flags),
        "fecv_minimal_subset_rate": _mean(fecv_minimal_subset_flags),
        "fecv_negative_specificity_rate": _mean(fecv_negative_specificity_flags),
        "fecv_trigger_drop_effect": _mean(fecv_trigger_drop_effects),
        "fecv_confirmation_drop_effect": _mean(fecv_confirmation_drop_effects),
        "fecv_precursor_drop_effect": _mean(fecv_precursor_drop_effects),
        "fecv_finalize_readiness_drop": _mean(fecv_finalize_readiness_drops),
        "counterfactual_type_accuracy": _mean(counterfactual_hits),
        "event_chain_f1": _mean(event_chain_f1_values),
        "event_chain_finalize_rate": _mean(event_chain_finalize_flags),
        "severity_mae": _mean(severity_errors),
        "mean_inspected_clip_ratio": _mean(inspected_clip_ratios),
        "mean_num_turns": _mean(total_turns),
        "tool_call_validity_rate": _mean(tool_validity_values),
        "formal_turn_validity_rate": _mean(formal_turn_validity_values),
        "protocol_compliance_rate": _mean(protocol_flags),
        "null_final_answer_rate": _mean(null_final_answer_flags),
        "finalized_case_rate": _mean(finalized_case_flags),
        "max_turn_termination_rate": _mean(max_turn_termination_flags),
        "verify_coverage_rate": _mean(verify_coverage_flags),
        "verify_finalize_followthrough_rate": _safe_rate(
            verify_finalize_followthrough_numerator,
            verify_finalize_followthrough_denominator,
        ),
        **verify_health,
    }
    if include_diagnostic_summary:
        summary["diagnostic_summary"] = {
            "primary_status_ratios": primary_ratios,
            "score_summary": summarize_scored_rollouts(scored_records),
        }
    return _order_metric_summary_for_reporting(summary)
