from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple


TEMPORAL_GROUNDING_THRESHOLDS = (0.3, 0.5, 0.7)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def normalize_interval(interval: Sequence[float] | None) -> Optional[Tuple[float, float]]:
    if not interval or len(interval) != 2:
        return None
    start_sec = _safe_float(interval[0], 0.0)
    end_sec = _safe_float(interval[1], 0.0)
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec
    return start_sec, end_sec


def interval_iou(interval_a: Sequence[float] | None, interval_b: Sequence[float] | None) -> float:
    normalized_a = normalize_interval(interval_a)
    normalized_b = normalize_interval(interval_b)
    if normalized_a is None or normalized_b is None:
        return 0.0
    overlap = max(0.0, min(normalized_a[1], normalized_b[1]) - max(normalized_a[0], normalized_b[0]))
    if overlap <= 0.0:
        return 0.0
    length_a = max(0.0, normalized_a[1] - normalized_a[0])
    length_b = max(0.0, normalized_b[1] - normalized_b[0])
    union = max(length_a + length_b - overlap, 1e-6)
    return max(0.0, min(1.0, overlap / union))


def _interpolated_precision_recall(precision: Sequence[float], recall: Sequence[float]) -> float:
    if not precision or not recall or len(precision) != len(recall):
        return 0.0
    merged_precision = [0.0, *[float(value) for value in precision], 0.0]
    merged_recall = [0.0, *[float(value) for value in recall], 1.0]
    for index in range(len(merged_precision) - 2, -1, -1):
        merged_precision[index] = max(merged_precision[index], merged_precision[index + 1])
    ap = 0.0
    for index in range(1, len(merged_recall)):
        if merged_recall[index] != merged_recall[index - 1]:
            ap += (merged_recall[index] - merged_recall[index - 1]) * merged_precision[index]
    return float(ap)


def _average_precision_for_query(
    predicted_windows: Sequence[Tuple[float, float]],
    reference_windows: Sequence[Tuple[float, float]],
    *,
    iou_threshold: float,
) -> float:
    if not reference_windows:
        return 0.0
    if not predicted_windows:
        return 0.0
    matched_reference = set()
    true_positive: list[float] = []
    false_positive: list[float] = []
    for predicted in predicted_windows:
        best_index = None
        best_iou = 0.0
        for ref_index, reference in enumerate(reference_windows):
            if ref_index in matched_reference:
                continue
            current_iou = interval_iou(predicted, reference)
            if current_iou > best_iou:
                best_iou = current_iou
                best_index = ref_index
        if best_index is not None and best_iou >= float(iou_threshold):
            matched_reference.add(best_index)
            true_positive.append(1.0)
            false_positive.append(0.0)
        else:
            true_positive.append(0.0)
            false_positive.append(1.0)
    cumulative_tp: list[float] = []
    cumulative_fp: list[float] = []
    tp_total = 0.0
    fp_total = 0.0
    for tp_value, fp_value in zip(true_positive, false_positive):
        tp_total += tp_value
        fp_total += fp_value
        cumulative_tp.append(tp_total)
        cumulative_fp.append(fp_total)
    recall = [value / float(len(reference_windows)) for value in cumulative_tp]
    precision = [
        cumulative_tp[index] / float(max(cumulative_tp[index] + cumulative_fp[index], 1e-6))
        for index in range(len(cumulative_tp))
    ]
    return _interpolated_precision_recall(precision, recall)


def _threshold_key(threshold: float) -> str:
    return str(threshold).replace(".", "_")


def compute_temporal_grounding_summary(
    temporal_samples: Iterable[tuple[Sequence[Tuple[float, float]], Sequence[Tuple[float, float]]]],
    *,
    thresholds: Sequence[float] = TEMPORAL_GROUNDING_THRESHOLDS,
) -> dict[str, float | int]:
    normalized_thresholds = [float(threshold) for threshold in thresholds]
    per_threshold_r1 = {threshold: [] for threshold in normalized_thresholds}
    per_threshold_ap = {threshold: [] for threshold in normalized_thresholds}
    invalid_pred_num = 0
    query_count = 0

    for predicted_windows, reference_windows in list(temporal_samples):
        if not reference_windows:
            continue
        query_count += 1
        normalized_predicted = [
            interval
            for interval in (normalize_interval(window) for window in predicted_windows)
            if interval is not None
        ]
        normalized_reference = [
            interval
            for interval in (normalize_interval(window) for window in reference_windows)
            if interval is not None
        ]
        if not normalized_reference:
            query_count -= 1
            continue
        if not normalized_predicted:
            invalid_pred_num += 1
            best_top1_iou = 0.0
        else:
            best_top1_iou = max(interval_iou(normalized_predicted[0], reference) for reference in normalized_reference)
        for threshold in normalized_thresholds:
            per_threshold_r1[threshold].append(1.0 if best_top1_iou >= threshold else 0.0)
            per_threshold_ap[threshold].append(
                _average_precision_for_query(
                    normalized_predicted,
                    normalized_reference,
                    iou_threshold=threshold,
                )
            )

    summary: dict[str, float | int] = {
        "temporal_invalid_pred_num": int(invalid_pred_num),
    }
    r1_values = []
    ap_values = []
    for threshold in normalized_thresholds:
        key = _threshold_key(threshold)
        recall_value = (
            float(sum(per_threshold_r1[threshold])) / float(query_count)
            if query_count > 0
            else 0.0
        )
        ap_value = (
            float(sum(per_threshold_ap[threshold])) / float(query_count)
            if query_count > 0
            else 0.0
        )
        summary[f"temporal_r1_at_{key}"] = recall_value
        summary[f"temporal_map_at_{key}"] = ap_value
        r1_values.append(recall_value)
        ap_values.append(ap_value)
    summary["temporal_r1_avg"] = float(sum(r1_values)) / float(len(r1_values)) if r1_values else 0.0
    summary["temporal_map_avg"] = float(sum(ap_values)) / float(len(ap_values)) if ap_values else 0.0
    return summary
