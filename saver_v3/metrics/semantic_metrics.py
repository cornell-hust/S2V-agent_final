from __future__ import annotations

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from saver_v3.core.categories import canonicalize_saver_category
from saver_v3.core.llm_judge import OpenAICompatibleLlmJudge
from saver_v3.metrics.offline_scoring import ReferenceDataProvider
from saver_v3.core.semantic_answer import (
    SEMANTIC_EVENT_CHAIN_STAGES,
    build_semantic_answer_payload,
    normalize_semantic_answer_payload,
    normalize_text_match,
)
from saver_v3.metrics.temporal_grounding_metrics import interval_iou, normalize_interval


QA_ACCURACY_FIELDS = (
    "existence",
    "category",
    "temporal",
    "precursor",
    "trigger",
    "confirmation",
)
DEFAULT_SEMANTIC_METRICS = ("rouge", "bertscore", "qa_accuracy", "qa_relaxed")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _tokenize(text: Any) -> List[str]:
    normalized = normalize_text_match(text)
    return [token for token in normalized.split(" ") if token]


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values)) / float(len(values))


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1))


def _f1_from_overlap(pred_count: Counter, ref_count: Counter) -> float:
    overlap = sum(min(pred_count[key], ref_count[key]) for key in pred_count.keys() & ref_count.keys())
    pred_total = sum(pred_count.values())
    ref_total = sum(ref_count.values())
    if pred_total <= 0 or ref_total <= 0 or overlap <= 0:
        return 0.0
    precision = float(overlap) / float(pred_total)
    recall = float(overlap) / float(ref_total)
    if precision + recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def rouge_n_f1(prediction: Any, reference: Any, *, n: int = 1) -> float:
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    return _f1_from_overlap(_ngram_counts(pred_tokens, n), _ngram_counts(ref_tokens, n))


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


def rouge_l_f1(prediction: Any, reference: Any) -> float:
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


def _token_f1(prediction: Any, reference: Any) -> float:
    return _f1_from_overlap(
        Counter(_tokenize(prediction)),
        Counter(_tokenize(reference)),
    )


def _sequence_similarity(prediction: Any, reference: Any) -> float:
    normalized_prediction = normalize_text_match(prediction)
    normalized_reference = normalize_text_match(reference)
    if not normalized_prediction or not normalized_reference:
        return 0.0
    return float(SequenceMatcher(None, normalized_prediction, normalized_reference).ratio())


def _reference_semantic_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    return build_semantic_answer_payload(
        structured_target=record.get("structured_target") or {},
        qa_pairs=record.get("qa_pairs") or [],
        evidence_moments=((record.get("evidence") or {}).get("evidence_moments") or []),
        finalized_case=record.get("structured_target") or {},
    )


def _field_value(payload: Dict[str, Any] | None, field_name: str) -> str:
    payload = payload or {}
    if field_name in {"precursor", "trigger", "confirmation"}:
        return str(((payload.get("event_chain_summary") or {}).get(field_name)) or "").strip()
    return str(((payload.get("qa_focus_answers") or {}).get(field_name)) or "").strip()


def _normalize_existence_value(payload: Dict[str, Any] | None, text_value: str) -> str:
    decision = dict((payload or {}).get("decision") or {})
    existence = str(decision.get("existence") or "").strip().lower()
    if existence in {"normal", "anomaly"}:
        return existence
    normalized = normalize_text_match(text_value)
    if "no anomaly" in normalized or normalized.startswith("no ") or " normal " in f" {normalized} ":
        return "normal"
    if "yes" in normalized or "anomaly" in normalized:
        return "anomaly"
    return ""


def _normalize_category_value(payload: Dict[str, Any] | None, text_value: str) -> str:
    decision = dict((payload or {}).get("decision") or {})
    category = canonicalize_saver_category(
        decision.get("category"),
        existence=decision.get("existence"),
    )
    if category:
        return category
    return canonicalize_saver_category(text_value)


def _compare_qa_field(field_name: str, prediction: Dict[str, Any] | None, reference: Dict[str, Any]) -> bool:
    predicted_text = _field_value(prediction, field_name)
    reference_text = _field_value(reference, field_name)
    if field_name == "existence":
        return _normalize_existence_value(prediction, predicted_text) == _normalize_existence_value(reference, reference_text)
    if field_name == "category":
        return _normalize_category_value(prediction, predicted_text) == _normalize_category_value(reference, reference_text)
    return normalize_text_match(predicted_text) == normalize_text_match(reference_text)


def _collect_summary_strings(
    rollouts: Sequence[Dict[str, Any]],
    *,
    reference_data: ReferenceDataProvider,
) -> tuple[List[str], List[str]]:
    predictions: List[str] = []
    references: List[str] = []
    for rollout in list(rollouts):
        video_id = str(rollout.get("video_id") or "").strip()
        if not video_id or video_id not in reference_data.by_video_id:
            continue
        reference_payload = _reference_semantic_payload(reference_data.by_video_id[video_id])
        prediction_payload = normalize_semantic_answer_payload(rollout.get("semantic_answer"))
        predictions.append(str((prediction_payload or {}).get("summary") or ""))
        references.append(str(reference_payload.get("summary") or ""))
    return predictions, references


def _build_field_stats() -> Dict[str, Dict[str, float | int]]:
    return {
        field_name: {"correct": 0, "total": 0, "coverage": 0}
        for field_name in QA_ACCURACY_FIELDS
    }


def _finalize_binary_field_stats(field_stats: Dict[str, Dict[str, float | int]], *, prefix: str) -> Dict[str, Any]:
    total = sum(int(stat["total"]) for stat in field_stats.values())
    correct = sum(int(stat["correct"]) for stat in field_stats.values())
    coverage = sum(int(stat["coverage"]) for stat in field_stats.values())
    for stat in field_stats.values():
        field_total = int(stat["total"])
        stat["accuracy"] = float(stat["correct"]) / float(field_total) if field_total > 0 else 0.0
        stat["coverage_rate"] = float(stat["coverage"]) / float(field_total) if field_total > 0 else 0.0
    if prefix == "qa_accuracy":
        return {
            "qa_accuracy_overall": float(correct) / float(total) if total > 0 else 0.0,
            "qa_accuracy_coverage": float(coverage) / float(total) if total > 0 else 0.0,
            "qa_accuracy_fields": field_stats,
        }
    return {
        f"{prefix}_summary": {
            "overall": float(correct) / float(total) if total > 0 else 0.0,
            "coverage": int(coverage),
            "coverage_rate": float(coverage) / float(total) if total > 0 else 0.0,
            "fields": field_stats,
        }
    }


def _evaluate_rouge(
    rollouts: Sequence[Dict[str, Any]],
    *,
    reference_data: ReferenceDataProvider,
) -> Dict[str, Any]:
    predictions, references = _collect_summary_strings(rollouts, reference_data=reference_data)
    rouge_1_scores = [rouge_n_f1(pred, ref, n=1) for pred, ref in zip(predictions, references)]
    rouge_l_scores = [rouge_l_f1(pred, ref) for pred, ref in zip(predictions, references)]
    coverage = sum(1 for prediction in predictions if str(prediction).strip())
    total = len(references)
    return {
        "rouge_1_f1": _safe_mean(rouge_1_scores),
        "rouge_l_f1": _safe_mean(rouge_l_scores),
        "coverage": int(coverage),
        "coverage_rate": float(coverage) / float(total) if total > 0 else 0.0,
    }


def _evaluate_bertscore(
    rollouts: Sequence[Dict[str, Any]],
    *,
    reference_data: ReferenceDataProvider,
    bertscore_model_path: str | Path = "",
) -> Dict[str, Any]:
    predictions, references = _collect_summary_strings(rollouts, reference_data=reference_data)
    coverage = sum(1 for prediction in predictions if str(prediction).strip())
    total = len(references)
    try:
        from bert_score import score as bert_score
    except Exception:
        return {
            "available": False,
            "skipped_reason": "BERTScore requested but the `bert_score` package is not installed.",
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "coverage": int(coverage),
            "coverage_rate": float(coverage) / float(total) if total > 0 else 0.0,
        }
    if not references:
        return {
            "available": True,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "coverage": 0,
            "coverage_rate": 0.0,
        }
    model_type = str(bertscore_model_path or "").strip()
    bertscore_kwargs: Dict[str, Any] = {
        "lang": "en",
        "verbose": False,
    }
    if model_type:
        bertscore_kwargs["model_type"] = model_type
    precision, recall, f1 = bert_score(predictions, references, **bertscore_kwargs)
    return {
        "available": True,
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
        "f1": float(f1.mean().item()),
        "coverage": int(coverage),
        "coverage_rate": float(coverage) / float(total) if total > 0 else 0.0,
        "model_type": model_type,
    }


def _evaluate_qa_accuracy(
    rollouts: Sequence[Dict[str, Any]],
    *,
    reference_data: ReferenceDataProvider,
) -> Dict[str, Any]:
    field_stats = _build_field_stats()
    for rollout in list(rollouts):
        video_id = str(rollout.get("video_id") or "").strip()
        if not video_id or video_id not in reference_data.by_video_id:
            continue
        reference_payload = _reference_semantic_payload(reference_data.by_video_id[video_id])
        prediction_payload = normalize_semantic_answer_payload(rollout.get("semantic_answer"))
        for field_name in QA_ACCURACY_FIELDS:
            reference_value = _field_value(reference_payload, field_name)
            if not reference_value:
                continue
            field_stats[field_name]["total"] += 1
            predicted_value = _field_value(prediction_payload, field_name)
            if predicted_value:
                field_stats[field_name]["coverage"] += 1
            if _compare_qa_field(field_name, prediction_payload, reference_payload):
                field_stats[field_name]["correct"] += 1
    return _finalize_binary_field_stats(field_stats, prefix="qa_accuracy")


def _field_interval_key(field_name: str) -> str:
    if field_name == "temporal":
        return "anomaly_interval_sec"
    if field_name == "precursor":
        return "precursor_interval_sec"
    return ""


def _extract_interval_from_text(text: str, *, video_meta: Dict[str, Any]) -> Optional[tuple[float, float]]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return None
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", raw_text)
    if len(matches) < 2:
        return None
    start_value = _safe_float(matches[0], 0.0)
    end_value = _safe_float(matches[1], 0.0)
    normalized_text = raw_text.lower()
    fps = _safe_float((video_meta or {}).get("fps"), 0.0)
    duration_sec = _safe_float((video_meta or {}).get("duration_sec"), 0.0)
    likely_frames = "frame" in normalized_text or (
        fps > 0.0 and duration_sec > 0.0 and max(abs(start_value), abs(end_value)) > max(duration_sec + 1.0, duration_sec * 1.5)
    )
    if likely_frames and fps > 0.0:
        start_value /= fps
        end_value /= fps
    return normalize_interval((start_value, end_value))


def _extract_interval_for_field(
    field_name: str,
    *,
    payload: Dict[str, Any] | None,
    text_value: str,
    video_meta: Dict[str, Any],
) -> Optional[tuple[float, float]]:
    decision = dict((payload or {}).get("decision") or {})
    interval_key = _field_interval_key(field_name)
    if interval_key:
        normalized = normalize_interval(decision.get(interval_key))
        if normalized is not None:
            return normalized
    return _extract_interval_from_text(text_value, video_meta=video_meta)


def _relaxed_text_match(prediction: str, reference: str) -> bool:
    normalized_prediction = normalize_text_match(prediction)
    normalized_reference = normalize_text_match(reference)
    if not normalized_prediction or not normalized_reference:
        return False
    if normalized_prediction == normalized_reference:
        return True
    if _token_f1(prediction, reference) >= 0.5:
        return True
    if rouge_l_f1(prediction, reference) >= 0.5:
        return True
    return _sequence_similarity(prediction, reference) >= 0.5


def _compare_qa_field_relaxed(
    field_name: str,
    prediction: Dict[str, Any] | None,
    reference: Dict[str, Any],
    *,
    reference_record: Dict[str, Any],
) -> bool:
    predicted_text = _field_value(prediction, field_name)
    reference_text = _field_value(reference, field_name)
    if field_name == "existence":
        return _normalize_existence_value(prediction, predicted_text) == _normalize_existence_value(reference, reference_text)
    if field_name == "category":
        return _normalize_category_value(prediction, predicted_text) == _normalize_category_value(reference, reference_text)
    if normalize_text_match(predicted_text) == normalize_text_match(reference_text):
        return True
    video_meta = dict(reference_record.get("video_meta") or {})
    predicted_interval = _extract_interval_for_field(
        field_name,
        payload=prediction,
        text_value=predicted_text,
        video_meta=video_meta,
    )
    reference_interval = _extract_interval_for_field(
        field_name,
        payload=reference,
        text_value=reference_text,
        video_meta=video_meta,
    )
    if predicted_interval is not None and reference_interval is not None and interval_iou(predicted_interval, reference_interval) >= 0.5:
        return True
    return _relaxed_text_match(predicted_text, reference_text)


def _evaluate_qa_relaxed(
    rollouts: Sequence[Dict[str, Any]],
    *,
    reference_data: ReferenceDataProvider,
) -> Dict[str, Any]:
    field_stats = _build_field_stats()
    for rollout in list(rollouts):
        video_id = str(rollout.get("video_id") or "").strip()
        if not video_id or video_id not in reference_data.by_video_id:
            continue
        reference_record = reference_data.by_video_id[video_id]
        reference_payload = _reference_semantic_payload(reference_record)
        prediction_payload = normalize_semantic_answer_payload(rollout.get("semantic_answer"))
        for field_name in QA_ACCURACY_FIELDS:
            reference_value = _field_value(reference_payload, field_name)
            if not reference_value:
                continue
            field_stats[field_name]["total"] += 1
            predicted_value = _field_value(prediction_payload, field_name)
            if predicted_value:
                field_stats[field_name]["coverage"] += 1
            if _compare_qa_field_relaxed(
                field_name,
                prediction_payload,
                reference_payload,
                reference_record=reference_record,
            ):
                field_stats[field_name]["correct"] += 1
    return _finalize_binary_field_stats(field_stats, prefix="qa_relaxed")


def _judge_question_text(reference_record: Dict[str, Any], *, field_name: str = "") -> str:
    task_prompt = str(((reference_record.get("agent_task") or {}).get("task_prompt")) or "").strip()
    if field_name:
        return f"{task_prompt or 'Evaluate the SAVER answer.'} Focus field: {field_name}."
    return task_prompt or "Evaluate the SAVER answer."


def _empty_judge_field_summary() -> Dict[str, Dict[str, float | int]]:
    return {
        field_name: {"score": 0.0, "total": 0, "coverage": 0, "coverage_rate": 0.0}
        for field_name in QA_ACCURACY_FIELDS
    }


def _evaluate_qa_judge(
    rollouts: Sequence[Dict[str, Any]],
    *,
    reference_data: ReferenceDataProvider,
    judge: OpenAICompatibleLlmJudge,
) -> Dict[str, Any]:
    if not judge.is_configured:
        return {
            "configured": False,
            "overall": 0.0,
            "coverage": 0,
            "coverage_rate": 0.0,
            "fields": _empty_judge_field_summary(),
            "skipped_reason": "semantic judge is not configured",
        }
    field_totals = {field_name: 0 for field_name in QA_ACCURACY_FIELDS}
    field_coverages = {field_name: 0 for field_name in QA_ACCURACY_FIELDS}
    field_scores = {field_name: [] for field_name in QA_ACCURACY_FIELDS}
    aggregate_scores: list[float] = []
    total = 0
    coverage = 0
    for rollout in list(rollouts):
        video_id = str(rollout.get("video_id") or "").strip()
        if not video_id or video_id not in reference_data.by_video_id:
            continue
        reference_record = reference_data.by_video_id[video_id]
        reference_payload = _reference_semantic_payload(reference_record)
        prediction_payload = normalize_semantic_answer_payload(rollout.get("semantic_answer"))
        for field_name in QA_ACCURACY_FIELDS:
            reference_value = _field_value(reference_payload, field_name)
            if not reference_value:
                continue
            total += 1
            field_totals[field_name] += 1
            predicted_value = _field_value(prediction_payload, field_name)
            if predicted_value:
                coverage += 1
                field_coverages[field_name] += 1
            score = judge.score(
                question=_judge_question_text(reference_record, field_name=field_name),
                reference=reference_value,
                prediction=predicted_value,
            ) if predicted_value else 0.0
            aggregate_scores.append(float(score))
            field_scores[field_name].append(float(score))
    field_summary = _empty_judge_field_summary()
    for field_name in QA_ACCURACY_FIELDS:
        field_total = int(field_totals[field_name])
        field_coverage = int(field_coverages[field_name])
        field_summary[field_name] = {
            "score": _safe_mean(field_scores[field_name]),
            "total": field_total,
            "coverage": field_coverage,
            "coverage_rate": float(field_coverage) / float(field_total) if field_total > 0 else 0.0,
        }
    return {
        "configured": True,
        "overall": _safe_mean(aggregate_scores),
        "coverage": int(coverage),
        "coverage_rate": float(coverage) / float(total) if total > 0 else 0.0,
        "fields": field_summary,
        "judge_model": judge.model,
    }


def _render_explanation_text(payload: Dict[str, Any] | None) -> str:
    normalized = normalize_semantic_answer_payload(payload)
    if normalized is None:
        return ""
    parts: list[str] = []
    summary = str(normalized.get("summary") or "").strip()
    rationale = str(normalized.get("rationale") or "").strip()
    if summary:
        parts.append(f"Summary: {summary}")
    if rationale:
        parts.append(f"Rationale: {rationale}")
    event_chain_summary = dict(normalized.get("event_chain_summary") or {})
    rendered_event_chain = {
        stage: str(event_chain_summary.get(stage) or "").strip()
        for stage in SEMANTIC_EVENT_CHAIN_STAGES
        if str(event_chain_summary.get(stage) or "").strip()
    }
    if rendered_event_chain:
        parts.append(f"Event chain: {json.dumps(rendered_event_chain, ensure_ascii=False, sort_keys=True)}")
    return "\n".join(parts)


def _causal_judge_prompt(
    *,
    question: str,
    reference_text: str,
    prediction_text: str,
) -> str:
    return (
        "Rate the model explanation on a 0-5 scale.\n"
        "5 = semantically equivalent, causally complete, evidence-grounded.\n"
        "4 = mostly correct with only minor omissions.\n"
        "3 = partially correct but meaningfully incomplete.\n"
        "2 = limited overlap or weak causal linkage.\n"
        "1 = mostly incorrect.\n"
        "0 = wrong, missing, or contradictory.\n\n"
        f"Question: {question}\n"
        f"Reference explanation:\n{reference_text}\n\n"
        f"Model explanation:\n{prediction_text}\n\n"
        "Score:"
    )


def _evaluate_causal_judge(
    rollouts: Sequence[Dict[str, Any]],
    *,
    reference_data: ReferenceDataProvider,
    judge: OpenAICompatibleLlmJudge,
) -> Dict[str, Any]:
    if not judge.is_configured:
        return {
            "configured": False,
            "overall": 0.0,
            "coverage": 0,
            "coverage_rate": 0.0,
            "num_records": 0,
            "skipped_reason": "semantic judge is not configured",
        }
    scores: list[float] = []
    total = 0
    coverage = 0
    for rollout in list(rollouts):
        video_id = str(rollout.get("video_id") or "").strip()
        if not video_id or video_id not in reference_data.by_video_id:
            continue
        reference_record = reference_data.by_video_id[video_id]
        reference_text = _render_explanation_text(_reference_semantic_payload(reference_record))
        if not reference_text:
            continue
        total += 1
        prediction_text = _render_explanation_text(rollout.get("semantic_answer"))
        if prediction_text:
            coverage += 1
        fallback_score = max(
            rouge_l_f1(prediction_text, reference_text),
            _sequence_similarity(prediction_text, reference_text),
        )
        scores.append(
            judge.score_rubric(
                prompt=_causal_judge_prompt(
                    question=_judge_question_text(reference_record),
                    reference_text=reference_text,
                    prediction_text=prediction_text,
                ),
                rubric_name="causal explanation quality",
                min_score=0.0,
                max_score=5.0,
                fallback_score=fallback_score,
            ) if prediction_text else 0.0
        )
    return {
        "configured": True,
        "overall": _safe_mean(scores),
        "coverage": int(coverage),
        "coverage_rate": float(coverage) / float(total) if total > 0 else 0.0,
        "num_records": int(total),
        "judge_model": judge.model,
    }


def evaluate_semantic_rollouts(
    rollouts: Sequence[Dict[str, Any]],
    *,
    data_path: str | Path,
    metrics: Sequence[str] | None = None,
    judge_base_url: str = "",
    judge_model: str = "",
    judge_cache_path: str | Path = "",
    judge_timeout_sec: float = 30.0,
    bertscore_model_path: str | Path = "",
) -> Dict[str, Any]:
    requested_metrics = [
        str(metric).strip().lower()
        for metric in list(metrics or DEFAULT_SEMANTIC_METRICS)
        if str(metric).strip()
    ]
    reference_data = ReferenceDataProvider(data_path=data_path)
    result: Dict[str, Any] = {
        "num_records": len(list(rollouts)),
        "metrics": requested_metrics,
    }
    if "rouge" in requested_metrics:
        result["rouge_summary"] = _evaluate_rouge(rollouts, reference_data=reference_data)
    if "bertscore" in requested_metrics:
        result["bertscore_summary"] = _evaluate_bertscore(
            rollouts,
            reference_data=reference_data,
            bertscore_model_path=bertscore_model_path,
        )
    if "qa_accuracy" in requested_metrics:
        result.update(_evaluate_qa_accuracy(rollouts, reference_data=reference_data))
    if "qa_relaxed" in requested_metrics:
        result.update(_evaluate_qa_relaxed(rollouts, reference_data=reference_data))
    judge = None
    if "qa_judge" in requested_metrics or "causal_judge" in requested_metrics:
        judge = OpenAICompatibleLlmJudge(
            base_url=str(judge_base_url or "").strip(),
            model=str(judge_model or "").strip(),
            cache_path=str(judge_cache_path or "").strip(),
            timeout_sec=float(judge_timeout_sec),
        )
    if "qa_judge" in requested_metrics:
        result["qa_judge_summary"] = _evaluate_qa_judge(
            rollouts,
            reference_data=reference_data,
            judge=judge or OpenAICompatibleLlmJudge(),
        )
    if "causal_judge" in requested_metrics:
        result["causal_judge_summary"] = _evaluate_causal_judge(
            rollouts,
            reference_data=reference_data,
            judge=judge or OpenAICompatibleLlmJudge(),
        )
    return result
