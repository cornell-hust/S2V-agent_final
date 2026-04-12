from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from saver_v3.core.protocol_guidance import (
    event_chain_stage_for_role,
    normalize_event_chain_stages,
    normalize_stage_selected_moment_ids,
)
from saver_v3.core.schema import SaverEnvironmentState


DEFAULT_SUPPORT_WEIGHTS = {
    "exist_support": 0.20,
    "category_support": 0.20,
    "temporal_support": 0.20,
    "precursor_support": 0.10,
    "finalize_support": 0.20,
    "counterfactual_support": 0.10,
}
SUPPORT_COMPONENT_KEYS = tuple(DEFAULT_SUPPORT_WEIGHTS.keys())

PRIMARY_STATUS_VALUES = {"complete", "incomplete", "redundant", "misaligned"}
_VERIFIER_RUNTIME_CACHE: Dict[Tuple[str, str, str, str, int], Any] = {}


def _normalize_interval(interval: Sequence[float] | None) -> Optional[Tuple[float, float]]:
    if not interval or len(interval) != 2:
        return None
    start_sec = float(interval[0])
    end_sec = float(interval[1])
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec
    return start_sec, end_sec


def _interval_overlap(interval_a: Sequence[float] | None, interval_b: Sequence[float] | None) -> float:
    a = _normalize_interval(interval_a)
    b = _normalize_interval(interval_b)
    if a is None or b is None:
        return 0.0
    start_sec = max(a[0], b[0])
    end_sec = min(a[1], b[1])
    return max(0.0, end_sec - start_sec)


def _interval_duration(interval: Sequence[float] | None) -> float:
    normalized = _normalize_interval(interval)
    if normalized is None:
        return 0.0
    return max(0.0, normalized[1] - normalized[0])


def _overlap_ratio(interval_a: Sequence[float] | None, interval_b: Sequence[float] | None) -> float:
    duration = _interval_duration(interval_b)
    if duration <= 0:
        return 0.0
    return max(0.0, min(1.0, _interval_overlap(interval_a, interval_b) / duration))


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


def _coverage_ratio(windows: Sequence[Dict[str, Any]], target_interval: Sequence[float] | None) -> float:
    target = _normalize_interval(target_interval)
    if target is None:
        return 0.0
    clipped = []
    for window in windows:
        interval = _normalize_interval((window.get("start_sec"), window.get("end_sec")))
        if interval is None:
            continue
        overlap = _interval_overlap(interval, target)
        if overlap <= 0:
            continue
        clipped.append((max(interval[0], target[0]), min(interval[1], target[1])))
    if not clipped:
        return 0.0
    total = sum(end_sec - start_sec for start_sec, end_sec in _merge_intervals(clipped))
    target_duration = max(target[1] - target[0], 1e-6)
    return max(0.0, min(1.0, total / target_duration))


def _extract_target(
    multimodal_cache: Dict[str, Any],
    *,
    use_reference_supervision: bool = True,
) -> Dict[str, Any]:
    if not use_reference_supervision:
        return {}
    return dict(multimodal_cache.get("structured_target") or {})


def _extract_oracle_windows(
    multimodal_cache: Dict[str, Any],
    *,
    use_reference_supervision: bool = True,
) -> List[Dict[str, Any]]:
    if not use_reference_supervision:
        return []
    tool_io = multimodal_cache.get("tool_io") or {}
    raw_windows = tool_io.get("oracle_windows_sec") or []
    if raw_windows:
        windows = []
        for entry in raw_windows:
            interval = entry.get("window") or entry.get("window_sec")
            normalized = _normalize_interval(interval)
            if normalized is None:
                continue
            windows.append(
                {
                    "moment_id": entry.get("moment_id"),
                    "role": str(entry.get("role") or "").lower(),
                    "start_sec": normalized[0],
                    "end_sec": normalized[1],
                    "description": entry.get("description"),
                }
            )
        if windows:
            return windows

    target = _extract_target(multimodal_cache, use_reference_supervision=use_reference_supervision)
    evidence_windows = target.get("evidence_windows_sec") or []
    windows = []
    for entry in evidence_windows:
        interval = entry.get("window_sec") or entry.get("window")
        normalized = _normalize_interval(interval)
        if normalized is None:
            continue
        windows.append(
            {
                "moment_id": entry.get("moment_id"),
                "role": str(entry.get("role") or "").lower(),
                "start_sec": normalized[0],
                "end_sec": normalized[1],
                "description": entry.get("description"),
            }
        )
    return windows


def _event_chain_target(claim: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(claim.get("event_chain_target"), dict):
        return dict(claim.get("event_chain_target") or {})
    if isinstance(target.get("event_chain_target"), dict):
        return dict(target.get("event_chain_target") or {})
    return {}


def _required_event_chain_stages(
    *,
    claim: Dict[str, Any],
    target: Dict[str, Any],
    oracle_windows: Sequence[Dict[str, Any]],
) -> List[str]:
    chain_target = _event_chain_target(claim, target)
    required_stages = normalize_event_chain_stages(chain_target.get("required_stages"))
    if required_stages:
        return required_stages
    existence = str(claim.get("existence") or target.get("existence") or "").strip().lower()
    if existence != "anomaly":
        return []
    inferred: List[str] = []
    if chain_target.get("stage_to_moment_ids"):
        stage_to_moment_ids = normalize_stage_selected_moment_ids(chain_target.get("stage_to_moment_ids"))
        if stage_to_moment_ids.get("precursor"):
            inferred.append("precursor")
        inferred.append("trigger")
        if stage_to_moment_ids.get("confirmation"):
            inferred.append("confirmation")
        return normalize_event_chain_stages(inferred)
    if _normalize_interval(claim.get("precursor_interval_sec") or target.get("precursor_interval_sec")) is not None:
        inferred.append("precursor")
    inferred.append("trigger")
    if any(event_chain_stage_for_role(entry.get("role")) == "confirmation" for entry in oracle_windows):
        inferred.append("confirmation")
    return normalize_event_chain_stages(inferred)


def _selected_event_chain_state(
    *,
    selected_windows: Sequence[Dict[str, Any]],
    claim: Dict[str, Any],
    target: Dict[str, Any],
    oracle_windows: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, List[str]], List[str]]:
    chain_target = _event_chain_target(claim, target)
    moment_stage_map: Dict[str, str] = {}
    for stage, moment_ids in normalize_stage_selected_moment_ids(chain_target.get("stage_to_moment_ids")).items():
        for moment_id in moment_ids:
            moment_stage_map[str(moment_id)] = stage
    for entry in list(oracle_windows or []):
        moment_id = str(entry.get("moment_id") or "").strip()
        if not moment_id:
            continue
        stage = event_chain_stage_for_role(entry.get("role"))
        if stage is not None:
            moment_stage_map[moment_id] = stage

    stage_selected_moment_ids: Dict[str, List[str]] = {}
    covered_stage_set = set()
    for window in list(selected_windows or []):
        moment_id = str(window.get("moment_id") or "").strip()
        stage = event_chain_stage_for_role(window.get("role"))
        if stage is None and moment_id:
            stage = moment_stage_map.get(moment_id)
        if stage is None:
            continue
        covered_stage_set.add(stage)
        if not moment_id:
            continue
        stage_selected_moment_ids.setdefault(stage, [])
        if moment_id not in stage_selected_moment_ids[stage]:
            stage_selected_moment_ids[stage].append(moment_id)
    covered_stages = normalize_event_chain_stages(covered_stage_set)
    return normalize_stage_selected_moment_ids(stage_selected_moment_ids), covered_stages


def _resolve_windows(
    state: SaverEnvironmentState,
    *,
    candidate_window_ids: Optional[Sequence[str]] = None,
    candidate_evidence_ids: Optional[Sequence[str]] = None,
    candidate_evidence_moment_ids: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    by_window_id = {
        entry.get("window_id"): entry
        for entry in state.evidence_ledger
        if str(entry.get("window_id") or "").strip()
    }
    by_evidence_id = {entry.get("evidence_id"): entry for entry in state.evidence_ledger}
    by_moment_id: Dict[str, List[Dict[str, Any]]] = {}
    for entry in state.evidence_ledger:
        moment_id = entry.get("moment_id")
        if moment_id is None:
            continue
        by_moment_id.setdefault(str(moment_id), []).append(entry)
    resolved: List[Dict[str, Any]] = []
    seen = set()
    selectors_provided = bool(candidate_window_ids) or bool(candidate_evidence_ids) or bool(candidate_evidence_moment_ids)

    for window_id in candidate_window_ids or []:
        if window_id in by_window_id and window_id not in seen:
            seen.add(window_id)
            resolved.append(by_window_id[window_id])
    for evidence_id in candidate_evidence_ids or []:
        entry = by_evidence_id.get(evidence_id)
        if entry is None:
            continue
        window_id = entry.get("window_id")
        if window_id in seen:
            continue
        seen.add(window_id)
        resolved.append(entry)
    for moment_id in candidate_evidence_moment_ids or []:
        for entry in by_moment_id.get(str(moment_id), []):
            window_id = entry.get("window_id")
            if window_id in seen:
                continue
            seen.add(window_id)
            resolved.append(entry)

    if resolved:
        return resolved
    if selectors_provided:
        return []
    return list(state.evidence_ledger)


def _window_ids(windows: Sequence[Dict[str, Any]]) -> List[str]:
    return [str(entry.get("window_id")) for entry in windows if entry.get("window_id")]


def _role_scores(view_windows: Sequence[Dict[str, Any]], oracle_windows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for oracle in oracle_windows:
        role = str(oracle.get("role") or "").lower()
        oracle_interval = (oracle.get("start_sec"), oracle.get("end_sec"))
        best = 0.0
        for window in view_windows:
            candidate_interval = (window.get("start_sec"), window.get("end_sec"))
            best = max(best, _overlap_ratio(candidate_interval, oracle_interval))
        scores[role] = max(scores.get(role, 0.0), best)
    return scores


def _fallback_view_scores(view_windows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    if not view_windows:
        return {
            "exist_support": 0.0,
            "category_support": 0.0,
            "temporal_support": 0.0,
            "precursor_support": 0.0,
            "finalize_support": 0.0,
            "counterfactual_support": 0.0,
            "overall_support": 0.0,
        }
    coverage = min(1.0, 0.25 + 0.15 * len(view_windows))
    scores = {
        "exist_support": coverage,
        "category_support": coverage * 0.9,
        "temporal_support": coverage * 0.85,
        "precursor_support": coverage * 0.4,
        "finalize_support": coverage * 0.75,
        "counterfactual_support": coverage * 0.8,
    }
    scores["overall_support"] = _weighted_support(scores)
    return scores


def _video_coverage_ratio(view_windows: Sequence[Dict[str, Any]], multimodal_cache: Dict[str, Any]) -> float:
    intervals = _merge_intervals(
        (window.get("start_sec"), window.get("end_sec"))
        for window in view_windows
    )
    if not intervals:
        return 0.0
    covered_duration = sum(max(0.0, end_sec - start_sec) for start_sec, end_sec in intervals)
    duration = 0.0
    try:
        duration = float(multimodal_cache.get("duration") or 0.0)
    except Exception:
        duration = 0.0
    if duration <= 0.0:
        duration = max(float(intervals[-1][1]), 1e-6)
    return max(0.0, min(1.0, covered_duration / max(duration, 1e-6)))


def _fallback_normal_view_scores(
    view_windows: Sequence[Dict[str, Any]],
    multimodal_cache: Dict[str, Any],
) -> Dict[str, float]:
    if not view_windows:
        return {
            "exist_support": 0.0,
            "category_support": 0.0,
            "temporal_support": 0.0,
            "precursor_support": 0.0,
            "finalize_support": 0.0,
            "counterfactual_support": 0.0,
            "overall_support": 0.0,
        }

    coverage = _video_coverage_ratio(view_windows, multimodal_cache)
    search_windows = float(len(view_windows))
    exist_support = min(1.0, 0.25 + 0.60 * coverage + 0.05 * search_windows)
    category_support = min(1.0, 0.20 + 0.55 * coverage + 0.05 * search_windows)
    temporal_support = min(1.0, 0.15 + 0.60 * coverage + 0.05 * max(0.0, search_windows - 1.0))
    finalize_support = min(1.0, 0.20 + 0.60 * coverage + 0.05 * search_windows)
    counterfactual_support = min(1.0, 0.10 + 0.50 * coverage + 0.05 * search_windows)

    scores = {
        "exist_support": round(exist_support, 6),
        "category_support": round(category_support, 6),
        "temporal_support": round(temporal_support, 6),
        "precursor_support": 0.0,
        "finalize_support": round(finalize_support, 6),
        "counterfactual_support": round(counterfactual_support, 6),
    }
    scores["overall_support"] = round(
        _weighted_support(
            scores,
            weights=_support_weights(include_precursor=False),
        ),
        6,
    )
    return scores


def _support_weights(*, include_precursor: bool = True) -> Dict[str, float]:
    weights = dict(DEFAULT_SUPPORT_WEIGHTS)
    if not include_precursor:
        weights.pop("precursor_support", None)
    return weights


def _weighted_support(scores: Dict[str, float], *, weights: Optional[Dict[str, float]] = None) -> float:
    active_weights = dict(weights or DEFAULT_SUPPORT_WEIGHTS)
    total_weight = float(sum(active_weights.values()))
    if total_weight <= 0.0:
        return 0.0
    total = 0.0
    for key, weight in active_weights.items():
        total += float(scores.get(key, 0.0)) * float(weight)
    return max(0.0, min(1.0, total / total_weight))


def _precursor_is_applicable(
    *,
    claim: Dict[str, Any],
    target: Dict[str, Any],
) -> bool:
    precursor_interval = claim.get("precursor_interval_sec")
    if precursor_interval is None:
        precursor_interval = target.get("precursor_interval_sec")
    return _normalize_interval(precursor_interval) is not None


def _renormalize_view_scores(
    view_scores: Dict[str, Dict[str, float]],
    *,
    include_precursor: bool,
) -> Dict[str, Dict[str, float]]:
    weights = _support_weights(include_precursor=include_precursor)
    normalized: Dict[str, Dict[str, float]] = {}
    for view_name, scores in (view_scores or {}).items():
        normalized_scores = dict(scores or {})
        normalized_scores["overall_support"] = round(_weighted_support(normalized_scores, weights=weights), 6)
        normalized[view_name] = normalized_scores
    return normalized


def _score_view(
    view_windows: Sequence[Dict[str, Any]],
    *,
    claim: Dict[str, Any],
    multimodal_cache: Dict[str, Any],
    use_reference_supervision: bool = True,
) -> Dict[str, float]:
    oracle_windows = _extract_oracle_windows(
        multimodal_cache,
        use_reference_supervision=use_reference_supervision,
    )
    target = _extract_target(
        multimodal_cache,
        use_reference_supervision=use_reference_supervision,
    )
    claim_existence = str(claim.get("existence") or target.get("existence") or "").strip().lower()
    if claim_existence == "normal" and not oracle_windows:
        return _fallback_normal_view_scores(view_windows, multimodal_cache)
    if not oracle_windows and not target:
        return _fallback_view_scores(view_windows)

    scores_by_role = _role_scores(view_windows, oracle_windows)
    precursor_score = scores_by_role.get("precursor", 0.0)
    trigger_score = scores_by_role.get("trigger", 0.0)
    peak_score = max(scores_by_role.get("peak_action", 0.0), scores_by_role.get("peak", 0.0))
    confirm_score = max(scores_by_role.get("confirmation", 0.0), scores_by_role.get("aftermath", 0.0))

    anomaly_interval = claim.get("anomaly_interval_sec") or target.get("anomaly_interval_sec")
    precursor_interval = claim.get("precursor_interval_sec") or target.get("precursor_interval_sec")
    anomaly_coverage = _coverage_ratio(view_windows, anomaly_interval)
    precursor_coverage = _coverage_ratio(view_windows, precursor_interval)
    include_precursor = _precursor_is_applicable(claim=claim, target=target)
    required_stages = _required_event_chain_stages(
        claim=claim,
        target=target,
        oracle_windows=oracle_windows,
    )
    available_stage_set = {
        stage
        for stage in (
            "precursor" if precursor_score >= 0.25 else None,
            "trigger" if max(trigger_score, peak_score) >= 0.25 else None,
            "confirmation" if confirm_score >= 0.25 else None,
        )
        if stage is not None
    }
    required_stage_set = set(required_stages)
    if required_stage_set:
        stage_coverage = float(len(required_stage_set & available_stage_set)) / float(len(required_stage_set))
    elif max(trigger_score, peak_score, confirm_score) > 0.0:
        stage_coverage = 1.0
    else:
        stage_coverage = 0.0

    if trigger_score >= 0.25 and (peak_score >= 0.25 or confirm_score >= 0.25):
        category_support = 0.92
    elif trigger_score >= 0.25:
        category_support = 0.78
    elif peak_score >= 0.25 and confirm_score >= 0.25:
        category_support = 0.74
    elif peak_score >= 0.25:
        category_support = 0.48
    elif confirm_score >= 0.25:
        category_support = 0.36
    elif precursor_score >= 0.25:
        category_support = 0.20
    elif anomaly_coverage > 0:
        category_support = 0.10
    else:
        category_support = 0.0

    temporal_support = min(1.0, 0.5 * anomaly_coverage + 0.5 * max(trigger_score, peak_score, confirm_score))
    exist_support = max(category_support, anomaly_coverage * 0.9)
    precursor_support = max(precursor_coverage, precursor_score * 0.8)
    if stage_coverage >= 0.999 and category_support >= 0.78:
        finalize_support = 0.92
    elif stage_coverage >= 0.5 and max(trigger_score, peak_score) >= 0.25:
        finalize_support = 0.72
    elif max(trigger_score, peak_score) >= 0.25:
        finalize_support = 0.48
    elif confirm_score >= 0.25:
        finalize_support = 0.30
    elif precursor_score >= 0.25:
        finalize_support = 0.15
    else:
        finalize_support = 0.0

    counterfactual_support = min(1.0, 0.5 * category_support + 0.5 * temporal_support)

    scores = {
        "exist_support": round(exist_support, 6),
        "category_support": round(category_support, 6),
        "temporal_support": round(temporal_support, 6),
        "precursor_support": round(precursor_support, 6),
        "finalize_support": round(finalize_support, 6),
        "counterfactual_support": round(counterfactual_support, 6),
    }
    scores["overall_support"] = round(
        _weighted_support(
            scores,
            weights=_support_weights(include_precursor=include_precursor),
        ),
        6,
    )
    return scores


def _derive_scores(view_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    full_score = float(view_scores["full"].get("overall_support", 0.0))
    keep_score = float(view_scores["keep"].get("overall_support", 0.0))
    drop_score = float(view_scores["drop"].get("overall_support", 0.0))
    finalize_score = float(view_scores["keep"].get("finalize_support", 0.0))

    sufficiency = keep_score
    necessity = max(0.0, min(1.0, full_score - drop_score))
    consistency = max(0.0, min(1.0, 1.0 - abs(full_score - keep_score)))
    counterfactual_faithfulness = max(0.0, min(1.0, 0.5 * sufficiency + 0.5 * necessity))
    return {
        "sufficiency": round(sufficiency, 6),
        "necessity": round(necessity, 6),
        "consistency": round(consistency, 6),
        "finalize_readiness": round(finalize_score, 6),
        "counterfactual_faithfulness": round(counterfactual_faithfulness, 6),
    }


def _view_payload(
    *,
    view_name: str,
    windows: Sequence[Dict[str, Any]],
    multimodal_cache: Dict[str, Any],
) -> Dict[str, Any]:
    frames = multimodal_cache.get("video")
    fps = float(multimodal_cache.get("fps") or 1.0)
    images = []
    timestamps: List[float] = []
    for window in windows:
        for timestamp in window.get("selected_timestamps") or []:
            timestamps.append(float(timestamp))
            if frames is None:
                continue
            frame_index = int(round(float(timestamp) * fps))
            frame_index = max(0, min(frame_index, len(frames) - 1))
            images.append(frames[frame_index])
    window_summary = "; ".join(
        f"{entry.get('window_id')}[{float(entry.get('start_sec', 0.0)):.3f},{float(entry.get('end_sec', 0.0)):.3f}]"
        for entry in windows
    ) or "no windows"
    return {
        "name": view_name,
        "window_ids": _window_ids(windows),
        "timestamps": timestamps[:8],
        "images": images[:8],
        "summary_text": f"windows={window_summary}",
    }


def _resolve_qwen_verifier_runtime(multimodal_cache: Dict[str, Any]) -> Any:
    if multimodal_cache.get("verifier_runtime") is not None:
        return multimodal_cache["verifier_runtime"]

    model_path = str(multimodal_cache.get("verifier_model_path") or "")
    if not model_path:
        try:
            from saver_v3.core.qwen_verifier import DEFAULT_VERIFIER_MODEL_PATH

            model_path = DEFAULT_VERIFIER_MODEL_PATH
        except Exception:
            model_path = ""
    torch_dtype = str(multimodal_cache.get("verifier_torch_dtype") or "auto")
    device_map = str(multimodal_cache.get("verifier_device_map") or "auto")
    attn_implementation = str(multimodal_cache.get("verifier_attn_implementation") or "")
    max_new_tokens = int(multimodal_cache.get("verifier_max_new_tokens") or 512)
    cache_key = (model_path, torch_dtype, device_map, attn_implementation, max_new_tokens)
    if cache_key in _VERIFIER_RUNTIME_CACHE:
        runtime = _VERIFIER_RUNTIME_CACHE[cache_key]
        multimodal_cache["verifier_runtime"] = runtime
        return runtime

    try:
        from saver_v3.core.qwen_verifier import QwenSelfVerifier

        runtime = QwenSelfVerifier.from_pretrained(
            model_path=model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation or None,
            max_new_tokens=max_new_tokens,
        )
    except Exception as exc:
        raise ValueError("qwen_self_verifier backend requires a loadable verifier runtime") from exc

    _VERIFIER_RUNTIME_CACHE[cache_key] = runtime
    multimodal_cache["verifier_runtime"] = runtime
    return runtime


def _reduce_primary_status(
    *,
    view_scores: Dict[str, Dict[str, float]],
    derived_scores: Dict[str, float],
) -> str:
    keep_scores = view_scores["keep"]
    full_scores = view_scores["full"]
    drop_scores = view_scores["drop"]

    if keep_scores["exist_support"] < 0.2 and keep_scores["temporal_support"] < 0.2:
        return "misaligned"
    if keep_scores["category_support"] < 0.15 and keep_scores["overall_support"] < 0.35:
        return "misaligned"
    if keep_scores["overall_support"] < 0.55 and full_scores["overall_support"] >= 0.60:
        return "incomplete"
    if keep_scores["overall_support"] >= 0.60 and drop_scores["overall_support"] >= 0.65:
        return "redundant"
    if (
        keep_scores["overall_support"] >= 0.60
        and derived_scores["necessity"] >= 0.20
        and derived_scores["consistency"] >= 0.75
    ):
        return "complete"
    return "incomplete"


def _recommended_action(primary_status: str, verification_mode: str) -> str:
    if primary_status == "misaligned":
        return "revise_claim"
    if primary_status == "incomplete":
        return "continue_search"
    if primary_status == "redundant":
        return "refine_evidence"
    if primary_status == "complete":
        if verification_mode in {"final_check", "full_keep_drop", "reward_only", "search_step_check"}:
            return "finalize"
        return "finalize"
    return "continue_search"


def _failure_reasons(primary_status: str) -> List[str]:
    reasons: List[str] = []
    if primary_status == "incomplete":
        reasons.append("selected_evidence_not_sufficient")
    elif primary_status == "redundant":
        reasons.append("selected_evidence_not_necessary_enough")
    elif primary_status == "misaligned":
        reasons.append("selected_evidence_not_aligned_with_claim")
    return reasons


def run_counterfactual_verifier(
    *,
    state: SaverEnvironmentState,
    multimodal_cache: Dict[str, Any],
    verification_mode: str,
    claim: Optional[Dict[str, Any]] = None,
    candidate_window_ids: Optional[Sequence[str]] = None,
    candidate_evidence_ids: Optional[Sequence[str]] = None,
    candidate_evidence_moment_ids: Optional[Sequence[str]] = None,
    alert: Optional[Dict[str, Any]] = None,
    query: str = "",
    backend: str = "qwen_self_verifier",
    use_reference_supervision: bool = True,
) -> Dict[str, Any]:
    requested_backend = str(backend or "qwen_self_verifier")
    if requested_backend != "qwen_self_verifier":
        raise ValueError(f"Unsupported verifier backend: {requested_backend}")
    claim = dict(claim or {})
    target = _extract_target(
        multimodal_cache,
        use_reference_supervision=use_reference_supervision,
    )
    oracle_windows = _extract_oracle_windows(
        multimodal_cache,
        use_reference_supervision=use_reference_supervision,
    )
    merged_claim = {**target, **claim} if use_reference_supervision else dict(claim)
    selected_windows = _resolve_windows(
        state,
        candidate_window_ids=candidate_window_ids,
        candidate_evidence_ids=candidate_evidence_ids,
        candidate_evidence_moment_ids=candidate_evidence_moment_ids,
    )

    selected_window_ids = set(_window_ids(selected_windows))
    full_windows = list(state.evidence_ledger)
    if not full_windows:
        full_windows = [
            entry for entry in state.visited_windows
            if str(entry.get("kind") or "") != "scan"
        ]
    drop_windows = [entry for entry in full_windows if entry.get("window_id") not in selected_window_ids]

    include_precursor = _precursor_is_applicable(claim=merged_claim, target=target)
    runtime = _resolve_qwen_verifier_runtime(multimodal_cache)
    view_payloads = {
        "full": _view_payload(view_name="full", windows=full_windows, multimodal_cache=multimodal_cache),
        "keep": _view_payload(view_name="keep", windows=selected_windows, multimodal_cache=multimodal_cache),
        "drop": _view_payload(view_name="drop", windows=drop_windows, multimodal_cache=multimodal_cache),
    }
    view_scores = runtime.score_views(
        views=view_payloads,
        claim=merged_claim,
        verification_mode=verification_mode,
        question=str(multimodal_cache.get("question") or query),
    )
    view_scores = _renormalize_view_scores(
        view_scores,
        include_precursor=include_precursor,
    )
    derived_scores = _derive_scores(view_scores)
    primary_status = _reduce_primary_status(view_scores=view_scores, derived_scores=derived_scores)
    stage_selected_moment_ids, covered_stages = _selected_event_chain_state(
        selected_windows=selected_windows,
        claim=merged_claim,
        target=target,
        oracle_windows=oracle_windows,
    )
    required_stages = _required_event_chain_stages(
        claim=merged_claim,
        target=target,
        oracle_windows=oracle_windows,
    )
    missing_required_stages = [stage for stage in required_stages if stage not in set(covered_stages)]
    if missing_required_stages and primary_status != "misaligned":
        primary_status = "incomplete"
    recommended_action = _recommended_action(primary_status, verification_mode)

    verified_window_ids = _window_ids(selected_windows)
    selectors_provided = bool(candidate_window_ids) or bool(candidate_evidence_ids) or bool(candidate_evidence_moment_ids)
    best_effort_window_ids = list(verified_window_ids)
    if not best_effort_window_ids and full_windows and not selectors_provided:
        best_effort_window_ids = _window_ids(full_windows[:1])

    failure_reasons = _failure_reasons(primary_status)
    if missing_required_stages:
        failure_reasons.append("missing_required_event_chain_stages")
    explanation = (
        f"Verification mode {verification_mode}: selected evidence is {primary_status} "
        f"for finalize readiness."
    )
    if covered_stages or missing_required_stages:
        explanation += (
            " Event-chain coverage: covered="
            + ",".join(covered_stages or ["none"])
            + " missing="
            + ",".join(missing_required_stages or ["none"])
            + "."
        )

    verdict = {
        "verification_mode": verification_mode,
        "primary_status": primary_status,
        "recommended_action": recommended_action,
        "covered_stages": covered_stages,
        "missing_required_stages": missing_required_stages,
        "stage_selected_moment_ids": stage_selected_moment_ids,
        "view_scores": view_scores,
        "derived_scores": derived_scores,
        "verified_window_ids": verified_window_ids,
        "best_effort_window_ids": best_effort_window_ids,
        "failure_reasons": failure_reasons,
        "explanation": explanation,
        "requested_verifier_backend": requested_backend,
        "verifier_backend": "qwen_self_verifier",
        "candidate_window_ids": list(candidate_window_ids or []),
        "candidate_evidence_ids": list(candidate_evidence_ids or []),
        "candidate_evidence_moment_ids": list(candidate_evidence_moment_ids or []),
        "query": query,
        "claim": merged_claim,
        "use_reference_supervision": bool(use_reference_supervision),
    }
    if primary_status not in PRIMARY_STATUS_VALUES:
        raise ValueError(f"Unexpected verifier primary status: {primary_status}")
    return verdict


def _coerce_state_like(state_like: SaverEnvironmentState | Dict[str, Any]) -> SaverEnvironmentState:
    if isinstance(state_like, SaverEnvironmentState):
        return state_like
    payload = dict(state_like or {})
    return SaverEnvironmentState(
        visited_windows=list(payload.get("visited_windows") or []),
        evidence_ledger=list(payload.get("evidence_ledger") or []),
        verification_records=list(payload.get("verification_records") or []),
        finalized_case=dict(payload["finalized_case"]) if isinstance(payload.get("finalized_case"), dict) else None,
        last_claim=dict(payload["last_claim"]) if isinstance(payload.get("last_claim"), dict) else None,
        active_evidence_window_ids=list(payload.get("active_evidence_window_ids") or []),
        verifier_cache=list(payload.get("verifier_cache") or []),
        next_evidence_id=int(payload.get("next_evidence_id") or 1),
        next_window_id=int(payload.get("next_window_id") or 1),
    )


def _group_relative_advantages(records: List[Dict[str, Any]], *, eps: float = 1e-6) -> List[Dict[str, Any]]:
    if not records:
        return []
    rewards = [float(record.get("branch_reward") or 0.0) for record in records]
    mean_reward = sum(rewards) / float(len(rewards))
    variance = sum((reward - mean_reward) ** 2 for reward in rewards) / float(len(rewards))
    std_reward = variance ** 0.5
    updated: List[Dict[str, Any]] = []
    for record, reward in zip(records, rewards):
        local_advantage = 0.0 if std_reward <= eps else (reward - mean_reward) / (std_reward + eps)
        enriched = dict(record)
        enriched["local_advantage"] = round(float(local_advantage), 6)
        enriched["group_reward_mean"] = round(float(mean_reward), 6)
        enriched["group_reward_std"] = round(float(std_reward), 6)
        updated.append(enriched)
    return updated


def _primary_status_bonus(primary_status: str) -> float:
    return {
        "complete": 0.3,
        "incomplete": -0.2,
        "redundant": -0.3,
        "misaligned": -0.6,
    }.get(str(primary_status), 0.0)


def _evidence_ids_for_window_ids(state: SaverEnvironmentState, window_ids: Sequence[str]) -> List[str]:
    selected = set(str(value) for value in window_ids or [])
    if not selected:
        return []
    evidence_ids: List[str] = []
    for entry in state.evidence_ledger:
        window_id = str(entry.get("window_id") or "")
        evidence_id = entry.get("evidence_id")
        if window_id in selected and evidence_id:
            evidence_ids.append(str(evidence_id))
    return evidence_ids


def _choose_minimal_subset_window_ids(
    state: SaverEnvironmentState,
    selected_window_ids: Sequence[str],
    *,
    subset_size: int = 2,
) -> List[str]:
    entries = [entry for entry in state.evidence_ledger if str(entry.get("window_id") or "") in set(selected_window_ids or [])]
    if not entries:
        return []
    role_priority = {
        "trigger": 0,
        "peak_action": 1,
        "peak": 1,
        "precursor": 2,
        "confirmation": 3,
        "aftermath": 3,
        "": 4,
        "none": 4,
        None: 4,
    }
    entries = sorted(
        entries,
        key=lambda entry: (
            role_priority.get(str(entry.get("role") or "").lower(), 4),
            -float(entry.get("end_sec") or 0.0),
            str(entry.get("window_id") or ""),
        ),
    )
    keep = max(1, min(int(subset_size), len(entries)))
    return [str(entry.get("window_id")) for entry in entries[:keep] if entry.get("window_id")]


def _entry_identifier(entry: Dict[str, Any]) -> Tuple[str, str, float, float]:
    return (
        str(entry.get("window_id") or ""),
        str(entry.get("evidence_id") or ""),
        float(entry.get("start_sec") or 0.0),
        float(entry.get("end_sec") or 0.0),
    )


def _build_delta_only_state(
    state_before: SaverEnvironmentState,
    state_after: SaverEnvironmentState,
) -> SaverEnvironmentState:
    before_visited = {_entry_identifier(entry) for entry in state_before.visited_windows}
    before_evidence = {_entry_identifier(entry) for entry in state_before.evidence_ledger}
    delta_visited = [dict(entry) for entry in state_after.visited_windows if _entry_identifier(entry) not in before_visited]
    delta_evidence = [dict(entry) for entry in state_after.evidence_ledger if _entry_identifier(entry) not in before_evidence]
    delta_window_ids = [str(entry.get("window_id")) for entry in delta_evidence if entry.get("window_id")]
    return SaverEnvironmentState(
        visited_windows=delta_visited,
        evidence_ledger=delta_evidence,
        verification_records=list(state_after.verification_records),
        finalized_case=dict(state_after.finalized_case) if isinstance(state_after.finalized_case, dict) else None,
        last_claim=dict(state_after.last_claim) if isinstance(state_after.last_claim, dict) else None,
        active_evidence_window_ids=delta_window_ids,
        verifier_cache=list(state_after.verifier_cache),
        next_evidence_id=int(state_after.next_evidence_id),
        next_window_id=int(state_after.next_window_id),
    )


def _state_clip_ratio(state: SaverEnvironmentState, duration_sec: float) -> float:
    if duration_sec <= 1e-8:
        return 0.0
    intervals: List[Tuple[float, float]] = []
    source_entries = list(state.visited_windows or []) or list(state.evidence_ledger or [])
    for entry in source_entries:
        try:
            start_sec = float(entry.get("start_sec") or 0.0)
            end_sec = float(entry.get("end_sec") or start_sec)
        except Exception:
            continue
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec
        if end_sec <= start_sec:
            continue
        intervals.append((start_sec, end_sec))
    if not intervals:
        return 0.0
    intervals.sort()
    merged: List[Tuple[float, float]] = []
    current_start, current_end = intervals[0]
    for start_sec, end_sec in intervals[1:]:
        if start_sec <= current_end:
            current_end = max(current_end, end_sec)
            continue
        merged.append((current_start, current_end))
        current_start, current_end = start_sec, end_sec
    merged.append((current_start, current_end))
    covered = sum(max(0.0, end_sec - start_sec) for start_sec, end_sec in merged)
    return min(1.0, max(0.0, covered / float(duration_sec)))


def score_evidence_counterfactual_group(
    *,
    state: SaverEnvironmentState | Dict[str, Any],
    multimodal_cache: Dict[str, Any],
    claim: Dict[str, Any],
    selected_window_ids: Sequence[str],
    anchor_turn_index: int,
    alert: Optional[Dict[str, Any]] = None,
    verifier_backend: str = "qwen_self_verifier",
    use_reference_supervision: bool = False,
    minimal_subset_window_ids: Optional[Sequence[str]] = None,
    subset_size_penalty: float = 0.10,
) -> List[Dict[str, Any]]:
    runtime_state = _coerce_state_like(state)
    full_window_ids = [str(entry.get("window_id")) for entry in runtime_state.evidence_ledger if entry.get("window_id")]
    keep_window_ids = [str(value) for value in selected_window_ids or [] if value]
    drop_window_ids = [window_id for window_id in full_window_ids if window_id not in set(keep_window_ids)]
    minimal_window_ids = (
        [str(value) for value in minimal_subset_window_ids or [] if value]
        if minimal_subset_window_ids
        else _choose_minimal_subset_window_ids(runtime_state, keep_window_ids)
    )
    branch_specs = [
        ("full_ledger", full_window_ids),
        ("keep_selected", keep_window_ids),
        ("drop_selected", drop_window_ids),
        ("minimal_subset", minimal_window_ids),
    ]
    records: List[Dict[str, Any]] = []
    full_ledger_size = max(1, len(full_window_ids))
    for branch_type, branch_window_ids in branch_specs:
        verdict = run_counterfactual_verifier(
            state=runtime_state,
            multimodal_cache=multimodal_cache,
            verification_mode="full_keep_drop",
            claim=claim,
            candidate_window_ids=branch_window_ids,
            alert=alert,
            backend=verifier_backend,
            use_reference_supervision=use_reference_supervision,
        )
        derived = verdict.get("derived_scores") or {}
        subset_ratio = float(len(branch_window_ids)) / float(full_ledger_size)
        reward_components = {
            "sufficiency_reward": 0.35 * float(derived.get("sufficiency", 0.0) or 0.0),
            "necessity_reward": 0.35 * float(derived.get("necessity", 0.0) or 0.0),
            "counterfactual_reward": 0.20 * float(derived.get("counterfactual_faithfulness", 0.0) or 0.0),
            "primary_status_bonus": 0.10 * _primary_status_bonus(str(verdict.get("primary_status") or "")),
            "subset_size_penalty": -abs(float(subset_size_penalty)) * subset_ratio,
        }
        branch_reward = round(sum(float(value) for value in reward_components.values()), 6)
        records.append(
            {
                "group_kind": "evidence",
                "branch_type": branch_type,
                "anchor_turn_index": int(anchor_turn_index),
                "source_turn_index": int(anchor_turn_index),
                "selected_window_ids": list(branch_window_ids),
                "selected_evidence_ids": _evidence_ids_for_window_ids(runtime_state, branch_window_ids),
                "branch_reward_components": reward_components,
                "branch_reward": branch_reward,
                "verifier_verdict": verdict,
                "primary_status": verdict.get("primary_status"),
            }
        )
    return _group_relative_advantages(records)


def score_search_counterfactual_group(
    *,
    state_before: SaverEnvironmentState | Dict[str, Any],
    state_after: SaverEnvironmentState | Dict[str, Any],
    multimodal_cache: Dict[str, Any],
    claim: Dict[str, Any],
    anchor_turn_index: int,
    alert: Optional[Dict[str, Any]] = None,
    verifier_backend: str = "qwen_self_verifier",
    use_reference_supervision: bool = False,
    search_cost_penalty: float = 0.10,
) -> List[Dict[str, Any]]:
    runtime_before = _coerce_state_like(state_before)
    runtime_after = _coerce_state_like(state_after)
    delta_only_state = _build_delta_only_state(runtime_before, runtime_after)
    duration_sec = float(multimodal_cache.get("duration") or 0.0)
    before_window_ids = {
        str(entry.get("window_id") or "")
        for entry in runtime_before.evidence_ledger
        if str(entry.get("window_id") or "").strip()
    }
    total_new_window_ids = {
        str(entry.get("window_id") or "")
        for entry in runtime_after.evidence_ledger
        if str(entry.get("window_id") or "").strip() and str(entry.get("window_id") or "") not in before_window_ids
    }
    branch_specs = [
        ("skip_search", runtime_before),
        ("use_search", runtime_after),
        ("delta_only", delta_only_state),
    ]
    records: List[Dict[str, Any]] = []
    for branch_type, branch_state in branch_specs:
        verdict = run_counterfactual_verifier(
            state=branch_state,
            multimodal_cache=multimodal_cache,
            verification_mode="search_step_check",
            claim=claim,
            alert=alert,
            backend=verifier_backend,
            use_reference_supervision=use_reference_supervision,
        )
        derived = verdict.get("derived_scores") or {}
        clip_ratio = _state_clip_ratio(branch_state, duration_sec)
        branch_new_window_ids = {
            str(entry.get("window_id") or "")
            for entry in branch_state.evidence_ledger
            if str(entry.get("window_id") or "").strip() and str(entry.get("window_id") or "") not in before_window_ids
        }
        incremental_evidence_reward = 0.0
        if total_new_window_ids:
            incremental_evidence_reward = 0.10 * (
                float(len(branch_new_window_ids)) / float(len(total_new_window_ids))
            )
        reward_components = {
            "sufficiency_reward": 0.35 * float(derived.get("sufficiency", 0.0) or 0.0),
            "necessity_reward": 0.20 * float(derived.get("necessity", 0.0) or 0.0),
            "counterfactual_reward": 0.20 * float(derived.get("counterfactual_faithfulness", 0.0) or 0.0),
            "finalize_readiness_reward": 0.10 * float(derived.get("finalize_readiness", 0.0) or 0.0),
            "primary_status_bonus": 0.15 * _primary_status_bonus(str(verdict.get("primary_status") or "")),
            "incremental_evidence_reward": incremental_evidence_reward,
            "search_cost_penalty": -abs(float(search_cost_penalty)) * clip_ratio,
        }
        branch_reward = round(sum(float(value) for value in reward_components.values()), 6)
        selected_window_ids = (
            list(verdict.get("verified_window_ids") or verdict.get("best_effort_window_ids") or [])
            or [str(value) for value in branch_state.active_evidence_window_ids if value]
        )
        records.append(
            {
                "group_kind": "search",
                "branch_type": branch_type,
                "anchor_turn_index": int(anchor_turn_index),
                "source_turn_index": int(anchor_turn_index),
                "selected_window_ids": selected_window_ids,
                "selected_evidence_ids": _evidence_ids_for_window_ids(branch_state, selected_window_ids),
                "branch_reward_components": reward_components,
                "branch_reward": branch_reward,
                "verifier_verdict": verdict,
                "primary_status": verdict.get("primary_status"),
            }
        )
    return _group_relative_advantages(records)
