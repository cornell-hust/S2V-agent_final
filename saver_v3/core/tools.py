from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from saver_v3.core.categories import canonicalize_category_payload, validate_canonical_category_payload
from saver_v3.core.protocol_guidance import event_chain_stage_for_role, summarize_evidence_ledger
from saver_v3.core.proposal import (
    coerce_feature_cache_payload,
    feature_guided_frame_proposal,
    normalize_query_text,
)
from saver_v3.core.schema import SaverEnvironmentState, validate_required_fields
from saver_v3.core.semantic_answer import augment_finalize_case_schema, split_finalize_case_payload
from saver_v3.core.self_verification import (
    normalize_self_verification_mode,
    parse_self_verification_payload,
    validate_policy_self_verification_payload,
)


MAX_NUM_KEY_FRAMES = 8
SELF_VERIFICATION_VERDICT_KEYS = (
    "verification_decision",
    "primary_status",
    "recommended_action",
    "sufficiency_score",
    "necessity_score",
    "finalize_readiness_score",
    "counterfactual_faithfulness",
    "derived_scores",
    "failure_reasons",
    "rationale",
    "explanation",
)
OPTIONAL_FINALIZE_SEMANTIC_FIELDS = (
    "summary",
    "rationale",
    "event_chain_summary",
    "qa_focus_answers",
)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _frame_bounds(multimodal_cache: Dict) -> Tuple[int, float, int]:
    frames = multimodal_cache.get("video")
    fps = float(multimodal_cache.get("fps") or 1.0)
    if frames is not None:
        total_frames = int(frames.shape[0])
    else:
        duration = float(multimodal_cache.get("duration") or 0.0)
        total_frames = max(int(math.ceil(duration * fps)), 1)
    return total_frames, fps, max(total_frames - 1, 0)


def _select_uniform_indices(start_idx: int, end_idx: int, num_frames: int) -> List[int]:
    if end_idx < start_idx:
        return []
    total = end_idx - start_idx + 1
    num_frames = max(1, min(int(num_frames), total, MAX_NUM_KEY_FRAMES))
    if total <= num_frames:
        return list(range(start_idx, end_idx + 1))
    return np.round(np.linspace(start_idx, end_idx, num_frames)).astype(int).tolist()


def _select_stride_indices(start_idx: int, end_idx: int, stride_frames: int) -> List[int]:
    if end_idx < start_idx:
        return []
    stride_frames = max(1, int(stride_frames))
    indices = list(range(start_idx, end_idx + 1, stride_frames))
    if not indices or indices[-1] != end_idx:
        indices.append(end_idx)
    indices = list(dict.fromkeys(int(index) for index in indices))
    if len(indices) > MAX_NUM_KEY_FRAMES:
        return _select_uniform_indices(indices[0], indices[-1], MAX_NUM_KEY_FRAMES)
    return indices


def _resolve_window(args: Dict[str, Any], multimodal_cache: Dict) -> Tuple[float, float, List[int], float]:
    total_frames, fps, last_frame_idx = _frame_bounds(multimodal_cache)
    duration = float(multimodal_cache.get("duration") or (total_frames / fps))
    start_sec = max(0.0, _coerce_float(args.get("start_sec", args.get("start_time", 0.0)), 0.0))
    end_sec = _coerce_float(args.get("end_sec", args.get("end_time", duration)), duration)
    end_sec = min(max(start_sec, end_sec), duration)
    start_idx = max(0, min(int(math.floor(start_sec * fps)), last_frame_idx))
    end_idx = max(start_idx, min(int(math.floor(end_sec * fps)), last_frame_idx))
    if args.get("stride_sec") is not None:
        stride_sec = _coerce_float(args.get("stride_sec"), 0.0)
        stride_frames = max(1, int(round(stride_sec * fps))) if stride_sec > 0 else 1
        selected_indices = _select_stride_indices(start_idx, end_idx, stride_frames)
    else:
        num_frames = _coerce_int(args.get("num_frames", MAX_NUM_KEY_FRAMES), MAX_NUM_KEY_FRAMES)
        selected_indices = _select_uniform_indices(start_idx, end_idx, num_frames)
    return start_sec, end_sec, selected_indices, fps


def _indices_to_timestamps(indices: List[int], fps: float) -> List[float]:
    return [round(float(idx) / fps, 6) for idx in indices]


def _build_visual_content(indices: List[int], multimodal_cache: Dict, footer: str) -> List[Dict[str, Any]]:
    fps = float(multimodal_cache.get("fps") or 1.0)
    frames = multimodal_cache.get("video")
    frame_indices = multimodal_cache.get("frame_indices") or []
    video_path = str(multimodal_cache.get("video_path") or "").strip()
    timestamps = _indices_to_timestamps(indices, fps)
    content: List[Dict[str, Any]] = []
    for i, timestamp in zip(indices, timestamps):
        content.append({"type": "text", "text": f"{timestamp:.3f}s"})
        if frames is not None and 0 <= i < len(frames):
            image_item = {
                "type": "image",
                "image": frames[i],
                "sampled_frame_index": int(i),
                "timestamp_sec": float(timestamp),
            }
            if 0 <= int(i) < len(frame_indices):
                image_item["raw_frame_index"] = int(frame_indices[int(i)])
            content.append(image_item)
        elif video_path:
            raw_frame_index = int(frame_indices[int(i)]) if 0 <= int(i) < len(frame_indices) else int(i)
            image_ref = {
                "video_path": str(video_path),
                "raw_frame_index": int(raw_frame_index),
                "timestamp_sec": float(timestamp),
            }
            if 0 <= int(i) < len(frame_indices):
                image_ref["sampled_frame_index"] = int(i)
            content.append(
                {
                    "type": "image",
                    "image_ref": image_ref,
                    "timestamp_sec": float(timestamp),
                }
            )
    content.append({"type": "text", "text": footer})
    return content


def _dedupe_string_list(values: List[Any] | Tuple[Any, ...] | None) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for value in values or []:
        text = str(value).strip()
        if not text or text in seen:
            continue
        deduped.append(text)
        seen.add(text)
    return deduped


def _normalize_interval_tuple(value: Any) -> Tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start_sec = float(value[0])
        end_sec = float(value[1])
    except Exception:
        return None
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec
    return start_sec, end_sec


def _interval_iou(interval_a: Any, interval_b: Any) -> float:
    a = _normalize_interval_tuple(interval_a)
    b = _normalize_interval_tuple(interval_b)
    if a is None or b is None:
        return 0.0
    overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    if overlap <= 0.0:
        return 0.0
    union = max(a[1] - a[0], 0.0) + max(b[1] - b[0], 0.0) - overlap
    if union <= 0.0:
        return 0.0
    return max(0.0, min(1.0, overlap / union))


def _oracle_moment_stage_lookup(multimodal_cache: Dict[str, Any]) -> Dict[str, str]:
    stage_by_moment_id: Dict[str, str] = {}
    structured_target = dict(multimodal_cache.get("structured_target") or {})
    chain_target = dict(structured_target.get("event_chain_target") or {})
    for stage, moment_ids in dict(chain_target.get("stage_to_moment_ids") or {}).items():
        normalized_stage = event_chain_stage_for_role(stage) or str(stage or "").strip().lower()
        if normalized_stage not in {"precursor", "trigger", "confirmation"}:
            continue
        for moment_id in list(moment_ids or []):
            moment_id_text = str(moment_id or "").strip()
            if moment_id_text:
                stage_by_moment_id[moment_id_text] = normalized_stage
    for entry in list(((multimodal_cache.get("tool_io") or {}).get("oracle_windows_sec")) or []):
        moment_id = str(entry.get("moment_id") or "").strip()
        stage = event_chain_stage_for_role(entry.get("role"))
        if moment_id and stage:
            stage_by_moment_id[moment_id] = stage
    return stage_by_moment_id


def _oracle_window_entries(multimodal_cache: Dict[str, Any]) -> List[Dict[str, Any]]:
    windows: List[Dict[str, Any]] = []
    for entry in list(((multimodal_cache.get("tool_io") or {}).get("oracle_windows_sec")) or []):
        moment_id = str(entry.get("moment_id") or "").strip()
        interval = entry.get("window") or entry.get("window_sec")
        normalized_interval = _normalize_interval_tuple(interval)
        stage = event_chain_stage_for_role(entry.get("role"))
        if not moment_id or normalized_interval is None:
            continue
        windows.append(
            {
                "moment_id": moment_id,
                "stage": stage,
                "interval": normalized_interval,
            }
        )
    return windows


def _runtime_entry_stage(entry: Dict[str, Any], *, stage_by_moment_id: Dict[str, str]) -> str:
    stage = event_chain_stage_for_role(entry.get("role"))
    if stage:
        return stage
    moment_id = str(entry.get("moment_id") or "").strip()
    return stage_by_moment_id.get(moment_id, "")


def _selected_evidence_entries_for_finalization(
    state: SaverEnvironmentState,
    decision_arguments: Dict[str, Any],
) -> List[Dict[str, Any]]:
    by_window_id = {
        str(entry.get("window_id")): entry
        for entry in state.evidence_ledger
        if str(entry.get("window_id") or "").strip()
    }
    by_evidence_id = {
        str(entry.get("evidence_id")): entry
        for entry in state.evidence_ledger
        if str(entry.get("evidence_id") or "").strip()
    }
    selected_ids = _dedupe_string_list(
        list(decision_arguments.get("selected_window_ids") or [])
        + list(decision_arguments.get("verified_window_ids") or [])
        + list(decision_arguments.get("best_effort_window_ids") or [])
        + list(getattr(state, "active_evidence_window_ids", []) or [])
    )
    selected_evidence_ids = _dedupe_string_list(
        list(decision_arguments.get("selected_evidence_ids") or [])
        + [
            value
            for value in list(decision_arguments.get("evidence_moment_ids") or [])
            if str(value or "").strip().startswith("e")
        ]
    )
    selected: List[Dict[str, Any]] = []
    seen_window_ids = set()
    for window_id in selected_ids:
        entry = by_window_id.get(str(window_id))
        if entry is None:
            continue
        resolved_window_id = str(entry.get("window_id") or "").strip()
        if not resolved_window_id or resolved_window_id in seen_window_ids:
            continue
        seen_window_ids.add(resolved_window_id)
        selected.append(entry)
    for evidence_id in selected_evidence_ids:
        entry = by_evidence_id.get(str(evidence_id))
        if entry is None:
            continue
        resolved_window_id = str(entry.get("window_id") or "").strip()
        if not resolved_window_id or resolved_window_id in seen_window_ids:
            continue
        seen_window_ids.add(resolved_window_id)
        selected.append(entry)
    if selected:
        return selected
    return list(state.evidence_ledger or [])


def _resolve_finalized_moment_ids_from_state(
    *,
    decision_arguments: Dict[str, Any],
    multimodal_cache: Dict[str, Any],
    state: SaverEnvironmentState,
) -> Tuple[List[str], Dict[str, List[str]]]:
    stage_by_moment_id = _oracle_moment_stage_lookup(multimodal_cache)
    valid_moment_ids = set(stage_by_moment_id)
    selected_entries = _selected_evidence_entries_for_finalization(state, decision_arguments)
    selected_runtime_moment_ids = _dedupe_string_list([entry.get("moment_id") for entry in selected_entries])
    requested_moment_ids = _dedupe_string_list(
        [
            value
            for value in list(decision_arguments.get("evidence_moment_ids") or [])
            if str(value or "").strip() in valid_moment_ids
        ]
    )
    moment_ids = _dedupe_string_list(requested_moment_ids + [value for value in selected_runtime_moment_ids if value in valid_moment_ids])

    if not moment_ids:
        oracle_windows = _oracle_window_entries(multimodal_cache)
        for entry in selected_entries:
            entry_interval = _normalize_interval_tuple((entry.get("start_sec"), entry.get("end_sec")))
            entry_stage = _runtime_entry_stage(entry, stage_by_moment_id=stage_by_moment_id)
            best: Tuple[float, str] = (0.0, "")
            for oracle in oracle_windows:
                oracle_stage = str(oracle.get("stage") or "")
                if entry_stage and oracle_stage and entry_stage != oracle_stage:
                    continue
                score = _interval_iou(entry_interval, oracle.get("interval"))
                if score > best[0]:
                    best = (score, str(oracle.get("moment_id") or ""))
            if best[0] > 0.0 and best[1]:
                moment_ids.append(best[1])
        moment_ids = _dedupe_string_list(moment_ids)

    if not moment_ids:
        moment_ids = _dedupe_string_list(
            [
                value
                for value in list(decision_arguments.get("evidence_moment_ids") or [])
                if str(value or "").strip() in valid_moment_ids
            ]
        )

    stage_selected: Dict[str, List[str]] = {}
    for moment_id in moment_ids:
        stage = stage_by_moment_id.get(moment_id)
        if not stage:
            continue
        stage_selected.setdefault(stage, [])
        if moment_id not in stage_selected[stage]:
            stage_selected[stage].append(moment_id)
    return moment_ids, stage_selected


def _append_window(
    state: SaverEnvironmentState,
    *,
    kind: str,
    query: str | None,
    query_normalized: str | None,
    query_source: str | None,
    moment_id: str | None,
    role: str | None,
    start_sec: float,
    end_sec: float,
    selected_indices: List[int],
    fps: float,
    record_as_evidence: bool = True,
    search_anchor: Dict[str, Any] | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    window_id = f"w{state.next_window_id:04d}"
    state.next_window_id += 1
    evidence_id = f"e{state.next_evidence_id:04d}"
    state.next_evidence_id += 1
    entry = {
        "window_id": window_id,
        "evidence_id": evidence_id,
        "kind": kind,
        "query": query,
        "query_normalized": query_normalized,
        "query_source": query_source,
        "moment_id": moment_id,
        "role": role,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "selected_frame_indices": [int(index) for index in selected_indices],
        "selected_timestamps": _indices_to_timestamps(selected_indices, fps),
        "selected_frame_count": len(selected_indices),
    }
    if search_anchor:
        entry["search_anchor"] = dict(search_anchor)
    if metadata:
        entry.update(dict(metadata))
    state.visited_windows.append(entry)
    if record_as_evidence:
        state.evidence_ledger.append(entry)
    return entry


def _resolve_scene_text(multimodal_cache: Dict[str, Any]) -> str:
    raw_scene = multimodal_cache.get("scene")
    if isinstance(raw_scene, dict):
        return str(raw_scene.get("scenario") or raw_scene.get("name") or "").strip()
    if isinstance(raw_scene, str):
        return raw_scene.strip()
    return ""


def _build_active_hypothesis_text(state: SaverEnvironmentState) -> str:
    claim = dict(state.last_claim or {})
    existence = str(claim.get("existence") or "").strip().lower()
    category = str(claim.get("category") or "").strip().lower()
    if existence == "normal":
        return "checking whether the clip remains normal"
    if existence == "anomaly" and category and category != "normal":
        return f"possible {category}"
    if category and category != "normal":
        return f"possible {category}"
    return ""


def _latest_search_feedback(state: SaverEnvironmentState) -> str:
    if not state.evidence_ledger:
        return ""
    latest = dict(state.evidence_ledger[-1] or {})
    query = str(latest.get("query_normalized") or latest.get("query") or "").strip()
    role = str(latest.get("role") or "").strip()
    if query and role:
        return f"previous search already checked {role} evidence for {query}"
    if query:
        return f"previous search already checked {query}"
    return ""


def _expected_stage_for_search(arguments: Dict[str, Any], state: SaverEnvironmentState) -> str:
    role_stage = event_chain_stage_for_role(arguments.get("role"))
    if role_stage:
        return role_stage
    if state.verification_records:
        latest_verification = dict(state.verification_records[-1] or {})
        missing_required_stages = list(latest_verification.get("missing_required_stages") or [])
        if missing_required_stages:
            return str(missing_required_stages[0]).strip()
    return ""


def _build_search_anchor(
    arguments: Dict[str, Any],
    *,
    multimodal_cache: Dict[str, Any],
    state: SaverEnvironmentState,
) -> Dict[str, Any]:
    latest_verification = dict(state.verification_records[-1] or {}) if state.verification_records else {}
    unresolved_targets = [
        str(value).strip()
        for value in list(latest_verification.get("missing_required_stages") or [])
        if str(value).strip()
    ]
    anchor = {
        "scene": _resolve_scene_text(multimodal_cache),
        "active_hypothesis": _build_active_hypothesis_text(state),
        "expected_stage": _expected_stage_for_search(arguments, state),
        "unresolved_targets": unresolved_targets,
        "resolved_evidence_summary": summarize_evidence_ledger(state.evidence_ledger),
        "last_search_feedback": _latest_search_feedback(state),
    }
    cleaned_anchor: Dict[str, Any] = {}
    for key, value in anchor.items():
        if value is None:
            continue
        if isinstance(value, str) and not value:
            continue
        if isinstance(value, list) and not value:
            continue
        cleaned_anchor[key] = value
    return cleaned_anchor


def _has_self_verification_payload(arguments: Dict[str, Any]) -> bool:
    return any(key in arguments for key in SELF_VERIFICATION_VERDICT_KEYS)


def _normalize_verification_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(arguments or {})
    normalized["verification_mode"] = normalize_self_verification_mode(
        normalized.get("verification_mode"),
        default="reward_only",
        public_only=True,
    )
    if isinstance(normalized.get("claim"), dict):
        normalized["claim"] = validate_canonical_category_payload(
            normalized["claim"],
            payload_name="claim",
            require_category_for_anomaly=True,
        )
    return normalized


def _resolve_selected_window_ids(
    state: SaverEnvironmentState,
    *,
    selected_window_ids: List[str],
    selected_evidence_ids: List[str],
    selected_evidence_moment_ids: List[str],
    candidate_window_ids: List[str],
) -> Dict[str, Any]:
    requested_selected_window_ids = _dedupe_string_list(selected_window_ids)
    selected_evidence_ids = _dedupe_string_list(selected_evidence_ids)
    selected_evidence_moment_ids = _dedupe_string_list(selected_evidence_moment_ids)
    candidate_window_ids = _dedupe_string_list(candidate_window_ids)
    raw_window_selector_ids = _dedupe_string_list(requested_selected_window_ids + candidate_window_ids)

    resolved: List[str] = []
    seen = set()
    invalid_selected_window_ids: List[str] = []
    resolution_sources: List[str] = []
    valid_window_ids = {
        str(entry.get("window_id")).strip()
        for entry in state.evidence_ledger
        if str(entry.get("window_id") or "").strip()
    }
    by_evidence_id = {
        str(entry.get("evidence_id")): entry for entry in state.evidence_ledger if entry.get("evidence_id")
    }
    by_moment_id: Dict[str, List[Dict[str, Any]]] = {}
    for entry in state.evidence_ledger:
        moment_id = entry.get("moment_id")
        if moment_id is None:
            continue
        by_moment_id.setdefault(str(moment_id), []).append(entry)

    selected_window_added = False
    for raw_window_id in requested_selected_window_ids:
        if raw_window_id not in valid_window_ids:
            if raw_window_id not in invalid_selected_window_ids:
                invalid_selected_window_ids.append(raw_window_id)
            continue
        if raw_window_id not in seen:
            seen.add(raw_window_id)
            resolved.append(raw_window_id)
            selected_window_added = True
    if selected_window_added:
        resolution_sources.append("selected_window_ids")

    evidence_added = False
    for evidence_id in selected_evidence_ids:
        entry = by_evidence_id.get(str(evidence_id))
        if entry is None:
            continue
        window_id = str(entry.get("window_id") or "").strip()
        if window_id and window_id not in seen:
            seen.add(window_id)
            resolved.append(window_id)
            evidence_added = True
    if evidence_added:
        resolution_sources.append("selected_evidence_ids")

    moment_added = False
    for moment_id in selected_evidence_moment_ids:
        for entry in by_moment_id.get(str(moment_id), []):
            window_id = str(entry.get("window_id") or "").strip()
            if window_id and window_id not in seen:
                seen.add(window_id)
                resolved.append(window_id)
                moment_added = True
    if moment_added:
        resolution_sources.append("selected_evidence_moment_ids")

    valid_candidate_window_ids: List[str] = []
    candidate_added = False
    candidate_fallback_allowed = not resolved
    for window_id in candidate_window_ids:
        if window_id not in valid_window_ids:
            continue
        if window_id not in valid_candidate_window_ids:
            valid_candidate_window_ids.append(window_id)
        if not candidate_fallback_allowed:
            continue
        if window_id not in seen:
            seen.add(window_id)
            resolved.append(window_id)
            candidate_added = True
    if candidate_added:
        resolution_sources.append("candidate_window_ids")

    selection_requested = bool(
        requested_selected_window_ids or selected_evidence_ids or selected_evidence_moment_ids or candidate_window_ids
    )
    # Fallback: when nothing resolves but evidence exists in ledger, auto-select all evidence windows
    if not resolved and valid_window_ids:
        for fallback_window_id in sorted(valid_window_ids):
            if fallback_window_id not in seen:
                seen.add(fallback_window_id)
                resolved.append(fallback_window_id)
        if resolved:
            resolution_sources.append("auto_fallback_from_ledger")

    if resolution_sources:
        selection_resolution_source = "+".join(resolution_sources)
    elif selection_requested:
        selection_resolution_source = "unresolved"
    else:
        selection_resolution_source = "none"

    return {
        "resolved_window_ids": resolved,
        "requested_selected_window_ids": requested_selected_window_ids,
        "invalid_selected_window_ids": invalid_selected_window_ids,
        "selection_resolution_source": selection_resolution_source,
        "selection_requested": selection_requested,
        "selection_unresolved": bool(selection_requested and not resolved),
        "selected_evidence_ids": selected_evidence_ids,
        "selected_evidence_moment_ids": selected_evidence_moment_ids,
        "valid_candidate_window_ids": valid_candidate_window_ids,
        "window_selector_ids": raw_window_selector_ids,
    }


def _finalize_verification_payload(
    verification: Dict[str, Any],
    *,
    selection_info: Dict[str, Any],
    verification_parse_mode: str,
) -> Dict[str, Any]:
    finalized = dict(verification or {})
    finalized["verified_window_ids"] = _dedupe_string_list(finalized.get("verified_window_ids") or [])
    finalized["best_effort_window_ids"] = _dedupe_string_list(finalized.get("best_effort_window_ids") or [])
    if finalized.get("self_verification_selected_window_ids") is not None:
        finalized["self_verification_selected_window_ids"] = _dedupe_string_list(
            finalized.get("self_verification_selected_window_ids") or []
        )

    failure_reasons = _dedupe_string_list(finalized.get("failure_reasons") or [])
    if bool(selection_info.get("selection_unresolved")):
        failure_reasons = _dedupe_string_list(
            failure_reasons + ["selected_evidence_not_resolved_to_known_windows"]
        )
        finalized["verified_window_ids"] = []
        finalized["best_effort_window_ids"] = []
        if finalized.get("self_verification_selected_window_ids") is not None:
            finalized["self_verification_selected_window_ids"] = []
    finalized["failure_reasons"] = failure_reasons
    finalized["verification_parse_mode"] = str(verification_parse_mode)
    finalized["requested_selected_window_ids"] = list(selection_info.get("requested_selected_window_ids") or [])
    finalized["invalid_selected_window_ids"] = list(selection_info.get("invalid_selected_window_ids") or [])
    finalized["selection_resolution_source"] = str(selection_info.get("selection_resolution_source") or "none")
    return finalized


def scan_timeline(arguments: Dict[str, Any], multimodal_cache: Dict, state: SaverEnvironmentState):
    start_sec, end_sec, selected_indices, fps = _resolve_window(arguments, multimodal_cache)
    entry = _append_window(
        state,
        kind="scan",
        query=arguments.get("purpose"),
        query_normalized=normalize_query_text(str(arguments.get("purpose") or "")),
        query_source="scan_purpose",
        moment_id=None,
        role=None,
        start_sec=start_sec,
        end_sec=end_sec,
        selected_indices=selected_indices,
        fps=fps,
        record_as_evidence=False,
        search_anchor=None,
        metadata={
            "proposal_backend": "uniform",
            "feature_cache_used": False,
            "proposal_candidate_count": 0,
            "proposal_candidate_frame_indices": [int(index) for index in selected_indices],
            "proposal_candidate_frame_scores": [],
            "proposal_candidate_windows": [],
            "selected_frame_scores": [],
            "proposal_fallback_reason": "scan_timeline_uniform",
        },
    )
    footer = (
        f"Scanned timeline window [{start_sec:.3f}, {end_sec:.3f}] and selected "
        f"{len(selected_indices)} frames."
    )
    return _build_visual_content(selected_indices, multimodal_cache, footer), state, entry


def seek_evidence(arguments: Dict[str, Any], multimodal_cache: Dict, state: SaverEnvironmentState):
    start_sec, end_sec, selected_indices, fps = _resolve_window(arguments, multimodal_cache)
    query = str(arguments.get("query") or "").strip()
    search_anchor = _build_search_anchor(arguments, multimodal_cache=multimodal_cache, state=state)
    feature_cache = coerce_feature_cache_payload(
        multimodal_cache.get("embedding"),
        fps=fps,
        frame_indices=multimodal_cache.get("frame_indices") or [],
    )
    if bool(multimodal_cache.get("strict_feature_guided_proposal")):
        if feature_cache is None:
            raise ValueError(
                "seek_evidence requires feature_cache when strict_feature_guided_proposal is enabled."
            )
        if multimodal_cache.get("proposal_runtime") is None:
            raise ValueError(
                "seek_evidence requires proposal_runtime when strict_feature_guided_proposal is enabled."
            )
    proposal_metadata = feature_guided_frame_proposal(
        feature_cache=feature_cache,
        proposal_runtime=multimodal_cache.get("proposal_runtime"),
        query=query,
        query_package=None,
        role=str(arguments.get("role") or ""),
        search_anchor=search_anchor,
        start_sec=start_sec,
        end_sec=end_sec,
        fps=fps,
        num_frames=int(arguments.get("num_frames") or 0),
        top_k_candidates=int(arguments.get("top_k_candidates") or 8),
        candidate_merge_gap_sec=float(arguments.get("candidate_merge_gap_sec") or 1.0),
        query_source=str(arguments.get("query_source") or "model"),
    )
    if proposal_metadata.get("selected_frame_indices"):
        selected_indices = [int(index) for index in proposal_metadata["selected_frame_indices"]]
    else:
        proposal_metadata["selected_frame_indices"] = [int(index) for index in selected_indices]
    entry = _append_window(
        state,
        kind="evidence",
        query=query,
        query_normalized=str(proposal_metadata.get("query_normalized") or normalize_query_text(query)),
        query_source=str(proposal_metadata.get("query_source") or arguments.get("query_source") or "model"),
        moment_id=str(arguments.get("moment_id")) if arguments.get("moment_id") is not None else None,
        role=str(arguments.get("role")) if arguments.get("role") is not None else None,
        start_sec=start_sec,
        end_sec=end_sec,
        selected_indices=selected_indices,
        fps=fps,
        record_as_evidence=True,
        search_anchor=search_anchor,
        metadata=proposal_metadata,
    )
    evidence_window_id = str(entry.get("window_id") or "").strip()
    evidence_id = str(entry.get("evidence_id") or "").strip()
    moment_id = str(entry.get("moment_id") or "").strip()
    role = str(entry.get("role") or "").strip()
    registration_parts = []
    if evidence_window_id:
        registration_parts.append(f"window_id={evidence_window_id}")
    if evidence_id:
        registration_parts.append(f"evidence_id={evidence_id}")
    if role:
        registration_parts.append(f"role={role}")
    if moment_id:
        registration_parts.append(f"moment_id={moment_id}")
    verification_hint = ""
    if evidence_window_id:
        verification_hint = f' For verify_hypothesis use selected_window_ids=["{evidence_window_id}"].'
        if moment_id:
            verification_hint += f' You may also include selected_evidence_moment_ids=["{moment_id}"].'
    ledger_summary = summarize_evidence_ledger(state.evidence_ledger)
    stage_hint = str(search_anchor.get("expected_stage") or role or "").strip()
    stage_hint_text = f" Stage hint: {stage_hint}." if stage_hint else ""
    footer = (
        f"Evidence registered: {' '.join(registration_parts)}. "
        f"Query='{query}'. Window [{start_sec:.3f}, {end_sec:.3f}]. Selected {len(selected_indices)} frames."
        f"{verification_hint}"
        f"{stage_hint_text}"
        f" {ledger_summary}"
    )
    return _build_visual_content(selected_indices, multimodal_cache, footer), state, entry


def verify_hypothesis(arguments: Dict[str, Any], multimodal_cache: Dict, state: SaverEnvironmentState):
    arguments = _normalize_verification_arguments(arguments)
    selected_window_ids = [str(value) for value in (arguments.get("selected_window_ids") or []) if str(value).strip()]
    selected_evidence_ids = [str(value) for value in (arguments.get("selected_evidence_ids") or []) if str(value).strip()]
    selected_evidence_moment_ids = [
        str(value)
        for value in (
            arguments.get("selected_evidence_moment_ids")
            or arguments.get("evidence_moment_ids")
            or []
        )
        if str(value).strip()
    ]
    candidate_window_ids = [str(value) for value in (arguments.get("candidate_window_ids") or []) if str(value).strip()]
    candidate_evidence_ids = [
        str(value)
        for value in (arguments.get("candidate_evidence_ids") or arguments.get("evidence_ids") or [])
        if str(value).strip()
    ]
    candidate_evidence_moment_ids = [
        str(value)
        for value in (
            arguments.get("candidate_evidence_moment_ids")
            or arguments.get("evidence_moment_ids")
            or arguments.get("selected_evidence_moment_ids")
            or []
        )
        if str(value).strip()
    ]
    selection_info = _resolve_selected_window_ids(
        state,
        selected_window_ids=selected_window_ids,
        selected_evidence_ids=selected_evidence_ids,
        selected_evidence_moment_ids=selected_evidence_moment_ids,
        candidate_window_ids=candidate_window_ids,
    )

    has_self_verification_payload = _has_self_verification_payload(arguments)

    if has_self_verification_payload:
        payload = dict(arguments)
        payload["selected_window_ids"] = list(selection_info["resolved_window_ids"])
        payload["selected_evidence_ids"] = list(selection_info["selected_evidence_ids"])
        payload["selected_evidence_moment_ids"] = list(selection_info["selected_evidence_moment_ids"])
        payload["candidate_window_ids"] = list(selection_info["valid_candidate_window_ids"])
        payload = validate_policy_self_verification_payload(payload)
        verification = parse_self_verification_payload(
            payload,
            fallback_claim=arguments.get("claim") or state.last_claim or {},
            verification_mode=str(arguments.get("verification_mode") or "reward_only"),
        )
        verification = _finalize_verification_payload(
            verification,
            selection_info=selection_info,
            verification_parse_mode="self_report",
        )
    else:
        raise ValueError(
            "verify_hypothesis must provide a verdict-bearing self-verification payload."
        )
    state.last_claim = dict(verification.get("claim") or {})
    state.active_evidence_window_ids = list(
        verification.get("verified_window_ids") or verification.get("best_effort_window_ids") or []
    )
    state.verification_records.append(verification)
    state.verifier_cache.append(verification)
    content = [{"type": "text", "text": json.dumps(verification, ensure_ascii=False)}]
    return content, state, verification


def finalize_case(arguments: Dict[str, Any], multimodal_cache: Dict, state: SaverEnvironmentState):
    decision_arguments, semantic_answer = split_finalize_case_payload(arguments)
    decision_arguments = validate_canonical_category_payload(
        decision_arguments,
        payload_name="finalize_case",
        require_category_for_anomaly=True,
    )
    resolved_moment_ids, resolved_stage_selected_moment_ids = _resolve_finalized_moment_ids_from_state(
        decision_arguments=decision_arguments,
        multimodal_cache=multimodal_cache,
        state=state,
    )
    if resolved_moment_ids:
        decision_arguments["evidence_moment_ids"] = list(resolved_moment_ids)
    elif "evidence_moment_ids" in decision_arguments:
        decision_arguments["evidence_moment_ids"] = []
    if resolved_stage_selected_moment_ids:
        decision_arguments["stage_selected_moment_ids"] = {
            stage: list(moment_ids)
            for stage, moment_ids in resolved_stage_selected_moment_ids.items()
        }
        resolved_covered_stages = [stage for stage, moment_ids in resolved_stage_selected_moment_ids.items() if moment_ids]
        decision_arguments["covered_stages"] = resolved_covered_stages
        current_missing = _dedupe_string_list(list(decision_arguments.get("missing_required_stages") or []))
        if current_missing:
            decision_arguments["missing_required_stages"] = [
                stage for stage in current_missing if stage not in resolved_covered_stages
            ]
    normalized_arguments = dict(arguments or {})
    normalized_arguments.update(decision_arguments)
    if semantic_answer is not None:
        normalized_arguments["summary"] = semantic_answer.get("summary")
        normalized_arguments["rationale"] = semantic_answer.get("rationale")
        normalized_arguments["event_chain_summary"] = dict(semantic_answer.get("event_chain_summary") or {})
        normalized_arguments["qa_focus_answers"] = dict(semantic_answer.get("qa_focus_answers") or {})
    schema = augment_finalize_case_schema(multimodal_cache.get("tool_io", {}).get("finalize_case_schema"))
    allowed_decision_fields = set(str(field_name) for field_name in dict(schema.get("properties") or {}).keys())
    if allowed_decision_fields:
        decision_arguments = {
            str(key): value
            for key, value in dict(decision_arguments or {}).items()
            if str(key) in allowed_decision_fields
        }
        normalized_arguments = {
            **{
                str(key): value
                for key, value in dict(normalized_arguments or {}).items()
                if str(key) in allowed_decision_fields
            },
            **{
                key: normalized_arguments[key]
                for key in OPTIONAL_FINALIZE_SEMANTIC_FIELDS
                if key in normalized_arguments
            },
        }
    validate_required_fields(normalized_arguments, schema)
    state.finalized_case = dict(decision_arguments)
    state.finalized_semantic_answer = dict(semantic_answer) if isinstance(semantic_answer, dict) else None
    state.last_claim = dict(decision_arguments)
    response_payload: Dict[str, Any] = {"status": "finalized", "finalized_case": state.finalized_case}
    if isinstance(state.finalized_semantic_answer, dict):
        response_payload["semantic_answer"] = dict(state.finalized_semantic_answer)
    content = [
        {
            "type": "text",
            "text": json.dumps(response_payload, ensure_ascii=False),
        }
    ]
    return content, state, state.finalized_case
