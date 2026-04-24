from __future__ import annotations

import copy
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch

from saver_v3.core.environment import parse_actions_and_contents
from saver_v3.model.model_loading import build_hf_model_init_kwargs, ensure_flash_attention_supported_dtype
from saver_v3.model.qwen_policy import (
    _build_generation_kwargs,
    _configure_qwen_processor,
    _to_pil_image,
    load_auto_processor_with_compat,
)
from saver_v3.core.self_verification import (
    PRIMARY_STATUS_TO_DECISION,
    SELF_VERIFICATION_DECISIONS,
    parse_self_verification_payload,
)


DEFAULT_TEACHER_JUDGE_MODEL_PATH = os.environ.get(
    "SAVER_QWEN_TEACHER_JUDGE_MODEL_PATH",
    "/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/Qwen3-VL-32B-Instruct",
)
DEFAULT_TEACHER_JUDGE_INPUT_MODE = os.environ.get("SAVER_TEACHER_JUDGE_INPUT_MODE", "auto")
TEACHER_JUDGE_INPUT_MODES = {"text_only", "multimodal_visual", "auto"}
TEACHER_JUDGE_PACKAGE_SCHEMA_VERSION = 3
TEACHER_JUDGE_PACKAGE_STORAGE_FORMAT = "materialized_torch_messages_v1"
TEACHER_JUDGE_ENTRY_SCHEMA_VERSION = 1
TEACHER_JUDGE_LABEL_KEYS = (
    "teacher_judge_scores",
    "teacher_judge_decision",
    "teacher_judge_rationale",
)
TEACHER_JUDGE_SCORE_KEYS = (
    "sufficiency",
    "necessity",
    "finalize_readiness",
)


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, score))


def _sanitize_teacher_next_tool_text(text: str) -> str:
    sanitized = str(text or "")
    replacements = (
        (r"\bcontinue_search\b", "seek_evidence"),
        (r"\brefine_evidence\b", "seek_evidence"),
        (r"\brevise_claim\b", "seek_evidence"),
        (r"\bcontinue searching\b", "call seek_evidence"),
        (r"\bcontinue the search\b", "call seek_evidence"),
        (r"\bcontinuing the search\b", "calling seek_evidence"),
        (r"\brefine the selected evidence subset\b", "call seek_evidence on a better evidence subset"),
        (r"\brevise the claim\b", "call seek_evidence to gather aligned evidence"),
        (r"\bshould finalize\b", "should call finalize_case"),
        (r"\bbefore finalize\b", "before calling finalize_case"),
    )
    for pattern, replacement in replacements:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)
    return sanitized


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = str(text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _output_schema_example() -> str:
    payload = {
        "teacher_judge_scores": {
            "sufficiency": 0.0,
            "necessity": 0.0,
            "finalize_readiness": 0.0,
        },
        "teacher_judge_decision": "insufficient",
        "teacher_judge_rationale": "",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _json_text_length(payload: Any) -> int:
    try:
        return len(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    except Exception:
        return len(str(payload or ""))


def _extract_verify_payload(example: Dict[str, Any]) -> Dict[str, Any]:
    target_response = str(example.get("target_response") or "")
    actions, contents = parse_actions_and_contents([target_response])
    if not actions or actions[0] != "tool_call":
        return {}
    parsed_content = contents[0]
    if not isinstance(parsed_content, dict):
        return {}
    function_payload = parsed_content.get("function") or {}
    if str(function_payload.get("name") or "") != "verify_hypothesis":
        return {}
    arguments = function_payload.get("arguments") or {}
    return copy.deepcopy(arguments) if isinstance(arguments, dict) else {}


def _normalize_teacher_decision(payload: Dict[str, Any], scores: Dict[str, float]) -> str:
    decision = str(payload.get("teacher_judge_decision") or payload.get("verification_decision") or "").strip().lower()
    if decision in SELF_VERIFICATION_DECISIONS:
        return decision
    primary_status = str(payload.get("primary_status") or "").strip().lower()
    if primary_status == "complete":
        return "sufficient"
    if primary_status == "misaligned":
        return "misaligned"
    if primary_status == "redundant":
        return "redundant"
    next_tool = str(payload.get("next_tool") or "").strip().lower()
    if next_tool == "finalize_case":
        return "sufficient"
    if scores["sufficiency"] >= 0.75 and scores["necessity"] >= 0.35:
        return "sufficient"
    return "insufficient"


def normalize_teacher_judge_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(payload or {})
    score_payload = payload.get("teacher_judge_scores") or {}
    if not isinstance(score_payload, dict):
        score_payload = {}
    scores = {
        "sufficiency": _clamp_score(score_payload.get("sufficiency", payload.get("sufficiency_score", payload.get("sufficiency")))),
        "necessity": _clamp_score(score_payload.get("necessity", payload.get("necessity_score", payload.get("necessity")))),
        "finalize_readiness": _clamp_score(
            score_payload.get(
                "finalize_readiness",
                payload.get(
                    "finalize_readiness_score",
                    payload.get("finalize_readiness"),
                ),
            )
        ),
    }
    decision = _normalize_teacher_decision(payload, scores)
    rationale = str(
        payload.get("teacher_judge_rationale") or payload.get("rationale") or payload.get("explanation") or ""
    ).strip()
    rationale = _sanitize_teacher_next_tool_text(rationale)
    return {
        "teacher_judge_scores": {key: round(value, 6) for key, value in scores.items()},
        "teacher_judge_decision": decision,
        "teacher_judge_rationale": rationale,
    }


def parse_teacher_judge_response(text: str) -> Dict[str, Any]:
    return normalize_teacher_judge_result(_extract_json_object(text))


def is_teacher_judge_candidate(example: Dict[str, Any]) -> bool:
    target_action = str(example.get("target_action") or "").strip().lower()
    if target_action and target_action != "tool_call":
        return False
    tool_name = str(example.get("tool_name") or "").strip()
    if tool_name:
        return tool_name == "verify_hypothesis"
    return bool(_extract_verify_payload(example))


def has_teacher_judge_labels(example: Dict[str, Any]) -> bool:
    return any(key in example for key in TEACHER_JUDGE_LABEL_KEYS)


def _extract_verify_decision_from_example(example: Dict[str, Any]) -> str:
    verify_payload = _extract_verify_payload(example)
    decision = str(verify_payload.get("verification_decision") or "").strip().lower()
    if decision in SELF_VERIFICATION_DECISIONS:
        return decision
    next_tool = str(verify_payload.get("next_tool") or "").strip().lower()
    if next_tool == "finalize_case":
        return "sufficient"
    if next_tool == "seek_evidence":
        return "insufficient"
    return ""


def compute_teacher_judge_alignment(example: Dict[str, Any]) -> Optional[float]:
    if not has_teacher_judge_labels(example):
        return None
    verify_decision = _extract_verify_decision_from_example(example)
    teacher_decision = str(example.get("teacher_judge_decision") or "").strip().lower()
    if not verify_decision or not teacher_decision:
        return None
    return 1.0 if verify_decision == teacher_decision else 0.0


def compute_teacher_judge_weight_multiplier(example: Dict[str, Any]) -> float:
    if not is_teacher_judge_candidate(example) or not has_teacher_judge_labels(example):
        return 1.0

    score_payload = dict(example.get("teacher_judge_scores") or {})
    score_values = [
        _clamp_score(score_payload.get(key))
        for key in ("sufficiency", "necessity", "finalize_readiness")
        if key in score_payload
    ]
    confidence = sum(score_values) / float(len(score_values)) if score_values else 0.5
    confidence_factor = 0.85 + 0.5 * confidence

    teacher_decision = str(example.get("teacher_judge_decision") or "").strip().lower()
    decision_factor = 1.15 if teacher_decision in {"insufficient", "misaligned", "redundant"} else 1.0

    alignment = compute_teacher_judge_alignment(example)
    if alignment is None:
        alignment_factor = 1.0
    elif alignment >= 0.5:
        alignment_factor = 1.15
    else:
        alignment_factor = 0.7

    return round(max(0.35, min(2.0, confidence_factor * decision_factor * alignment_factor)), 6)


def _extract_policy_verification_signature(source: Dict[str, Any]) -> tuple[str, Dict[str, float]]:
    payload: Dict[str, Any] = {}
    if is_teacher_judge_candidate(source):
        payload = _extract_verify_payload(source)
    if payload:
        normalized = parse_self_verification_payload(payload)
        return (
            str(normalized.get("verification_decision") or "").strip().lower(),
            {
                key: _clamp_score((normalized.get("self_verification_scores") or normalized.get("derived_scores") or {}).get(key))
                for key in TEACHER_JUDGE_SCORE_KEYS
            },
        )

    decision = str(source.get("self_verification_decision") or source.get("verification_decision") or "").strip().lower()
    if decision not in SELF_VERIFICATION_DECISIONS:
        primary_status = str(source.get("verifier_primary_status") or source.get("primary_status") or "").strip().lower()
        decision = PRIMARY_STATUS_TO_DECISION.get(primary_status, "")
    score_payload = dict(source.get("self_verification_scores") or source.get("verifier_derived_scores") or {})
    return decision, {key: _clamp_score(score_payload.get(key)) for key in TEACHER_JUDGE_SCORE_KEYS}


def compute_teacher_judge_signal(source: Dict[str, Any]) -> Dict[str, Any]:
    teacher_payload = dict(source.get("teacher_judge_scores") or {})
    teacher_scores = {
        key: _clamp_score(teacher_payload.get(key))
        for key in TEACHER_JUDGE_SCORE_KEYS
        if key in teacher_payload
    }
    teacher_decision = str(source.get("teacher_judge_decision") or "").strip().lower()
    teacher_present = bool(teacher_scores or teacher_decision)
    if not teacher_present:
        return {
            "teacher_judge_present": False,
            "teacher_judge_policy_decision": "",
            "teacher_judge_alignment": None,
            "teacher_judge_score_agreement": 0.0,
            "teacher_judge_confidence": 0.0,
            "teacher_judge_reward": 0.0,
        }

    policy_decision, policy_scores = _extract_policy_verification_signature(source)
    alignment: Optional[float]
    if teacher_decision and policy_decision:
        alignment = 1.0 if teacher_decision == policy_decision else 0.0
    else:
        alignment = None

    score_terms: List[float] = []
    for key, teacher_score in teacher_scores.items():
        if key not in policy_scores:
            continue
        score_terms.append(max(0.0, 1.0 - abs(float(policy_scores[key]) - float(teacher_score))))
    score_agreement = sum(score_terms) / float(len(score_terms)) if score_terms else 0.5
    confidence = sum(float(value) for value in teacher_scores.values()) / float(len(teacher_scores)) if teacher_scores else 0.5
    decision_term = 0.0 if alignment is None else (2.0 * float(alignment) - 1.0)
    score_term = 2.0 * float(score_agreement) - 1.0
    reward = confidence * (0.7 * decision_term + 0.3 * score_term)
    reward = max(-1.0, min(1.0, float(reward)))
    return {
        "teacher_judge_present": True,
        "teacher_judge_policy_decision": policy_decision,
        "teacher_judge_decision": teacher_decision,
        "teacher_judge_alignment": alignment,
        "teacher_judge_score_agreement": round(float(score_agreement), 6),
        "teacher_judge_confidence": round(float(confidence), 6),
        "teacher_judge_reward": round(float(reward), 6),
    }


def _has_existing_teacher_reweight(example: Dict[str, Any]) -> bool:
    return any(
        key in example
        for key in (
            "teacher_judge_effective_sample_weight",
            "teacher_judge_weight_multiplier",
            "teacher_judge_base_sample_weight",
        )
    )


def apply_teacher_judge_reweighting(example: Dict[str, Any]) -> Dict[str, Any]:
    updated = copy.deepcopy(example)
    if not is_teacher_judge_candidate(updated) or not has_teacher_judge_labels(updated):
        return updated
    alignment = compute_teacher_judge_alignment(updated)
    updated["teacher_judge_alignment"] = alignment
    current_sample_weight = float(updated.get("sample_weight", 1.0) or 1.0)
    if _has_existing_teacher_reweight(updated):
        stored_multiplier = float(updated.get("teacher_judge_weight_multiplier", 0.0) or 0.0)
        inferred_base_sample_weight = current_sample_weight
        if abs(stored_multiplier) > 1e-8:
            inferred_base_sample_weight = current_sample_weight / stored_multiplier
        updated.setdefault("teacher_judge_base_sample_weight", round(float(inferred_base_sample_weight), 6))
        updated.setdefault("teacher_judge_effective_sample_weight", float(current_sample_weight))
        return updated

    base_sample_weight = current_sample_weight
    multiplier = compute_teacher_judge_weight_multiplier(updated)
    updated["teacher_judge_base_sample_weight"] = round(float(base_sample_weight), 6)
    updated["teacher_judge_weight_multiplier"] = multiplier
    updated["sample_weight"] = round(base_sample_weight * multiplier, 6)
    updated["teacher_judge_effective_sample_weight"] = float(updated["sample_weight"])
    return updated


def reweight_teacher_judge_examples(examples: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    reweighted_examples: List[Dict[str, Any]] = []
    num_reweighted = 0
    for example in examples:
        updated = apply_teacher_judge_reweighting(example)
        if updated != example and is_teacher_judge_candidate(updated) and has_teacher_judge_labels(updated):
            num_reweighted += 1
        reweighted_examples.append(updated)
    return reweighted_examples, {
        "num_teacher_judge_reweighted": num_reweighted,
    }


def _iter_observation_items(example: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for message in list(example.get("messages") or []):
        role = str(message.get("role") or "").strip().lower()
        if role not in {"user", "tool"}:
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict):
                yield item


def _resolve_observation_image(
    item: Dict[str, Any],
    *,
    image_resolver: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Any:
    if "image" in item:
        return item.get("image")
    image_ref = item.get("image_ref")
    if image_ref is not None and callable(image_resolver):
        try:
            return image_resolver(image_ref)
        except Exception:
            return None
    return None


def _extract_image_timestamp_sec(item: Dict[str, Any]) -> Optional[float]:
    for source in (item, item.get("image_ref") or {}):
        if not isinstance(source, dict):
            continue
        value = source.get("timestamp_sec")
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _build_observation_image_entry(
    item: Dict[str, Any],
    *,
    resolve_image: bool = False,
    image_resolver: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Dict[str, Any]:
    image_entry: Dict[str, Any] = {}
    resolved_image = None
    if bool(resolve_image):
        resolved_image = _resolve_observation_image(item, image_resolver=image_resolver)
    if resolved_image is not None:
        image_entry["image"] = resolved_image
    elif "image" in item:
        image_entry["image"] = item.get("image")
    if item.get("image_ref") is not None:
        image_entry["image_ref"] = item.get("image_ref")
    timestamp_sec = _extract_image_timestamp_sec(item)
    if timestamp_sec is not None:
        image_entry["timestamp_sec"] = float(timestamp_sec)
    return image_entry


def _collect_observation_windows(
    example: Dict[str, Any],
    *,
    resolve_images: bool = False,
    image_resolver: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> tuple[List[str], List[Dict[str, Any]]]:
    observation_lines: List[str] = []
    window_records: List[Dict[str, Any]] = []
    next_window_index = 1
    next_evidence_index = 1

    for message in list(example.get("messages") or []):
        role = str(message.get("role") or "").strip().lower()
        if role not in {"user", "tool"}:
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue

        message_texts: List[str] = []
        image_entries: List[Dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip().lower()
            if item_type == "text":
                text = str(item.get("text") or "").strip()
                if text:
                    observation_lines.append(text)
                    message_texts.append(text)
                continue
            if item_type != "image":
                continue
            image_entry = _build_observation_image_entry(
                item,
                resolve_image=bool(resolve_images),
                image_resolver=image_resolver,
            )
            if image_entry:
                image_entries.append(image_entry)

        if role != "tool":
            continue
        tool_name = str(message.get("name") or "").strip()
        if tool_name not in {"scan_timeline", "seek_evidence"}:
            continue
        arguments = dict(message.get("arguments") or {})
        timestamps_sec = [
            float(entry["timestamp_sec"])
            for entry in image_entries
            if entry.get("timestamp_sec") is not None
        ]
        window_records.append(
            {
                "window_id": f"w{next_window_index:04d}",
                "evidence_id": f"e{next_evidence_index:04d}",
                "tool_name": tool_name,
                "moment_id": str(arguments.get("moment_id") or "").strip(),
                "role": str(arguments.get("role") or "").strip(),
                "start_sec": arguments.get("start_sec"),
                "end_sec": arguments.get("end_sec"),
                "query": str(arguments.get("query") or "").strip(),
                "images": image_entries,
                "timestamps_sec": timestamps_sec,
                "summary_text": "\n".join(message_texts).strip(),
            }
        )
        next_window_index += 1
        next_evidence_index += 1

    return observation_lines, window_records


def _resolve_selected_window_ids_from_records(
    window_records: List[Dict[str, Any]],
    *,
    selected_window_ids: List[str],
    selected_evidence_ids: List[str],
    selected_evidence_moment_ids: List[str],
) -> List[str]:
    known_window_ids = {
        str(record.get("window_id") or "").strip()
        for record in window_records
        if str(record.get("window_id") or "").strip()
    }
    resolved = [str(value) for value in selected_window_ids or [] if str(value).strip() in known_window_ids]
    seen = {str(value) for value in resolved if str(value).strip()}
    selected_evidence_id_set = {str(value) for value in selected_evidence_ids if str(value).strip()}
    selected_moment_id_set = {str(value) for value in selected_evidence_moment_ids if str(value).strip()}
    for record in window_records:
        window_id = str(record.get("window_id") or "").strip()
        evidence_id = str(record.get("evidence_id") or "").strip()
        moment_id = str(record.get("moment_id") or "").strip()
        if not window_id or window_id in seen:
            continue
        if evidence_id and evidence_id in selected_evidence_id_set:
            resolved.append(window_id)
            seen.add(window_id)
            continue
        if moment_id and moment_id in selected_moment_id_set:
            resolved.append(window_id)
            seen.add(window_id)
    return resolved


def _limit_image_entries(image_entries: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if limit <= 0 or len(image_entries) <= limit:
        return [copy.deepcopy(entry) for entry in image_entries]
    if limit == 1:
        return [copy.deepcopy(image_entries[0])]
    selected_indices = {
        int(round(index * (len(image_entries) - 1) / float(limit - 1)))
        for index in range(limit)
    }
    return [copy.deepcopy(image_entries[index]) for index in sorted(selected_indices)]


def _summarize_view_windows(window_records: List[Dict[str, Any]], *, empty_text: str) -> str:
    if not window_records:
        return empty_text
    lines: List[str] = []
    for record in window_records:
        window_id = str(record.get("window_id") or "")
        tool_name = str(record.get("tool_name") or "")
        role = str(record.get("role") or "")
        moment_id = str(record.get("moment_id") or "")
        query = str(record.get("query") or "")
        timestamps_sec = [round(float(value), 3) for value in list(record.get("timestamps_sec") or [])]
        line = f"{window_id} {tool_name}"
        if role:
            line += f" role={role}"
        if moment_id:
            line += f" moment_id={moment_id}"
        if query:
            line += f" query={query}"
        if timestamps_sec:
            line += f" timestamps={json.dumps(timestamps_sec, ensure_ascii=False)}"
        lines.append(line)
    return "\n".join(lines)


def _build_view_payload(
    window_records: List[Dict[str, Any]],
    *,
    topk_frames_per_view: int,
    empty_text: str,
) -> Dict[str, Any]:
    flattened_images: List[Dict[str, Any]] = []
    timestamps_sec: List[float] = []
    for record in window_records:
        flattened_images.extend(list(record.get("images") or []))
        timestamps_sec.extend(float(value) for value in list(record.get("timestamps_sec") or []))
    return {
        "summary_text": _summarize_view_windows(window_records, empty_text=empty_text),
        "timestamps_sec": [round(float(value), 6) for value in timestamps_sec],
        "images": _limit_image_entries(flattened_images, int(topk_frames_per_view)),
    }


def _normalize_input_mode(input_mode: str | None) -> str:
    normalized = str(input_mode or DEFAULT_TEACHER_JUDGE_INPUT_MODE).strip().lower()
    if normalized not in TEACHER_JUDGE_INPUT_MODES:
        return DEFAULT_TEACHER_JUDGE_INPUT_MODE
    return normalized


def _count_package_images(package: Dict[str, Any]) -> int:
    image_count = 0
    for view_payload in (package.get("views") or {}).values():
        if isinstance(view_payload, dict):
            image_count += len(view_payload.get("images") or [])
    return int(image_count)


def resolve_teacher_judge_input_mode_metadata(
    package: Dict[str, Any],
    *,
    requested_mode: str = DEFAULT_TEACHER_JUDGE_INPUT_MODE,
    max_images: int = 8,
) -> Dict[str, Any]:
    normalized_mode = _normalize_input_mode(requested_mode)
    if not isinstance(package, dict):
        return {
            "actual_input_mode": "text_only",
            "auto_mode_reason": "auto_text_only_default",
            "auto_mode_verification_decision": "",
            "auto_mode_selected_window_count": 0,
            "auto_mode_image_count": 0,
        }

    image_count = _count_package_images(package)
    policy_self_verification = dict(package.get("policy_self_verification") or {})
    verification_decision = str(policy_self_verification.get("verification_decision") or "").strip().lower()
    selected_window_count = len([value for value in list(package.get("selected_window_ids") or []) if str(value).strip()])
    metadata = {
        "actual_input_mode": "text_only",
        "auto_mode_reason": "auto_text_only_default",
        "auto_mode_verification_decision": verification_decision,
        "auto_mode_selected_window_count": int(selected_window_count),
        "auto_mode_image_count": int(image_count),
    }
    if normalized_mode == "text_only":
        metadata["auto_mode_reason"] = "explicit_text_only"
        return metadata
    if normalized_mode == "multimodal_visual":
        metadata["actual_input_mode"] = "multimodal_visual"
        metadata["auto_mode_reason"] = "explicit_multimodal_visual"
        return metadata
    if verification_decision not in {"misaligned", "redundant"}:
        metadata["auto_mode_reason"] = "auto_text_only_decision_not_visual"
        return metadata
    if selected_window_count <= 1:
        metadata["auto_mode_reason"] = "auto_text_only_single_window"
        return metadata
    if image_count <= 0:
        metadata["auto_mode_reason"] = "auto_text_only_no_images"
        return metadata
    if image_count > max(0, int(max_images)):
        metadata["auto_mode_reason"] = "auto_text_only_too_many_images"
        return metadata
    metadata["actual_input_mode"] = "multimodal_visual"
    metadata["auto_mode_reason"] = f"auto_multimodal_{verification_decision}_multi_window_light_images"
    return metadata


def _infer_package_video_group_key(example: Dict[str, Any], package: Dict[str, Any]) -> str:
    for view_payload in (package.get("views") or {}).values():
        if not isinstance(view_payload, dict):
            continue
        for image_entry in list(view_payload.get("images") or []):
            image_ref = image_entry.get("image_ref") or {}
            video_path = str(image_ref.get("video_path") or "").strip()
            if video_path:
                return f"video_path::{video_path}"
    video_id = str(example.get("video_id") or "").strip()
    if video_id:
        return f"video_id::{video_id}"
    return f"source::{str(example.get('source') or '')}::row"


def estimate_teacher_judge_package_cost(package: Dict[str, Any], *, actual_input_mode: str) -> int:
    image_count = 0
    for view_payload in (package.get("views") or {}).values():
        if isinstance(view_payload, dict):
            image_count += len(view_payload.get("images") or [])
    summary_len = _json_text_length(package.get("claim")) + _json_text_length(package.get("policy_self_verification"))
    for view_payload in (package.get("views") or {}).values():
        if isinstance(view_payload, dict):
            summary_len += len(str(view_payload.get("summary_text") or ""))
    base_cost = max(1, summary_len // 512)
    if str(actual_input_mode or "").strip().lower() == "multimodal_visual":
        base_cost += 8 + 4 * int(image_count)
    else:
        base_cost += 1
    return int(base_cost)


def build_teacher_judge_package(
    example: Dict[str, Any],
    *,
    topk_frames_per_view: int = 4,
    image_resolver: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Dict[str, Any]:
    verify_payload = _extract_verify_payload(example)
    parsed_payload = parse_self_verification_payload(verify_payload) if verify_payload else {}
    normalized_preview = normalize_teacher_judge_result(verify_payload)

    topk_frames_per_view = max(1, int(topk_frames_per_view))
    observation_lines, window_records = _collect_observation_windows(
        example,
        resolve_images=False,
        image_resolver=image_resolver,
    )
    selected_window_ids = _resolve_selected_window_ids_from_records(
        window_records,
        selected_window_ids=list(parsed_payload.get("selected_window_ids") or []),
        selected_evidence_ids=list(parsed_payload.get("selected_evidence_ids") or []),
        selected_evidence_moment_ids=list(parsed_payload.get("selected_evidence_moment_ids") or []),
    )
    selected_evidence_ids = list(parsed_payload.get("selected_evidence_ids") or [])
    selected_evidence_moment_ids = list(parsed_payload.get("selected_evidence_moment_ids") or [])
    decision = str(parsed_payload.get("verification_decision") or normalized_preview.get("teacher_judge_decision") or "")
    next_tool = str(parsed_payload.get("next_tool") or "")
    selected_window_id_set = {str(value) for value in selected_window_ids if str(value).strip()}
    keep_window_records = [
        record for record in window_records if str(record.get("window_id") or "").strip() in selected_window_id_set
    ]
    drop_window_records = [
        record for record in window_records if str(record.get("window_id") or "").strip() not in selected_window_id_set
    ]

    hard_reasons: List[str] = []
    if decision in {"insufficient", "misaligned", "redundant"}:
        hard_reasons.append(f"verification_decision={decision}")
    if next_tool and next_tool != "finalize_case":
        hard_reasons.append(f"next_tool={next_tool}")
    if len(selected_window_ids) > 1:
        hard_reasons.append("multiple_selected_windows")

    rounded_timestamps = [
        round(float(entry.get("timestamp_sec")), 6)
        for record in window_records
        for entry in list(record.get("images") or [])
        if entry.get("timestamp_sec") is not None
    ]
    return {
        "claim": copy.deepcopy(parsed_payload.get("claim") or verify_payload.get("claim") or {}),
        "verification_mode": str(parsed_payload.get("verification_mode") or verify_payload.get("verification_mode") or ""),
        "policy_self_verification": copy.deepcopy(parsed_payload) if parsed_payload else copy.deepcopy(normalized_preview),
        "selected_window_ids": selected_window_ids,
        "selected_evidence_ids": selected_evidence_ids,
        "selected_evidence_moment_ids": selected_evidence_moment_ids,
        "timestamps_sec": rounded_timestamps,
        "views": {
            "full": _build_view_payload(
                window_records,
                topk_frames_per_view=topk_frames_per_view,
                empty_text="\n".join(list(observation_lines) or ["No textual observation summary was available."]),
            ),
            "keep": _build_view_payload(
                keep_window_records,
                topk_frames_per_view=topk_frames_per_view,
                empty_text=(
                    "No selected evidence subset was available.\n"
                    f"Selected window ids: {json.dumps(selected_window_ids, ensure_ascii=False)}\n"
                    f"Selected evidence ids: {json.dumps(selected_evidence_ids, ensure_ascii=False)}\n"
                    f"Selected evidence moment ids: {json.dumps(selected_evidence_moment_ids, ensure_ascii=False)}"
                ),
            ),
            "drop": _build_view_payload(
                drop_window_records,
                topk_frames_per_view=topk_frames_per_view,
                empty_text="Counterfactual drop view has no remaining evidence outside the selected subset.",
            ),
        },
        "hard_case": bool(hard_reasons),
        "hard_reasons": hard_reasons,
    }


def build_teacher_judge_package_record(
    example: Dict[str, Any],
    *,
    source_row_index: int,
    requested_mode: str = DEFAULT_TEACHER_JUDGE_INPUT_MODE,
    max_images: int = 8,
    topk_frames_per_view: int = 4,
    image_resolver: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> Dict[str, Any]:
    package = build_teacher_judge_package(
        example,
        topk_frames_per_view=topk_frames_per_view,
        image_resolver=image_resolver,
    )
    input_mode_metadata = resolve_teacher_judge_input_mode_metadata(
        package,
        requested_mode=requested_mode,
        max_images=max_images,
    )
    actual_input_mode = str(input_mode_metadata["actual_input_mode"])
    image_ref_count = int(input_mode_metadata["auto_mode_image_count"])
    return {
        "package_schema_version": int(TEACHER_JUDGE_PACKAGE_SCHEMA_VERSION),
        "source_row_index": int(source_row_index),
        "video_id": str(example.get("video_id") or ""),
        "tool_name": str(example.get("tool_name") or ""),
        "requested_input_mode": _normalize_input_mode(requested_mode),
        "actual_input_mode": str(actual_input_mode),
        "max_images": int(max_images),
        "topk_frames_per_view": int(topk_frames_per_view),
        "image_ref_count": int(image_ref_count),
        "video_group_key": _infer_package_video_group_key(example, package),
        "estimated_cost": int(estimate_teacher_judge_package_cost(package, actual_input_mode=actual_input_mode)),
        "package_payload": package,
        **input_mode_metadata,
    }


def select_teacher_judge_input_mode(
    package: Dict[str, Any],
    *,
    requested_mode: str = DEFAULT_TEACHER_JUDGE_INPUT_MODE,
    max_images: int = 8,
) -> str:
    return str(
        resolve_teacher_judge_input_mode_metadata(
            package,
            requested_mode=requested_mode,
            max_images=max_images,
        )["actual_input_mode"]
    )


def build_teacher_judge_messages(
    example: Dict[str, Any],
    *,
    input_mode: str = DEFAULT_TEACHER_JUDGE_INPUT_MODE,
    max_images: int = 8,
    topk_frames_per_view: int = 4,
    image_resolver: Optional[Callable[[Dict[str, Any]], Any]] = None,
) -> List[Dict[str, Any]]:
    package = (
        copy.deepcopy(example)
        if isinstance(example, dict) and "views" in example and "policy_self_verification" in example
        else build_teacher_judge_package(
            example,
            topk_frames_per_view=topk_frames_per_view,
            image_resolver=image_resolver,
        )
    )
    actual_input_mode = select_teacher_judge_input_mode(
        package,
        requested_mode=input_mode,
        max_images=max_images,
    )
    system_prompt = (
        "You are a teacher judge for SAVER self-verification supervision. "
        "Judge whether the policy's selected evidence subset is sufficient, necessary enough, "
        "finalize-ready, and counterfactually faithful for the stated anomaly claim. "
        "Return valid JSON only. Begin with { and end with }. "
        "Use double quotes for all keys and strings. Do not use markdown fences."
    )
    views = package.get("views") or {}
    user_content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Teacher judge task: assess the policy's verify_hypothesis output.\n"
                "Use the policy self-verification output together with the full / keep / drop views "
                "to judge whether the policy should call seek_evidence or finalize_case next.\n"
                "Return JSON with this exact schema:\n"
                f"{_output_schema_example()}\n"
                f"Active judge input mode: {actual_input_mode}\n"
                f"Claim:\n{json.dumps(package.get('claim') or {}, ensure_ascii=False, indent=2)}\n"
                f"Policy self-verification:\n{json.dumps(package.get('policy_self_verification') or {}, ensure_ascii=False, indent=2)}\n"
                f"Counterfactual view summaries:\n"
                f"{json.dumps({name: {'summary_text': (payload or {}).get('summary_text', ''), 'timestamps_sec': (payload or {}).get('timestamps_sec', [])} for name, payload in views.items()}, ensure_ascii=False, indent=2)}\n"
                f"Hard-case reasons: {json.dumps(package.get('hard_reasons') or [], ensure_ascii=False)}"
            ),
        }
    ]

    image_count = 0
    for view_name in ("full", "keep", "drop"):
        view_payload = views.get(view_name) or {}
        summary_text = str(view_payload.get("summary_text") or "").strip()
        if summary_text:
            user_content.append({"type": "text", "text": f"{view_name} summary:\n{summary_text}"})
        if actual_input_mode != "multimodal_visual":
            continue
        for image_entry in view_payload.get("images") or []:
            if image_count >= int(max_images):
                break
            resolved_image = image_entry.get("image")
            if resolved_image is None and image_entry.get("image_ref") is not None and callable(image_resolver):
                try:
                    resolved_image = image_resolver(image_entry["image_ref"])
                except Exception:
                    resolved_image = None
            if resolved_image is None:
                continue
            timestamp_sec = image_entry.get("timestamp_sec")
            if timestamp_sec is not None:
                user_content.append(
                    {
                        "type": "text",
                        "text": f"{view_name} frame timestamp: {float(timestamp_sec):.3f}s",
                    }
                )
            user_content.append({"type": "image", "image": _to_pil_image(resolved_image)})
            image_count += 1
        if image_count >= int(max_images):
            break

    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]


def validate_materialized_teacher_judge_messages(messages: Any) -> List[Dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("Materialized teacher-judge entry is missing non-empty messages.")
    validated_messages = copy.deepcopy(messages)
    for message in validated_messages:
        if not isinstance(message, dict):
            raise ValueError("Materialized teacher-judge message must be a dict.")
        content = message.get("content")
        if not isinstance(content, list):
            raise ValueError("Materialized teacher-judge message content must be a list.")
        for item in content:
            if not isinstance(item, dict):
                raise ValueError("Materialized teacher-judge message item must be a dict.")
            if item.get("type") != "image":
                continue
            if "image_ref" in item:
                raise ValueError("Materialized teacher-judge image items must not keep image_ref.")
            if "image" not in item:
                raise ValueError("Materialized teacher-judge image items must include image.")
    return validated_messages


def attach_teacher_judge_labels(example: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    updated = copy.deepcopy(example)
    updated.update(normalize_teacher_judge_result(result))
    return updated


def annotate_teacher_judge_examples(
    examples: List[Dict[str, Any]],
    *,
    judge: Any,
    input_mode: Optional[str] = None,
    batch_size: int = 1,
    overwrite_existing: bool = False,
    progress_every: int = 0,
    log_fn: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    annotated_examples: List[Optional[Dict[str, Any]]] = [None] * len(examples)
    total_examples = len(examples)
    candidate_examples = 0
    annotated_count = 0
    skipped_existing = 0
    pending_indices: List[int] = []
    pending_examples: List[Dict[str, Any]] = []

    normalized_batch_size = max(1, int(batch_size))

    def _emit_progress(*, phase: str, completed: int, total: int) -> None:
        if progress_callback is None:
            return
        progress_callback(
            {
                "phase": str(phase),
                "completed": int(completed),
                "total": int(total),
                "candidate_examples": int(candidate_examples),
                "annotated_count": int(annotated_count),
                "skipped_existing": int(skipped_existing),
                "pending_examples": int(len(pending_examples)),
                "num_examples": int(total_examples),
            }
        )

    def _annotate_batch(batch_examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not batch_examples:
            return []
        if normalized_batch_size > 1 and hasattr(judge, "annotate_examples"):
            return list(judge.annotate_examples(batch_examples, input_mode=input_mode))
        return [judge.annotate_example(example, input_mode=input_mode) for example in batch_examples]

    for idx, example in enumerate(examples, start=1):
        if not is_teacher_judge_candidate(example):
            annotated_examples[idx - 1] = copy.deepcopy(example)
        elif not overwrite_existing and has_teacher_judge_labels(example):
            annotated_examples[idx - 1] = copy.deepcopy(example)
            candidate_examples += 1
            skipped_existing += 1
        else:
            candidate_examples += 1
            pending_indices.append(idx - 1)
            pending_examples.append(example)

        if log_fn is not None:
            should_log = idx == 1 or idx == total_examples or (progress_every > 0 and idx % int(progress_every) == 0)
            if should_log:
                log_fn(
                    "teacher judge progress: "
                    f"examples={idx}/{total_examples} "
                    f"candidates={candidate_examples} "
                    f"annotated={annotated_count} "
                    f"skipped_existing={skipped_existing}"
                )
        _emit_progress(phase="scan", completed=idx, total=total_examples)

    if log_fn is not None:
        log_fn(
            "teacher judge scan complete: "
            f"examples={total_examples} "
            f"candidates={candidate_examples} "
            f"pending_annotation={len(pending_examples)} "
            f"skipped_existing={skipped_existing} "
            f"batch_size={normalized_batch_size}"
        )
    _emit_progress(phase="annotate", completed=0, total=len(pending_examples))

    for start in range(0, len(pending_examples), normalized_batch_size):
        batch_examples = pending_examples[start : start + normalized_batch_size]
        batch_indices = pending_indices[start : start + normalized_batch_size]
        batch_results = _annotate_batch(batch_examples)
        if len(batch_results) != len(batch_examples):
            raise ValueError(
                "Teacher judge batch annotation returned a different number of results than requested. "
                f"requested={len(batch_examples)} returned={len(batch_results)}"
            )
        for example_index, annotated_example in zip(batch_indices, batch_results):
            annotated_examples[example_index] = annotated_example
        annotated_count += len(batch_results)
        if log_fn is not None:
            batch_completed = min(start + len(batch_results), len(pending_examples))
            should_log_batch = (
                batch_completed == len(pending_examples)
                or batch_completed == len(batch_results)
                or (progress_every > 0 and batch_completed % int(progress_every) == 0)
            )
            if should_log_batch:
                log_fn(
                    "teacher judge annotate progress: "
                    f"annotated={batch_completed}/{len(pending_examples)} "
                    f"total_examples={total_examples} "
                    f"candidates={candidate_examples}"
                )
        _emit_progress(
            phase="annotate",
            completed=min(start + len(batch_results), len(pending_examples)),
            total=len(pending_examples),
        )

    finalized_examples = [example for example in annotated_examples if example is not None]
    if len(finalized_examples) != len(examples):
        raise ValueError(
            "Teacher judge annotation did not fill every example slot. "
            f"expected={len(examples)} actual={len(finalized_examples)}"
        )
    summary = {
        "num_examples": total_examples,
        "num_teacher_judge_candidates": candidate_examples,
        "num_teacher_judge_annotated": annotated_count,
        "num_teacher_judge_skipped_existing": skipped_existing,
    }
    return finalized_examples, summary


class QwenTeacherJudge:
    """Offline teacher judge used for SFT/RL supervision, not inference-time control."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        input_mode: str = DEFAULT_TEACHER_JUDGE_INPUT_MODE,
        max_new_tokens: int = 384,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        max_images: int = 8,
        topk_frames_per_view: int = 4,
        image_resolver: Optional[Callable[[Dict[str, Any]], Any]] = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        self.model = model
        self.processor = processor
        self.input_mode = str(input_mode or DEFAULT_TEACHER_JUDGE_INPUT_MODE)
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_images = int(max_images)
        self.topk_frames_per_view = int(topk_frames_per_view)
        self.image_resolver = image_resolver
        self.log_fn = log_fn
        self._normalize_generation_config()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path = DEFAULT_TEACHER_JUDGE_MODEL_PATH,
        *,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        attn_implementation: Optional[str] = None,
        input_mode: str = DEFAULT_TEACHER_JUDGE_INPUT_MODE,
        max_new_tokens: int = 384,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        max_images: int = 8,
        topk_frames_per_view: int = 4,
        image_resolver: Optional[Callable[[Dict[str, Any]], Any]] = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ) -> "QwenTeacherJudge":
        try:
            from transformers import Qwen3VLForConditionalGeneration
        except Exception as exc:
            raise ImportError(
                "QwenTeacherJudge requires a recent transformers build with Qwen3-VL support. "
                "Install it with `pip install git+https://github.com/huggingface/transformers accelerate`."
            ) from exc

        model_init_kwargs = build_hf_model_init_kwargs(
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
            prefer_inference_dtype=True,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_path), **model_init_kwargs)
        ensure_flash_attention_supported_dtype(
            model,
            attn_implementation=attn_implementation,
        )
        model.eval()
        processor = _configure_qwen_processor(load_auto_processor_with_compat(str(model_path)))
        return cls(
            model=model,
            processor=processor,
            input_mode=input_mode,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_images=max_images,
            topk_frames_per_view=topk_frames_per_view,
            image_resolver=image_resolver,
            log_fn=log_fn,
        )

    def judge_example(
        self,
        example: Dict[str, Any],
        *,
        input_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        results = self.judge_examples([example], input_mode=input_mode)
        return results[0] if results else normalize_teacher_judge_result({})

    def judge_examples(
        self,
        examples: List[Dict[str, Any]],
        *,
        input_mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not examples:
            return []
        messages_batch = [
            build_teacher_judge_messages(
                example,
                input_mode=input_mode or self.input_mode,
                max_images=self.max_images,
                topk_frames_per_view=self.topk_frames_per_view,
                image_resolver=self.image_resolver,
            )
            for example in examples
        ]
        output_texts = self._generate_batch(messages_batch)
        return [parse_teacher_judge_response(output_text) for output_text in output_texts]

    def annotate_example(
        self,
        example: Dict[str, Any],
        *,
        input_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        results = self.annotate_examples([example], input_mode=input_mode)
        return results[0] if results else copy.deepcopy(example)

    def annotate_examples(
        self,
        examples: List[Dict[str, Any]],
        *,
        input_mode: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        judgments = self.judge_examples(examples, input_mode=input_mode)
        return [
            attach_teacher_judge_labels(example, judgment)
            for example, judgment in zip(examples, judgments)
        ]

    def judge_package_records(self, package_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not package_records:
            return []
        messages_batch: List[List[Dict[str, Any]]] = []
        for record in package_records:
            prebuilt_messages = record.get("messages")
            if not isinstance(prebuilt_messages, list) or not prebuilt_messages:
                raise ValueError(
                    "Teacher judge package records must contain prebuilt materialized messages. "
                    f"source_row_index={record.get('source_row_index')}"
                )
            messages_batch.append(validate_materialized_teacher_judge_messages(prebuilt_messages))
        self._log(
            "teacher judge batch start: "
            f"records={len(messages_batch)} "
            f"images={sum(self._count_images(messages) for messages in messages_batch)}"
        )
        output_texts = self._generate_batch(messages_batch)
        self._log(f"teacher judge batch parse start: records={len(output_texts)}")
        parse_start = time.perf_counter()
        results = [parse_teacher_judge_response(output_text) for output_text in output_texts]
        self._log(
            "teacher judge batch parse finish: "
            f"records={len(output_texts)} elapsed_sec={time.perf_counter() - parse_start:.3f}"
        )
        return results

    def _generate(self, messages: List[Dict[str, Any]]) -> str:
        inputs = self._build_inputs(messages)
        inputs = self._move_to_model_device(inputs)
        output_ids = self.model.generate(**inputs, **self._generation_kwargs())
        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0] if output_text else ""

    def _generate_batch(self, messages_batch: List[List[Dict[str, Any]]]) -> List[str]:
        if not messages_batch:
            return []
        start = time.perf_counter()
        self._log(f"teacher judge batch tensorize start: records={len(messages_batch)}")
        inputs = self._build_inputs_batch(messages_batch)
        self._log(
            "teacher judge batch tensorize finish: "
            f"records={len(messages_batch)} elapsed_sec={time.perf_counter() - start:.3f}"
        )
        inputs = self._move_to_model_device(inputs)
        generate_start = time.perf_counter()
        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        prompt_tokens = int(input_ids.numel()) if hasattr(input_ids, "numel") else 0
        self._log(
            "teacher judge batch generate start: "
            f"records={len(messages_batch)} prompt_tokens={prompt_tokens}"
        )
        output_ids = self.model.generate(**inputs, **self._generation_kwargs())
        self._log(
            "teacher judge batch generate finish: "
            f"records={len(messages_batch)} elapsed_sec={time.perf_counter() - generate_start:.3f}"
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, output_ids)
        ]
        decode_start = time.perf_counter()
        self._log(f"teacher judge batch decode start: records={len(messages_batch)}")
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        self._log(
            "teacher judge batch decode finish: "
            f"records={len(messages_batch)} elapsed_sec={time.perf_counter() - decode_start:.3f}"
        )
        return [str(text) for text in output_texts]

    def _build_inputs_batch(self, messages_batch: List[List[Dict[str, Any]]]) -> Any:
        try:
            return self.processor.apply_chat_template(
                messages_batch,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except (TypeError, ValueError):
            prompt_texts: List[str] = []
            image_inputs_batch: List[List[Any]] = []
            for messages in messages_batch:
                prompt_texts.append(
                    self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
                image_inputs: List[Any] = []
                for message in messages:
                    content = message.get("content")
                    if not isinstance(content, list):
                        continue
                    for item in content:
                        if item.get("type") == "image" and "image" in item:
                            image_inputs.append(item["image"])
                image_inputs_batch.append(image_inputs)
            processor_kwargs: Dict[str, Any] = {
                "text": prompt_texts,
                "padding": True,
                "return_tensors": "pt",
            }
            if any(image_inputs_batch):
                processor_kwargs["images"] = image_inputs_batch
            return self.processor(**processor_kwargs)

    def _count_images(self, messages: List[Dict[str, Any]]) -> int:
        image_count = 0
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    image_count += 1
        return image_count

    def _log(self, message: str) -> None:
        if callable(self.log_fn):
            self.log_fn(str(message))

    def _normalize_generation_config(self) -> None:
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is None:
            return
        if not self.do_sample:
            for key in ("temperature", "top_p", "top_k"):
                if hasattr(generation_config, key):
                    try:
                        setattr(generation_config, key, None)
                    except Exception:
                        pass

    def _build_inputs(self, messages: List[Dict[str, Any]]) -> Any:
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs: List[Any] = []
            for message in messages:
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for item in content:
                    if item.get("type") == "image" and "image" in item:
                        image_inputs.append(item["image"])
            processor_kwargs: Dict[str, Any] = {
                "text": prompt_text,
                "padding": True,
                "return_tensors": "pt",
            }
            if image_inputs:
                processor_kwargs["images"] = image_inputs
            return self.processor(**processor_kwargs)

    def _move_to_model_device(self, inputs: Any) -> Any:
        if not hasattr(inputs, "to"):
            return inputs
        device = getattr(self.model, "device", None)
        if device is None:
            return inputs
        return inputs.to(device)

    def _generation_kwargs(self) -> Dict[str, Any]:
        return _build_generation_kwargs(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )
