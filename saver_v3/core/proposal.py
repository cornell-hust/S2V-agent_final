from __future__ import annotations

import hashlib
import json
import math
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch


_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "for",
    "with",
    "before",
    "after",
    "near",
    "another",
    "other",
    "potential",
    "look",
}

_CLOTHING_TERMS = (
    "shirt",
    "jacket",
    "coat",
    "hoodie",
    "vest",
    "uniform",
    "pants",
    "dress",
    "hat",
    "helmet",
)
_SCENE_TERMS = {
    "intersection",
    "shop",
    "store",
    "sidewalk",
    "road",
    "street",
    "restaurant",
    "terrace",
    "parking",
    "driveway",
    "workbench",
    "site",
}
_RELATION_TO_QUERY = (
    (("attack", "attacked", "assault", "hit", "kick", "punch", "lunge", "fight", "altercation"), "physical struggle"),
    (("snatch", "rob", "robbery", "steal", "grab bag"), "bag snatch"),
    (("fall", "falls", "falling", "fell", "trip", "stumble"), "person falling"),
    (("ground", "lying", "fallen"), "person on ground"),
    (("fire", "flame", "flames"), "fire"),
    (("smoke",), "smoke"),
)

_QUERY_PACKAGE_KEYS = (
    "event_cue",
    "key_objects",
    "scene_context",
    "hypothesis",
    "negative_constraints",
    "rewrite_reason",
)

_DESCRIPTION_PRIMARY_REWRITE_REASONS = {"oracle_moment_description_primary"}


def normalize_query_text(text: str) -> str:
    normalized = str(text or "").strip().lower()
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\b(?:the|a|an)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in normalize_query_text(text).split()
        if token and token not in _STOPWORDS
    }


def _classify_query_kind(text: str) -> str:
    tokens = _tokenize(text)
    if not tokens:
        return "object"
    if tokens & _SCENE_TERMS:
        return "scene_context"
    if any(term in text for term in _CLOTHING_TERMS):
        return "attribute_object"
    if any(keyword in text for keywords, _ in _RELATION_TO_QUERY for keyword in keywords):
        return "event_relation"
    return "object"


def _append_query_entry(entries: list[dict[str, Any]], text: str, *, kind: Optional[str] = None, weight: float = 1.0) -> None:
    normalized = normalize_query_text(text)
    if not normalized:
        return
    resolved_kind = kind or _classify_query_kind(normalized)
    for entry in entries:
        if entry["text"] == normalized:
            entry["weight"] = max(float(entry.get("weight") or 0.0), float(weight))
            if entry.get("kind") == "object" and resolved_kind != "object":
                entry["kind"] = resolved_kind
            return
    entries.append(
        {
            "text": normalized,
            "kind": resolved_kind,
            "weight": float(weight),
        }
    )


def normalize_key_object_phrases(key_objects: Sequence[str]) -> List[Dict[str, Any]]:
    normalized_queries: List[Dict[str, Any]] = []
    for raw_text in key_objects or []:
        normalized = normalize_query_text(raw_text)
        if not normalized:
            continue

        # Split compact conjunctions like "pedestrians and vehicles".
        for piece in re.split(r"\s+(?:and|/)\s+", normalized):
            if piece and len(piece.split()) <= 4:
                _append_query_entry(normalized_queries, piece, weight=0.9)

        clothing_pattern = re.compile(
            rf"\b(person|man|woman|individual|victim|attacker)\s+in\s+([a-z0-9 ]{{1,40}}?(?:{'|'.join(_CLOTHING_TERMS)}))\b"
        )
        for match in clothing_pattern.finditer(normalized):
            subject = match.group(1)
            if subject in {"victim", "attacker", "individual"}:
                subject = "person"
            _append_query_entry(normalized_queries, f"{subject} in {match.group(2)}", kind="attribute_object", weight=1.0)

        if len(normalized.split()) <= 4:
            _append_query_entry(normalized_queries, normalized, weight=1.0)

        for keywords, query_text in _RELATION_TO_QUERY:
            if any(keyword in normalized for keyword in keywords):
                weight = 0.8 if query_text not in {"physical struggle", "person on ground"} else 0.9
                _append_query_entry(normalized_queries, query_text, kind="event_relation", weight=weight)

    return normalized_queries


def normalize_description_query_phrases(description: str) -> List[Dict[str, Any]]:
    normalized_description = normalize_query_text(description)
    if not normalized_description:
        return []
    normalized_queries: List[Dict[str, Any]] = []
    _append_query_entry(
        normalized_queries,
        normalized_description,
        kind="event_relation",
        weight=1.2,
    )
    for entry in normalize_key_object_phrases([description]):
        _append_query_entry(
            normalized_queries,
            str(entry.get("text") or ""),
            kind=str(entry.get("kind") or "") or None,
            weight=float(entry.get("weight") or 0.0),
        )
    return normalized_queries


def compose_scene_anchored_query(description: str, scene_context: str = "") -> str:
    normalized_description = normalize_query_text(description)
    if not normalized_description:
        return ""
    normalized_scene = normalize_query_text(scene_context)
    if normalized_scene:
        return f"{normalized_description} in the {normalized_scene}"
    return f"{normalized_description} in the scene"


def build_proposal_supervision(
    *,
    key_objects: Sequence[str],
    evidence_moments: Sequence[Dict[str, Any]],
    scene_context: str = "",
) -> Dict[str, Any]:
    queries: List[Dict[str, Any]] = []
    moments = [dict(moment) for moment in (evidence_moments or [])]
    next_query_index = 1
    for moment in moments:
        moment_id = str(moment.get("moment_id") or "").strip()
        role = str(moment.get("role") or "").strip()
        raw_text = str(moment.get("description") or "").strip()
        normalized_queries = normalize_description_query_phrases(raw_text)
        if not raw_text or not normalized_queries:
            continue
        query_text = compose_scene_anchored_query(raw_text, scene_context)
        queries.append(
            {
                "query_id": f"pq{next_query_index}",
                "raw_text": raw_text,
                "query_text": query_text or raw_text,
                "normalized_queries": normalized_queries,
                "linked_moment_ids": [moment_id] if moment_id else [],
                "linked_roles": [role] if role else [],
                "linked_windows_sec": [
                    [float(moment.get("start_sec") or 0.0), float(moment.get("end_sec") or 0.0)]
                ],
                "alignment_source": "moment_description",
                "query_origin": "moment_description_primary",
                "moment_id": moment_id,
                "role": role,
            }
        )
        next_query_index += 1

    for raw_text in key_objects or []:
        normalized_queries = normalize_key_object_phrases([str(raw_text or "")])
        if not normalized_queries:
            continue
        linked_moments: List[Dict[str, Any]] = []
        normalized_tokens = [_tokenize(entry["text"]) for entry in normalized_queries]
        for moment in moments:
            role = str(moment.get("role") or "")
            moment_tokens = _tokenize(f"{moment.get('description') or ''} {role}")
            best_overlap = 0.0
            for query_tokens in normalized_tokens:
                if not query_tokens:
                    continue
                overlap = len(query_tokens & moment_tokens) / float(max(len(query_tokens), 1))
                best_overlap = max(best_overlap, overlap)
            if best_overlap <= 0.0:
                continue
            linked_moments.append(moment)
        queries.append(
            {
                "query_id": f"pq{next_query_index}",
                "raw_text": str(raw_text or ""),
                "normalized_queries": normalized_queries,
                "linked_moment_ids": [str(moment.get("moment_id") or "") for moment in linked_moments if moment.get("moment_id")],
                "linked_roles": [str(moment.get("role") or "") for moment in linked_moments if moment.get("role")],
                "linked_windows_sec": [
                    [float(moment.get("start_sec") or 0.0), float(moment.get("end_sec") or 0.0)]
                    for moment in linked_moments
                ],
                "alignment_source": "weak_alignment",
                "query_origin": "key_objects_auxiliary",
            }
        )
        next_query_index += 1
    return {
        "queries": queries,
        "has_oracle_supervision": bool(queries),
    }


def select_query_for_moment(
    proposal_supervision: Optional[Dict[str, Any]],
    *,
    moment: Dict[str, Any],
    fallback_query: str,
) -> tuple[str, str]:
    if not isinstance(proposal_supervision, dict):
        return fallback_query, "role_fallback"
    moment_id = str(moment.get("moment_id") or "")
    role = str(moment.get("role") or "")
    moment_tokens = _tokenize(f"{moment.get('description') or ''} {role}")
    for query_group in proposal_supervision.get("queries") or []:
        query_origin = str(query_group.get("query_origin") or "").strip().lower()
        if query_origin != "moment_description_primary":
            continue
        if moment_id and str(query_group.get("moment_id") or "").strip() == moment_id:
            query_text = str(query_group.get("query_text") or query_group.get("raw_text") or "").strip()
            if query_text:
                return query_text, "description_primary"
    best_text = normalize_query_text(fallback_query) or str(fallback_query or "")
    best_source = "hypothesis_fallback"
    best_score = -1.0
    for query_group in proposal_supervision.get("queries") or []:
        linked_moment_ids = {str(value) for value in query_group.get("linked_moment_ids") or [] if value}
        linked_roles = {str(value) for value in query_group.get("linked_roles") or [] if value}
        if moment_id and linked_moment_ids and moment_id not in linked_moment_ids and role not in linked_roles:
            continue
        query_origin = str(query_group.get("query_origin") or "").strip().lower()
        origin_bonus = 2.0 if query_origin == "moment_description_primary" else 0.0
        for entry in query_group.get("normalized_queries") or []:
            text = str(entry.get("text") or "")
            if not text:
                continue
            query_tokens = _tokenize(text)
            overlap = len(query_tokens & moment_tokens) / float(max(len(query_tokens), 1)) if query_tokens else 0.0
            score = float(entry.get("weight") or 0.0) + overlap + origin_bonus
            if score > best_score:
                best_score = score
                best_text = str(query_group.get("query_text") or text or "")
                best_source = (
                    "description_primary"
                    if query_origin == "moment_description_primary"
                    else "object_fallback"
                )
    return best_text, best_source


def _append_weighted_text(
    entries: list[dict[str, Any]],
    text: str,
    *,
    weight: float,
    source: str,
) -> None:
    normalized = normalize_query_text(text)
    if not normalized:
        return
    for entry in entries:
        if entry["text"] == normalized:
            entry["weight"] = max(float(entry.get("weight") or 0.0), float(weight))
            return
    entries.append({"text": normalized, "weight": float(weight), "source": source})


def render_query_context_texts(
    query: str,
    *,
    role: str = "",
    search_anchor: Optional[Dict[str, Any]] = None,
    query_source: str = "model",
    query_package: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    package = None
    if isinstance(query_package, dict) and any(query_package.values()):
        package = normalize_query_package(query_package, fallback_query=query)
        legacy_rendering = render_query_package_texts(package)
        legacy_rendering["query_summary"] = summarize_query_package(package) or normalize_query_text(query)
        legacy_rendering["query_context"] = {
            "role": str(role or "").strip(),
            "query_source": str(query_source or "model"),
            "search_anchor": dict(search_anchor or {}),
        }
        return legacy_rendering

    normalized_query = normalize_query_text(query)
    anchor = dict(search_anchor or {})
    positive_texts: List[Dict[str, Any]] = []
    negative_texts: List[Dict[str, Any]] = []

    if normalized_query:
        _append_weighted_text(positive_texts, normalized_query, weight=0.55, source="query")

    scene = str(anchor.get("scene") or "").strip()
    if normalized_query and scene:
        _append_weighted_text(
            positive_texts,
            f"{normalized_query} in {scene}",
            weight=0.20,
            source="scene",
        )

    active_hypothesis = str(anchor.get("active_hypothesis") or "").strip()
    if normalized_query and active_hypothesis:
        _append_weighted_text(
            positive_texts,
            f"{normalized_query} {active_hypothesis}",
            weight=0.15,
            source="active_hypothesis",
        )

    expected_stage = normalize_query_text(anchor.get("expected_stage") or role)
    if normalized_query and expected_stage:
        _append_weighted_text(
            positive_texts,
            f"{expected_stage} evidence around {normalized_query}",
            weight=0.10,
            source="expected_stage",
        )

    last_search_feedback = str(anchor.get("last_search_feedback") or "").strip()
    if last_search_feedback:
        _append_weighted_text(
            negative_texts,
            last_search_feedback,
            weight=0.05,
            source="last_search_feedback",
        )

    return {
        "query_context": {
            "role": str(role or "").strip(),
            "query_source": str(query_source or "model"),
            "search_anchor": anchor,
        },
        "positive_texts": positive_texts,
        "negative_texts": negative_texts,
        "query_summary": normalized_query,
    }


def is_description_primary_query_package(query_package: Optional[Dict[str, Any]]) -> bool:
    rewrite_reason = str((query_package or {}).get("rewrite_reason") or "").strip().lower()
    return rewrite_reason in _DESCRIPTION_PRIMARY_REWRITE_REASONS


def normalize_query_package(
    query_package: Optional[Dict[str, Any]],
    *,
    fallback_query: str = "",
    fallback_scene_context: str = "",
    rewrite_reason: str = "",
) -> Dict[str, Any]:
    package = dict(query_package or {})
    event_cue = str(package.get("event_cue") or "").strip()
    key_objects = [str(value).strip() for value in list(package.get("key_objects") or []) if str(value).strip()]
    scene_context = str(package.get("scene_context") or fallback_scene_context or "").strip()
    hypothesis = str(package.get("hypothesis") or "").strip()
    negative_constraints = [
        str(value).strip()
        for value in list(package.get("negative_constraints") or [])
        if str(value).strip()
    ]
    resolved_rewrite_reason = str(package.get("rewrite_reason") or rewrite_reason or "").strip()

    fallback_normalized = normalize_query_text(fallback_query)
    if not event_cue:
        event_cue = fallback_normalized or str(fallback_query or "").strip()
    if not hypothesis:
        hypothesis = event_cue

    normalized_key_objects: List[str] = []
    seen_key_objects = set()
    for value in key_objects:
        normalized = normalize_query_text(value) or value
        if not normalized or normalized in seen_key_objects:
            continue
        normalized_key_objects.append(normalized)
        seen_key_objects.add(normalized)

    normalized_negative_constraints: List[str] = []
    seen_negative_constraints = set()
    for value in negative_constraints:
        normalized = normalize_query_text(value) or value
        if not normalized or normalized in seen_negative_constraints:
            continue
        normalized_negative_constraints.append(normalized)
        seen_negative_constraints.add(normalized)

    return {
        "event_cue": normalize_query_text(event_cue) or event_cue,
        "key_objects": normalized_key_objects,
        "scene_context": normalize_query_text(scene_context) or scene_context,
        "hypothesis": normalize_query_text(hypothesis) or hypothesis,
        "negative_constraints": normalized_negative_constraints,
        "rewrite_reason": resolved_rewrite_reason,
    }


def render_query_package_texts(query_package: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    package = normalize_query_package(query_package)
    description_primary = is_description_primary_query_package(package)
    positive_texts: List[Dict[str, Any]] = []
    negative_texts: List[Dict[str, Any]] = []

    def _append_text(entries: List[Dict[str, Any]], text: str, *, weight: float, source: str) -> None:
        normalized = normalize_query_text(text)
        if not normalized:
            return
        for entry in entries:
            if entry["text"] == normalized:
                entry["weight"] = max(float(entry["weight"]), float(weight))
                return
        entries.append({"text": normalized, "weight": float(weight), "source": source})

    if package["event_cue"]:
        _append_text(
            positive_texts,
            package["event_cue"],
            weight=0.30 if description_primary else 0.40,
            source="event_cue",
        )
    for value in package["key_objects"]:
        _append_text(
            positive_texts,
            value,
            weight=0.15 if description_primary else 0.20,
            source="key_objects",
        )
    if package["scene_context"]:
        _append_text(positive_texts, package["scene_context"], weight=0.10, source="scene_context")
    if package["hypothesis"]:
        _append_text(
            positive_texts,
            package["hypothesis"],
            weight=0.45 if description_primary else 0.25,
            source="hypothesis",
        )
    for value in package["negative_constraints"]:
        _append_text(negative_texts, value, weight=0.15, source="negative_constraints")

    return {
        "query_package": package,
        "positive_texts": positive_texts,
        "negative_texts": negative_texts,
        "rewrite_reason": package["rewrite_reason"],
    }


def summarize_query_package(query_package: Optional[Dict[str, Any]]) -> str:
    rendered = render_query_package_texts(query_package)
    if is_description_primary_query_package(rendered.get("query_package")):
        hypothesis = str(((rendered.get("query_package") or {}).get("hypothesis")) or "").strip()
        if hypothesis:
            return hypothesis
    positive_texts = list(rendered.get("positive_texts") or [])
    if positive_texts:
        return str(positive_texts[0].get("text") or "").strip()
    package = rendered.get("query_package") or {}
    return str(package.get("event_cue") or package.get("hypothesis") or "").strip()


def compute_frame_cache_signature(*, fps: float, frame_indices: Sequence[int], num_frames: int) -> str:
    payload = {
        "fps": round(float(fps), 6),
        "frame_indices": [int(index) for index in frame_indices],
        "num_frames": int(num_frames),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


def coerce_feature_cache_payload(
    payload: Any,
    *,
    fps: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
) -> Optional[Dict[str, Any]]:
    if payload is None:
        return None
    if isinstance(payload, torch.Tensor):
        resolved_frame_indices = [int(index) for index in (frame_indices or range(int(payload.shape[0])))]
        resolved_fps = float(fps or 1.0)
        timestamps = [round(float(index) / max(resolved_fps, 1e-6), 6) for index in range(len(resolved_frame_indices))]
        return {
            "version": "saver_feature_cache_v0",
            "model_name": "legacy_tensor",
            "fps": resolved_fps,
            "frame_indices": resolved_frame_indices,
            "timestamps_sec": timestamps,
            "embeddings": payload,
            "embedding_dim": int(payload.shape[-1]) if payload.ndim >= 2 else 1,
            "normalized": False,
            "frame_cache_signature": compute_frame_cache_signature(
                fps=resolved_fps,
                frame_indices=resolved_frame_indices,
                num_frames=len(resolved_frame_indices),
            ),
        }
    if not isinstance(payload, dict):
        return None
    embeddings = payload.get("embeddings")
    if isinstance(embeddings, torch.Tensor):
        resolved_frame_indices = [int(index) for index in (payload.get("frame_indices") or frame_indices or range(int(embeddings.shape[0])))]
        resolved_fps = float(payload.get("fps") or fps or 1.0)
        timestamps = payload.get("timestamps_sec")
        if not isinstance(timestamps, list) or len(timestamps) != len(resolved_frame_indices):
            timestamps = [round(float(index) / max(resolved_fps, 1e-6), 6) for index in range(len(resolved_frame_indices))]
        return {
            "version": str(payload.get("version") or "saver_feature_cache_v1"),
            "model_name": str(payload.get("model_name") or "unknown"),
            "fps": resolved_fps,
            "frame_indices": resolved_frame_indices,
            "timestamps_sec": [float(value) for value in timestamps],
            "embeddings": embeddings,
            "embedding_dim": int(payload.get("embedding_dim") or (embeddings.shape[-1] if embeddings.ndim >= 2 else 1)),
            "normalized": bool(payload.get("normalized", False)),
            "frame_cache_signature": str(
                payload.get("frame_cache_signature")
                or compute_frame_cache_signature(
                    fps=resolved_fps,
                    frame_indices=resolved_frame_indices,
                    num_frames=len(resolved_frame_indices),
                )
            ),
        }
    return None


def _l2_normalize(tensor: torch.Tensor) -> torch.Tensor:
    denom = tensor.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return tensor / denom


def _resolve_feature_tensor_device(*, proposal_runtime: Any = None, feature_cache: Optional[Dict[str, Any]] = None) -> torch.device:
    if proposal_runtime is not None:
        device_value = getattr(proposal_runtime, "device", None)
        if device_value is not None:
            try:
                return torch.device(str(device_value))
            except Exception:
                pass
        model = getattr(proposal_runtime, "model", None)
        if model is not None:
            try:
                return next(model.parameters()).device
            except Exception:
                pass
    if isinstance(feature_cache, dict):
        embeddings = feature_cache.get("embeddings")
        if isinstance(embeddings, torch.Tensor):
            return embeddings.device
    return torch.device("cpu")


def _to_feature_device_tensor(value: torch.Tensor, *, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    tensor = value.detach()
    if tensor.device != device:
        tensor = tensor.to(device)
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    return tensor


def coerce_encoder_feature_tensor(
    value: Any,
    *,
    preferred_keys: Sequence[str],
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value

    def _extract(candidate: Any) -> Optional[torch.Tensor]:
        if isinstance(candidate, torch.Tensor):
            return candidate
        if isinstance(candidate, (list, tuple)) and candidate:
            for item in reversed(candidate):
                if isinstance(item, torch.Tensor):
                    return item
        return None

    for key in preferred_keys:
        candidate = None
        if isinstance(value, dict) and key in value:
            candidate = value.get(key)
        elif hasattr(value, key):
            candidate = getattr(value, key)
        tensor = _extract(candidate)
        if tensor is None:
            continue
        if key in {"last_hidden_state", "hidden_state", "hidden_states"} and tensor.ndim >= 3:
            return tensor.mean(dim=1)
        return tensor

    raise TypeError(
        "Could not coerce encoder output to a feature tensor. "
        f"Tried keys={list(preferred_keys)} on type={type(value).__name__}."
    )


def _cluster_sorted_indices(indices: Sequence[int], *, max_gap: int) -> List[List[int]]:
    clusters: List[List[int]] = []
    for index in sorted(int(value) for value in indices):
        if not clusters or index - clusters[-1][-1] > max_gap:
            clusters.append([index])
        else:
            clusters[-1].append(index)
    return clusters


def _compute_dynamic_num_frames(
    *,
    window_span_sec: float,
    candidate_frame_scores: Sequence[float],
    candidate_windows: Sequence[Dict[str, Any]],
    requested_num_frames: int = 0,
    max_num_frames: int = 8,
) -> tuple[int, Dict[str, Any]]:
    effective_cap = max(1, int(max_num_frames))
    if requested_num_frames > 0:
        effective_cap = max(1, min(effective_cap, int(requested_num_frames)))
    base_k = 1 if float(window_span_sec) < 2.0 else 2
    ambiguity_bonus = 0
    if len(candidate_frame_scores) >= 5:
        top1 = float(candidate_frame_scores[0])
        top5 = float(candidate_frame_scores[min(4, len(candidate_frame_scores) - 1)])
        ambiguity_bonus = 1 if top1 - top5 < 0.08 else 0
    density_bonus = 1 if len(candidate_windows) >= 3 else 0
    adaptive_num_frames = max(1, min(effective_cap, base_k + ambiguity_bonus + density_bonus))
    return adaptive_num_frames, {
        "base_k": int(base_k),
        "ambiguity_bonus": int(ambiguity_bonus),
        "density_bonus": int(density_bonus),
        "requested_num_frames": int(requested_num_frames or 0),
        "effective_cap": int(effective_cap),
    }


def _build_dpp_kernel(
    *,
    candidate_embeddings: torch.Tensor,
    candidate_scores: torch.Tensor,
    candidate_timestamps: torch.Tensor,
    alpha: float = 1.0,
    tau_sec: float = 2.0,
) -> torch.Tensor:
    if candidate_embeddings.ndim != 2:
        raise ValueError("candidate_embeddings must be rank-2.")
    similarity = torch.matmul(candidate_embeddings, candidate_embeddings.T).clamp(min=0.0, max=1.0)
    if tau_sec > 0:
        temporal_distance = torch.abs(candidate_timestamps[:, None] - candidate_timestamps[None, :])
        temporal_kernel = torch.exp(-temporal_distance / float(tau_sec))
        similarity = similarity * temporal_kernel
    if candidate_scores.numel() == 0:
        quality = torch.ones((candidate_embeddings.shape[0],), dtype=torch.float32)
    else:
        score_min = torch.min(candidate_scores)
        score_max = torch.max(candidate_scores)
        if float(score_max - score_min) > 1e-8:
            normalized_scores = (candidate_scores - score_min) / (score_max - score_min)
        else:
            normalized_scores = torch.ones_like(candidate_scores)
        quality = torch.exp(float(alpha) * normalized_scores).float()
    quality_outer = quality[:, None] * quality[None, :]
    kernel = similarity * quality_outer
    diagonal = torch.diagonal(kernel).clamp_min(1e-6)
    diagonal_indices = torch.arange(kernel.shape[0], dtype=torch.long)
    kernel[diagonal_indices, diagonal_indices] = diagonal
    return kernel


def _encode_query_text_entries(
    proposal_runtime: Any,
    text_entries: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    entries = [dict(entry) for entry in text_entries if str(entry.get("text") or "").strip()]
    if not entries:
        return []

    texts = [str(entry["text"]) for entry in entries]
    encoded_entries: List[Dict[str, Any]] = []
    feature_device = _resolve_feature_tensor_device(proposal_runtime=proposal_runtime)

    try:
        embeddings = proposal_runtime.encode_texts(texts)
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError("proposal_runtime.encode_texts must return a torch.Tensor.")
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        if int(embeddings.shape[0]) != len(entries):
            raise ValueError("proposal_runtime.encode_texts returned a mismatched batch size.")
        embeddings = _to_feature_device_tensor(embeddings, device=feature_device, dtype=torch.float32)
        for entry, embedding in zip(entries, embeddings):
            encoded_entry = dict(entry)
            encoded_entry["embedding"] = embedding.detach()
            encoded_entries.append(encoded_entry)
        return encoded_entries
    except Exception:
        pass

    for entry in entries:
        text = str(entry["text"])
        try:
            embedding = proposal_runtime.encode_texts([text])
        except Exception:
            continue
        if not isinstance(embedding, torch.Tensor):
            raise TypeError("proposal_runtime.encode_texts must return a torch.Tensor.")
        if embedding.ndim == 2:
            if int(embedding.shape[0]) < 1:
                continue
            embedding = embedding[0]
        elif embedding.ndim != 1:
            raise ValueError("proposal_runtime.encode_texts returned an unexpected tensor rank.")
        encoded_entry = dict(entry)
        encoded_entry["embedding"] = _to_feature_device_tensor(embedding, device=feature_device, dtype=torch.float32)
        encoded_entries.append(encoded_entry)
    return encoded_entries


def _greedy_map_dpp(kernel: torch.Tensor, *, num_select: int) -> List[int]:
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be a square matrix.")
    total = int(kernel.shape[0])
    if total <= 0 or num_select <= 0:
        return []
    num_select = min(int(num_select), total)
    selected: List[int] = []
    remaining = list(range(total))
    eye_cache: Dict[int, torch.Tensor] = {}
    while remaining and len(selected) < num_select:
        best_index = None
        best_score = None
        for candidate in remaining:
            subset = selected + [candidate]
            subset_tensor = torch.tensor(subset, dtype=torch.long)
            subkernel = kernel.index_select(0, subset_tensor).index_select(1, subset_tensor)
            matrix_size = int(subkernel.shape[0])
            if matrix_size not in eye_cache:
                eye_cache[matrix_size] = torch.eye(matrix_size, dtype=subkernel.dtype)
            stabilized = subkernel + 1e-6 * eye_cache[matrix_size]
            sign, logabsdet = torch.linalg.slogdet(stabilized)
            score = float(logabsdet.item()) if float(sign.item()) > 0 else float("-inf")
            if best_score is None or score > best_score:
                best_score = score
                best_index = candidate
        if best_index is None:
            break
        selected.append(int(best_index))
        remaining.remove(int(best_index))
    return selected


def feature_guided_frame_proposal(
    *,
    feature_cache: Optional[Dict[str, Any]],
    proposal_runtime: Any,
    query: str,
    query_package: Optional[Dict[str, Any]] = None,
    role: str = "",
    search_anchor: Optional[Dict[str, Any]] = None,
    start_sec: float,
    end_sec: float,
    fps: float,
    num_frames: int,
    top_k_candidates: int = 8,
    candidate_merge_gap_sec: float = 1.0,
    query_source: str = "model",
) -> Dict[str, Any]:
    normalized_package = None
    if isinstance(query_package, dict) and any(query_package.values()):
        normalized_package = normalize_query_package(query_package, fallback_query=query)
    query_rendering = render_query_context_texts(
        query,
        role=role,
        search_anchor=search_anchor,
        query_source=query_source,
        query_package=normalized_package,
    )
    normalized_query = str(query_rendering.get("query_summary") or normalize_query_text(query) or "").strip()
    base_metadata = {
        "proposal_backend": "uniform",
        "feature_cache_used": feature_cache is not None,
        "query_raw": str(normalized_query or query or ""),
        "query_normalized": normalized_query,
        "query_source": str(query_source or "model"),
        "query_rendering": query_rendering,
        "search_anchor": dict(search_anchor or {}),
        "proposal_candidate_count": 0,
        "proposal_candidate_frame_indices": [],
        "proposal_candidate_frame_scores": [],
        "proposal_candidate_windows": [],
        "selected_frame_indices": [],
        "selected_frame_scores": [],
        "adaptive_num_frames": 0,
    }
    if normalized_package is not None:
        base_metadata["query_package"] = normalized_package
    if feature_cache is None:
        return {**base_metadata, "feature_cache_used": False, "proposal_fallback_reason": "missing_feature_cache"}
    positive_texts = [dict(entry) for entry in list(query_rendering.get("positive_texts") or [])]
    negative_texts = [dict(entry) for entry in list(query_rendering.get("negative_texts") or [])]
    if proposal_runtime is None or not positive_texts:
        return {
            **base_metadata,
            "proposal_fallback_reason": "missing_query_encoder" if proposal_runtime is None else "empty_query_context",
        }

    embeddings = feature_cache.get("embeddings")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2 or embeddings.shape[0] == 0:
        return {**base_metadata, "proposal_fallback_reason": "invalid_feature_cache"}

    positive_encoded_entries = _encode_query_text_entries(proposal_runtime, positive_texts)
    negative_encoded_entries = _encode_query_text_entries(proposal_runtime, negative_texts)
    if not positive_encoded_entries:
        return {**base_metadata, "proposal_fallback_reason": "empty_query_embeddings"}
    feature_device = _resolve_feature_tensor_device(proposal_runtime=proposal_runtime, feature_cache=feature_cache)
    frame_embeddings = _to_feature_device_tensor(embeddings, device=feature_device, dtype=torch.float32)
    positive_embeddings = _to_feature_device_tensor(
        torch.stack([entry["embedding"] for entry in positive_encoded_entries], dim=0),
        device=feature_device,
        dtype=torch.float32,
    )
    if not bool(feature_cache.get("normalized", False)):
        frame_embeddings = _l2_normalize(frame_embeddings)
    positive_embeddings = _l2_normalize(positive_embeddings)
    negative_embeddings = None
    if negative_encoded_entries:
        negative_embeddings = _to_feature_device_tensor(
            torch.stack([entry["embedding"] for entry in negative_encoded_entries], dim=0),
            device=feature_device,
            dtype=torch.float32,
        )
        negative_embeddings = _l2_normalize(negative_embeddings)

    start_index = max(int(math.floor(float(start_sec) * float(fps))), 0)
    end_index = min(int(math.floor(float(end_sec) * float(fps))), int(frame_embeddings.shape[0]) - 1)
    if end_index < start_index:
        end_index = start_index
    search_embeddings = frame_embeddings[start_index : end_index + 1]
    positive_weight_tensor = torch.tensor(
        [float(entry.get("weight") or 0.0) for entry in positive_encoded_entries],
        dtype=torch.float32,
        device=feature_device,
    )
    positive_scores = torch.matmul(search_embeddings, positive_embeddings.T)
    scores = torch.matmul(positive_scores, positive_weight_tensor)
    if negative_embeddings is not None and negative_encoded_entries:
        negative_weight_tensor = torch.tensor(
            [float(entry.get("weight") or 0.0) for entry in negative_encoded_entries],
            dtype=torch.float32,
            device=feature_device,
        )
        negative_scores = torch.matmul(search_embeddings, negative_embeddings.T)
        scores = scores - torch.matmul(negative_scores, negative_weight_tensor)
    requested_num_frames = max(0, int(num_frames or 0))
    top_k_hint = max(requested_num_frames, 4)
    effective_top_k = min(int(scores.shape[0]), max(min(4 * top_k_hint, 64), int(top_k_candidates), 1))
    top_scores, top_local_indices = torch.topk(scores, k=effective_top_k)
    candidate_global_indices = [int(index) + start_index for index in top_local_indices.tolist()]
    candidate_frame_scores = [round(float(score), 6) for score in top_scores.tolist()]

    max_gap_frames = max(1, int(round(float(candidate_merge_gap_sec) * float(max(fps, 1e-6)))))
    candidate_windows: List[Dict[str, Any]] = []
    for cluster in _cluster_sorted_indices(candidate_global_indices, max_gap=max_gap_frames):
        cluster_scores = [float(scores[int(index - start_index)].item()) for index in cluster]
        candidate_windows.append(
            {
                "start_frame_index": int(cluster[0]),
                "end_frame_index": int(cluster[-1]),
                "start_sec": round(float(cluster[0]) / max(float(fps), 1e-6), 6),
                "end_sec": round(float(cluster[-1]) / max(float(fps), 1e-6), 6),
                "frame_indices": [int(index) for index in cluster],
                "score_max": round(max(cluster_scores), 6),
                "score_mean": round(sum(cluster_scores) / float(len(cluster_scores)), 6),
            }
        )
    candidate_windows.sort(key=lambda item: (-float(item["score_max"]), -float(item["score_mean"]), int(item["start_frame_index"])))

    if not candidate_windows:
        return {
            **base_metadata,
            "proposal_candidate_frame_indices": candidate_global_indices,
            "proposal_candidate_frame_scores": candidate_frame_scores,
            "proposal_fallback_reason": "empty_candidate_windows",
        }
    adaptive_num_frames, adaptive_meta = _compute_dynamic_num_frames(
        window_span_sec=max(float(end_sec) - float(start_sec), 1e-6),
        candidate_frame_scores=candidate_frame_scores,
        candidate_windows=candidate_windows,
        requested_num_frames=requested_num_frames,
        max_num_frames=8,
    )
    candidate_index_tensor = torch.tensor([int(index - start_index) for index in candidate_global_indices], dtype=torch.long, device=feature_device)
    candidate_embeddings = search_embeddings.index_select(0, candidate_index_tensor)
    candidate_timestamps = torch.tensor(
        [float(index) / max(float(fps), 1e-6) for index in candidate_global_indices],
        dtype=torch.float32,
        device=feature_device,
    )
    candidate_score_tensor = torch.tensor(candidate_frame_scores, dtype=torch.float32, device=feature_device)
    dpp_kernel = _build_dpp_kernel(
        candidate_embeddings=candidate_embeddings,
        candidate_scores=candidate_score_tensor,
        candidate_timestamps=candidate_timestamps,
        alpha=6.0,
        tau_sec=2.0,
    )
    try:
        selected_candidate_positions = _greedy_map_dpp(dpp_kernel, num_select=adaptive_num_frames)
        selected_indices = sorted(int(candidate_global_indices[index]) for index in selected_candidate_positions)
        selected_scores = [
            round(float(scores[int(index - start_index)].item()), 6)
            for index in selected_indices
        ]
        proposal_backend = "siglip_dpp"
        fallback_reason = None
    except Exception:
        selected_indices = sorted(candidate_global_indices[:adaptive_num_frames])
        selected_scores = [
            round(float(scores[int(index - start_index)].item()), 6)
            for index in selected_indices
        ]
        proposal_backend = "feature_topk"
        fallback_reason = "dpp_failure"
    metadata = {
        **base_metadata,
        "proposal_backend": proposal_backend,
        "proposal_candidate_count": len(candidate_windows),
        "proposal_candidate_frame_indices": candidate_global_indices,
        "proposal_candidate_frame_scores": candidate_frame_scores,
        "proposal_candidate_windows": candidate_windows,
        "selected_frame_indices": selected_indices,
        "selected_frame_scores": selected_scores,
        "adaptive_num_frames": int(adaptive_num_frames),
        "adaptive_frame_budget_meta": adaptive_meta,
        "proposal_dpp_kernel_meta": {
            "alpha": 6.0,
            "tau_sec": 2.0,
            "candidate_count": int(len(candidate_global_indices)),
            "selection_count": int(len(selected_indices)),
        },
        "proposal_fallback_reason": fallback_reason,
    }
    if normalized_package is not None:
        metadata["query_package"] = normalized_package
    return metadata


@dataclass
class SiglipFeatureEncoder:
    model: Any
    processor: Any
    device: str = "cpu"
    model_name: str = ""
    max_text_cache_size: int = 4096
    _text_feature_cache: "OrderedDict[str, torch.Tensor]" = field(default_factory=OrderedDict)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        torch_dtype: str | torch.dtype = "auto",
        device: str = "cpu",
    ) -> "SiglipFeatureEncoder":
        from transformers import AutoModel, AutoProcessor

        dtype_arg = None
        if isinstance(torch_dtype, str):
            if torch_dtype not in {"", "auto", "none"}:
                dtype_arg = getattr(torch, torch_dtype)
        else:
            dtype_arg = torch_dtype
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        model = AutoModel.from_pretrained(model_path, torch_dtype=dtype_arg, local_files_only=True)
        model.to(device)
        model.eval()
        return cls(model=model, processor=processor, device=device, model_name=model_path)

    def _to_pil_images(self, images: torch.Tensor) -> List[Any]:
        from PIL import Image

        pil_images: List[Any] = []
        for frame in images.detach().cpu():
            if frame.ndim != 3:
                raise ValueError("Expected image tensor in CHW format.")
            array = frame.permute(1, 2, 0).contiguous().numpy()
            if array.dtype != np.uint8:
                array = array.astype("float32")
                array = array.clip(0.0, 255.0).astype("uint8")
            pil_images.append(Image.fromarray(array))
        return pil_images

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        pil_images = self._to_pil_images(images)
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                features = self.model.get_image_features(**inputs)
            else:
                outputs = self.model(**inputs)
                features = outputs
        features = coerce_encoder_feature_tensor(
            features,
            preferred_keys=("image_embeds", "pooler_output", "last_hidden_state", "hidden_states"),
        )
        return _l2_normalize(features.detach().cpu())

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        normalized_texts = [str(text) for text in texts]
        if not normalized_texts:
            return torch.empty((0, 0), dtype=torch.float32)

        missing_texts: List[str] = []
        seen_missing = set()
        for text in normalized_texts:
            if text in self._text_feature_cache:
                self._text_feature_cache.move_to_end(text)
                continue
            if text in seen_missing:
                continue
            seen_missing.add(text)
            missing_texts.append(text)

        if missing_texts:
            inputs = self.processor(text=missing_texts, return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                if hasattr(self.model, "get_text_features"):
                    features = self.model.get_text_features(**inputs)
                else:
                    outputs = self.model(**inputs)
                    features = outputs
            features = coerce_encoder_feature_tensor(
                features,
                preferred_keys=("text_embeds", "pooler_output", "last_hidden_state", "hidden_states"),
            )
            features = _l2_normalize(features.detach().to(self.device, dtype=torch.float32))
            for text, feature in zip(missing_texts, features):
                self._text_feature_cache[text] = feature.clone()
                self._text_feature_cache.move_to_end(text)
                while len(self._text_feature_cache) > max(1, int(self.max_text_cache_size)):
                    self._text_feature_cache.popitem(last=False)

        stacked_features = [self._text_feature_cache[text] for text in normalized_texts]
        return torch.stack(stacked_features, dim=0)
