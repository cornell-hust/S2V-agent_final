#!/usr/bin/env python3
"""Convert canonical SAVER-style annotations into agent-ready training views.

This script keeps the benchmark-facing canonical JSONL untouched and derives
stable training/evaluation views for a TimeSearch-R-style SAVER agent.

Current adapters:
- ``msad_saver_qwen``: canonical records from
  ``msad_saver_with_qwen.jsonl``

Output modes:
- ``canonical_passthrough``: enrich canonical records with derived second-based
  fields while preserving the original structure.
- ``agent_train``: add stable task/schema/tool supervision for the future agent.
- ``oracle_sft``: add a stage-query-centric oracle trajectory for warm-start
  SFT.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from split_utils import parse_include_splits
from saver_v3.core.categories import CANONICAL_POLICY_CATEGORIES, canonicalize_saver_category
from saver_v3.core.protocol_guidance import (
    EVENT_CHAIN_STAGES,
    build_counterfactual_type_schema,
    build_stage_selected_moment_ids_schema,
    event_chain_stage_for_role,
    normalize_event_chain_stages,
    normalize_stage_selected_moment_ids,
)
from saver_v3.core.proposal import (
    build_proposal_supervision,
    compose_scene_anchored_query,
    normalize_query_package,
    normalize_query_text,
    select_query_for_moment,
)
from saver_v3.core.self_verification import build_policy_self_verification_payload
from saver_v3.core.semantic_answer import build_semantic_answer_payload


SCHEMA_VERSION = "saver_agent.v2"
DEFAULT_DERIVED_PRECURSOR_SECONDS = 2.0
DEFAULT_DERIVED_PRECURSOR_FRACTION = 0.2
ALLOWED_TOOLS = [
    "scan_timeline",
    "seek_evidence",
    "verify_hypothesis",
    "finalize_case",
]

FINALIZE_CASE_SCHEMA = {
    "type": "object",
    "properties": {
        "existence": {"type": "string", "enum": ["normal", "anomaly"]},
        "category": {"type": "string", "enum": list(CANONICAL_POLICY_CATEGORIES)},
        "severity": {"type": "integer"},
        "anomaly_interval_sec": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            ]
        },
        "precursor_interval_sec": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            ]
        },
        "earliest_actionable_sec": {"oneOf": [{"type": "null"}, {"type": "number"}]},
        "evidence_moment_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "covered_stages": {
            "type": "array",
            "items": {"type": "string", "enum": list(EVENT_CHAIN_STAGES)},
        },
        "missing_required_stages": {
            "type": "array",
            "items": {"type": "string", "enum": list(EVENT_CHAIN_STAGES)},
        },
        "stage_selected_moment_ids": build_stage_selected_moment_ids_schema(),
        "counterfactual_type": build_counterfactual_type_schema(),
        "summary": {"type": "string"},
        "rationale": {"type": "string"},
        "event_chain_summary": {
            "type": "object",
            "properties": {
                "precursor": {"type": "string"},
                "trigger": {"type": "string"},
                "confirmation": {"type": "string"},
            },
        },
        "qa_focus_answers": {
            "type": "object",
            "properties": {
                "existence": {"type": "string"},
                "category": {"type": "string"},
                "temporal": {"type": "string"},
            },
        },
    },
    "required": [
        "existence",
        "category",
        "severity",
        "anomaly_interval_sec",
        "precursor_interval_sec",
        "covered_stages",
        "stage_selected_moment_ids",
        "counterfactual_type",
    ],
}

FINALIZE_CASE_PRIMARY_FIELDS = (
    "existence",
    "category",
    "severity",
    "anomaly_interval_sec",
    "precursor_interval_sec",
    "earliest_actionable_sec",
    "evidence_moment_ids",
    "covered_stages",
    "missing_required_stages",
    "stage_selected_moment_ids",
    "counterfactual_type",
    "summary",
    "rationale",
    "event_chain_summary",
    "qa_focus_answers",
)


def build_finalize_case_payload(structured_target: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for field_name in FINALIZE_CASE_PRIMARY_FIELDS:
        if field_name in structured_target:
            payload[field_name] = copy.deepcopy(structured_target[field_name])
    return payload


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def round6(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)


def frame_to_second(
    frame: Optional[int],
    *,
    fps: float,
    frame_index_base: int,
    duration_sec: Optional[float] = None,
) -> Optional[float]:
    """Convert a frame index to the start time of that frame."""
    if frame is None:
        return None
    second = (float(frame) - float(frame_index_base)) / float(fps)
    if duration_sec is not None:
        second = clamp(second, 0.0, float(duration_sec))
    return round6(second)


def frame_interval_to_seconds(
    interval_frames: Optional[List[int]],
    *,
    fps: float,
    frame_index_base: int,
    duration_sec: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Convert an inclusive frame interval into a second interval.

    The end timestamp follows an inclusive-end policy:
    ``end_sec = (end_frame - base + 1) / fps``.
    This gives the right duration for inclusive frame annotations and behaves
    well with retrieval tools that operate on seconds.
    """
    if not interval_frames:
        return None, None
    start_frame, end_frame = int(interval_frames[0]), int(interval_frames[1])
    start_sec = frame_to_second(
        start_frame, fps=fps, frame_index_base=frame_index_base, duration_sec=duration_sec
    )
    end_sec = (float(end_frame) - float(frame_index_base) + 1.0) / float(fps)
    if duration_sec is not None:
        end_sec = clamp(end_sec, 0.0, float(duration_sec))
    return start_sec, round6(end_sec)


def ensure_frame_interval(interval_frames: Optional[List[int]]) -> Optional[List[int]]:
    if not interval_frames:
        return None
    start_frame, end_frame = int(interval_frames[0]), int(interval_frames[1])
    if start_frame > end_frame:
        start_frame, end_frame = end_frame, start_frame
    return [start_frame, end_frame]


def union_frame_intervals(intervals: Iterable[List[int]]) -> Optional[List[int]]:
    cleaned = [ensure_frame_interval(interval) for interval in intervals if interval]
    cleaned = [interval for interval in cleaned if interval]
    if not cleaned:
        return None
    start_frame = min(interval[0] for interval in cleaned)
    end_frame = max(interval[1] for interval in cleaned)
    return [start_frame, end_frame]


def _is_strict_precursor_interval(
    interval_frames: Optional[List[int]],
    *,
    anomaly_start_frame: Optional[int],
) -> bool:
    interval = ensure_frame_interval(interval_frames)
    if interval is None or anomaly_start_frame is None:
        return interval is not None
    # The second-domain interval uses an exclusive end timestamp:
    # end_sec = (end_frame - base + 1) / fps. Therefore a precursor that ends
    # on the same frame as anomaly_start already overlaps the anomaly in time.
    return int(interval[1]) < int(anomaly_start_frame)


def _sanitize_qa_pairs(
    qa_pairs: Any,
    *,
    precursor_interval_sec: Optional[List[float]],
    precursor_resolution_source: Optional[str],
) -> List[Dict[str, Any]]:
    cleaned_pairs: List[Dict[str, Any]] = []
    keep_precursor_temporal = (
        precursor_interval_sec is not None and str(precursor_resolution_source or "") == "annotation"
    )
    for qa in list(qa_pairs or []):
        if not isinstance(qa, dict):
            continue
        if str(qa.get("type") or "") == "precursor_temporal" and not keep_precursor_temporal:
            continue
        cleaned_pairs.append(dict(qa))
    return cleaned_pairs


def _normalize_evidence_role(
    role: Any,
    *,
    start_frame: int,
    end_frame: int,
    anomaly_start_frame: Optional[int],
) -> str:
    normalized_role = str(role or "unspecified")
    if normalized_role != "precursor" or anomaly_start_frame is None:
        return normalized_role
    if _is_strict_precursor_interval(
        [start_frame, end_frame],
        anomaly_start_frame=anomaly_start_frame,
    ):
        return normalized_role
    if int(start_frame) >= int(anomaly_start_frame):
        return "trigger"
    return "evidence"


def normalize_evidence_moment(
    moment: Dict[str, Any],
    *,
    fps: float,
    frame_index_base: int,
    duration_sec: float,
    anomaly_start_frame: Optional[int] = None,
) -> Dict[str, Any]:
    start_frame = int(moment["start_frame"])
    end_frame = int(moment["end_frame"])
    start_sec, end_sec = frame_interval_to_seconds(
        [start_frame, end_frame],
        fps=fps,
        frame_index_base=frame_index_base,
        duration_sec=duration_sec,
    )
    normalized_role = _normalize_evidence_role(
        moment.get("role"),
        start_frame=start_frame,
        end_frame=end_frame,
        anomaly_start_frame=anomaly_start_frame,
    )
    return {
        "moment_id": moment.get("moment_id", f"{normalized_role or 'ev'}_{start_frame}_{end_frame}"),
        "role": normalized_role,
        "description": moment.get("description"),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "start_sec": start_sec,
        "end_sec": end_sec,
    }


def complete_precursor_interval(record: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve or synthesize a precursor interval for anomaly videos."""
    temporal = record["temporal"]
    label = record["label"]
    fps = float(record["video_meta"]["fps"])
    duration_sec = float(record["video_meta"]["duration_sec"])
    frame_index_base = int(record.get("frame_index_base", 0))

    if not label.get("is_anomaly", False):
        return {
            "frames": None,
            "seconds": None,
            "source": "not_applicable_normal",
            "auto_completed": False,
        }

    annotated = ensure_frame_interval(temporal.get("precursor_interval_frames"))
    anomaly_interval = ensure_frame_interval(temporal.get("anomaly_interval_frames"))
    anomaly_start = anomaly_interval[0] if anomaly_interval else None
    saw_explicit_non_strict_precursor = False

    if annotated:
        if _is_strict_precursor_interval(annotated, anomaly_start_frame=anomaly_start):
            seconds = frame_interval_to_seconds(
                annotated,
                fps=fps,
                frame_index_base=frame_index_base,
                duration_sec=duration_sec,
            )
            return {
                "frames": annotated,
                "seconds": list(seconds),
                "source": "annotation",
                "auto_completed": False,
            }
        saw_explicit_non_strict_precursor = True

    evidence_moments = record.get("evidence", {}).get("evidence_moments", [])

    precursor_intervals = []
    preceding_intervals = []
    for moment in evidence_moments:
        interval = ensure_frame_interval([moment["start_frame"], moment["end_frame"]])
        if moment.get("role") == "precursor":
            if _is_strict_precursor_interval(interval, anomaly_start_frame=anomaly_start):
                precursor_intervals.append(interval)
            else:
                saw_explicit_non_strict_precursor = True
        if anomaly_start is not None and interval[1] < anomaly_start:
            preceding_intervals.append(interval)

    if precursor_intervals:
        resolved = union_frame_intervals(precursor_intervals)
        seconds = frame_interval_to_seconds(
            resolved,
            fps=fps,
            frame_index_base=frame_index_base,
            duration_sec=duration_sec,
        )
        return {
            "frames": resolved,
            "seconds": list(seconds),
            "source": "evidence_precursor_role",
            "auto_completed": False,
        }

    if saw_explicit_non_strict_precursor:
        return {
            "frames": None,
            "seconds": None,
            "source": "non_strict_precursor_dropped",
            "auto_completed": False,
        }

    if preceding_intervals:
        resolved = union_frame_intervals(preceding_intervals)
        seconds = frame_interval_to_seconds(
            resolved,
            fps=fps,
            frame_index_base=frame_index_base,
            duration_sec=duration_sec,
        )
        return {
            "frames": resolved,
            "seconds": list(seconds),
            "source": "evidence_preceding_event",
            "auto_completed": True,
        }

    if anomaly_start is None:
        return {
            "frames": None,
            "seconds": None,
            "source": "missing_without_anomaly_interval",
            "auto_completed": True,
        }

    anomaly_length = anomaly_interval[1] - anomaly_interval[0] + 1
    derived_window_frames = max(
        int(round(float(DEFAULT_DERIVED_PRECURSOR_SECONDS) * fps)),
        int(round(float(DEFAULT_DERIVED_PRECURSOR_FRACTION) * anomaly_length)),
    )
    end_frame = anomaly_start - 1
    start_frame = max(frame_index_base, end_frame - derived_window_frames + 1)
    if end_frame < start_frame:
        return {
            "frames": None,
            "seconds": None,
            "source": "derived_preceding_window_unavailable",
            "auto_completed": True,
        }

    resolved = [start_frame, end_frame]
    seconds = frame_interval_to_seconds(
        resolved,
        fps=fps,
        frame_index_base=frame_index_base,
        duration_sec=duration_sec,
    )
    return {
        "frames": resolved,
        "seconds": list(seconds),
        "source": "derived_preceding_window",
        "auto_completed": True,
    }


@dataclass
class ConverterConfig:
    pass


class CanonicalSaverAdapter:
    """Adapter for canonical SAVER benchmark records."""

    name = "msad_saver_qwen"

    def __init__(self, config: ConverterConfig):
        self.config = config

    def convert(self, record: Dict[str, Any], mode: str) -> Dict[str, Any]:
        base = self._build_base_view(record)
        if mode == "canonical_passthrough":
            return base
        if mode == "agent_train":
            return self._build_agent_train_view(base)
        if mode == "oracle_sft":
            agent_view = self._build_agent_train_view(base)
            agent_view["oracle_sft"] = self._build_oracle_sft(agent_view)
            return agent_view
        raise ValueError(f"Unsupported conversion mode: {mode}")

    def _build_base_view(self, record: Dict[str, Any]) -> Dict[str, Any]:
        fps = float(record["video_meta"]["fps"])
        duration_sec = float(record["video_meta"]["duration_sec"])
        frame_index_base = int(record.get("frame_index_base", 0))

        anomaly_interval_frames = ensure_frame_interval(record["temporal"].get("anomaly_interval_frames"))
        anomaly_start_frame = anomaly_interval_frames[0] if anomaly_interval_frames else None
        anomaly_interval_sec = (
            list(
                frame_interval_to_seconds(
                    anomaly_interval_frames,
                    fps=fps,
                    frame_index_base=frame_index_base,
                    duration_sec=duration_sec,
                )
            )
            if anomaly_interval_frames
            else None
        )

        precursor_info = complete_precursor_interval(record)
        earliest_actionable_frame = record["temporal"].get("earliest_alert_frame")
        earliest_actionable_sec = frame_to_second(
            earliest_actionable_frame,
            fps=fps,
            frame_index_base=frame_index_base,
            duration_sec=duration_sec,
        )

        evidence_moments = [
            normalize_evidence_moment(
                moment,
                fps=fps,
                frame_index_base=frame_index_base,
                duration_sec=duration_sec,
                anomaly_start_frame=anomaly_start_frame,
            )
            for moment in record.get("evidence", {}).get("evidence_moments", [])
        ]

        temporal = {
            key: value
            for key, value in dict(record["temporal"]).items()
            if key not in {"earliest_alert_frame", "earliest_alert_sec"}
        }

        base = {
            "schema_version": SCHEMA_VERSION,
            "record_origin": self.name,
            "video_id": record["video_id"],
            "file_name": record["file_name"],
            "video_path": record["video_path"],
            "source_dataset": record["source_dataset"],
            "source_split": record["source_split"],
            "split": record["split"],
            "frame_index_base": frame_index_base,
            "video_meta": {
                **record["video_meta"],
                "time_basis": {
                    "frame_index_base": frame_index_base,
                    "interval_end_policy": "inclusive_frame_end_converted_to_exclusive_second_end",
                },
            },
            "scene": record.get("scene", {}),
            "key_objects": record.get("key_objects", []),
            "label": {
                **record["label"],
                "category": canonicalize_saver_category(
                    record.get("label", {}).get("category"),
                    existence="anomaly" if record.get("label", {}).get("is_anomaly") else "normal",
                ),
            },
            "temporal": {
                **temporal,
                "anomaly_interval_sec": anomaly_interval_sec,
                "precursor_interval_frames": precursor_info["frames"],
                "precursor_interval_sec": precursor_info["seconds"],
                "earliest_actionable_frame": earliest_actionable_frame,
                "earliest_actionable_sec": earliest_actionable_sec,
                "precursor_resolution": {
                    "source": precursor_info["source"],
                    "auto_completed": precursor_info["auto_completed"],
                },
            },
            "evidence": {
                **record.get("evidence", {}),
                "evidence_moments": evidence_moments,
            },
            "counterfactual": record.get("counterfactual", {}),
            "language": record.get("language", {}),
            "qa_pairs": _sanitize_qa_pairs(
                record.get("qa_pairs", []),
                precursor_interval_sec=precursor_info["seconds"],
                precursor_resolution_source=precursor_info["source"],
            ),
            "provenance": record.get("provenance", {}),
            "qwen_preannotation": record.get("qwen_preannotation", {}),
            "auto_completed": {
                "precursor_interval": bool(precursor_info["auto_completed"]),
            },
        }
        base["proposal_supervision"] = build_proposal_supervision(
            key_objects=base.get("key_objects") or [],
            evidence_moments=evidence_moments,
            scene_context=str((base.get("scene") or {}).get("scenario") or ""),
        )
        return base

    @staticmethod
    def _synthetic_stage_moment(
        *,
        moment_id: str,
        role: str,
        interval_sec: Sequence[float],
        description: str,
    ) -> Dict[str, Any]:
        return {
            "moment_id": moment_id,
            "role": role,
            "description": description,
            "start_sec": round6(interval_sec[0]),
            "end_sec": round6(interval_sec[1]),
        }

    def _build_event_chain_stage_moments(
        self,
        base: Dict[str, Any],
        *,
        include_synthetic: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        label = base.get("label") or {}
        temporal = base.get("temporal") or {}
        language = base.get("language") or {}
        stage_to_moments: Dict[str, List[Dict[str, Any]]] = {stage: [] for stage in EVENT_CHAIN_STAGES}
        for moment in self._sorted_evidence_moments((base.get("evidence") or {}).get("evidence_moments") or []):
            stage = event_chain_stage_for_role(moment.get("role"))
            if stage is None:
                continue
            stage_to_moments[stage].append(moment)
        if not label.get("is_anomaly") or not include_synthetic:
            return {stage: list(stage_to_moments.get(stage) or []) for stage in EVENT_CHAIN_STAGES}

        precursor_interval = temporal.get("precursor_interval_sec")
        if precursor_interval and not stage_to_moments["precursor"]:
            stage_to_moments["precursor"].append(
                self._synthetic_stage_moment(
                    moment_id="oracle_precursor",
                    role="precursor",
                    interval_sec=precursor_interval,
                    description="lead-up activity before the decisive event",
                )
            )
        anomaly_interval = temporal.get("anomaly_interval_sec")
        if anomaly_interval and not stage_to_moments["trigger"]:
            summary = str(language.get("summary") or "decisive anomalous event").strip()
            stage_to_moments["trigger"].append(
                self._synthetic_stage_moment(
                    moment_id="oracle_trigger",
                    role="trigger",
                    interval_sec=anomaly_interval,
                    description=summary or "decisive anomalous event",
                )
            )
        return {stage: list(stage_to_moments.get(stage) or []) for stage in EVENT_CHAIN_STAGES}

    def _build_event_chain_target(self, base: Dict[str, Any]) -> Dict[str, Any]:
        label = base.get("label") or {}
        temporal = base.get("temporal") or {}
        stage_to_moments = self._build_event_chain_stage_moments(base, include_synthetic=True)
        stage_to_moment_ids = {
            stage: [
                str(moment.get("moment_id"))
                for moment in list(stage_to_moments.get(stage) or [])
                if str(moment.get("moment_id") or "").strip()
            ]
            for stage in EVENT_CHAIN_STAGES
        }
        required_stages: List[str] = []
        if label.get("is_anomaly"):
            if stage_to_moment_ids.get("precursor") or temporal.get("precursor_interval_sec"):
                required_stages.append("precursor")
            required_stages.append("trigger")
            if stage_to_moment_ids.get("confirmation"):
                required_stages.append("confirmation")
        required_stages = normalize_event_chain_stages(required_stages)
        available_stages = normalize_event_chain_stages(
            [stage for stage, moment_ids in stage_to_moment_ids.items() if moment_ids]
        )
        return {
            "stage_order": list(EVENT_CHAIN_STAGES),
            "required_stages": required_stages,
            "available_stages": available_stages,
            "stage_to_moment_ids": normalize_stage_selected_moment_ids(stage_to_moment_ids),
        }

    def _build_search_supervision(self, base: Dict[str, Any]) -> Dict[str, Any]:
        label = base.get("label") or {}
        scene_context = str((base.get("scene") or {}).get("scenario") or "").strip()
        category = canonicalize_saver_category(
            label.get("category"),
            existence="anomaly" if label.get("is_anomaly") else "normal",
        )
        stage_to_moments = self._build_event_chain_stage_moments(base, include_synthetic=True)
        stage_queries: Dict[str, Dict[str, Any]] = {}
        for stage in EVENT_CHAIN_STAGES:
            stage_moments = list(stage_to_moments.get(stage) or [])
            if not stage_moments:
                continue
            moment = dict(stage_moments[0])
            raw_description = str(moment.get("description") or "").strip()
            normalized_stage_query = self._moment_description_query(
                moment,
                role=str(moment.get("role") or stage),
                category=category,
                scene_context=scene_context,
            )
            if not normalized_stage_query:
                normalized_stage_query = compose_scene_anchored_query(
                    normalize_query_text(raw_description or stage),
                    scene_context,
                )
            stage_queries[stage] = {
                "moment_id": str(moment.get("moment_id") or ""),
                "role": str(moment.get("role") or stage),
                "raw_description": raw_description,
                "normalized_stage_query": normalized_stage_query,
                "window_sec": [
                    round6(moment.get("start_sec")),
                    round6(moment.get("end_sec")),
                ],
            }

        if not label.get("is_anomaly"):
            return {
                "stage_queries": stage_queries,
                "finalize_policy": {
                    "finalize_ready_after_stage": None,
                    "earliest_actionable_sec": None,
                    "minimal_sufficient_moment_ids": [],
                    "minimal_sufficient_stage_order": [],
                },
                "stage_supervision": {
                    "required_stage_order": [],
                    "finalize_ready_after_stage": None,
                    "default_next_stage": None,
                    "stage_query_source": "moment_description_primary",
                },
            }

        if stage_queries.get("confirmation"):
            finalize_ready_after_stage = "confirmation"
        elif stage_queries.get("trigger"):
            finalize_ready_after_stage = "trigger"
        elif stage_queries.get("precursor"):
            finalize_ready_after_stage = "precursor"
        else:
            finalize_ready_after_stage = None

        minimal_sufficient_stage_order: List[str] = []
        minimal_sufficient_moment_ids: List[str] = []
        earliest_actionable_sec = round6(
            (base.get("temporal") or {}).get("earliest_actionable_sec")
        )
        if finalize_ready_after_stage in EVENT_CHAIN_STAGES:
            ready_stage_index = EVENT_CHAIN_STAGES.index(finalize_ready_after_stage)
            for stage in EVENT_CHAIN_STAGES[: ready_stage_index + 1]:
                stage_query = stage_queries.get(stage)
                if not stage_query:
                    continue
                minimal_sufficient_stage_order.append(stage)
                moment_id = str(stage_query.get("moment_id") or "").strip()
                if moment_id:
                    minimal_sufficient_moment_ids.append(moment_id)
        if earliest_actionable_sec is None and stage_queries.get("trigger"):
            earliest_actionable_sec = round6((stage_queries.get("trigger") or {}).get("window_sec", [None])[0])
        return {
            "stage_queries": stage_queries,
            "finalize_policy": {
                "finalize_ready_after_stage": finalize_ready_after_stage,
                "earliest_actionable_sec": earliest_actionable_sec,
                "minimal_sufficient_moment_ids": minimal_sufficient_moment_ids,
                "minimal_sufficient_stage_order": minimal_sufficient_stage_order,
            },
            "stage_supervision": {
                "required_stage_order": minimal_sufficient_stage_order,
                "finalize_ready_after_stage": finalize_ready_after_stage,
                "default_next_stage": minimal_sufficient_stage_order[0] if minimal_sufficient_stage_order else None,
                "stage_query_source": "moment_description_primary",
            },
        }

    def _build_agent_train_view(self, base: Dict[str, Any]) -> Dict[str, Any]:
        label = base["label"]
        temporal = base["temporal"]
        evidence_moments = base["evidence"]["evidence_moments"]
        language = base.get("language") or {}
        existence = "anomaly" if label["is_anomaly"] else "normal"
        anomaly_interval_sec = temporal.get("anomaly_interval_sec")
        precursor_interval_sec = temporal.get("precursor_interval_sec")
        search_supervision = self._build_search_supervision(base)
        earliest_actionable_sec = (
            (search_supervision.get("finalize_policy") or {}).get("earliest_actionable_sec")
            if label["is_anomaly"]
            else None
        )
        if earliest_actionable_sec is None:
            earliest_actionable_sec = temporal.get("earliest_actionable_sec")
        event_chain_target = self._build_event_chain_target(base)
        stage_selected_moment_ids = normalize_stage_selected_moment_ids(
            event_chain_target.get("stage_to_moment_ids")
        )
        covered_stages = normalize_event_chain_stages(stage_selected_moment_ids.keys())
        missing_required_stages = [
            stage
            for stage in list(event_chain_target.get("required_stages") or [])
            if stage not in covered_stages
        ]

        task_prompt = self._build_task_prompt(base)
        structured_target = {
            "existence": existence,
            "category": canonicalize_saver_category(label["category"], existence=existence),
            "severity": label["severity"],
            "hard_normal": label["hard_normal"],
            "anomaly_interval_sec": anomaly_interval_sec,
            "precursor_interval_sec": precursor_interval_sec,
            "earliest_actionable_sec": earliest_actionable_sec,
            "evidence_moment_ids": [moment["moment_id"] for moment in evidence_moments],
            "event_chain_target": event_chain_target,
            "covered_stages": covered_stages,
            "missing_required_stages": missing_required_stages,
            "stage_selected_moment_ids": stage_selected_moment_ids,
            "evidence_windows_sec": [
                {
                    "moment_id": moment["moment_id"],
                    "role": moment["role"],
                    "window_sec": [moment["start_sec"], moment["end_sec"]],
                }
                for moment in evidence_moments
            ],
            "counterfactual_type": base["counterfactual"].get("type", "none"),
            "counterfactual_text": base["counterfactual"].get("text"),
            "summary": language.get("summary"),
            "rationale": language.get("rationale"),
        }
        semantic_payload = build_semantic_answer_payload(
            structured_target=structured_target,
            qa_pairs=base.get("qa_pairs") or [],
            evidence_moments=evidence_moments,
            finalized_case=structured_target,
        )
        structured_target["summary"] = semantic_payload.get("summary")
        structured_target["rationale"] = semantic_payload.get("rationale")
        structured_target["event_chain_summary"] = copy.deepcopy(
            semantic_payload.get("event_chain_summary") or {}
        )
        structured_target["qa_focus_answers"] = copy.deepcopy(
            semantic_payload.get("qa_focus_answers") or {}
        )
        tool_io = {
            "allowed_tools": ALLOWED_TOOLS,
            "initial_scan_window_frames": [
                base["frame_index_base"],
                int(base["video_meta"]["total_frames"]) + base["frame_index_base"] - 1,
            ],
            "initial_scan_window_sec": [0.0, round6(base["video_meta"]["duration_sec"])],
            "oracle_windows_frames": [
                {
                    "moment_id": moment["moment_id"],
                    "role": moment["role"],
                    "window": [moment["start_frame"], moment["end_frame"]],
                    "description": moment["description"],
                }
                for moment in evidence_moments
            ],
            "oracle_windows_sec": [
                {
                    "moment_id": moment["moment_id"],
                    "role": moment["role"],
                    "window": [moment["start_sec"], moment["end_sec"]],
                    "description": moment["description"],
                }
                for moment in evidence_moments
            ],
            "finalize_case_schema": FINALIZE_CASE_SCHEMA,
        }
        agent_task = {
            "task_type": "video_anomaly_search_verify_finalize",
            "query_mode": "stage_conditioned_retrieval",
            "task_prompt": task_prompt,
            "success_criteria": [
                "Discover whether an actionable anomaly exists under limited search.",
                "Use concrete stage queries that match visible evidence.",
                "Ground the decision in visited evidence only.",
                "Verify the current evidence subset before finalizing the case.",
            ],
        }
        base["agent_task"] = agent_task
        base["structured_target"] = structured_target
        base["tool_io"] = tool_io
        base["search_supervision"] = search_supervision
        base["stage_supervision"] = copy.deepcopy(search_supervision.get("stage_supervision") or {})
        return base

    def _build_task_prompt(self, base: Dict[str, Any]) -> str:
        scene = base.get("scene", {}).get("scenario")
        duration = round6(base["video_meta"]["duration_sec"])
        target_text = (
            "Determine whether the video contains an actionable anomaly or is normal, search for a complete event chain "
            "covering precursor, trigger, and confirmation evidence when relevant, use concrete stage-conditioned queries grounded in visible events, "
            "verify which stages are already covered by the searched evidence, and only finalize once the evidence is sufficient."
        )
        return (
            f"Video duration: {duration} seconds. "
            f"Scene: {scene}. "
            f"{target_text}"
        )

    @staticmethod
    def _select_proposal_query_group(
        moment: Dict[str, Any],
        *,
        proposal_supervision: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(proposal_supervision, dict):
            return None
        moment_id = str(moment.get("moment_id") or "")
        role = str(moment.get("role") or "")
        moment_tokens = {
            token
            for token in normalize_query_text(f"{moment.get('description') or ''} {role}").split()
            if token
        }
        best_group: Optional[Dict[str, Any]] = None
        best_score = -1.0
        for query_group in proposal_supervision.get("queries") or []:
            query_origin = str(query_group.get("query_origin") or "").strip().lower()
            linked_moment_ids = {str(value) for value in query_group.get("linked_moment_ids") or [] if value}
            linked_roles = {str(value) for value in query_group.get("linked_roles") or [] if value}
            if moment_id and linked_moment_ids and moment_id not in linked_moment_ids and role not in linked_roles:
                continue
            score = 0.0
            if moment_id and moment_id in linked_moment_ids:
                score += 4.0
            if role and role in linked_roles:
                score += 2.0
            if query_origin == "moment_description_primary":
                score += 3.0
            best_entry_score = 0.0
            for entry in query_group.get("normalized_queries") or []:
                text = str(entry.get("text") or "").strip()
                if not text:
                    continue
                query_tokens = {
                    token
                    for token in normalize_query_text(text).split()
                    if token
                }
                overlap = len(query_tokens & moment_tokens) / float(max(len(query_tokens), 1)) if query_tokens else 0.0
                best_entry_score = max(best_entry_score, float(entry.get("weight") or 0.0) + overlap)
            score += best_entry_score
            if score > best_score:
                best_score = score
                best_group = dict(query_group)
        return best_group

    @staticmethod
    def _generic_hypothesis_for_role(role: str) -> str:
        if role == "precursor":
            return "suspected pre-anomaly cue"
        if role in {"trigger", "peak_action"}:
            return "suspected actionable event"
        if role == "confirmation":
            return "suspected anomaly confirmation"
        if role == "normal_check":
            return "check whether this interval remains normal"
        if role == "context":
            return "suspected anomaly context"
        return "suspected anomaly evidence"

    def _build_generic_query_package(
        self,
        base: Dict[str, Any],
        *,
        role: str,
        query_text: str,
        query_source: str,
        key_objects: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        scene_context = str((base.get("scene") or {}).get("scenario") or "").strip()
        return normalize_query_package(
            {
                "event_cue": str(query_text or "").strip(),
                "key_objects": list(key_objects or []),
                "scene_context": scene_context,
                "hypothesis": self._generic_hypothesis_for_role(role),
                "negative_constraints": [],
                "rewrite_reason": str(query_source or "role_fallback"),
            },
            fallback_query=str(query_text or "").strip(),
            fallback_scene_context=scene_context,
            rewrite_reason=str(query_source or "role_fallback"),
        )

    def _build_query_package_for_moment(
        self,
        base: Dict[str, Any],
        *,
        moment: Dict[str, Any],
        query_text: str,
        query_source: str,
        proposal_supervision: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        role = str(moment.get("role") or "")
        query_group = self._select_proposal_query_group(moment, proposal_supervision=proposal_supervision)
        normalized_entries = list((query_group or {}).get("normalized_queries") or [])
        query_origin = str((query_group or {}).get("query_origin") or "").strip().lower()
        event_candidates = [
            str(entry.get("text") or "").strip()
            for entry in normalized_entries
            if str(entry.get("kind") or "").strip() == "event_relation" and str(entry.get("text") or "").strip()
        ]
        key_object_candidates = [
            str(entry.get("text") or "").strip()
            for entry in normalized_entries
            if str(entry.get("kind") or "").strip() in {"object", "attribute_object"}
            and str(entry.get("text") or "").strip()
        ]
        if query_origin == "moment_description_primary":
            scene_context = str((base.get("scene") or {}).get("scenario") or "").strip()
            moment_description = normalize_query_text(
                str((query_group or {}).get("raw_text") or moment.get("description") or "")
            )
            return normalize_query_package(
                {
                    "event_cue": moment_description,
                    "key_objects": key_object_candidates[:4],
                    "scene_context": scene_context,
                    "hypothesis": moment_description,
                    "negative_constraints": [],
                    "rewrite_reason": str(query_source or "oracle_moment_description_primary"),
                },
                fallback_query=moment_description or str(query_text or "").strip(),
                fallback_scene_context=scene_context,
                rewrite_reason=str(query_source or "oracle_moment_description_primary"),
            )
        event_cue = event_candidates[0] if event_candidates else str(query_text or "").strip()
        if not key_object_candidates and query_text and query_text != event_cue:
            key_object_candidates = [str(query_text).strip()]
        return self._build_generic_query_package(
            base,
            role=role,
            query_text=event_cue or str(query_text or "").strip(),
            query_source=query_source,
            key_objects=key_object_candidates[:4],
        )

    @staticmethod
    def _sanitize_query_phrase(text: Any, *, category: str) -> str:
        normalized = normalize_query_text(str(text or ""))
        normalized = " ".join(
            token
            for token in normalized.split()
            if token not in {"anomaly", "abnormal", "event"}
        )
        return normalized.strip()

    def _moment_description_query(
        self,
        moment: Dict[str, Any],
        *,
        role: str,
        category: str,
        scene_context: str = "",
    ) -> str:
        description = self._sanitize_query_phrase(moment.get("description"), category=category)
        if len(description.split()) < 2:
            return ""
        return compose_scene_anchored_query(description, scene_context)

    def _build_oracle_sft(self, base: Dict[str, Any]) -> Dict[str, Any]:
        trajectory: List[Dict[str, Any]] = []
        language = base.get("language") or {}
        structured_target = base.get("structured_target") or {}
        duration = round6(base["video_meta"]["duration_sec"])
        next_window_id = 1
        next_evidence_id = 1
        searched_real_moments: List[Dict[str, Any]] = []
        searched_real_refs: List[Dict[str, Any]] = []
        searched_supplemental_refs: List[Dict[str, Any]] = []
        normal_search_refs: List[Dict[str, Any]] = []

        def append_step(
            step: Dict[str, Any],
            *,
            ref_bucket: Optional[List[Dict[str, Any]]] = None,
        ) -> Optional[Dict[str, Any]]:
            nonlocal next_window_id, next_evidence_id
            trajectory.append(step)
            tool_name = str(step.get("tool") or "")
            if tool_name not in {"scan_timeline", "seek_evidence"}:
                return None
            runtime_ref = {
                "window_id": f"w{next_window_id:04d}",
                "evidence_id": f"e{next_evidence_id:04d}",
                "tool": tool_name,
                "moment_id": (step.get("arguments") or {}).get("moment_id"),
                "role": (step.get("arguments") or {}).get("role"),
            }
            next_window_id += 1
            next_evidence_id += 1
            if ref_bucket is not None:
                ref_bucket.append(runtime_ref)
            return runtime_ref

        append_step(
            {
                "tool": "scan_timeline",
                "arguments": {
                    "start_sec": 0.0,
                    "end_sec": duration,
                    "stride_sec": max(round6(duration / 8.0) or 0.5, 0.5),
                    "purpose": "global_overview",
                },
            },
            ref_bucket=normal_search_refs,
        )

        evidence_moments = base["evidence"]["evidence_moments"]
        label = base["label"]
        temporal = base["temporal"]

        if label["is_anomaly"]:
            category = canonicalize_saver_category(label.get("category") or "anomaly", existence="anomaly")
            proposal_supervision = base.get("proposal_supervision")
            event_chain_target = structured_target.get("event_chain_target") or self._build_event_chain_target(base)
            search_supervision = base.get("search_supervision") or self._build_search_supervision(base)
            finalize_policy = dict(search_supervision.get("finalize_policy") or {})
            oracle_required_stages = normalize_event_chain_stages(
                finalize_policy.get("minimal_sufficient_stage_order")
                or event_chain_target.get("required_stages")
            )
            stage_to_moments = self._build_event_chain_stage_moments(base, include_synthetic=True)
            earliest_actionable_sec = structured_target.get("earliest_actionable_sec")

            def current_stage_selected_moment_ids() -> Dict[str, List[str]]:
                grouped: Dict[str, List[str]] = {}
                for stage in EVENT_CHAIN_STAGES:
                    moment_ids = []
                    seen = set()
                    for moment in searched_real_moments:
                        if event_chain_stage_for_role(moment.get("role")) != stage:
                            continue
                        moment_id = str(moment.get("moment_id") or "").strip()
                        if not moment_id or moment_id in seen:
                            continue
                        seen.add(moment_id)
                        moment_ids.append(moment_id)
                    if moment_ids:
                        grouped[stage] = moment_ids
                return grouped

            def current_covered_stages() -> List[str]:
                return normalize_event_chain_stages(current_stage_selected_moment_ids().keys())

            def current_missing_required_stages() -> List[str]:
                covered = set(current_covered_stages())
                return [stage for stage in oracle_required_stages if stage not in covered]

            def append_real_seek(moment: Dict[str, Any]) -> None:
                runtime_ref = append_step(
                    self._seek_evidence_step(
                        base,
                        moment,
                        category=category,
                        proposal_supervision=proposal_supervision,
                    ),
                    ref_bucket=searched_real_refs,
                )
                searched_real_moments.append(moment)
                if runtime_ref is not None:
                    runtime_ref["moment_id"] = moment.get("moment_id")
                    runtime_ref["role"] = moment.get("role")

            def append_stage_seek(stage: str) -> Optional[Dict[str, Any]]:
                already_selected = {
                    str(moment.get("moment_id") or "").strip()
                    for moment in searched_real_moments
                    if str(moment.get("moment_id") or "").strip()
                }
                for moment in list(stage_to_moments.get(stage) or []):
                    moment_id = str(moment.get("moment_id") or "").strip()
                    if moment_id and moment_id in already_selected:
                        continue
                    append_real_seek(moment)
                    return moment
                return None

            for stage_index, stage in enumerate(oracle_required_stages):
                if stage not in current_covered_stages():
                    append_stage_seek(stage)

                stage_map = current_stage_selected_moment_ids()
                selected_evidence_moment_ids = [
                    moment_id
                    for iter_stage in EVENT_CHAIN_STAGES
                    for moment_id in list(stage_map.get(iter_stage) or [])
                ]
                covered_stages = current_covered_stages()
                missing_required_stages = current_missing_required_stages()
                event_chain_complete = not missing_required_stages
                progress_ratio = (
                    len(covered_stages) / float(len(oracle_required_stages))
                    if oracle_required_stages
                    else 0.0
                )
                current_finalize_readiness = max(
                    0.08,
                    min(
                        0.92,
                        0.18
                        + 0.34 * progress_ratio
                        + (0.18 if "trigger" in covered_stages else 0.0)
                        + (0.12 if event_chain_complete else 0.0),
                    ),
                )
                trajectory.append(
                    {
                        "tool": "verify_hypothesis",
                        "arguments": {
                            "verification_mode": "final_check" if event_chain_complete else "full_keep_drop",
                            "selected_evidence_moment_ids": selected_evidence_moment_ids,
                            "covered_stages": covered_stages,
                            "missing_required_stages": missing_required_stages,
                            "stage_selected_moment_ids": stage_map,
                            "claim": {
                                "existence": "anomaly",
                                "category": category,
                                "earliest_actionable_sec": earliest_actionable_sec,
                            },
                        },
                        "oracle_verifier_feedback": self._oracle_verifier_feedback(
                            verification_mode="final_check" if event_chain_complete else "full_keep_drop",
                            verification_decision="sufficient" if event_chain_complete else "insufficient",
                            recommended_action="finalize" if event_chain_complete else "continue_search",
                            selected_refs=searched_real_refs,
                            selected_evidence_moment_ids=selected_evidence_moment_ids,
                            covered_stages=covered_stages,
                            missing_required_stages=missing_required_stages,
                            stage_selected_moment_ids=stage_map,
                            sufficiency_score=0.28 + 0.62 * progress_ratio if not event_chain_complete else 0.92,
                            necessity_score=0.24 + 0.48 * progress_ratio if not event_chain_complete else 0.76,
                            finalize_readiness_score=current_finalize_readiness,
                            counterfactual_faithfulness=0.32 + 0.44 * progress_ratio if not event_chain_complete else 0.84,
                            rationale=(
                                f"The current evidence covers {', '.join(covered_stages) or 'no required stage'} and still misses "
                                f"{', '.join(missing_required_stages)}, so search should continue before finalization."
                                if not event_chain_complete
                                else "The searched evidence now covers the full required event chain, so the case can be finalized."
                            ),
                        ),
                    }
                )
        else:
            for step in self._normal_followup_scan_steps(base, count=1):
                append_step(step, ref_bucket=normal_search_refs)
            trajectory.append(
                {
                    "tool": "verify_hypothesis",
                    "arguments": {
                        "verification_mode": "final_check",
                        "selected_window_ids": [
                            str(ref.get("window_id"))
                            for ref in list(normal_search_refs or [])
                            if str(ref.get("window_id") or "").strip()
                        ],
                        "claim": {
                            "existence": "normal",
                            "category": canonicalize_saver_category(label.get("category"), existence="normal"),
                        },
                    },
                    "oracle_verifier_feedback": self._oracle_verifier_feedback(
                        verification_mode="final_check",
                        verification_decision="sufficient",
                        recommended_action="finalize",
                        selected_refs=normal_search_refs,
                        selected_evidence_moment_ids=[],
                        sufficiency_score=0.9,
                        necessity_score=0.58,
                        finalize_readiness_score=0.0,
                        counterfactual_faithfulness=0.74,
                        rationale="The searched windows are enough to justify a normal decision, so the case can be finalized.",
                    ),
                }
            )

        trajectory.append(
            {
                "tool": "finalize_case",
                "arguments": build_finalize_case_payload(base["structured_target"]),
            }
        )
        for step in trajectory:
            if str(step.get("tool") or "") != "verify_hypothesis":
                continue
            feedback = step.get("oracle_verifier_feedback") or {}
            if not isinstance(feedback, dict):
                continue
            merged_arguments = dict(step.get("arguments") or {})
            merged_arguments.update(feedback)
            step["arguments"] = build_policy_self_verification_payload(merged_arguments)
        return {
            "trajectory": trajectory,
            "final_decision": build_finalize_case_payload(base["structured_target"]),
        }

    @staticmethod
    def _window_key(start_sec: Any, end_sec: Any) -> Tuple[Optional[float], Optional[float]]:
        return round6(start_sec), round6(end_sec)

    @staticmethod
    def _sorted_evidence_moments(evidence_moments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        role_priority = {
            "precursor": 0,
            "trigger": 1,
            "peak_action": 2,
            "peak": 2,
            "confirmation": 3,
            "aftermath": 4,
        }
        return sorted(
            evidence_moments,
            key=lambda moment: (
                role_priority.get(str(moment.get("role") or "").lower(), 5),
                float(moment.get("start_sec") or 0.0),
                float(moment.get("end_sec") or 0.0),
                str(moment.get("moment_id") or ""),
            ),
        )

    def _seek_evidence_step(
        self,
        base: Dict[str, Any],
        moment: Dict[str, Any],
        *,
        category: str,
        proposal_supervision: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        role = str(moment.get("role") or "")
        fallback_query = self._query_for_role(role, category)
        query_text, query_source = select_query_for_moment(
            moment=moment,
            proposal_supervision=proposal_supervision,
            fallback_query=fallback_query,
        )
        description_query = self._moment_description_query(
            moment,
            role=role,
            category=category,
            scene_context=str((base.get("scene") or {}).get("scenario") or ""),
        )
        normalized_query = normalize_query_text(query_text)
        if description_query and (
            query_source != "object_fallback"
            or len(normalized_query.split()) <= 1
            or normalized_query in {"person", "people", "man", "woman", "bag", "car", "vehicle", "worker", "object"}
        ):
            query_text = description_query
            if query_source == "description_primary":
                query_source = "description_primary"
            else:
                query_source = "description_fallback"
        return {
            "tool": "seek_evidence",
            "arguments": {
                "query": query_text,
                "start_sec": moment["start_sec"],
                "end_sec": moment["end_sec"],
                "moment_id": moment["moment_id"],
                "role": role,
                "query_source": query_source,
            },
        }

    def _supplemental_seek_steps(
        self,
        base: Dict[str, Any],
        *,
        category: str,
        used_window_keys: set[Tuple[Optional[float], Optional[float]]],
        count: int,
    ) -> List[Dict[str, Any]]:
        if count <= 0:
            return []
        duration = round6(base["video_meta"]["duration_sec"]) or 0.0
        temporal = base.get("temporal") or {}
        anomaly_interval = temporal.get("anomaly_interval_sec") or [0.0, duration]
        precursor_interval = temporal.get("precursor_interval_sec")
        anomaly_start = float((anomaly_interval or [0.0, duration])[0] or 0.0)
        anomaly_end = float((anomaly_interval or [0.0, duration])[1] or duration)
        if anomaly_end <= anomaly_start:
            anomaly_end = max(anomaly_start + 0.5, duration)
        anomaly_span = max(anomaly_end - anomaly_start, 0.5)

        candidate_specs: List[Dict[str, Any]] = []
        if precursor_interval:
            candidate_specs.append(
                {
                    "query": self._query_for_role("precursor", category),
                    "start_sec": precursor_interval[0],
                    "end_sec": precursor_interval[1],
                    "role": "precursor",
                    "query_source": "oracle_role_fallback",
                }
            )
        candidate_specs.extend(
            [
                {
                    "query": self._query_for_role("peak_action", category),
                    "start_sec": anomaly_start + 0.15 * anomaly_span,
                    "end_sec": anomaly_start + 0.55 * anomaly_span,
                    "role": "peak_action",
                    "query_source": "oracle_role_fallback",
                },
                {
                    "query": self._query_for_role("confirmation", category),
                    "start_sec": max(anomaly_start, anomaly_end - max(0.35 * anomaly_span, 0.5)),
                    "end_sec": anomaly_end,
                    "role": "confirmation",
                    "query_source": "oracle_role_fallback",
                },
                {
                    "query": "look for broader temporal context around the suspected anomaly",
                    "start_sec": max(0.0, anomaly_start - max(0.15 * anomaly_span, 0.5)),
                    "end_sec": min(duration, anomaly_end + max(0.15 * anomaly_span, 0.5)),
                    "role": "context",
                    "query_source": "oracle_context_broad",
                },
                {
                    "query": "look for the full temporal context of the suspected anomaly",
                    "start_sec": anomaly_start,
                    "end_sec": anomaly_end,
                    "role": "context",
                    "query_source": "oracle_context_full",
                },
            ]
        )

        supplemental_steps: List[Dict[str, Any]] = []
        for spec in candidate_specs:
            start_sec = round6(spec.get("start_sec"))
            end_sec = round6(spec.get("end_sec"))
            if start_sec is None or end_sec is None:
                continue
            if end_sec <= start_sec:
                end_sec = round6(min(duration, float(start_sec) + 0.5))
            window_key = self._window_key(start_sec, end_sec)
            if window_key in used_window_keys:
                continue
            used_window_keys.add(window_key)
            supplemental_steps.append(
                {
                    "tool": "seek_evidence",
                    "arguments": {
                        "query": spec["query"],
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "role": spec.get("role"),
                        "query_source": spec.get("query_source"),
                    },
                }
            )
            if len(supplemental_steps) >= count:
                break
        return supplemental_steps

    def _normal_followup_scan_steps(self, base: Dict[str, Any], *, count: int = 3) -> List[Dict[str, Any]]:
        duration = round6(base["video_meta"]["duration_sec"]) or 0.0
        if duration <= 0.0 or count <= 0:
            return []
        boundaries = [round6(duration * float(idx) / float(count)) for idx in range(count + 1)]
        steps: List[Dict[str, Any]] = []
        for idx in range(count):
            start_sec = boundaries[idx]
            end_sec = boundaries[idx + 1]
            if end_sec is None or start_sec is None:
                continue
            if idx == count - 1:
                end_sec = duration
            if end_sec <= start_sec:
                end_sec = round6(min(duration, float(start_sec) + max(duration / max(count, 1), 0.5)))
            if end_sec <= start_sec:
                continue
            steps.append(
                {
                    "tool": "seek_evidence",
                    "arguments": {
                        "query": f"check whether segment {idx + 1} contains any actionable anomaly evidence or supports a normal conclusion",
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "role": "normal_check",
                        "query_source": "oracle_normal_search",
                    },
                }
            )
        return steps

    @staticmethod
    def _oracle_verifier_feedback(
        *,
        verification_mode: str,
        verification_decision: str,
        recommended_action: str,
        selected_refs: Optional[List[Dict[str, Any]]] = None,
        selected_evidence_moment_ids: Optional[List[str]] = None,
        covered_stages: Optional[List[str]] = None,
        missing_required_stages: Optional[List[str]] = None,
        stage_selected_moment_ids: Optional[Dict[str, List[str]]] = None,
        sufficiency_score: float,
        necessity_score: float,
        finalize_readiness_score: float,
        counterfactual_faithfulness: float,
        rationale: str,
        teacher_judge_scores: Optional[Dict[str, Any]] = None,
        teacher_judge_decision: Optional[str] = None,
        teacher_judge_rationale: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = build_policy_self_verification_payload(
            {
                "verification_mode": verification_mode,
                "verification_decision": verification_decision,
                "recommended_action": recommended_action,
                "selected_window_ids": [
                    str(ref.get("window_id"))
                    for ref in list(selected_refs or [])
                    if ref.get("window_id") is not None
                ],
                "selected_evidence_moment_ids": [
                    str(moment_id)
                    for moment_id in list(selected_evidence_moment_ids or [])
                    if str(moment_id).strip()
                ],
                "covered_stages": normalize_event_chain_stages(covered_stages),
                "missing_required_stages": normalize_event_chain_stages(missing_required_stages),
                "stage_selected_moment_ids": normalize_stage_selected_moment_ids(stage_selected_moment_ids),
                "sufficiency_score": round6(sufficiency_score),
                "necessity_score": round6(necessity_score),
                "finalize_readiness_score": round6(finalize_readiness_score),
                "counterfactual_faithfulness": round6(counterfactual_faithfulness),
                "rationale": rationale,
            }
        )
        if teacher_judge_scores is not None:
            payload["teacher_judge_scores"] = dict(teacher_judge_scores)
        if teacher_judge_decision is not None:
            payload["teacher_judge_decision"] = str(teacher_judge_decision)
        if teacher_judge_rationale is not None:
            payload["teacher_judge_rationale"] = str(teacher_judge_rationale)
        return payload

    @staticmethod
    def _query_for_role(role: str, category: str) -> str:
        if role == "precursor":
            return "look for lead-up risk cues, suspicious preparation, or approach behavior before the decisive event"
        if role == "trigger":
            return "look for the first decisive interaction, collision, ignition, fall, or other clearly actionable event"
        if role == "peak_action":
            return "look for the strongest anomalous interaction or risky motion"
        if role == "confirmation":
            return "look for aftermath or persistent evidence confirming the event really happened"
        return "look for event-level evidence relevant to the suspected anomaly"


ADAPTERS = {
    "msad_saver_qwen": CanonicalSaverAdapter,
    "canonical_saver_v1": CanonicalSaverAdapter,
}


def convert_record(
    record: Dict[str, Any],
    *,
    mode: str,
    adapter_name: str = "msad_saver_qwen",
) -> Dict[str, Any]:
    adapter_cls = ADAPTERS[adapter_name]
    adapter = adapter_cls(ConverterConfig())
    return adapter.convert(record, mode)


def iter_jsonl(
    path: Path,
    *,
    include_splits: Optional[str | List[str]] = None,
    skip_invalid_lines: bool = False,
) -> Iterable[Dict[str, Any]]:
    allowed_splits = set(parse_include_splits(include_splits) or [])
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                preview = line.replace("\t", " ")
                if len(preview) > 240:
                    preview = preview[:240] + "..."
                message = f"Invalid JSONL at {path}:{line_number}: {exc}. Line preview: {preview}"
                if not skip_invalid_lines:
                    raise ValueError(message) from exc
                print(json.dumps({"warning": "skipped_invalid_jsonl_line", "message": message}, ensure_ascii=False))
                continue
            if allowed_splits and str(record.get("split") or "").strip() not in allowed_splits:
                continue
            yield record


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input canonical JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument(
        "--mode",
        default="agent_train",
        choices=["canonical_passthrough", "agent_train", "oracle_sft"],
        help="Derived view to generate.",
    )
    parser.add_argument(
        "--adapter",
        default="msad_saver_qwen",
        choices=sorted(ADAPTERS.keys()),
        help="Input adapter.",
    )
    parser.add_argument(
        "--include-splits",
        default="",
        help="Optional comma-separated split whitelist, e.g. train or train,val.",
    )
    parser.add_argument(
        "--skip-invalid-jsonl-lines",
        action="store_true",
        help="Skip malformed JSONL lines instead of failing immediately. Prefer regenerating the source file when possible.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    input_records = iter_jsonl(
        input_path,
        include_splits=args.include_splits,
        skip_invalid_lines=args.skip_invalid_jsonl_lines,
    )
    converted_rows = (
        convert_record(
            record,
            mode=args.mode,
            adapter_name=args.adapter,
        )
        for record in input_records
    )
    write_jsonl(output_path, converted_rows)
    print("finished\n--------------------------------\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
