from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from saver_v3.cli.common import load_yaml_mapping, write_json
from saver_v3.common.runtime import (
    distributed_barrier,
    distributed_runtime_from_env,
    init_torch_distributed,
    runtime_log,
    shard_sequence,
    should_log_progress,
)
from saver_v3.core.categories import canonicalize_category_payload
from saver_v3.core.protocol_guidance import event_chain_stage_for_role, normalize_event_chain_stages
from saver_v3.core.semantic_answer import normalize_semantic_answer_payload
from saver_v3.data.config import InitialObservationConfig, PreviewConfig, SaverAgentConfig
from saver_v3.data.dataset import SaverAgentDataset
from saver_v3.inference.message_runtime import MessageSamplingConfig, TorchQwen3MessageRuntime
from saver_v3.metrics.legacy_metrics import summarize_saver_metrics
from saver_v3.metrics.offline_scoring import ReferenceDataProvider, save_rollout_records
from saver_v3.metrics.semantic_metrics import evaluate_semantic_rollouts


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return value or {}


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(value)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _system_prompt() -> str:
    return (
        "You are a fixed-observation video anomaly understanding baseline. "
        "You do not have tools, search, verification, or multi-turn interaction. "
        "Use only provided preview frames. Return exactly one JSON object and nothing else."
    )


def _response_scaffold() -> dict[str, Any]:
    return {
        "decision": {
            "existence": "anomaly",
            "category": "assault",
            "anomaly_interval_sec": [0.0, 1.0],
            "precursor_interval_sec": [0.0, 0.5],
        },
        "semantic_answer": {
            "summary": "one concise case summary",
            "rationale": "brief evidence-grounded rationale",
            "event_chain_summary": {
                "precursor": "optional precursor evidence summary",
                "trigger": "optional trigger evidence summary",
                "confirmation": "optional confirmation or aftermath summary",
            },
            "qa_focus_answers": {
                "existence": "direct answer to whether an anomaly exists",
                "category": "direct answer to which anomaly category applies",
                "temporal": "direct answer to when the anomaly occurs",
            },
        },
        "evidence_topk": [
            {
                "rank": 1,
                "start_sec": 0.0,
                "end_sec": 0.5,
                "role": "precursor",
                "description": "short visible evidence description",
            }
        ],
    }


def build_fixed_baseline_messages(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    record = dict(item or {})
    cache = dict(record.get("multimodal_cache") or {})
    video_meta = dict(record.get("video_meta") or cache.get("video_meta") or {})
    duration_sec = _safe_float(video_meta.get("duration_sec"), _safe_float(cache.get("duration"), 0.0))
    scene = str(((record.get("scene") or {}).get("scenario")) or "unknown").strip() or "unknown"
    source_dataset = str(record.get("source_dataset") or "").strip()
    prompt_lines = [
        f"Task: inspect fixed preview frames from a surveillance video of duration {duration_sec:.3f}s.",
        f"Scene: {scene}.",
    ]
    if source_dataset:
        prompt_lines.append(f"Dataset: {source_dataset}.")
    prompt_lines.extend(
        [
            "Decide whether the video is normal or contains an actionable anomaly.",
            "If anomalous, provide anomaly category, temporal interval, optional precursor interval, and up to 3 evidence windows.",
            "Use seconds, not frame indices.",
            "Roles must be precursor, trigger, or confirmation.",
            "For normal videos, set existence=normal, category=normal, intervals=null, evidence_topk=[].",
            "Return strict JSON only with this schema:",
            json.dumps(_response_scaffold(), ensure_ascii=False),
        ]
    )
    user_prompt = "\n".join(prompt_lines)

    preview_frames = cache.get("preview_frames")
    preview_timestamps = list(cache.get("preview_timestamps") or [])
    preview_frame_indices = list(cache.get("preview_frame_indices") or [])
    frame_indices = list(cache.get("frame_indices") or [])
    user_content: List[Dict[str, Any]] = []
    if preview_frames is not None and len(preview_timestamps) > 0:
        for timestamp, frame, sampled_frame_index in zip(preview_timestamps, preview_frames, preview_frame_indices):
            user_content.append({"type": "text", "text": f"{float(timestamp):.3f}s"})
            image_item = {
                "type": "image",
                "image": frame,
                "sampled_frame_index": int(sampled_frame_index),
                "timestamp_sec": float(timestamp),
            }
            if 0 <= int(sampled_frame_index) < len(frame_indices):
                image_item["raw_frame_index"] = int(frame_indices[int(sampled_frame_index)])
            user_content.append(image_item)
    user_content.append({"type": "text", "text": user_prompt})
    return [
        {"role": "system", "content": [{"type": "text", "text": _system_prompt()}]},
        {"role": "user", "content": user_content},
    ]


def parse_fixed_baseline_response_text(response_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    text = str(response_text or "").strip()
    if not text:
        return None, "empty_response"
    if "```" in text:
        return None, "fenced_output"
    if not (text.startswith("{") and text.endswith("}")):
        return None, "not_pure_json_object"
    try:
        payload = json.loads(text)
    except Exception as exc:
        return None, f"json_decode_error:{exc}"
    if not isinstance(payload, dict):
        return None, "json_not_object"
    return payload, None


def _normalize_existence(value: Any) -> str:
    if isinstance(value, bool):
        return "anomaly" if value else "normal"
    text = _clean_text(value).lower()
    if not text:
        # Empty input is unresolvable — do NOT bias toward "normal".
        return ""
    if text in {"normal", "none", "no", "false", "0"}:
        return "normal"
    if text in {"anomaly", "yes", "true", "1"}:
        return "anomaly"
    if "no anomaly" in text or text.startswith("normal"):
        return "normal"
    return "anomaly"


def _normalize_interval_sec(value: Any, *, duration_sec: float) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, Mapping):
        raw = value.get("interval") or value.get("window") or value.get("window_sec")
    else:
        raw = value
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    start_sec = _safe_float(raw[0], math.nan)
    end_sec = _safe_float(raw[1], math.nan)
    if math.isnan(start_sec) or math.isnan(end_sec):
        return None
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec
    start_sec = max(0.0, start_sec)
    end_sec = max(0.0, end_sec)
    if duration_sec > 0.0:
        start_sec = min(start_sec, duration_sec)
        end_sec = min(end_sec, duration_sec)
    if end_sec < start_sec:
        return None
    return [round(float(start_sec), 6), round(float(end_sec), 6)]


def _normalize_role(value: Any) -> str:
    stage = event_chain_stage_for_role(value)
    return stage or ""


def _fallback_semantic_answer(
    decision: Dict[str, Any],
    evidence_topk: Sequence[Dict[str, Any]],
    raw_semantic: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    raw_semantic = dict(raw_semantic or {})
    # Historical bias-to-normal fallback removed: treat an absent existence
    # label as "unknown" and let downstream text branches handle it, rather
    # than silently labelling every un-decided rollout as normal.
    existence_raw = decision.get("existence")
    existence = _normalize_existence(existence_raw) if existence_raw is not None else ""
    category = str(decision.get("category") or "").strip()
    anomaly_interval = list(decision.get("anomaly_interval_sec") or [])
    temporal_answer = (
        f"The anomaly occurs from {float(anomaly_interval[0]):.3f}s to {float(anomaly_interval[1]):.3f}s."
        if len(anomaly_interval) == 2
        else ("There is no anomaly interval in this video." if existence == "normal" else "The anomaly timing is uncertain.")
    )
    by_stage: Dict[str, str] = {"precursor": "", "trigger": "", "confirmation": ""}
    for entry in evidence_topk:
        stage = str(entry.get("role") or "").strip().lower()
        if stage not in by_stage or by_stage[stage]:
            continue
        by_stage[stage] = _clean_text(entry.get("description"))
    semantic_candidate = {
        "decision": dict(decision),
        "summary": _clean_text(raw_semantic.get("summary")) or (
            "The video appears normal." if existence == "normal" else f"The video shows {category or 'anomalous activity'}."
        ),
        "rationale": _clean_text(raw_semantic.get("rationale")) or (
            "No anomaly is visible in the provided preview frames."
            if existence == "normal"
            else "The provided preview frames contain visible evidence supporting the anomaly decision."
        ),
        "event_chain_summary": {
            stage: _clean_text((raw_semantic.get("event_chain_summary") or {}).get(stage)) or by_stage[stage]
            for stage in ("precursor", "trigger", "confirmation")
        },
        "qa_focus_answers": {
            "existence": _clean_text((raw_semantic.get("qa_focus_answers") or {}).get("existence"))
            or ("No. No anomaly is visible in this video." if existence == "normal" else "Yes, there is an anomaly in this video."),
            "category": _clean_text((raw_semantic.get("qa_focus_answers") or {}).get("category"))
            or ("The video is normal." if existence == "normal" else f"The anomaly is {category or 'uncertain'}."),
            "temporal": _clean_text((raw_semantic.get("qa_focus_answers") or {}).get("temporal")) or temporal_answer,
        },
    }
    return normalize_semantic_answer_payload(semantic_candidate) or semantic_candidate


def normalize_fixed_baseline_prediction(payload: Any, *, duration_sec: float, evidence_top_k: int = 3) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}
    raw_decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else payload
    existence = _normalize_existence((raw_decision or {}).get("existence"))
    decision: Dict[str, Any] = canonicalize_category_payload(
        {
            "existence": existence,
            "category": (raw_decision or {}).get("category"),
            "anomaly_interval_sec": _normalize_interval_sec((raw_decision or {}).get("anomaly_interval_sec"), duration_sec=duration_sec),
            "precursor_interval_sec": _normalize_interval_sec((raw_decision or {}).get("precursor_interval_sec"), duration_sec=duration_sec),
        }
    )
    if existence == "normal":
        decision["category"] = "normal"
        decision["anomaly_interval_sec"] = None
        decision["precursor_interval_sec"] = None

    normalized_evidence: List[Dict[str, Any]] = []
    seen = set()
    for index, raw_item in enumerate(list(payload.get("evidence_topk") or []), start=1):
        if not isinstance(raw_item, dict):
            continue
        role = _normalize_role(raw_item.get("role"))
        interval = _normalize_interval_sec(
            raw_item.get("interval_sec")
            or raw_item.get("window_sec")
            or [raw_item.get("start_sec"), raw_item.get("end_sec")],
            duration_sec=duration_sec,
        )
        if not role or interval is None:
            continue
        key = (role, interval[0], interval[1])
        if key in seen:
            continue
        seen.add(key)
        normalized_evidence.append(
            {
                "rank": _safe_int(raw_item.get("rank"), index),
                "start_sec": interval[0],
                "end_sec": interval[1],
                "role": role,
                "description": _clean_text(raw_item.get("description")),
            }
        )
    normalized_evidence.sort(key=lambda item: (int(item.get("rank") or 0), float(item.get("start_sec") or 0.0)))
    normalized_evidence = normalized_evidence[: max(0, int(evidence_top_k))]

    raw_semantic = payload.get("semantic_answer") if isinstance(payload.get("semantic_answer"), dict) else {}
    semantic_answer = _fallback_semantic_answer(decision, normalized_evidence, raw_semantic=raw_semantic)
    return {
        "decision": decision,
        "semantic_answer": semantic_answer,
        "evidence_topk": normalized_evidence,
    }


def adapt_fixed_baseline_prediction_to_rollout(
    record: Dict[str, Any],
    normalized_prediction: Dict[str, Any],
    *,
    raw_response_text: str,
    parse_ok: bool,
    parse_error: str | None,
) -> Dict[str, Any]:
    decision = dict((normalized_prediction or {}).get("decision") or {})
    semantic_answer = dict((normalized_prediction or {}).get("semantic_answer") or {})
    evidence_topk = list((normalized_prediction or {}).get("evidence_topk") or [])
    stage_selected_moment_ids: Dict[str, List[str]] = {}
    evidence_ledger: List[Dict[str, Any]] = []
    covered_stages: List[str] = []
    for index, entry in enumerate(evidence_topk, start=1):
        stage = str(entry.get("role") or "").strip().lower()
        if stage not in {"precursor", "trigger", "confirmation"}:
            continue
        window_id = f"baseline_ev{index}"
        stage_selected_moment_ids.setdefault(stage, []).append(window_id)
        if stage not in covered_stages:
            covered_stages.append(stage)
        evidence_ledger.append(
            {
                "window_id": window_id,
                "start_sec": float(entry.get("start_sec") or 0.0),
                "end_sec": float(entry.get("end_sec") or 0.0),
                "role": stage,
                "description": _clean_text(entry.get("description")),
            }
        )
    covered_stages = normalize_event_chain_stages(covered_stages)
    final_answer = dict(decision)
    final_answer["covered_stages"] = covered_stages
    final_answer["stage_selected_moment_ids"] = stage_selected_moment_ids
    return {
        "video_id": record.get("video_id"),
        "file_name": record.get("file_name"),
        "source_dataset": record.get("source_dataset"),
        "split": record.get("split"),
        "raw_response_text": str(raw_response_text or ""),
        "parse_ok": bool(parse_ok),
        "parse_error": str(parse_error or "") or None,
        "final_answer": final_answer,
        "semantic_answer": semantic_answer,
        "state": {
            "active_evidence_window_ids": [entry["window_id"] for entry in evidence_ledger],
            "evidence_ledger": evidence_ledger,
        },
        "turns": [],
        "invalid_attempts": [],
        "num_turns": 0,
        "terminated_reason": "direct_fixed_baseline_single_shot",
    }


def _batch(values: Sequence[int], batch_size: int) -> Iterable[List[int]]:
    resolved_batch_size = max(1, int(batch_size))
    for start in range(0, len(values), resolved_batch_size):
        yield list(values[start : start + resolved_batch_size])


def _summarize_dataset_label(records: Sequence[Dict[str, Any]]) -> str:
    labels = sorted({str(record.get("source_dataset") or "").strip() for record in records if str(record.get("source_dataset") or "").strip()})
    if not labels:
        return "unknown"
    if len(labels) == 1:
        return labels[0]
    return "+".join(labels)


def _resolve_single_split(records: Sequence[Dict[str, Any]]) -> str:
    splits = sorted({str(record.get("split") or "").strip() for record in records if str(record.get("split") or "").strip()})
    if not splits:
        return ""
    if len(splits) == 1:
        return splits[0]
    return ",".join(splits)


def _paper_metrics(summary: Mapping[str, Any], semantic_metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "Existence Acc.": float(summary.get("existence_accuracy", 0.0) or 0.0),
        "Temporal mIoU": float(summary.get("temporal_miou", 0.0) or 0.0),
        "QA Accuracy": float(semantic_metrics.get("qa_accuracy_overall", 0.0) or 0.0),
        "Event-Chain F1": float(summary.get("event_chain_f1", 0.0) or 0.0),
        "Evidence F1@3": float(summary.get("evidence_f1_at_3", 0.0) or 0.0),
        "FECV Sufficiency": None,
    }


@dataclass
class FixedBaselineEvalConfig:
    base_model: str
    data_path: str
    output_dir: str
    data_root: str = ""
    include_splits: str = ""
    max_records: int = 0
    progress_every: int = 10
    batch_size: int = 1
    policy_max_new_tokens: int = 768
    max_total_images: int = 8
    max_seq_length: int = 8192
    max_image_side: int = 640
    max_image_pixels: int = 0
    num_preview_frames: int = 8
    evidence_top_k: int = 3
    enable_semantic_metrics: bool = True
    semantic_metrics: Sequence[str] | str = "qa_accuracy"
    semantic_bertscore_model_path: str = ""
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_3"
    use_generation_cache: bool = True
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    strict_json: bool = True
    use_storyboard: bool = False
    prompt_version: str = "v1"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "FixedBaselineEvalConfig":
        server = dict(_mapping(mapping.get("server")))
        client = dict(_mapping(mapping.get("client")))
        io_cfg = dict(_mapping(mapping.get("io")))
        baseline = dict(_mapping(mapping.get("baseline")))
        evaluation = dict(_mapping(mapping.get("evaluation")))
        data_path = str(io_cfg.get("data_path") or "").strip()
        if not data_path:
            raise ValueError("fixed baseline eval requires io.data_path pointing to a runtime JSONL manifest.")
        output_dir = str(io_cfg.get("output_dir") or "").strip()
        if not output_dir:
            raise ValueError("fixed baseline eval requires io.output_dir.")
        return cls(
            base_model=str(mapping.get("base_model") or "").strip(),
            data_path=data_path,
            output_dir=output_dir,
            data_root=str(io_cfg.get("data_root") or "").strip(),
            include_splits=str(io_cfg.get("include_splits") or "").strip(),
            max_records=int(io_cfg.get("max_records", 0) or 0),
            progress_every=int(evaluation.get("progress_every", 10) or 10),
            batch_size=int(client.get("batch_size", 1) or 1),
            policy_max_new_tokens=int(client.get("max_tokens", 768) or 768),
            max_total_images=int(client.get("max_total_images", 8) or 8),
            max_seq_length=int(client.get("max_seq_length", 8192) or 8192),
            max_image_side=int(client.get("max_image_side", 640) or 640),
            max_image_pixels=int(client.get("max_image_pixels", 0) or 0),
            num_preview_frames=int(client.get("num_preview_frames", 8) or 8),
            evidence_top_k=int(baseline.get("evidence_top_k", 3) or 3),
            enable_semantic_metrics=_bool(evaluation.get("enable_semantic_metrics"), True),
            semantic_metrics=evaluation.get("semantic_metrics", "qa_accuracy"),
            semantic_bertscore_model_path=(
                str(evaluation.get("semantic_bertscore_model_path") or evaluation.get("bertscore_model_path") or "").strip()
            ),
            torch_dtype=str(mapping.get("torch_dtype") or "bfloat16").strip() or "bfloat16",
            attn_implementation=str(mapping.get("attn_implementation") or "flash_attention_3").strip() or "flash_attention_3",
            use_generation_cache=_bool(client.get("use_generation_cache"), True),
            temperature=float(client.get("temperature", 0.0) or 0.0),
            top_p=float(client.get("top_p", 1.0) or 1.0),
            top_k=int(client.get("top_k", -1) or -1),
            strict_json=_bool(baseline.get("strict_json"), True),
            use_storyboard=_bool(baseline.get("use_storyboard"), False),
            prompt_version=str(baseline.get("prompt_version") or "v1").strip() or "v1",
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FixedBaselineEvalConfig":
        return cls.from_mapping(load_yaml_mapping(path))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_fixed_baseline_eval_job(config: FixedBaselineEvalConfig) -> Dict[str, Any]:
    if int(config.batch_size) < 1:
        raise ValueError("Fixed baseline direct-HF path requires client.batch_size>=1.")
    attn_implementation = str(config.attn_implementation or "").strip().lower()
    allowed_attention_backends = {"flash_attention_3", "flash_attention_2", "sdpa", "eager"}
    if attn_implementation and attn_implementation not in allowed_attention_backends:
        raise ValueError(
            "Fixed baseline direct-HF path requires attn_implementation to be one of "
            f"{sorted(allowed_attention_backends)}, got {config.attn_implementation!r}."
        )
    runtime = distributed_runtime_from_env()
    init_torch_distributed(runtime)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_shard_dir = output_dir / "raw_predictions"
    normalized_shard_dir = output_dir / "normalized_predictions"
    scored_shard_dir = output_dir / "scored_predictions"
    raw_shard_dir.mkdir(parents=True, exist_ok=True)
    normalized_shard_dir.mkdir(parents=True, exist_ok=True)
    scored_shard_dir.mkdir(parents=True, exist_ok=True)

    dataset = SaverAgentDataset(
        config.data_path,
        data_root=config.data_root,
        include_splits=config.include_splits,
        config=SaverAgentConfig(
            preview=PreviewConfig(
                num_preview_frames=int(config.num_preview_frames),
                max_preview_frames=int(config.num_preview_frames),
            ),
            initial_observation=InitialObservationConfig(mode="preview"),
        ),
        load_feature_cache=False,
        require_feature_cache=False,
    )
    total_records = len(dataset)
    all_indices = list(range(total_records))
    if int(config.max_records) > 0:
        all_indices = all_indices[: int(config.max_records)]
    local_indices = shard_sequence(all_indices, num_shards=runtime.world_size, shard_index=runtime.rank)
    runtime_log(
        f"fixed baseline eval: total={len(all_indices)} local={len(local_indices)} data_path={config.data_path}",
        runtime=runtime,
    )

    raw_records: List[Dict[str, Any]] = []
    normalized_records: List[Dict[str, Any]] = []
    if local_indices:
        local_runtime = TorchQwen3MessageRuntime(
            model_path=config.base_model,
            torch_dtype=config.torch_dtype,
            attn_implementation=config.attn_implementation,
            max_new_tokens=int(config.policy_max_new_tokens),
            max_total_images=int(config.max_total_images),
            max_seq_length=int(config.max_seq_length),
            max_image_side=int(config.max_image_side),
            max_image_pixels=int(config.max_image_pixels),
            temperature=float(config.temperature),
            top_p=float(config.top_p),
            top_k=int(config.top_k),
            use_generation_cache=bool(config.use_generation_cache),
        )
        sampling = MessageSamplingConfig(
            max_tokens=int(config.policy_max_new_tokens),
            temperature=float(config.temperature),
            top_p=float(config.top_p),
            top_k=int(config.top_k),
        )
        completed = 0
        local_total = len(local_indices)
        for batch_indices in _batch(local_indices, int(config.batch_size)):
            items = [dataset[idx] for idx in batch_indices]
            messages_batch = [build_fixed_baseline_messages(item) for item in items]
            response_texts = local_runtime.generate(messages_batch, sampling=sampling)
            for item, response_text in zip(items, response_texts):
                payload, parse_error = parse_fixed_baseline_response_text(response_text)
                normalized_prediction = normalize_fixed_baseline_prediction(
                    payload,
                    duration_sec=_safe_float(((item.get("video_meta") or {}).get("duration_sec")), 0.0),
                    evidence_top_k=int(config.evidence_top_k),
                )
                raw_records.append(
                    {
                        "video_id": item.get("video_id"),
                        "source_dataset": item.get("source_dataset"),
                        "split": item.get("split"),
                        "parse_ok": payload is not None,
                        "parse_error": parse_error,
                        "raw_response_text": str(response_text or ""),
                        "prediction": normalized_prediction,
                    }
                )
                normalized_records.append(
                    adapt_fixed_baseline_prediction_to_rollout(
                        item,
                        normalized_prediction,
                        raw_response_text=str(response_text or ""),
                        parse_ok=(payload is not None),
                        parse_error=parse_error,
                    )
                )
                completed += 1
                if should_log_progress(completed, local_total, int(config.progress_every)):
                    runtime_log(f"fixed baseline eval progress: {completed}/{local_total}", runtime=runtime)

    raw_shard_path = raw_shard_dir / f"part.rank{runtime.rank:02d}-of-{runtime.world_size:02d}.jsonl"
    normalized_shard_path = normalized_shard_dir / f"part.rank{runtime.rank:02d}-of-{runtime.world_size:02d}.jsonl"
    scored_shard_path = scored_shard_dir / f"part.rank{runtime.rank:02d}-of-{runtime.world_size:02d}.jsonl"
    save_rollout_records(raw_records, raw_shard_path, metadata={"input_kind": "jsonl"})
    save_rollout_records(normalized_records, normalized_shard_path, metadata={"input_kind": "jsonl"})
    save_rollout_records(normalized_records, scored_shard_path, metadata={"input_kind": "jsonl"})
    distributed_barrier(runtime)

    if not runtime.is_main_process:
        return {
            "rank": runtime.rank,
            "world_size": runtime.world_size,
            "local_records": len(normalized_records),
            "output_dir": str(output_dir),
        }

    merged_raw_records: List[Dict[str, Any]] = []
    merged_normalized_records: List[Dict[str, Any]] = []
    for rank in range(runtime.world_size):
        raw_path = raw_shard_dir / f"part.rank{rank:02d}-of-{runtime.world_size:02d}.jsonl"
        normalized_path = normalized_shard_dir / f"part.rank{rank:02d}-of-{runtime.world_size:02d}.jsonl"
        if raw_path.exists():
            merged_raw_records.extend(json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip())
        if normalized_path.exists():
            merged_normalized_records.extend(json.loads(line) for line in normalized_path.read_text(encoding="utf-8").splitlines() if line.strip())

    merged_raw_path = output_dir / "raw_predictions.merged.jsonl"
    merged_normalized_path = output_dir / "normalized_predictions.merged.jsonl"
    merged_scored_path = output_dir / "scored_predictions.merged.jsonl"
    metrics_path = output_dir / "metrics.json"
    semantic_metrics_path = output_dir / "semantic_metrics.json"
    table_metrics_path = output_dir / "table_metrics.json"
    config_path = output_dir / "config.snapshot.json"

    save_rollout_records(merged_raw_records, merged_raw_path, metadata={"input_kind": "jsonl"})
    save_rollout_records(merged_normalized_records, merged_normalized_path, metadata={"input_kind": "jsonl"})
    save_rollout_records(merged_normalized_records, merged_scored_path, metadata={"input_kind": "jsonl"})
    write_json(config.to_dict(), config_path)

    reference_data = ReferenceDataProvider(data_path=config.data_path, data_root=config.data_root)
    summary = summarize_saver_metrics(
        merged_normalized_records,
        reference_data=reference_data,
        evidence_top_k=int(config.evidence_top_k),
        include_diagnostic_summary=False,
    )
    semantic_metrics: Dict[str, Any] = {}
    if bool(config.enable_semantic_metrics):
        requested_metrics = (
            [str(metric).strip() for metric in config.semantic_metrics.split(",") if str(metric).strip()]
            if isinstance(config.semantic_metrics, str)
            else [str(metric).strip() for metric in config.semantic_metrics if str(metric).strip()]
        )
        semantic_metrics = evaluate_semantic_rollouts(
            merged_normalized_records,
            data_path=config.data_path,
            metrics=requested_metrics,
            bertscore_model_path=config.semantic_bertscore_model_path,
        )
    if semantic_metrics:
        write_json(semantic_metrics, semantic_metrics_path)
        summary["semantic_metrics"] = semantic_metrics

    paper_metrics = _paper_metrics(summary, semantic_metrics)
    dataset_label = _summarize_dataset_label(reference_data.records)
    filtered_reference_records = (
        [
            record
            for record in reference_data.records
            if str(record.get("split") or "").strip() in {value.strip() for value in str(config.include_splits).split(",") if value.strip()}
        ]
        if str(config.include_splits).strip()
        else reference_data.records
    )
    split_label = _resolve_single_split(filtered_reference_records)
    table_metrics = {
        "dataset": dataset_label,
        "split": split_label,
        "num_records": len(merged_normalized_records),
        "paper_metrics": paper_metrics,
        "raw_metric_keys": {
            "existence_accuracy": float(summary.get("existence_accuracy", 0.0) or 0.0),
            "temporal_miou": float(summary.get("temporal_miou", 0.0) or 0.0),
            "qa_accuracy_overall": float(semantic_metrics.get("qa_accuracy_overall", 0.0) or 0.0),
            "event_chain_f1": float(summary.get("event_chain_f1", 0.0) or 0.0),
            "evidence_f1_at_3": float(summary.get("evidence_f1_at_3", 0.0) or 0.0),
        },
    }
    summary["paper_metrics"] = paper_metrics
    summary["dataset"] = dataset_label
    summary["split"] = split_label
    summary["num_scored_records"] = len(merged_normalized_records)
    summary["baseline_protocol"] = "fixed_observation_single_shot"

    write_json(summary, metrics_path)
    write_json(table_metrics, table_metrics_path)
    runtime_log(f"fixed baseline eval artifacts saved to {output_dir}", runtime=runtime, main_process_only=True)
    return summary
