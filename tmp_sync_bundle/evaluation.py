from __future__ import annotations

import copy
import gc
import os
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from run_saver_rollout import _serialize_result
from saver_v3.core.adapter import TimeSearchRolloutAdapter
from saver_v3.data.config import DEFAULT_POLICY_MAX_NEW_TOKENS, DEFAULT_ROLLOUT_MAX_TURNS, SaverAgentConfig
from saver_v3.data.dataset import SaverAgentDataset
from saver_v3.data.materialized_cache import (
    MATERIALIZED_RUNTIME_ITEMS_FORMAT,
    MaterializedRuntimeItemDataset,
    ensure_materialized_cache_metadata,
)
from saver_v3.data.protocol_signature import DEFAULT_TEACHER_ROLE, build_protocol_signature
from saver_v3.common.experiment_logging import append_jsonl, utc_timestamp
from saver_v3.metrics.legacy_metrics import summarize_saver_metrics
from saver_v3.metrics.offline_scoring import (
    ReferenceDataProvider,
    load_rollout_records,
    save_rollout_records,
    score_rollout_records,
)
from saver_v3.core.protocol_guidance import summarize_evidence_ledger
from saver_v3.core.proposal import SiglipFeatureEncoder
from saver_v3.core.qwen_verifier import DEFAULT_VERIFIER_MODEL_PATH, QwenSelfVerifier
from saver_v3.core.rollout import SaverRolloutRunner
from saver_v3.common.runtime import (
    DistributedRuntime,
    claim_next_dynamic_task_index,
    distributed_barrier,
    distributed_runtime_from_env,
    initialize_dynamic_task_queue,
    init_torch_distributed,
    record_dynamic_task_completion,
    resolve_inference_device_map,
    resolve_shard_spec,
    runtime_log,
    shard_sequence,
    should_log_progress,
)
from saver_v3.core.semantic_answer import (
    build_public_semantic_replay_scaffold,
    merge_public_semantic_replay_with_decision,
    normalize_semantic_answer_payload,
    normalize_public_semantic_replay_payload,
    public_semantic_replay_to_text,
    semantic_answer_to_text,
)
from saver_v3.metrics.semantic_metrics import DEFAULT_SEMANTIC_METRICS, evaluate_semantic_rollouts

DEFAULT_DISTRIBUTED_WAIT_TIMEOUT_SEC = 0.0
DEFAULT_EVAL_SEMANTIC_METRICS = DEFAULT_SEMANTIC_METRICS
_PER_VIDEO_FILENAME_SANITIZE_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class RolloutEvaluationConfig:
    data_path: str | Path
    data_root: str | Path = ""
    materialized_items_path: str | Path = ""
    require_materialized_cache: bool = False
    include_splits: Optional[Sequence[str] | str] = None
    max_records: int = 0
    inline_rollout_eval: bool = False
    rollout_max_turns: int = DEFAULT_ROLLOUT_MAX_TURNS
    rollout_batch_size: int = 1
    policy_max_new_tokens: int = DEFAULT_POLICY_MAX_NEW_TOKENS
    use_generation_cache: bool = True
    generation_cache_oom_fallback: bool = True
    enable_semantic_replay: bool = True
    semantic_replay_max_new_tokens: int = 512
    max_total_images: int = 0
    max_tool_message_frames: int = 0
    max_total_video_frames: int = 0
    max_seq_length: int = 0
    keep_recent_tool_image_messages: int = 0
    keep_recent_text_messages: int = 0
    max_image_side: int = 0
    max_image_pixels: int = 0
    proposal_model_path: str | Path = ""
    proposal_torch_dtype: str = "auto"
    proposal_device: str = ""
    verifier_backend: str = "qwen_self_verifier"
    verifier_model_path: str | Path = DEFAULT_VERIFIER_MODEL_PATH
    verifier_torch_dtype: str = "auto"
    verifier_device_map: Any = "auto"
    verifier_attn_implementation: str = ""
    verifier_max_new_tokens: int = 512
    attach_reference_diagnostics: bool = False
    progress_every: int = 1
    saver_config: Optional[SaverAgentConfig] = None
    enable_semantic_metrics: bool = True
    semantic_metrics: Sequence[str] | str = DEFAULT_EVAL_SEMANTIC_METRICS
    semantic_judge_base_url: str = ""
    semantic_judge_model: str = ""
    semantic_judge_cache_path: str | Path = ""
    semantic_judge_timeout_sec: float = 30.0
    semantic_bertscore_model_path: str | Path = ""


def _semantic_replay_enabled(eval_config: RolloutEvaluationConfig) -> bool:
    return bool(eval_config.enable_semantic_replay) and int(eval_config.semantic_replay_max_new_tokens) > 0


def _resolve_rollout_batch_size(eval_config: RolloutEvaluationConfig) -> int:
    return max(1, int(getattr(eval_config, "rollout_batch_size", 1) or 1))


def _resolve_rollout_eval_saver_configs(
    eval_config: RolloutEvaluationConfig,
) -> tuple[SaverAgentConfig, SaverAgentConfig]:
    artifact_saver_config = (
        copy.deepcopy(eval_config.saver_config) if eval_config.saver_config is not None else SaverAgentConfig()
    )
    runtime_saver_config = copy.deepcopy(artifact_saver_config)
    runtime_saver_config.rollout_trace.record_message_history = False
    runtime_saver_config.rollout_trace.record_observation_content = False
    # Search/query metrics depend on per-turn state deltas and new evidence traces.
    runtime_saver_config.rollout_trace.record_state_deltas = True
    runtime_saver_config.rollout_trace.record_counterfactual_trace = False
    return artifact_saver_config, runtime_saver_config


def _claim_dynamic_rollout_batch_indices(
    *,
    all_indices: Sequence[int],
    task_queue_dir: Path,
    runtime: DistributedRuntime,
    batch_size: int,
) -> list[int]:
    claimed_dataset_indices: list[int] = []
    for _ in range(max(1, int(batch_size))):
        task_index = claim_next_dynamic_task_index(task_queue_dir, runtime=runtime)
        if task_index is None:
            break
        claimed_dataset_indices.append(int(all_indices[int(task_index)]))
    return claimed_dataset_indices


def _attach_verifier_context(
    item: Dict[str, Any],
    *,
    eval_config: RolloutEvaluationConfig,
    verifier_runtime: Any,
    verifier_device_map: Any,
) -> None:
    cache = item["multimodal_cache"]
    cache["verifier_backend"] = eval_config.verifier_backend
    cache["verifier_model_path"] = str(eval_config.verifier_model_path)
    cache["verifier_torch_dtype"] = eval_config.verifier_torch_dtype
    cache["verifier_device_map"] = verifier_device_map
    cache["verifier_attn_implementation"] = eval_config.verifier_attn_implementation
    cache["verifier_max_new_tokens"] = int(eval_config.verifier_max_new_tokens)
    if verifier_runtime is not None:
        cache["verifier_runtime"] = verifier_runtime


def _attach_proposal_context(
    item: Dict[str, Any],
    *,
    proposal_runtime: Any,
    strict_feature_guided_proposal: bool = False,
) -> None:
    cache = item.setdefault("multimodal_cache", {})
    if bool(strict_feature_guided_proposal):
        cache["strict_feature_guided_proposal"] = True
    if proposal_runtime is not None:
        cache["proposal_runtime"] = proposal_runtime


def _attach_reference_free_eval_guard(
    item: Dict[str, Any],
) -> None:
    cache = item.setdefault("multimodal_cache", {})
    cache.pop("allow_external_verifier_fallback", None)


def _records_require_feature_guided_proposal(records: Sequence[Dict[str, Any]] | None) -> bool:
    for record in list(records or []):
        allowed_tools = list(((record.get("tool_io") or {}).get("allowed_tools") or []))
        if any(str(tool_name or "").strip() == "seek_evidence" for tool_name in allowed_tools):
            return True
        for step in list(record.get("oracle_trajectory") or []):
            if str(step.get("tool") or "").strip() == "seek_evidence":
                return True
    return False


def _resolve_proposal_device(
    explicit_device: str | None,
    *,
    runtime: DistributedRuntime,
) -> str:
    explicit = str(explicit_device or "").strip()
    if explicit and max(1, int(getattr(runtime, "world_size", 1) or 1)) > 1 and re.fullmatch(r"(?:cuda:\d+|\d+)", explicit.lower()):
        runtime_log(
            (
                "distributed eval proposal device warning: "
                f"explicit proposal_device={explicit!r} pins every rank's SigLIP encoder to the same CUDA device. "
                "Leave proposal_device empty to use per-rank local_rank placement."
            ),
            runtime=runtime,
            main_process_only=True,
        )
    if str(explicit_device or "").strip():
        return str(explicit_device)
    try:
        import torch
    except Exception:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    try:
        visible_cuda_devices = int(torch.cuda.device_count())
    except Exception:
        visible_cuda_devices = 0
    if visible_cuda_devices <= 0:
        return "cpu"
    local_rank = int(getattr(runtime, "local_rank", 0) or 0)
    if 0 <= local_rank < visible_cuda_devices:
        return f"cuda:{local_rank}"
    runtime_log(
        (
            "eval proposal device fallback: "
            f"local_rank={local_rank} is outside visible_cuda_devices={visible_cuda_devices}; using cuda:0"
        ),
        runtime=runtime,
    )
    return "cuda:0"


def _load_verifier_runtime(
    *,
    eval_config: RolloutEvaluationConfig,
    runtime: DistributedRuntime,
) -> Any:
    if not bool(eval_config.attach_reference_diagnostics):
        return None
    if eval_config.verifier_backend != "qwen_self_verifier":
        return None
    resolved_device_map = resolve_inference_device_map(eval_config.verifier_device_map, runtime=runtime)
    runtime_log(
        f"loading eval verifier from {eval_config.verifier_model_path} with device_map={resolved_device_map}",
        runtime=runtime,
    )
    return QwenSelfVerifier.from_pretrained(
        eval_config.verifier_model_path,
        torch_dtype=eval_config.verifier_torch_dtype,
        device_map=resolved_device_map,
        attn_implementation=eval_config.verifier_attn_implementation or None,
        max_new_tokens=eval_config.verifier_max_new_tokens,
    )


def _load_proposal_runtime(
    *,
    eval_config: RolloutEvaluationConfig,
    runtime: DistributedRuntime,
) -> Any:
    if not str(eval_config.proposal_model_path or "").strip():
        return None
    resolved_device = _resolve_proposal_device(eval_config.proposal_device, runtime=runtime)
    runtime_log(
        f"loading eval proposal model from {eval_config.proposal_model_path} on device={resolved_device}",
        runtime=runtime,
    )
    return SiglipFeatureEncoder.from_pretrained(
        str(eval_config.proposal_model_path),
        torch_dtype=eval_config.proposal_torch_dtype,
        device=resolved_device,
    )


def _cleanup_cuda_cache(*, runtime: DistributedRuntime, reason: str) -> None:
    gc.collect()
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
        runtime_log(
            f"rollout eval memory cleanup: {reason}",
            runtime=runtime,
        )
    except Exception:
        return


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc or "").strip().lower()
    if not text:
        return False
    return "out of memory" in text or "cuda oom" in text


def _run_with_generation_cache_oom_fallback(
    fn,
    *,
    policy: Any,
    eval_config: RolloutEvaluationConfig,
    runtime: DistributedRuntime,
    stage_name: str,
    video_id: str,
):
    try:
        return fn()
    except Exception as exc:
        if not bool(eval_config.generation_cache_oom_fallback):
            raise
        if not bool(getattr(policy, "use_generation_cache", False)):
            raise
        if not _is_cuda_oom(exc):
            raise
        runtime_log(
            (
                f"rollout eval {stage_name} hit CUDA OOM for video_id={video_id}; "
                "retrying the current video once with generation cache disabled on this rank"
            ),
            runtime=runtime,
        )
        try:
            setattr(policy, "use_generation_cache", False)
        except Exception:
            raise
        _cleanup_cuda_cache(runtime=runtime, reason=f"after rollout-eval {stage_name} CUDA OOM")
        return fn()


def _clear_stale_json_shards(shard_dir: Path) -> int:
    removed = 0
    for pattern in ("*.json", "*.jsonl", ".tmp.*.jsonl"):
        for shard_path in shard_dir.glob(pattern):
            if not shard_path.is_file():
                continue
            shard_path.unlink()
            removed += 1
    return removed


def _clear_rollout_eval_sync_files(*, eval_root: Path) -> int:
    removed = 0
    for path in (
        eval_root / "metrics.json",
        eval_root / "semantic_metrics.json",
        eval_root / "failure.json",
        eval_root / "metadata.json",
    ):
        if path.exists():
            path.unlink()
            removed += 1
    return removed


def _sanitize_per_video_filename_component(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return "video"
    text = _PER_VIDEO_FILENAME_SANITIZE_PATTERN.sub("_", text).strip("._-")
    return text or "video"


def _build_per_video_filename_map(
    *,
    dataset: SaverAgentDataset,
    dataset_indices: Sequence[int],
) -> dict[int, str]:
    dataset_records = list(getattr(dataset, "records", []) or [])
    if not dataset_records:
        dataset_records = list(getattr(dataset, "items", []) or [])
    stems_by_index: dict[int, str] = {}
    stem_counts: dict[str, int] = {}
    for dataset_index in dataset_indices:
        record = {}
        if 0 <= int(dataset_index) < len(dataset_records):
            record = dict(dataset_records[int(dataset_index)] or {})
        video_id = record.get("video_id") or f"video_{int(dataset_index):05d}"
        stem = _sanitize_per_video_filename_component(video_id)
        stems_by_index[int(dataset_index)] = stem
        stem_counts[stem] = int(stem_counts.get(stem, 0)) + 1
    filenames: dict[int, str] = {}
    for dataset_index, stem in stems_by_index.items():
        if int(stem_counts.get(stem, 0)) > 1:
            filenames[int(dataset_index)] = f"{stem}__idx{int(dataset_index):05d}.jsonl"
        else:
            filenames[int(dataset_index)] = f"{stem}.jsonl"
    return filenames


def _append_per_video_rollout_record(
    *,
    per_video_dir: Path,
    filename: str,
    record: Dict[str, Any],
    record_stage: str,
    epoch_index: int,
    runtime: DistributedRuntime,
    dataset_index: int,
) -> None:
    payload = copy.deepcopy(record)
    payload["record_stage"] = str(record_stage)
    payload["epoch_index"] = int(epoch_index)
    payload["rank"] = int(runtime.rank)
    payload["dataset_index"] = int(dataset_index)
    payload["saved_at"] = utc_timestamp()
    path = per_video_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _normalize_rollout_semantic_fields(rollout: Dict[str, Any]) -> None:
    semantic_answer = normalize_semantic_answer_payload(rollout.get("semantic_answer"))
    rollout["semantic_answer"] = semantic_answer
    rollout["semantic_answer_text"] = semantic_answer_to_text(semantic_answer)
    if semantic_answer is None:
        rollout.pop("semantic_answer_source", None)
        return
    source = str(rollout.get("semantic_answer_source") or "").strip()
    if source == "finalize_case":
        rollout["semantic_answer_source"] = "finalize_case_inline"
        return
    if source == "semantic_replay":
        rollout["semantic_answer_source"] = "semantic_replay_fallback"
        return
    if not str(rollout.get("semantic_answer_source") or "").strip():
        if str(rollout.get("final_answer_source") or "") == "finalize_case":
            rollout["semantic_answer_source"] = "finalize_case_inline"
        elif str(rollout.get("final_answer_source") or "") == "answer":
            rollout["semantic_answer_source"] = "answer"


def _rollout_has_semantic_answer(rollout: Dict[str, Any]) -> bool:
    return normalize_semantic_answer_payload(rollout.get("semantic_answer")) is not None


def _normalize_eval_include_splits(include_splits: Optional[Sequence[str] | str]) -> list[str]:
    if include_splits is None:
        return []
    if isinstance(include_splits, str):
        value = include_splits.strip()
        return [value] if value else []
    return [str(value).strip() for value in include_splits if str(value).strip()]


def _build_rollout_eval_metadata(eval_config: RolloutEvaluationConfig) -> Dict[str, Any]:
    return {
        "data_path": str(eval_config.data_path),
        "data_root": str(eval_config.data_root),
        "include_splits": _normalize_eval_include_splits(eval_config.include_splits),
        "max_records": int(eval_config.max_records),
        "rollout_max_turns": int(eval_config.rollout_max_turns),
        "rollout_batch_size": int(_resolve_rollout_batch_size(eval_config)),
        "policy_max_new_tokens": int(eval_config.policy_max_new_tokens),
        "use_generation_cache": bool(eval_config.use_generation_cache),
        "generation_cache_oom_fallback": bool(eval_config.generation_cache_oom_fallback),
        "enable_semantic_replay": bool(eval_config.enable_semantic_replay),
        "semantic_replay_max_new_tokens": int(eval_config.semantic_replay_max_new_tokens),
        "max_total_images": int(eval_config.max_total_images),
        "max_tool_message_frames": int(eval_config.max_tool_message_frames),
        "max_total_video_frames": int(eval_config.max_total_video_frames),
        "max_seq_length": int(eval_config.max_seq_length),
        "keep_recent_tool_image_messages": int(eval_config.keep_recent_tool_image_messages),
        "keep_recent_text_messages": int(eval_config.keep_recent_text_messages),
        "max_image_side": int(eval_config.max_image_side),
        "max_image_pixels": int(eval_config.max_image_pixels),
        "proposal_model_path": str(eval_config.proposal_model_path or ""),
        "verifier_backend": str(eval_config.verifier_backend),
        "attach_reference_diagnostics": bool(eval_config.attach_reference_diagnostics),
        "enable_semantic_metrics": bool(eval_config.enable_semantic_metrics),
        "semantic_metrics": _normalize_eval_semantic_metrics(eval_config.semantic_metrics),
        "semantic_judge_base_url": str(eval_config.semantic_judge_base_url or ""),
        "semantic_judge_model": str(eval_config.semantic_judge_model or ""),
        "semantic_bertscore_model_path": str(eval_config.semantic_bertscore_model_path or ""),
    }


def _normalize_eval_semantic_metrics(metrics: Sequence[str] | str | None) -> list[str]:
    if metrics is None:
        return list(DEFAULT_EVAL_SEMANTIC_METRICS)
    if isinstance(metrics, str):
        values = [value.strip() for value in metrics.split(",") if value.strip()]
        return values or list(DEFAULT_EVAL_SEMANTIC_METRICS)
    values = [str(value).strip() for value in metrics if str(value).strip()]
    return values or list(DEFAULT_EVAL_SEMANTIC_METRICS)


def _window_record_lookup(rollout: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    state = dict(rollout.get("state") or {})
    for source_key in ("visited_windows", "evidence_ledger"):
        for entry in list(state.get(source_key) or []):
            window_id = str(entry.get("window_id") or "").strip()
            if not window_id or window_id in lookup:
                continue
            lookup[window_id] = copy.deepcopy(entry)
    return lookup


def _infer_selected_window_ids(rollout: Dict[str, Any]) -> list[str]:
    for turn in reversed(list(rollout.get("turns") or [])):
        if str(turn.get("tool_name") or "").strip() != "verify_hypothesis":
            continue
        selected = [
            str(value).strip()
            for value in (
                turn.get("verifier_verified_window_ids")
                or turn.get("verified_window_ids")
                or turn.get("selected_window_ids")
                or []
            )
            if str(value).strip()
        ]
        if selected:
            return selected
    state = dict(rollout.get("state") or {})
    return [
        str(value).strip()
        for value in list(state.get("active_evidence_window_ids") or [])
        if str(value).strip()
    ]


def _semantic_replay_window_ids(rollout: Dict[str, Any]) -> list[str]:
    window_ids = _infer_selected_window_ids(rollout)
    if window_ids:
        return window_ids
    state = dict(rollout.get("state") or {})
    fallback: list[str] = []
    seen = set()
    for entry in list(state.get("evidence_ledger") or []):
        window_id = str(entry.get("window_id") or "").strip()
        if not window_id or window_id in seen:
            continue
        fallback.append(window_id)
        seen.add(window_id)
    return fallback


def _selected_window_records(rollout: Dict[str, Any], window_ids: Sequence[str]) -> list[Dict[str, Any]]:
    lookup = _window_record_lookup(rollout)
    selected_records: list[Dict[str, Any]] = []
    for value in list(window_ids or []):
        window_id = str(value).strip()
        if window_id and window_id in lookup:
            selected_records.append(copy.deepcopy(lookup[window_id]))
    return selected_records


def _build_public_semantic_replay_messages(
    item: Dict[str, Any],
    *,
    rollout: Dict[str, Any],
    max_images: int = 12,
) -> Optional[list[Dict[str, Any]]]:
    finalized_case = dict(rollout.get("final_answer") or (rollout.get("state") or {}).get("finalized_case") or {})
    if not finalized_case:
        return None
    window_ids = _semantic_replay_window_ids(rollout)
    selected_records = _selected_window_records(rollout, window_ids)
    if not selected_records:
        return None

    multimodal_cache = dict(item.get("multimodal_cache") or {})
    question = str(multimodal_cache.get("question") or "Summarize the finalized anomaly decision from the provided evidence.")
    scaffold = build_public_semantic_replay_scaffold()
    user_content: list[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "This is a post-finalize semantic replay. "
                "Do not call tools. Use only the provided evidence package.\n"
                f"Task: {question}\n"
                f"Finalized decision: {json.dumps(finalized_case, ensure_ascii=False)}\n"
                f"Selected evidence window ids: {json.dumps(list(window_ids), ensure_ascii=False)}\n"
                f"Evidence ledger summary: {summarize_evidence_ledger(selected_records)}\n"
                f"Return exactly one <answer></answer> JSON in this shape: {scaffold}"
            ),
        }
    ]

    video = multimodal_cache.get("video")
    image_count = 0
    for record in selected_records:
        frame_indices = list(record.get("selected_frame_indices") or [])
        timestamps = list(record.get("selected_timestamps") or [])
        for frame_index, timestamp in zip(frame_indices, timestamps):
            if image_count >= int(max_images):
                break
            if video is None:
                continue
            try:
                image = video[int(frame_index)]
            except Exception:
                continue
            user_content.append({"type": "text", "text": f"{float(timestamp):.3f}s"})
            user_content.append(
                {
                    "type": "image",
                    "image": image,
                    "sampled_frame_index": int(frame_index),
                    "timestamp_sec": float(timestamp),
                }
            )
            image_count += 1
        if image_count >= int(max_images):
            break

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are SAVER semantic replay. "
                        "This is a separate post-finalize explanation pass. "
                        "You must not call tools."
                    ),
                }
            ],
        },
        {"role": "user", "content": user_content},
    ]


def _generate_with_optional_token_override(
    policy: Any,
    *,
    messages: list[Dict[str, Any]],
    max_new_tokens: int,
) -> str:
    if not hasattr(policy, "generate_from_messages"):
        raise AttributeError("Semantic replay requires a policy with generate_from_messages(messages).")
    original_max_new_tokens = getattr(policy, "max_new_tokens", None)
    override_applied = isinstance(original_max_new_tokens, int)
    if override_applied:
        policy.max_new_tokens = int(max_new_tokens)
    try:
        return str(policy.generate_from_messages(messages) or "")
    finally:
        if override_applied:
            policy.max_new_tokens = original_max_new_tokens


def _run_public_semantic_replay(
    policy: Any,
    *,
    item: Dict[str, Any],
    rollout: Dict[str, Any],
    max_new_tokens: int,
) -> Dict[str, Any]:
    finalized_case = dict(rollout.get("final_answer") or (rollout.get("state") or {}).get("finalized_case") or {})
    if not finalized_case:
        return {
            "available": False,
            "messages": None,
            "response_text": None,
            "semantic_replay": None,
            "semantic_replay_text": None,
            "semantic_answer": None,
            "semantic_answer_text": None,
            "unavailable_reason": "no_finalized_case",
        }
    try:
        messages = _build_public_semantic_replay_messages(item, rollout=rollout)
    except Exception as exc:
        return {
            "available": False,
            "messages": None,
            "response_text": None,
            "semantic_replay": None,
            "semantic_replay_text": None,
            "semantic_answer": None,
            "semantic_answer_text": None,
            "unavailable_reason": f"message_build_failed:{exc}",
        }
    if not messages:
        return {
            "available": False,
            "messages": None,
            "response_text": None,
            "semantic_replay": None,
            "semantic_replay_text": None,
            "semantic_answer": None,
            "semantic_answer_text": None,
            "unavailable_reason": "no_selected_evidence_windows",
        }
    try:
        response_text = _generate_with_optional_token_override(
            policy,
            messages=messages,
            max_new_tokens=max_new_tokens,
        )
    except Exception as exc:
        return {
            "available": False,
            "messages": messages,
            "response_text": None,
            "semantic_replay": None,
            "semantic_replay_text": None,
            "semantic_answer": None,
            "semantic_answer_text": None,
            "unavailable_reason": f"generation_failed:{exc}",
        }

    answer_text = TimeSearchRolloutAdapter.parse_answer_text(response_text)
    parsed_payload: Optional[Dict[str, Any]] = None
    if answer_text:
        try:
            parsed_json = json.loads(answer_text)
        except Exception:
            parsed_json = None
        if isinstance(parsed_json, dict):
            parsed_payload = normalize_public_semantic_replay_payload(parsed_json)
    semantic_answer = merge_public_semantic_replay_with_decision(
        parsed_payload,
        finalized_case=finalized_case,
    )
    return {
        "available": bool(parsed_payload is not None and semantic_answer is not None),
        "messages": messages,
        "response_text": response_text,
        "semantic_replay": parsed_payload,
        "semantic_replay_text": public_semantic_replay_to_text(parsed_payload),
        "semantic_answer": semantic_answer,
        "semantic_answer_text": semantic_answer_to_text(semantic_answer),
        "unavailable_reason": None if parsed_payload is not None and semantic_answer is not None else "invalid_semantic_replay_payload",
    }


def _expected_scored_shard_paths(
    *,
    scored_shard_dir: Path,
    runtime: DistributedRuntime,
) -> list[Path]:
    return [
        scored_shard_dir / f"part.rank{rank:02d}-of-{runtime.world_size:02d}.jsonl"
        for rank in range(int(runtime.world_size))
    ]


def _expected_raw_shard_paths(
    *,
    raw_shard_dir: Path,
    runtime: DistributedRuntime,
) -> list[Path]:
    return [
        raw_shard_dir / f"part.rank{rank:02d}-of-{runtime.world_size:02d}.jsonl"
        for rank in range(int(runtime.world_size))
    ]


def _write_rollout_eval_failure_marker(*, failure_path: Path, exc: Exception) -> None:
    _write_rollout_eval_failure_marker_with_context(failure_path=failure_path, exc=exc)


def _write_rollout_eval_failure_marker_with_context(
    *,
    failure_path: Path,
    exc: Exception,
    runtime: Optional[DistributedRuntime] = None,
    phase: str = "",
    overwrite_existing: bool = False,
) -> None:
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    if failure_path.exists() and not overwrite_existing:
        return
    failure_payload = {
        "error_type": exc.__class__.__name__,
        "error": str(exc),
    }
    if runtime is not None:
        failure_payload["rank"] = int(runtime.rank)
        failure_payload["world_size"] = int(runtime.world_size)
    if str(phase).strip():
        failure_payload["phase"] = str(phase)
    failure_path.write_text(json.dumps(failure_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_wait_deadline(timeout_sec: float | None) -> float | None:
    if timeout_sec is None:
        return None
    try:
        resolved_timeout = float(timeout_sec)
    except Exception:
        return None
    if resolved_timeout <= 0.0:
        return None
    return time.time() + max(1.0, resolved_timeout)


def _format_wait_timeout_suffix(deadline: float | None) -> str:
    if deadline is None:
        return "without a timeout"
    remaining_sec = max(0.0, deadline - time.time())
    return f"with {remaining_sec:.0f}s remaining before timeout"


def _wait_for_current_scored_records(
    *,
    scored_shard_dir: Path,
    failure_path: Path,
    runtime: DistributedRuntime,
    timeout_sec: float = DEFAULT_DISTRIBUTED_WAIT_TIMEOUT_SEC,
    poll_interval_sec: float = 1.0,
) -> list[Dict[str, Any]]:
    deadline = _resolve_wait_deadline(timeout_sec)
    logged_wait_message = False
    shard_status: dict[str, str] = {}
    expected_paths = _expected_scored_shard_paths(scored_shard_dir=scored_shard_dir, runtime=runtime)
    while True:
        if failure_path.exists():
            try:
                payload = json.loads(failure_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {"error": failure_path.read_text(encoding="utf-8")}
            raise RuntimeError(
                "rollout eval failed before all scored shards were ready: "
                + json.dumps(payload, ensure_ascii=False)
            )
        shard_status.clear()
        merged_records: list[Dict[str, Any]] = []
        all_ready = True
        for shard_path in expected_paths:
            if not shard_path.exists():
                all_ready = False
                shard_status[str(shard_path)] = "missing"
                continue
            try:
                shard_records, _ = load_rollout_records(shard_path)
            except Exception as exc:
                all_ready = False
                shard_status[str(shard_path)] = f"unreadable: {exc}"
                continue
            merged_records.extend(shard_records)
            shard_status[str(shard_path)] = f"ready:{len(shard_records)}"
        if all_ready:
            return merged_records
        if not logged_wait_message:
            missing_count = sum(1 for status in shard_status.values() if status != "ready:0" and not status.startswith("ready:"))
            runtime_log(
                (
                    "rollout eval is waiting for scored shard outputs "
                    f"({len(expected_paths) - missing_count}/{len(expected_paths)} ready; "
                    f"continuing {_format_wait_timeout_suffix(deadline)})"
                ),
                runtime=runtime,
                main_process_only=True,
            )
            logged_wait_message = True
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError(
                "Timed out while waiting for rollout-eval scored shard outputs: "
                + json.dumps(shard_status, ensure_ascii=False)
            )
        time.sleep(max(0.05, float(poll_interval_sec)))


def _wait_for_current_raw_records(
    *,
    raw_shard_dir: Path,
    failure_path: Path,
    runtime: DistributedRuntime,
    timeout_sec: float = DEFAULT_DISTRIBUTED_WAIT_TIMEOUT_SEC,
    poll_interval_sec: float = 1.0,
) -> list[Dict[str, Any]]:
    deadline = _resolve_wait_deadline(timeout_sec)
    logged_wait_message = False
    shard_status: dict[str, str] = {}
    expected_paths = _expected_raw_shard_paths(raw_shard_dir=raw_shard_dir, runtime=runtime)
    while True:
        if failure_path.exists():
            try:
                payload = json.loads(failure_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {"error": failure_path.read_text(encoding="utf-8")}
            raise RuntimeError(
                "rollout eval failed before all raw shards were ready: "
                + json.dumps(payload, ensure_ascii=False)
            )
        shard_status.clear()
        merged_records: list[Dict[str, Any]] = []
        all_ready = True
        for shard_path in expected_paths:
            if not shard_path.exists():
                all_ready = False
                shard_status[str(shard_path)] = "missing"
                continue
            try:
                shard_records, _ = load_rollout_records(shard_path)
            except Exception as exc:
                all_ready = False
                shard_status[str(shard_path)] = f"unreadable: {exc}"
                continue
            merged_records.extend(shard_records)
            shard_status[str(shard_path)] = f"ready:{len(shard_records)}"
        if all_ready:
            return merged_records
        if not logged_wait_message:
            missing_count = sum(1 for status in shard_status.values() if status != "ready:0" and not status.startswith("ready:"))
            runtime_log(
                (
                    "rollout eval is waiting for raw shard outputs "
                    f"({len(expected_paths) - missing_count}/{len(expected_paths)} ready; "
                    f"continuing {_format_wait_timeout_suffix(deadline)})"
                ),
                runtime=runtime,
                main_process_only=True,
            )
            logged_wait_message = True
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError(
                "Timed out while waiting for rollout-eval raw shard outputs: "
                + json.dumps(shard_status, ensure_ascii=False)
            )
        time.sleep(max(0.05, float(poll_interval_sec)))


def _load_current_scored_records(
    *,
    scored_shard_dir: Path,
    runtime: DistributedRuntime,
) -> list[Dict[str, Any]]:
    expected_paths = _expected_scored_shard_paths(scored_shard_dir=scored_shard_dir, runtime=runtime)
    missing_paths = [str(path) for path in expected_paths if not path.exists()]
    if missing_paths:
        raise RuntimeError(
            "rollout eval is missing scored shard outputs for the current distributed run: "
            + ", ".join(missing_paths)
        )
    merged_records: list[Dict[str, Any]] = []
    for shard_path in expected_paths:
        shard_records, _ = load_rollout_records(shard_path)
        merged_records.extend(shard_records)
    return merged_records


def _score_rollout_records_with_dynamic_claiming(
    *,
    all_indices: Sequence[int],
    raw_records_by_dataset_index: Dict[int, Dict[str, Any]],
    task_queue_dir: Path,
    runtime: DistributedRuntime,
    active_participant: bool,
    progress_every: int,
    progress_label: str,
    score_fn: Any = score_rollout_records,
    score_fn_kwargs: Optional[Dict[str, Any]] = None,
) -> list[Dict[str, Any]]:
    initialize_dynamic_task_queue(
        task_queue_dir,
        num_tasks=len(all_indices),
        runtime=runtime,
        timeout_sec=5.0,
        poll_interval_sec=0.05,
    )
    if not active_participant:
        return []

    local_scored_records: list[Dict[str, Any]] = []
    shared_score_kwargs = dict(score_fn_kwargs or {})
    while True:
        task_index = claim_next_dynamic_task_index(task_queue_dir, runtime=runtime)
        if task_index is None:
            break
        dataset_index = int(all_indices[int(task_index)])
        raw_record = raw_records_by_dataset_index.get(dataset_index)
        if not isinstance(raw_record, dict):
            raise KeyError(f"Missing raw rollout record for dataset_index={dataset_index}")
        scored_batch = list(
            score_fn(
                [copy.deepcopy(raw_record)],
                progress_every=0,
                progress_label=str(progress_label),
                show_progress_bar=False,
                runtime=runtime,
                **shared_score_kwargs,
            )
            or []
        )
        if len(scored_batch) != 1:
            raise ValueError(
                "Distributed rollout-eval scoring expected exactly one scored record per claimed task, "
                f"but received {len(scored_batch)} for dataset_index={dataset_index}"
            )
        local_scored_records.extend(scored_batch)
        progress_completed = record_dynamic_task_completion(task_queue_dir, runtime=runtime)
        if should_log_progress(progress_completed, len(all_indices), int(progress_every)):
            runtime_log(
                (
                    f"{progress_label}: {progress_completed}/{len(all_indices)} "
                    f"video_id={scored_batch[0].get('video_id', '')}"
                ),
                runtime=runtime,
            )
    return local_scored_records


def _wait_for_rollout_eval_completion(
    *,
    metrics_path: Path,
    failure_path: Path,
    runtime: DistributedRuntime,
    timeout_sec: float = DEFAULT_DISTRIBUTED_WAIT_TIMEOUT_SEC,
    poll_interval_sec: float = 1.0,
) -> None:
    deadline = _resolve_wait_deadline(timeout_sec)
    logged_wait_message = False
    while True:
        if metrics_path.exists():
            return
        if failure_path.exists():
            try:
                payload = json.loads(failure_path.read_text(encoding="utf-8"))
            except Exception:
                payload = {"error": failure_path.read_text(encoding="utf-8")}
            raise RuntimeError(
                "rollout eval failed on the main process: " + json.dumps(payload, ensure_ascii=False)
            )
        if not logged_wait_message:
            runtime_log(
                (
                    "rollout eval worker is waiting for the completion marker "
                    f"at {metrics_path} ({_format_wait_timeout_suffix(deadline)})"
                ),
                runtime=runtime,
                main_process_only=True,
            )
            logged_wait_message = True
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError(f"Timed out while waiting for rollout-eval completion marker at {metrics_path}")
        time.sleep(max(0.05, float(poll_interval_sec)))


def run_rollout_evaluation(
    policy: Any,
    *,
    eval_config: RolloutEvaluationConfig,
    output_dir: str | Path,
    epoch_index: int,
    runtime: Optional[DistributedRuntime] = None,
) -> Optional[Dict[str, Any]]:
    runtime = runtime or distributed_runtime_from_env()
    init_torch_distributed(runtime)
    shard_spec = resolve_shard_spec(runtime=runtime)
    artifact_saver_config, saver_config = _resolve_rollout_eval_saver_configs(eval_config)
    runtime_log(
        (
            "rollout eval policy budget: "
            f"max_new_tokens_per_turn={int(eval_config.policy_max_new_tokens)} "
            f"rollout_batch_size={int(_resolve_rollout_batch_size(eval_config))} "
            f"use_generation_cache={bool(eval_config.use_generation_cache)} "
            f"max_total_images={int(eval_config.max_total_images) if int(eval_config.max_total_images) > 0 else 'all'} "
            f"max_tool_message_frames={int(eval_config.max_tool_message_frames) or 'all'} "
            f"max_total_video_frames={int(eval_config.max_total_video_frames) or 'all'} "
            f"max_seq_length={int(eval_config.max_seq_length) or 'off'} "
            f"keep_recent_tool_image_messages={int(eval_config.keep_recent_tool_image_messages) or 'all'} "
            f"keep_recent_text_messages={int(eval_config.keep_recent_text_messages) or 'all'} "
            f"max_image_side={int(eval_config.max_image_side) or 'off'} "
            f"max_image_pixels={int(eval_config.max_image_pixels) or 'off'}"
        ),
        runtime=runtime,
        main_process_only=True,
    )
    if hasattr(policy, "use_generation_cache"):
        try:
            policy.use_generation_cache = bool(eval_config.use_generation_cache)
        except Exception:
            pass
    if hasattr(policy, "max_seq_length"):
        try:
            policy.max_seq_length = int(eval_config.max_seq_length)
        except Exception:
            pass
    if hasattr(policy, "keep_recent_text_messages"):
        try:
            policy.keep_recent_text_messages = int(eval_config.keep_recent_text_messages)
        except Exception:
            pass
    if hasattr(policy, "keep_recent_tool_image_messages"):
        try:
            policy.keep_recent_tool_image_messages = int(eval_config.keep_recent_tool_image_messages)
        except Exception:
            pass
    if hasattr(policy, "max_tool_message_frames"):
        try:
            policy.max_tool_message_frames = int(eval_config.max_tool_message_frames)
        except Exception:
            pass
    if hasattr(policy, "max_total_video_frames"):
        try:
            policy.max_total_video_frames = int(eval_config.max_total_video_frames)
        except Exception:
            pass
    strict_feature_guided_proposal = bool(str(eval_config.proposal_model_path or "").strip())
    if str(eval_config.materialized_items_path or "").strip():
        expected_protocol_signature = build_protocol_signature(
            config=artifact_saver_config,
            max_turns=int(eval_config.rollout_max_turns),
            policy_max_new_tokens=int(eval_config.policy_max_new_tokens),
            teacher_role=DEFAULT_TEACHER_ROLE,
        )
        ensure_materialized_cache_metadata(
            eval_config.materialized_items_path,
            expected_format=MATERIALIZED_RUNTIME_ITEMS_FORMAT,
            expected_source_path=eval_config.data_path,
            expected_include_splits=eval_config.include_splits,
            expected_config=artifact_saver_config,
            expected_protocol_signature=expected_protocol_signature,
            require_config_match=True,
            require_source=True,
        )
        dataset = MaterializedRuntimeItemDataset(
            eval_config.materialized_items_path,
            include_splits=eval_config.include_splits,
            config=artifact_saver_config,
            require_frame_cache=True,
            require_feature_cache=True,
        )
    else:
        if bool(eval_config.require_materialized_cache):
            raise ValueError(
                "Rollout evaluation requires io.materialized_items_path when require_materialized_cache is enabled."
            )
        dataset = SaverAgentDataset(
            eval_config.data_path,
            data_root=eval_config.data_root,
            config=saver_config,
            include_splits=eval_config.include_splits,
            require_frame_cache=True,
            require_feature_cache=True,
            strict_feature_guided_proposal=strict_feature_guided_proposal,
        )
    if _records_require_feature_guided_proposal(getattr(dataset, "records", None)) and not strict_feature_guided_proposal:
        raise ValueError(
            "Rollout evaluation requires proposal_model_path because the eval data exposes seek_evidence."
        )
    if hasattr(dataset, "format_frame_cache_status"):
        runtime_log(
            dataset.format_frame_cache_status(prefix="rollout eval frame cache"),
            runtime=runtime,
            main_process_only=True,
        )
    all_indices = list(range(len(dataset)))
    if int(eval_config.max_records) > 0:
        all_indices = all_indices[: int(eval_config.max_records)]
    static_local_indices = shard_sequence(all_indices, num_shards=shard_spec.num_shards, shard_index=shard_spec.shard_index)
    use_dynamic_claiming = shard_spec.is_sharded
    active_rank_count = min(len(all_indices), int(shard_spec.num_shards)) if use_dynamic_claiming else 1
    active_participant = (not use_dynamic_claiming) or (runtime.rank < active_rank_count)

    eval_root = Path(output_dir) / "rollout_eval" / f"epoch_{int(epoch_index):03d}"
    metrics_path = eval_root / "metrics.json"
    semantic_metrics_path = eval_root / "semantic_metrics.json"
    failure_path = eval_root / "failure.json"
    metadata_path = eval_root / "metadata.json"
    raw_shard_dir = eval_root / "raw_shards"
    scored_shard_dir = eval_root / "scored_shards"
    per_video_dir = eval_root / "per_video"
    task_queue_dir = eval_root / "rollout_task_queue"
    score_task_queue_dir = eval_root / "score_task_queue"
    raw_shard_dir.mkdir(parents=True, exist_ok=True)
    scored_shard_dir.mkdir(parents=True, exist_ok=True)
    per_video_dir.mkdir(parents=True, exist_ok=True)
    if runtime.is_main_process:
        removed_raw_shards = _clear_stale_json_shards(raw_shard_dir)
        removed_scored_shards = _clear_stale_json_shards(scored_shard_dir)
        removed_per_video_files = _clear_stale_json_shards(per_video_dir)
        removed_sync_files = _clear_rollout_eval_sync_files(eval_root=eval_root)
        if removed_raw_shards > 0:
            runtime_log(
                f"cleared {removed_raw_shards} stale rollout-eval raw shard files from {raw_shard_dir}",
                runtime=runtime,
                main_process_only=True,
            )
        if removed_scored_shards > 0:
            runtime_log(
                f"cleared {removed_scored_shards} stale rollout-eval scored shard files from {scored_shard_dir}",
                runtime=runtime,
                main_process_only=True,
            )
        if removed_per_video_files > 0:
            runtime_log(
                f"cleared {removed_per_video_files} stale rollout-eval per-video files from {per_video_dir}",
                runtime=runtime,
                main_process_only=True,
            )
        if removed_sync_files > 0:
            runtime_log(
                f"cleared {removed_sync_files} stale rollout-eval sync files from {eval_root}",
                runtime=runtime,
                main_process_only=True,
            )
        metadata_path.write_text(
            json.dumps(_build_rollout_eval_metadata(eval_config), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    distributed_barrier(runtime)
    per_video_filename_by_index = _build_per_video_filename_map(dataset=dataset, dataset_indices=all_indices)
    runtime_log(
        f"per-video rollout logging enabled: {per_video_dir}",
        runtime=runtime,
        main_process_only=True,
    )
    if use_dynamic_claiming:
        try:
            initialize_dynamic_task_queue(
                task_queue_dir,
                num_tasks=len(all_indices),
                runtime=runtime,
                timeout_sec=5.0,
                poll_interval_sec=0.05,
            )
        except TimeoutError:
            use_dynamic_claiming = False
            active_rank_count = 1
            active_participant = True
            runtime_log(
                "dynamic rollout-eval task queue was not initialized by the main process; falling back to static shard assignment",
                runtime=runtime,
            )

    proposal_runtime = None
    verifier_runtime = None
    reference_data = None
    summary: Optional[Dict[str, Any]] = None
    try:
        _cleanup_cuda_cache(runtime=runtime, reason="before loading rollout-eval auxiliary runtimes")
        proposal_runtime = _load_proposal_runtime(eval_config=eval_config, runtime=runtime) if active_participant and all_indices else None
        verifier_runtime = _load_verifier_runtime(eval_config=eval_config, runtime=runtime) if active_participant and all_indices else None
        verifier_kwargs = {
            "verifier_backend": eval_config.verifier_backend,
            "verifier_model_path": str(eval_config.verifier_model_path),
            "verifier_torch_dtype": eval_config.verifier_torch_dtype,
            "verifier_device_map": resolve_inference_device_map(eval_config.verifier_device_map, runtime=runtime),
            "verifier_attn_implementation": eval_config.verifier_attn_implementation,
            "verifier_max_new_tokens": int(eval_config.verifier_max_new_tokens),
        }
        if verifier_runtime is not None:
            verifier_kwargs["verifier_runtime"] = verifier_runtime

        runner = SaverRolloutRunner(
            adapter=TimeSearchRolloutAdapter(config=saver_config),
            max_turns=int(eval_config.rollout_max_turns),
            config=saver_config,
        )
        rollout_batch_size = _resolve_rollout_batch_size(eval_config)
        local_rollouts = []
        completed = 0
        while True:
            if use_dynamic_claiming:
                if not active_participant:
                    break
                batch_dataset_indices = _claim_dynamic_rollout_batch_indices(
                    all_indices=all_indices,
                    task_queue_dir=task_queue_dir,
                    runtime=runtime,
                    batch_size=rollout_batch_size,
                )
                if not batch_dataset_indices:
                    break
            else:
                if completed >= len(static_local_indices):
                    break
                batch_dataset_indices = [
                    int(dataset_index)
                    for dataset_index in static_local_indices[completed : completed + rollout_batch_size]
                ]

            def _build_item(dataset_index: int) -> Dict[str, Any]:
                item = dataset[int(dataset_index)]
                _attach_reference_free_eval_guard(item)
                _attach_proposal_context(
                    item,
                    proposal_runtime=proposal_runtime,
                    strict_feature_guided_proposal=strict_feature_guided_proposal,
                )
                return item

            batch_items = [_build_item(int(dataset_index)) for dataset_index in batch_dataset_indices]
            batch_video_ids = [str(item.get("video_id") or dataset_index) for item, dataset_index in zip(batch_items, batch_dataset_indices)]
            batch_video_label = (
                batch_video_ids[0]
                if len(batch_video_ids) == 1
                else f"{batch_video_ids[0]} +{len(batch_video_ids) - 1} more"
            )

            def _run_raw_rollout_batch():
                return runner.run_episodes(batch_items, policy)

            result = _run_with_generation_cache_oom_fallback(
                _run_raw_rollout_batch,
                policy=policy,
                eval_config=eval_config,
                runtime=runtime,
                stage_name="rollout",
                video_id=batch_video_label,
            )
            batch_progress_completed = None
            for dataset_index, item, item_result in zip(batch_dataset_indices, batch_items, result):
                serialized = _serialize_result(item_result)
                serialized["dataset_index"] = int(dataset_index)
                serialized["semantic_fallback_used"] = False
                _normalize_rollout_semantic_fields(serialized)
                if _semantic_replay_enabled(eval_config) and not _rollout_has_semantic_answer(serialized):
                    replay_result = _run_with_generation_cache_oom_fallback(
                        lambda item=item, rollout=serialized: _run_public_semantic_replay(
                            policy,
                            item=item,
                            rollout=rollout,
                            max_new_tokens=int(eval_config.semantic_replay_max_new_tokens),
                        ),
                        policy=policy,
                        eval_config=eval_config,
                        runtime=runtime,
                        stage_name="semantic_replay",
                        video_id=str(serialized.get("video_id") or dataset_index),
                    )
                    if replay_result.get("semantic_replay") is not None:
                        serialized["semantic_replay"] = copy.deepcopy(replay_result.get("semantic_replay"))
                        serialized["semantic_replay_text"] = replay_result.get("semantic_replay_text")
                    if replay_result.get("semantic_answer") is not None:
                        serialized["semantic_answer"] = copy.deepcopy(replay_result.get("semantic_answer"))
                        serialized["semantic_answer_text"] = replay_result.get("semantic_answer_text")
                        serialized["semantic_answer_source"] = "semantic_replay_fallback"
                        serialized["semantic_fallback_used"] = True
                    if replay_result.get("response_text") is not None:
                        serialized["semantic_replay_response_text"] = replay_result.get("response_text")
                    if replay_result.get("messages") is not None:
                        serialized["semantic_replay_available"] = bool(replay_result.get("available"))
                        if replay_result.get("unavailable_reason") is not None:
                            serialized["semantic_replay_unavailable_reason"] = replay_result.get("unavailable_reason")
                _normalize_rollout_semantic_fields(serialized)
                _append_per_video_rollout_record(
                    per_video_dir=per_video_dir,
                    filename=per_video_filename_by_index.get(dataset_index, f"video_{dataset_index:05d}.jsonl"),
                    record=serialized,
                    record_stage="rollout_raw",
                    epoch_index=epoch_index,
                    runtime=runtime,
                    dataset_index=int(dataset_index),
                )
                local_rollouts.append(serialized)
                completed += 1
                progress_completed = (
                    record_dynamic_task_completion(task_queue_dir, runtime=runtime)
                    if use_dynamic_claiming
                    else completed
                )
                batch_progress_completed = int(progress_completed)
            if result:
                if batch_progress_completed is None:
                    batch_progress_completed = int(completed)
                if should_log_progress(batch_progress_completed, len(all_indices), int(eval_config.progress_every)):
                    runtime_log(
                        (
                            f"rollout eval progress: {batch_progress_completed}/{len(all_indices)} "
                            f"video_id={str(result[-1].get('video_id') or batch_video_ids[-1] or '')} "
                            f"batch_size={len(batch_dataset_indices)}"
                        ),
                        runtime=runtime,
                    )
        local_raw_path = raw_shard_dir / f"part.rank{runtime.rank:02d}-of-{runtime.world_size:02d}.jsonl"
        _tmp_raw_path = raw_shard_dir / f".tmp.part.rank{runtime.rank:02d}-of-{runtime.world_size:02d}.jsonl"
        save_rollout_records(local_rollouts, _tmp_raw_path, metadata={"input_kind": "jsonl"})
        with open(_tmp_raw_path, "rb") as _fh:
            os.fsync(_fh.fileno())
        _tmp_raw_path.rename(local_raw_path)

        reference_data = ReferenceDataProvider(data_path=eval_config.data_path, data_root=eval_config.data_root)
        score_progress_label = f"epoch {int(epoch_index)} eval score progress"
        if use_dynamic_claiming:
            merged_raw_records = _wait_for_current_raw_records(
                raw_shard_dir=raw_shard_dir,
                failure_path=failure_path,
                runtime=runtime,
            )
            raw_records_by_dataset_index: Dict[int, Dict[str, Any]] = {}
            for raw_record in merged_raw_records:
                try:
                    dataset_index = int(raw_record.get("dataset_index", -1))
                except Exception:
                    dataset_index = -1
                if dataset_index < 0:
                    continue
                if dataset_index in raw_records_by_dataset_index:
                    raise ValueError(
                        "rollout eval encountered duplicate raw rollout records for "
                        f"dataset_index={dataset_index}"
                    )
                raw_records_by_dataset_index[dataset_index] = raw_record
            missing_dataset_indices = [
                int(dataset_index)
                for dataset_index in all_indices
                if int(dataset_index) not in raw_records_by_dataset_index
            ]
            if missing_dataset_indices:
                raise RuntimeError(
                    "rollout eval is missing raw rollout records for dataset indices: "
                    + ", ".join(str(value) for value in missing_dataset_indices)
                )
            local_scored_records = _score_rollout_records_with_dynamic_claiming(
                all_indices=all_indices,
                raw_records_by_dataset_index=raw_records_by_dataset_index,
                task_queue_dir=score_task_queue_dir,
                runtime=runtime,
                active_participant=active_participant,
                progress_every=int(eval_config.progress_every),
                progress_label=score_progress_label,
                score_fn=score_rollout_records,
                score_fn_kwargs={
                    "reference_data": reference_data,
                    "policy": policy,
                    "verifier_backend": eval_config.verifier_backend,
                    "force_reverify": bool(eval_config.attach_reference_diagnostics),
                    "attach_reference_offline_verifier": bool(eval_config.attach_reference_diagnostics),
                    "verifier_kwargs": verifier_kwargs,
                },
            )
        else:
            local_scored_records = score_rollout_records(
                local_rollouts,
                reference_data=reference_data,
                policy=policy,
                verifier_backend=eval_config.verifier_backend,
                force_reverify=bool(eval_config.attach_reference_diagnostics),
                attach_reference_offline_verifier=bool(eval_config.attach_reference_diagnostics),
                verifier_kwargs=verifier_kwargs,
                progress_every=eval_config.progress_every,
                progress_label=score_progress_label,
                show_progress_bar=False,
                runtime=runtime,
            )
        local_scored_path = scored_shard_dir / f"part.rank{runtime.rank:02d}-of-{runtime.world_size:02d}.jsonl"
        _tmp_scored_path = scored_shard_dir / f".tmp.part.rank{runtime.rank:02d}-of-{runtime.world_size:02d}.jsonl"
        save_rollout_records(local_scored_records, _tmp_scored_path, metadata={"input_kind": "jsonl"})
        with open(_tmp_scored_path, "rb") as _fh:
            os.fsync(_fh.fileno())
        _tmp_scored_path.rename(local_scored_path)
        for scored_record in local_scored_records:
            raw_dataset_index = scored_record.get("dataset_index", -1)
            try:
                dataset_index = int(raw_dataset_index)
            except Exception:
                dataset_index = -1
            if dataset_index < 0:
                continue
            _append_per_video_rollout_record(
                per_video_dir=per_video_dir,
                filename=per_video_filename_by_index.get(dataset_index, f"video_{dataset_index:05d}.jsonl"),
                record=scored_record,
                record_stage="final",
                epoch_index=epoch_index,
                runtime=runtime,
                dataset_index=dataset_index,
            )

        if runtime.is_main_process:
            try:
                merged_scored_records = _wait_for_current_scored_records(
                    scored_shard_dir=scored_shard_dir,
                    failure_path=failure_path,
                    runtime=runtime,
                )
                summary = summarize_saver_metrics(
                    merged_scored_records,
                    reference_data=reference_data,
                    include_diagnostic_summary=bool(eval_config.attach_reference_diagnostics),
                )
                if bool(eval_config.enable_semantic_metrics):
                    semantic_metrics = evaluate_semantic_rollouts(
                        merged_scored_records,
                        data_path=eval_config.data_path,
                        metrics=_normalize_eval_semantic_metrics(eval_config.semantic_metrics),
                        judge_base_url=eval_config.semantic_judge_base_url,
                        judge_model=eval_config.semantic_judge_model,
                        judge_cache_path=eval_config.semantic_judge_cache_path,
                        judge_timeout_sec=eval_config.semantic_judge_timeout_sec,
                        bertscore_model_path=eval_config.semantic_bertscore_model_path,
                    )
                    semantic_metrics_path.write_text(
                        json.dumps(semantic_metrics, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    summary["semantic_metrics"] = semantic_metrics
                summary["epoch_index"] = int(epoch_index)
                summary["num_scored_records"] = len(merged_scored_records)
                metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                runtime_log(
                    f"epoch {int(epoch_index)} rollout eval metrics saved to {metrics_path}",
                    runtime=runtime,
                    main_process_only=True,
                )
            except Exception as exc:
                _write_rollout_eval_failure_marker_with_context(
                    failure_path=failure_path,
                    exc=exc,
                    runtime=runtime,
                    phase="main_wait_or_summarize",
                    overwrite_existing=False,
                )
                raise
        elif runtime.is_distributed:
            _wait_for_rollout_eval_completion(
                metrics_path=metrics_path,
                failure_path=failure_path,
                runtime=runtime,
            )
        return summary
    except Exception as exc:
        _write_rollout_eval_failure_marker_with_context(
            failure_path=failure_path,
            exc=exc,
            runtime=runtime,
            phase="distributed_rollout_eval",
            overwrite_existing=False,
        )
        raise
    finally:
        proposal_runtime = None
        verifier_runtime = None
        reference_data = None
        _cleanup_cuda_cache(runtime=runtime, reason="after releasing rollout-eval auxiliary runtimes")
