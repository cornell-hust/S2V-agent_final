#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from split_utils import parse_include_splits
import torch.distributed as dist

from saver_v3.data.prepared_metadata import (
    build_jsonl_provenance,
    config_from_prepared_sft_metadata,
    ensure_prepared_sft_metadata,
    prepared_sft_metadata_path,
)
from saver_v3.data.protocol_signature import TEACHER_ROLE_AUXILIARY, build_protocol_signature
from saver_v3.core.environment import parse_actions_and_contents
from saver_v3.core.proposal import SiglipFeatureEncoder
from saver_v3.common.runtime import (
    create_progress_bar,
    distributed_runtime_from_env,
    init_torch_distributed,
    resolve_inference_device_map,
    resolve_shard_spec,
    runtime_log,
    sharded_output_path,
)
from saver_v3.teacher.teacher_judge import (
    QwenTeacherJudge,
    build_teacher_judge_package_record,
    is_teacher_judge_candidate,
)
from saver_v3.sft.training import (
    _FrameReferenceResolver,
    expand_compact_trace_row_to_step_rows,
)
from saver_v3.data.training_data import (
    _build_teacher_override_verify_payload,
    convert_step_examples_to_episode_records,
    is_compact_trace_sft_record,
    rebuild_teacher_rollout_primary_examples,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run distributed online teacher-judge annotation over compact_trace_v4 prepared SFT JSONL."
    )
    parser.add_argument("--input", required=True, help="Prepared SFT JSONL path to annotate online.")
    parser.add_argument("--output", required=True, help="Output JSONL path for annotated examples.")
    parser.add_argument("--model-path", required=True, help="Local Qwen teacher judge model path.")
    parser.add_argument("--include-splits", default="", help="Optional comma-separated split whitelist.")
    parser.add_argument("--skip-invalid-jsonl-lines", action="store_true", help="Skip malformed JSONL lines instead of failing.")
    parser.add_argument(
        "--input-mode",
        choices=["text_only", "multimodal_visual", "auto"],
        default="auto",
        help="Teacher judge input mode.",
    )
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype for the teacher judge.")
    parser.add_argument("--device-map", default="auto", help="device_map for the teacher judge.")
    parser.add_argument("--attn-implementation", default="", help="Optional attention backend for the teacher judge.")
    parser.add_argument("--proposal-model-path", default="", help="Optional proposal encoder path for strict compact-trace replay.")
    parser.add_argument("--proposal-torch-dtype", default="auto", help="Torch dtype for the proposal encoder.")
    parser.add_argument("--proposal-device", default="", help="Optional device for the proposal encoder.")
    parser.add_argument("--max-new-tokens", type=int, default=384, help="Generation length for the teacher judge.")
    parser.add_argument("--max-images", type=int, default=8, help="Maximum images passed to the teacher judge per example.")
    parser.add_argument(
        "--topk-frames-per-view",
        type=int,
        default=4,
        help="Maximum number of frames sampled into each teacher-judge view package.",
    )
    parser.add_argument(
        "--frame-cache-max-cached-videos",
        type=int,
        default=64,
        help="How many frame_cache tensors/video readers to keep open while resolving image_ref payloads.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Teacher-judge micro-batch size inside each shard process.",
    )
    parser.add_argument("--overwrite-existing", action="store_true", help="Overwrite existing teacher-judge labels.")
    parser.add_argument("--no-progress-bar", action="store_true", help="Disable interactive tqdm progress bars.")
    parser.add_argument("--num-shards", type=int, default=0, help="Optional number of shard workers.")
    parser.add_argument("--shard-index", type=int, default=-1, help="Optional shard index for this process.")
    parser.set_defaults(materialize_teacher_rollout_primary=True)
    parser.add_argument(
        "--materialize-teacher-rollout-primary",
        dest="materialize_teacher_rollout_primary",
        action="store_true",
        help="Deprecated no-op. The output is always the final teacher-annotated compact_trace_v4 prepared SFT file.",
    )
    parser.add_argument(
        "--no-materialize-teacher-rollout-primary",
        dest="materialize_teacher_rollout_primary",
        action="store_false",
        help="Deprecated no-op retained for CLI compatibility. The output remains the final teacher-annotated compact_trace_v4 file.",
    )
    return parser.parse_args(argv)


def _resolve_proposal_device(explicit_device: str | None, *, runtime: Any) -> str:
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
    return "cuda:0"


def _load_proposal_runtime(args: argparse.Namespace, *, runtime: Any) -> Any:
    if not str(getattr(args, "proposal_model_path", "") or "").strip():
        return None
    resolved_device = _resolve_proposal_device(getattr(args, "proposal_device", ""), runtime=runtime)
    return SiglipFeatureEncoder.from_pretrained(
        str(args.proposal_model_path),
        torch_dtype=str(getattr(args, "proposal_torch_dtype", "auto") or "auto"),
        device=resolved_device,
    )


def _jsonl_decode_error_message(path: str | Path, line_number: int, line: str, exc: Exception) -> str:
    preview = line.strip().replace("\t", " ")
    if len(preview) > 240:
        preview = preview[:240] + "..."
    return f"Invalid JSONL at {path}:{line_number}: {exc}. Line preview: {preview}"


def _load_jsonl(
    path: str | Path,
    *,
    skip_invalid_lines: bool = False,
    include_splits: Optional[str | List[str]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    invalid_messages: List[str] = []
    allowed_splits = set(parse_include_splits(include_splits) or [])
    with Path(path).open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                message = _jsonl_decode_error_message(path, line_number, line, exc)
                if not skip_invalid_lines:
                    raise ValueError(message) from exc
                invalid_messages.append(message)
                continue
            if allowed_splits and str(row.get("split") or "").strip() not in allowed_splits:
                continue
            rows.append(row)
    if invalid_messages:
        print(
            json.dumps(
                {
                    "warning": "skipped_invalid_jsonl_lines",
                    "path": str(path),
                    "num_skipped": len(invalid_messages),
                    "first_error": invalid_messages[0],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return rows


def _prepared_rows_require_feature_guided_proposal(rows: List[Dict[str, Any]]) -> bool:
    for row in rows:
        for step in list(row.get("oracle_trajectory") or []):
            if str(step.get("tool") or "").strip() == "seek_evidence":
                return True
    return False


def _write_jsonl(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_episode_row(row: Dict[str, Any]) -> bool:
    return isinstance(row.get("messages"), list) and isinstance(row.get("assistant_supervision"), list)


def _extract_message_text(message: Dict[str, Any]) -> str:
    for item in list(message.get("content") or []):
        if isinstance(item, dict) and item.get("type") == "text":
            return str(item.get("text") or "")
    return ""


def _step_source_from_episode_source(source: Any) -> str:
    normalized = str(source or "").strip()
    return normalized[: -len("_episode")] if normalized.endswith("_episode") else normalized


def _parse_step_target(response_text: str) -> tuple[str, Optional[str]]:
    actions, contents = parse_actions_and_contents([str(response_text or "")])
    action = actions[0] if actions else None
    content = contents[0] if contents else None
    if action == "tool_call":
        function_payload = ((content or {}).get("function") or {}) if isinstance(content, dict) else {}
        tool_name = str(function_payload.get("name") or "").strip() or None
        if tool_name is None:
            raise ValueError("Supervised assistant tool call is missing a tool name.")
        return "tool_call", tool_name
    if action == "answer":
        return "answer", None
    raise ValueError(
        "Episode-format teacher-judge inputs must supervise assistant messages that decode to exactly one "
        "<tool_call> or <answer> payload."
    )


def _expand_episode_row_to_step_rows(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    messages = list(row.get("messages") or [])
    assistant_supervision = list(row.get("assistant_supervision") or [])
    base_step_row = copy.deepcopy(dict(row))
    base_step_row.pop("assistant_supervision", None)
    base_step_row.pop("episode_weight", None)
    base_step_row["source"] = _step_source_from_episode_source(base_step_row.get("source"))

    step_rows: List[Dict[str, Any]] = []
    for step_index, supervision in enumerate(assistant_supervision, start=1):
        try:
            assistant_message_index = int(supervision.get("assistant_message_index"))
        except Exception as exc:
            raise ValueError(
                "Episode-format teacher-judge input has an invalid assistant_message_index: "
                f"video_id={row.get('video_id')} source={row.get('source')} supervision={supervision}"
            ) from exc
        if assistant_message_index < 0 or assistant_message_index >= len(messages):
            raise ValueError(
                "Episode-format teacher-judge input references an out-of-range assistant message: "
                f"video_id={row.get('video_id')} source={row.get('source')} "
                f"assistant_message_index={assistant_message_index} num_messages={len(messages)}"
            )
        assistant_message = messages[assistant_message_index]
        if str(assistant_message.get("role") or "") != "assistant":
            raise ValueError(
                "Episode-format teacher-judge input assistant_supervision must point to assistant messages: "
                f"video_id={row.get('video_id')} source={row.get('source')} "
                f"assistant_message_index={assistant_message_index} role={assistant_message.get('role')}"
            )
        target_response = _extract_message_text(assistant_message)
        if not target_response:
            raise ValueError(
                "Episode-format teacher-judge input assistant supervision message is missing text content: "
                f"video_id={row.get('video_id')} source={row.get('source')} "
                f"assistant_message_index={assistant_message_index}"
            )
        target_action, tool_name = _parse_step_target(target_response)
        step_row = copy.deepcopy(base_step_row)
        step_row["messages"] = copy.deepcopy(messages[:assistant_message_index])
        step_row["step_index"] = step_index
        step_row["target_response"] = target_response
        step_row["target_action"] = target_action
        step_row["tool_name"] = tool_name
        step_rows.append(step_row)
    return step_rows


def _expand_episode_rows_to_step_rows(rows: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    summary = {
        "num_input_step_rows": 0,
        "num_input_episode_rows": 0,
        "num_expanded_step_rows": 0,
        "num_output_rows": 0,
    }
    expanded_rows: List[Dict[str, Any]] = []
    for row in rows:
        if _is_episode_row(row):
            summary["num_input_episode_rows"] += 1
            step_rows = _expand_episode_row_to_step_rows(row)
            expanded_rows.extend(step_rows)
            summary["num_expanded_step_rows"] += len(step_rows)
            continue
        summary["num_input_step_rows"] += 1
        expanded_rows.append(copy.deepcopy(row))
    summary["num_output_rows"] = len(expanded_rows)
    return expanded_rows, summary


def _expand_compact_trace_rows_to_step_rows(
    rows: List[Dict[str, Any]],
    *,
    prepared_metadata: Dict[str, Any],
    proposal_runtime: Any = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    config = config_from_prepared_sft_metadata(prepared_metadata)
    expanded_rows: List[Dict[str, Any]] = []
    for source_row_index, row in enumerate(rows):
        step_rows = expand_compact_trace_row_to_step_rows(
            row,
            config=config,
            load_frame_cache=False,
            load_feature_cache=True,
            proposal_runtime=proposal_runtime,
            strict_feature_guided_proposal=True,
        )
        for step_row in step_rows:
            step_row["_compact_source_row_index"] = int(source_row_index)
        expanded_rows.extend(step_rows)
    summary = {
        "num_input_compact_trace_rows": len(rows),
        "num_expanded_step_rows": len(expanded_rows),
        "num_output_rows": len(expanded_rows),
    }
    return expanded_rows, summary


def _iter_compact_trace_step_rows(
    rows: List[Dict[str, Any]],
    *,
    prepared_metadata: Dict[str, Any],
    proposal_runtime: Any = None,
) -> Iterable[tuple[int, Dict[str, Any]]]:
    config = config_from_prepared_sft_metadata(prepared_metadata)
    global_step_row_index = 0
    for compact_source_row_index, row in enumerate(rows):
        step_rows = expand_compact_trace_row_to_step_rows(
            row,
            config=config,
            load_frame_cache=False,
            load_feature_cache=True,
            proposal_runtime=proposal_runtime,
            strict_feature_guided_proposal=True,
        )
        for step_row in step_rows:
            step_row["_compact_source_row_index"] = int(compact_source_row_index)
            yield int(global_step_row_index), step_row
            global_step_row_index += 1


def _candidate_manifest_entry(candidate_record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source_row_index": int(candidate_record.get("source_row_index", -1)),
        "video_id": str(candidate_record.get("video_id") or ""),
        "tool_name": str(candidate_record.get("tool_name") or ""),
        "requested_input_mode": str(candidate_record.get("requested_input_mode") or ""),
        "actual_input_mode": str(candidate_record.get("actual_input_mode") or ""),
        "max_images": int(candidate_record.get("max_images", 0) or 0),
        "topk_frames_per_view": int(candidate_record.get("topk_frames_per_view", 0) or 0),
        "image_ref_count": int(candidate_record.get("image_ref_count", 0) or 0),
        "video_group_key": str(candidate_record.get("video_group_key") or ""),
        "estimated_cost": int(candidate_record.get("estimated_cost", 1) or 1),
        "auto_mode_reason": str(candidate_record.get("auto_mode_reason") or ""),
        "auto_mode_verification_decision": str(candidate_record.get("auto_mode_verification_decision") or ""),
        "auto_mode_selected_window_count": int(candidate_record.get("auto_mode_selected_window_count", 0) or 0),
        "auto_mode_image_count": int(candidate_record.get("auto_mode_image_count", 0) or 0),
    }


def _build_candidate_manifest(
    raw_rows: List[Dict[str, Any]],
    *,
    prepared_metadata: Dict[str, Any],
    requested_mode: str,
    max_images: int,
    topk_frames_per_view: int,
    overwrite_existing: bool,
    proposal_runtime: Any = None,
) -> tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    candidate_manifest: List[Dict[str, Any]] = []
    input_format_summary = {
        "num_input_compact_trace_rows": len(raw_rows),
        "num_expanded_step_rows": 0,
        "num_output_rows": 0,
    }
    candidate_summary = {
        "num_teacher_judge_candidates": 0,
        "num_teacher_judge_skipped_existing": 0,
    }
    for source_row_index, step_row in _iter_compact_trace_step_rows(
        raw_rows,
        prepared_metadata=prepared_metadata,
        proposal_runtime=proposal_runtime,
    ):
        input_format_summary["num_expanded_step_rows"] += 1
        if not is_teacher_judge_candidate(step_row):
            continue
        candidate_summary["num_teacher_judge_candidates"] += 1
        if (not overwrite_existing) and any(
            key in step_row for key in ("teacher_judge_scores", "teacher_judge_decision", "teacher_judge_rationale")
        ):
            candidate_summary["num_teacher_judge_skipped_existing"] += 1
            continue
        candidate_record = build_teacher_judge_package_record(
            step_row,
            source_row_index=source_row_index,
            requested_mode=requested_mode,
            max_images=max_images,
            topk_frames_per_view=topk_frames_per_view,
        )
        candidate_manifest.append(_candidate_manifest_entry(candidate_record))
    input_format_summary["num_output_rows"] = int(input_format_summary["num_expanded_step_rows"])
    return candidate_manifest, input_format_summary, candidate_summary


def _materialize_step_rows_by_source_index(
    raw_rows: List[Dict[str, Any]],
    *,
    prepared_metadata: Dict[str, Any],
    source_row_indices: Iterable[int],
    proposal_runtime: Any = None,
) -> Dict[int, Dict[str, Any]]:
    target_indices = {int(index) for index in source_row_indices}
    if not target_indices:
        return {}
    step_rows_by_source_index: Dict[int, Dict[str, Any]] = {}
    for source_row_index, step_row in _iter_compact_trace_step_rows(
        raw_rows,
        prepared_metadata=prepared_metadata,
        proposal_runtime=proposal_runtime,
    ):
        if source_row_index not in target_indices:
            continue
        step_rows_by_source_index[int(source_row_index)] = copy.deepcopy(step_row)
        if len(step_rows_by_source_index) == len(target_indices):
            break
    missing_indices = sorted(target_indices.difference(step_rows_by_source_index.keys()))
    if missing_indices:
        raise ValueError(
            "Failed to materialize requested teacher-judge step rows. "
            f"missing_source_row_indices={missing_indices[:16]}"
        )
    return step_rows_by_source_index


def _materialize_local_candidate_packages(
    raw_rows: List[Dict[str, Any]],
    *,
    prepared_metadata: Dict[str, Any],
    target_candidates: List[Dict[str, Any]],
    overwrite_existing: bool,
    proposal_runtime: Any = None,
) -> tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    if not target_candidates:
        return [], {}
    target_order = {
        int(candidate.get("source_row_index", -1)): position
        for position, candidate in enumerate(target_candidates)
    }
    step_rows_by_source_index = _materialize_step_rows_by_source_index(
        raw_rows,
        prepared_metadata=prepared_metadata,
        source_row_indices=target_order.keys(),
        proposal_runtime=proposal_runtime,
    )
    package_records: List[Dict[str, Any]] = []
    for source_row_index in sorted(step_rows_by_source_index.keys(), key=lambda value: target_order[value]):
        step_row = step_rows_by_source_index[int(source_row_index)]
        if not is_teacher_judge_candidate(step_row):
            raise ValueError(
                "Shard plan assigned a non-candidate teacher-judge step row. "
                f"source_row_index={source_row_index} video_id={step_row.get('video_id')}"
            )
        if (not overwrite_existing) and any(
            key in step_row for key in ("teacher_judge_scores", "teacher_judge_decision", "teacher_judge_rationale")
        ):
            raise ValueError(
                "Shard plan assigned a teacher-judge example that already has labels. "
                f"source_row_index={source_row_index} video_id={step_row.get('video_id')}"
            )
        manifest_record = target_candidates[target_order[source_row_index]]
        package_records.append(
            build_teacher_judge_package_record(
                step_row,
                source_row_index=int(source_row_index),
                requested_mode=str(manifest_record.get("requested_input_mode") or "auto"),
                max_images=int(manifest_record.get("max_images", 0) or 0),
                topk_frames_per_view=int(manifest_record.get("topk_frames_per_view", 0) or 0),
            )
        )
    return package_records, step_rows_by_source_index


def _broadcast_from_main_process(payload: Any, *, runtime, dist_initialized: bool) -> Any:
    if not (runtime.is_distributed and dist_initialized):
        return payload
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Torch distributed is expected to be initialized before broadcasting teacher-judge metadata.")
    object_list = [payload if runtime.is_main_process else None]
    dist.broadcast_object_list(object_list, src=0)
    if object_list[0] is None:
        raise RuntimeError("Broadcasted teacher-judge metadata payload is empty.")
    return object_list[0]


def _apply_teacher_judgments_to_compact_trace_rows(
    raw_rows: List[Dict[str, Any]],
    step_rows_by_source_index: Dict[int, Dict[str, Any]],
    judgments: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    merged_rows = [copy.deepcopy(row) for row in raw_rows]
    num_overrides = 0
    for judgment in judgments:
        row_index = int(judgment.get("source_row_index", -1))
        if row_index < 0:
            raise ValueError(f"Teacher judge returned an invalid source_row_index: {row_index}")
        if row_index not in step_rows_by_source_index:
            raise ValueError(f"Teacher judge returned a source_row_index that was not materialized: {row_index}")
        step_row = copy.deepcopy(step_rows_by_source_index[row_index])
        compact_source_row_index = int(step_row.get("_compact_source_row_index", -1))
        compact_step_index = int(step_row.get("_compact_trace_step_index", step_row.get("step_index", -1)))
        if compact_source_row_index < 0 or compact_source_row_index >= len(merged_rows):
            raise ValueError(f"Teacher judge compact source row index is out of range: {compact_source_row_index}")
        trajectory = list(merged_rows[compact_source_row_index].get("oracle_trajectory") or [])
        if compact_step_index <= 0 or compact_step_index > len(trajectory):
            raise ValueError(
                f"Teacher judge compact step index is out of range: {compact_step_index} "
                f"for video_id={merged_rows[compact_source_row_index].get('video_id')}"
            )
        step_row.update({key: value for key, value in judgment.items() if key != "source_row_index"})
        override_payload = _build_teacher_override_verify_payload(step_row)
        if not override_payload:
            continue
        override_payload["teacher_judge_scores"] = copy.deepcopy(step_row.get("teacher_judge_scores") or {})
        override_payload["teacher_judge_decision"] = str(step_row.get("teacher_judge_decision") or "")
        override_payload["teacher_judge_rationale"] = str(step_row.get("teacher_judge_rationale") or "")
        trajectory[compact_step_index - 1]["oracle_verifier_feedback"] = override_payload
        merged_rows[compact_source_row_index]["oracle_trajectory"] = trajectory
        num_overrides += 1
    return merged_rows, {"num_teacher_override_examples": int(num_overrides)}


def _convert_step_rows_to_episode_rows(
    rows: List[Dict[str, Any]],
    *,
    prepared_metadata: Dict[str, Any],
    runtime,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    config = config_from_prepared_sft_metadata(prepared_metadata)
    episode_rows, summary = convert_step_examples_to_episode_records(
        rows,
        config=config,
    )
    renamed_summary = {
        "num_output_episode_records": int(summary.get("num_episode_records", 0)),
        "num_output_episode_upgraded_step_groups": int(summary.get("num_upgraded_step_groups", 0)),
        "num_output_passthrough_episode_records": int(summary.get("num_passthrough_episode_records", 0)),
    }
    runtime_log(
        (
            "converted teacher-judge output back to episode-format: "
            f"episode_records={renamed_summary['num_output_episode_records']} "
            f"upgraded_step_groups={renamed_summary['num_output_episode_upgraded_step_groups']}"
        ),
        runtime=runtime,
    )
    return episode_rows, renamed_summary


def _collate_package_batch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


def _log_teacher_judge_stage(message: str, *, runtime) -> None:
    runtime_log(message, runtime=runtime)


def _resolve_teacher_judge_shard_indices(
    rows: List[Dict[str, Any]],
    *,
    num_shards: int,
) -> List[List[int]]:
    if int(num_shards) < 1:
        raise ValueError("num_shards must be at least 1.")
    shard_indices_by_shard: List[List[int]] = [[] for _ in range(int(num_shards))]
    verify_candidate_index = 0
    for row_index, row in enumerate(rows):
        if is_teacher_judge_candidate(row):
            assigned_shard = verify_candidate_index % int(num_shards)
            verify_candidate_index += 1
        else:
            assigned_shard = row_index % int(num_shards)
        shard_indices_by_shard[int(assigned_shard)].append(int(row_index))
    return shard_indices_by_shard


def _expected_shard_indices(
    *,
    total_rows: int,
    num_shards: int,
    shard_index: int,
    shard_indices_by_shard: Optional[List[List[int]]] = None,
) -> List[int]:
    if shard_indices_by_shard is not None:
        if not 0 <= int(shard_index) < len(shard_indices_by_shard):
            raise ValueError(
                f"shard_index={shard_index} is outside the provided shard mapping range "
                f"[0, {len(shard_indices_by_shard) - 1}]."
            )
        return list(shard_indices_by_shard[int(shard_index)])
    return list(range(int(shard_index), int(total_rows), int(num_shards)))


def _merge_sharded_outputs(
    output_path: str | Path,
    *,
    total_rows: int,
    num_shards: int,
    shard_indices_by_shard: Optional[List[List[int]]] = None,
) -> List[Dict[str, Any]]:
    base_output_path = Path(output_path)
    if int(num_shards) <= 1:
        return _load_jsonl(base_output_path)
    merged_rows: List[Optional[Dict[str, Any]]] = [None] * int(total_rows)
    for shard_index in range(int(num_shards)):
        shard_output_path = sharded_output_path(base_output_path, num_shards=int(num_shards), shard_index=shard_index)
        shard_rows = _load_jsonl(shard_output_path)
        expected_indices = _expected_shard_indices(
            total_rows=int(total_rows),
            num_shards=int(num_shards),
            shard_index=shard_index,
            shard_indices_by_shard=shard_indices_by_shard,
        )
        if len(shard_rows) != len(expected_indices):
            raise ValueError(
                "Shard output row count does not match expected shard partition size. "
                f"shard={shard_index + 1}/{num_shards} expected={len(expected_indices)} actual={len(shard_rows)} "
                f"path={shard_output_path}"
            )
        for source_index, row in zip(expected_indices, shard_rows):
            merged_rows[source_index] = row
    if any(row is None for row in merged_rows):
        raise ValueError("Merged teacher-judge output is incomplete after combining shard files.")
    finalized_rows = [row for row in merged_rows if row is not None]
    _write_jsonl(base_output_path, finalized_rows)
    return finalized_rows


def _wait_for_sharded_outputs(
    output_path: str | Path,
    *,
    total_rows: int,
    num_shards: int,
    shard_indices_by_shard: Optional[List[List[int]]] = None,
    timeout_sec: float = 3600.0,
    poll_interval_sec: float = 1.0,
) -> None:
    base_output_path = Path(output_path)
    deadline = time.time() + max(1.0, float(timeout_sec))
    shard_status: Dict[str, str] = {}
    while time.time() < deadline:
        all_ready = True
        shard_status.clear()
        for shard_index in range(int(num_shards)):
            shard_output_path = sharded_output_path(base_output_path, num_shards=int(num_shards), shard_index=shard_index)
            expected_indices = _expected_shard_indices(
                total_rows=int(total_rows),
                num_shards=int(num_shards),
                shard_index=shard_index,
                shard_indices_by_shard=shard_indices_by_shard,
            )
            expected_count = len(expected_indices)
            if not shard_output_path.exists():
                all_ready = False
                shard_status[str(shard_output_path)] = "missing"
                continue
            try:
                shard_rows = _load_jsonl(shard_output_path)
            except Exception as exc:
                all_ready = False
                shard_status[str(shard_output_path)] = f"unreadable: {exc}"
                continue
            if len(shard_rows) != expected_count:
                all_ready = False
                shard_status[str(shard_output_path)] = f"rows={len(shard_rows)}/{expected_count}"
                continue
            shard_status[str(shard_output_path)] = "ready"
        if all_ready:
            return
        time.sleep(max(0.05, float(poll_interval_sec)))
    raise TimeoutError(
        "Timed out while waiting for sharded teacher-judge outputs to become ready: "
        + json.dumps(shard_status, ensure_ascii=False)
    )


def _materialize_teacher_rollout_primary_rows(
    rows: List[Dict[str, Any]],
    *,
    prepared_metadata: Dict[str, Any],
    runtime,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    config = config_from_prepared_sft_metadata(prepared_metadata)
    rebuilt_rows, summary = rebuild_teacher_rollout_primary_examples(
        rows,
        config=config,
    )
    if int(summary.get("num_teacher_rollout_records", 0)) > 0:
        runtime_log(
            (
                "teacher-rollout-primary materialization: "
                f"records={summary['num_teacher_rollout_records']} "
                f"overrides={summary['num_teacher_override_examples']} "
                f"synthetic_finalize={summary['num_synthetic_finalize_examples']}"
            ),
            runtime=runtime,
        )
    return rebuilt_rows, summary


def _input_signature(path: Path, metadata: Dict[str, Any]) -> str:
    stat = path.stat()
    payload = {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "metadata": metadata,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _apply_aux_teacher_protocol_signature(
    rows: List[Dict[str, Any]],
    *,
    prepared_metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    signature = build_protocol_signature(
        config=config_from_prepared_sft_metadata(prepared_metadata),
        teacher_role=TEACHER_ROLE_AUXILIARY,
    )
    patched_rows: List[Dict[str, Any]] = []
    for row in rows:
        patched = copy.deepcopy(row)
        patched["protocol_signature"] = copy.deepcopy(signature)
        patched_rows.append(patched)
    return patched_rows


def _build_candidate_records(
    rows: List[Dict[str, Any]],
    *,
    requested_mode: str,
    max_images: int,
    topk_frames_per_view: int,
    overwrite_existing: bool,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    candidate_records: List[Dict[str, Any]] = []
    summary = {
        "num_teacher_judge_candidates": 0,
        "num_teacher_judge_skipped_existing": 0,
    }
    for row_index, row in enumerate(rows):
        if not is_teacher_judge_candidate(row):
            continue
        summary["num_teacher_judge_candidates"] += 1
        if (not overwrite_existing) and any(key in row for key in ("teacher_judge_scores", "teacher_judge_decision", "teacher_judge_rationale")):
            summary["num_teacher_judge_skipped_existing"] += 1
            continue
        candidate_records.append(
            build_teacher_judge_package_record(
                row,
                source_row_index=row_index,
                requested_mode=requested_mode,
                max_images=max_images,
                topk_frames_per_view=topk_frames_per_view,
            )
        )
    return candidate_records, summary


def _assign_package_indices_by_video_group(
    package_records: List[Dict[str, Any]],
    *,
    num_shards: int,
) -> List[List[int]]:
    if int(num_shards) < 1:
        raise ValueError("num_shards must be at least 1.")
    grouped: Dict[str, List[int]] = {}
    group_costs: Dict[str, int] = {}
    for package_index, record in enumerate(package_records):
        group_key = str(record.get("video_group_key") or f"package::{package_index}")
        grouped.setdefault(group_key, []).append(package_index)
        group_costs[group_key] = group_costs.get(group_key, 0) + int(record.get("estimated_cost", 1) or 1)
    shard_indices: List[List[int]] = [[] for _ in range(int(num_shards))]
    shard_costs: List[int] = [0 for _ in range(int(num_shards))]
    for group_key in sorted(grouped.keys(), key=lambda key: (-group_costs[key], key)):
        shard_index = min(range(int(num_shards)), key=lambda index: (shard_costs[index], index))
        shard_indices[shard_index].extend(grouped[group_key])
        shard_costs[shard_index] += int(group_costs[group_key])
    for indices in shard_indices:
        indices.sort()
    return shard_indices


def _wait_for_sharded_judgment_outputs(
    output_path: str | Path,
    *,
    num_shards: int,
    shard_indices_by_shard: List[List[int]],
    timeout_sec: float = 3600.0,
    poll_interval_sec: float = 1.0,
) -> None:
    base_output_path = Path(output_path)
    deadline = time.time() + max(1.0, float(timeout_sec))
    shard_status: Dict[str, str] = {}
    while time.time() < deadline:
        all_ready = True
        shard_status.clear()
        for shard_index in range(int(num_shards)):
            shard_output_path = sharded_output_path(base_output_path, num_shards=int(num_shards), shard_index=shard_index)
            expected_count = len(shard_indices_by_shard[shard_index])
            if not shard_output_path.exists():
                all_ready = False
                shard_status[str(shard_output_path)] = "missing"
                continue
            try:
                shard_rows = _load_jsonl(shard_output_path)
            except Exception as exc:
                all_ready = False
                shard_status[str(shard_output_path)] = f"unreadable: {exc}"
                continue
            if len(shard_rows) != expected_count:
                all_ready = False
                shard_status[str(shard_output_path)] = f"rows={len(shard_rows)}/{expected_count}"
                continue
            shard_status[str(shard_output_path)] = "ready"
        if all_ready:
            return
        time.sleep(max(0.05, float(poll_interval_sec)))
    raise TimeoutError(
        "Timed out while waiting for sharded teacher-judge outputs to become ready: "
        + json.dumps(shard_status, ensure_ascii=False)
    )


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    runtime = distributed_runtime_from_env()
    dist_initialized = init_torch_distributed(runtime)
    shard_spec = resolve_shard_spec(num_shards=args.num_shards, shard_index=args.shard_index, runtime=runtime)
    output_path = sharded_output_path(args.output, num_shards=shard_spec.num_shards, shard_index=shard_spec.shard_index)
    input_path = Path(args.input)
    input_metadata = ensure_prepared_sft_metadata(input_path)
    effective_source_prepared_splits = parse_include_splits(args.include_splits) or list(input_metadata.get("include_splits") or [])
    raw_rows = _load_jsonl(
        input_path,
        skip_invalid_lines=args.skip_invalid_jsonl_lines,
        include_splits=args.include_splits,
    )
    if raw_rows and not all(is_compact_trace_sft_record(row) for row in raw_rows):
        raise ValueError(
            "annotate_teacher_judge_sft.py now only accepts compact_trace_v4 prepared rows. "
            "Regenerate the prepared SFT JSONL with build_saver_data.py and rerun teacher judge."
        )
    strict_feature_guided_proposal = _prepared_rows_require_feature_guided_proposal(raw_rows)
    if strict_feature_guided_proposal and not str(args.proposal_model_path or "").strip():
        raise ValueError(
            "annotate_teacher_judge_sft.py requires --proposal-model-path because compact_trace replay includes "
            "seek_evidence and strict feature-guided proposal is enabled."
        )
    runtime_log(
        (
            f"teacher judge preparing shard plan: compact_rows={len(raw_rows)} "
            f"num_shards={shard_spec.num_shards} input_mode={args.input_mode}"
        ),
        runtime=runtime,
    )
    proposal_runtime = _load_proposal_runtime(args, runtime=runtime)
    if runtime.is_main_process or not (runtime.is_distributed and dist_initialized):
        candidate_manifest, input_format_summary, candidate_summary = _build_candidate_manifest(
            raw_rows,
            prepared_metadata=input_metadata,
            requested_mode=args.input_mode,
            max_images=args.max_images,
            topk_frames_per_view=args.topk_frames_per_view,
            overwrite_existing=args.overwrite_existing,
            proposal_runtime=proposal_runtime,
        )
    else:
        candidate_manifest = []
        input_format_summary = {}
        candidate_summary = {}
    candidate_manifest = _broadcast_from_main_process(
        candidate_manifest,
        runtime=runtime,
        dist_initialized=dist_initialized,
    )
    input_format_summary = _broadcast_from_main_process(
        input_format_summary,
        runtime=runtime,
        dist_initialized=dist_initialized,
    )
    candidate_summary = _broadcast_from_main_process(
        candidate_summary,
        runtime=runtime,
        dist_initialized=dist_initialized,
    )
    shard_indices_by_shard = _assign_package_indices_by_video_group(candidate_manifest, num_shards=shard_spec.num_shards)
    local_package_indices = _expected_shard_indices(
        total_rows=len(candidate_manifest),
        num_shards=shard_spec.num_shards,
        shard_index=shard_spec.shard_index,
        shard_indices_by_shard=shard_indices_by_shard,
    )
    local_target_candidates = [candidate_manifest[index] for index in local_package_indices]
    local_package_records, local_step_rows_by_source_index = _materialize_local_candidate_packages(
        raw_rows,
        prepared_metadata=input_metadata,
        target_candidates=local_target_candidates,
        overwrite_existing=args.overwrite_existing,
        proposal_runtime=proposal_runtime,
    )
    effective_device_map = resolve_inference_device_map(args.device_map, runtime=runtime)
    runtime_log(
        (
            f"teacher judge startup: expanded_rows={int(input_format_summary.get('num_output_rows', 0))} "
            f"total_candidates={len(candidate_manifest)} "
            f"local_candidates={len(local_package_records)} batch_size={args.batch_size} "
            f"input_mode={args.input_mode} model_path={args.model_path} output={output_path}"
        ),
        runtime=runtime,
    )
    progress_bar = (
        create_progress_bar(
            total=len(local_package_records),
            desc=f"rank{runtime.rank} judge",
            runtime=runtime,
            unit="example",
            leave=True,
        )
        if not args.no_progress_bar
        else None
    )
    summary: Dict[str, Any] = {}
    reweight_summary: Dict[str, Any] = {}
    materialize_summary: Dict[str, Any] = {}
    output_episode_summary: Dict[str, Any] = {}
    merged_output_path: Optional[Path] = None
    try:
        image_resolver = None
        if str(args.input_mode or "").strip().lower() != "text_only":
            image_resolver = _FrameReferenceResolver(
                max_cached_videos=args.frame_cache_max_cached_videos
            )._resolve_image_ref
        _log_teacher_judge_stage(
            f"teacher judge loading model: model_path={args.model_path} device_map={effective_device_map}",
            runtime=runtime,
        )
        judge = QwenTeacherJudge.from_pretrained(
            args.model_path,
            torch_dtype=args.torch_dtype,
            device_map=effective_device_map,
            attn_implementation=args.attn_implementation or None,
            input_mode=args.input_mode,
            max_new_tokens=args.max_new_tokens,
            max_images=args.max_images,
            topk_frames_per_view=args.topk_frames_per_view,
            image_resolver=image_resolver,
            log_fn=None,
        )
        _log_teacher_judge_stage("teacher judge model loaded", runtime=runtime)
        local_judgments: List[Dict[str, Any]] = []
        total_batches = (len(local_package_records) + max(1, int(args.batch_size)) - 1) // max(1, int(args.batch_size))
        for batch_index in range(1, total_batches + 1):
            start = (batch_index - 1) * max(1, int(args.batch_size))
            batch_records = local_package_records[start : start + max(1, int(args.batch_size))]
            batch_examples = [
                copy.deepcopy(local_step_rows_by_source_index[int(record.get("source_row_index", -1))])
                for record in batch_records
            ]
            annotated_examples = judge.annotate_examples(batch_examples, input_mode=args.input_mode)
            if len(annotated_examples) != len(batch_examples):
                raise ValueError(
                    "Teacher judge batch returned a different number of results than requested. "
                    f"requested={len(batch_examples)} returned={len(annotated_examples)}"
                )
            for package_record, annotated_example in zip(batch_records, annotated_examples):
                local_judgments.append(
                    {
                        "source_row_index": int(package_record.get("source_row_index", -1)),
                        "teacher_judge_scores": copy.deepcopy(annotated_example.get("teacher_judge_scores") or {}),
                        "teacher_judge_decision": str(annotated_example.get("teacher_judge_decision") or ""),
                        "teacher_judge_rationale": str(annotated_example.get("teacher_judge_rationale") or ""),
                    }
                )
            if progress_bar is not None:
                progress_bar.update(len(batch_records))
                progress_bar.set_postfix_str(
                    f"annotated={len(local_judgments)}/{len(local_package_records)}",
                    refresh=False,
                )
        summary = {
            "num_examples": int(input_format_summary.get("num_output_rows", 0)),
            **candidate_summary,
            "num_teacher_judge_annotated": len(local_judgments),
        }
        local_output_rows = local_judgments
        if not shard_spec.is_sharded:
            merged_rows, materialize_summary = _apply_teacher_judgments_to_compact_trace_rows(
                raw_rows,
                local_step_rows_by_source_index,
                local_judgments,
            )
            local_output_rows = _apply_aux_teacher_protocol_signature(
                merged_rows,
                prepared_metadata=input_metadata,
            )
        _write_jsonl(output_path, local_output_rows)
        runtime_log(f"saved {len(local_output_rows)} teacher-judge records to {output_path}", runtime=runtime)
        if not shard_spec.is_sharded:
            metadata_path = prepared_sft_metadata_path(output_path)
            metadata = dict(input_metadata)
            metadata["num_records"] = int(len(local_output_rows))
            metadata["include_splits"] = sorted({str(item).strip() for item in effective_source_prepared_splits if str(item).strip()})
            metadata["source_prepared"] = build_jsonl_provenance(
                input_path,
                include_splits=effective_source_prepared_splits,
            )
            metadata["teacher_annotated"] = True
            metadata["teacher_rollout_primary_materialized"] = True
            metadata["protocol_signature"] = build_protocol_signature(
                config=config_from_prepared_sft_metadata(metadata),
                teacher_role=TEACHER_ROLE_AUXILIARY,
            )
            metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if shard_spec.is_sharded and runtime.is_distributed and dist_initialized:
            if runtime.is_main_process:
                _wait_for_sharded_judgment_outputs(
                    args.output,
                    num_shards=shard_spec.num_shards,
                    shard_indices_by_shard=shard_indices_by_shard,
                )
                merged_judgments: List[Dict[str, Any]] = []
                for shard_index in range(int(shard_spec.num_shards)):
                    shard_output_path = sharded_output_path(args.output, num_shards=int(shard_spec.num_shards), shard_index=shard_index)
                    merged_judgments.extend(_load_jsonl(shard_output_path))
                merged_step_rows_by_source_index = _materialize_step_rows_by_source_index(
                    raw_rows,
                    prepared_metadata=input_metadata,
                    source_row_indices=[
                        int(judgment.get("source_row_index", -1))
                        for judgment in merged_judgments
                    ],
                    proposal_runtime=proposal_runtime,
                )
                merged_rows, materialize_summary = _apply_teacher_judgments_to_compact_trace_rows(
                    raw_rows,
                    merged_step_rows_by_source_index,
                    merged_judgments,
                )
                merged_rows = _apply_aux_teacher_protocol_signature(
                    merged_rows,
                    prepared_metadata=input_metadata,
                )
                _write_jsonl(args.output, merged_rows)
                merged_output_path = Path(args.output)
                runtime_log(
                    f"merged {len(merged_rows)} sharded teacher-judge outputs into {merged_output_path}",
                    runtime=runtime,
                )
                merged_metadata = dict(input_metadata)
                merged_metadata["num_records"] = int(len(merged_rows))
                merged_metadata["include_splits"] = sorted(
                    {str(item).strip() for item in effective_source_prepared_splits if str(item).strip()}
                )
                merged_metadata["source_prepared"] = build_jsonl_provenance(
                    input_path,
                    include_splits=effective_source_prepared_splits,
                )
                merged_metadata["teacher_annotated"] = True
                merged_metadata["teacher_rollout_primary_materialized"] = True
                merged_metadata["protocol_signature"] = build_protocol_signature(
                    config=config_from_prepared_sft_metadata(merged_metadata),
                    teacher_role=TEACHER_ROLE_AUXILIARY,
                )
                prepared_sft_metadata_path(merged_output_path).write_text(
                    json.dumps(merged_metadata, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
    finally:
        if progress_bar is not None:
            progress_bar.close()
    if runtime.is_main_process:
        print(
            json.dumps(
                {
                    "input": str(input_path),
                    "output": str(merged_output_path or output_path),
                    "local_output": str(output_path),
                    "num_examples": int(input_format_summary.get("num_output_rows", 0)),
                    "num_local_candidates": len(local_package_records),
                    **input_format_summary,
                    **summary,
                    **materialize_summary,
                    **reweight_summary,
                    **output_episode_summary,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
