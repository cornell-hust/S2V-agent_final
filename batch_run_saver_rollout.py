#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from split_utils import parse_include_splits

from run_saver_rollout import _serialize_result
from saver_v3.core.adapter import TimeSearchRolloutAdapter
from saver_v3.data.config import (
    DEFAULT_POLICY_MAX_NEW_TOKENS,
    DEFAULT_ROLLOUT_MAX_TURNS,
    DEFAULT_RECOMMENDED_KEEP_RECENT_TEXT_MESSAGES,
    DEFAULT_RECOMMENDED_MAX_SEQ_LENGTH,
    DEFAULT_RECOMMENDED_MAX_TOTAL_IMAGES,
    DEFAULT_TOTAL_VISUAL_BUDGET,
    InitialObservationConfig,
    PromptConfig,
    PreviewConfig,
    RolloutTraceConfig,
    SaverAgentConfig,
)
from saver_v3.data.dataset import SaverAgentDataset
from saver_v3.common.experiment_logging import resolve_experiment_log_dir, utc_timestamp, write_json
from saver_v3.metrics.offline_scoring import load_rollout_records
from saver_v3.core.proposal import SiglipFeatureEncoder
from saver_v3.model.qwen_policy import DEFAULT_MODEL_PATH, QwenGenerationPolicy
from saver_v3.common.runtime import (
    claim_next_dynamic_task_index,
    distributed_runtime_from_env,
    initialize_dynamic_task_queue,
    record_dynamic_task_completion,
    resolve_inference_device_map,
    resolve_shard_spec,
    runtime_log,
    shard_sequence,
    sharded_output_path,
    should_log_progress,
)
from saver_v3.core.rollout import ReplayPolicy, SaverRolloutRunner
from saver_v3.model.vllm_generation import (
    VllmQwenGenerationPolicy,
    build_vllm_policy_from_model_path,
    create_vllm_runtime,
)


class _StoreMaxTotalImages(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, "_max_total_images_explicit", True)


def _parse_bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean flag value, got: {value}")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-run SAVER rollouts over a dataset slice.")
    parser.set_defaults(_max_total_images_explicit=False)
    parser.add_argument("--data", required=True, help="Path to saver_agent JSONL data.")
    parser.add_argument("--data-root", default="", help="Root path used to resolve relative video paths.")
    parser.add_argument("--include-splits", default="", help="Optional comma-separated split whitelist for --data.")
    parser.add_argument(
        "--indices",
        default="",
        help="Optional explicit dataset indices, e.g. '0,1,5-7'. Overrides --start-index/--count.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start dataset index for batch rollout.")
    parser.add_argument(
        "--count",
        type=int,
        default=0,
        help="Number of samples to roll out from start-index. Use 0 to run until the end of the filtered dataset.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_ROLLOUT_MAX_TURNS,
        help="Maximum rollout turns per sample.",
    )
    parser.add_argument(
        "--policy-backend",
        choices=["replay", "qwen"],
        default="replay",
        help="Use replayed responses or real Qwen generation.",
    )
    parser.add_argument(
        "--response",
        action="append",
        default=[],
        help="Replayed model response for one turn. Reused for every sample when policy-backend=replay.",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local Qwen model path.")
    parser.add_argument("--proposal-model-path", default="", help="Optional local SigLIP/CLIP path for feature-guided proposal.")
    parser.add_argument("--proposal-torch-dtype", default="auto", help="Torch dtype for the proposal encoder.")
    parser.add_argument("--proposal-device", default="", help="Optional device for the proposal encoder. Defaults to cpu or current local cuda device.")
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype passed to from_pretrained.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map argument.")
    parser.add_argument("--attn-implementation", default="", help="Optional attention backend.")
    parser.add_argument(
        "--use-vllm",
        type=_parse_bool_flag,
        default=True,
        help="Use the shared vLLM runtime for Qwen rollout generation. Defaults to true.",
    )
    parser.add_argument(
        "--vllm-mode",
        choices=["colocate", "server"],
        default="colocate",
        help="vLLM execution mode for rollout generation.",
    )
    parser.add_argument(
        "--vllm-tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for colocated vLLM workers.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=0.35,
        help="GPU memory utilization target for colocated vLLM workers.",
    )
    parser.add_argument(
        "--vllm-guided-decoding-regex",
        default="",
        help="Optional guided decoding regex passed through the shared vLLM runtime.",
    )
    parser.add_argument("--vllm-server-host", default="127.0.0.1", help="Optional vLLM server host.")
    parser.add_argument("--vllm-server-port", type=int, default=8000, help="Optional vLLM server port.")
    parser.add_argument(
        "--vllm-server-timeout",
        type=float,
        default=240.0,
        help="Connection timeout in seconds for vLLM server mode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_POLICY_MAX_NEW_TOKENS,
        help="Generation length for Qwen policy.",
    )
    parser.add_argument(
        "--total-visual-budget",
        type=int,
        default=0,
        help="Alias for a coarse visual budget. Resolved as --max-total-images when the latter is unset.",
    )
    parser.add_argument(
        "--max-total-images",
        type=int,
        default=DEFAULT_RECOMMENDED_MAX_TOTAL_IMAGES,
        action=_StoreMaxTotalImages,
        help="Optional hard cap on total images preserved in the rollout prompt. 0 keeps all images.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_RECOMMENDED_MAX_SEQ_LENGTH,
        help="Optional tokenizer/processor max_length for rollout prompts. 0 disables prompt fitting.",
    )
    parser.add_argument(
        "--keep-recent-text-messages",
        type=int,
        default=DEFAULT_RECOMMENDED_KEEP_RECENT_TEXT_MESSAGES,
        help="If >0, keep full text only for the N most recent non-initial history messages in rollout prompts.",
    )
    parser.add_argument(
        "--keep-recent-tool-image-messages",
        type=int,
        default=0,
        help="If >0, keep the most recent N tool image messages in rollout prompts.",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=0,
        help="Optional rollout-time max image side length in pixels. 0 disables resizing.",
    )
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=0,
        help="Optional rollout-time max image area in pixels. 0 disables resizing.",
    )
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling for Qwen policy.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Sampling top-p.")
    parser.add_argument("--top-k", type=int, default=None, help="Sampling top-k.")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Optional repetition penalty.")
    parser.add_argument(
        "--initial-observation-mode",
        choices=["preview", "explicit_first_scan"],
        default="explicit_first_scan",
        help="Canonical initial observation mode for SEEK mainline.",
    )
    parser.add_argument(
        "--initial-scan-num-frames",
        type=int,
        default=8,
        help='Number of frames retained by the canonical first scan_timeline call when initial-observation-mode=explicit_first_scan.',
    )
    parser.add_argument(
        "--protect-initial-scan-from-visual-budget",
        type=_parse_bool_flag,
        default=True,
        help="Keep the canonical first global scan exempt from visual-budget pruning.",
    )
    parser.add_argument(
        "--error-on-initial-scan-seq-prune",
        type=_parse_bool_flag,
        default=True,
        help="Raise an explicit error if max_seq_length fitting would need to prune the canonical first global scan.",
    )
    parser.add_argument(
        "--num-preview-frames",
        type=int,
        default=8,
        help="Legacy preview frame count. Used only when --initial-observation-mode=preview.",
    )
    parser.add_argument(
        "--preview-sampling-fps",
        type=float,
        default=None,
        help="Legacy preview sampling fps. Used only when --initial-observation-mode=preview.",
    )
    parser.add_argument("--initial-user-template", default="", help="Optional custom template for the first user prompt.")
    parser.add_argument("--preview-instruction", default="", help="Optional custom preview instruction.")
    parser.add_argument("--tool-response-template", default="", help="Optional custom tool follow-up prompt template.")
    parser.add_argument("--record-observation-content", action="store_true", help="Store full tool observation content.")
    parser.add_argument(
        "--no-record-message-history",
        action="store_true",
        help="Disable storing the full message history in the rollout output.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Batch rollout output path (.jsonl, .json, or directory).",
    )
    parser.add_argument(
        "--rollout-batch-size",
        type=int,
        default=4,
        help="How many videos to batch together per rollout generation step.",
    )
    parser.add_argument("--log-dir", default="", help="Optional directory for batch rollout logs.")
    parser.add_argument("--num-shards", type=int, default=0, help="Optional number of shard workers.")
    parser.add_argument("--shard-index", type=int, default=-1, help="Optional shard index for this process.")
    parser.add_argument("--progress-every", type=int, default=1, help="Log rollout progress every N local samples.")
    return parser.parse_args(argv)


def _resolve_max_total_images(args: argparse.Namespace) -> int:
    explicit_max_total_images = int(getattr(args, "max_total_images", 0) or 0)
    if bool(getattr(args, "_max_total_images_explicit", False)):
        return max(0, explicit_max_total_images)
    alias_max_total_images = max(0, int(getattr(args, "total_visual_budget", 0) or 0))
    if alias_max_total_images > 0:
        return alias_max_total_images
    return max(0, explicit_max_total_images)


def _build_config(args: argparse.Namespace) -> SaverAgentConfig:
    return SaverAgentConfig(
        preview=PreviewConfig(
            num_preview_frames=args.num_preview_frames,
            preview_sampling_fps=args.preview_sampling_fps,
            max_preview_frames=args.num_preview_frames,
        ),
        initial_observation=InitialObservationConfig(
            mode=str(args.initial_observation_mode or "explicit_first_scan"),
            scan_num_frames=max(1, int(args.initial_scan_num_frames or 8)),
            scan_purpose="global_overview",
            protect_from_visual_budget=bool(args.protect_initial_scan_from_visual_budget),
            error_on_seq_prune=bool(args.error_on_initial_scan_seq_prune),
        ),
        prompt=PromptConfig(
            initial_user_template=args.initial_user_template or PromptConfig().initial_user_template,
            preview_instruction=args.preview_instruction or PromptConfig().preview_instruction,
            tool_response_template=args.tool_response_template or PromptConfig().tool_response_template,
        ),
        rollout_trace=RolloutTraceConfig(
            record_observation_content=args.record_observation_content,
            record_state_deltas=True,
            record_message_history=not args.no_record_message_history,
        ),
    )


def _parse_indices(indices_text: str) -> List[int]:
    values: List[int] = []
    if not indices_text.strip():
        return values
    for chunk in indices_text.split(","):
        token = chunk.strip()
        if not token:
            continue
        range_match = re.fullmatch(r"(-?\d+)\s*-\s*(-?\d+)", token)
        if range_match:
            start = int(range_match.group(1))
            end = int(range_match.group(2))
            step = 1 if end >= start else -1
            values.extend(list(range(start, end + step, step)))
            continue
        values.append(int(token))
    return values


def _resolve_dataset_indices(args: argparse.Namespace, dataset_size: int) -> List[int]:
    if args.indices:
        indices = _parse_indices(args.indices)
    else:
        if args.start_index < 0 or args.start_index >= dataset_size:
            raise SystemExit(
                f"Dataset index out of range: {args.start_index}. Valid range is [0, {max(dataset_size - 1, 0)}]."
            )
        if args.count < 0:
            raise SystemExit("Provide either --indices or a non-negative --count for batch rollout.")
        if args.count == 0:
            indices = list(range(args.start_index, dataset_size))
        else:
            indices = list(range(args.start_index, args.start_index + args.count))

    if not indices:
        raise SystemExit("No dataset indices were resolved for batch rollout.")
    invalid = [index for index in indices if index < 0 or index >= dataset_size]
    if invalid:
        raise SystemExit(
            f"Dataset index out of range: {invalid[0]}. Valid range is [0, {max(dataset_size - 1, 0)}]."
        )
    return indices


def _record_requires_feature_guided_proposal(record: Dict[str, Any]) -> bool:
    allowed_tools = list(((record.get("tool_io") or {}).get("allowed_tools") or []))
    if any(str(tool_name or "").strip() == "seek_evidence" for tool_name in allowed_tools):
        return True
    for step in list(record.get("oracle_trajectory") or []):
        if str(step.get("tool") or "").strip() == "seek_evidence":
            return True
    return False


def _records_require_feature_guided_proposal(records: List[Dict[str, Any]]) -> bool:
    return any(_record_requires_feature_guided_proposal(record) for record in list(records or []))


def _attach_proposal_context(
    item: Dict[str, Any],
    *,
    proposal_runtime: Any,
    strict_feature_guided_proposal: bool,
) -> None:
    if bool(strict_feature_guided_proposal):
        item["multimodal_cache"]["strict_feature_guided_proposal"] = True
    if proposal_runtime is not None:
        item["multimodal_cache"]["proposal_runtime"] = proposal_runtime


def _build_qwen_policy(args: argparse.Namespace, *, runtime: Any) -> QwenGenerationPolicy:
    if bool(getattr(args, "use_vllm", True)):
        policy = build_vllm_policy_from_model_path(
            args=args,
            runtime=runtime,
            model_path=args.model_path,
            prefer_direct_local_rank_runtime=True,
            max_new_tokens=args.max_new_tokens,
            max_total_images=_resolve_max_total_images(args),
            max_seq_length=args.max_seq_length,
            keep_recent_tool_image_messages=args.keep_recent_tool_image_messages,
            keep_recent_text_messages=args.keep_recent_text_messages,
            max_image_side=args.max_image_side,
            max_image_pixels=args.max_image_pixels,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            use_generation_cache=True,
            step_resolver=lambda: 0,
            policy_class=VllmQwenGenerationPolicy,
        )
        return policy
    return QwenGenerationPolicy.from_pretrained(
        args.model_path,
        torch_dtype=args.torch_dtype,
        device_map=resolve_inference_device_map(args.device_map, runtime=runtime),
        attn_implementation=args.attn_implementation or None,
        max_new_tokens=args.max_new_tokens,
        max_total_images=_resolve_max_total_images(args),
        max_seq_length=args.max_seq_length,
        keep_recent_text_messages=args.keep_recent_text_messages,
        keep_recent_tool_image_messages=args.keep_recent_tool_image_messages,
        max_image_side=args.max_image_side,
        max_image_pixels=args.max_image_pixels,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )


def _close_qwen_policy(policy: Any) -> None:
    runtime = getattr(policy, "vllm_runtime", None)
    if runtime is not None:
        close_fn = getattr(runtime, "close", None)
        if callable(close_fn):
            close_fn()


def _claim_dynamic_batch_indices(
    *,
    task_queue_dir: Path,
    runtime: Any,
    indices: List[int],
    batch_size: int,
) -> List[int]:
    claimed_indices: List[int] = []
    for _ in range(max(1, int(batch_size))):
        task_index = claim_next_dynamic_task_index(task_queue_dir, runtime=runtime)
        if task_index is None:
            break
        claimed_indices.append(int(indices[int(task_index)]))
    return claimed_indices


def _build_proposal_runtime(args: argparse.Namespace, *, runtime: Any) -> SiglipFeatureEncoder | None:
    if not args.proposal_model_path:
        return None
    if args.proposal_device:
        device = args.proposal_device
    else:
        try:
            import torch
        except Exception:
            device = "cpu"
        else:
            device = f"cuda:{int(runtime.local_rank)}" if torch.cuda.is_available() else "cpu"
    return SiglipFeatureEncoder.from_pretrained(
        args.proposal_model_path,
        torch_dtype=args.proposal_torch_dtype,
        device=device,
    )


def _save_batch_results(records: List[Dict[str, Any]], output_path: Path) -> None:
    if output_path.suffix in {".jsonl", ".json"}:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix == ".jsonl":
            with output_path.open("w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return
        output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    for record in records:
        safe_video_id = str(record.get("video_id") or "sample").replace("/", "_")
        dataset_index = int(record.get("dataset_index", 0))
        file_path = output_path / f"{dataset_index:06d}_{safe_video_id}.json"
        file_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def _wait_for_sharded_outputs(
    output_path: str | Path,
    *,
    num_shards: int,
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
            shard_output_path = sharded_output_path(
                base_output_path,
                num_shards=int(num_shards),
                shard_index=shard_index,
            )
            ready_marker_path = _shard_ready_marker_path(shard_output_path)
            if not ready_marker_path.exists():
                all_ready = False
                shard_status[str(ready_marker_path)] = "missing"
                continue
            if not shard_output_path.exists():
                all_ready = False
                shard_status[str(shard_output_path)] = "missing"
                continue
            try:
                shard_records, _ = load_rollout_records(shard_output_path)
            except Exception as exc:
                all_ready = False
                shard_status[str(shard_output_path)] = f"unreadable: {exc}"
                continue
            shard_status[str(shard_output_path)] = f"ready:{len(shard_records)}"
        if all_ready:
            return
        time.sleep(max(0.05, float(poll_interval_sec)))
    raise TimeoutError(
        "Timed out while waiting for sharded batch-rollout outputs to become ready: "
        + json.dumps(shard_status, ensure_ascii=False)
    )


def _merge_sharded_outputs(output_path: str | Path, *, num_shards: int) -> List[Dict[str, Any]]:
    base_output_path = Path(output_path)
    if int(num_shards) <= 1:
        records, _ = load_rollout_records(base_output_path)
        return records
    merged_records: List[Dict[str, Any]] = []
    for shard_index in range(int(num_shards)):
        shard_output_path = sharded_output_path(
            base_output_path,
            num_shards=int(num_shards),
            shard_index=shard_index,
        )
        shard_records, _ = load_rollout_records(shard_output_path)
        merged_records.extend(shard_records)
    merged_records.sort(key=lambda record: (int(record.get("dataset_index", 0) or 0), str(record.get("video_id") or "")))
    _save_batch_results(merged_records, base_output_path)
    return merged_records


def _shard_ready_marker_path(output_path: str | Path) -> Path:
    output_path = Path(output_path)
    return output_path.with_name(output_path.name + ".ready")


def main(argv: List[str] | None = None) -> Dict[str, Any]:
    args = parse_args(argv)
    if args.policy_backend == "replay" and not args.response:
        raise SystemExit("At least one --response is required for replay rollout.")

    runtime = distributed_runtime_from_env()
    log_dir = resolve_experiment_log_dir(args.log_dir, fallback_paths=[args.output])
    shard_spec = resolve_shard_spec(num_shards=args.num_shards, shard_index=args.shard_index, runtime=runtime)
    config = _build_config(args)
    dataset = SaverAgentDataset(
        args.data,
        data_root=args.data_root,
        config=config,
        include_splits=parse_include_splits(args.include_splits),
        require_frame_cache=True,
        require_feature_cache=True,
    )
    if hasattr(dataset, "format_frame_cache_status"):
        runtime_log(
            dataset.format_frame_cache_status(prefix="rollout frame cache"),
            runtime=runtime,
            main_process_only=True,
        )
    indices = _resolve_dataset_indices(args, len(dataset))
    strict_feature_guided_proposal = _records_require_feature_guided_proposal(
        [dataset.records[int(index)] for index in indices]
    )
    if strict_feature_guided_proposal and not str(args.proposal_model_path or "").strip():
        raise ValueError(
            "Batch rollout requires --proposal-model-path because the selected samples expose seek_evidence."
        )
    use_dynamic_claiming = shard_spec.is_sharded
    active_rank_count = min(len(indices), int(shard_spec.num_shards)) if use_dynamic_claiming else 1
    active_participant = (not use_dynamic_claiming) or (runtime.rank < active_rank_count)
    task_queue_dir = Path(args.output).parent / f".{Path(args.output).name}.batch_rollout_task_queue"
    local_indices = shard_sequence(indices, num_shards=shard_spec.num_shards, shard_index=shard_spec.shard_index)
    if use_dynamic_claiming:
        try:
            initialize_dynamic_task_queue(
                task_queue_dir,
                num_tasks=len(indices),
                runtime=runtime,
                timeout_sec=5.0,
                poll_interval_sec=0.05,
            )
        except TimeoutError:
            use_dynamic_claiming = False
            active_rank_count = 1
            active_participant = True
            runtime_log(
                "dynamic batch-rollout task queue was not initialized by the main process; falling back to static shard assignment",
                runtime=runtime,
            )
    output_path = sharded_output_path(args.output, num_shards=shard_spec.num_shards, shard_index=shard_spec.shard_index)
    effective_policy_device_map = resolve_inference_device_map(args.device_map, runtime=runtime)
    if runtime.is_main_process and log_dir is not None:
        write_json(
            log_dir / "batch_run_saver_rollout_run_config.json",
            {
                "timestamp_utc": utc_timestamp(),
                "data": args.data,
                "data_root": args.data_root,
                "include_splits": parse_include_splits(args.include_splits) or [],
                "output": args.output,
                "log_dir": str(log_dir),
                "policy_backend": args.policy_backend,
                "model_path": args.model_path,
                "use_vllm": bool(args.use_vllm),
                "vllm_mode": str(args.vllm_mode),
                "proposal_model_path": args.proposal_model_path,
                "strict_feature_guided_proposal": bool(strict_feature_guided_proposal),
                "max_turns": int(args.max_turns),
                "rollout_batch_size": int(args.rollout_batch_size),
                "max_new_tokens": int(args.max_new_tokens),
            },
        )

    runtime_log(
        "batch rollout startup: "
        f"dataset_size={len(dataset)} total_indices={len(indices)} "
        f"local_indices={'dynamic' if use_dynamic_claiming else len(local_indices)} "
        f"strict_feature_guided_proposal={bool(strict_feature_guided_proposal)} "
        f"policy_backend={args.policy_backend} include_splits={parse_include_splits(args.include_splits) or 'all'} "
        f"output={output_path}",
        runtime=runtime,
    )

    proposal_runtime = None
    if args.proposal_model_path and active_participant and indices:
        runtime_log(
            f"loading proposal model from {args.proposal_model_path}",
            runtime=runtime,
        )
        proposal_runtime = _build_proposal_runtime(args, runtime=runtime)
    qwen_policy = None
    if args.policy_backend == "qwen" and active_participant and indices:
        runtime_log(
            f"loading policy model from {args.model_path} with device_map={effective_policy_device_map}",
            runtime=runtime,
        )
        qwen_policy = _build_qwen_policy(args, runtime=runtime)
    runner = SaverRolloutRunner(
        adapter=TimeSearchRolloutAdapter(config=config),
        max_turns=args.max_turns,
        config=config,
    )

    results: List[Dict[str, Any]] = []
    completed = 0
    claimed_indices: List[int] = []
    summary_payload: Dict[str, Any] = {}
    try:
        while True:
            if use_dynamic_claiming:
                if not active_participant:
                    break
                batch_dataset_indices = _claim_dynamic_batch_indices(
                    task_queue_dir=task_queue_dir,
                    runtime=runtime,
                    indices=indices,
                    batch_size=args.rollout_batch_size,
                )
                if not batch_dataset_indices:
                    break
            else:
                if completed >= len(local_indices):
                    break
                batch_dataset_indices = [
                    int(dataset_index)
                    for dataset_index in local_indices[completed : completed + max(1, int(args.rollout_batch_size))]
                ]
            claimed_indices.extend(batch_dataset_indices)
            batch_items = [dataset[dataset_index] for dataset_index in batch_dataset_indices]
            for item in batch_items:
                _attach_proposal_context(
                    item,
                    proposal_runtime=proposal_runtime,
                    strict_feature_guided_proposal=strict_feature_guided_proposal,
                )
            policy = qwen_policy if qwen_policy is not None else ReplayPolicy(args.response)
            batch_results = runner.run_episodes(batch_items, policy)
            for dataset_index, result in zip(batch_dataset_indices, batch_results):
                serialized = _serialize_result(result)
                serialized["dataset_index"] = int(dataset_index)
                results.append(serialized)
                completed += 1
                progress_completed = (
                    record_dynamic_task_completion(task_queue_dir, runtime=runtime)
                    if use_dynamic_claiming
                    else completed
                )
                if should_log_progress(progress_completed, len(indices), args.progress_every):
                    runtime_log(
                        f"rollout progress: {progress_completed}/{len(indices)} dataset_index={dataset_index} "
                        f"video_id={serialized.get('video_id', '')}",
                        runtime=runtime,
                    )

        _save_batch_results(results, output_path)
        _shard_ready_marker_path(output_path).write_text(
            json.dumps({"num_records": len(results), "dataset_indices": claimed_indices}, ensure_ascii=False),
            encoding="utf-8",
        )
        runtime_log(f"saved {len(results)} rollout records to {output_path}", runtime=runtime)
        merged_result_count = len(results)
        if shard_spec.is_sharded and runtime.is_main_process:
            _wait_for_sharded_outputs(
                args.output,
                num_shards=shard_spec.num_shards,
            )
            merged_records = _merge_sharded_outputs(args.output, num_shards=shard_spec.num_shards)
            merged_result_count = len(merged_records)
            runtime_log(
                f"merged {merged_result_count} sharded rollout records into {args.output}",
                runtime=runtime,
                main_process_only=True,
            )
        summary_payload = {
            "timestamp_utc": utc_timestamp(),
            "output_path": str(args.output if shard_spec.is_sharded else output_path),
            "num_results": int(merged_result_count),
            "num_local_indices": len(results),
            "num_shards": int(shard_spec.num_shards),
        }
        if runtime.is_main_process and log_dir is not None:
            write_json(
                log_dir / "batch_run_saver_rollout_summary.json",
                summary_payload,
            )
    finally:
        if qwen_policy is not None:
            _close_qwen_policy(qwen_policy)
    return summary_payload


if __name__ == "__main__":
    main()
