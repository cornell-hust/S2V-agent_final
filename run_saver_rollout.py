#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from split_utils import parse_include_splits

from saver_v3.data.config import (
    DEFAULT_POLICY_MAX_NEW_TOKENS,
    DEFAULT_RECOMMENDED_KEEP_RECENT_TEXT_MESSAGES,
    DEFAULT_RECOMMENDED_MAX_SEQ_LENGTH,
    DEFAULT_RECOMMENDED_MAX_TOTAL_IMAGES,
    DEFAULT_TOTAL_VISUAL_BUDGET,
    PromptConfig,
    PreviewConfig,
    RolloutTraceConfig,
    SaverAgentConfig,
)
from saver_v3.core.adapter import TimeSearchRolloutAdapter
from saver_v3.data.dataset import SaverAgentDataset
from saver_v3.core.proposal import SiglipFeatureEncoder
from saver_v3.model.qwen_policy import DEFAULT_MODEL_PATH, QwenGenerationPolicy
from saver_v3.core.rollout import ReplayPolicy, SaverRolloutRunner
from saver_v3.model.vllm_generation import (
    VllmQwenGenerationPolicy,
    build_vllm_policy_from_model_path,
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
    parser = argparse.ArgumentParser(description="Run a minimal SAVER rollout with replayed responses.")
    parser.set_defaults(_max_total_images_explicit=False)
    parser.add_argument("--data", required=True, help="Path to saver_agent JSONL data.")
    parser.add_argument("--data-root", default="", help="Root path used to resolve relative video paths.")
    parser.add_argument("--include-splits", default="", help="Optional comma-separated split whitelist for --data.")
    parser.add_argument("--index", type=int, default=0, help="Dataset sample index.")
    parser.add_argument("--max-turns", type=int, default=14, help="Maximum rollout turns.")
    parser.add_argument(
        "--policy-backend",
        choices=["replay", "qwen"],
        default="replay",
        help="Use replayed responses or real Qwen generation.",
    )
    parser.add_argument("--response", action="append", default=[], help="Replayed model response for one turn.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local Qwen model path.")
    parser.add_argument("--proposal-model-path", default="", help="Optional local SigLIP/CLIP path for feature-guided proposal.")
    parser.add_argument("--proposal-torch-dtype", default="auto", help="Torch dtype for the proposal encoder.")
    parser.add_argument("--proposal-device", default="cpu", help="Device for the proposal encoder, e.g. cpu or cuda:0.")
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype passed to from_pretrained.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map argument.")
    parser.add_argument("--attn-implementation", default="", help="Optional attention backend, e.g. flash_attention_2.")
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
    parser.add_argument("--num-preview-frames", type=int, default=8, help="Maximum preview frames injected into the first user turn.")
    parser.add_argument("--preview-sampling-fps", type=float, default=None, help="Target preview sampling fps before capping by preview frame count.")
    parser.add_argument("--initial-user-template", default="", help="Optional custom template for the first user prompt.")
    parser.add_argument("--preview-instruction", default="", help="Optional custom preview instruction.")
    parser.add_argument("--tool-response-template", default="", help="Optional custom tool follow-up prompt template.")
    parser.add_argument("--record-observation-content", action="store_true", help="Store full tool observation content in rollout traces.")
    parser.add_argument(
        "--no-record-message-history",
        action="store_true",
        help="Disable storing the full message history in the rollout output.",
    )
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args(argv)


def _resolve_max_total_images(args: argparse.Namespace) -> int:
    explicit_max_total_images = int(getattr(args, "max_total_images", 0) or 0)
    if bool(getattr(args, "_max_total_images_explicit", False)):
        return max(0, explicit_max_total_images)
    alias_max_total_images = max(0, int(getattr(args, "total_visual_budget", 0) or 0))
    if alias_max_total_images > 0:
        return alias_max_total_images
    return max(0, explicit_max_total_images)


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
    cache = item.setdefault("multimodal_cache", {})
    if bool(strict_feature_guided_proposal):
        cache["strict_feature_guided_proposal"] = True
    if proposal_runtime is not None:
        cache["proposal_runtime"] = proposal_runtime


def _serialize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return _to_jsonable(result)


def _serialize_message(message: Dict[str, Any]) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = []
    for item in message.get("content", []):
        if item.get("type") == "image":
            image = item.get("image")
            shape = list(image.shape) if hasattr(image, "shape") else None
            content.append({"type": "image", "shape": shape})
        else:
            content.append({"type": item.get("type"), "text": item.get("text")})
    return {
        "role": message.get("role"),
        "name": message.get("name"),
        "content": content,
    }


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        if {"role", "content"}.issubset(value.keys()):
            return _serialize_message(value)
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return {"type": "tensor", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    return value


def _close_qwen_policy(policy: Any) -> None:
    runtime = getattr(policy, "vllm_runtime", None)
    if runtime is not None:
        close_fn = getattr(runtime, "close", None)
        if callable(close_fn):
            close_fn()


def main() -> None:
    args = parse_args()
    if args.policy_backend == "replay" and not args.response:
        raise SystemExit("At least one --response is required for replay rollout.")

    config = SaverAgentConfig(
        preview=PreviewConfig(
            num_preview_frames=args.num_preview_frames,
            preview_sampling_fps=args.preview_sampling_fps,
            max_preview_frames=args.num_preview_frames,
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

    dataset = SaverAgentDataset(
        args.data,
        data_root=args.data_root,
        config=config,
        include_splits=parse_include_splits(args.include_splits),
        require_frame_cache=True,
        require_feature_cache=True,
    )
    strict_feature_guided_proposal = _records_require_feature_guided_proposal([dataset.records[args.index]])
    if strict_feature_guided_proposal and not str(args.proposal_model_path or "").strip():
        raise ValueError(
            "Rollout requires --proposal-model-path because the selected sample exposes seek_evidence."
        )
    proposal_runtime = None
    if args.proposal_model_path:
        proposal_runtime = SiglipFeatureEncoder.from_pretrained(
            args.proposal_model_path,
            torch_dtype=args.proposal_torch_dtype,
            device=args.proposal_device,
        )
    item = dataset[args.index]
    _attach_proposal_context(
        item,
        proposal_runtime=proposal_runtime,
        strict_feature_guided_proposal=strict_feature_guided_proposal,
    )
    runner = SaverRolloutRunner(
        adapter=TimeSearchRolloutAdapter(config=config),
        max_turns=args.max_turns,
        config=config,
    )
    if args.policy_backend == "qwen":
        if bool(getattr(args, "use_vllm", True)):
            policy = build_vllm_policy_from_model_path(
                args=args,
                runtime=None,
                model_path=args.model_path,
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
        else:
            policy = QwenGenerationPolicy.from_pretrained(
                args.model_path,
                torch_dtype=args.torch_dtype,
                device_map=args.device_map,
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
    else:
        policy = ReplayPolicy(args.response)
    try:
        result = runner.run_episode(item, policy)
        serialized = _serialize_result(result)
    finally:
        if args.policy_backend == "qwen":
            _close_qwen_policy(policy)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(serialized, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
