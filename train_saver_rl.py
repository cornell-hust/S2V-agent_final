#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from split_utils import parse_include_splits

from saver_v3.data.config import (
    DEFAULT_POLICY_MAX_NEW_TOKENS,
    DEFAULT_RECOMMENDED_KEEP_RECENT_TEXT_MESSAGES,
    DEFAULT_RECOMMENDED_MAX_SEQ_LENGTH,
    DEFAULT_RECOMMENDED_MAX_TOTAL_IMAGES,
    PreviewConfig,
    PromptConfig,
    RolloutTraceConfig,
    SaverAgentConfig,
)
from saver_v3.metrics.evaluation import RolloutEvaluationConfig
from saver_v3.common.experiment_logging import resolve_experiment_log_dir, utc_timestamp, write_json
from saver_v3.rl.grpo_trainer_env import run_trainer_native_grpo
from saver_v3.model.qwen_policy import DEFAULT_MODEL_PATH, QwenGenerationPolicy
from saver_v3.core.qwen_verifier import DEFAULT_VERIFIER_MODEL_PATH
from saver_v3.core.reward import DEFAULT_RL_REWARD_VERSION
from saver_v3.common.runtime import (
    distributed_runtime_from_env,
    init_torch_distributed,
    resolve_inference_device_map,
)
from saver_v3.sft.training import run_rollout_eval_from_checkpoint


class _StoreEvalMaxTotalImages(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, "_eval_max_total_images_explicit", True)


def _parse_bool_flag(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _is_adapter_only_checkpoint(path_value: str) -> bool:
    text = str(path_value or "").strip()
    if not text:
        return False
    path = Path(text).expanduser()
    if not path.exists():
        return False
    if path.is_file():
        return path.name in {"adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"}
    return any((path / name).exists() for name in ("adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAVER RL trainer-native GRPO entrypoint with inline/deferred epoch-end rollout eval recovery."
    )
    parser.set_defaults(_eval_max_total_images_explicit=False, inline_rollout_eval=False)
    parser.add_argument("--data", default="", help="Path to SAVER agent/oracle JSONL data.")
    parser.add_argument("--data-root", default="", help="Root path used to resolve relative video paths.")
    parser.add_argument("--include-splits", default="", help="Optional comma-separated split whitelist for --data.")
    parser.add_argument("--output-dir", required=True, help="Directory to store RL outputs.")
    parser.add_argument("--log-dir", default="", help="Optional directory for RL logs. Defaults to <output-dir>/logs.")
    parser.add_argument(
        "--rollout-eval-output-dir",
        default="",
        help="Optional directory for RL epoch-end rollout eval outputs. Defaults to iteration-scoped subdirs under --output-dir.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default="",
        help="Optional authority checkpoint used to replay a missing RL epoch-end rollout eval.",
    )
    parser.add_argument(
        "--resume-rollout-eval-only",
        action="store_true",
        help="Only replay the missing RL epoch-end rollout eval for --resume-from-checkpoint, then exit.",
    )
    parser.add_argument(
        "--inline-rollout-eval",
        dest="inline_rollout_eval",
        action="store_true",
        help="Run RL epoch-end rollout eval inline immediately after each iteration.",
    )
    parser.add_argument(
        "--defer-rollout-eval",
        dest="inline_rollout_eval",
        action="store_false",
        help="Defer RL epoch-end rollout eval to the external recovery path. This is the default behavior.",
    )
    parser.add_argument(
        "--rl-reward-version",
        choices=["legacy", "timesearch_v1", "timesearch_v2"],
        default=DEFAULT_RL_REWARD_VERSION,
        help="Reward version used by Stage 5 RL only.",
    )
    parser.add_argument(
        "--rl-open-ended-judge-enabled",
        type=_parse_bool_flag,
        default=True,
        help="Enable external OpenAI-compatible judging for open-ended RL accuracy reward components when configured.",
    )
    parser.add_argument("--rl-open-ended-judge-base-url", default="", help="Optional OpenAI-compatible base URL.")
    parser.add_argument("--rl-open-ended-judge-model", default="", help="Optional OpenAI-compatible model name.")
    parser.add_argument("--rl-open-ended-judge-cache-path", default="", help="Optional JSON cache path for RL judge calls.")
    parser.add_argument("--rl-open-ended-judge-timeout-sec", type=float, default=30.0, help="Timeout in seconds for RL judge requests.")
    parser.add_argument("--rl-reward-config-json", default="", help="Optional JSON object overriding RL reward configuration fields.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Initial Qwen policy checkpoint.")
    parser.add_argument("--proposal-model-path", default="", help="Optional local SigLIP/CLIP path for RL rollout collection.")
    parser.add_argument("--proposal-torch-dtype", default="auto", help="Torch dtype for the proposal encoder.")
    parser.add_argument("--proposal-device", default="", help="Optional device for the proposal encoder.")
    parser.add_argument(
        "--reference-model-path",
        default="",
        help="Optional fixed reference policy checkpoint. Defaults to the initial --model-path.",
    )
    parser.add_argument("--num-iterations", type=int, default=1, help="Number of RL iterations.")
    parser.add_argument("--rollout-count", type=int, default=16, help="Number of videos per iteration.")
    parser.add_argument("--num-generations", type=int, default=4, help="Number of sampled rollouts per video per iteration.")
    parser.add_argument("--rollout-start-index", type=int, default=0, help="Start index for the first iteration.")
    parser.add_argument("--rollout-max-turns", type=int, default=14, help="Maximum rollout turns.")
    parser.add_argument("--dry-run", action="store_true", help="Collect/score examples but skip gradient updates.")
    parser.add_argument("--min-weight", type=float, default=0.1, help="Minimum absolute rollout advantage kept for updates.")
    parser.add_argument("--advantage-clip", type=float, default=3.0, help="Absolute clip value for group-relative advantages.")
    parser.add_argument("--ppo-clip-epsilon", type=float, default=0.2, help="PPO/GRPO clipping epsilon.")
    parser.add_argument("--kl-beta", type=float, default=0.0, help="KL regularization weight against the fixed reference policy.")
    parser.add_argument("--max-train-examples", type=int, default=0, help="Optional limit on RL training examples per iteration.")
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype passed to from_pretrained.")
    parser.add_argument("--attn-implementation", default="", help="Optional attention backend for policy model.")
    parser.add_argument("--policy-device-map", default="auto", help="device_map used for rollout policy inference.")
    parser.add_argument("--policy-do-sample", action="store_true", help="Enable sampling for rollout generation.")
    parser.add_argument("--policy-temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--policy-top-p", type=float, default=None, help="Sampling top-p.")
    parser.add_argument("--policy-top-k", type=int, default=None, help="Sampling top-k.")
    parser.add_argument("--policy-repetition-penalty", type=float, default=None, help="Sampling repetition penalty.")
    parser.add_argument("--policy-max-new-tokens", type=int, default=DEFAULT_POLICY_MAX_NEW_TOKENS, help="Policy generation length.")
    parser.add_argument("--rl-rollout-use-cache", type=_parse_bool_flag, default=True, help="Enable KV cache during trainer-native RL rollout generation.")
    parser.add_argument("--rl-fecv-use-cache", type=_parse_bool_flag, default=True, help="Enable KV cache during trainer-native RL online FECV replay.")
    parser.add_argument("--rl-compute-loss-microbatch-size", type=int, default=2, help="Completion-only forward microbatch size in trainer-native GRPO compute_loss().")
    parser.add_argument("--rl-steps-per-generation", type=int, default=1, help="Reuse one rollout generation batch across this many trainer steps.")
    parser.add_argument("--rl-replay-buffer-enable", type=_parse_bool_flag, default=True, help="Enable replay-buffer fallback for trainer-native GRPO empty batches.")
    parser.add_argument("--rl-replay-buffer-type", choices=["none", "ssr", "dapo"], default="ssr", help="Replay buffer policy used by trainer-native GRPO.")
    parser.add_argument("--rl-replay-buffer-capacity", type=int, default=16, help="Maximum number of replay experiences retained for fallback.")
    parser.add_argument("--rl-replay-buffer-alpha", type=float, default=1.0, help="Priority exponent used by the replay buffer.")
    parser.add_argument("--rl-fecv-failure-policy", choices=["degrade", "drop", "fail"], default="degrade", help="How trainer-native GRPO handles online FECV failures.")
    parser.add_argument("--rl-all-empty-policy", choices=["true_skip", "zero_loss"], default="true_skip", help="How trainer-native GRPO handles batches where all ranks have no trainable samples.")
    parser.add_argument("--rl-log-empty-batch-rank-summary", type=_parse_bool_flag, default=True, help="Log per-rank empty-batch summaries when trainer-native GRPO skips a batch.")
    parser.add_argument("--verifier-backend", choices=["qwen_self_verifier"], default="qwen_self_verifier", help="Diagnostic verifier backend only.")
    parser.add_argument("--verifier-model-path", default=DEFAULT_VERIFIER_MODEL_PATH, help="Diagnostic verifier model path.")
    parser.add_argument("--verifier-torch-dtype", default="auto", help="Torch dtype for verifier.")
    parser.add_argument("--verifier-device-map", default="auto", help="device_map used for verifier inference.")
    parser.add_argument("--verifier-attn-implementation", default="", help="Optional attention backend for verifier.")
    parser.add_argument("--verifier-max-new-tokens", type=int, default=512, help="Verifier generation length.")
    parser.add_argument("--teacher-judge-model-path", default="", help="Optional offline Qwen teacher judge model path.")
    parser.add_argument("--teacher-judge-input-mode", choices=["text_only", "multimodal_visual", "auto"], default="auto", help="Teacher judge input mode used during RL teacher annotation.")
    parser.add_argument("--teacher-judge-anchor-policy", choices=["verify_only", "verify_and_finalize", "hard_examples"], default="verify_only", help="Teacher annotation anchor policy.")
    parser.add_argument("--teacher-judge-topk-frames-per-view", type=int, default=4, help="Maximum raw frames sampled for each teacher-judge view.")
    parser.add_argument("--teacher-judge-torch-dtype", default="auto", help="Torch dtype for the RL teacher judge.")
    parser.add_argument("--teacher-judge-device-map", default="auto", help="device_map used for RL teacher judge inference.")
    parser.add_argument("--teacher-judge-attn-implementation", default="", help="Optional attention backend for the RL teacher judge.")
    parser.add_argument("--teacher-judge-max-new-tokens", type=int, default=384, help="Teacher judge generation length during RL annotation.")
    parser.add_argument("--teacher-judge-max-images", type=int, default=8, help="Maximum images passed to the RL teacher judge per verify turn.")
    parser.add_argument("--teacher-judge-progress-every", type=int, default=25, help="Log RL teacher annotation progress every N verify turns.")
    parser.add_argument("--diagnostic-attach-reference-offline-verifier", action="store_true", help="Attach reference-conditioned offline verifier results during RL scoring.")
    parser.add_argument("--diagnostic-force-reverify", action="store_true", help="Force rerunning the diagnostic offline verifier when enabled.")
    parser.add_argument("--turn-advantage-gamma", type=float, default=0.9, help="Discount factor used when logging per-turn credit returns.")
    parser.add_argument("--turn-advantage-alpha", type=float, default=0.5, help="Strength of turn-level credit redistribution.")
    parser.add_argument("--turn-search-bonus", type=float, default=0.05, help="Small positive shaping for valid search turns.")
    parser.add_argument("--turn-evidence-bonus", type=float, default=0.1, help="Bonus per newly added evidence item.")
    parser.add_argument("--turn-finalize-bonus", type=float, default=0.2, help="Bonus for successful finalize_case turns.")
    parser.add_argument("--turn-invalid-penalty", type=float, default=0.75, help="Penalty for invalid turns.")
    parser.add_argument("--progress-every", type=int, default=1, help="Log rollout/score progress every N local items.")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable model gradient checkpointing.")
    parser.add_argument("--deepspeed", default="", help="Optional DeepSpeed config json passed to Trainer/TRL training arguments.")
    parser.add_argument("--learning-rate", type=float, default=5e-6, help="Update learning rate.")
    parser.add_argument("--num-train-epochs", type=float, default=1.0, help="Update epochs per iteration.")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Per-device update batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="Gradient accumulation steps.")
    parser.add_argument("--dataloader-num-workers", type=int, default=0, help="DataLoader worker count for each RL update stage.")
    parser.add_argument("--dataloader-prefetch-factor", type=int, default=0, help="Optional DataLoader prefetch_factor for RL updates.")
    parser.add_argument("--dataloader-persistent-workers", action="store_true", help="Keep RL update DataLoader workers alive across epochs.")
    parser.add_argument("--logging-steps", type=int, default=10, help="Trainer logging steps.")
    parser.add_argument("--save-steps", type=int, default=100, help="Trainer save steps.")
    parser.add_argument("--save-total-limit", type=int, default=2, help="Trainer save_total_limit.")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Trainer warmup ratio.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Trainer weight decay.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Trainer max grad norm.")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 training.")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16 training.")
    parser.add_argument("--lora", action="store_true", help="Use PEFT LoRA adapters.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--lora-target-modules", default="", help="Comma-separated LoRA target module names.")
    parser.add_argument("--max-image-side", type=int, default=640, help="Optional training-time max image side length in pixels for RL.")
    parser.add_argument("--max-image-pixels", type=int, default=0, help="Optional training-time max image area in pixels for RL.")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_RECOMMENDED_MAX_SEQ_LENGTH, help="Explicit tokenizer/processor max_length for RL update examples.")
    parser.add_argument("--keep-recent-tool-image-messages", type=int, default=0, help="If >0, keep images only for the N most recent tool messages during RL.")
    parser.add_argument("--keep-recent-text-messages", type=int, default=DEFAULT_RECOMMENDED_KEEP_RECENT_TEXT_MESSAGES, help="If >0, keep full text only for the N most recent non-initial history messages during RL.")
    parser.add_argument("--max-total-images", type=int, default=DEFAULT_RECOMMENDED_MAX_TOTAL_IMAGES, help="Optional hard cap on total images kept in each RL example.")
    parser.add_argument("--num-preview-frames", type=int, default=8, help="Preview frames for initial prompt.")
    parser.add_argument("--preview-sampling-fps", type=float, default=None, help="Preview sampling fps.")
    parser.add_argument("--eval-data", default="", help="Optional raw saver_agent/oracle JSONL used for rollout metrics after each update epoch.")
    parser.add_argument("--eval-data-root", default="", help="Root path used to resolve relative video paths for epoch-end rollout eval.")
    parser.add_argument("--eval-include-splits", default="", help="Optional comma-separated split whitelist for --eval-data.")
    parser.add_argument("--eval-max-records", type=int, default=0, help="Optional cap on eval records per epoch-end rollout eval.")
    parser.add_argument("--eval-rollout-max-turns", type=int, default=14, help="Maximum rollout turns for epoch-end rollout eval.")
    parser.add_argument("--eval-max-new-tokens-per-turn", type=int, default=DEFAULT_POLICY_MAX_NEW_TOKENS, help="Generation length budget for each epoch-end rollout eval turn.")
    parser.add_argument("--eval-total-visual-budget", type=int, default=0, help="Alias for a coarse epoch-end rollout visual budget.")
    parser.add_argument("--eval-max-total-images", type=int, default=0, action=_StoreEvalMaxTotalImages, help="Optional hard cap on total images preserved in each epoch-end rollout eval prompt.")
    parser.add_argument("--eval-proposal-model-path", default="", help="Optional proposal encoder path for epoch-end rollout eval.")
    parser.add_argument("--eval-proposal-torch-dtype", default="auto", help="Torch dtype for epoch-end eval proposal encoder.")
    parser.add_argument("--eval-proposal-device", default="", help="Optional device for epoch-end eval proposal encoder.")
    parser.add_argument("--eval-verifier-backend", choices=["qwen_self_verifier"], default="qwen_self_verifier", help="Diagnostic offline verifier backend used only when diagnostics are enabled.")
    parser.add_argument("--eval-verifier-model-path", default="", help="Optional verifier model path for epoch-end rollout eval.")
    parser.add_argument("--eval-verifier-torch-dtype", default="auto", help="Torch dtype for epoch-end eval verifier.")
    parser.add_argument("--eval-verifier-device-map", default="auto", help="device_map for epoch-end eval verifier.")
    parser.add_argument("--eval-verifier-attn-implementation", default="", help="Attention backend for epoch-end eval verifier.")
    parser.add_argument("--eval-verifier-max-new-tokens", type=int, default=512, help="Generation length for epoch-end eval verifier.")
    parser.add_argument("--eval-attach-reference-diagnostics", action="store_true", help="Attach reference-conditioned offline verifier diagnostics during epoch-end rollout eval.")
    parser.add_argument("--eval-progress-every", type=int, default=1, help="Log epoch-end rollout eval progress every N local items.")
    args = parser.parse_args(argv)
    if not bool(getattr(args, "_eval_max_total_images_explicit", False)) and int(
        getattr(args, "eval_max_total_images", 0) or 0
    ) <= 0:
        args.eval_max_total_images = max(0, int(getattr(args, "max_total_images", 0) or 0))
    reward_config_json = str(getattr(args, "rl_reward_config_json", "") or "").strip()
    if reward_config_json:
        parsed_reward_config = json.loads(reward_config_json)
        if not isinstance(parsed_reward_config, dict):
            raise ValueError("--rl-reward-config-json must decode to a JSON object.")
        args.rl_reward_config = parsed_reward_config
    else:
        args.rl_reward_config = {}
    if bool(getattr(args, "lora", False)):
        raise ValueError("idea2_v3 targets full-model RL only; --lora is disabled.")
    if _is_adapter_only_checkpoint(str(getattr(args, "model_path", "") or "")):
        raise ValueError("idea2_v3 RL does not accept adapter-only checkpoint paths for --model-path.")
    if _is_adapter_only_checkpoint(str(getattr(args, "reference_model_path", "") or "")):
        raise ValueError("idea2_v3 RL does not accept adapter-only checkpoint paths for --reference-model-path.")
    return args


def select_iteration_indices(dataset_size: int, rollout_count: int, start_index: int, iteration: int) -> List[int]:
    if dataset_size <= 0:
        return []
    offset = (int(start_index) + int(iteration) * int(rollout_count)) % int(dataset_size)
    return [(offset + i) % int(dataset_size) for i in range(int(rollout_count))]


def _resolve_eval_max_total_images(args: argparse.Namespace) -> int:
    explicit_max_total_images = int(getattr(args, "eval_max_total_images", 0) or 0)
    if bool(getattr(args, "_eval_max_total_images_explicit", False)):
        return max(0, explicit_max_total_images)
    alias_max_total_images = max(0, int(getattr(args, "eval_total_visual_budget", 0) or 0))
    if alias_max_total_images > 0:
        return alias_max_total_images
    training_max_total_images = max(0, int(getattr(args, "max_total_images", 0) or 0))
    if training_max_total_images > 0:
        return training_max_total_images
    return max(0, explicit_max_total_images)


def _resolve_resume_epoch_index(checkpoint_path: str | Path) -> int:
    checkpoint_dir = Path(checkpoint_path)
    base_name = checkpoint_dir.name
    if base_name.startswith("epoch_"):
        suffix = base_name.split("_", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    raise ValueError(f"Unable to resolve epoch index from resume checkpoint path: {checkpoint_dir}")


def _resolve_resume_iteration_index(checkpoint_path: str | Path) -> int:
    checkpoint_dir = Path(checkpoint_path)
    for candidate in (checkpoint_dir,) + tuple(checkpoint_dir.parents):
        base_name = candidate.name
        if base_name.startswith("iter_"):
            suffix = base_name.split("_", 1)[1]
            if suffix.isdigit():
                return int(suffix)
    raise ValueError(f"Unable to resolve iteration index from resume checkpoint path: {checkpoint_dir}")


def _resolve_resume_checkpoint_record_output_dir(checkpoint_path: str | Path, *, fallback_output_dir: str | Path) -> Path:
    checkpoint_dir = Path(checkpoint_path)
    if checkpoint_dir.name.startswith("epoch_") and checkpoint_dir.parent.name == "epoch_resume":
        return checkpoint_dir.parent.parent
    return Path(fallback_output_dir)


def resolve_reference_model_path(model_path: str | Path, reference_model_path: str | Path | None) -> str:
    reference_text = str(reference_model_path or "").strip()
    return reference_text or str(model_path)


def _build_config(args: argparse.Namespace) -> SaverAgentConfig:
    return SaverAgentConfig(
        preview=PreviewConfig(
            num_preview_frames=args.num_preview_frames,
            preview_sampling_fps=args.preview_sampling_fps,
            max_preview_frames=args.num_preview_frames,
        ),
        prompt=PromptConfig(),
        rollout_trace=RolloutTraceConfig(
            record_observation_content=False,
            record_state_deltas=True,
            record_message_history=True,
        ),
    )


def _build_policy(model_path: str | Path, args: argparse.Namespace, *, runtime: Any) -> QwenGenerationPolicy:
    return QwenGenerationPolicy.from_pretrained(
        model_path,
        torch_dtype=args.torch_dtype,
        device_map=resolve_inference_device_map(args.policy_device_map, runtime=runtime),
        attn_implementation=args.attn_implementation or None,
        max_new_tokens=args.policy_max_new_tokens,
        max_total_images=args.max_total_images,
        max_seq_length=args.max_seq_length,
        keep_recent_tool_image_messages=args.keep_recent_tool_image_messages,
        keep_recent_text_messages=args.keep_recent_text_messages,
        max_image_side=args.max_image_side,
        max_image_pixels=args.max_image_pixels,
        do_sample=args.policy_do_sample,
        temperature=args.policy_temperature,
        top_p=args.policy_top_p,
        top_k=args.policy_top_k,
        repetition_penalty=args.policy_repetition_penalty,
    )


def _build_rollout_eval_config(
    *,
    args: argparse.Namespace,
    current_model_path: str | Path,
    reference_model_path: str | Path,
    config: SaverAgentConfig,
) -> Optional[RolloutEvaluationConfig]:
    del reference_model_path
    if not args.eval_data:
        return None
    return RolloutEvaluationConfig(
        data_path=Path(args.eval_data),
        data_root=args.eval_data_root or args.data_root,
        include_splits=parse_include_splits(args.eval_include_splits),
        max_records=args.eval_max_records,
        inline_rollout_eval=bool(args.inline_rollout_eval),
        rollout_max_turns=args.eval_rollout_max_turns,
        policy_max_new_tokens=args.eval_max_new_tokens_per_turn,
        max_total_images=_resolve_eval_max_total_images(args),
        max_seq_length=int(args.max_seq_length),
        keep_recent_tool_image_messages=int(args.keep_recent_tool_image_messages),
        keep_recent_text_messages=int(args.keep_recent_text_messages),
        max_image_side=args.max_image_side,
        max_image_pixels=args.max_image_pixels,
        proposal_model_path=args.eval_proposal_model_path or args.proposal_model_path,
        proposal_torch_dtype=args.eval_proposal_torch_dtype,
        proposal_device=args.eval_proposal_device,
        verifier_backend=args.eval_verifier_backend,
        verifier_model_path=args.eval_verifier_model_path or args.verifier_model_path or current_model_path,
        verifier_torch_dtype=args.eval_verifier_torch_dtype,
        verifier_device_map=args.eval_verifier_device_map,
        verifier_attn_implementation=args.eval_verifier_attn_implementation,
        verifier_max_new_tokens=args.eval_verifier_max_new_tokens,
        attach_reference_diagnostics=args.eval_attach_reference_diagnostics,
        progress_every=args.eval_progress_every,
        saver_config=config,
    )


def main() -> None:
    args = parse_args()
    if not args.resume_rollout_eval_only and not args.data:
        raise ValueError("--data is required unless --resume-rollout-eval-only is used.")
    if args.resume_rollout_eval_only and not args.resume_from_checkpoint:
        raise ValueError("--resume-rollout-eval-only requires --resume-from-checkpoint.")

    runtime = distributed_runtime_from_env()
    init_torch_distributed(runtime)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = resolve_experiment_log_dir(args.log_dir, output_dir=args.output_dir)
    rollout_eval_output_dir = str(args.rollout_eval_output_dir or "").strip()

    if runtime.is_main_process and log_dir is not None:
        write_json(
            Path(log_dir) / "train_saver_rl_run_config.json",
            {
                "timestamp_utc": utc_timestamp(),
                "script_entrypoint": "train_saver_rl.py",
                "data": args.data,
                "data_root": args.data_root,
                "include_splits": parse_include_splits(args.include_splits) or [],
                "output_dir": args.output_dir,
                "log_dir": str(log_dir),
                "rollout_eval_output_dir": rollout_eval_output_dir,
                "resume_from_checkpoint": args.resume_from_checkpoint,
                "resume_rollout_eval_only": bool(args.resume_rollout_eval_only),
                "inline_rollout_eval": bool(args.inline_rollout_eval),
                "model_path": args.model_path,
                "reference_model_path": args.reference_model_path,
                "num_iterations": int(args.num_iterations),
                "rollout_count": int(args.rollout_count),
                "num_generations": int(args.num_generations),
                "rollout_max_turns": int(args.rollout_max_turns),
                "rl_reward_version": str(args.rl_reward_version),
                "kl_beta": float(args.kl_beta),
                "rl_rollout_use_cache": bool(args.rl_rollout_use_cache),
                "rl_fecv_use_cache": bool(args.rl_fecv_use_cache),
                "rl_compute_loss_microbatch_size": int(args.rl_compute_loss_microbatch_size),
                "rl_steps_per_generation": int(args.rl_steps_per_generation),
                "rl_replay_buffer_enable": bool(args.rl_replay_buffer_enable),
                "rl_replay_buffer_type": str(args.rl_replay_buffer_type),
                "rl_replay_buffer_capacity": int(args.rl_replay_buffer_capacity),
                "rl_replay_buffer_alpha": float(args.rl_replay_buffer_alpha),
                "rl_fecv_failure_policy": str(args.rl_fecv_failure_policy),
                "rl_all_empty_policy": str(args.rl_all_empty_policy),
                "rl_log_empty_batch_rank_summary": bool(args.rl_log_empty_batch_rank_summary),
                "rl_open_ended_judge_enabled": bool(args.rl_open_ended_judge_enabled),
                "rl_open_ended_judge_base_url": str(args.rl_open_ended_judge_base_url or ""),
                "rl_open_ended_judge_model": str(args.rl_open_ended_judge_model or ""),
                "rl_open_ended_judge_cache_path": str(args.rl_open_ended_judge_cache_path or ""),
                "rl_open_ended_judge_timeout_sec": float(args.rl_open_ended_judge_timeout_sec),
                "proposal_model_path": str(args.proposal_model_path or ""),
                "teacher_judge_model_path": str(args.teacher_judge_model_path or ""),
                "teacher_judge_input_mode": str(args.teacher_judge_input_mode or ""),
                "eval_data": str(args.eval_data or ""),
            },
        )

    if args.resume_rollout_eval_only:
        rollout_eval_config = _build_rollout_eval_config(
            args=args,
            current_model_path=args.model_path,
            reference_model_path=resolve_reference_model_path(args.model_path, args.reference_model_path),
            config=_build_config(args),
        )
        if rollout_eval_config is None:
            raise ValueError("--resume-rollout-eval-only requires --eval-data so the missing RL rollout eval can be replayed.")
        checkpoint_path = Path(args.resume_from_checkpoint)
        epoch_index = _resolve_resume_epoch_index(checkpoint_path)
        iteration_index = _resolve_resume_iteration_index(checkpoint_path)
        recovery_kwargs = dict(
            checkpoint_path=checkpoint_path,
            output_dir=_resolve_resume_checkpoint_record_output_dir(checkpoint_path, fallback_output_dir=output_dir),
            rollout_eval_config=rollout_eval_config,
            epoch_index=epoch_index,
            model_path=args.model_path,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation or None,
            runtime=runtime,
        )
        if rollout_eval_output_dir:
            recovery_kwargs["rollout_eval_output_dir"] = rollout_eval_output_dir
        result = run_rollout_eval_from_checkpoint(**recovery_kwargs)
        final_summary = {
            "resume_from_checkpoint": str(checkpoint_path),
            "resume_rollout_eval_only": True,
            "resume_iteration_index": int(iteration_index),
            "resume_epoch_index": int(epoch_index),
            **(result or {}),
        }
        if runtime.is_main_process and log_dir is not None:
            write_json(Path(log_dir) / "train_saver_rl_summary.json", final_summary)
        if runtime.is_main_process:
            print(json.dumps(final_summary, ensure_ascii=False, indent=2))
        return

    result = run_trainer_native_grpo(
        args=args,
        runtime=runtime,
        log_dir=log_dir,
        config_builder=_build_config,
        eval_config_builder=_build_rollout_eval_config,
        reference_model_resolver=resolve_reference_model_path,
        select_iteration_indices_fn=select_iteration_indices,
    )
    if runtime.is_main_process:
        print(json.dumps({"latest_checkpoint": str(result.get("latest_checkpoint", args.model_path))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
