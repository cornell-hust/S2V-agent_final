from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from split_utils import parse_include_splits

from saver_v3.common.runtime import (
    distributed_runtime_from_env,
    init_torch_distributed,
    resolve_inference_device_map,
)
from saver_v3.core.qwen_verifier import DEFAULT_VERIFIER_MODEL_PATH
from saver_v3.data.config import (
    DEFAULT_POLICY_MAX_NEW_TOKENS,
    DEFAULT_ROLLOUT_MAX_TURNS,
    DEFAULT_RECOMMENDED_KEEP_RECENT_TEXT_MESSAGES,
    DEFAULT_RECOMMENDED_MAX_SEQ_LENGTH,
    DEFAULT_RECOMMENDED_MAX_TOTAL_IMAGES,
    PreviewConfig,
    PromptConfig,
    RolloutTraceConfig,
    SaverAgentConfig,
)
from saver_v3.metrics.evaluation import RolloutEvaluationConfig
from saver_v3.model.qwen_policy import DEFAULT_MODEL_PATH, QwenGenerationPolicy


REMOVED_ACTIVE_RL_FLAGS: Dict[str, str] = {
    "--rl-replay-buffer-enable": "replay buffer was removed because active RL now only supports pure-pack episode_inputs.",
    "--rl-replay-buffer-type": "replay buffer was removed because active RL now only supports pure-pack episode_inputs.",
    "--rl-replay-buffer-capacity": "replay buffer was removed because active RL now only supports pure-pack episode_inputs.",
    "--rl-replay-buffer-alpha": "replay buffer was removed because active RL now only supports pure-pack episode_inputs.",
    "--rl-all-empty-policy": "legacy empty-batch policy was removed because active RL now always uses donor no-op padding on pure-pack episode_inputs.",
}


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


def fail_on_removed_active_rl_flags(argv: List[str]) -> None:
    for token in list(argv or []):
        text = str(token or "")
        for flag, reason in REMOVED_ACTIVE_RL_FLAGS.items():
            if text == flag or text.startswith(f"{flag}="):
                raise SystemExit(
                    f"{flag} has been removed from active RL; {reason} "
                    "Migrate to saver_v3.cli.train_rl_ds with the current pure-pack GRPO route."
                )


def build_active_rl_arg_parser(*, description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
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
        choices=["legacy", "timesearch_v1", "timesearch_v2", "timesearch_v3"],
        default="timesearch_v3",
        help="Reward version used by active RL.",
    )
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
    parser.add_argument(
        "--rollout-max-turns",
        type=int,
        default=DEFAULT_ROLLOUT_MAX_TURNS,
        help="Maximum rollout turns.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Collect/score examples but skip gradient updates.")
    parser.add_argument("--min-weight", type=float, default=0.1, help="Minimum absolute rollout advantage kept for updates.")
    parser.add_argument("--advantage-clip", type=float, default=3.0, help="Absolute clip value for group-relative advantages.")
    parser.add_argument("--ppo-clip-epsilon", type=float, default=0.2, help="PPO/GRPO clipping epsilon.")
    parser.add_argument("--kl-beta", type=float, default=0.0, help="KL regularization weight against the fixed reference policy.")
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype passed to from_pretrained.")
    parser.add_argument("--attn-implementation", default="", help="Optional attention backend for policy model.")
    parser.add_argument("--policy-device-map", default="auto", help="device_map used for rollout policy inference.")
    parser.add_argument("--policy-do-sample", action="store_true", help="Enable sampling for rollout generation.")
    parser.add_argument("--policy-temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--policy-top-p", type=float, default=None, help="Sampling top-p.")
    parser.add_argument("--policy-top-k", type=int, default=None, help="Sampling top-k.")
    parser.add_argument("--policy-repetition-penalty", type=float, default=None, help="Sampling repetition penalty.")
    parser.add_argument("--policy-max-new-tokens", type=int, default=DEFAULT_POLICY_MAX_NEW_TOKENS, help="Policy generation length.")
    parser.add_argument("--rl-rollout-use-cache", type=_parse_bool_flag, default=True, help="Enable KV cache during RL rollout generation.")
    parser.add_argument("--rl-fecv-use-cache", type=_parse_bool_flag, default=True, help="Enable KV cache during RL online FECV replay.")
    parser.add_argument("--rl-compute-loss-microbatch-size", type=int, default=2, help="Deprecated compatibility knob. Active GRPO now computes each merged episode batch in one completion-only forward without loss microbatch slicing.")
    parser.add_argument("--rl-steps-per-generation", type=int, default=1, help="Reuse one rollout generation batch across this many trainer steps.")
    parser.add_argument("--use-liger-loss", type=_parse_bool_flag, default=False, help="Enable Liger fused GRPO loss on the active pure-pack path. Disabled by default in idea2_v3 because the current Qwen3-VL RL stack is incompatible/unstable with Liger.")
    parser.add_argument("--rollout-stage-batch-size", type=int, default=16, help="Chunk size for rollout stage processing.")
    parser.add_argument("--fecv-stage-batch-size", type=int, default=16, help="Chunk size for FECV stage processing.")
    parser.add_argument("--rl-fecv-failure-policy", choices=["degrade", "drop", "fail"], default="degrade", help="How active RL handles online FECV failures.")
    parser.add_argument("--rl-log-empty-batch-rank-summary", type=_parse_bool_flag, default=True, help="Log per-rank empty-batch summaries when active RL sees no trainable episode inputs.")
    parser.add_argument("--verifier-backend", choices=["qwen_self_verifier"], default="qwen_self_verifier", help="Diagnostic verifier backend only.")
    parser.add_argument("--verifier-model-path", default=DEFAULT_VERIFIER_MODEL_PATH, help="Diagnostic verifier model path.")
    parser.add_argument("--verifier-torch-dtype", default="auto", help="Torch dtype for verifier.")
    parser.add_argument("--verifier-device-map", default="auto", help="device_map used for verifier inference.")
    parser.add_argument("--verifier-attn-implementation", default="", help="Optional attention backend for verifier.")
    parser.add_argument("--verifier-max-new-tokens", type=int, default=512, help="Verifier generation length.")
    parser.add_argument("--teacher-judge-model-path", default="", help="Optional offline Qwen teacher judge model path.")
    parser.add_argument("--teacher-judge-input-mode", choices=["text_only", "multimodal_visual", "auto"], default="auto", help="Teacher judge input mode used during RL teacher annotation.")
    parser.add_argument("--teacher-judge-torch-dtype", default="auto", help="Torch dtype for the RL teacher judge.")
    parser.add_argument("--teacher-judge-device-map", default="auto", help="device_map used for RL teacher judge inference.")
    parser.add_argument("--teacher-judge-attn-implementation", default="", help="Optional attention backend for the RL teacher judge.")
    parser.add_argument("--teacher-judge-max-new-tokens", type=int, default=384, help="Teacher judge generation length during RL annotation.")
    parser.add_argument("--teacher-judge-max-images", type=int, default=8, help="Maximum images passed to the RL teacher judge per verify turn.")
    parser.add_argument("--teacher-judge-progress-every", type=int, default=25, help="Log RL teacher annotation progress every N verify turns.")
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
    parser.add_argument("--keep-recent-tool-image-messages", type=int, default=3, help="If >0, keep images only for the N most recent tool messages during RL.")
    parser.add_argument("--keep-recent-text-messages", type=int, default=DEFAULT_RECOMMENDED_KEEP_RECENT_TEXT_MESSAGES, help="If >0, keep full text only for the N most recent non-initial history messages during RL.")
    parser.add_argument("--max-total-images", type=int, default=DEFAULT_RECOMMENDED_MAX_TOTAL_IMAGES, help="Optional hard cap on total images kept in each RL example.")
    parser.add_argument("--num-preview-frames", type=int, default=8, help="Preview frames for initial prompt.")
    parser.add_argument("--preview-sampling-fps", type=float, default=None, help="Preview sampling fps.")
    parser.add_argument("--eval-data", default="", help="Optional raw saver_agent/oracle JSONL used for rollout metrics after each update epoch.")
    parser.add_argument("--eval-data-root", default="", help="Root path used to resolve relative video paths for epoch-end rollout eval.")
    parser.add_argument("--eval-include-splits", default="", help="Optional comma-separated split whitelist for --eval-data.")
    parser.add_argument("--eval-max-records", type=int, default=0, help="Optional cap on eval records per epoch-end rollout eval.")
    parser.add_argument(
        "--eval-rollout-max-turns",
        type=int,
        default=DEFAULT_ROLLOUT_MAX_TURNS,
        help="Maximum rollout turns for epoch-end rollout eval.",
    )
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
    return parser


def parse_active_rl_args(argv: Optional[List[str]], *, description: str) -> argparse.Namespace:
    raw_argv = list(argv) if argv is not None else []
    fail_on_removed_active_rl_flags(raw_argv)
    parser = build_active_rl_arg_parser(description=description)
    args = parser.parse_args(raw_argv or None)
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
    if str(getattr(args, "reference_model_path", "") or "").strip():
        raise ValueError(
            "idea2_v3 RL no longer accepts --reference-model-path; "
            "reference now follows the per-iteration trainer init model in TimeSearch-R style."
        )
    if _is_adapter_only_checkpoint(str(getattr(args, "model_path", "") or "")):
        raise ValueError("idea2_v3 RL does not accept adapter-only checkpoint paths for --model-path.")
    return args


def select_iteration_indices(
    dataset_size: int,
    rollout_count: int,
    start_index: int,
    iteration: int,
    *,
    seed: int = 42,
) -> List[int]:
    if dataset_size <= 0:
        return []
    total_needed = max(0, int(rollout_count))
    absolute_offset = max(0, int(start_index)) + max(0, int(iteration)) * total_needed
    selected: List[int] = []
    current_cycle = int(absolute_offset // int(dataset_size))
    offset_in_cycle = int(absolute_offset % int(dataset_size))
    remaining = total_needed
    while remaining > 0:
        order = list(range(int(dataset_size)))
        rng = random.Random(int(seed) + int(current_cycle))
        rng.shuffle(order)
        current_order = order[offset_in_cycle:]
        take = min(remaining, len(current_order))
        selected.extend(current_order[:take])
        remaining -= take
        current_cycle += 1
        offset_in_cycle = 0
    return selected


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


def resolve_resume_epoch_index(checkpoint_path: str | Path) -> int:
    checkpoint_dir = Path(checkpoint_path)
    base_name = checkpoint_dir.name
    if base_name.startswith("epoch_"):
        suffix = base_name.split("_", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    raise ValueError(f"Unable to resolve epoch index from resume checkpoint path: {checkpoint_dir}")


def resolve_resume_iteration_index(checkpoint_path: str | Path) -> int:
    checkpoint_dir = Path(checkpoint_path)
    for candidate in (checkpoint_dir,) + tuple(checkpoint_dir.parents):
        base_name = candidate.name
        if base_name.startswith("iter_"):
            suffix = base_name.split("_", 1)[1]
            if suffix.isdigit():
                return int(suffix)
    raise ValueError(f"Unable to resolve iteration index from resume checkpoint path: {checkpoint_dir}")


def resolve_resume_checkpoint_record_output_dir(checkpoint_path: str | Path, *, fallback_output_dir: str | Path) -> Path:
    checkpoint_dir = Path(checkpoint_path)
    if checkpoint_dir.name.startswith("epoch_") and checkpoint_dir.parent.name == "epoch_resume":
        return checkpoint_dir.parent.parent
    return Path(fallback_output_dir)


def resolve_reference_model_path(model_path: str | Path, reference_model_path: str | Path | None) -> str:
    reference_text = str(reference_model_path or "").strip()
    return reference_text or str(model_path)


def build_saver_config(args: argparse.Namespace) -> SaverAgentConfig:
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


def build_rollout_eval_config(
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


def build_policy(model_path: str | Path, args: argparse.Namespace, *, runtime: Any) -> QwenGenerationPolicy:
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
