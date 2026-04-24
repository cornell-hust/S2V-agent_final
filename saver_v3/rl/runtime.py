from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import train_saver_rl_trl as legacy_train_saver_rl_trl

from saver_v3.core.reward import (
    DEFAULT_RL_REWARD_VERSION,
    DEFAULT_COMPONENT_WEIGHTS,
)
from saver_v3.cli.common import apply_config_overrides, load_yaml_mapping, resolve_path, write_json
from saver_v3.common import ensure_fa3_training_ready
from saver_v3.data.config import DEFAULT_ROLLOUT_MAX_TURNS, saver_config_from_mapping
from saver_v3.rl.resume import load_trainer_resume_state


REMOVED_ACTIVE_RL_CONFIG_FIELDS = {
    "optimization.rl_replay_buffer_enable": "replay buffer was removed because active RL now only supports pure-pack episode_inputs.",
    "optimization.rl_replay_buffer_type": "replay buffer was removed because active RL now only supports pure-pack episode_inputs.",
    "optimization.rl_replay_buffer_capacity": "replay buffer was removed because active RL now only supports pure-pack episode_inputs.",
    "optimization.rl_replay_buffer_alpha": "replay buffer was removed because active RL now only supports pure-pack episode_inputs.",
    "optimization.replay_buffer_enable": "replay buffer was removed because active RL now only supports pure-pack episode_inputs.",
    "optimization.replay_buffer_type": "replay buffer was removed because active RL now only supports pure-pack episode_inputs.",
    "optimization.replay_buffer_capacity": "replay buffer was removed because active RL now only supports pure-pack episode_inputs.",
    "optimization.replay_buffer_alpha": "replay buffer was removed because active RL now only supports pure-pack episode_inputs.",
    "optimization.rl_all_empty_policy": "legacy empty-batch policy was removed because active RL now always uses donor no-op padding on pure-pack episode_inputs.",
    "optimization.all_empty_policy": "legacy empty-batch policy was removed because active RL now always uses donor no-op padding on pure-pack episode_inputs.",
    "rewards.open_ended_judge_enabled": "external LLM judge toggles were removed from active RL; semantic QA reward now uses the fixed local-Qwen judge backend.",
    "rewards.open_ended_judge_base_url": "external LLM judge URLs were removed from active RL; semantic QA reward now uses the fixed local-Qwen judge backend.",
    "rewards.open_ended_judge_model": "external LLM judge model selection was removed from active RL; semantic QA reward now uses the fixed local-Qwen judge backend.",
    "rewards.open_ended_judge_cache_path": "external LLM judge cache paths were removed from active RL; semantic QA reward now uses the fixed local-Qwen judge backend.",
    "rewards.open_ended_judge_timeout_sec": "external LLM judge timeouts were removed from active RL; semantic QA reward now uses the fixed local-Qwen judge backend.",
    "optimization.fecv_stage_batch_size": "FECV has been removed from active RL.",
    "optimization.rl_fecv_use_cache": "FECV has been removed from active RL.",
    "optimization.rl_fecv_failure_policy": "FECV has been removed from active RL.",
    "optimization.counterfactual_max_images": "Counterfactual verification has been removed from active RL.",
}

ACTIVE_RL_USE_LIGER_LOSS = False


def _resolve_bool(mapping: Mapping[str, Any], key: str, default: bool) -> bool:
    if key not in mapping:
        return bool(default)
    value = mapping.get(key)
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return bool(default)


def _normalize_vllm_mode(*, engine: Any, launcher: Any) -> str:
    normalized = str(launcher or engine or "colocate").strip().lower()
    if normalized in {
        "colocate",
        "external_launcher",
        "local_rank",
        "torchrun_external_launcher",
        "vllm_local_rank",
    }:
        return "colocate"
    if normalized in {"managed_http_server", "managed_server", "server"}:
        return "server"
    return normalized or "colocate"


def _deepspeed_config_uses_param_offload(config_path: str | Path | None) -> bool:
    config_text = str(config_path or "").strip()
    if not config_text:
        return False
    try:
        payload = json.loads(Path(config_text).read_text(encoding="utf-8"))
    except Exception:
        return False
    zero_optimization = payload.get("zero_optimization") or {}
    offload_param = zero_optimization.get("offload_param") or {}
    if not isinstance(offload_param, Mapping) or not offload_param:
        return False
    device = str(offload_param.get("device") or "").strip().lower()
    return device not in {"", "none"}


def _resolve_active_rl_num_train_epochs(optimization: Mapping[str, Any]) -> float:
    raw_value = optimization.get("num_train_epochs", optimization.get("num_epochs"))
    if raw_value is None:
        return 1.0
    resolved = float(raw_value or 1.0)
    if resolved != 1.0:
        raise ValueError(
            "Active continuous RL derives internal trainer epochs from num_iterations; "
            "optimization.num_train_epochs/num_epochs must be omitted or set to 1.0."
        )
    return 1.0


def _resolve_liger_compatible_deepspeed_config(
    config_path: str | Path | None,
    *,
    use_liger_loss: bool,
) -> tuple[str | None, bool, str]:
    config_text = str(config_path or "").strip()
    if not config_text:
        return None, False, ""
    resolved = Path(config_text).expanduser().resolve()
    if not bool(use_liger_loss):
        return str(resolved), False, ""
    if not _deepspeed_config_uses_param_offload(resolved):
        return str(resolved), False, ""
    fallback = resolved.with_name("zero3_full_model.json")
    if not fallback.exists():
        raise FileNotFoundError(
            "Liger-compatible DeepSpeed config requires a non-offload ZeRO-3 config next to the current RL config: "
            f"requested={resolved} expected_fallback={fallback}"
        )
    reason = (
        "Active RL uses Liger loss by default, and Liger + ZeRO-3 parameter offload causes extremely slow "
        "compute_loss forwards in idea2_v3. Switching to zero3_full_model.json keeps training semantics unchanged "
        "while avoiding ZeRO-3 parameter offload."
    )
    return str(fallback), True, reason


def _supported_reward_weight_keys(reward_version: str) -> set[str]:
    normalized = str(reward_version or DEFAULT_RL_REWARD_VERSION).strip().lower()
    if normalized == DEFAULT_RL_REWARD_VERSION:
        return set(DEFAULT_COMPONENT_WEIGHTS)
    raise ValueError(f"Unsupported RL reward version: {reward_version!r}")


def _extract_reward_weight_mapping(rewards: Mapping[str, Any], *, reward_version: str) -> Dict[str, float]:
    supported_keys = _supported_reward_weight_keys(reward_version)
    weight_mapping: Dict[str, float] = {}
    unsupported: list[str] = []
    for key, value in rewards.items():
        text = str(key).strip()
        if not text.endswith("_weight"):
            continue
        component_name = text[: -len("_weight")]
        if component_name not in supported_keys:
            unsupported.append(component_name)
            continue
        weight_mapping[component_name] = float(value)
    if unsupported:
        supported_text = ", ".join(sorted(supported_keys))
        unsupported_text = ", ".join(sorted(unsupported))
        raise ValueError(
            f"Unsupported RL reward weight keys for reward_version={reward_version!r}: {unsupported_text}. "
            f"Supported keys are: {supported_text}."
        )
    return weight_mapping


def _reject_removed_active_rl_config_fields(config: Mapping[str, Any]) -> None:
    for dotted_key, reason in REMOVED_ACTIVE_RL_CONFIG_FIELDS.items():
        cursor: Any = config
        found = True
        for segment in dotted_key.split("."):
            if isinstance(cursor, Mapping) and segment in cursor:
                cursor = cursor[segment]
            else:
                found = False
                break
        if found:
            raise ValueError(
                f"Removed active RL config field `{dotted_key}` is no longer supported: {reason} "
                "Migrate to the current default saver_v3.cli.train_rl_ds pure-pack episode GRPO route."
            )


@dataclass
class RLJobConfig:
    run_name: str
    output_dir: str
    train_manifest: str
    eval_manifest: str | None
    data_root: str
    eval_data_root: str
    include_splits: str
    eval_include_splits: str
    policy_init_from: str
    reference_model: str
    base_model: str
    torch_dtype: str
    attn_implementation: str
    gradient_checkpointing: bool
    rollout_backend: str
    rollout_config: str
    deepspeed_config_path: str | None
    requested_deepspeed_config_path: str | None = None
    liger_deepspeed_auto_switched: bool = False
    liger_deepspeed_switch_reason: str = ""
    materialized_train_items_path: str = ""
    materialized_eval_items_path: str = ""
    require_materialized_runtime_cache: bool = False
    reward_version: str = DEFAULT_RL_REWARD_VERSION
    reward_config: Dict[str, Any] = field(default_factory=dict)
    resume_from_checkpoint: str = ""
    rollout_eval_output_dir: str = ""
    inline_rollout_eval: bool = False
    rollout_eval_start_iteration: int = 1
    rollout_eval_interval_iterations: int = 1
    final_rollout_eval: bool = False
    num_iterations: int = 1
    num_train_epochs: float = 1.0
    learning_rate: float = 5e-7
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    min_weight: float = 0.1
    advantage_clip: float = 3.0
    ppo_clip_epsilon: float = 0.2
    kl_beta: float = 0.0
    rl_steps_per_generation: int = 4
    rollout_stage_batch_size: int = 16
    rollout_count: int = 16
    num_generations: int = 8
    rollout_max_turns: int = DEFAULT_ROLLOUT_MAX_TURNS
    policy_do_sample: bool = False
    policy_temperature: float | None = None
    policy_top_p: float | None = None
    policy_top_k: int | None = None
    policy_repetition_penalty: float | None = None
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 2
    bf16: bool = True
    fp16: bool = False
    max_seq_length: int = 8192
    max_total_images: int = 28
    max_image_side: int = 640
    max_image_pixels: int = 0
    keep_recent_text_messages: int = 20
    keep_recent_tool_image_messages: int = 3
    initial_observation_mode: str = "explicit_first_scan"
    initial_scan_num_frames: int = 8
    protect_initial_scan_from_visual_budget: bool = True
    error_on_initial_scan_seq_prune: bool = True
    num_preview_frames: int = 8
    max_tool_message_frames: int = 0
    max_total_video_frames: int = 0
    compute_loss_microbatch_size: int = 2
    proposal_model_path: str = ""
    proposal_torch_dtype: str = "auto"
    proposal_device: str = ""
    eval_proposal_model_path: str = ""
    eval_proposal_torch_dtype: str = "auto"
    eval_proposal_device: str = ""
    eval_enable_semantic_metrics: bool | None = None
    eval_semantic_metrics: Sequence[str] | str | None = None
    eval_semantic_judge_base_url: str = ""
    eval_semantic_judge_model: str = ""
    eval_semantic_judge_cache_path: str = ""
    eval_semantic_judge_timeout_sec: float = 30.0
    eval_bertscore_model_path: str = ""
    vllm_mode: str = "colocate"
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.35
    vllm_max_num_seqs: int = 4
    vllm_fallback_max_num_seqs: int = 2
    vllm_guided_decoding_regex: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_files(
        cls,
        *,
        config_path: str,
        model_config_path: str,
        attention_config_path: str,
        deepspeed_config_path: str | None = None,
        config_overrides: Sequence[str] | None = None,
    ) -> "RLJobConfig":
        resolved_config_path = resolve_path(config_path)
        config_anchor = (resolved_config_path or Path(config_path).expanduser()).resolve().parent
        config = apply_config_overrides(load_yaml_mapping(config_path), config_overrides)
        _reject_removed_active_rl_config_fields(config)
        model_config = load_yaml_mapping(model_config_path)
        attention_config = load_yaml_mapping(attention_config_path)
        rollout_config_path = str(config.get("rollout_config") or "").strip()
        resolved_rollout_config_path = resolve_path(rollout_config_path, anchor=config_anchor) if rollout_config_path else None
        rollout_config = load_yaml_mapping(resolved_rollout_config_path) if resolved_rollout_config_path else {}
        optimization = dict(config.get("optimization") or {})
        data = dict(config.get("data") or {})
        distributed = dict(config.get("distributed") or {})
        logging_cfg = dict(config.get("logging") or {})
        rewards = dict(config.get("rewards") or {})
        proposal_cfg = dict(config.get("proposal") or {})
        semantic_cfg = dict(rollout_config.get("semantic_metrics") or {})
        saver_config = saver_config_from_mapping(config)
        explicit_reference_model = str(config.get("reference_model") or "").strip()
        if explicit_reference_model:
            raise ValueError(
                "Active RL no longer accepts `reference_model` in config; "
                "reference now follows the per-iteration trainer init model in TimeSearch-R style."
            )
        rollout_engine = str(rollout_config.get("engine") or "").strip()
        server_cfg = dict(rollout_config.get("server") or {})
        client_cfg = dict(rollout_config.get("client") or {})
        normalized_vllm_mode = _normalize_vllm_mode(engine=rollout_engine, launcher=server_cfg.get("launcher"))
        if normalized_vllm_mode != "colocate":
            raise ValueError("idea2_v3 RL supports only local-rank/external-launcher vLLM mode.")
        resolved_deepspeed_config_path = (
            resolve_path(deepspeed_config_path or str(config.get("deepspeed_config") or "").strip(), anchor=config_anchor)
            if (deepspeed_config_path or str(config.get("deepspeed_config") or "").strip())
            else None
        )
        requested_deepspeed_config_path = (
            str(resolved_deepspeed_config_path) if resolved_deepspeed_config_path is not None else None
        )
        liger_deepspeed_auto_switched = False
        liger_deepspeed_switch_reason = ""
        if resolved_deepspeed_config_path is not None:
            (
                resolved_deepspeed_config_path,
                liger_deepspeed_auto_switched,
                liger_deepspeed_switch_reason,
            ) = _resolve_liger_compatible_deepspeed_config(
                resolved_deepspeed_config_path,
                use_liger_loss=ACTIVE_RL_USE_LIGER_LOSS,
            )
        if str(attention_config.get("policy_name") or "").strip() != "fa3_only":
            raise ValueError("idea2_v3 requires attention policy fa3_only for RL")
        if str(model_config.get("attn_implementation") or "").strip() != "flash_attention_3":
            raise ValueError("idea2_v3 RL expects model attn_implementation=flash_attention_3")
        if str(config.get("rollout_backend") or "vllm").strip().lower() != "vllm":
            raise ValueError("idea2_v3 RL currently supports only rollout_backend=vllm")
        reward_config: Dict[str, Any] = {}
        reward_version = str(rewards.get("reward_version") or DEFAULT_RL_REWARD_VERSION)
        weight_mapping = _extract_reward_weight_mapping(rewards, reward_version=reward_version)
        if weight_mapping:
            reward_config["weights"] = weight_mapping
        return cls(
            run_name=str(config.get("run_name") or "qwen3_vl_8b_grpo_ds8"),
            output_dir=str(config.get("output_dir") or "artifacts/rl/qwen3_vl_8b_grpo"),
            train_manifest=str((data.get("train_manifest") or "")).strip(),
            eval_manifest=str((data.get("eval_manifest") or "")).strip() or None,
            materialized_train_items_path=str((data.get("materialized_train_items_path") or "")).strip(),
            materialized_eval_items_path=str((data.get("materialized_eval_items_path") or "")).strip(),
            require_materialized_runtime_cache=bool(data.get("require_materialized_runtime_cache", False)),
            data_root=str((data.get("data_root") or "")).strip(),
            eval_data_root=str((data.get("eval_data_root") or data.get("data_root") or "")).strip(),
            include_splits=str((data.get("include_splits") or "")).strip(),
            eval_include_splits=str((data.get("eval_include_splits") or data.get("include_splits") or "")).strip(),
            policy_init_from=str(config.get("policy_init_from") or "").strip(),
            reference_model="",
            base_model=str(model_config.get("base_model") or "").strip(),
            torch_dtype=str(model_config.get("torch_dtype") or "bfloat16"),
            attn_implementation=str(model_config.get("attn_implementation") or "flash_attention_3"),
            gradient_checkpointing=_resolve_bool(model_config, "gradient_checkpointing", True),
            rollout_backend="vllm",
            rollout_config=str(resolved_rollout_config_path) if resolved_rollout_config_path is not None else rollout_config_path,
            deepspeed_config_path=(str(resolved_deepspeed_config_path) if resolved_deepspeed_config_path is not None else None),
            requested_deepspeed_config_path=requested_deepspeed_config_path,
            liger_deepspeed_auto_switched=bool(liger_deepspeed_auto_switched),
            liger_deepspeed_switch_reason=str(liger_deepspeed_switch_reason or ""),
            reward_version=reward_version,
            reward_config=reward_config,
            resume_from_checkpoint=str(config.get("resume_from_checkpoint") or "").strip(),
            rollout_eval_output_dir=str(logging_cfg.get("rollout_eval_output_dir") or "").strip(),
            inline_rollout_eval=_resolve_bool(logging_cfg, "inline_rollout_eval", False),
            rollout_eval_start_iteration=int(logging_cfg.get("rollout_eval_start_iteration", 1) or 1),
            rollout_eval_interval_iterations=int(logging_cfg.get("rollout_eval_interval_iterations", 1) or 1),
            final_rollout_eval=_resolve_bool(logging_cfg, "final_rollout_eval", False),
            num_iterations=int(optimization.get("num_iterations", optimization.get("num_updates", 1)) or 1),
            num_train_epochs=_resolve_active_rl_num_train_epochs(optimization),
            learning_rate=float(optimization.get("learning_rate", 5e-7) or 5e-7),
            per_device_train_batch_size=int(optimization.get("per_device_batch_size", optimization.get("per_device_train_batch_size", 1)) or 1),
            gradient_accumulation_steps=int(optimization.get("gradient_accumulation_steps", 8) or 8),
            min_weight=(
                float(optimization.get("min_weight"))
                if optimization.get("min_weight") is not None
                else 0.1
            ),
            advantage_clip=(
                float(optimization.get("advantage_clip"))
                if optimization.get("advantage_clip") is not None
                else 3.0
            ),
            ppo_clip_epsilon=(
                float(optimization.get("ppo_clip_epsilon"))
                if optimization.get("ppo_clip_epsilon") is not None
                else 0.2
            ),
            kl_beta=(
                float(optimization.get("kl_beta"))
                if optimization.get("kl_beta") is not None
                else 0.0
            ),
            rl_steps_per_generation=int(optimization.get("rl_steps_per_generation", optimization.get("steps_per_generation", 4)) or 4),
            rollout_stage_batch_size=int(optimization.get("rollout_stage_batch_size", 16) or 16),
            rollout_count=int(optimization.get("rollout_count", 16) or 16),
            num_generations=int(optimization.get("num_generations", 8) or 8),
            rollout_max_turns=int(
                optimization.get("rollout_max_turns", DEFAULT_ROLLOUT_MAX_TURNS) or DEFAULT_ROLLOUT_MAX_TURNS
            ),
            policy_do_sample=_resolve_bool(optimization, "policy_do_sample", False),
            policy_temperature=(
                float(optimization.get("policy_temperature"))
                if optimization.get("policy_temperature") is not None
                else (
                    float(client_cfg.get("temperature"))
                    if client_cfg.get("temperature") is not None
                    else None
                )
            ),
            policy_top_p=(
                float(optimization.get("policy_top_p"))
                if optimization.get("policy_top_p") is not None
                else (
                    float(client_cfg.get("top_p"))
                    if client_cfg.get("top_p") is not None
                    else None
                )
            ),
            policy_top_k=(
                int(optimization.get("policy_top_k"))
                if optimization.get("policy_top_k") is not None
                else (
                    int(client_cfg.get("top_k"))
                    if client_cfg.get("top_k") is not None
                    else None
                )
            ),
            policy_repetition_penalty=(
                float(optimization.get("policy_repetition_penalty"))
                if optimization.get("policy_repetition_penalty") is not None
                else None
            ),
            logging_steps=int(logging_cfg.get("logging_steps", logging_cfg.get("log_every_n_steps", 10)) or 10),
            save_steps=int(logging_cfg.get("save_steps", logging_cfg.get("save_every_n_steps", 100)) or 100),
            save_total_limit=int(logging_cfg.get("save_total_limit", 2) or 2),
            bf16=_resolve_bool(distributed, "bf16", True),
            fp16=_resolve_bool(distributed, "fp16", False),
            max_seq_length=int(((model_config.get("sequence") or {}).get("max_length", 8192)) or 8192),
            max_total_images=int(((model_config.get("vision") or {}).get("max_images_per_sample", 28)) or 28),
            max_image_side=int(((model_config.get("vision") or {}).get("max_image_side", 640)) or 640),
            max_image_pixels=int(((model_config.get("vision") or {}).get("max_image_pixels", 0)) or 0),
            keep_recent_text_messages=int(optimization.get("keep_recent_text_messages", 20) or 20),
            keep_recent_tool_image_messages=int(optimization.get("keep_recent_tool_image_messages", 3) or 3),
            initial_observation_mode=str(saver_config.initial_observation.mode or "explicit_first_scan"),
            initial_scan_num_frames=max(1, int(saver_config.initial_observation.scan_num_frames or 8)),
            protect_initial_scan_from_visual_budget=bool(saver_config.initial_observation.protect_from_visual_budget),
            error_on_initial_scan_seq_prune=bool(saver_config.initial_observation.error_on_seq_prune),
            num_preview_frames=int(saver_config.preview.num_preview_frames or 8),
            max_tool_message_frames=int(optimization.get("max_tool_message_frames", 0) or 0),
            max_total_video_frames=int(optimization.get("max_total_video_frames", 0) or 0),
            compute_loss_microbatch_size=int(
                optimization.get(
                    "compute_loss_microbatch_size",
                    optimization.get("rl_compute_loss_microbatch_size", 2),
                )
                or 2
            ),
            proposal_model_path=str(proposal_cfg.get("model_path") or "").strip(),
            proposal_torch_dtype=str(proposal_cfg.get("torch_dtype") or "auto"),
            proposal_device=str(proposal_cfg.get("device") or "").strip(),
            eval_proposal_model_path=str(proposal_cfg.get("eval_model_path") or proposal_cfg.get("model_path") or "").strip(),
            eval_proposal_torch_dtype=str(proposal_cfg.get("eval_torch_dtype") or proposal_cfg.get("torch_dtype") or "auto"),
            eval_proposal_device=str(proposal_cfg.get("eval_device") or proposal_cfg.get("device") or "").strip(),
            eval_enable_semantic_metrics=(
                _resolve_bool(semantic_cfg, "enabled", True)
                if "enabled" in semantic_cfg
                else None
            ),
            eval_semantic_metrics=(semantic_cfg.get("metrics") if "metrics" in semantic_cfg else None),
            eval_semantic_judge_base_url=str(semantic_cfg.get("judge_base_url") or "").strip(),
            eval_semantic_judge_model=str(semantic_cfg.get("judge_model") or "").strip(),
            eval_semantic_judge_cache_path=str(semantic_cfg.get("judge_cache_path") or "").strip(),
            eval_semantic_judge_timeout_sec=float(semantic_cfg.get("judge_timeout_sec", 30.0) or 30.0),
            eval_bertscore_model_path=str(semantic_cfg.get("bertscore_model_path") or "").strip(),
            vllm_mode=normalized_vllm_mode,
            vllm_tensor_parallel_size=int(server_cfg.get("tensor_parallel_size", 1) or 1),
            vllm_gpu_memory_utilization=float(server_cfg.get("gpu_memory_utilization", 0.9) or 0.9),
            vllm_max_num_seqs=int(optimization.get("vllm_max_num_seqs", 4) or 4),
            vllm_fallback_max_num_seqs=int(optimization.get("vllm_fallback_max_num_seqs", 2) or 2),
            vllm_guided_decoding_regex=str(client_cfg.get("guided_decoding_regex") or "").strip(),
        )


@dataclass
class RLRunResult:
    run_name: str
    output_dir: str
    supported: bool
    latest_checkpoint: str = ""
    summary_path: str = ""
    launch_manifest_path: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _append_flag(argv: list[str], flag: str, value: str | None) -> None:
    if value is None:
        return
    text = str(value).strip()
    if text:
        argv.extend([flag, text])


def build_active_rl_trl_argv(job: RLJobConfig) -> list[str]:
    argv: list[str] = [
        "--data", job.train_manifest,
        "--output-dir", job.output_dir,
        "--model-path", job.policy_init_from,
        "--rollout-eval-start-iteration", str(job.rollout_eval_start_iteration),
        "--rollout-eval-interval-iterations", str(job.rollout_eval_interval_iterations),
        "--num-iterations", str(job.num_iterations),
        "--rollout-count", str(job.rollout_count),
        "--num-generations", str(job.num_generations),
        "--rollout-max-turns", str(job.rollout_max_turns),
        "--learning-rate", str(job.learning_rate),
        "--num-train-epochs", str(job.num_train_epochs),
        "--per-device-train-batch-size", str(job.per_device_train_batch_size),
        "--gradient-accumulation-steps", str(job.gradient_accumulation_steps),
        "--min-weight", str(job.min_weight),
        "--advantage-clip", str(job.advantage_clip),
        "--ppo-clip-epsilon", str(job.ppo_clip_epsilon),
        "--kl-beta", str(job.kl_beta),
        "--logging-steps", str(job.logging_steps),
        "--save-steps", str(job.save_steps),
        "--save-total-limit", str(job.save_total_limit),
        "--torch-dtype", job.torch_dtype,
        "--attn-implementation", job.attn_implementation,
        "--max-seq-length", str(job.max_seq_length),
        "--max-total-images", str(job.max_total_images),
        "--max-image-side", str(job.max_image_side),
        "--max-image-pixels", str(job.max_image_pixels),
        "--keep-recent-text-messages", str(job.keep_recent_text_messages),
        "--keep-recent-tool-image-messages", str(job.keep_recent_tool_image_messages),
        "--initial-observation-mode", str(job.initial_observation_mode),
        "--initial-scan-num-frames", str(job.initial_scan_num_frames),
        "--protect-initial-scan-from-visual-budget", "true" if job.protect_initial_scan_from_visual_budget else "false",
        "--error-on-initial-scan-seq-prune", "true" if job.error_on_initial_scan_seq_prune else "false",
        "--num-preview-frames", str(job.num_preview_frames),
        "--max-tool-message-frames", str(job.max_tool_message_frames),
        "--max-total-video-frames", str(job.max_total_video_frames),
        "--rl-reward-version", job.reward_version,
        "--rl-compute-loss-microbatch-size", str(job.compute_loss_microbatch_size),
        "--rl-steps-per-generation", str(job.rl_steps_per_generation),
        "--rollout-stage-batch-size", str(job.rollout_stage_batch_size),
        "--vllm-tensor-parallel-size", str(job.vllm_tensor_parallel_size),
        "--vllm-gpu-memory-utilization", str(job.vllm_gpu_memory_utilization),
        "--vllm-max-num-seqs", str(job.vllm_max_num_seqs),
        "--vllm-fallback-max-num-seqs", str(job.vllm_fallback_max_num_seqs),
        "--rl-reward-config-json", json.dumps(job.reward_config, ensure_ascii=False),
    ]
    if job.gradient_checkpointing:
        argv.append("--gradient-checkpointing")
    argv.append("--inline-rollout-eval" if job.inline_rollout_eval else "--defer-rollout-eval")
    argv.append("--final-rollout-eval" if job.final_rollout_eval else "--no-final-rollout-eval")
    if job.bf16:
        argv.append("--bf16")
    if job.fp16:
        argv.append("--fp16")
    if job.policy_do_sample:
        argv.append("--policy-do-sample")
    argv.extend(["--use-liger-loss", "true" if ACTIVE_RL_USE_LIGER_LOSS else "false"])
    _append_flag(argv, "--rollout-eval-output-dir", job.rollout_eval_output_dir)
    _append_flag(argv, "--include-splits", job.include_splits)
    _append_flag(argv, "--data-root", job.data_root)
    _append_flag(argv, "--resume-from-checkpoint", job.resume_from_checkpoint)
    _append_flag(argv, "--eval-data", job.eval_manifest)
    _append_flag(argv, "--eval-data-root", job.eval_data_root)
    _append_flag(argv, "--eval-include-splits", job.eval_include_splits)
    _append_flag(argv, "--proposal-model-path", job.proposal_model_path)
    _append_flag(argv, "--proposal-torch-dtype", job.proposal_torch_dtype)
    _append_flag(argv, "--proposal-device", job.proposal_device)
    _append_flag(argv, "--materialized-train-items-path", job.materialized_train_items_path)
    _append_flag(argv, "--materialized-eval-items-path", job.materialized_eval_items_path)
    _append_flag(argv, "--require-materialized-runtime-cache", "true" if job.require_materialized_runtime_cache else "false")
    _append_flag(argv, "--eval-proposal-model-path", job.eval_proposal_model_path)
    _append_flag(argv, "--eval-proposal-torch-dtype", job.eval_proposal_torch_dtype)
    _append_flag(argv, "--eval-proposal-device", job.eval_proposal_device)
    if job.eval_enable_semantic_metrics is not None:
        _append_flag(argv, "--eval-enable-semantic-metrics", "true" if job.eval_enable_semantic_metrics else "false")
    if job.eval_semantic_metrics is not None:
        eval_semantic_metrics_value = (
            str(job.eval_semantic_metrics).strip()
            if isinstance(job.eval_semantic_metrics, str)
            else ",".join(str(metric).strip() for metric in job.eval_semantic_metrics if str(metric).strip())
        )
        _append_flag(argv, "--eval-semantic-metrics", eval_semantic_metrics_value)
    _append_flag(argv, "--eval-semantic-judge-base-url", job.eval_semantic_judge_base_url)
    _append_flag(argv, "--eval-semantic-judge-model", job.eval_semantic_judge_model)
    _append_flag(argv, "--eval-semantic-judge-cache-path", job.eval_semantic_judge_cache_path)
    _append_flag(argv, "--eval-semantic-judge-timeout-sec", job.eval_semantic_judge_timeout_sec)
    _append_flag(argv, "--eval-bertscore-model-path", job.eval_bertscore_model_path)
    _append_flag(argv, "--deepspeed", job.deepspeed_config_path)
    _append_flag(argv, "--vllm-guided-decoding-regex", job.vllm_guided_decoding_regex)
    _append_flag(argv, "--policy-temperature", job.policy_temperature)
    _append_flag(argv, "--policy-top-p", job.policy_top_p)
    _append_flag(argv, "--policy-top-k", job.policy_top_k)
    _append_flag(argv, "--policy-repetition-penalty", job.policy_repetition_penalty)
    return argv


def run_rl_job(job: RLJobConfig) -> RLRunResult:
    if not job.train_manifest:
        raise ValueError("RL train manifest is required")
    if not job.policy_init_from:
        raise ValueError("RL policy_init_from is required")
    ensure_fa3_training_ready(require_gpu=True)
    output_dir = Path(job.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = build_active_rl_trl_argv(job)
    launch_manifest_path = write_json(
        {
            "entrypoint": "train_saver_rl_trl.py",
            "episode_grpo_pure_pack": True,
            "use_liger_loss": False,
            "rl_replay_buffer_supported": False,
            "rl_replay_buffer_mode": "disabled_episode_grpo",
            "argv": argv,
            "config": job.to_dict(),
            "requested_deepspeed_config_path": job.requested_deepspeed_config_path,
            "effective_deepspeed_config_path": job.deepspeed_config_path,
            "liger_deepspeed_auto_switched": bool(job.liger_deepspeed_auto_switched),
            "liger_deepspeed_switch_reason": str(job.liger_deepspeed_switch_reason or ""),
            "resume_state": (
                load_trainer_resume_state(job.resume_from_checkpoint, source="job.resume_from_checkpoint").to_dict()
                if str(job.resume_from_checkpoint or "").strip()
                else None
            ),
        },
        output_dir / "rl_launch_manifest.json",
    )
    result = legacy_train_saver_rl_trl.main(argv)
    summary_payload = dict(result or {})
    summary_payload.update(
        {
            "run_name": job.run_name,
            "output_dir": str(output_dir),
            "entrypoint": "train_saver_rl_trl.py",
            "episode_grpo_pure_pack": True,
            "use_liger_loss": False,
            "rl_replay_buffer_supported": False,
            "rl_replay_buffer_mode": "disabled_episode_grpo",
            "launch_manifest_path": str(launch_manifest_path),
        }
    )
    summary_path = write_json(summary_payload, output_dir / "rl_summary.json")
    return RLRunResult(
        run_name=job.run_name,
        output_dir=str(output_dir),
        supported=True,
        latest_checkpoint=str(summary_payload.get("latest_checkpoint") or ""),
        summary_path=str(summary_path),
        launch_manifest_path=str(launch_manifest_path),
        notes=[
            "RL now dispatches through the TRL + vLLM GRPO route.",
            f"reference_model_mode=per_iteration_trainer_init source={job.policy_init_from}",
            f"deepspeed={job.deepspeed_config_path or '(none)'}",
            *(
                [
                    f"requested_deepspeed={job.requested_deepspeed_config_path or '(none)'}",
                    f"liger_deepspeed_auto_switched=true reason={job.liger_deepspeed_switch_reason}",
                ]
                if bool(job.liger_deepspeed_auto_switched)
                else []
            ),
        ],
    )
