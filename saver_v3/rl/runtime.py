from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

import train_saver_rl_trl as legacy_train_saver_rl_trl

from saver_v3.core.reward import (
    DEFAULT_COMPONENT_WEIGHTS,
    LEGACY_COMPONENT_ALIASES,
    TIMESARCH_COMPONENT_ALIASES,
    TIMESARCH_V1_COMPONENT_WEIGHTS,
    TIMESARCH_V2_COMPONENT_WEIGHTS,
)
from saver_v3.cli.common import load_yaml_mapping, resolve_path, write_json
from saver_v3.common import ensure_fa3_training_ready


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


def _supported_reward_weight_keys(reward_version: str) -> set[str]:
    normalized = str(reward_version or "timesearch_v2").strip().lower()
    if normalized == "timesearch_v1":
        return set(TIMESARCH_V1_COMPONENT_WEIGHTS) | set(TIMESARCH_COMPONENT_ALIASES)
    if normalized == "timesearch_v2":
        return set(TIMESARCH_V2_COMPONENT_WEIGHTS) | set(TIMESARCH_COMPONENT_ALIASES)
    if normalized == "legacy":
        return set(DEFAULT_COMPONENT_WEIGHTS) | set(LEGACY_COMPONENT_ALIASES)
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
    reward_version: str = "timesearch_v2"
    reward_config: Dict[str, Any] = field(default_factory=dict)
    num_iterations: int = 1
    num_train_epochs: float = 1.0
    learning_rate: float = 5e-7
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    rollout_count: int = 16
    num_generations: int = 8
    rollout_max_turns: int = 14
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
    keep_recent_tool_image_messages: int = 0
    num_preview_frames: int = 8
    vllm_mode: str = "colocate"
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.35
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
    ) -> "RLJobConfig":
        resolved_config_path = resolve_path(config_path)
        config_anchor = (resolved_config_path or Path(config_path).expanduser()).resolve().parent
        config = load_yaml_mapping(config_path)
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
        if str(attention_config.get("policy_name") or "").strip() != "fa3_only":
            raise ValueError("idea2_v3 requires attention policy fa3_only for RL")
        if str(model_config.get("attn_implementation") or "").strip() != "flash_attention_3":
            raise ValueError("idea2_v3 RL expects model attn_implementation=flash_attention_3")
        if str(config.get("rollout_backend") or "vllm").strip().lower() != "vllm":
            raise ValueError("idea2_v3 RL currently supports only rollout_backend=vllm")
        reward_config: Dict[str, Any] = {}
        reward_version = str(rewards.get("reward_version") or "timesearch_v2")
        weight_mapping = _extract_reward_weight_mapping(rewards, reward_version=reward_version)
        if weight_mapping:
            reward_config["weights"] = weight_mapping
        if "open_ended_judge_enabled" in rewards:
            reward_config["open_ended_judge_enabled"] = _resolve_bool(rewards, "open_ended_judge_enabled", True)
        if "open_ended_judge_base_url" in rewards:
            reward_config["open_ended_judge_base_url"] = str(rewards.get("open_ended_judge_base_url") or "")
        if "open_ended_judge_model" in rewards:
            reward_config["open_ended_judge_model"] = str(rewards.get("open_ended_judge_model") or "")
        if "open_ended_judge_cache_path" in rewards:
            reward_config["open_ended_judge_cache_path"] = str(rewards.get("open_ended_judge_cache_path") or "")
        if "open_ended_judge_timeout_sec" in rewards:
            reward_config["open_ended_judge_timeout_sec"] = float(rewards.get("open_ended_judge_timeout_sec") or 30.0)
        return cls(
            run_name=str(config.get("run_name") or "qwen3_vl_8b_grpo_ds8"),
            output_dir=str(config.get("output_dir") or "artifacts/rl/qwen3_vl_8b_grpo"),
            train_manifest=str((data.get("train_manifest") or "")).strip(),
            eval_manifest=str((data.get("eval_manifest") or "")).strip() or None,
            data_root=str((data.get("data_root") or "")).strip(),
            eval_data_root=str((data.get("eval_data_root") or data.get("data_root") or "")).strip(),
            include_splits=str((data.get("include_splits") or "")).strip(),
            eval_include_splits=str((data.get("eval_include_splits") or data.get("include_splits") or "")).strip(),
            policy_init_from=str(config.get("policy_init_from") or "").strip(),
            reference_model=str(config.get("reference_model") or model_config.get("base_model") or "").strip(),
            base_model=str(model_config.get("base_model") or "").strip(),
            torch_dtype=str(model_config.get("torch_dtype") or "bfloat16"),
            attn_implementation=str(model_config.get("attn_implementation") or "flash_attention_3"),
            gradient_checkpointing=_resolve_bool(model_config, "gradient_checkpointing", True),
            rollout_backend="vllm",
            rollout_config=str(resolved_rollout_config_path) if resolved_rollout_config_path is not None else rollout_config_path,
            deepspeed_config_path=(str(resolved_deepspeed_config_path) if resolved_deepspeed_config_path is not None else None),
            reward_version=reward_version,
            reward_config=reward_config,
            num_iterations=int(optimization.get("num_iterations", optimization.get("num_updates", 1)) or 1),
            num_train_epochs=float(optimization.get("num_train_epochs", optimization.get("num_epochs", 1.0)) or 1.0),
            learning_rate=float(optimization.get("learning_rate", 5e-7) or 5e-7),
            per_device_train_batch_size=int(optimization.get("per_device_batch_size", optimization.get("per_device_train_batch_size", 1)) or 1),
            gradient_accumulation_steps=int(optimization.get("gradient_accumulation_steps", 8) or 8),
            rollout_count=int(optimization.get("rollout_count", 16) or 16),
            num_generations=int(optimization.get("num_generations", 8) or 8),
            rollout_max_turns=int(optimization.get("rollout_max_turns", 14) or 14),
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
            keep_recent_tool_image_messages=int(optimization.get("keep_recent_tool_image_messages", 0) or 0),
            num_preview_frames=int(optimization.get("num_preview_frames", 8) or 8),
            vllm_mode=normalized_vllm_mode,
            vllm_tensor_parallel_size=int(server_cfg.get("tensor_parallel_size", 1) or 1),
            vllm_gpu_memory_utilization=float(server_cfg.get("gpu_memory_utilization", 0.9) or 0.9),
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


def build_legacy_rl_trl_argv(job: RLJobConfig) -> list[str]:
    argv: list[str] = [
        "--data", job.train_manifest,
        "--output-dir", job.output_dir,
        "--model-path", job.policy_init_from,
        "--reference-model-path", job.reference_model or job.base_model,
        "--num-iterations", str(job.num_iterations),
        "--rollout-count", str(job.rollout_count),
        "--num-generations", str(job.num_generations),
        "--rollout-max-turns", str(job.rollout_max_turns),
        "--learning-rate", str(job.learning_rate),
        "--num-train-epochs", str(job.num_train_epochs),
        "--per-device-train-batch-size", str(job.per_device_train_batch_size),
        "--gradient-accumulation-steps", str(job.gradient_accumulation_steps),
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
        "--num-preview-frames", str(job.num_preview_frames),
        "--rl-reward-version", job.reward_version,
        "--rl-steps-per-generation", "1",
        "--use-vllm", "true",
        "--vllm-mode", job.vllm_mode,
        "--vllm-tensor-parallel-size", str(job.vllm_tensor_parallel_size),
        "--vllm-gpu-memory-utilization", str(job.vllm_gpu_memory_utilization),
        "--rl-open-ended-judge-enabled", str(bool(job.reward_config.get("open_ended_judge_enabled", True))).lower(),
        "--rl-reward-config-json", json.dumps(job.reward_config, ensure_ascii=False),
    ]
    if job.gradient_checkpointing:
        argv.append("--gradient-checkpointing")
    if job.bf16:
        argv.append("--bf16")
    if job.fp16:
        argv.append("--fp16")
    _append_flag(argv, "--include-splits", job.include_splits)
    _append_flag(argv, "--data-root", job.data_root)
    _append_flag(argv, "--eval-data", job.eval_manifest)
    _append_flag(argv, "--eval-data-root", job.eval_data_root)
    _append_flag(argv, "--eval-include-splits", job.eval_include_splits)
    _append_flag(argv, "--deepspeed", job.deepspeed_config_path)
    _append_flag(argv, "--vllm-guided-decoding-regex", job.vllm_guided_decoding_regex)
    return argv


def run_rl_job(job: RLJobConfig) -> RLRunResult:
    if not job.train_manifest:
        raise ValueError("RL train manifest is required")
    if not job.policy_init_from:
        raise ValueError("RL policy_init_from is required")
    ensure_fa3_training_ready(require_gpu=True)
    output_dir = Path(job.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = build_legacy_rl_trl_argv(job)
    launch_manifest_path = write_json(
        {
            "entrypoint": "train_saver_rl_trl.py",
            "argv": argv,
            "config": job.to_dict(),
        },
        output_dir / "rl_launch_manifest.json",
    )
    result = legacy_train_saver_rl_trl.main(argv)
    summary_payload = dict(result or {})
    summary_payload.update(
        {
            "run_name": job.run_name,
            "output_dir": str(output_dir),
            "legacy_entrypoint": "train_saver_rl_trl.py",
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
            f"reference_model={job.reference_model or job.base_model}",
            f"deepspeed={job.deepspeed_config_path or '(none)'}",
        ],
    )
