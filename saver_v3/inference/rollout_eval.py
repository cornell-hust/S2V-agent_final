from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from saver_v3.metrics.evaluation import RolloutEvaluationConfig, run_rollout_evaluation
from saver_v3.metrics.legacy_metrics import PAPER_MAIN_METRIC_KEYS
from saver_v3.model.vllm_generation import build_vllm_policy_from_model_path
from saver_v3.cli.common import load_yaml_mapping, write_json
from saver_v3.common import distributed_runtime_from_env, runtime_log
from saver_v3.data.config import (
    DEFAULT_POLICY_MAX_NEW_TOKENS,
    DEFAULT_ROLLOUT_MAX_TURNS,
    SaverAgentConfig,
    saver_config_from_mapping,
)


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


@dataclass
class StepRolloutEvalConfig:
    base_model: str
    data_path: str
    output_dir: str
    data_root: str = ""
    materialized_items_path: str = ""
    require_materialized_cache: bool = False
    include_splits: str = ""
    max_records: int = 0
    epoch_index: int = 0
    max_turns: int = DEFAULT_ROLLOUT_MAX_TURNS
    rollout_batch_size: int = 4
    progress_every: int = 10
    policy_max_new_tokens: int = DEFAULT_POLICY_MAX_NEW_TOKENS
    use_generation_cache: bool = True
    enable_semantic_replay: bool = True
    semantic_replay_max_new_tokens: int = DEFAULT_POLICY_MAX_NEW_TOKENS
    max_total_images: int = 28
    max_seq_length: int = 8192
    keep_recent_text_messages: int = 20
    keep_recent_tool_image_messages: int = 0
    max_image_side: int = 640
    max_image_pixels: int = 0
    proposal_model_path: str = ""
    proposal_torch_dtype: str = "auto"
    proposal_device: str = ""
    verifier_backend: str = "qwen_self_verifier"
    verifier_model_path: str = ""
    verifier_torch_dtype: str = "auto"
    verifier_device_map: str = "auto"
    verifier_attn_implementation: str = ""
    verifier_max_new_tokens: int = 512
    attach_reference_diagnostics: bool = False
    enable_semantic_metrics: bool = True
    semantic_metrics: Sequence[str] | str = "qa_accuracy,bertscore"
    semantic_judge_base_url: str = ""
    semantic_judge_model: str = ""
    semantic_judge_cache_path: str = ""
    semantic_judge_timeout_sec: float = 30.0
    semantic_bertscore_model_path: str = ""
    torch_dtype: str = "auto"
    device_map: str = "auto"
    attn_implementation: str = ""
    vllm_mode: str = "colocate"
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.9
    vllm_guided_decoding_regex: str = ""
    vllm_server_host: str = "127.0.0.1"
    vllm_server_port: int = 8000
    vllm_server_timeout: float = 240.0
    vllm_server_auto_launch: bool = False
    vllm_server_per_rank: bool = False
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    saver_config: Dict[str, Any] = field(default_factory=lambda: SaverAgentConfig().to_dict())

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "StepRolloutEvalConfig":
        server = dict(_mapping(mapping.get("server")))
        client = dict(_mapping(mapping.get("client")))
        io_cfg = dict(_mapping(mapping.get("io")))
        evaluation = dict(_mapping(mapping.get("evaluation")))
        proposal = dict(_mapping(mapping.get("proposal")))
        verifier = dict(_mapping(mapping.get("verifier")))
        semantic = dict(_mapping(mapping.get("semantic_metrics")))
        saver_config = saver_config_from_mapping(mapping)
        data_path = str(io_cfg.get("data_path") or "").strip()
        if not data_path:
            raise ValueError(
                "run_sft_rollout_eval_vllm now requires io.data_path pointing to raw SAVER JSONL. "
                "The legacy io.input_manifest compact-trace eval path has been retired."
            )
        semantic_metrics = semantic.get("metrics", "qa_accuracy,bertscore")
        return cls(
            base_model=str(mapping.get("base_model") or "").strip(),
            data_path=data_path,
            output_dir=str(io_cfg.get("output_dir") or "").strip(),
            data_root=str(io_cfg.get("data_root") or "").strip(),
            materialized_items_path=str(io_cfg.get("materialized_items_path") or "").strip(),
            require_materialized_cache=_bool(io_cfg.get("require_materialized_cache"), False),
            include_splits=str(io_cfg.get("include_splits") or "").strip(),
            max_records=int(io_cfg.get("max_records", 0) or 0),
            epoch_index=int(evaluation.get("epoch_index", 0) or 0),
            max_turns=int(evaluation.get("max_turns", DEFAULT_ROLLOUT_MAX_TURNS) or DEFAULT_ROLLOUT_MAX_TURNS),
            rollout_batch_size=int(evaluation.get("rollout_batch_size", client.get("rollout_batch_size", 4)) or 4),
            progress_every=int(evaluation.get("progress_every", 10) or 10),
            policy_max_new_tokens=int(
                client.get("max_tokens", DEFAULT_POLICY_MAX_NEW_TOKENS) or DEFAULT_POLICY_MAX_NEW_TOKENS
            ),
            use_generation_cache=_bool(client.get("use_generation_cache"), True),
            enable_semantic_replay=_bool(evaluation.get("enable_semantic_replay"), True),
            semantic_replay_max_new_tokens=int(
                evaluation.get("semantic_replay_max_new_tokens", DEFAULT_POLICY_MAX_NEW_TOKENS)
                or DEFAULT_POLICY_MAX_NEW_TOKENS
            ),
            max_total_images=int(client.get("max_total_images", 28) or 28),
            max_seq_length=int(client.get("max_seq_length", 8192) or 8192),
            keep_recent_text_messages=int(client.get("keep_recent_text_messages", 20) or 20),
            keep_recent_tool_image_messages=int(client.get("keep_recent_tool_image_messages", 0) or 0),
            max_image_side=int(client.get("max_image_side", 640) or 640),
            max_image_pixels=int(client.get("max_image_pixels", 0) or 0),
            proposal_model_path=str(proposal.get("model_path") or evaluation.get("proposal_model_path") or "").strip(),
            proposal_torch_dtype=str(proposal.get("torch_dtype") or "auto"),
            proposal_device=str(proposal.get("device") or "").strip(),
            verifier_backend=str(verifier.get("backend") or "qwen_self_verifier"),
            verifier_model_path=str(verifier.get("model_path") or "").strip(),
            verifier_torch_dtype=str(verifier.get("torch_dtype") or "auto"),
            verifier_device_map=str(verifier.get("device_map") or "auto"),
            verifier_attn_implementation=str(verifier.get("attn_implementation") or "").strip(),
            verifier_max_new_tokens=int(verifier.get("max_new_tokens", 512) or 512),
            attach_reference_diagnostics=_bool(verifier.get("attach_reference_diagnostics"), False),
            enable_semantic_metrics=_bool(semantic.get("enabled"), True),
            semantic_metrics=semantic_metrics,
            semantic_judge_base_url=str(semantic.get("judge_base_url") or "").strip(),
            semantic_judge_model=str(semantic.get("judge_model") or "").strip(),
            semantic_judge_cache_path=str(semantic.get("judge_cache_path") or "").strip(),
            semantic_judge_timeout_sec=float(semantic.get("judge_timeout_sec", 30.0) or 30.0),
            semantic_bertscore_model_path=str(semantic.get("bertscore_model_path") or "").strip(),
            torch_dtype=str(mapping.get("torch_dtype") or server.get("dtype") or "auto"),
            device_map=str(mapping.get("device_map") or "auto"),
            attn_implementation=str(mapping.get("attn_implementation") or "").strip(),
            vllm_mode=str(server.get("mode") or "colocate"),
            vllm_tensor_parallel_size=int(server.get("tensor_parallel_size", 1) or 1),
            vllm_gpu_memory_utilization=float(server.get("gpu_memory_utilization", 0.9) or 0.9),
            vllm_guided_decoding_regex=str(client.get("guided_decoding_regex") or "").strip(),
            vllm_server_host=str(server.get("host") or "127.0.0.1"),
            vllm_server_port=int(server.get("port", 8000) or 8000),
            vllm_server_timeout=float(server.get("timeout_sec", 240.0) or 240.0),
            vllm_server_auto_launch=_bool(server.get("auto_launch"), False),
            vllm_server_per_rank=_bool(server.get("per_rank"), False),
            temperature=float(client.get("temperature", 0.0) or 0.0),
            top_p=(None if client.get("top_p") is None else float(client.get("top_p") or 1.0)),
            top_k=(None if client.get("top_k") is None else int(client.get("top_k") or -1)),
            repetition_penalty=(
                None if client.get("repetition_penalty") is None else float(client.get("repetition_penalty") or 1.0)
            ),
            saver_config=saver_config.to_dict(),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "StepRolloutEvalConfig":
        return cls.from_mapping(load_yaml_mapping(path))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build_vllm_args(config: StepRolloutEvalConfig) -> argparse.Namespace:
    return argparse.Namespace(
        use_vllm=True,
        vllm_mode=config.vllm_mode,
        vllm_tensor_parallel_size=config.vllm_tensor_parallel_size,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        vllm_guided_decoding_regex=config.vllm_guided_decoding_regex,
        vllm_server_host=config.vllm_server_host,
        vllm_server_port=config.vllm_server_port,
        vllm_server_timeout=config.vllm_server_timeout,
        vllm_server_auto_launch=config.vllm_server_auto_launch,
        vllm_server_per_rank=config.vllm_server_per_rank,
        torch_dtype=config.torch_dtype,
        device_map=config.device_map,
        attn_implementation=config.attn_implementation,
        max_seq_length=int(config.max_seq_length),
        policy_max_new_tokens=int(config.policy_max_new_tokens),
        max_new_tokens=int(config.policy_max_new_tokens),
        max_total_images=int(config.max_total_images),
        keep_recent_text_messages=int(config.keep_recent_text_messages),
        keep_recent_tool_image_messages=int(config.keep_recent_tool_image_messages),
        max_image_side=int(config.max_image_side),
        max_image_pixels=int(config.max_image_pixels),
    )


def _paper_main_metrics(summary: Mapping[str, Any]) -> Dict[str, float]:
    return {key: float(summary.get(key, 0.0) or 0.0) for key in PAPER_MAIN_METRIC_KEYS}


def _close_policy(policy: Any) -> None:
    runtime = getattr(policy, "vllm_runtime", None)
    close_fn = getattr(runtime, "close", None)
    if callable(close_fn):
        close_fn()


def run_step_rollout_eval_job(config: StepRolloutEvalConfig) -> Dict[str, Any]:
    if not config.base_model:
        raise ValueError("rollout eval base_model is required")
    if not config.output_dir:
        raise ValueError("rollout eval io.output_dir is required")

    runtime = distributed_runtime_from_env()
    runtime_log(
        (
            "SFT rollout eval startup through saver_v3.metrics.evaluation: "
            f"data={config.data_path} output_dir={config.output_dir} model={config.base_model} "
            f"include_splits={config.include_splits or 'all'}"
        ),
        runtime=runtime,
        main_process_only=True,
    )
    saver_config = saver_config_from_mapping(config.saver_config)
    eval_config = RolloutEvaluationConfig(
        data_path=config.data_path,
        data_root=config.data_root,
        materialized_items_path=config.materialized_items_path,
        require_materialized_cache=config.require_materialized_cache,
        include_splits=config.include_splits or None,
        max_records=config.max_records,
        rollout_max_turns=config.max_turns,
        rollout_batch_size=config.rollout_batch_size,
        policy_max_new_tokens=config.policy_max_new_tokens,
        use_generation_cache=config.use_generation_cache,
        enable_semantic_replay=config.enable_semantic_replay,
        semantic_replay_max_new_tokens=config.semantic_replay_max_new_tokens,
        max_total_images=config.max_total_images,
        max_seq_length=config.max_seq_length,
        keep_recent_text_messages=config.keep_recent_text_messages,
        keep_recent_tool_image_messages=config.keep_recent_tool_image_messages,
        max_image_side=config.max_image_side,
        max_image_pixels=config.max_image_pixels,
        proposal_model_path=config.proposal_model_path,
        proposal_torch_dtype=config.proposal_torch_dtype,
        proposal_device=config.proposal_device,
        verifier_backend=config.verifier_backend,
        verifier_model_path=config.verifier_model_path,
        verifier_torch_dtype=config.verifier_torch_dtype,
        verifier_device_map=config.verifier_device_map,
        verifier_attn_implementation=config.verifier_attn_implementation,
        verifier_max_new_tokens=config.verifier_max_new_tokens,
        attach_reference_diagnostics=config.attach_reference_diagnostics,
        progress_every=config.progress_every,
        enable_semantic_metrics=config.enable_semantic_metrics,
        semantic_metrics=config.semantic_metrics,
        semantic_judge_base_url=config.semantic_judge_base_url,
        semantic_judge_model=config.semantic_judge_model,
        semantic_judge_cache_path=config.semantic_judge_cache_path,
        semantic_judge_timeout_sec=config.semantic_judge_timeout_sec,
        semantic_bertscore_model_path=config.semantic_bertscore_model_path,
        saver_config=saver_config,
    )
    policy = build_vllm_policy_from_model_path(
        args=_build_vllm_args(config),
        runtime=runtime,
        model_path=config.base_model,
        prefer_direct_local_rank_runtime=True,
        max_new_tokens=config.policy_max_new_tokens,
        max_total_images=config.max_total_images,
        max_seq_length=config.max_seq_length,
        keep_recent_tool_image_messages=config.keep_recent_tool_image_messages,
        keep_recent_text_messages=config.keep_recent_text_messages,
        max_image_side=config.max_image_side,
        max_image_pixels=config.max_image_pixels,
        do_sample=config.temperature > 0,
        temperature=config.temperature if config.temperature > 0 else None,
        top_p=config.top_p,
        top_k=config.top_k,
        repetition_penalty=config.repetition_penalty,
        use_generation_cache=config.use_generation_cache,
    )
    try:
        summary = dict(
            run_rollout_evaluation(
                policy,
                eval_config=eval_config,
                output_dir=config.output_dir,
                epoch_index=config.epoch_index,
                runtime=runtime,
            )
            or {}
        )
    finally:
        _close_policy(policy)
    summary["paper_main_metrics"] = _paper_main_metrics(summary)
    summary["entrypoint"] = "saver_v3.cli.run_sft_rollout_eval_vllm"
    summary["config"] = config.to_dict()
    if runtime.is_main_process:
        output_dir = Path(config.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(summary, output_dir / "rollout_eval_wrapper_summary.json")
    return summary
