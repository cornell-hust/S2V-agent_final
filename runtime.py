from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from saver_v3.sft.training import run_standard_sft
from saver_v3.cli.common import apply_config_overrides, load_yaml_mapping, write_json
from saver_v3.common import distributed_runtime_from_env, ensure_fa3_training_ready, runtime_log
from saver_v3.data.config import SaverAgentConfig, saver_config_from_mapping
from saver_v3.data.prepared_metadata import ensure_prepared_sft_metadata, load_prepared_sft_metadata
from saver_v3.data.materialized_cache import MATERIALIZED_SFT_MESSAGES_FORMAT, ensure_materialized_cache_metadata
from saver_v3.data.protocol_signature import build_protocol_signature, infer_teacher_role_from_metadata


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return value or {}


def _saver_config_from_mapping(mapping: Mapping[str, Any]) -> SaverAgentConfig:
    return saver_config_from_mapping(mapping)


def _saver_config_from_dict(payload: Mapping[str, Any] | None) -> SaverAgentConfig:
    return _saver_config_from_mapping(payload or {})


@dataclass
class SFTJobConfig:
    run_name: str
    output_dir: str
    prepared_data_path: str
    include_splits: str
    num_workers: int
    dataloader_prefetch_factor: int
    dataloader_persistent_workers: bool
    epochs: float
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_grad_norm: float
    log_every_n_steps: int
    save_every_n_steps: int
    save_total_limit: int
    report_to: list[str]
    seed: int
    ddp_find_unused_parameters: bool | None
    bf16: bool
    fp16: bool
    model_path: str
    torch_dtype: str
    gradient_checkpointing: bool
    attn_implementation: str
    max_seq_length: int
    max_total_images: int
    max_image_side: int
    max_image_pixels: int
    keep_recent_text_messages: int
    keep_recent_tool_image_messages: int
    trust_remote_code: bool
    train_mode: str
    proposal_model_path: str = ""
    proposal_torch_dtype: str = "auto"
    proposal_device: str = ""
    deepspeed_config_path: str | None = None
    log_dir: str = ""
    rollout_eval_output_dir: str = ""
    resume_from_checkpoint: str = ""
    use_sample_weights: bool = False
    materialized_messages_path: str = ""
    require_materialized_cache: bool = False
    saver_config: Dict[str, Any] = field(default_factory=lambda: SaverAgentConfig().to_dict())

    @classmethod
    def from_files(
        cls,
        *,
        config_path: str,
        model_config_path: str,
        attention_config_path: str,
        deepspeed_config_path: str | None = None,
        config_overrides: Sequence[str] | None = None,
    ) -> "SFTJobConfig":
        config = apply_config_overrides(load_yaml_mapping(config_path), config_overrides)
        model_config = load_yaml_mapping(model_config_path)
        attention_config = load_yaml_mapping(attention_config_path)
        if str(attention_config.get("policy_name") or "").strip() != "fa3_only":
            raise ValueError("idea2_v3 requires attention policy fa3_only")
        if str(model_config.get("attn_implementation") or "").strip() != "flash_attention_3":
            raise ValueError("model config must lock attn_implementation to flash_attention_3")
        if str(model_config.get("train_mode") or "").strip().lower() != "full":
            raise ValueError("idea2_v3 SFT only supports full-model training")

        data = dict(_mapping(config.get("data")))
        optimization = dict(_mapping(config.get("optimization")))
        logging_cfg = dict(_mapping(config.get("logging")))
        distributed = dict(_mapping(config.get("distributed")))
        proposal = dict(_mapping(config.get("proposal")))
        sequence_cfg = dict(_mapping(model_config.get("sequence")))
        vision_cfg = dict(_mapping(model_config.get("vision")))
        saver_config = _saver_config_from_mapping(config)

        prepared_data_path = str(data.get("prepared_data_path") or "").strip()
        if not prepared_data_path:
            raise ValueError(
                "idea2_v3 SFT now requires data.prepared_data_path pointing to compact_trace_v4 JSONL. "
                "The legacy data.train_manifest path has been removed from the official training entrypoint."
            )

        return cls(
            run_name=str(config.get("run_name") or "qwen3_vl_8b_full_sft_ds8"),
            output_dir=str(config.get("output_dir") or "artifacts/sft/qwen3_vl_8b_full"),
            prepared_data_path=prepared_data_path,
            materialized_messages_path=str(data.get("materialized_messages_path") or "").strip(),
            require_materialized_cache=bool(data.get("require_materialized_cache", False)),
            include_splits=str(data.get("include_splits") or "train").strip(),
            num_workers=int(data.get("num_workers", 0) or 0),
            dataloader_prefetch_factor=int(data.get("dataloader_prefetch_factor", 0) or 0),
            dataloader_persistent_workers=bool(data.get("dataloader_persistent_workers", False)),
            epochs=float(optimization.get("epochs", 1.0) or 1.0),
            learning_rate=float(optimization.get("learning_rate", 1e-5) or 1e-5),
            weight_decay=float(optimization.get("weight_decay", 0.0) or 0.0),
            warmup_ratio=float(optimization.get("warmup_ratio", 0.0) or 0.0),
            lr_scheduler_type=str(optimization.get("lr_scheduler_type") or "cosine"),
            per_device_train_batch_size=int(optimization.get("per_device_train_batch_size", 1) or 1),
            gradient_accumulation_steps=int(optimization.get("gradient_accumulation_steps", 1) or 1),
            max_grad_norm=float(optimization.get("max_grad_norm", 1.0) or 1.0),
            log_every_n_steps=int(logging_cfg.get("log_every_n_steps", 10) or 10),
            save_every_n_steps=int(logging_cfg.get("save_every_n_steps", 500) or 500),
            save_total_limit=int(logging_cfg.get("save_total_limit", 2) or 2),
            report_to=[str(item) for item in list(logging_cfg.get("report_to") or [])],
            seed=int(distributed.get("seed", 42) or 42),
            ddp_find_unused_parameters=(
                None if distributed.get("ddp_find_unused_parameters") is None else bool(distributed.get("ddp_find_unused_parameters"))
            ),
            bf16=bool(distributed.get("bf16", True)),
            fp16=bool(distributed.get("fp16", False)),
            model_path=str(config.get("base_model") or model_config.get("base_model") or "").strip(),
            torch_dtype=str(model_config.get("torch_dtype") or "bfloat16"),
            gradient_checkpointing=bool(model_config.get("gradient_checkpointing", True)),
            attn_implementation=str(model_config.get("attn_implementation") or "flash_attention_3"),
            max_seq_length=int(sequence_cfg.get("max_length", 8192) or 8192),
            max_total_images=int(vision_cfg.get("max_images_per_sample", vision_cfg.get("max_total_images", 28)) or 28),
            max_image_side=int(vision_cfg.get("max_image_side", 640) or 640),
            max_image_pixels=int(vision_cfg.get("max_image_pixels", 0) or 0),
            keep_recent_text_messages=int(optimization.get("keep_recent_text_messages", 20) or 20),
            keep_recent_tool_image_messages=int(optimization.get("keep_recent_tool_image_messages", 0) or 0),
            trust_remote_code=bool(model_config.get("trust_remote_code", True)),
            train_mode=str(model_config.get("train_mode") or "full"),
            proposal_model_path=str(proposal.get("model_path") or "").strip(),
            proposal_torch_dtype=str(proposal.get("torch_dtype") or "auto"),
            proposal_device=str(proposal.get("device") or "").strip(),
            deepspeed_config_path=deepspeed_config_path or str(config.get("deepspeed_config") or "").strip() or None,
            log_dir=str(logging_cfg.get("log_dir") or "").strip(),
            rollout_eval_output_dir=str(logging_cfg.get("rollout_eval_output_dir") or "").strip(),
            resume_from_checkpoint=str(config.get("resume_from_checkpoint") or "").strip(),
            use_sample_weights=bool(optimization.get("use_sample_weights", False)),
            saver_config=saver_config.to_dict(),
        )


@dataclass
class SFTTrainingResult:
    run_name: str
    output_dir: str
    num_train_examples: int
    train_loss: float = 0.0
    summary_path: str = ""
    launch_manifest_path: str = ""
    standard_result: Dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_sft_job(job: SFTJobConfig) -> SFTTrainingResult:
    if not job.prepared_data_path:
        raise ValueError("SFT prepared_data_path is required")
    saver_config = _saver_config_from_dict(job.saver_config)
    prepared_metadata = load_prepared_sft_metadata(job.prepared_data_path)
    expected_protocol_signature = build_protocol_signature(
        config=saver_config,
        teacher_role=infer_teacher_role_from_metadata(prepared_metadata),
    )
    ensure_prepared_sft_metadata(
        job.prepared_data_path,
        config=saver_config,
        require_config_match=True,
        expected_protocol_signature=expected_protocol_signature,
    )
    if job.materialized_messages_path:
        ensure_materialized_cache_metadata(
            job.materialized_messages_path,
            expected_format=MATERIALIZED_SFT_MESSAGES_FORMAT,
            expected_source_path=job.prepared_data_path,
            expected_include_splits=job.include_splits or None,
            expected_config=saver_config,
            expected_protocol_signature=expected_protocol_signature,
            require_config_match=True,
            require_source=True,
        )
    ensure_fa3_training_ready(require_gpu=True)
    runtime = distributed_runtime_from_env()
    output_dir = Path(job.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_log(
        (
            "delegating v3 SFT to saver_agent.run_standard_sft: "
            f"prepared_data={job.prepared_data_path} include_splits={job.include_splits or 'all'} "
            f"materialized_messages={job.materialized_messages_path or '(none)'} "
            f"model_path={job.model_path} deepspeed={job.deepspeed_config_path or '(none)'}"
        ),
        runtime=runtime,
        main_process_only=True,
    )
    launch_manifest_path = write_json(
        {
            "entrypoint": "saver_v3.sft.training.run_standard_sft",
            "config": job.to_dict(),
        },
        output_dir / "sft_launch_manifest.json",
    )
    standard_result = dict(
        run_standard_sft(
            prepared_data_path=job.prepared_data_path,
            materialized_messages_path=job.materialized_messages_path,
            require_materialized_cache=job.require_materialized_cache,
            include_splits=job.include_splits or None,
            model_path=job.model_path,
            output_dir=str(output_dir),
            log_dir=job.log_dir,
            rollout_eval_output_dir=job.rollout_eval_output_dir,
            resume_from_checkpoint=job.resume_from_checkpoint,
            torch_dtype=job.torch_dtype,
            attn_implementation=job.attn_implementation,
            gradient_checkpointing=job.gradient_checkpointing,
            use_lora=False,
            learning_rate=job.learning_rate,
            num_train_epochs=job.epochs,
            per_device_train_batch_size=job.per_device_train_batch_size,
            gradient_accumulation_steps=job.gradient_accumulation_steps,
            logging_steps=job.log_every_n_steps,
            save_steps=job.save_every_n_steps,
            save_total_limit=job.save_total_limit,
            warmup_ratio=job.warmup_ratio,
            weight_decay=job.weight_decay,
            max_grad_norm=job.max_grad_norm,
            bf16=job.bf16,
            fp16=job.fp16,
            max_image_side=job.max_image_side,
            max_image_pixels=job.max_image_pixels,
            keep_recent_tool_image_messages=job.keep_recent_tool_image_messages,
            max_total_images=job.max_total_images,
            max_seq_length=job.max_seq_length,
            keep_recent_text_messages=job.keep_recent_text_messages,
            dataloader_num_workers=job.num_workers,
            dataloader_prefetch_factor=job.dataloader_prefetch_factor,
            dataloader_persistent_workers=job.dataloader_persistent_workers,
            lr_scheduler_type=job.lr_scheduler_type,
            report_to=job.report_to,
            seed=job.seed,
            ddp_find_unused_parameters=job.ddp_find_unused_parameters,
            deepspeed=job.deepspeed_config_path or "",
            saver_config=saver_config,
            use_sample_weights=job.use_sample_weights,
            proposal_model_path=job.proposal_model_path,
            proposal_torch_dtype=job.proposal_torch_dtype,
            proposal_device=job.proposal_device,
        )
        or {}
    )
    summary_payload = dict(standard_result)
    summary_payload.update(
        {
            "run_name": job.run_name,
            "output_dir": str(output_dir),
            "launch_manifest_path": str(launch_manifest_path),
            "entrypoint": "saver_v3.cli.train_sft_ds",
        }
    )
    summary_path = write_json(summary_payload, output_dir / "sft_summary.json")
    return SFTTrainingResult(
        run_name=job.run_name,
        output_dir=str(output_dir),
        num_train_examples=int(standard_result.get("num_examples", 0) or 0),
        train_loss=float(standard_result.get("train_loss", 0.0) or 0.0),
        summary_path=str(summary_path),
        launch_manifest_path=str(launch_manifest_path),
        standard_result=summary_payload,
        notes=[
            "SFT now delegates to saver_agent.run_standard_sft.",
            f"prepared_data_path={job.prepared_data_path}",
            f"materialized_messages_path={job.materialized_messages_path or '(none)'}",
            f"include_splits={job.include_splits or 'all'}",
        ],
    )


def _sft_job_to_dict(job: SFTJobConfig) -> Dict[str, Any]:
    return asdict(job)


SFTJobConfig.to_dict = _sft_job_to_dict  # type: ignore[attr-defined]
