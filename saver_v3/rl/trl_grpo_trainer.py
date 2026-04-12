from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from saver_v3.data.dataset import SaverAgentDataset
from saver_v3.common.experiment_logging import append_jsonl, utc_timestamp, write_json
from saver_v3.rl.grpo_trainer_env import (
    _build_rl_authority_checkpoint_callback,
    _teardown_trainer_iteration_runtime,
    _write_rl_checkpoint_authority_metadata,
)
from saver_v3.model.qwen_policy import QwenGenerationPolicy, load_generation_processor_for_checkpoint
from saver_v3.core.reward import DEFAULT_RL_REWARD_VERSION
from saver_v3.core.rollout import SaverRolloutRunner
from saver_v3.common.runtime import distributed_runtime_from_env, runtime_log
from saver_v3.rl.timesearch_aligned_grpo_trainer import create_timesearch_aligned_grpo_trainer
from saver_v3.sft.training import _unwrap_model, load_qwen_model_and_processor
from saver_v3.rl.trl_compat import patch_vllm_guided_decoding_params
from saver_v3.model import vllm_generation as shared_vllm_generation


def _build_limit_mm_per_prompt(args: Any) -> Dict[str, int]:
    return shared_vllm_generation._build_limit_mm_per_prompt(args)


def _resolve_rollout_episode_batch_size(args: Any) -> int:
    return shared_vllm_generation._resolve_rollout_episode_batch_size(args)


def build_vllm_runtime_settings(args: Any) -> Dict[str, Any]:
    return shared_vllm_generation.build_vllm_runtime_settings(args)


def _resolve_vllm_base_model_path(model_path: str | Path) -> str:
    return shared_vllm_generation._resolve_vllm_base_model_path(model_path)


def _maybe_reset_vllm_prefix_cache(llm: Any) -> None:
    return shared_vllm_generation._maybe_reset_vllm_prefix_cache(llm)


def _iter_named_weights_for_vllm(model: Any):
    yield from shared_vllm_generation._iter_named_weights_for_vllm(model)


class _SAVERVLLMClient(shared_vllm_generation._SAVERVLLMClient):
    pass


class _VllmColocateRuntime(shared_vllm_generation._VllmColocateRuntime):
    pass


class _VllmServerRuntime(shared_vllm_generation._VllmServerRuntime):
    def __init__(
        self,
        *,
        args: Any,
        runtime: Any,
        model_path: str | Path,
    ) -> None:
        self.args = args
        self.runtime = runtime or distributed_runtime_from_env()
        self.settings = build_vllm_runtime_settings(args)
        self.enabled = bool(self.settings["use_vllm"])
        self.base_model_path = _resolve_vllm_base_model_path(model_path)
        self.client: Any = None
        self._last_loaded_step: Optional[int] = None
        if not self.enabled:
            return
        if self.settings["vllm_mode"] != "server":
            raise ValueError(f"Unsupported vLLM mode: {self.settings['vllm_mode']}")
        if bool(getattr(self.runtime, "is_main_process", True)):
            self.client = _SAVERVLLMClient(
                host=str(self.settings["vllm_server_host"]),
                server_port=int(self.settings["vllm_server_port"]),
                connection_timeout=float(self.settings["vllm_server_timeout"]),
            )
            self.client.init_communicator()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()


def create_vllm_runtime(
    *,
    args: Any,
    runtime: Any,
    model_path: str | Path,
) -> Any:
    settings = build_vllm_runtime_settings(args)
    if not bool(settings["use_vllm"]):
        return None
    if settings["vllm_mode"] == "server":
        return _VllmServerRuntime(args=args, runtime=runtime, model_path=model_path)
    if settings["vllm_mode"] == "colocate":
        return _VllmColocateRuntime(args=args, runtime=runtime, model_path=model_path)
    raise ValueError(f"Unsupported vLLM mode: {settings['vllm_mode']}")


class VllmQwenGenerationPolicy(shared_vllm_generation.VllmQwenGenerationPolicy):
    pass


def _build_inline_vllm_policy_factory(
    *,
    args: Any,
    vllm_runtime: Any,
) -> Callable[..., QwenGenerationPolicy]:
    return shared_vllm_generation.build_inline_vllm_policy_factory(
        args=args,
        vllm_runtime=vllm_runtime,
        policy_class=VllmQwenGenerationPolicy,
    )


def build_recovery_vllm_policy_factory(
    *,
    args: Any,
) -> Callable[..., Any]:
    return shared_vllm_generation.build_recovery_vllm_policy_factory(
        args=args,
        runtime_builder=create_vllm_runtime,
        policy_class=VllmQwenGenerationPolicy,
        processor_loader=load_generation_processor_for_checkpoint,
        hf_policy_class=QwenGenerationPolicy,
    )


def _build_vllm_trainer_class_transform(
    *,
    args: Any,
    vllm_runtime: Optional[Any],
) -> Callable[[type], type]:
    if vllm_runtime is None:
        raise RuntimeError("vLLM route requested, but no runtime was provided.")

    def _transform(base_trainer_class: type) -> type:
        class _VllmNativeGRPOTrainer(base_trainer_class):
            def __init__(self, *trainer_args: Any, **trainer_kwargs: Any) -> None:
                super().__init__(*trainer_args, **trainer_kwargs)
                self._vllm_runtime = vllm_runtime

            def _build_policy(self, model: Any, *, use_generation_cache: bool) -> QwenGenerationPolicy:
                return VllmQwenGenerationPolicy(
                    runtime=vllm_runtime,
                    source_model=_unwrap_model(model),
                    step_resolver=lambda: int(getattr(getattr(self, "state", None), "global_step", 0) or 0),
                    guided_decoding_regex=str(getattr(args, "vllm_guided_decoding_regex", "") or ""),
                    processor=self.processor,
                    max_new_tokens=self.policy_max_new_tokens,
                    max_total_images=self.max_total_images,
                    max_seq_length=self.max_seq_length,
                    keep_recent_tool_image_messages=self.keep_recent_tool_image_messages,
                    keep_recent_text_messages=self.keep_recent_text_messages,
                    max_image_side=self.max_image_side,
                    max_image_pixels=self.max_image_pixels,
                    do_sample=self.policy_do_sample,
                    temperature=self.policy_temperature,
                    top_p=self.policy_top_p,
                    top_k=self.policy_top_k,
                    repetition_penalty=self.policy_repetition_penalty,
                    use_generation_cache=bool(use_generation_cache),
                )

        _VllmNativeGRPOTrainer.__name__ = f"Vllm{base_trainer_class.__name__}"
        _VllmNativeGRPOTrainer.__qualname__ = f"Vllm{base_trainer_class.__qualname__}"
        return _VllmNativeGRPOTrainer

    return _transform


def _build_runtime_policy_builder(
    *,
    args: Any,
    vllm_runtime: Optional[Any],
) -> Callable[..., Any]:
    def _builder(
        *,
        model: Any,
        use_generation_cache: bool,
        step_resolver: Optional[Callable[[], int]],
        processor: Any,
        max_new_tokens: int,
        max_total_images: int,
        max_seq_length: int,
        keep_recent_tool_image_messages: int,
        keep_recent_text_messages: int,
        max_image_side: int,
        max_image_pixels: int,
        do_sample: bool,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        repetition_penalty: Optional[float],
    ) -> Any:
        if bool(getattr(args, "use_vllm", True)) and vllm_runtime is not None:
            return VllmQwenGenerationPolicy(
                runtime=vllm_runtime,
                source_model=_unwrap_model(model),
                step_resolver=step_resolver or (lambda: 0),
                guided_decoding_regex=str(getattr(args, "vllm_guided_decoding_regex", "") or ""),
                processor=processor,
                max_new_tokens=int(max_new_tokens),
                max_total_images=int(max_total_images),
                max_seq_length=int(max_seq_length),
                keep_recent_tool_image_messages=int(keep_recent_tool_image_messages),
                keep_recent_text_messages=int(keep_recent_text_messages),
                max_image_side=int(max_image_side),
                max_image_pixels=int(max_image_pixels),
                do_sample=bool(do_sample),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                use_generation_cache=bool(use_generation_cache),
            )
        return QwenGenerationPolicy.from_components(
            model=_unwrap_model(model),
            processor=processor,
            max_new_tokens=int(max_new_tokens),
            max_total_images=int(max_total_images),
            max_seq_length=int(max_seq_length),
            keep_recent_tool_image_messages=int(keep_recent_tool_image_messages),
            keep_recent_text_messages=int(keep_recent_text_messages),
            max_image_side=int(max_image_side),
            max_image_pixels=int(max_image_pixels),
            do_sample=bool(do_sample),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            use_generation_cache=bool(use_generation_cache),
        )

    return _builder


def _attach_vllm_route_to_trainer(*args: Any, **kwargs: Any) -> Optional[_VllmColocateRuntime]:
    del args, kwargs
    raise RuntimeError(
        "_attach_vllm_route_to_trainer is deprecated. trl_vllm_grpo must provide the vLLM route via trainer_class_transform."
    )


def create_trl_vllm_grpo_trainer(
    *,
    args: Any,
    model: Any,
    processor: Any,
    reference_model: Any,
    train_items: List[Dict[str, Any]],
    checkpoint_dir: str | Path,
    iteration_index: int,
    num_iterations: int,
    config: Any,
    rollout_eval_callback: Any,
    vllm_runtime: Optional[Any],
    use_lora_reference_disable_adapter: bool,
) -> Any:
    reward_config = {
        "reward_version": str(getattr(args, "rl_reward_version", DEFAULT_RL_REWARD_VERSION)),
        "open_ended_judge_enabled": bool(getattr(args, "rl_open_ended_judge_enabled", True)),
        "open_ended_judge_base_url": str(getattr(args, "rl_open_ended_judge_base_url", "") or ""),
        "open_ended_judge_model": str(getattr(args, "rl_open_ended_judge_model", "") or ""),
        "open_ended_judge_cache_path": str(getattr(args, "rl_open_ended_judge_cache_path", "") or ""),
        "open_ended_judge_timeout_sec": float(getattr(args, "rl_open_ended_judge_timeout_sec", 30.0)),
    }
    extra_reward_config = getattr(args, "rl_reward_config", None)
    if isinstance(extra_reward_config, dict):
        reward_config.update(extra_reward_config)
    return create_timesearch_aligned_grpo_trainer(
        model=model,
        processor=processor,
        train_items=train_items,
        output_dir=checkpoint_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        dataloader_persistent_workers=args.dataloader_persistent_workers,
        reference_model=reference_model,
        use_lora_reference_disable_adapter=use_lora_reference_disable_adapter,
        kl_beta=args.kl_beta,
        ppo_clip_epsilon=args.ppo_clip_epsilon,
        rollout_runner=SaverRolloutRunner(
            max_turns=args.rollout_max_turns,
            config=config,
        ),
        num_generations=args.num_generations,
        min_weight=args.min_weight,
        advantage_clip=args.advantage_clip,
        policy_max_new_tokens=args.policy_max_new_tokens,
        max_image_side=args.max_image_side,
        max_image_pixels=args.max_image_pixels,
        max_total_images=args.max_total_images,
        keep_recent_tool_image_messages=args.keep_recent_tool_image_messages,
        keep_recent_text_messages=args.keep_recent_text_messages,
        max_seq_length=args.max_seq_length,
        counterfactual_max_images=12,
        policy_do_sample=args.policy_do_sample,
        policy_temperature=args.policy_temperature,
        policy_top_p=args.policy_top_p,
        policy_top_k=args.policy_top_k,
        policy_repetition_penalty=args.policy_repetition_penalty,
        rollout_use_generation_cache=args.rl_rollout_use_cache,
        fecv_use_generation_cache=args.rl_fecv_use_cache,
        compute_loss_microbatch_size=args.rl_compute_loss_microbatch_size,
        steps_per_generation=max(1, int(getattr(args, "rl_steps_per_generation", 1))),
        replay_buffer_enable=args.rl_replay_buffer_enable,
        replay_buffer_type=args.rl_replay_buffer_type,
        replay_buffer_capacity=args.rl_replay_buffer_capacity,
        replay_buffer_alpha=args.rl_replay_buffer_alpha,
        fecv_failure_policy=args.rl_fecv_failure_policy,
        log_empty_batch_rank_summary=args.rl_log_empty_batch_rank_summary,
        reward_version=getattr(args, "rl_reward_version", DEFAULT_RL_REWARD_VERSION),
        reward_config=reward_config,
        iteration_index=int(iteration_index),
        num_iterations=int(num_iterations),
        rollout_eval_callback=rollout_eval_callback,
        deepspeed=str(getattr(args, "deepspeed", "") or "") or None,
        policy_builder=_build_runtime_policy_builder(
            args=args,
            vllm_runtime=vllm_runtime,
        ),
    )


class TrlVllmGrpoRunner:
    def __init__(
        self,
        *,
        args: Any,
        runtime: Any,
        log_dir: str | Path = "",
        config_builder: Any,
        eval_config_builder: Any,
        reference_model_resolver: Any,
        select_iteration_indices_fn: Any,
    ) -> None:
        self.args = args
        self.runtime = runtime or distributed_runtime_from_env()
        self.log_dir = log_dir
        self.config_builder = config_builder
        self.eval_config_builder = eval_config_builder
        self.reference_model_resolver = reference_model_resolver
        self.select_iteration_indices_fn = select_iteration_indices_fn

        self.output_dir = Path(self.args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_eval_output_root = (
            Path(str(self.args.rollout_eval_output_dir).strip())
            if str(getattr(self.args, "rollout_eval_output_dir", "") or "").strip()
            else self.output_dir
        )
        self.rollout_eval_output_root.mkdir(parents=True, exist_ok=True)
        self.resolved_log_dir = (
            Path(str(log_dir).strip()) if str(log_dir or "").strip() else self.output_dir / "logs"
        )

        self.raw_records = self._load_raw_records()
        self.dataset = SaverAgentDataset(
            self.args.data,
            data_root=self.args.data_root,
            config=self.config_builder(self.args),
            include_splits=getattr(self.args, "include_splits", ""),
            require_frame_cache=True,
            require_feature_cache=True,
        )
        self.current_model_path = str(self.args.model_path)
        self.latest_checkpoint = self.current_model_path
        self.reference_model_path = self.reference_model_resolver(
            self.args.model_path,
            self.args.reference_model_path,
        )
        self.use_lora_reference_disable_adapter = bool(self.args.lora) and float(self.args.kl_beta) > 0.0

        self.policy_model = None
        self.processor = None
        self.reference_model = None
        self.persistent_vllm_runtime: Optional[Any] = None

    def _load_raw_records(self) -> List[Dict[str, Any]]:
        records = [
            json.loads(line)
            for line in Path(self.args.data).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if getattr(self.args, "include_splits", ""):
            allowed = {
                str(value).strip()
                for value in str(self.args.include_splits).split(",")
                if str(value).strip()
            }
            records = [record for record in records if str(record.get("split") or "").strip() in allowed]
        return records

    def _ensure_models_loaded(self) -> Tuple[Any, Any, Any]:
        if self.policy_model is None or self.processor is None:
            self.policy_model, self.processor = load_qwen_model_and_processor(
                self.current_model_path,
                torch_dtype=self.args.torch_dtype,
                attn_implementation=self.args.attn_implementation or None,
                gradient_checkpointing=self.args.gradient_checkpointing,
                use_lora=self.args.lora,
                lora_r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                lora_target_modules=[
                    module.strip()
                    for module in str(self.args.lora_target_modules or "").split(",")
                    if module.strip()
                ]
                or None,
            )
        if (
            self.reference_model is None
            and not self.use_lora_reference_disable_adapter
            and float(self.args.kl_beta) > 0.0
            and str(self.reference_model_path or "").strip()
        ):
            self.reference_model, _ = load_qwen_model_and_processor(
                self.reference_model_path,
                torch_dtype=self.args.torch_dtype,
                attn_implementation=self.args.attn_implementation or None,
                gradient_checkpointing=False,
                use_lora=False,
            )
        return self.policy_model, self.processor, self.reference_model

    def _build_iteration_summary(
        self,
        *,
        iteration: int,
        items: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "iteration": int(iteration),
            "num_groups": len(items),
            "num_generations": int(self.args.num_generations),
            "current_model_path": str(self.current_model_path),
            "reference_model_path": str(self.reference_model_path),
            "rollout_eval_output_dir": str(self.rollout_eval_output_root / f"iter_{int(iteration):03d}"),
            "rl_backend": "trl_vllm_grpo",
            "rl_reward_version": str(getattr(self.args, "rl_reward_version", DEFAULT_RL_REWARD_VERSION)),
            "use_vllm": bool(getattr(self.args, "use_vllm", True)),
            "vllm_mode": str(getattr(self.args, "vllm_mode", "colocate")),
            "vllm_tensor_parallel_size": int(getattr(self.args, "vllm_tensor_parallel_size", 1)),
            "vllm_gpu_memory_utilization": float(getattr(self.args, "vllm_gpu_memory_utilization", 0.35)),
            "rl_rollout_use_cache": bool(self.args.rl_rollout_use_cache),
            "rl_fecv_use_cache": bool(self.args.rl_fecv_use_cache),
            "rl_compute_loss_microbatch_size": int(self.args.rl_compute_loss_microbatch_size),
            "rl_steps_per_generation": int(getattr(self.args, "rl_steps_per_generation", 1)),
            "rl_replay_buffer_enable": bool(self.args.rl_replay_buffer_enable),
            "rl_replay_buffer_type": str(self.args.rl_replay_buffer_type),
            "rl_replay_buffer_capacity": int(self.args.rl_replay_buffer_capacity),
            "rl_replay_buffer_alpha": float(self.args.rl_replay_buffer_alpha),
            "rl_fecv_failure_policy": str(self.args.rl_fecv_failure_policy),
            "rl_log_empty_batch_rank_summary": bool(self.args.rl_log_empty_batch_rank_summary),
            "rl_open_ended_judge_enabled": bool(getattr(self.args, "rl_open_ended_judge_enabled", True)),
            "rl_open_ended_judge_base_url": str(getattr(self.args, "rl_open_ended_judge_base_url", "") or ""),
            "rl_open_ended_judge_model": str(getattr(self.args, "rl_open_ended_judge_model", "") or ""),
            "rl_open_ended_judge_cache_path": str(getattr(self.args, "rl_open_ended_judge_cache_path", "") or ""),
        }

    def _run_iteration(self, *, iteration: int) -> None:
        iter_dir = self.output_dir / f"iter_{int(iteration):03d}"
        checkpoint_dir = iter_dir / "checkpoint"
        iter_dir.mkdir(parents=True, exist_ok=True)
        indices = self.select_iteration_indices_fn(
            len(self.raw_records),
            self.args.rollout_count,
            self.args.rollout_start_index,
            iteration,
        )
        items = [self.dataset[int(index)] for index in indices]
        summary = self._build_iteration_summary(iteration=iteration, items=items)
        if self.args.dry_run or not items:
            summary["latest_checkpoint"] = str(self.current_model_path)
            write_json(iter_dir / "summary.json", summary)
            if self.runtime.is_main_process:
                append_jsonl(self.resolved_log_dir / "rl_iteration_metrics.jsonl", summary)
            return

        runtime_log(
            (
                f"iteration {int(iteration)}: starting trl-vllm GRPO update "
                f"with groups={len(items)} num_generations={int(self.args.num_generations)} "
                f"steps_per_generation={int(getattr(self.args, 'rl_steps_per_generation', 1))}"
            ),
            runtime=self.runtime,
            main_process_only=True,
        )
        model, processor, reference_model = self._ensure_models_loaded()
        iteration_config = self.config_builder(self.args)
        rollout_eval_config = self.eval_config_builder(
            args=self.args,
            current_model_path=self.current_model_path,
            reference_model_path=self.reference_model_path,
            config=iteration_config,
        )
        rollout_eval_callback = _build_rl_authority_checkpoint_callback(
            processor=processor,
            rollout_eval_config=rollout_eval_config,
            rollout_eval_output_dir=self.rollout_eval_output_root / f"iter_{int(iteration):03d}",
            iteration_index=int(iteration),
            policy_factory=(
                _build_inline_vllm_policy_factory(args=self.args, vllm_runtime=self.persistent_vllm_runtime)
                if self.persistent_vllm_runtime is not None
                else None
            ),
        )
        trainer = create_trl_vllm_grpo_trainer(
            args=self.args,
            model=model,
            processor=processor,
            reference_model=reference_model,
            train_items=items,
            checkpoint_dir=checkpoint_dir,
            iteration_index=int(iteration),
            num_iterations=int(self.args.num_iterations),
            config=iteration_config,
            rollout_eval_callback=rollout_eval_callback,
            vllm_runtime=self.persistent_vllm_runtime,
            use_lora_reference_disable_adapter=self.use_lora_reference_disable_adapter,
        )
        try:
            train_result = trainer.train()
            budget_stats = trainer.get_budgeting_stats() if hasattr(trainer, "get_budgeting_stats") else None
            budget_drop_metrics = (
                trainer.get_budget_drop_metrics() if hasattr(trainer, "get_budget_drop_metrics") else {}
            )
            if budget_stats is not None:
                runtime_log(
                    f"RL iteration {int(iteration)} budgeting: {budget_stats.as_dict()}",
                    runtime=self.runtime,
                    main_process_only=True,
                )
            authority_checkpoint = getattr(rollout_eval_callback, "last_authority_checkpoint_path", None)
            authority_epoch_index = getattr(rollout_eval_callback, "last_authority_epoch_index", None)
            checkpoint_strategy = str(getattr(rollout_eval_callback, "checkpoint_strategy", "epoch_resume_only"))
            if authority_checkpoint is None:
                raise RuntimeError(f"trl-vllm RL iteration {int(iteration)} did not publish an authority checkpoint.")
            authority_checkpoint = Path(authority_checkpoint)
            _write_rl_checkpoint_authority_metadata(
                checkpoint_root=checkpoint_dir,
                authority_checkpoint=authority_checkpoint,
                iteration_index=int(iteration),
                epoch_index=int(authority_epoch_index or 1),
                runtime=self.runtime,
            )
            runtime_log(
                (
                    f"RL duplicate root save skipped: iter={int(iteration)} "
                    f"checkpoint_root={checkpoint_dir} authority_checkpoint={authority_checkpoint}"
                ),
                runtime=self.runtime,
                main_process_only=True,
            )
            self.latest_checkpoint = str(authority_checkpoint)
            self.current_model_path = self.latest_checkpoint
            summary.update(
                {
                    "latest_checkpoint": str(self.latest_checkpoint),
                    "checkpoint_root": str(checkpoint_dir),
                    "checkpoint_strategy": checkpoint_strategy,
                    "authority_epoch_index": int(authority_epoch_index or 1),
                    "train_loss": float(getattr(train_result, "training_loss", 0.0)),
                }
            )
            summary.update({key: value for key, value in budget_drop_metrics.items()})
            write_json(iter_dir / "summary.json", summary)
            if self.runtime.is_main_process:
                append_jsonl(self.resolved_log_dir / "rl_iteration_metrics.jsonl", summary)
            runtime_log(
                (
                    f"RL authority checkpoint published: iter={int(iteration)} "
                    f"latest_checkpoint={self.latest_checkpoint} checkpoint_root={checkpoint_dir} "
                    f"strategy={checkpoint_strategy}"
                ),
                runtime=self.runtime,
                main_process_only=True,
            )
        finally:
            if self.persistent_vllm_runtime is not None:
                setattr(trainer, "_vllm_runtime", None)
            _teardown_trainer_iteration_runtime(trainer)
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run(self) -> Dict[str, Any]:
        runtime_log(
            (
                "trl-vllm RL startup: "
                f"num_iterations={int(self.args.num_iterations)} rollout_count={int(self.args.rollout_count)} "
                f"num_generations={int(self.args.num_generations)} model_path={self.current_model_path} "
                f"steps_per_generation={int(getattr(self.args, 'rl_steps_per_generation', 1))} "
                f"reward_version={str(getattr(self.args, 'rl_reward_version', DEFAULT_RL_REWARD_VERSION))} "
                f"use_vllm={bool(getattr(self.args, 'use_vllm', True))} "
                f"vllm_mode={str(getattr(self.args, 'vllm_mode', 'colocate'))} "
                f"rollout_cache={bool(self.args.rl_rollout_use_cache)} fecv_cache={bool(self.args.rl_fecv_use_cache)} "
                f"loss_microbatch={int(self.args.rl_compute_loss_microbatch_size)} "
                f"rollout_eval_mode={'inline' if bool(getattr(self.args, 'inline_rollout_eval', False)) else 'deferred'} "
                f"rollout_eval_output_root={self.rollout_eval_output_root}"
            ),
            runtime=self.runtime,
            main_process_only=True,
        )
        if bool(getattr(self.args, "use_vllm", True)):
            self.persistent_vllm_runtime = create_vllm_runtime(
                args=self.args,
                runtime=self.runtime,
                model_path=self.current_model_path,
            )
            runtime_log(
                (
                    "trl-vllm RL attached persistent vLLM runtime: "
                    f"base_model_path={self.persistent_vllm_runtime.base_model_path} "
                    f"mode={str(getattr(self.args, 'vllm_mode', 'colocate'))} "
                    f"gpu_memory_utilization={float(getattr(self.args, 'vllm_gpu_memory_utilization', 0.35) or 0.35):.3f} "
                    f"guided_decoding={'on' if bool(getattr(self.args, 'vllm_guided_decoding_regex', '')) else 'off'}"
                ),
                runtime=self.runtime,
                main_process_only=True,
            )

        try:
            for iteration in range(int(self.args.num_iterations)):
                self._run_iteration(iteration=int(iteration))
        finally:
            if self.persistent_vllm_runtime is not None:
                self.persistent_vllm_runtime.close()

        final_summary = {
            "timestamp_utc": utc_timestamp(),
            "output_dir": str(self.output_dir),
            "rollout_eval_output_root": str(self.rollout_eval_output_root),
            "latest_checkpoint": str(self.latest_checkpoint),
            "num_iterations": int(self.args.num_iterations),
            "reference_model_path": str(self.reference_model_path),
            "rl_backend": "trl_vllm_grpo",
            "rl_reward_version": str(getattr(self.args, "rl_reward_version", DEFAULT_RL_REWARD_VERSION)),
            "checkpoint_strategy": "epoch_resume_only",
            "use_vllm": bool(getattr(self.args, "use_vllm", True)),
            "vllm_mode": str(getattr(self.args, "vllm_mode", "colocate")),
            "vllm_tensor_parallel_size": int(getattr(self.args, "vllm_tensor_parallel_size", 1)),
            "vllm_gpu_memory_utilization": float(getattr(self.args, "vllm_gpu_memory_utilization", 0.35)),
            "rl_rollout_use_cache": bool(self.args.rl_rollout_use_cache),
            "rl_fecv_use_cache": bool(self.args.rl_fecv_use_cache),
            "rl_compute_loss_microbatch_size": int(self.args.rl_compute_loss_microbatch_size),
            "rl_steps_per_generation": int(getattr(self.args, "rl_steps_per_generation", 1)),
            "rl_replay_buffer_enable": bool(self.args.rl_replay_buffer_enable),
            "rl_replay_buffer_type": str(self.args.rl_replay_buffer_type),
            "rl_replay_buffer_capacity": int(self.args.rl_replay_buffer_capacity),
            "rl_replay_buffer_alpha": float(self.args.rl_replay_buffer_alpha),
            "rl_fecv_failure_policy": str(self.args.rl_fecv_failure_policy),
            "rl_log_empty_batch_rank_summary": bool(self.args.rl_log_empty_batch_rank_summary),
        }
        if self.runtime.is_main_process:
            (self.output_dir / "latest_checkpoint.txt").write_text(str(self.latest_checkpoint), encoding="utf-8")
            write_json(self.resolved_log_dir / "train_saver_rl_final_summary.json", final_summary)
        return final_summary


def _run_vllm_grpo_iterations(
    *,
    args: Any,
    runtime: Any,
    log_dir: str | Path = "",
    config_builder: Any,
    eval_config_builder: Any,
    reference_model_resolver: Any,
    select_iteration_indices_fn: Any,
) -> Dict[str, Any]:
    return TrlVllmGrpoRunner(
        args=args,
        runtime=runtime,
        log_dir=log_dir,
        config_builder=config_builder,
        eval_config_builder=eval_config_builder,
        reference_model_resolver=reference_model_resolver,
        select_iteration_indices_fn=select_iteration_indices_fn,
    ).run()


def run_trainer_vllm_grpo(
    *,
    args: Any,
    runtime: Any,
    log_dir: str | Path = "",
    config_builder: Any,
    eval_config_builder: Any,
    reference_model_resolver: Any,
    select_iteration_indices_fn: Any,
) -> Dict[str, Any]:
    return _run_vllm_grpo_iterations(
        args=args,
        runtime=runtime,
        log_dir=log_dir,
        config_builder=config_builder,
        eval_config_builder=eval_config_builder,
        reference_model_resolver=reference_model_resolver,
        select_iteration_indices_fn=select_iteration_indices_fn,
    )
