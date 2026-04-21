from __future__ import annotations

import copy
import gc
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from saver_v3.data.dataset import SaverAgentDataset
from saver_v3.data.materialized_cache import (
    MATERIALIZED_RUNTIME_ITEMS_FORMAT,
    MaterializedRuntimeItemDataset,
    ensure_materialized_cache_metadata,
)
from saver_v3.common.experiment_logging import append_jsonl, utc_timestamp, write_json
from saver_v3.rl.grpo_trainer_env import (
    _build_rl_authority_checkpoint_callback,
    create_native_grpo_trainer,
    _load_training_proposal_runtime,
    _raw_records_require_feature_guided_proposal,
    _teardown_trainer_iteration_runtime,
    _write_rl_checkpoint_authority_metadata,
)
from saver_v3.model.qwen_policy import QwenGenerationPolicy, load_generation_processor_for_checkpoint
from saver_v3.core.reward import DEFAULT_RL_REWARD_VERSION
from saver_v3.core.rollout import SaverRolloutRunner
from saver_v3.common.runtime import distributed_runtime_from_env, runtime_log
from saver_v3.sft.training import _unwrap_model, load_qwen_model_and_processor, run_rollout_eval_with_policy
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


def _save_loadable_hf_authority_checkpoint(
    *,
    trainer: Any,
    processor: Any,
    checkpoint_root: str | Path,
    epoch_index: int,
    runtime: Any,
) -> Path:
    checkpoint_root = Path(checkpoint_root)
    loadable_dir = checkpoint_root / "authority_hf" / f"epoch_{int(epoch_index):03d}"
    loadable_dir.mkdir(parents=True, exist_ok=True)
    accelerator = getattr(trainer, "accelerator", None)
    wrapped_model = getattr(trainer, "model_wrapped", None) or getattr(trainer, "model", None)
    if accelerator is None or wrapped_model is None:
        raise RuntimeError(
            "Saving a loadable RL authority checkpoint requires trainer.accelerator and a wrapped training model."
        )
    unwrap_model = getattr(accelerator, "unwrap_model", None)
    model_to_save = unwrap_model(wrapped_model) if callable(unwrap_model) else _unwrap_model(wrapped_model)
    if runtime.is_main_process:
        get_state_dict = getattr(accelerator, "get_state_dict", None)
        if not callable(get_state_dict):
            raise RuntimeError("Accelerator.get_state_dict is required to save a loadable RL authority checkpoint.")
        state_dict = get_state_dict(wrapped_model, unwrap=False)
        model_to_save.save_pretrained(str(loadable_dir), state_dict=state_dict)
        if hasattr(processor, "save_pretrained"):
            processor.save_pretrained(str(loadable_dir))
        write_json(
            loadable_dir / "authority_metadata.json",
            {
                "epoch_index": int(epoch_index),
                "checkpoint_kind": "rl_loadable_authority_hf",
                "checkpoint_root": str(checkpoint_root),
            },
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    return loadable_dir


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
        preferred_max_num_seqs = max(1, int(getattr(args, "vllm_max_num_seqs", 4) or 4))
        fallback_max_num_seqs = max(1, int(getattr(args, "vllm_fallback_max_num_seqs", 2) or 2))
        last_exc: Optional[BaseException] = None
        attempted: list[int] = []
        for candidate in [preferred_max_num_seqs, fallback_max_num_seqs]:
            if candidate in attempted:
                continue
            attempted.append(candidate)
            candidate_args = _rl_with_vllm_seq_overrides(args, max_num_seqs=candidate)
            try:
                runtime_log(
                    f"rl-only vLLM runtime init attempt: local_rank={int(getattr(runtime, 'local_rank', 0) or 0)} max_num_seqs={candidate}",
                    runtime=runtime,
                    main_process_only=False,
                )
                return shared_vllm_generation.create_vllm_runtime(
                    args=candidate_args,
                    runtime=runtime,
                    model_path=model_path,
                    prefer_direct_local_rank_runtime=True,
                )
            except Exception as exc:
                last_exc = exc
                if candidate == preferred_max_num_seqs and candidate != fallback_max_num_seqs and _is_vllm_init_oom_error(exc):
                    runtime_log(
                        f"rl-only vLLM runtime falling back from max_num_seqs={preferred_max_num_seqs} to {fallback_max_num_seqs} after init OOM: {exc}",
                        runtime=runtime,
                        main_process_only=True,
                    )
                    continue
                raise
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Unable to initialize RL vLLM runtime")
    raise ValueError(f"Unsupported vLLM mode: {settings['vllm_mode']}")


class VllmQwenGenerationPolicy(shared_vllm_generation.VllmQwenGenerationPolicy):
    pass


class MutableIterationDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[Dict[str, Any]]):
        self.items = list(items or [])

    def replace_items(self, items: List[Dict[str, Any]]) -> None:
        self.items = list(items or [])

    def snapshot_items(self) -> List[Dict[str, Any]]:
        return [copy.deepcopy(item) for item in self.items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return copy.deepcopy(self.items[int(index)])


def _is_vllm_init_oom_error(exc: BaseException) -> bool:
    text = str(exc or "").lower()
    return any(
        marker in text
        for marker in (
            "cuda out of memory",
            "outofmemoryerror",
            "not enough gpu memory",
            "no available memory for the cache blocks",
            "failed to load model - not enough gpu memory",
        )
    )


def _rl_with_vllm_seq_overrides(args: Any, *, max_num_seqs: int) -> Any:
    payload = dict(vars(args)) if hasattr(args, "__dict__") else {}
    payload["vllm_max_num_seqs"] = int(max_num_seqs)
    return type("RLVllmArgs", (), payload)()


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
                # Let reward judge reuse the same-rank training vLLM engine so
                # accuracy_reward semantic sub-rewards can batch through the
                # already-loaded Qwen3-VL-8B without extra VRAM or external API.
                judge = getattr(self, "reward_judge", None)
                if judge is not None and vllm_runtime is not None:
                    engine = getattr(vllm_runtime, "llm", None)
                    if engine is not None and hasattr(judge, "attach_local_vllm"):
                        judge.attach_local_vllm(engine)

            def _build_policy(self, model: Any, *, use_generation_cache: bool) -> QwenGenerationPolicy:
                policy = VllmQwenGenerationPolicy(
                    runtime=vllm_runtime,
                    source_model=_unwrap_model(model),
                    step_resolver=lambda: int(getattr(getattr(self, "state", None), "global_step", 0) or 0),
                    guided_decoding_regex=str(getattr(args, "vllm_guided_decoding_regex", "") or ""),
                    processor=self.processor,
                    max_new_tokens=self.policy_max_new_tokens,
                    max_total_images=self.max_total_images,
                    max_tool_message_frames=self.max_tool_message_frames,
                    max_total_video_frames=self.max_total_video_frames,
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
                return policy

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
        max_tool_message_frames: int,
        max_total_video_frames: int,
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
            policy = VllmQwenGenerationPolicy(
                runtime=vllm_runtime,
                source_model=_unwrap_model(model),
                step_resolver=step_resolver or (lambda: 0),
                guided_decoding_regex=str(getattr(args, "vllm_guided_decoding_regex", "") or ""),
                processor=processor,
                max_new_tokens=int(max_new_tokens),
                max_total_images=int(max_total_images),
                max_tool_message_frames=int(max_tool_message_frames),
                max_total_video_frames=int(max_total_video_frames),
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
            return policy
        return QwenGenerationPolicy.from_components(
            model=_unwrap_model(model),
            processor=processor,
            max_new_tokens=int(max_new_tokens),
            max_total_images=int(max_total_images),
            max_tool_message_frames=int(max_tool_message_frames),
            max_total_video_frames=int(max_total_video_frames),
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


def _iteration_dir_name(iteration_index: int) -> str:
    return f"iter_{int(iteration_index):03d}"


def _resolve_standard_trainer_checkpoint_path(output_dir: str | Path, global_step: int) -> Path:
    return Path(output_dir) / f"checkpoint-{int(global_step)}"


def _sample_iteration_items(
    *,
    base_dataset: Any,
    raw_records: Optional[List[Dict[str, Any]]],
    raw_record_count: int,
    select_iteration_indices_fn: Any,
    rollout_count: int,
    rollout_start_index: int,
    iteration_index: int,
    seed: int,
) -> List[Dict[str, Any]]:
    try:
        indices = select_iteration_indices_fn(
            raw_record_count,
            rollout_count,
            rollout_start_index,
            iteration_index,
            seed=seed,
            records=raw_records,
        )
    except TypeError:
        indices = select_iteration_indices_fn(
            raw_record_count,
            rollout_count,
            rollout_start_index,
            iteration_index,
        )
    return [base_dataset[int(index)] for index in indices]


def _continuous_rl_args(args: Any) -> Any:
    payload = dict(vars(args)) if hasattr(args, "__dict__") else {}
    payload["num_train_epochs"] = float(int(getattr(args, "num_iterations", 1) or 1))
    return type("ContinuousRLArgs", (), payload)()


def _build_continuous_iteration_callback(
    *,
    owner: Any,
    mutable_dataset: MutableIterationDataset,
    trainer: Any,
    processor: Any,
    eval_start_iteration: int,
    eval_every_iterations: int,
    final_rollout_eval: bool = False,
) -> Any:
    try:
        from transformers import TrainerCallback
        from transformers.trainer_utils import get_last_checkpoint
    except Exception as exc:
        raise ImportError("Continuous RL iteration callback requires transformers.") from exc

    class ContinuousIterationCallback(TrainerCallback):
        def __init__(self) -> None:
            self.owner = owner
            self.mutable_dataset = mutable_dataset
            self.trainer = trainer
            self.processor = processor
            self.eval_start_iteration = max(1, int(eval_start_iteration))
            self.eval_every_iterations = max(1, int(eval_every_iterations))
            self.final_rollout_eval = bool(final_rollout_eval)
            self.last_saved_checkpoint: Optional[Path] = None
            self._final_eval_completed = False

        def _should_trigger_checkpoint(self, iteration_index: int) -> bool:
            return should_run_inline_rollout_eval(
                iteration_index,
                eval_start_iteration=self.eval_start_iteration,
                eval_every_iterations=self.eval_every_iterations,
            )

        @staticmethod
        def _iteration_from_epoch_begin(state: Any) -> int:
            epoch_value = float(getattr(state, "epoch", 0.0) or 0.0)
            return max(0, int(epoch_value))

        @staticmethod
        def _iteration_from_epoch_end(state: Any) -> int:
            epoch_value = float(getattr(state, "epoch", 0.0) or 0.0)
            return max(0, int(round(epoch_value)) - 1)

        def _refresh_iteration_items(self, iteration_index: int) -> List[Dict[str, Any]]:
            items = _sample_iteration_items(
                base_dataset=self.owner.dataset,
                raw_records=self.owner.raw_records,
                raw_record_count=len(self.owner.raw_records),
                select_iteration_indices_fn=self.owner.select_iteration_indices_fn,
                rollout_count=int(self.owner.args.rollout_count),
                rollout_start_index=int(self.owner.args.rollout_start_index),
                iteration_index=int(iteration_index),
                seed=int(getattr(self.owner.args, "seed", 42) or 42),
            )
            self.mutable_dataset.replace_items(items)
            self.trainer.train_dataset = self.mutable_dataset
            self.trainer.reference_model_source_path = str(self.owner.current_model_path)
            native_progress = getattr(self.trainer, "_native_grpo_progress", None)
            if native_progress is not None:
                native_progress.iteration_index = int(iteration_index)
                native_progress.num_iterations = int(self.owner.args.num_iterations)
                native_progress.set_total_groups(len(items))
            if hasattr(self.trainer, "_buffered_generation_step_payloads"):
                self.trainer._buffered_generation_step_payloads = []
            if hasattr(self.trainer, "_buffered_generation_batch_key"):
                self.trainer._buffered_generation_batch_key = None
            return items

        def on_train_begin(self, args, state, control, **kwargs):
            del kwargs
            iteration_index = self._iteration_from_epoch_begin(state)
            self._refresh_iteration_items(iteration_index)
            return control

        def on_epoch_begin(self, args, state, control, **kwargs):
            del args, kwargs
            iteration_index = self._iteration_from_epoch_begin(state)
            items = self._refresh_iteration_items(iteration_index)
            runtime_log(
                (
                    f"continuous RL epoch begin: iteration={int(iteration_index)} "
                    f"groups={len(items)} num_generations={int(self.owner.args.num_generations)}"
                ),
                runtime=self.owner.runtime,
                main_process_only=True,
            )
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            del args, kwargs
            iteration_index = self._iteration_from_epoch_end(state)
            should_save = self._should_trigger_checkpoint(int(iteration_index))
            control.should_save = bool(should_save)
            return control

        def on_save(self, args, state, control, model=None, **kwargs):
            del control, kwargs
            iteration_index = self._iteration_from_epoch_end(state)
            if not self._should_trigger_checkpoint(int(iteration_index)):
                return
            checkpoint = get_last_checkpoint(args.output_dir)
            if not checkpoint:
                raise RuntimeError("Continuous RL expected a standard Trainer checkpoint after save, but none was found.")
            checkpoint_path = Path(checkpoint)
            self.last_saved_checkpoint = checkpoint_path
            self.owner.latest_checkpoint = str(checkpoint_path)
            self.owner.current_model_path = str(checkpoint_path)
            runtime_log(
                (
                    f"continuous RL standard checkpoint published: iteration={int(iteration_index)} "
                    f"checkpoint={checkpoint_path}"
                ),
                runtime=self.owner.runtime,
                main_process_only=True,
            )
            summary = self.owner._build_iteration_summary(
                iteration=int(iteration_index),
                items=self.mutable_dataset.snapshot_items(),
            )
            summary.update(
                {
                    "latest_checkpoint": str(checkpoint_path),
                    "checkpoint_root": str(checkpoint_path),
                    "checkpoint_strategy": _checkpoint_strategy_label(
                        eval_start_iteration=self.eval_start_iteration,
                        eval_every_iterations=self.eval_every_iterations,
                    ),
                    "reference_model_source_path": str(getattr(self.trainer, "reference_model_source_path", self.owner.reference_model_source_path)),
                    "reference_model_backend": str(getattr(self.trainer, "reference_model_backend", self.owner.reference_model_backend)),
                    "use_liger_loss_requested": bool(getattr(self.trainer, "use_liger_loss_requested", self.owner.use_liger_loss_requested)),
                    "use_liger_loss_effective": bool(getattr(self.trainer, "use_liger_loss_effective", self.owner.use_liger_loss_effective or False)),
                    "liger_disable_reason": str(getattr(self.trainer, "_liger_runtime_disable_reason", "") or ""),
                    "liger_linear_head_path": str(getattr(self.trainer, "_liger_linear_head_path", "") or ""),
                    "liger_hidden_state_path": str(getattr(self.trainer, "_liger_hidden_state_path", "") or ""),
                }
            )
            budget_stats = self.trainer.get_budgeting_stats() if hasattr(self.trainer, "get_budgeting_stats") else None
            budget_drop_metrics = self.trainer.get_budget_drop_metrics() if hasattr(self.trainer, "get_budget_drop_metrics") else {}
            if budget_stats is not None:
                runtime_log(
                    f"RL iteration {int(iteration_index)} budgeting: {budget_stats.as_dict()}",
                    runtime=self.owner.runtime,
                    main_process_only=True,
                )
            summary.update({key: value for key, value in budget_drop_metrics.items()})
            iter_dir = self.owner.output_dir / _iteration_dir_name(int(iteration_index))
            iter_dir.mkdir(parents=True, exist_ok=True)
            write_json(iter_dir / "summary.json", summary)
            if self.owner.runtime.is_main_process:
                append_jsonl(self.owner.resolved_log_dir / "rl_iteration_metrics.jsonl", summary)
            rollout_eval_config = self.owner.eval_config_builder(
                args=self.owner.args,
                current_model_path=str(checkpoint_path),
                reference_model_path=str(checkpoint_path),
                config=self.owner.config_builder(self.owner.args),
            )
            if bool(getattr(rollout_eval_config, "inline_rollout_eval", False)):
                eval_model = _unwrap_model(model)
                if callable(self.owner.inline_policy_factory):
                    policy = self.owner.inline_policy_factory(
                        eval_model=eval_model,
                        processor=self.processor,
                        rollout_eval_config=rollout_eval_config,
                        state=state,
                        runtime=self.owner.runtime,
                    )
                else:
                    policy = QwenGenerationPolicy.from_components(
                        model=eval_model,
                        processor=self.processor,
                        max_new_tokens=int(rollout_eval_config.policy_max_new_tokens),
                        max_total_images=int(rollout_eval_config.max_total_images),
                        max_tool_message_frames=int(getattr(rollout_eval_config, "max_tool_message_frames", 0)),
                        max_total_video_frames=int(getattr(rollout_eval_config, "max_total_video_frames", 0)),
                        max_seq_length=int(rollout_eval_config.max_seq_length),
                        keep_recent_tool_image_messages=int(getattr(rollout_eval_config, "keep_recent_tool_image_messages", 0)),
                        keep_recent_text_messages=int(rollout_eval_config.keep_recent_text_messages),
                        max_image_side=int(rollout_eval_config.max_image_side),
                        max_image_pixels=int(rollout_eval_config.max_image_pixels),
                        do_sample=False,
                        use_generation_cache=bool(rollout_eval_config.use_generation_cache),
                    )
                run_rollout_eval_with_policy(
                    policy,
                    rollout_eval_config=rollout_eval_config,
                    output_dir=args.output_dir,
                    rollout_eval_output_dir=str(self.owner.rollout_eval_output_root / _iteration_dir_name(int(iteration_index))),
                    epoch_index=int(iteration_index) + 1,
                    epoch_value=float(getattr(state, "epoch", float(iteration_index + 1)) or float(iteration_index + 1)),
                    runtime=self.owner.runtime,
                )
            else:
                runtime_log(
                    f"continuous RL rollout eval deferred: checkpoint={checkpoint_path}",
                    runtime=self.owner.runtime,
                    main_process_only=True,
                )

        def on_train_end(self, args, state, control, model=None, **kwargs):
            del kwargs
            if not self.final_rollout_eval:
                return control
            if self._final_eval_completed:
                return control
            runtime_log(
                "continuous RL final eval begin: saving terminal checkpoint before rollout evaluation",
                runtime=self.owner.runtime,
                main_process_only=True,
            )
            try:
                self.trainer.save_model(str(args.output_dir))
            except Exception as exc:
                runtime_log(
                    f"final save_model failed, falling back to last saved checkpoint: {exc}",
                    runtime=self.owner.runtime,
                    main_process_only=True,
                )
            checkpoint = get_last_checkpoint(args.output_dir) or str(args.output_dir)
            checkpoint_path = Path(checkpoint)
            self.last_saved_checkpoint = checkpoint_path
            self.owner.latest_checkpoint = str(checkpoint_path)
            self.owner.current_model_path = str(checkpoint_path)
            iteration_index = max(0, int(self._iteration_from_epoch_end(state)))
            if iteration_index < int(getattr(self.owner.args, "num_iterations", 1) or 1) - 1:
                iteration_index = int(getattr(self.owner.args, "num_iterations", 1) or 1) - 1
            runtime_log(
                f"continuous RL final checkpoint saved: checkpoint={checkpoint_path} iteration={iteration_index}",
                runtime=self.owner.runtime,
                main_process_only=True,
            )
            rollout_eval_config = self.owner.eval_config_builder(
                args=self.owner.args,
                current_model_path=str(checkpoint_path),
                reference_model_path=str(checkpoint_path),
                config=self.owner.config_builder(self.owner.args),
            )
            eval_model = _unwrap_model(model)
            if callable(self.owner.inline_policy_factory):
                policy = self.owner.inline_policy_factory(
                    eval_model=eval_model,
                    processor=self.processor,
                    rollout_eval_config=rollout_eval_config,
                    state=state,
                    runtime=self.owner.runtime,
                )
            else:
                policy = QwenGenerationPolicy.from_components(
                    model=eval_model,
                    processor=self.processor,
                    max_new_tokens=int(rollout_eval_config.policy_max_new_tokens),
                    max_total_images=int(rollout_eval_config.max_total_images),
                    max_tool_message_frames=int(getattr(rollout_eval_config, "max_tool_message_frames", 0)),
                    max_total_video_frames=int(getattr(rollout_eval_config, "max_total_video_frames", 0)),
                    max_seq_length=int(rollout_eval_config.max_seq_length),
                    keep_recent_tool_image_messages=int(getattr(rollout_eval_config, "keep_recent_tool_image_messages", 0)),
                    keep_recent_text_messages=int(rollout_eval_config.keep_recent_text_messages),
                    max_image_side=int(rollout_eval_config.max_image_side),
                    max_image_pixels=int(rollout_eval_config.max_image_pixels),
                    do_sample=False,
                    use_generation_cache=bool(rollout_eval_config.use_generation_cache),
                )
            run_rollout_eval_with_policy(
                policy,
                rollout_eval_config=rollout_eval_config,
                output_dir=args.output_dir,
                rollout_eval_output_dir=str(self.owner.rollout_eval_output_root / "final"),
                epoch_index=int(iteration_index) + 1,
                epoch_value=float(getattr(state, "epoch", float(iteration_index + 1)) or float(iteration_index + 1)),
                runtime=self.owner.runtime,
            )
            self._final_eval_completed = True
            runtime_log(
                f"continuous RL final eval complete: checkpoint={checkpoint_path}",
                runtime=self.owner.runtime,
                main_process_only=True,
            )
            return control

    return ContinuousIterationCallback()


def should_run_inline_rollout_eval(
    iteration_index: int,
    *,
    eval_start_iteration: int,
    eval_every_iterations: int,
) -> bool:
    iteration_number = int(iteration_index) + 1
    start_iteration = max(1, int(eval_start_iteration))
    interval = max(1, int(eval_every_iterations))
    if iteration_number < start_iteration:
        return False
    return ((iteration_number - start_iteration) % interval) == 0


def _checkpoint_strategy_label(*, eval_start_iteration: int, eval_every_iterations: int) -> str:
    return (
        "standard_trainer_checkpoint_inline_eval"
        f"_start_{max(1, int(eval_start_iteration))}"
        f"_every_{max(1, int(eval_every_iterations))}_iterations"
    )


def create_trl_vllm_grpo_trainer(
    *,
    args: Any,
    model: Any,
    processor: Any,
    trainer_init_model_path: str | Path,
    train_items: List[Dict[str, Any]],
    train_dataset: Any = None,
    checkpoint_dir: str | Path,
    iteration_index: int,
    num_iterations: int,
    config: Any,
    rollout_eval_callback: Any,
    vllm_runtime: Optional[Any],
    proposal_runtime: Any = None,
    strict_feature_guided_proposal: bool = False,
    save_strategy: str = "no",
) -> Any:
    reward_config = {
        "reward_version": str(getattr(args, "rl_reward_version", DEFAULT_RL_REWARD_VERSION)),
    }
    extra_reward_config = getattr(args, "rl_reward_config", None)
    if isinstance(extra_reward_config, dict):
        reward_config.update(extra_reward_config)
    reference_model = None
    if float(getattr(args, "kl_beta", 0.0) or 0.0) > 0.0:
        reference_model, _ = load_qwen_model_and_processor(
            str(trainer_init_model_path),
            torch_dtype=str(args.torch_dtype or "auto"),
            attn_implementation=args.attn_implementation or None,
            gradient_checkpointing=False,
            use_lora=False,
        )
    return create_native_grpo_trainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
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
        old_policy_model=None,
        reference_model=reference_model,
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
        max_tool_message_frames=int(getattr(args, "max_tool_message_frames", 0) or 0),
        max_total_video_frames=int(getattr(args, "max_total_video_frames", 0) or 0),
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
        replay_buffer_enable=False,
        replay_buffer_type="none",
        replay_buffer_capacity=0,
        replay_buffer_alpha=1.0,
        fecv_failure_policy=args.rl_fecv_failure_policy,
        all_empty_policy="true_skip",
        log_empty_batch_rank_summary=args.rl_log_empty_batch_rank_summary,
        reward_version=getattr(args, "rl_reward_version", DEFAULT_RL_REWARD_VERSION),
        reward_config=reward_config,
        iteration_index=int(iteration_index),
        num_iterations=int(num_iterations),
        rollout_eval_callback=rollout_eval_callback,
        deepspeed=str(getattr(args, "deepspeed", "") or "") or None,
        save_strategy=str(save_strategy),
        trainer_class_transform=_build_vllm_trainer_class_transform(
            args=args,
            vllm_runtime=vllm_runtime,
        ) if vllm_runtime is not None else None,
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
        (
            self.strict_feature_guided_proposal,
            self.proposal_runtime,
        ) = self._resolve_training_proposal_support()
        self.dataset = self._build_dataset()
        self.current_model_path = str(self.args.model_path)
        self.latest_checkpoint = self.current_model_path
        self.reference_model_mode = "per_iteration_trainer_init"
        self.reference_model_source_path = str(self.current_model_path)
        self.reference_model_backend = "none"
        self.use_liger_loss_requested = bool(getattr(self.args, "use_liger_loss", True))
        self.use_liger_loss_effective: Optional[bool] = None
        self.liger_disable_reason = ""
        self.liger_linear_head_path = ""
        self.liger_hidden_state_path = ""

        self.policy_model = None
        self.processor = None
        self.persistent_vllm_runtime: Optional[Any] = None
        self.inline_policy_factory: Optional[Any] = None

    def _load_raw_records(self) -> List[Dict[str, Any]]:
        materialized_train_items_path = str(getattr(self.args, "materialized_train_items_path", "") or "").strip()
        include_splits_value = getattr(self.args, "include_splits", "")
        if materialized_train_items_path:
            ensure_materialized_cache_metadata(
                materialized_train_items_path,
                expected_format=MATERIALIZED_RUNTIME_ITEMS_FORMAT,
                expected_source_path=self.args.data,
                expected_include_splits=include_splits_value,
                require_source=True,
            )
            dataset = MaterializedRuntimeItemDataset(
                materialized_train_items_path,
                include_splits=include_splits_value,
                require_frame_cache=True,
                require_feature_cache=True,
            )
            return [dict(record) for record in list(getattr(dataset, "records", []) or [])]
        if bool(getattr(self.args, "require_materialized_runtime_cache", False)):
            raise ValueError(
                "Active RL requires --materialized-train-items-path when --require-materialized-runtime-cache=true."
            )
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

    def _resolve_training_proposal_support(self) -> Tuple[bool, Any]:
        strict_feature_guided_proposal = _raw_records_require_feature_guided_proposal(self.raw_records)
        if strict_feature_guided_proposal and not str(getattr(self.args, "proposal_model_path", "") or "").strip():
            raise ValueError(
                "Active RL requires proposal_model_path because the rollout environment exposes seek_evidence."
            )
        proposal_runtime = (
            _load_training_proposal_runtime(
                proposal_model_path=str(getattr(self.args, "proposal_model_path", "") or ""),
                proposal_torch_dtype=str(getattr(self.args, "proposal_torch_dtype", "auto") or "auto"),
                proposal_device=str(getattr(self.args, "proposal_device", "") or ""),
                runtime=self.runtime,
            )
            if strict_feature_guided_proposal
            else None
        )
        return bool(strict_feature_guided_proposal), proposal_runtime

    def _build_dataset(self) -> Any:
        materialized_train_items_path = str(getattr(self.args, "materialized_train_items_path", "") or "").strip()
        include_splits_value = getattr(self.args, "include_splits", "")
        if bool(getattr(self.args, "require_materialized_runtime_cache", False)) and not materialized_train_items_path:
            raise ValueError(
                "Active RL requires --materialized-train-items-path when --require-materialized-runtime-cache=true."
            )
        if materialized_train_items_path:
            return MaterializedRuntimeItemDataset(
                materialized_train_items_path,
                include_splits=include_splits_value,
                require_frame_cache=True,
                require_feature_cache=True,
            )
        return SaverAgentDataset(
            self.args.data,
            data_root=self.args.data_root,
            config=self.config_builder(self.args),
            include_splits=include_splits_value,
            require_frame_cache=True,
            require_feature_cache=True,
        )

    def _ensure_models_loaded(self) -> Tuple[Any, Any]:
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
        return self.policy_model, self.processor

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
            "reference_model_mode": str(self.reference_model_mode),
            "reference_model_source_path": str(self.current_model_path),
            "reference_model_backend": str(self.reference_model_backend),
            "rollout_eval_output_dir": str(self.rollout_eval_output_root / f"iter_{int(iteration):03d}"),
            "inline_rollout_eval": bool(getattr(self.args, "inline_rollout_eval", False)),
            "rollout_eval_start_iteration": int(getattr(self.args, "rollout_eval_start_iteration", 1) or 1),
            "rollout_eval_interval_iterations": int(getattr(self.args, "rollout_eval_interval_iterations", 1) or 1),
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
            "use_liger_loss_requested": bool(getattr(self.args, "use_liger_loss", True)),
            "use_liger_loss_effective": None,
            "liger_disable_reason": "",
            "rl_replay_buffer_supported": False,
            "rl_replay_buffer_mode": "disabled_episode_grpo",
            "rl_fecv_failure_policy": str(self.args.rl_fecv_failure_policy),
            "rl_log_empty_batch_rank_summary": bool(self.args.rl_log_empty_batch_rank_summary),
        }

    def _build_initial_iteration_items(self) -> List[Dict[str, Any]]:
        return _sample_iteration_items(
            base_dataset=self.dataset,
            raw_records=self.raw_records,
            raw_record_count=len(self.raw_records),
            select_iteration_indices_fn=self.select_iteration_indices_fn,
            rollout_count=int(self.args.rollout_count),
            rollout_start_index=int(self.args.rollout_start_index),
            iteration_index=0,
            seed=int(getattr(self.args, "seed", 42) or 42),
        )

    def _run_continuous_training(self) -> Dict[str, Any]:
        continuous_args = _continuous_rl_args(self.args)
        model, processor = self._ensure_models_loaded()
        initial_items = self._build_initial_iteration_items()
        mutable_dataset = MutableIterationDataset(initial_items)
        trainer = create_trl_vllm_grpo_trainer(
            args=continuous_args,
            model=model,
            processor=processor,
            trainer_init_model_path=self.current_model_path,
            train_items=initial_items,
            train_dataset=mutable_dataset,
            checkpoint_dir=self.output_dir,
            iteration_index=0,
            num_iterations=int(self.args.num_iterations),
            config=self.config_builder(self.args),
            rollout_eval_callback=None,
            vllm_runtime=self.persistent_vllm_runtime,
            proposal_runtime=self.proposal_runtime,
            strict_feature_guided_proposal=self.strict_feature_guided_proposal,
            save_strategy="epoch",
        )
        continuous_callback = _build_continuous_iteration_callback(
            owner=self,
            mutable_dataset=mutable_dataset,
            trainer=trainer,
            processor=processor,
            eval_start_iteration=int(getattr(self.args, "rollout_eval_start_iteration", 1) or 1),
            eval_every_iterations=int(getattr(self.args, "rollout_eval_interval_iterations", 1) or 1),
            final_rollout_eval=bool(getattr(self.args, "final_rollout_eval", False)),
        )
        trainer.add_callback(continuous_callback)
        try:
            train_result = trainer.train()
            self.reference_model_backend = str(getattr(trainer, "reference_model_backend", "none"))
            self.reference_model_source_path = str(
                getattr(trainer, "reference_model_source_path", self.reference_model_source_path)
            )
            self.use_liger_loss_effective = bool(
                getattr(trainer, "use_liger_loss_effective", getattr(trainer, "use_liger_loss", False))
            )
            self.liger_disable_reason = str(getattr(trainer, "_liger_runtime_disable_reason", "") or "")
            self.liger_linear_head_path = str(getattr(trainer, "_liger_linear_head_path", "") or "")
            self.liger_hidden_state_path = str(getattr(trainer, "_liger_hidden_state_path", "") or "")
            final_summary = {
                "timestamp_utc": utc_timestamp(),
                "output_dir": str(self.output_dir),
                "rollout_eval_output_root": str(self.rollout_eval_output_root),
                "latest_checkpoint": str(self.latest_checkpoint),
                "num_iterations": int(self.args.num_iterations),
                "reference_model_mode": str(self.reference_model_mode),
                "reference_model_source_path": str(self.reference_model_source_path),
                "reference_model_backend": str(self.reference_model_backend),
                "rl_backend": "trl_vllm_grpo",
                "rl_reward_version": str(getattr(self.args, "rl_reward_version", DEFAULT_RL_REWARD_VERSION)),
                "inline_rollout_eval": bool(getattr(self.args, "inline_rollout_eval", False)),
                "rollout_eval_start_iteration": int(getattr(self.args, "rollout_eval_start_iteration", 1) or 1),
                "rollout_eval_interval_iterations": int(getattr(self.args, "rollout_eval_interval_iterations", 1) or 1),
                "final_rollout_eval": bool(getattr(self.args, "final_rollout_eval", False)),
                "checkpoint_strategy": _checkpoint_strategy_label(
                    eval_start_iteration=int(getattr(self.args, "rollout_eval_start_iteration", 1) or 1),
                    eval_every_iterations=int(getattr(self.args, "rollout_eval_interval_iterations", 1) or 1),
                ),
                "use_vllm": bool(getattr(self.args, "use_vllm", True)),
                "vllm_mode": str(getattr(self.args, "vllm_mode", "colocate")),
                "vllm_tensor_parallel_size": int(getattr(self.args, "vllm_tensor_parallel_size", 1)),
                "vllm_gpu_memory_utilization": float(getattr(self.args, "vllm_gpu_memory_utilization", 0.35)),
                "rl_rollout_use_cache": bool(self.args.rl_rollout_use_cache),
                "rl_fecv_use_cache": bool(self.args.rl_fecv_use_cache),
                "rl_compute_loss_microbatch_size": int(self.args.rl_compute_loss_microbatch_size),
                "rl_steps_per_generation": int(getattr(self.args, "rl_steps_per_generation", 1)),
                "use_liger_loss_requested": bool(self.use_liger_loss_requested),
                "use_liger_loss_effective": self.use_liger_loss_effective,
                "liger_disable_reason": str(self.liger_disable_reason),
                "liger_linear_head_path": str(self.liger_linear_head_path),
                "liger_hidden_state_path": str(self.liger_hidden_state_path),
                "train_loss": float(getattr(train_result, "training_loss", 0.0)),
                "rl_replay_buffer_supported": False,
                "rl_replay_buffer_mode": "disabled_episode_grpo",
                "rl_fecv_failure_policy": str(self.args.rl_fecv_failure_policy),
                "rl_log_empty_batch_rank_summary": bool(self.args.rl_log_empty_batch_rank_summary),
            }
            if self.runtime.is_main_process:
                (self.output_dir / "latest_checkpoint.txt").write_text(str(self.latest_checkpoint), encoding="utf-8")
                write_json(self.resolved_log_dir / "train_saver_rl_final_summary.json", final_summary)
            return final_summary
        finally:
            if self.persistent_vllm_runtime is not None:
                setattr(trainer, "_vllm_runtime", None)
            _teardown_trainer_iteration_runtime(trainer)
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _run_iteration(self, *, iteration: int) -> None:
        iter_dir = self.output_dir / f"iter_{int(iteration):03d}"
        checkpoint_dir = iter_dir / "checkpoint"
        iter_dir.mkdir(parents=True, exist_ok=True)
        try:
            indices = self.select_iteration_indices_fn(
                len(self.raw_records),
                self.args.rollout_count,
                self.args.rollout_start_index,
                iteration,
                seed=getattr(self.args, "seed", 42),
                records=self.raw_records,
            )
        except TypeError:
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
        model, processor = self._ensure_models_loaded()
        iteration_config = self.config_builder(self.args)
        rollout_eval_config = self.eval_config_builder(
            args=self.args,
            current_model_path=self.current_model_path,
            reference_model_path=self.current_model_path,
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
            trainer_init_model_path=self.current_model_path,
            train_items=items,
            checkpoint_dir=checkpoint_dir,
            iteration_index=int(iteration),
            num_iterations=int(self.args.num_iterations),
            config=iteration_config,
            rollout_eval_callback=rollout_eval_callback,
            vllm_runtime=self.persistent_vllm_runtime,
            proposal_runtime=self.proposal_runtime,
            strict_feature_guided_proposal=self.strict_feature_guided_proposal,
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
            loadable_authority_checkpoint = _save_loadable_hf_authority_checkpoint(
                trainer=trainer,
                processor=processor,
                checkpoint_root=checkpoint_dir,
                epoch_index=int(authority_epoch_index or 1),
                runtime=self.runtime,
            )
            checkpoint_strategy = "epoch_resume_plus_loadable_hf_authority"
            _write_rl_checkpoint_authority_metadata(
                checkpoint_root=checkpoint_dir,
                authority_checkpoint=loadable_authority_checkpoint,
                iteration_index=int(iteration),
                epoch_index=int(authority_epoch_index or 1),
                runtime=self.runtime,
            )
            runtime_log(
                (
                    f"RL duplicate root save skipped: iter={int(iteration)} "
                    f"checkpoint_root={checkpoint_dir} authority_checkpoint={loadable_authority_checkpoint} "
                    f"epoch_resume_checkpoint={authority_checkpoint}"
                ),
                runtime=self.runtime,
                main_process_only=True,
            )
            self.latest_checkpoint = str(loadable_authority_checkpoint)
            self.current_model_path = self.latest_checkpoint
            self.reference_model_source_path = str(self.current_model_path)
            self.reference_model_backend = str(getattr(trainer, "reference_model_backend", "none"))
            self.use_liger_loss_effective = bool(getattr(trainer, "use_liger_loss_effective", getattr(trainer, "use_liger_loss", False)))
            self.liger_disable_reason = str(getattr(trainer, "_liger_runtime_disable_reason", "") or "")
            self.liger_linear_head_path = str(getattr(trainer, "_liger_linear_head_path", "") or "")
            self.liger_hidden_state_path = str(getattr(trainer, "_liger_hidden_state_path", "") or "")
            summary.update(
                {
                    "latest_checkpoint": str(self.latest_checkpoint),
                    "epoch_resume_checkpoint": str(authority_checkpoint),
                    "loadable_authority_checkpoint": str(loadable_authority_checkpoint),
                    "checkpoint_root": str(checkpoint_dir),
                    "checkpoint_strategy": checkpoint_strategy,
                    "authority_epoch_index": int(authority_epoch_index or 1),
                    "train_loss": float(getattr(train_result, "training_loss", 0.0)),
                    "reference_model_mode": str(getattr(trainer, "reference_model_mode", self.reference_model_mode)),
                    "reference_model_source_path": str(getattr(trainer, "reference_model_source_path", self.current_model_path)),
                    "reference_model_backend": str(getattr(trainer, "reference_model_backend", "none")),
                    "use_liger_loss_requested": bool(getattr(trainer, "use_liger_loss_requested", getattr(self.args, "use_liger_loss", True))),
                    "use_liger_loss_effective": bool(getattr(trainer, "use_liger_loss_effective", getattr(trainer, "use_liger_loss", False))),
                    "liger_disable_reason": str(getattr(trainer, "_liger_runtime_disable_reason", "") or ""),
                    "liger_linear_head_path": str(getattr(trainer, "_liger_linear_head_path", "") or ""),
                    "liger_hidden_state_path": str(getattr(trainer, "_liger_hidden_state_path", "") or ""),
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
                f"reference_mode={self.reference_model_mode} "
                f"steps_per_generation={int(getattr(self.args, 'rl_steps_per_generation', 1))} "
                f"reward_version={str(getattr(self.args, 'rl_reward_version', DEFAULT_RL_REWARD_VERSION))} "
                f"use_vllm={bool(getattr(self.args, 'use_vllm', True))} "
                f"vllm_mode={str(getattr(self.args, 'vllm_mode', 'colocate'))} "
                f"rollout_cache={bool(self.args.rl_rollout_use_cache)} fecv_cache={bool(self.args.rl_fecv_use_cache)} "
                f"loss_microbatch={int(self.args.rl_compute_loss_microbatch_size)} "
                f"rollout_eval_mode={'inline' if bool(getattr(self.args, 'inline_rollout_eval', False)) else 'deferred'} "
                f"rollout_eval_start_iteration={int(getattr(self.args, 'rollout_eval_start_iteration', 1) or 1)} "
                f"rollout_eval_interval_iterations={int(getattr(self.args, 'rollout_eval_interval_iterations', 1) or 1)} "
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
            self.inline_policy_factory = _build_inline_vllm_policy_factory(
                args=self.args,
                vllm_runtime=self.persistent_vllm_runtime,
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
            return self._run_continuous_training()
        finally:
            if self.persistent_vllm_runtime is not None:
                self.persistent_vllm_runtime.close()


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
