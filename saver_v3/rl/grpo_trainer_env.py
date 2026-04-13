from __future__ import annotations

import copy
import gc
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from saver_v3.core.counterfactual_verification import run_counterfactual_verification
from saver_v3.data.dataset import SaverAgentDataset
from saver_v3.data.materialized_cache import (
    MATERIALIZED_RUNTIME_ITEMS_FORMAT,
    MaterializedRuntimeItemDataset,
    ensure_materialized_cache_metadata,
)
from saver_v3.common.experiment_logging import append_jsonl, utc_timestamp, write_json
from saver_v3.model.qwen_policy import QwenGenerationPolicy
from saver_v3.core.reward import DEFAULT_RL_REWARD_VERSION, build_open_ended_reward_judge, score_rollout_trace
from saver_v3.core.rollout import SaverRolloutRunner
from saver_v3.common.runtime import distributed_barrier, distributed_runtime_from_env, runtime_log
from saver_v3.sft.training import (
    BatchBuildResult,
    BudgetingStats,
    _build_rl_completion_episode_spec_from_feature,
    _build_batch_from_feature,
    _format_budgeting_stats,
    _load_training_proposal_runtime,
    _normalize_batch_build_result,
    _save_epoch_resume_rng_state,
    _zero_loss_from_model,
    _unwrap_model,
    compute_completion_only_log_probs_from_ids,
    compute_completion_only_token_log_probs_from_ids,
    compute_completion_only_response_log_probs,
    compute_completion_only_response_token_log_probs,
    compute_grpo_surrogate_loss,
    compute_masked_response_token_log_probs,
    compute_masked_sampled_token_kl,
    load_qwen_model_and_processor,
    run_rollout_eval_with_policy,
)


def _raw_item_collator(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return features


def _raw_records_require_feature_guided_proposal(records: Sequence[Dict[str, Any]]) -> bool:
    for record in list(records or []):
        allowed_tools = list(((record.get("tool_io") or {}).get("allowed_tools") or []))
        if any(str(tool_name or "").strip() == "seek_evidence" for tool_name in allowed_tools):
            return True
    return False


class _RawItemDataset(torch.utils.data.Dataset):
    def __init__(self, items: Sequence[Dict[str, Any]]):
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return copy.deepcopy(self.items[int(index)])


class RepeatSampler(torch.utils.data.Sampler[int]):
    def __init__(
        self,
        *,
        data_source: Sequence[Any],
        mini_repeat_count: int = 1,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.data_source = data_source
        self.mini_repeat_count = max(1, int(mini_repeat_count))
        self.batch_size = max(1, int(batch_size))
        self.repeat_count = max(1, int(repeat_count))
        self.num_samples = max(0, int(len(data_source)))
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.generator = None
        if self.shuffle:
            self.generator = torch.Generator()
            if seed is not None:
                self.generator.manual_seed(int(seed))

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.num_samples, generator=self.generator).tolist()
        else:
            indices = list(range(self.num_samples))
        chunks = [indices[offset : offset + self.batch_size] for offset in range(0, len(indices), self.batch_size)]
        chunks = [chunk for chunk in chunks if len(chunk) == self.batch_size]
        for chunk in chunks:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        full_chunks = self.num_samples // self.batch_size
        return int(full_chunks * self.batch_size * self.mini_repeat_count * self.repeat_count)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _iter_dataloader_chain(dataloader: Any):
    current = dataloader
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "base_dataloader", None)


def _shutdown_dataloader_workers(dataloader: Any) -> None:
    for candidate in _iter_dataloader_chain(dataloader):
        iterator = getattr(candidate, "_iterator", None)
        if iterator is None:
            continue
        shutdown_fn = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_fn):
            try:
                shutdown_fn()
            except Exception:
                pass
        try:
            setattr(candidate, "_iterator", None)
        except Exception:
            pass


def _teardown_trainer_iteration_runtime(trainer: Any) -> None:
    if trainer is None:
        return
    callback_handler = getattr(trainer, "callback_handler", None)
    train_dataloader = getattr(callback_handler, "train_dataloader", None) if callback_handler is not None else None
    _shutdown_dataloader_workers(train_dataloader)
    if callback_handler is not None and hasattr(callback_handler, "train_dataloader"):
        try:
            callback_handler.train_dataloader = None
        except Exception:
            pass
    accelerator = getattr(trainer, "accelerator", None)
    optimizer = getattr(trainer, "optimizer", None)
    lr_scheduler = getattr(trainer, "lr_scheduler", None)
    if accelerator is not None and hasattr(accelerator, "free_memory"):
        try:
            released = accelerator.free_memory(optimizer, lr_scheduler)
        except TypeError:
            released = accelerator.free_memory()
        if isinstance(released, (list, tuple)) and len(released) >= 2:
            optimizer, lr_scheduler = released[:2]
    del optimizer, lr_scheduler
    trainer.optimizer = None
    trainer.lr_scheduler = None
    if hasattr(trainer, "_train_dataloader"):
        try:
            trainer._train_dataloader = None
        except Exception:
            pass
    if hasattr(trainer, "model_wrapped"):
        try:
            trainer.model_wrapped = None
        except Exception:
            pass


def _distributed_bool_consensus(local_value: bool, *, device: torch.device) -> Tuple[bool, bool]:
    local_flag = bool(local_value)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return local_flag, local_flag
    flag_tensor = torch.tensor([1 if local_flag else 0], dtype=torch.int32, device=device)
    min_tensor = flag_tensor.clone()
    max_tensor = flag_tensor.clone()
    torch.distributed.all_reduce(min_tensor, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(max_tensor, op=torch.distributed.ReduceOp.MAX)
    return bool(min_tensor.item()), bool(max_tensor.item())


def _distributed_sum_int(local_value: int, *, device: torch.device) -> int:
    local_total = int(local_value)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return local_total
    total_tensor = torch.tensor([local_total], dtype=torch.int64, device=device)
    torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)
    return int(total_tensor.item())


def _distributed_first_available_object(local_object: Any, *, device: Optional[torch.device] = None) -> Any:
    del device
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return copy.deepcopy(local_object) if local_object is not None else None
    world_size = max(1, int(torch.distributed.get_world_size()))
    gathered_objects: List[Any] = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(
        gathered_objects,
        copy.deepcopy(local_object) if local_object is not None else None,
    )
    for candidate in gathered_objects:
        if candidate is not None:
            return candidate
    return None


def _distributed_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        try:
            return max(1, int(torch.distributed.get_world_size()))
        except Exception:
            return 1
    runtime = distributed_runtime_from_env()
    return max(1, int(getattr(runtime, "world_size", 1) or 1))


def _truncate_error_message(message: Any, *, max_chars: int = 240) -> str:
    text = str(message or "").strip()
    if len(text) <= int(max_chars):
        return text
    return text[: max(0, int(max_chars) - 3)].rstrip() + "..."


def _degrade_reward_summary_for_fecv_failure(
    reward_summary: Dict[str, Any],
    *,
    error_message: str = "",
) -> Dict[str, Any]:
    degraded = copy.deepcopy(dict(reward_summary or {}))
    components = dict(degraded.get("components") or {})
    weights = dict(degraded.get("weights") or {})
    components["fecv_evidence_faithfulness_reward"] = 0.0
    components["fecv_decision_sufficiency_reward"] = 0.0
    components["fecv_specificity_reward"] = 0.0
    total_reward = degraded.get("total_reward", 0.0)
    if weights:
        total_reward = sum(float(weights.get(key, 0.0)) * float(value) for key, value in components.items())
    degraded["total_reward"] = round(float(total_reward or 0.0), 6)
    degraded["components"] = {
        key: round(float(value), 6)
        for key, value in components.items()
    }
    degraded["fecv_failed"] = True
    degraded["fecv_degraded"] = True
    degraded["fecv_failure_message"] = _truncate_error_message(error_message)
    degraded["fecv_decision_sufficiency"] = 0.0
    degraded["fecv_minimal_subset_sufficiency"] = 0.0
    degraded["fecv_negative_specificity_pass"] = 0.0
    degraded["fecv_grounded_decision"] = 0.0
    degraded["counterfactual_decision_sufficiency"] = 0.0
    degraded["counterfactual_minimal_subset_sufficiency"] = 0.0
    degraded["negative_specificity_pass"] = 0.0
    degraded["counterfactual_type_supported"] = 0.0
    degraded["fecv_counterfactual_type_supported"] = 0.0
    return degraded


def _replay_priority_from_experience(experience: Dict[str, Any]) -> float:
    eps = 1e-4
    episode_specs = list((experience or {}).get("episode_specs") or [])
    priorities: List[float] = []
    for episode_spec in episode_specs:
        feature = dict((episode_spec or {}).get("feature") or {})
        advantage = abs(float(feature.get("advantage", feature.get("sample_weight", 0.0)) or 0.0))
        if advantage > eps:
            priorities.append(advantage)
    if priorities:
        return max(eps, float(sum(priorities) / float(len(priorities))))
    return eps


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = max(0, int(capacity))
        self.buffer: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, experience: Dict[str, Any]) -> None:
        raise NotImplementedError

    def sample(self) -> Dict[str, Any]:
        if not self.buffer:
            raise ValueError("Replay buffer is empty.")
        return copy.deepcopy(self.buffer[0])


class SSRReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, alpha: float = 1.0):
        super().__init__(capacity)
        self.alpha = float(alpha)
        self.advantages: List[float] = []

    def add(self, experience: Dict[str, Any]) -> None:
        if self.capacity <= 0:
            return
        priority = _replay_priority_from_experience(experience)
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            self.advantages.pop(0)
        self.buffer.append(copy.deepcopy(dict(experience or {})))
        self.advantages.append(priority)

    def sample(self) -> Dict[str, Any]:
        if not self.buffer:
            raise ValueError("Replay buffer is empty.")
        scaled = [max(1e-6, float(priority)) ** self.alpha for priority in self.advantages]
        selected = random.choices(range(len(self.buffer)), weights=scaled, k=1)[0]
        return copy.deepcopy(self.buffer[int(selected)])


class DapoReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, alpha: float = 1.0):
        super().__init__(capacity)
        self.alpha = float(alpha)
        self.weights: List[float] = []

    def add(self, experience: Dict[str, Any]) -> None:
        if self.capacity <= 0:
            return
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
            self.weights.pop(0)
        self.buffer.append(copy.deepcopy(dict(experience or {})))
        self.weights.append(1.0)

    def sample(self) -> Dict[str, Any]:
        if not self.buffer:
            raise ValueError("Replay buffer is empty.")
        scaled = [max(1e-6, float(weight)) ** self.alpha for weight in self.weights]
        selected = random.choices(range(len(self.buffer)), weights=scaled, k=1)[0]
        return copy.deepcopy(self.buffer[int(selected)])


def get_replay_buffer(buffer_type: str, capacity: int, alpha: float = 1.0):
    normalized_type = str(buffer_type or "none").strip().lower()
    if normalized_type == "ssr":
        return SSRReplayBuffer(capacity, alpha=alpha)
    if normalized_type == "dapo":
        return DapoReplayBuffer(capacity, alpha=alpha)
    if normalized_type == "none":
        return None
    raise ValueError(f"Invalid replay buffer type: {buffer_type!r}")


class _NativeRLOptimizerStepProxy:
    def __init__(self, optimizer: Any, *, trainer: Any):
        self._optimizer = optimizer
        self._trainer = trainer

    def step(self, *args, **kwargs):
        if bool(getattr(self._trainer, "_native_rl_skip_next_optimizer_step", False)):
            try:
                setattr(self._trainer.accelerator, "optimizer_step_was_skipped", True)
            except Exception:
                pass
            self._trainer._native_rl_skip_next_optimizer_step = False
            self._trainer._optimizer_step_skips += 1
            return None
        try:
            setattr(self._trainer.accelerator, "optimizer_step_was_skipped", False)
        except Exception:
            pass
        result = self._optimizer.step(*args, **kwargs)
        self._trainer._effective_update_steps += 1
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._optimizer, name)


class _BudgetDropError(ValueError):
    pass


def _compute_samplewise_grpo_surrogate_loss(
    *,
    policy_log_probs: torch.Tensor,
    old_policy_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor:
    policy_log_probs = policy_log_probs.view(-1)
    old_policy_log_probs = old_policy_log_probs.to(policy_log_probs.device, dtype=torch.float32).view(-1)
    advantages = advantages.to(policy_log_probs.device, dtype=torch.float32).view(-1)
    if (
        policy_log_probs.numel() != old_policy_log_probs.numel()
        or policy_log_probs.numel() != advantages.numel()
    ):
        raise ValueError("policy_log_probs, old_policy_log_probs, and advantages must have matching lengths.")
    ratios = torch.exp(policy_log_probs - old_policy_log_probs)
    clipped_ratios = torch.clamp(ratios, 1.0 - float(clip_epsilon), 1.0 + float(clip_epsilon))
    surrogate_unclipped = ratios * advantages
    surrogate_clipped = clipped_ratios * advantages
    return -torch.minimum(surrogate_unclipped, surrogate_clipped)


def _compute_samplewise_masked_sampled_token_kl(
    *,
    policy_token_log_probs: torch.Tensor,
    reference_token_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    batch_size = int(policy_token_log_probs.shape[0]) if policy_token_log_probs.ndim > 0 else 1
    if not torch.any(response_mask):
        return policy_token_log_probs.new_zeros((batch_size,), dtype=torch.float32)
    delta = reference_token_log_probs.to(policy_token_log_probs.device) - policy_token_log_probs
    token_kl = torch.exp(delta) - delta - 1.0
    masked_token_kl = token_kl * response_mask.to(dtype=token_kl.dtype)
    token_counts = response_mask.sum(dim=-1)
    sample_kl = token_kl.new_zeros((batch_size,), dtype=torch.float32)
    valid = token_counts > 0
    if torch.any(valid):
        sample_kl[valid] = (
            masked_token_kl.sum(dim=-1)[valid] / token_counts[valid].to(dtype=masked_token_kl.dtype)
        ).to(torch.float32)
    return sample_kl


class _NativeGRPOProgressReporter:
    def __init__(
        self,
        *,
        runtime: Any,
        iteration_index: int,
        num_iterations: int,
        total_groups: int,
        num_generations: int,
        compute_loss_microbatch_size: int = 1,
        rollout_use_generation_cache: bool = False,
        fecv_use_generation_cache: bool = False,
    ):
        self.runtime = runtime
        self.iteration_index = max(0, int(iteration_index))
        self.num_iterations = max(1, int(num_iterations))
        self.total_groups = max(0, int(total_groups))
        self.num_generations = max(1, int(num_generations))
        self.compute_loss_microbatch_size = max(1, int(compute_loss_microbatch_size))
        self.rollout_use_generation_cache = bool(rollout_use_generation_cache)
        self.fecv_use_generation_cache = bool(fecv_use_generation_cache)
        self.trainer_step = 0
        self.processed_groups = 0
        self.batch_index = 0
        self.last_video_id = ""
        self.last_stage = ""

    def set_total_groups(self, total_groups: int) -> None:
        self.total_groups = max(0, int(total_groups))

    def set_trainer_step(self, trainer_step: int) -> None:
        self.trainer_step = max(0, int(trainer_step))

    def start_batch(self, *, num_items: int) -> None:
        del num_items
        self.batch_index += 1

    def extend_materialization_total(self, count: int) -> None:
        del count

    def advance_generation_stage(self, *, video_id: str, generation_id: int, stage: str) -> None:
        del generation_id
        self.last_video_id = str(video_id or "")
        self.last_stage = str(stage or "")

    def advance_materialization(self, *, video_id: str, completed: int, total: int) -> None:
        del completed, total
        self.last_video_id = str(video_id or "")
        self.last_stage = "materialize"

    def finish_item(self, *, video_id: str) -> None:
        self.processed_groups += 1
        self.last_video_id = str(video_id or "")
        total_display = int(self.total_groups) if self.total_groups > 0 else "?"
        runtime_log(
            (
                f"RL rank progress: iter={int(self.iteration_index) + 1}/{int(self.num_iterations)} "
                f"batch={int(self.batch_index)} groups={int(self.processed_groups)}/{total_display}"
            ),
            runtime=self.runtime,
            main_process_only=False,
        )

    def close_batch(self) -> None:
        return None

    def close(self) -> None:
        return None


def _estimate_local_total_groups(*, trainer: Any, args: Any) -> int:
    runtime = distributed_runtime_from_env()
    num_epochs = max(0.0, float(getattr(args, "num_train_epochs", 1.0) or 1.0))
    if runtime.is_distributed:
        train_dataset = getattr(trainer, "train_dataset", None)
        if train_dataset is not None and hasattr(train_dataset, "__len__"):
            try:
                global_total = max(0, int(len(train_dataset)))
            except Exception:
                global_total = 0
            if global_total > 0:
                local_total = int(math.ceil(float(global_total) / float(max(1, int(runtime.world_size)))))
                return max(1, int(math.ceil(float(local_total) * num_epochs)))

    local_total = 0
    try:
        train_dataloader = trainer.get_train_dataloader()
    except Exception:
        train_dataloader = None
    if train_dataloader is not None:
        sampler = getattr(train_dataloader, "sampler", None)
        batch_sampler = getattr(train_dataloader, "batch_sampler", None)
        if sampler is None and batch_sampler is not None:
            sampler = getattr(batch_sampler, "sampler", None)
        for candidate in (sampler, getattr(train_dataloader, "dataset", None), getattr(trainer, "train_dataset", None)):
            if candidate is None or not hasattr(candidate, "__len__"):
                continue
            try:
                local_total = max(0, int(len(candidate)))
            except Exception:
                local_total = 0
            if local_total > 0:
                break
    if local_total <= 0:
        return 0
    return max(1, int(math.ceil(float(local_total) * num_epochs)))


def _rl_epoch_resume_dir(output_dir: str | Path, epoch_index: int) -> Path:
    return Path(output_dir) / "epoch_resume" / f"epoch_{int(epoch_index):03d}"


def _write_rl_checkpoint_authority_metadata(
    *,
    checkpoint_root: str | Path,
    authority_checkpoint: str | Path,
    iteration_index: int,
    epoch_index: int,
    runtime: Any,
) -> None:
    if not bool(getattr(runtime, "is_main_process", False)):
        return
    checkpoint_root = Path(checkpoint_root)
    authority_checkpoint = Path(authority_checkpoint)
    write_json(
        checkpoint_root / "checkpoint_authority.json",
        {
            "iteration": int(iteration_index),
            "epoch_index": int(epoch_index),
            "checkpoint_root": str(checkpoint_root),
            "authority_checkpoint": str(authority_checkpoint),
            "checkpoint_strategy": "epoch_resume_only",
        },
    )


def _build_rl_authority_checkpoint_callback(
    *,
    processor: Any,
    rollout_eval_config: Any = None,
    rollout_eval_output_dir: str | Path = "",
    iteration_index: int = 0,
    policy_factory: Any = None,
):
    try:
        from transformers import TrainerCallback
    except Exception:
        class TrainerCallback:  # type: ignore[no-redef]
            pass

    class RLAuthorityCheckpointCallback(TrainerCallback):
        def __init__(self):
            self.processor = processor
            self.rollout_eval_config = rollout_eval_config
            self.rollout_eval_output_dir = str(rollout_eval_output_dir or "").strip()
            self.iteration_index = max(0, int(iteration_index))
            self.policy_factory = policy_factory
            self.runtime = distributed_runtime_from_env()
            self.checkpoint_strategy = "epoch_resume_only"
            self.last_authority_checkpoint_path: Optional[Path] = None
            self.last_authority_epoch_index: Optional[int] = None

        def on_epoch_end(self, args, state, control, model=None, **kwargs):
            if model is None:
                return control
            eval_model = _unwrap_model(model)
            was_training = bool(getattr(eval_model, "training", False))
            if hasattr(eval_model, "eval"):
                eval_model.eval()
            try:
                epoch_index = max(1, int(round(float(getattr(state, "epoch", 0.0) or 0.0))))
                checkpoint_root = Path(args.output_dir)
                authority_checkpoint = _rl_epoch_resume_dir(checkpoint_root, epoch_index)
                runtime_log(
                    (
                        f"RL authority checkpoint save start: iter={int(self.iteration_index)} "
                        f"epoch={int(epoch_index)} path={authority_checkpoint}"
                    ),
                    runtime=self.runtime,
                    main_process_only=True,
                )
                authority_checkpoint.mkdir(parents=True, exist_ok=True)
                if self.runtime.is_main_process:
                    eval_model.save_pretrained(str(authority_checkpoint))
                    if hasattr(self.processor, "save_pretrained"):
                        self.processor.save_pretrained(str(authority_checkpoint))
                    optimizer = kwargs.get("optimizer")
                    if optimizer is not None:
                        torch.save(optimizer.state_dict(), authority_checkpoint / "optimizer.pt")
                    lr_scheduler = kwargs.get("lr_scheduler")
                    if lr_scheduler is not None:
                        torch.save(lr_scheduler.state_dict(), authority_checkpoint / "scheduler.pt")
                    if state is not None and hasattr(state, "save_to_json"):
                        state.save_to_json(str(authority_checkpoint / "trainer_state.json"))
                    write_json(
                        authority_checkpoint / "resume_metadata.json",
                        {
                            "epoch_index": int(epoch_index),
                            "iteration": int(self.iteration_index),
                            "checkpoint_kind": "rl_epoch_end_authority",
                            "checkpoint_root": str(checkpoint_root),
                            "checkpoint_strategy": self.checkpoint_strategy,
                            "world_size": int(getattr(self.runtime, "world_size", 1)),
                            "global_step": int(getattr(state, "global_step", 0) or 0) if state is not None else 0,
                            "epoch": float(getattr(state, "epoch", float(epoch_index)) or float(epoch_index)),
                        },
                    )
                _save_epoch_resume_rng_state(authority_checkpoint, runtime=self.runtime)
                _write_rl_checkpoint_authority_metadata(
                    checkpoint_root=checkpoint_root,
                    authority_checkpoint=authority_checkpoint,
                    iteration_index=self.iteration_index,
                    epoch_index=epoch_index,
                    runtime=self.runtime,
                )
                runtime_log(
                    (
                        f"RL authority checkpoint save done: iter={int(self.iteration_index)} "
                        f"epoch={int(epoch_index)} path={authority_checkpoint}"
                    ),
                    runtime=self.runtime,
                    main_process_only=True,
                )
                runtime_log(
                    f"entering RL authority checkpoint barrier at {authority_checkpoint}",
                    runtime=self.runtime,
                    main_process_only=False,
                )
                distributed_barrier(self.runtime)
                runtime_log(
                    f"passed RL authority checkpoint barrier at {authority_checkpoint}",
                    runtime=self.runtime,
                    main_process_only=False,
                )
                self.last_authority_checkpoint_path = authority_checkpoint
                self.last_authority_epoch_index = int(epoch_index)
                if self.rollout_eval_config is not None:
                    if bool(getattr(self.rollout_eval_config, "inline_rollout_eval", False)):
                        runtime_log(
                            f"RL epoch-end rollout eval inline: authority_checkpoint={authority_checkpoint}",
                            runtime=self.runtime,
                            main_process_only=True,
                        )
                        if callable(self.policy_factory):
                            policy = self.policy_factory(
                                eval_model=eval_model,
                                processor=self.processor,
                                rollout_eval_config=self.rollout_eval_config,
                                state=state,
                                runtime=self.runtime,
                            )
                        else:
                            policy = QwenGenerationPolicy.from_components(
                                model=eval_model,
                                processor=self.processor,
                                max_new_tokens=int(self.rollout_eval_config.policy_max_new_tokens),
                                max_total_images=int(self.rollout_eval_config.max_total_images),
                                max_seq_length=int(self.rollout_eval_config.max_seq_length),
                                keep_recent_tool_image_messages=int(
                                    self.rollout_eval_config.keep_recent_tool_image_messages
                                ),
                                keep_recent_text_messages=int(self.rollout_eval_config.keep_recent_text_messages),
                                max_image_side=int(self.rollout_eval_config.max_image_side),
                                max_image_pixels=int(self.rollout_eval_config.max_image_pixels),
                                do_sample=False,
                                use_generation_cache=bool(self.rollout_eval_config.use_generation_cache),
                            )
                        run_rollout_eval_with_policy(
                            policy,
                            rollout_eval_config=self.rollout_eval_config,
                            output_dir=checkpoint_root,
                            rollout_eval_output_dir=self.rollout_eval_output_dir,
                            epoch_index=epoch_index,
                            epoch_value=float(getattr(state, "epoch", epoch_index) or epoch_index),
                            runtime=self.runtime,
                        )
                    else:
                        runtime_log(
                            f"RL epoch-end rollout eval deferred: authority_checkpoint={authority_checkpoint}",
                            runtime=self.runtime,
                            main_process_only=True,
                        )
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if was_training and hasattr(eval_model, "train"):
                    eval_model.train()
            return control

    return RLAuthorityCheckpointCallback()


def _build_native_grpo_progress_callback(*, trainer: Any):
    try:
        from transformers import TrainerCallback
    except Exception:
        class TrainerCallback:  # type: ignore[no-redef]
            pass

    class NativeGRPOProgressCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            progress = getattr(trainer, "_native_grpo_progress", None)
            if progress is not None:
                progress.set_total_groups(_estimate_local_total_groups(trainer=trainer, args=args))
                progress.set_trainer_step(int(getattr(state, "global_step", 0) or 0))
            return control

        def on_step_end(self, args, state, control, **kwargs):
            del args, kwargs
            progress = getattr(trainer, "_native_grpo_progress", None)
            if progress is not None:
                progress.set_trainer_step(int(getattr(state, "global_step", 0) or 0))
            return control

        def on_train_end(self, args, state, control, **kwargs):
            del args, state, kwargs
            progress = getattr(trainer, "_native_grpo_progress", None)
            if progress is not None:
                progress.close()
            return control

    return NativeGRPOProgressCallback()


def _is_episode_feature_from_feature(feature: Dict[str, Any]) -> bool:
    return isinstance(feature.get("messages"), list) and isinstance(feature.get("assistant_supervision"), list)


def _compute_group_relative_advantages(
    rollouts: Sequence[Dict[str, Any]],
    *,
    clip_value: Optional[float] = None,
    eps: float = 1e-6,
) -> List[Dict[str, Any]]:
    rewards = [float(((rollout.get("reward_summary") or {}).get("total_reward")) or 0.0) for rollout in rollouts]
    mean_reward = sum(rewards) / float(len(rewards)) if rewards else 0.0
    variance = sum((reward - mean_reward) ** 2 for reward in rewards) / float(len(rewards)) if rewards else 0.0
    std_reward = math.sqrt(max(variance, 0.0))
    updated: List[Dict[str, Any]] = []
    for rollout, reward in zip(rollouts, rewards):
        advantage = 0.0 if std_reward <= eps else (reward - mean_reward) / (std_reward + eps)
        if clip_value is not None:
            advantage = max(-float(clip_value), min(float(clip_value), float(advantage)))
        enriched = copy.deepcopy(rollout)
        enriched["group_reward"] = float(reward)
        enriched["group_reward_mean"] = float(mean_reward)
        enriched["group_reward_std"] = float(std_reward)
        enriched["group_advantage"] = round(float(advantage), 6)
        updated.append(enriched)
    return updated


def _flatten_rollout_to_episode_features(
    rollout: Dict[str, Any],
    *,
    min_abs_advantage: float = 0.0,
) -> List[Dict[str, Any]]:
    rollout_advantage = float(rollout.get("group_advantage", 0.0) or 0.0)
    if abs(rollout_advantage) < float(min_abs_advantage):
        return []

    features: List[Dict[str, Any]] = []
    for turn in list(rollout.get("turns") or []):
        if not bool(turn.get("valid_action")):
            continue
        prompt_messages = turn.get("_prompt_messages")
        target_response = str(turn.get("assistant_response_raw") or "").strip()
        if not isinstance(prompt_messages, list) or not target_response:
            continue
        episode_messages = copy.deepcopy(prompt_messages)
        episode_messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_response}],
            }
        )
        features.append(
            {
                "video_id": rollout.get("video_id"),
                "group_id": rollout.get("group_id"),
                "generation_id": rollout.get("generation_id"),
                "step_index": int(turn.get("step_index") or 0),
                "_rl_prompt_completion_native": True,
                "prompt_messages": copy.deepcopy(prompt_messages),
                "completion_text": target_response,
                "messages": episode_messages,
                "assistant_supervision": [
                    {
                        "assistant_message_index": len(episode_messages) - 1,
                        "kind": str(turn.get("action") or "assistant"),
                        "loss_weight": 1.0,
                    }
                ],
                "target_response": target_response,
                "sample_weight": float(rollout_advantage),
                "advantage": float(rollout_advantage),
                "target_action": turn.get("action"),
                "tool_name": turn.get("tool_name"),
            }
        )
    return features


_NATIVE_GRPO_TRAINER_CLASS_CACHE: Dict[type, type] = {}


def _build_native_grpo_trainer_class(trainer_base: type) -> type:
    trainer_class = _NATIVE_GRPO_TRAINER_CLASS_CACHE.get(trainer_base)
    if trainer_class is None:
        trainer_class = type("NativeGRPOTrainer", (_NativeGRPOTrainerMixin, trainer_base), {})
        _NATIVE_GRPO_TRAINER_CLASS_CACHE[trainer_base] = trainer_class
    return trainer_class


class _NativeGRPOTrainerMixin:
    def __init__(self, *trainer_args, native_grpo_config: Optional[Dict[str, Any]] = None, **trainer_kwargs):
        config = dict(native_grpo_config or {})
        if not config:
            raise ValueError("native_grpo_config is required when constructing the native GRPO trainer.")
        train_dataset = config.get("train_dataset")
        self.processor = config["processor"]
        self.old_policy_model = config["old_policy_model"]
        self.reference_model = config["reference_model"]
        self.use_lora_reference_disable_adapter = bool(config["use_lora_reference_disable_adapter"])
        self.kl_beta = float(config["kl_beta"])
        self.ppo_clip_epsilon = float(config["ppo_clip_epsilon"])
        self.rollout_runner = config["rollout_runner"]
        self.num_generations = max(1, int(config["num_generations"]))
        self.min_weight = max(0.0, float(config["min_weight"]))
        self.advantage_clip = float(config["advantage_clip"])
        self.policy_max_new_tokens = int(config["policy_max_new_tokens"])
        self.max_image_side = int(config["max_image_side"])
        self.max_image_pixels = int(config["max_image_pixels"])
        self.max_total_images = int(config["max_total_images"])
        self.keep_recent_tool_image_messages = int(config["keep_recent_tool_image_messages"])
        self.keep_recent_text_messages = int(config["keep_recent_text_messages"])
        self.max_seq_length = int(config["max_seq_length"])
        self.counterfactual_max_images = int(config["counterfactual_max_images"])
        self.policy_do_sample = bool(config["policy_do_sample"])
        self.policy_temperature = config["policy_temperature"]
        self.policy_top_p = config["policy_top_p"]
        self.policy_top_k = config["policy_top_k"]
        self.policy_repetition_penalty = config["policy_repetition_penalty"]
        self.rollout_use_generation_cache = bool(config["rollout_use_generation_cache"])
        self.fecv_use_generation_cache = bool(config["fecv_use_generation_cache"])
        self.compute_loss_microbatch_size = max(1, int(config["compute_loss_microbatch_size"]))
        self.steps_per_generation = max(1, int(config["steps_per_generation"]))
        self._generation_step_batch_size = max(1, int(config["per_device_train_batch_size"]))
        self._generation_batch_size = max(1, self._generation_step_batch_size * self.steps_per_generation)
        self._buffered_generation_step_payloads: List[Dict[str, Any]] = []
        self._buffered_generation_batch_key: Optional[Tuple[Any, ...]] = None
        self.replay_buffer_enable = bool(config["replay_buffer_enable"])
        self.replay_buffer_type = str(config["replay_buffer_type"] or "none").strip().lower()
        self.replay_buffer_capacity = max(0, int(config["replay_buffer_capacity"]))
        self.replay_buffer_alpha = float(config["replay_buffer_alpha"])
        self.fecv_failure_policy = str(config["fecv_failure_policy"] or "degrade").strip().lower()
        self.all_empty_policy = str(config["all_empty_policy"] or "true_skip").strip().lower()
        self.log_empty_batch_rank_summary = bool(config["log_empty_batch_rank_summary"])
        self.reward_version = str(config["reward_version"] or DEFAULT_RL_REWARD_VERSION).strip().lower()
        self.reward_config = dict(config["reward_config"] or {})
        self.reward_config.setdefault("reward_version", self.reward_version)
        self.reward_judge = build_open_ended_reward_judge(reward_config=self.reward_config)
        self._reference_model_device = None
        self._budgeting_stats = BudgetingStats()
        self._zero_response_dropped = 0
        self._materialize_fallback_batches = 0
        self._completion_only_grad_fallback_batches = 0
        self._ddp_global_empty_batch_skips = 0
        self._all_empty_batch_skips = 0
        self._effective_update_steps = 0
        self._optimizer_step_skips = 0
        self._replay_fill_batches = 0
        self._replay_fill_episode_specs = 0
        self._groups_all_zero_advantage = 0
        self._groups_filtered_by_min_weight = 0
        self._fecv_failure_count = 0
        self._fecv_degraded_rollout_count = 0
        self._native_rl_skip_next_optimizer_step = False
        self._native_rl_last_skip_reason = ""
        self.replay_buffer = get_replay_buffer(
            self.replay_buffer_type if self.replay_buffer_enable else "none",
            capacity=self.replay_buffer_capacity,
            alpha=self.replay_buffer_alpha,
        )
        try:
            initial_total_groups = max(0, int(len(train_dataset)))
        except Exception:
            initial_total_groups = 0
        self._native_grpo_progress = _NativeGRPOProgressReporter(
            runtime=distributed_runtime_from_env(),
            iteration_index=int(config["iteration_index"]),
            num_iterations=int(config["num_iterations"]),
            total_groups=initial_total_groups,
            num_generations=self.num_generations,
            compute_loss_microbatch_size=self.compute_loss_microbatch_size,
            rollout_use_generation_cache=self.rollout_use_generation_cache,
            fecv_use_generation_cache=self.fecv_use_generation_cache,
        )
        super().__init__(*trainer_args, **trainer_kwargs)
        if self.reference_model is not None:
            self.reference_model.eval()
            for parameter in self.reference_model.parameters():
                parameter.requires_grad_(False)

    def _prepare_for_training(self, max_steps, train_dataloader, resume_from_checkpoint):
        model, train_dataloader = super()._prepare_for_training(
            max_steps,
            train_dataloader,
            resume_from_checkpoint,
        )
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None and not isinstance(optimizer, _NativeRLOptimizerStepProxy):
            self.optimizer = _NativeRLOptimizerStepProxy(optimizer, trainer=self)
            if getattr(self, "callback_handler", None) is not None:
                self.callback_handler.optimizer = self.optimizer
        return model, train_dataloader

    def get_train_dataloader(self):
        train_dataset = getattr(self, "train_dataset", None)
        if train_dataset is None:
            raise ValueError("Trainer-native GRPO requires a train_dataset.")
        batch_size = int(
            getattr(
                self,
                "_train_batch_size",
                getattr(self.args, "per_device_train_batch_size", self._generation_step_batch_size),
            )
        )
        generation_batch_size = max(1, batch_size * self.steps_per_generation)
        dataloader_params: Dict[str, Any] = {
            "batch_size": generation_batch_size,
            "collate_fn": getattr(self, "data_collator", _raw_item_collator),
            "num_workers": int(getattr(self.args, "dataloader_num_workers", 0) or 0),
            "pin_memory": bool(getattr(self.args, "dataloader_pin_memory", False)),
            "persistent_workers": bool(getattr(self.args, "dataloader_persistent_workers", False)),
        }
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = bool(getattr(self.args, "dataloader_drop_last", False))
            try:
                from transformers.trainer_utils import seed_worker as _seed_worker
            except Exception:
                _seed_worker = None
            if _seed_worker is not None:
                dataloader_params["worker_init_fn"] = _seed_worker
            if int(dataloader_params["num_workers"]) > 0:
                prefetch_factor = getattr(self.args, "dataloader_prefetch_factor", None)
                if prefetch_factor is not None:
                    dataloader_params["prefetch_factor"] = int(prefetch_factor)
        dataloader = torch.utils.data.DataLoader(train_dataset, **dataloader_params)
        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None and hasattr(accelerator, "prepare"):
            return accelerator.prepare(dataloader)
        return dataloader

    def _get_train_sampler(self) -> torch.utils.data.Sampler[int]:
        return RepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=1,
            batch_size=max(1, int(self._generation_batch_size)),
            repeat_count=max(1, int(self.steps_per_generation)),
            shuffle=False,
            seed=getattr(self.args, "seed", None),
        )

    def _mark_skip_next_optimizer_step(self, *, reason: str, all_empty: bool = False) -> None:
        self._native_rl_skip_next_optimizer_step = True
        self._native_rl_last_skip_reason = str(reason or "")
        if all_empty:
            self._all_empty_batch_skips += 1

    def _maybe_log_empty_batch_rank_summary(
        self,
        *,
        reason: str,
        runtime_stats: Dict[str, Any],
        trainable_samples: int,
    ) -> None:
        if not self.log_empty_batch_rank_summary:
            return
        runtime = distributed_runtime_from_env()
        runtime_log(
            (
                f"RL empty-batch rank summary: reason={str(reason)} "
                f"episode_specs={int(runtime_stats.get('raw_local_episode_spec_count', 0))} "
                f"prepared_batches={int(runtime_stats.get('raw_local_prepared_batch_count', 0))} "
                f"trainable_samples={int(trainable_samples)} "
                f"fecv_failures={int(runtime_stats.get('local_fecv_failure_count', 0))} "
                f"min_weight_drops={int(runtime_stats.get('groups_filtered_by_min_weight', 0))} "
                f"replay_fills={int(runtime_stats.get('replay_fill_batches', 0))}"
            ),
            runtime=runtime,
            main_process_only=False,
        )

    def _add_episode_specs_to_replay_buffer(self, episode_specs: Sequence[Dict[str, Any]]) -> None:
        if self.replay_buffer is None or not episode_specs:
            return
        self.replay_buffer.add({"episode_specs": [copy.deepcopy(spec) for spec in episode_specs]})

    def _sample_episode_specs_from_replay_buffer(self) -> List[Dict[str, Any]]:
        if self.replay_buffer is None or len(self.replay_buffer) <= 0:
            return []
        sampled = self.replay_buffer.sample()
        return list(sampled.get("episode_specs") or [])

    def _ensure_aux_model_device(self, aux_model: Any, current_device: Any, device_attr_name: str) -> None:
        if aux_model is None:
            return
        if getattr(self, device_attr_name) != current_device:
            aux_model.to(current_device)
            aux_model.eval()
            setattr(self, device_attr_name, current_device)

    def _ensure_reference_model_device(self, model: Any) -> None:
        if self.reference_model is None:
            return
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            return
        self._ensure_aux_model_device(self.reference_model, target_device, "_reference_model_device")

    def _new_rollout_metric_lists(self) -> Dict[str, List[float]]:
        return {
            "reward_total": [],
            "reward_accuracy": [],
            "reward_fecv_evidence": [],
            "reward_protocol_finalize": [],
            "reward_fecv_decision": [],
            "reward_fecv_specificity": [],
        }

    def _new_runtime_stats(self) -> Dict[str, int]:
        return {
            "raw_local_episode_spec_count": 0,
            "raw_local_prepared_batch_count": 0,
            "raw_local_sample_count": 0,
            "local_fecv_failure_count": 0,
            "groups_filtered_by_min_weight": 0,
            "groups_all_zero_advantage": 0,
            "replay_fill_batches": 0,
            "replay_fill_episode_specs": 0,
        }

    def _build_generation_batch_key(self, generation_items: Sequence[Dict[str, Any]]) -> Tuple[Any, ...]:
        key: List[Any] = []
        for index, item in enumerate(generation_items):
            if isinstance(item, dict):
                key.append(
                    (
                        str(item.get("video_id") or ""),
                        str(item.get("split") or ""),
                        str(item.get("question_id") or ""),
                        str(item.get("dataset_index") if item.get("dataset_index") is not None else ""),
                        int(index),
                    )
                )
            else:
                key.append((int(index), repr(item)))
        return tuple(key)

    def _empty_generation_step_payload(self, *, video_ids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        return {
            "episode_specs": [],
            "rollout_metrics": {key: 0.0 for key in self._new_rollout_metric_lists().keys()},
            "budgeting_metrics": self.get_budget_drop_metrics(),
            "runtime_stats": self._new_runtime_stats(),
            "video_ids": [str(video_id or "") for video_id in (video_ids or [])],
        }

    def _build_generation_item_payload(
        self,
        item: Dict[str, Any],
        rollout_model: Any,
    ) -> Dict[str, Any]:
        video_id = str(item.get("video_id") or "")
        progress = getattr(self, "_active_generation_progress", None)
        rollout_metrics = self._new_rollout_metric_lists()
        runtime_stats = self._new_runtime_stats()
        item_episode_specs: List[Dict[str, Any]] = []
        scored_rollouts = self._generate_scored_rollouts(item, rollout_model, progress=None)
        item_materialization_total = 0
        item_materialized_completed = 0
        for rollout in scored_rollouts:
            if bool(rollout.get("fecv_failed")):
                runtime_stats["local_fecv_failure_count"] += 1
            reward_summary = dict(rollout.get("reward_summary") or {})
            components = dict(reward_summary.get("components") or {})
            rollout_metrics["reward_total"].append(_safe_float(reward_summary.get("total_reward")))
            rollout_metrics["reward_accuracy"].append(_safe_float(components.get("accuracy_reward")))
            rollout_metrics["reward_fecv_evidence"].append(
                _safe_float(components.get("fecv_evidence_faithfulness_reward"))
            )
            rollout_metrics["reward_protocol_finalize"].append(
                _safe_float(components.get("protocol_finalize_reward"))
            )
            rollout_metrics["reward_fecv_decision"].append(
                _safe_float(components.get("fecv_decision_sufficiency_reward"))
            )
            rollout_metrics["reward_fecv_specificity"].append(
                _safe_float(components.get("fecv_specificity_reward"))
            )
            rollout_advantage = abs(float(rollout.get("group_advantage", 0.0) or 0.0))
            if rollout_advantage < float(self.min_weight):
                runtime_stats["groups_filtered_by_min_weight"] += 1
                if rollout_advantage <= 0.0:
                    runtime_stats["groups_all_zero_advantage"] += 1
            features = _flatten_rollout_to_episode_features(
                rollout,
                min_abs_advantage=self.min_weight,
            )
            item_materialization_total += len(features)
            if progress is not None:
                progress.extend_materialization_total(len(features))
            for feature in features:
                episode_spec = self._cache_old_policy_log_probs(feature=feature)
                item_materialized_completed += 1
                if progress is not None:
                    progress.advance_materialization(
                        video_id=str(feature.get("video_id") or rollout.get("video_id") or video_id),
                        completed=item_materialized_completed,
                        total=item_materialization_total,
                    )
                if episode_spec is not None:
                    item_episode_specs.append(episode_spec)
        if item_episode_specs:
            self._populate_old_policy_log_probs(rollout_model, item_episode_specs)
        runtime_stats["raw_local_episode_spec_count"] = int(len(item_episode_specs))
        return {
            "video_id": video_id,
            "episode_specs": item_episode_specs,
            "rollout_metric_values": rollout_metrics,
            "runtime_stats": runtime_stats,
        }

    def _aggregate_generation_step_payload(
        self,
        item_payloads: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        episode_specs: List[Dict[str, Any]] = []
        rollout_metric_values = self._new_rollout_metric_lists()
        runtime_stats = self._new_runtime_stats()
        video_ids: List[str] = []
        for payload in item_payloads:
            video_ids.append(str(payload.get("video_id") or ""))
            episode_specs.extend(list(payload.get("episode_specs") or []))
            for key, values in dict(payload.get("rollout_metric_values") or {}).items():
                rollout_metric_values.setdefault(key, [])
                rollout_metric_values[key].extend([_safe_float(value) for value in (values or [])])
            for key, value in dict(payload.get("runtime_stats") or {}).items():
                runtime_stats[key] = int(runtime_stats.get(key, 0)) + int(value or 0)
        runtime_stats["raw_local_episode_spec_count"] = int(len(episode_specs))
        if runtime_stats["raw_local_episode_spec_count"] <= 0 and self.replay_buffer is not None:
            replay_episode_specs = self._sample_episode_specs_from_replay_buffer()
            if replay_episode_specs:
                episode_specs = replay_episode_specs
                runtime_stats["replay_fill_batches"] = int(runtime_stats.get("replay_fill_batches", 0)) + 1
                runtime_stats["replay_fill_episode_specs"] = int(len(replay_episode_specs))
                self._replay_fill_batches += 1
                self._replay_fill_episode_specs += int(len(replay_episode_specs))
        aggregated_metrics = {
            key: (sum(values) / float(len(values)) if values else 0.0)
            for key, values in rollout_metric_values.items()
        }
        return {
            "episode_specs": episode_specs,
            "rollout_metrics": aggregated_metrics,
            "budgeting_metrics": self.get_budget_drop_metrics(),
            "runtime_stats": runtime_stats,
            "video_ids": video_ids,
        }

    def _build_generation_step_payloads(
        self,
        generation_items: Sequence[Dict[str, Any]],
        rollout_model: Any,
    ) -> List[Dict[str, Any]]:
        item_payloads: List[Dict[str, Any]] = []
        all_episode_specs: List[Dict[str, Any]] = []
        with torch.inference_mode():
            for item in generation_items:
                payload = self._build_generation_item_payload(item, rollout_model)
                item_payloads.append(payload)
                all_episode_specs.extend(list(payload.get("episode_specs") or []))
                runtime_stats = dict(payload.get("runtime_stats") or {})
                self._groups_filtered_by_min_weight += int(runtime_stats.get("groups_filtered_by_min_weight", 0))
                self._groups_all_zero_advantage += int(runtime_stats.get("groups_all_zero_advantage", 0))
        if all_episode_specs:
            self._add_episode_specs_to_replay_buffer(all_episode_specs)
        step_payloads: List[Dict[str, Any]] = []
        step_size = max(1, int(self._generation_step_batch_size))
        for offset in range(0, len(item_payloads), step_size):
            step_payloads.append(
                self._aggregate_generation_step_payload(item_payloads[offset : offset + step_size])
            )
        return step_payloads

    def _pop_or_generate_generation_step_payload(
        self,
        generation_items: Sequence[Dict[str, Any]],
        wrapped_model: Any,
        rollout_model: Any,
        *,
        was_training: bool,
    ) -> Dict[str, Any]:
        batch_key = self._build_generation_batch_key(generation_items)
        buffered_ready = self._buffered_generation_step_payloads and self._buffered_generation_batch_key == batch_key
        if not buffered_ready:
            self._buffered_generation_step_payloads = []
            self._buffered_generation_batch_key = None
            if was_training and hasattr(wrapped_model, "eval"):
                wrapped_model.eval()
            try:
                self._buffered_generation_step_payloads = self._build_generation_step_payloads(
                    generation_items,
                    rollout_model,
                )
                if self._buffered_generation_step_payloads:
                    self._buffered_generation_batch_key = batch_key
            finally:
                if was_training and hasattr(wrapped_model, "train"):
                    wrapped_model.train()
        if self._buffered_generation_step_payloads:
            step_payload = self._buffered_generation_step_payloads.pop(0)
            if not self._buffered_generation_step_payloads:
                self._buffered_generation_batch_key = None
            return step_payload
        return self._empty_generation_step_payload(
            video_ids=[str(item.get("video_id") or "") for item in generation_items]
        )

    def _build_policy(self, model: Any, *, use_generation_cache: bool) -> QwenGenerationPolicy:
        return QwenGenerationPolicy.from_components(
            model=_unwrap_model(model),
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

    def _build_rollout_policy(self, model: Any) -> QwenGenerationPolicy:
        return self._build_policy(
            model,
            use_generation_cache=self.rollout_use_generation_cache,
        )

    def _build_fecv_policy(self, model: Any) -> QwenGenerationPolicy:
        return self._build_policy(
            model,
            use_generation_cache=self.fecv_use_generation_cache,
        )

    def _generate_scored_rollouts(
        self,
        item: Dict[str, Any],
        model: Any,
        *,
        progress: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        rollout_policy = self._build_rollout_policy(model)
        verification_policy = self._build_fecv_policy(model)
        rollouts: List[Dict[str, Any]] = []
        video_id = str(item.get("video_id") or "")
        rollout_items = [copy.deepcopy(item) for _ in range(self.num_generations)]
        batch_rollout_fn = getattr(self.rollout_runner, "run_episodes", None)
        if callable(batch_rollout_fn):
            generated_rollouts = list(
                batch_rollout_fn(
                    rollout_items,
                    rollout_policy,
                    capture_prompt_messages=True,
                )
            )
        else:
            generated_rollouts = [
                self.rollout_runner.run_episode(
                    rollout_item,
                    rollout_policy,
                    capture_prompt_messages=True,
                )
                for rollout_item in rollout_items
            ]
        for generation_id, rollout in enumerate(generated_rollouts):
            if progress is not None:
                progress.advance_generation_stage(
                    video_id=str(rollout.get("video_id") or video_id),
                    generation_id=int(generation_id),
                    stage="rollout",
                )
            rollout["group_id"] = str(item.get("video_id") or f"group_{generation_id}")
            rollout["generation_id"] = int(generation_id)
            if isinstance(item.get("structured_target"), dict):
                rollout["scoring_target"] = copy.deepcopy(item["structured_target"])
            if isinstance(item.get("qa_pairs"), list):
                rollout["scoring_qa_pairs"] = copy.deepcopy(item.get("qa_pairs") or [])
            evidence = item.get("evidence") or {}
            if isinstance(evidence, dict) and isinstance(evidence.get("evidence_moments"), list):
                rollout["scoring_evidence_moments"] = copy.deepcopy(evidence.get("evidence_moments") or [])
            try:
                rollout.update(
                    run_counterfactual_verification(
                        verification_policy,
                        item=item,
                        rollout=rollout,
                        reference_record=item,
                        max_images=self.counterfactual_max_images,
                        branch_profile="online_core",
                    )
                )
            except Exception as exc:
                if self.fecv_failure_policy == "fail":
                    raise
                self._fecv_failure_count += 1
                rollout["fecv_failed"] = True
                rollout["fecv_failure_message"] = _truncate_error_message(exc)
                rollout["fecv_failure_type"] = type(exc).__name__
                rollout["fecv_failure_policy"] = self.fecv_failure_policy
                if self.fecv_failure_policy == "drop":
                    rollout["drop_due_to_fecv_failure"] = True
            finally:
                if progress is not None:
                    progress.advance_generation_stage(
                        video_id=str(rollout.get("video_id") or video_id),
                        generation_id=int(generation_id),
                        stage="fecv",
                    )
            if bool(rollout.get("drop_due_to_fecv_failure")):
                continue
            try:
                rollout["reward_summary"] = score_rollout_trace(
                    rollout,
                    reward_version=self.reward_version,
                    reward_config=self.reward_config,
                    llm_judge=self.reward_judge,
                )
                if bool(rollout.get("fecv_failed")):
                    rollout["reward_summary"] = _degrade_reward_summary_for_fecv_failure(
                        rollout["reward_summary"],
                        error_message=str(rollout.get("fecv_failure_message") or ""),
                    )
                    self._fecv_degraded_rollout_count += 1
            finally:
                if progress is not None:
                    progress.advance_generation_stage(
                        video_id=str(rollout.get("video_id") or video_id),
                        generation_id=int(generation_id),
                        stage="score",
                    )
            rollouts.append(rollout)
        return _compute_group_relative_advantages(
            rollouts,
            clip_value=self.advantage_clip,
        )

    def _build_episode_spec(
        self,
        feature: Dict[str, Any],
    ) -> BatchBuildResult:
        return _build_rl_completion_episode_spec_from_feature(
            self.processor,
            feature,
            max_image_side=self.max_image_side,
            max_image_pixels=self.max_image_pixels,
            keep_recent_tool_image_messages=self.keep_recent_tool_image_messages,
            max_total_images=self.max_total_images,
            max_seq_length=self.max_seq_length,
            keep_recent_text_messages=self.keep_recent_text_messages,
        )

    def _reserved_episode_spec_keys(self) -> set[str]:
        return {
            "prompt_ids",
            "prompt_mask",
            "completion_ids",
            "completion_mask",
            "prompt_token_count",
            "completion_token_count",
            "sample_weight",
            "advantage",
            "old_policy_token_log_probs",
            "sample_loss_multiplier",
        }

    def _episode_spec_multimodal_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in batch.items()
            if key not in self._reserved_episode_spec_keys()
        }

    def _move_episode_spec_to_device(
        self,
        episode_spec: Dict[str, Any],
        *,
        device: torch.device,
    ) -> Dict[str, Any]:
        prepared: Dict[str, Any] = {}
        for key, value in episode_spec.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(device=device)
            else:
                prepared[key] = copy.deepcopy(value)
        return prepared

    def _episode_spec_sample_count(self, episode_spec: Dict[str, Any]) -> int:
        for key in ("completion_ids", "prompt_ids", "completion_mask", "sample_weight", "advantage"):
            value = episode_spec.get(key)
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return max(1, int(value.shape[0]))
        return 1

    def _clone_episode_spec_as_noop(self, episode_spec_entry: Dict[str, Any]) -> Dict[str, Any]:
        cloned = copy.deepcopy(dict(episode_spec_entry or {}))
        prepared_batch = dict(cloned.get("episode_spec") or {})
        sample_count = self._episode_spec_sample_count(prepared_batch)
        prepared_batch["sample_loss_multiplier"] = torch.zeros(sample_count, dtype=torch.float32)
        if isinstance(prepared_batch.get("sample_weight"), torch.Tensor):
            prepared_batch["sample_weight"] = torch.zeros_like(
                prepared_batch["sample_weight"],
                dtype=torch.float32,
            )
        if isinstance(prepared_batch.get("advantage"), torch.Tensor):
            prepared_batch["advantage"] = torch.zeros_like(
                prepared_batch["advantage"],
                dtype=torch.float32,
            )
        cloned["episode_spec"] = prepared_batch
        feature = dict(cloned.get("feature") or {})
        feature["_ddp_noop_padding"] = True
        cloned["feature"] = feature
        return cloned

    def _prepared_batch_sample_count(self, prepared_batch: Dict[str, Any]) -> int:
        for key in ("completion_ids", "prompt_ids", "completion_mask", "sample_weight", "advantage"):
            value = prepared_batch.get(key)
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return max(1, int(value.shape[0]))
        return 1

    def _clone_prepared_batch_as_noop(self, prepared_batch: Dict[str, Any]) -> Dict[str, Any]:
        cloned = copy.deepcopy(dict(prepared_batch or {}))
        sample_count = self._prepared_batch_sample_count(cloned)
        cloned["sample_loss_multiplier"] = torch.zeros(sample_count, dtype=torch.float32, device=next(
            (
                value.device
                for value in cloned.values()
                if isinstance(value, torch.Tensor)
            ),
            torch.device("cpu"),
        ))
        if isinstance(cloned.get("sample_weight"), torch.Tensor):
            cloned["sample_weight"] = torch.zeros_like(cloned["sample_weight"], dtype=torch.float32)
        if isinstance(cloned.get("advantage"), torch.Tensor):
            cloned["advantage"] = torch.zeros_like(cloned["advantage"], dtype=torch.float32)
        return cloned

    def _prepared_batch_cpu_copy(self, prepared_batch: Dict[str, Any]) -> Dict[str, Any]:
        return self._move_episode_spec_to_device(prepared_batch, device=torch.device("cpu"))

    def _sample_loss_multiplier(
        self,
        batch: Dict[str, Any],
        *,
        device: torch.device,
        sample_count: int,
    ) -> torch.Tensor:
        multiplier = batch.get("sample_loss_multiplier")
        if multiplier is None:
            return torch.ones(sample_count, dtype=torch.float32, device=device)
        if not isinstance(multiplier, torch.Tensor):
            multiplier = torch.tensor(multiplier, dtype=torch.float32, device=device)
        multiplier = multiplier.to(device=device, dtype=torch.float32).view(-1)
        if multiplier.numel() == 1 and int(sample_count) != 1:
            multiplier = multiplier.expand(int(sample_count))
        if int(multiplier.numel()) != int(sample_count):
            raise ValueError(
                "sample_loss_multiplier must align with the prepared batch sample count: "
                f"{tuple(multiplier.shape)} vs {int(sample_count)}"
            )
        return multiplier

    def _cache_old_policy_log_probs(
        self,
        *,
        feature: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        result = self._build_episode_spec(feature)
        self._budgeting_stats.record(result)
        if result.batch is None:
            return None
        return {
            "feature": copy.deepcopy(feature),
            "episode_spec": result.batch,
        }

    def _sequence_pad_values(self) -> Dict[str, Tuple[Any, str]]:
        return {
            "prompt_ids": (self._pad_token_id(), "left"),
            "prompt_mask": (0, "left"),
            "completion_ids": (self._pad_token_id(), "right"),
            "completion_mask": (0, "right"),
            "old_policy_token_log_probs": (0.0, "right"),
        }

    def _prepared_batch_merge_signature_entry(
        self,
        key: str,
        value: Any,
        prepared_batch: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Expected tensor values for prepared batch key {key!r}.")
        if key in self._sequence_pad_values():
            return ("sequence_pad", str(value.dtype), tuple(value.shape[1:-1]))
        if value.ndim == 0:
            return ("stack_scalar", str(value.dtype))
        if self._is_full_sequence_aligned_tensor(value, prepared_batch):
            return ("full_sequence_aligned", str(value.dtype), tuple(value.shape[1:-1]))
        return ("concat", str(value.dtype), tuple(value.shape[1:]))

    def _prepared_batch_merge_signature(
        self,
        prepared_batch: Dict[str, Any],
    ) -> Tuple[Tuple[str, Tuple[Any, ...]], ...]:
        return tuple(
            (
                str(key),
                self._prepared_batch_merge_signature_entry(str(key), prepared_batch[key], prepared_batch),
            )
            for key in sorted(prepared_batch.keys())
        )

    def _group_items_by_signature(
        self,
        items: Sequence[Any],
        *,
        signature_fn: Any,
    ) -> List[List[Any]]:
        grouped: Dict[Any, List[Any]] = {}
        ordered_groups: List[List[Any]] = []
        for item in items:
            signature = signature_fn(item)
            bucket = grouped.get(signature)
            if bucket is None:
                bucket = []
                grouped[signature] = bucket
                ordered_groups.append(bucket)
            bucket.append(item)
        return ordered_groups

    def _is_merge_fallback_error(self, exc: Exception) -> bool:
        error_message = str(exc)
        return (
            "Cannot merge prepared batches with inconsistent key presence" in error_message
            or "Unable to concatenate prepared batch key" in error_message
        )

    def _compute_old_policy_token_log_probs_for_prepared_batch(
        self,
        model: Any,
        *,
        prepared_batch: Dict[str, Any],
    ) -> torch.Tensor:
        with torch.inference_mode():
            old_policy_token_log_probs, _ = compute_completion_only_token_log_probs_from_ids(
                model=model,
                prompt_ids=prepared_batch["prompt_ids"],
                prompt_mask=prepared_batch["prompt_mask"],
                completion_ids=prepared_batch["completion_ids"],
                completion_mask=prepared_batch["completion_mask"],
                multimodal_inputs=self._episode_spec_multimodal_inputs(prepared_batch),
                temperature=self.policy_temperature,
            )
        return old_policy_token_log_probs.detach().cpu()

    def _populate_old_policy_log_probs(
        self,
        model: Any,
        episode_specs: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not episode_specs:
            return []
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")
        grouped_episode_specs = self._group_items_by_signature(
            list(episode_specs),
            signature_fn=lambda item: self._prepared_batch_merge_signature(item["episode_spec"]),
        )
        for bucket in grouped_episode_specs:
            prepared_batches = [
                self._move_episode_spec_to_device(entry["episode_spec"], device=target_device)
                for entry in bucket
            ]
            if len(prepared_batches) == 1:
                batched_log_probs = self._compute_old_policy_token_log_probs_for_prepared_batch(
                    model,
                    prepared_batch=prepared_batches[0],
                )
            else:
                try:
                    merged_batch = self._merge_prepared_batches(prepared_batches)
                except ValueError as exc:
                    if not self._is_merge_fallback_error(exc):
                        raise
                    for entry, prepared_batch in zip(bucket, prepared_batches):
                        entry["old_policy_token_log_probs"] = self._compute_old_policy_token_log_probs_for_prepared_batch(
                            model,
                            prepared_batch=prepared_batch,
                        )
                    continue
                batched_log_probs = self._compute_old_policy_token_log_probs_for_prepared_batch(
                    model,
                    prepared_batch=merged_batch,
                )
            for row_index, entry in enumerate(bucket):
                completion_length = int(entry["episode_spec"]["completion_ids"].shape[-1])
                entry["old_policy_token_log_probs"] = (
                    batched_log_probs[row_index : row_index + 1, :completion_length].clone()
                )
        return list(episode_specs)

    def _materialize_episode_spec(
        self,
        episode_spec: Dict[str, Any],
        *,
        device: torch.device,
    ) -> Dict[str, Any]:
        cached_episode_spec = episode_spec.get("episode_spec")
        if not isinstance(cached_episode_spec, dict):
            feature = dict(episode_spec.get("feature") or {})
            result = self._build_episode_spec(feature)
            if result.batch is None:
                raise _BudgetDropError(
                    f"Episode spec materialized to zero-response batch: {result.drop_reason or 'unknown'}"
                )
            cached_episode_spec = result.batch
        prepared_batch = self._move_episode_spec_to_device(cached_episode_spec, device=device)
        old_policy_token_log_probs = episode_spec.get("old_policy_token_log_probs")
        if old_policy_token_log_probs is None:
            legacy_old_policy_log_probs = episode_spec.get("old_policy_log_probs")
            if legacy_old_policy_log_probs is not None:
                raise ValueError(
                    "trainer-native GRPO no longer accepts legacy scalar old_policy_log_probs; "
                    "cache per-token old_policy_token_log_probs during rollout preparation instead."
                )
        if isinstance(old_policy_token_log_probs, torch.Tensor):
            old_policy_token_log_probs = old_policy_token_log_probs.to(device=device, dtype=torch.float32)
            if old_policy_token_log_probs.ndim == 1:
                old_policy_token_log_probs = old_policy_token_log_probs.view(1, -1)
            if tuple(old_policy_token_log_probs.shape) != tuple(prepared_batch["completion_ids"].shape):
                raise ValueError(
                    "old_policy_token_log_probs must align with completion_ids shape: "
                    f"{tuple(old_policy_token_log_probs.shape)} vs {tuple(prepared_batch['completion_ids'].shape)}"
                )
            prepared_batch["old_policy_token_log_probs"] = old_policy_token_log_probs
        return prepared_batch

    def _pad_token_id(self) -> int:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        for attr_name in ("pad_token_id", "eos_token_id"):
            value = getattr(tokenizer, attr_name, None)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    continue
        return 0

    def _pad_and_concat(
        self,
        tensors: Sequence[torch.Tensor],
        *,
        pad_value: Any,
        pad_side: str,
    ) -> torch.Tensor:
        if not tensors:
            raise ValueError("Cannot merge an empty tensor list.")
        max_seq_len = max(int(tensor.shape[-1]) for tensor in tensors)
        padded_tensors: List[torch.Tensor] = []
        for tensor in tensors:
            pad_width = max_seq_len - int(tensor.shape[-1])
            if pad_width <= 0:
                padded_tensors.append(tensor)
                continue
            pad_shape = list(tensor.shape)
            pad_shape[-1] = int(pad_width)
            pad_tensor = torch.full(
                pad_shape,
                pad_value,
                dtype=tensor.dtype,
                device=tensor.device,
            )
            if str(pad_side) == "right":
                padded_tensors.append(torch.cat([tensor, pad_tensor], dim=-1))
            else:
                padded_tensors.append(torch.cat([pad_tensor, tensor], dim=-1))
        return torch.cat(padded_tensors, dim=0)

    def _is_full_sequence_aligned_tensor(
        self,
        tensor: torch.Tensor,
        prepared_batch: Dict[str, Any],
    ) -> bool:
        prompt_ids = prepared_batch.get("prompt_ids")
        completion_ids = prepared_batch.get("completion_ids")
        if not isinstance(prompt_ids, torch.Tensor) or not isinstance(completion_ids, torch.Tensor):
            return False
        if tensor.ndim < 2 or prompt_ids.ndim != 2 or completion_ids.ndim != 2:
            return False
        batch_size = int(prompt_ids.shape[0])
        if int(completion_ids.shape[0]) != batch_size or int(tensor.shape[0]) != batch_size:
            return False
        expected_full_width = int(prompt_ids.shape[-1]) + int(completion_ids.shape[-1])
        return int(tensor.shape[-1]) == expected_full_width

    def _pad_full_sequence_aligned_and_concat(
        self,
        tensors: Sequence[torch.Tensor],
        prepared_batches: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        if not tensors or len(tensors) != len(prepared_batches):
            raise ValueError("Full-sequence tensor padding requires aligned tensor/batch lists.")
        prompt_lengths = [int(batch["prompt_ids"].shape[-1]) for batch in prepared_batches]
        completion_lengths = [int(batch["completion_ids"].shape[-1]) for batch in prepared_batches]
        max_prompt_length = max(prompt_lengths)
        max_completion_length = max(completion_lengths)
        padded_tensors: List[torch.Tensor] = []
        for tensor, prompt_length, completion_length in zip(tensors, prompt_lengths, completion_lengths):
            left_pad = max_prompt_length - int(prompt_length)
            right_pad = max_completion_length - int(completion_length)
            if left_pad < 0 or right_pad < 0:
                raise ValueError("Full-sequence padding widths must be non-negative.")
            if left_pad <= 0 and right_pad <= 0:
                padded_tensors.append(tensor)
                continue
            if left_pad > 0:
                left_shape = list(tensor.shape)
                left_shape[-1] = int(left_pad)
                left_tensor = torch.zeros(left_shape, dtype=tensor.dtype, device=tensor.device)
                tensor = torch.cat([left_tensor, tensor], dim=-1)
            if right_pad > 0:
                right_shape = list(tensor.shape)
                right_shape[-1] = int(right_pad)
                right_tensor = torch.zeros(right_shape, dtype=tensor.dtype, device=tensor.device)
                tensor = torch.cat([tensor, right_tensor], dim=-1)
            padded_tensors.append(tensor)
        return torch.cat(padded_tensors, dim=0)

    def _merge_prepared_batches(self, prepared_batches: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not prepared_batches:
            raise ValueError("Cannot merge zero prepared batches.")
        merged: Dict[str, Any] = {}
        sequence_pad_values = self._sequence_pad_values()
        ordered_keys: List[str] = []
        for batch in prepared_batches:
            for key in batch.keys():
                if key not in ordered_keys:
                    ordered_keys.append(key)
        for key in ordered_keys:
            values = [batch[key] for batch in prepared_batches if key in batch]
            if len(values) != len(prepared_batches):
                raise ValueError(f"Cannot merge prepared batches with inconsistent key presence for {key!r}.")
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise ValueError(f"Expected tensor values for merged prepared batch key {key!r}.")
            if key in sequence_pad_values:
                pad_value, pad_side = sequence_pad_values[key]
                merged[key] = self._pad_and_concat(values, pad_value=pad_value, pad_side=pad_side)
                continue
            if values[0].ndim == 0:
                merged[key] = torch.stack(values, dim=0)
                continue
            if all(
                self._is_full_sequence_aligned_tensor(value, batch)
                for value, batch in zip(values, prepared_batches)
            ):
                merged[key] = self._pad_full_sequence_aligned_and_concat(values, prepared_batches)
                continue
            try:
                merged[key] = torch.cat(values, dim=0)
            except Exception as exc:
                raise ValueError(f"Unable to concatenate prepared batch key {key!r}.") from exc
        return merged

    def _materialize_episode_spec_microbatch(
        self,
        episode_specs: Sequence[Dict[str, Any]],
        *,
        device: torch.device,
    ) -> List[Dict[str, Any]]:
        prepared_batches: List[Dict[str, Any]] = []
        for episode_spec in episode_specs:
            try:
                prepared_batch = self._materialize_episode_spec(episode_spec, device=device)
            except _BudgetDropError:
                self._zero_response_dropped += 1
                continue
            completion_mask = prepared_batch.get("completion_mask")
            if completion_mask is None or not bool(torch.any(completion_mask.to(dtype=torch.bool))):
                self._zero_response_dropped += 1
                continue
            prepared_batches.append(prepared_batch)
        if not prepared_batches:
            return []
        merged_batches: List[Dict[str, Any]] = []
        grouped_prepared_batches = self._group_items_by_signature(
            prepared_batches,
            signature_fn=self._prepared_batch_merge_signature,
        )
        for bucket in grouped_prepared_batches:
            if len(bucket) == 1:
                merged_batches.append(bucket[0])
                continue
            try:
                merged_batches.append(self._merge_prepared_batches(bucket))
            except ValueError as exc:
                if not self._is_merge_fallback_error(exc):
                    raise
                self._materialize_fallback_batches += 1
                merged_batches.extend(bucket)
        return merged_batches

    def _prepare_advantages(self, batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
        advantages = batch.get("advantage")
        if advantages is None:
            advantages = batch.get("sample_weight")
        if advantages is None:
            raise ValueError("Trainer-native GRPO requires rollout advantages for every episode batch.")
        return advantages.to(device=device, dtype=torch.float32).view(-1)

    def _compute_sample_losses_for_batch(
        self,
        *,
        model: Any,
        batch: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        response_mask = batch.get("completion_mask")
        if response_mask is None:
            return None
        response_mask = response_mask.to(dtype=torch.bool)
        if not bool(torch.any(response_mask)):
            return None
        policy_token_log_probs, response_mask = compute_completion_only_token_log_probs_from_ids(
            model=model,
            prompt_ids=batch["prompt_ids"],
            prompt_mask=batch["prompt_mask"],
            completion_ids=batch["completion_ids"],
            completion_mask=batch["completion_mask"],
            multimodal_inputs=self._episode_spec_multimodal_inputs(batch),
            temperature=self.policy_temperature,
        )
        if not bool(policy_token_log_probs.requires_grad):
            raise RuntimeError("Policy completion log-probs are detached in the completion-native GRPO path.")
        if tuple(policy_token_log_probs.shape) != tuple(batch["completion_ids"].shape):
            raise ValueError(
                "policy_token_log_probs must align with completion_ids shape: "
                f"{tuple(policy_token_log_probs.shape)} vs {tuple(batch['completion_ids'].shape)}"
            )
        old_policy_token_log_probs = batch.get("old_policy_token_log_probs")
        if old_policy_token_log_probs is None:
            old_policy_token_log_probs = policy_token_log_probs.detach()
        else:
            old_policy_token_log_probs = old_policy_token_log_probs.to(
                policy_token_log_probs.device,
                dtype=torch.float32,
            )
            if old_policy_token_log_probs.ndim == 1:
                old_policy_token_log_probs = old_policy_token_log_probs.view(1, -1)
        if tuple(old_policy_token_log_probs.shape) != tuple(policy_token_log_probs.shape):
            raise ValueError(
                "old_policy_token_log_probs must align with policy_token_log_probs shape: "
                f"{tuple(old_policy_token_log_probs.shape)} vs {tuple(policy_token_log_probs.shape)}"
            )
        advantages = self._prepare_advantages(batch, policy_token_log_probs.device)
        coef_1 = torch.exp(policy_token_log_probs - old_policy_token_log_probs.detach())
        coef_2 = torch.clamp(
            coef_1,
            1.0 - float(self.ppo_clip_epsilon),
            1.0 + float(self.ppo_clip_epsilon),
        )
        per_token_loss_1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss_2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.minimum(per_token_loss_1, per_token_loss_2)
        if self.kl_beta > 0.0:
            reference_token_log_probs = None
            if self.reference_model is not None:
                self._ensure_reference_model_device(model)
                with torch.inference_mode():
                    reference_token_log_probs, _ = compute_completion_only_token_log_probs_from_ids(
                        model=self.reference_model,
                        prompt_ids=batch["prompt_ids"],
                        prompt_mask=batch["prompt_mask"],
                        completion_ids=batch["completion_ids"],
                        completion_mask=batch["completion_mask"],
                        multimodal_inputs=self._episode_spec_multimodal_inputs(batch),
                        temperature=self.policy_temperature,
                    )
            elif self.use_lora_reference_disable_adapter:
                disable_context, reference_model = self._disable_adapter_context(model)
                with torch.inference_mode():
                    with disable_context:
                        reference_token_log_probs, _ = compute_completion_only_token_log_probs_from_ids(
                            model=reference_model,
                            prompt_ids=batch["prompt_ids"],
                            prompt_mask=batch["prompt_mask"],
                            completion_ids=batch["completion_ids"],
                            completion_mask=batch["completion_mask"],
                            multimodal_inputs=self._episode_spec_multimodal_inputs(batch),
                            temperature=self.policy_temperature,
                        )
            if reference_token_log_probs is not None:
                if tuple(reference_token_log_probs.shape) != tuple(policy_token_log_probs.shape):
                    raise ValueError(
                        "reference_token_log_probs must align with policy_token_log_probs shape: "
                        f"{tuple(reference_token_log_probs.shape)} vs {tuple(policy_token_log_probs.shape)}"
                    )
                delta = reference_token_log_probs.to(policy_token_log_probs.device) - policy_token_log_probs
                per_token_kl = torch.exp(delta) - delta - 1.0
                per_token_loss = per_token_loss + per_token_loss.new_tensor(self.kl_beta) * per_token_kl
        response_mask_f = response_mask.to(dtype=per_token_loss.dtype)
        token_counts = response_mask_f.sum(dim=-1).clamp(min=1.0)
        sample_losses = (per_token_loss * response_mask_f).sum(dim=-1) / token_counts
        sample_losses = sample_losses * self._sample_loss_multiplier(
            batch,
            device=sample_losses.device,
            sample_count=int(sample_losses.shape[0]),
        )
        return sample_losses

    def _disable_adapter_context(self, model: Any):
        if not self.use_lora_reference_disable_adapter:
            return nullcontext(), None
        unwrapped_model = _unwrap_model(model)
        disable_adapter = getattr(unwrapped_model, "disable_adapter", None)
        if not callable(disable_adapter):
            raise RuntimeError(
                "LoRA KL reference requested, but the current policy model does not expose disable_adapter()."
            )
        return disable_adapter(), unwrapped_model

    def get_budget_drop_metrics(self) -> Dict[str, Any]:
        metrics = self._budgeting_stats.as_dict()
        metrics.update(
            {
                "rl_zero_response_dropped": int(self._zero_response_dropped),
                "rl_materialize_fallback_batches": int(self._materialize_fallback_batches),
                "rl_completion_only_grad_fallback_batches": int(self._completion_only_grad_fallback_batches),
                "rl_ddp_global_empty_batch_skips": int(self._ddp_global_empty_batch_skips),
                "rl_all_empty_batch_skips": int(self._all_empty_batch_skips),
                "rl_effective_update_steps": int(self._effective_update_steps),
                "rl_optimizer_step_skips": int(self._optimizer_step_skips),
                "rl_replay_fill_batches": int(self._replay_fill_batches),
                "rl_replay_fill_episode_specs": int(self._replay_fill_episode_specs),
                "rl_groups_all_zero_advantage": int(self._groups_all_zero_advantage),
                "rl_groups_filtered_by_min_weight": int(self._groups_filtered_by_min_weight),
                "rl_fecv_failure_count": int(self._fecv_failure_count),
                "rl_fecv_degraded_rollout_count": int(self._fecv_degraded_rollout_count),
                "rl_compute_loss_microbatch_size_effective": int(self.compute_loss_microbatch_size),
            }
        )
        return metrics

    def get_budgeting_stats(self) -> BudgetingStats:
        stats = BudgetingStats()
        stats.merge(self._budgeting_stats)
        return stats

    def _prepare_inputs(self, inputs):
        if not isinstance(inputs, list):
            return super()._prepare_inputs(inputs)
        wrapped_model = self.model
        rollout_model = _unwrap_model(wrapped_model)
        progress = getattr(self, "_native_grpo_progress", None)
        was_training = bool(getattr(wrapped_model, "training", getattr(rollout_model, "training", False)))
        progress_video_ids: List[str] = []
        if progress is not None:
            progress.start_batch(
                num_items=min(
                    max(1, int(self._generation_step_batch_size)),
                    max(1, int(len(inputs))),
                )
            )
        try:
            if was_training:
                previous_generation_progress = getattr(self, "_active_generation_progress", None)
                self._active_generation_progress = progress
                try:
                    prepared = self._pop_or_generate_generation_step_payload(
                        inputs,
                        wrapped_model,
                        rollout_model,
                        was_training=was_training,
                    )
                finally:
                    self._active_generation_progress = previous_generation_progress
            else:
                if hasattr(wrapped_model, "eval"):
                    wrapped_model.eval()
                try:
                    prepared_batches = self._build_generation_step_payloads(inputs, rollout_model)
                finally:
                    if was_training and hasattr(wrapped_model, "train"):
                        wrapped_model.train()
                prepared = (
                    prepared_batches[0]
                    if prepared_batches
                    else self._empty_generation_step_payload(
                        video_ids=[str(item.get("video_id") or "") for item in inputs]
                    )
                )
            progress_video_ids = [str(video_id or "") for video_id in (prepared.get("video_ids") or [])]
            for video_id in progress_video_ids:
                if progress is not None:
                    progress.finish_item(video_id=video_id)
            return {
                "episode_specs": list(prepared.get("episode_specs") or []),
                "rollout_metrics": dict(prepared.get("rollout_metrics") or {}),
                "budgeting_metrics": dict(prepared.get("budgeting_metrics") or {}),
                "runtime_stats": dict(prepared.get("runtime_stats") or {}),
            }
        finally:
            if progress is not None:
                progress.close_batch()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        if return_outputs:
            raise ValueError("Trainer-native GRPO does not support returning model outputs.")
        episode_specs = list(inputs.get("episode_specs") or [])
        runtime_stats = dict(inputs.get("runtime_stats") or {})
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")
        local_episode_spec_count = int(len(episode_specs))
        global_episode_spec_count = _distributed_sum_int(local_episode_spec_count, device=target_device)
        all_ranks_have_episode_specs, any_rank_has_episode_specs = _distributed_bool_consensus(
            local_episode_spec_count > 0,
            device=target_device,
        )
        if any_rank_has_episode_specs and not all_ranks_have_episode_specs:
            donor_episode_spec = _distributed_first_available_object(
                episode_specs[0] if episode_specs else None,
                device=target_device,
            )
            if local_episode_spec_count <= 0 and donor_episode_spec is not None:
                episode_specs = [self._clone_episode_spec_as_noop(donor_episode_spec)]
                local_episode_spec_count = int(len(episode_specs))
                runtime_stats["ddp_noop_padded_episode_specs"] = int(local_episode_spec_count)
        if not any_rank_has_episode_specs or global_episode_spec_count <= 0:
            if self.all_empty_policy == "true_skip":
                self._mark_skip_next_optimizer_step(reason="all_empty_episode_specs", all_empty=True)
            self._maybe_log_empty_batch_rank_summary(
                reason="all_empty_episode_specs",
                runtime_stats=runtime_stats,
                trainable_samples=0,
            )
            return _zero_loss_from_model(model)

        total_loss_sum = None
        total_samples = 0
        microbatch_size = max(1, int(self.compute_loss_microbatch_size))
        prepared_microbatches: List[Dict[str, Any]] = []
        for start_index in range(0, len(episode_specs), microbatch_size):
            chunk = episode_specs[start_index : start_index + microbatch_size]
            prepared_microbatches.extend(
                self._materialize_episode_spec_microbatch(chunk, device=target_device)
            )
        runtime_stats["raw_local_prepared_batch_count"] = int(len(prepared_microbatches))
        local_prepared_batch_count = int(len(prepared_microbatches))
        global_prepared_batch_count = _distributed_sum_int(local_prepared_batch_count, device=target_device)
        all_ranks_have_prepared_batches, any_rank_has_prepared_batches = _distributed_bool_consensus(
            local_prepared_batch_count > 0,
            device=target_device,
        )
        if any_rank_has_prepared_batches and not all_ranks_have_prepared_batches:
            donor_prepared_batch = _distributed_first_available_object(
                self._prepared_batch_cpu_copy(prepared_microbatches[0]) if prepared_microbatches else None,
                device=target_device,
            )
            if local_prepared_batch_count <= 0 and donor_prepared_batch is not None:
                prepared_microbatches = [
                    self._clone_prepared_batch_as_noop(
                        self._move_episode_spec_to_device(donor_prepared_batch, device=target_device)
                    )
                ]
                local_prepared_batch_count = int(len(prepared_microbatches))
                runtime_stats["ddp_noop_padded_prepared_batches"] = int(local_prepared_batch_count)
        if not any_rank_has_prepared_batches or global_prepared_batch_count <= 0:
            if self.all_empty_policy == "true_skip":
                self._mark_skip_next_optimizer_step(reason="all_empty_prepared_batches", all_empty=True)
            self._maybe_log_empty_batch_rank_summary(
                reason="all_empty_prepared_batches",
                runtime_stats=runtime_stats,
                trainable_samples=0,
            )
            return _zero_loss_from_model(model)

        for batch in prepared_microbatches:
            sample_losses = self._compute_sample_losses_for_batch(
                model=model,
                batch=batch,
            )
            if sample_losses is None or sample_losses.numel() <= 0:
                continue
            sample_loss_multiplier = self._sample_loss_multiplier(
                batch,
                device=sample_losses.device,
                sample_count=int(sample_losses.numel()),
            )
            total_samples += int((sample_loss_multiplier > 0).sum().item())
            batch_loss_sum = sample_losses.sum()
            total_loss_sum = batch_loss_sum if total_loss_sum is None else total_loss_sum + batch_loss_sum
        runtime_stats["raw_local_sample_count"] = int(total_samples)
        global_total_samples = _distributed_sum_int(int(total_samples), device=target_device)
        if total_loss_sum is None or global_total_samples <= 0:
            if self.all_empty_policy == "true_skip":
                self._mark_skip_next_optimizer_step(reason="all_empty_trainable_samples", all_empty=True)
            self._maybe_log_empty_batch_rank_summary(
                reason="all_empty_trainable_samples",
                runtime_stats=runtime_stats,
                trainable_samples=int(total_samples),
            )
            return _zero_loss_from_model(model)
        world_size = max(1, int(_distributed_world_size()))
        return total_loss_sum * float(world_size) / float(max(1, int(global_total_samples)))


def create_native_grpo_trainer(
    *,
    model: Any,
    processor: Any,
    train_dataset: torch.utils.data.Dataset,
    output_dir: str | Path,
    learning_rate: float,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    logging_steps: int,
    save_steps: int,
    save_total_limit: int,
    warmup_ratio: float,
    weight_decay: float,
    max_grad_norm: float,
    bf16: bool,
    fp16: bool,
    dataloader_num_workers: int,
    dataloader_prefetch_factor: int,
    dataloader_persistent_workers: bool,
    old_policy_model: Any,
    reference_model: Any,
    kl_beta: float,
    ppo_clip_epsilon: float,
    rollout_runner: SaverRolloutRunner,
    num_generations: int,
    min_weight: float,
    advantage_clip: float,
    policy_max_new_tokens: int,
    max_image_side: int,
    max_image_pixels: int,
    max_total_images: int,
    keep_recent_tool_image_messages: int = 0,
    keep_recent_text_messages: int = 0,
    max_seq_length: int = 0,
    counterfactual_max_images: int,
    policy_do_sample: bool,
    policy_temperature: Optional[float],
    policy_top_p: Optional[float],
    policy_top_k: Optional[int],
    policy_repetition_penalty: Optional[float],
    rollout_use_generation_cache: bool = True,
    fecv_use_generation_cache: bool = True,
    compute_loss_microbatch_size: int = 2,
    use_lora_reference_disable_adapter: bool = False,
    iteration_index: int = 0,
    num_iterations: int = 1,
    rollout_eval_callback: Any = None,
    replay_buffer_enable: bool = True,
    replay_buffer_type: str = "ssr",
    replay_buffer_capacity: int = 16,
    replay_buffer_alpha: float = 1.0,
    fecv_failure_policy: str = "degrade",
    all_empty_policy: str = "true_skip",
    log_empty_batch_rank_summary: bool = True,
    reward_version: str = DEFAULT_RL_REWARD_VERSION,
    reward_config: Optional[Dict[str, Any]] = None,
    steps_per_generation: int = 1,
    ddp_find_unused_parameters: bool = False,
    trainer_class_transform: Optional[Any] = None,
):
    try:
        from transformers import Trainer, TrainingArguments
    except Exception as exc:
        raise ImportError("Trainer-native GRPO requires the `transformers` package.") from exc

    effective_persistent_workers = bool(dataloader_persistent_workers) and int(dataloader_num_workers) > 0
    if effective_persistent_workers and float(num_train_epochs) <= 1.0:
        runtime_log(
            (
                "trainer-native RL disabled dataloader_persistent_workers because "
                "num_train_epochs<=1 makes worker persistence across epochs ineffective and unstable "
                "for iteration-scoped Trainer recreation."
            ),
            runtime=distributed_runtime_from_env(),
            main_process_only=True,
        )
        effective_persistent_workers = False
    runtime_log(
        (
            "trainer-native RL ddp_find_unused_parameters="
            f"{bool(ddp_find_unused_parameters)}"
        ),
        runtime=distributed_runtime_from_env(),
        main_process_only=True,
    )
    training_args_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": float(learning_rate),
        "num_train_epochs": float(num_train_epochs),
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "logging_steps": int(logging_steps),
        "save_steps": int(save_steps),
        "save_total_limit": int(save_total_limit),
        "warmup_ratio": float(warmup_ratio),
        "weight_decay": float(weight_decay),
        "max_grad_norm": float(max_grad_norm),
        "bf16": bool(bf16),
        "fp16": bool(fp16),
        "remove_unused_columns": False,
        "report_to": [],
        "disable_tqdm": True,
        "save_strategy": "no",
        "dataloader_num_workers": max(0, int(dataloader_num_workers)),
        "dataloader_persistent_workers": bool(effective_persistent_workers),
        "ddp_find_unused_parameters": bool(ddp_find_unused_parameters),
    }
    if int(dataloader_num_workers) > 0 and int(dataloader_prefetch_factor) > 0:
        training_args_kwargs["dataloader_prefetch_factor"] = int(dataloader_prefetch_factor)
    training_args = TrainingArguments(**training_args_kwargs)
    native_grpo_config = {
        "processor": processor,
        "train_dataset": train_dataset,
        "old_policy_model": old_policy_model,
        "reference_model": reference_model,
        "use_lora_reference_disable_adapter": bool(use_lora_reference_disable_adapter),
        "kl_beta": float(kl_beta),
        "ppo_clip_epsilon": float(ppo_clip_epsilon),
        "rollout_runner": rollout_runner,
        "num_generations": int(num_generations),
        "min_weight": float(min_weight),
        "advantage_clip": float(advantage_clip),
        "policy_max_new_tokens": int(policy_max_new_tokens),
        "max_image_side": int(max_image_side),
        "max_image_pixels": int(max_image_pixels),
        "max_total_images": int(max_total_images),
        "keep_recent_tool_image_messages": int(keep_recent_tool_image_messages),
        "keep_recent_text_messages": int(keep_recent_text_messages),
        "max_seq_length": int(max_seq_length),
        "counterfactual_max_images": int(counterfactual_max_images),
        "policy_do_sample": bool(policy_do_sample),
        "policy_temperature": policy_temperature,
        "policy_top_p": policy_top_p,
        "policy_top_k": policy_top_k,
        "policy_repetition_penalty": policy_repetition_penalty,
        "rollout_use_generation_cache": bool(rollout_use_generation_cache),
        "fecv_use_generation_cache": bool(fecv_use_generation_cache),
        "compute_loss_microbatch_size": int(compute_loss_microbatch_size),
        "steps_per_generation": int(steps_per_generation),
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "replay_buffer_enable": bool(replay_buffer_enable),
        "replay_buffer_type": str(replay_buffer_type),
        "replay_buffer_capacity": int(replay_buffer_capacity),
        "replay_buffer_alpha": float(replay_buffer_alpha),
        "fecv_failure_policy": str(fecv_failure_policy),
        "all_empty_policy": str(all_empty_policy),
        "log_empty_batch_rank_summary": bool(log_empty_batch_rank_summary),
        "reward_version": str(reward_version),
        "reward_config": dict(reward_config or {}),
        "iteration_index": int(iteration_index),
        "num_iterations": int(num_iterations),
    }
    native_grpo_base_trainer_class = _build_native_grpo_trainer_class(Trainer)
    trainer_class = native_grpo_base_trainer_class
    if callable(trainer_class_transform):
        transformed_trainer_class = trainer_class_transform(native_grpo_base_trainer_class)
        if not isinstance(transformed_trainer_class, type) or not issubclass(
            transformed_trainer_class,
            native_grpo_base_trainer_class,
        ):
            raise TypeError(
                "trainer_class_transform must return a Trainer subclass derived from the native GRPO trainer base class."
            )
        trainer_class = transformed_trainer_class

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=_raw_item_collator,
        callbacks=[],
        native_grpo_config=native_grpo_config,
    )
    if rollout_eval_callback is not None:
        trainer.add_callback(rollout_eval_callback)
    trainer.add_callback(_build_native_grpo_progress_callback(trainer=trainer))
    return trainer


def run_trainer_native_grpo(
    *,
    args: Any,
    runtime: Any,
    log_dir: str | Path = "",
    config_builder: Any,
    eval_config_builder: Any,
    reference_model_resolver: Any,
    select_iteration_indices_fn: Any,
) -> Dict[str, Any]:
    runtime = runtime or distributed_runtime_from_env()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_eval_output_root = (
        Path(str(args.rollout_eval_output_dir).strip())
        if str(getattr(args, "rollout_eval_output_dir", "") or "").strip()
        else output_dir
    )
    rollout_eval_output_root.mkdir(parents=True, exist_ok=True)
    resolved_log_dir = Path(str(log_dir).strip()) if str(log_dir or "").strip() else output_dir / "logs"
    materialized_train_items_path = str(getattr(args, "materialized_train_items_path", "") or "").strip()
    include_splits_value = getattr(args, "include_splits", "")
    if materialized_train_items_path:
        ensure_materialized_cache_metadata(
            materialized_train_items_path,
            expected_format=MATERIALIZED_RUNTIME_ITEMS_FORMAT,
            expected_source_path=args.data,
            expected_include_splits=include_splits_value,
            require_source=True,
        )
        dataset = MaterializedRuntimeItemDataset(
            materialized_train_items_path,
            include_splits=include_splits_value,
            require_frame_cache=True,
            require_feature_cache=True,
            proposal_runtime=proposal_runtime,
            strict_feature_guided_proposal=strict_feature_guided_proposal,
        )
        raw_records = [dict(record) for record in list(getattr(dataset, "records", []) or [])]
    else:
        if bool(getattr(args, "require_materialized_runtime_cache", False)):
            raise ValueError(
                "Trainer-native RL requires --materialized-train-items-path when --require-materialized-runtime-cache=true."
            )
        raw_records = [
            json.loads(line)
            for line in Path(args.data).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if include_splits_value:
            allowed = set(str(value).strip() for value in str(include_splits_value).split(",") if str(value).strip())
            raw_records = [record for record in raw_records if str(record.get("split") or "").strip() in allowed]
    strict_feature_guided_proposal = _raw_records_require_feature_guided_proposal(raw_records)
    if strict_feature_guided_proposal and not str(getattr(args, "proposal_model_path", "") or "").strip():
        raise ValueError(
            "Trainer-native RL requires proposal_model_path because the rollout environment exposes seek_evidence."
        )
    proposal_runtime = (
        _load_training_proposal_runtime(
            proposal_model_path=str(getattr(args, "proposal_model_path", "") or ""),
            proposal_torch_dtype=str(getattr(args, "proposal_torch_dtype", "auto") or "auto"),
            proposal_device=str(getattr(args, "proposal_device", "") or ""),
            runtime=runtime,
        )
        if strict_feature_guided_proposal
        else None
    )
    if materialized_train_items_path and strict_feature_guided_proposal:
        if int(getattr(args, "dataloader_num_workers", 0) or 0) > 0:
            runtime_log(
                "trainer-native RL forcing dataloader_num_workers=0 because materialized runtime items still attach CUDA proposal_runtime during item loading.",
                runtime=runtime,
                main_process_only=True,
            )
            args.dataloader_num_workers = 0
            args.dataloader_prefetch_factor = 0
            args.dataloader_persistent_workers = False
    if not materialized_train_items_path:
        dataset = SaverAgentDataset(
            args.data,
            data_root=args.data_root,
            config=config_builder(args),
            include_splits=include_splits_value,
            require_frame_cache=True,
            require_feature_cache=True,
            proposal_runtime=proposal_runtime,
            strict_feature_guided_proposal=strict_feature_guided_proposal,
        )
    current_model_path = str(args.model_path)
    latest_checkpoint = current_model_path
    reference_model_path = reference_model_resolver(args.model_path, args.reference_model_path)
    use_lora_reference_disable_adapter = bool(args.lora) and float(args.kl_beta) > 0.0
    policy_model = None
    processor = None
    reference_model = None

    def _ensure_models_loaded() -> Tuple[Any, Any, Any]:
        nonlocal policy_model, processor, reference_model
        if policy_model is None or processor is None:
            policy_model, processor = load_qwen_model_and_processor(
                current_model_path,
                torch_dtype=args.torch_dtype,
                attn_implementation=args.attn_implementation or None,
                gradient_checkpointing=args.gradient_checkpointing,
                use_lora=args.lora,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                lora_target_modules=[
                    module.strip()
                    for module in str(args.lora_target_modules or "").split(",")
                    if module.strip()
                ]
                or None,
            )
        if (
            reference_model is None
            and not use_lora_reference_disable_adapter
            and float(args.kl_beta) > 0.0
            and str(reference_model_path or "").strip()
        ):
            reference_model, _ = load_qwen_model_and_processor(
                reference_model_path,
                torch_dtype=args.torch_dtype,
                attn_implementation=args.attn_implementation or None,
                gradient_checkpointing=False,
                use_lora=False,
            )
        return policy_model, processor, reference_model

    runtime_log(
        (
            "trainer-native RL startup: "
            f"num_iterations={int(args.num_iterations)} rollout_count={int(args.rollout_count)} "
            f"num_generations={int(args.num_generations)} model_path={current_model_path} "
            f"steps_per_generation={int(getattr(args, 'rl_steps_per_generation', 1))} "
            f"reward_version={str(getattr(args, 'rl_reward_version', DEFAULT_RL_REWARD_VERSION))} "
            f"rollout_cache={bool(args.rl_rollout_use_cache)} fecv_cache={bool(args.rl_fecv_use_cache)} "
            f"loss_microbatch={int(args.rl_compute_loss_microbatch_size)} "
            f"replay={bool(args.rl_replay_buffer_enable)}/{str(args.rl_replay_buffer_type)} "
            f"fecv_policy={str(args.rl_fecv_failure_policy)} "
            f"all_empty_policy={str(args.rl_all_empty_policy)} "
            f"rollout_eval_mode={'inline' if bool(getattr(args, 'inline_rollout_eval', False)) else 'deferred'} "
            f"rollout_eval_output_root={rollout_eval_output_root}"
        ),
        runtime=runtime,
        main_process_only=True,
    )

    for iteration in range(int(args.num_iterations)):
        iter_dir = output_dir / f"iter_{int(iteration):03d}"
        checkpoint_dir = iter_dir / "checkpoint"
        iter_dir.mkdir(parents=True, exist_ok=True)
        indices = select_iteration_indices_fn(
            len(raw_records),
            args.rollout_count,
            args.rollout_start_index,
            iteration,
        )
        items = [dataset[int(index)] for index in indices]
        summary: Dict[str, Any] = {
            "iteration": int(iteration),
            "num_groups": len(items),
            "num_generations": int(args.num_generations),
            "current_model_path": str(current_model_path),
            "reference_model_path": str(reference_model_path),
            "rollout_eval_output_dir": str(rollout_eval_output_root / f"iter_{int(iteration):03d}"),
            "rl_backend": "trainer_native_grpo",
            "rl_reward_version": str(getattr(args, "rl_reward_version", DEFAULT_RL_REWARD_VERSION)),
            "rl_rollout_use_cache": bool(args.rl_rollout_use_cache),
            "rl_fecv_use_cache": bool(args.rl_fecv_use_cache),
            "rl_compute_loss_microbatch_size": int(args.rl_compute_loss_microbatch_size),
            "rl_steps_per_generation": int(getattr(args, "rl_steps_per_generation", 1)),
            "rl_replay_buffer_enable": bool(args.rl_replay_buffer_enable),
            "rl_replay_buffer_type": str(args.rl_replay_buffer_type),
            "rl_replay_buffer_capacity": int(args.rl_replay_buffer_capacity),
            "rl_replay_buffer_alpha": float(args.rl_replay_buffer_alpha),
            "rl_fecv_failure_policy": str(args.rl_fecv_failure_policy),
            "rl_all_empty_policy": str(args.rl_all_empty_policy),
            "rl_log_empty_batch_rank_summary": bool(args.rl_log_empty_batch_rank_summary),
            "rl_open_ended_judge_enabled": bool(getattr(args, "rl_open_ended_judge_enabled", True)),
            "rl_open_ended_judge_base_url": str(getattr(args, "rl_open_ended_judge_base_url", "") or ""),
            "rl_open_ended_judge_model": str(getattr(args, "rl_open_ended_judge_model", "") or ""),
            "rl_open_ended_judge_cache_path": str(getattr(args, "rl_open_ended_judge_cache_path", "") or ""),
        }
        if args.dry_run or not items:
            summary["latest_checkpoint"] = str(current_model_path)
            write_json(iter_dir / "summary.json", summary)
            if runtime.is_main_process:
                append_jsonl(resolved_log_dir / "rl_iteration_metrics.jsonl", summary)
            continue

        runtime_log(
            (
                f"iteration {int(iteration)}: starting trainer-native GRPO update "
                f"with groups={len(items)} num_generations={int(args.num_generations)} "
                f"steps_per_generation={int(getattr(args, 'rl_steps_per_generation', 1))}"
            ),
            runtime=runtime,
            main_process_only=True,
        )
        model, processor, reference_model = _ensure_models_loaded()
        old_policy_model = None
        rollout_eval_config = eval_config_builder(
            args=args,
            current_model_path=current_model_path,
            reference_model_path=reference_model_path,
            config=config_builder(args),
        )
        rollout_eval_callback = _build_rl_authority_checkpoint_callback(
            processor=processor,
            rollout_eval_config=rollout_eval_config,
            rollout_eval_output_dir=rollout_eval_output_root / f"iter_{int(iteration):03d}",
            iteration_index=int(iteration),
        )
        trainer = create_native_grpo_trainer(
            model=model,
            processor=processor,
            train_dataset=_RawItemDataset(items),
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
            steps_per_generation=max(1, int(getattr(args, "rl_steps_per_generation", 1))),
            old_policy_model=old_policy_model,
            reference_model=reference_model,
            use_lora_reference_disable_adapter=use_lora_reference_disable_adapter,
            kl_beta=args.kl_beta,
            ppo_clip_epsilon=args.ppo_clip_epsilon,
            rollout_runner=SaverRolloutRunner(
                max_turns=args.rollout_max_turns,
                config=config_builder(args),
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
            replay_buffer_enable=args.rl_replay_buffer_enable,
            replay_buffer_type=args.rl_replay_buffer_type,
            replay_buffer_capacity=args.rl_replay_buffer_capacity,
            replay_buffer_alpha=args.rl_replay_buffer_alpha,
            fecv_failure_policy=args.rl_fecv_failure_policy,
            all_empty_policy=args.rl_all_empty_policy,
            log_empty_batch_rank_summary=args.rl_log_empty_batch_rank_summary,
            reward_version=getattr(args, "rl_reward_version", DEFAULT_RL_REWARD_VERSION),
            reward_config={
                "reward_version": str(getattr(args, "rl_reward_version", DEFAULT_RL_REWARD_VERSION)),
                "open_ended_judge_enabled": bool(getattr(args, "rl_open_ended_judge_enabled", True)),
                "open_ended_judge_base_url": str(getattr(args, "rl_open_ended_judge_base_url", "") or ""),
                "open_ended_judge_model": str(getattr(args, "rl_open_ended_judge_model", "") or ""),
                "open_ended_judge_cache_path": str(getattr(args, "rl_open_ended_judge_cache_path", "") or ""),
                "open_ended_judge_timeout_sec": float(getattr(args, "rl_open_ended_judge_timeout_sec", 30.0)),
            },
            iteration_index=int(iteration),
            num_iterations=int(args.num_iterations),
            rollout_eval_callback=rollout_eval_callback,
        )
        try:
            train_result = trainer.train()
            budget_stats = trainer.get_budgeting_stats() if hasattr(trainer, "get_budgeting_stats") else BudgetingStats()
            budget_drop_metrics = trainer.get_budget_drop_metrics() if hasattr(trainer, "get_budget_drop_metrics") else {}
            runtime_log(
                _format_budgeting_stats(f"RL iteration {int(iteration)} budgeting", budget_stats),
                runtime=runtime,
                main_process_only=True,
            )
            authority_checkpoint = getattr(rollout_eval_callback, "last_authority_checkpoint_path", None)
            authority_epoch_index = getattr(rollout_eval_callback, "last_authority_epoch_index", None)
            checkpoint_strategy = str(getattr(rollout_eval_callback, "checkpoint_strategy", "epoch_resume_only"))
            if authority_checkpoint is None:
                raise RuntimeError(
                    f"trainer-native RL iteration {int(iteration)} did not publish an authority checkpoint."
                )
            authority_checkpoint = Path(authority_checkpoint)
            _write_rl_checkpoint_authority_metadata(
                checkpoint_root=checkpoint_dir,
                authority_checkpoint=authority_checkpoint,
                iteration_index=int(iteration),
                epoch_index=int(authority_epoch_index or 1),
                runtime=runtime,
            )
            runtime_log(
                (
                    f"RL duplicate root save skipped: iter={int(iteration)} "
                    f"checkpoint_root={checkpoint_dir} authority_checkpoint={authority_checkpoint}"
                ),
                runtime=runtime,
                main_process_only=True,
            )
            latest_checkpoint = str(authority_checkpoint)
            current_model_path = latest_checkpoint
            summary.update(
                {
                    "latest_checkpoint": str(latest_checkpoint),
                    "checkpoint_root": str(checkpoint_dir),
                    "checkpoint_strategy": checkpoint_strategy,
                    "authority_epoch_index": int(authority_epoch_index or 1),
                    "train_loss": float(getattr(train_result, "training_loss", 0.0)),
                }
            )
            summary.update({key: value for key, value in budget_drop_metrics.items()})
            write_json(iter_dir / "summary.json", summary)
            if runtime.is_main_process:
                append_jsonl(resolved_log_dir / "rl_iteration_metrics.jsonl", summary)
            runtime_log(
                (
                    f"RL authority checkpoint published: iter={int(iteration)} "
                    f"latest_checkpoint={latest_checkpoint} checkpoint_root={checkpoint_dir} "
                    f"strategy={checkpoint_strategy}"
                ),
                runtime=runtime,
                main_process_only=True,
            )
        finally:
            _teardown_trainer_iteration_runtime(trainer)
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    final_summary = {
        "timestamp_utc": utc_timestamp(),
        "output_dir": str(output_dir),
        "rollout_eval_output_root": str(rollout_eval_output_root),
        "latest_checkpoint": str(latest_checkpoint),
        "num_iterations": int(args.num_iterations),
        "reference_model_path": str(reference_model_path),
        "rl_backend": "trainer_native_grpo",
        "rl_reward_version": str(getattr(args, "rl_reward_version", DEFAULT_RL_REWARD_VERSION)),
        "checkpoint_strategy": "epoch_resume_only",
        "rl_rollout_use_cache": bool(args.rl_rollout_use_cache),
        "rl_fecv_use_cache": bool(args.rl_fecv_use_cache),
        "rl_compute_loss_microbatch_size": int(args.rl_compute_loss_microbatch_size),
        "rl_replay_buffer_enable": bool(args.rl_replay_buffer_enable),
        "rl_replay_buffer_type": str(args.rl_replay_buffer_type),
        "rl_replay_buffer_capacity": int(args.rl_replay_buffer_capacity),
        "rl_replay_buffer_alpha": float(args.rl_replay_buffer_alpha),
        "rl_fecv_failure_policy": str(args.rl_fecv_failure_policy),
        "rl_all_empty_policy": str(args.rl_all_empty_policy),
        "rl_log_empty_batch_rank_summary": bool(args.rl_log_empty_batch_rank_summary),
    }
    if runtime.is_main_process:
        (output_dir / "latest_checkpoint.txt").write_text(str(latest_checkpoint), encoding="utf-8")
        write_json(resolved_log_dir / "train_saver_rl_final_summary.json", final_summary)
    return final_summary
