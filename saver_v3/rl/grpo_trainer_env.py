from __future__ import annotations

import copy
import gc
import inspect
import json
import math
import os
import random
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from saver_v3.data.dataset import SaverAgentDataset
from saver_v3.data.config import DEFAULT_ROLLOUT_MAX_TURNS
from saver_v3.data.materialized_cache import (
    MATERIALIZED_RUNTIME_ITEMS_FORMAT,
    MaterializedRuntimeItemDataset,
    ensure_materialized_cache_metadata,
)
from saver_v3.data.protocol_signature import DEFAULT_TEACHER_ROLE, build_protocol_signature
from saver_v3.common.experiment_logging import append_jsonl, utc_timestamp, write_json
from saver_v3.model.qwen_policy import QwenGenerationPolicy
from saver_v3.core.reward import (
    DEFAULT_RL_REWARD_VERSION,
    _collect_semantic_queries,
    build_open_ended_reward_judge,
    score_rollout_trace,
)
from saver_v3.core.rollout import SaverRolloutRunner, _build_episode_training_feature
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
    build_completion_only_model_inputs,
    compute_completion_only_log_probs_from_ids,
    compute_completion_only_token_log_probs_from_prepared_inputs,
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
        return dict(self.items[int(index)])


def _compute_rank_local_total_groups(total_groups: int, *, runtime: Any) -> int:
    total_groups = max(0, int(total_groups))
    world_size = max(1, int(getattr(runtime, "world_size", 1) or 1))
    rank = max(0, int(getattr(runtime, "rank", 0) or 0))
    if total_groups <= 0 or world_size <= 1:
        return total_groups
    base = total_groups // world_size
    remainder = total_groups % world_size
    return int(base + (1 if rank < remainder else 0))


def _build_seed_worker_init_fn(*, args: Any) -> Optional[Callable[..., Any]]:
    try:
        from transformers.trainer_utils import seed_worker as _seed_worker
    except Exception:
        return None

    try:
        parameter_names = tuple(inspect.signature(_seed_worker).parameters.keys())
    except (TypeError, ValueError):
        parameter_names = ()

    if not parameter_names or parameter_names == ("worker_id",):
        return _seed_worker

    worker_kwargs: Dict[str, Any] = {}
    if "num_workers" in parameter_names:
        worker_kwargs["num_workers"] = int(getattr(args, "dataloader_num_workers", 0) or 0)
    if "rank" in parameter_names:
        worker_kwargs["rank"] = int(
            getattr(
                args,
                "process_index",
                getattr(distributed_runtime_from_env(), "rank", 0),
            )
            or 0
        )
    if "seed" in parameter_names:
        seed_value = getattr(args, "data_seed", None)
        if seed_value is None:
            seed_value = getattr(args, "seed", None)
        if seed_value is not None:
            worker_kwargs["seed"] = int(seed_value)
    if not worker_kwargs:
        return _seed_worker
    return partial(_seed_worker, **worker_kwargs)


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
        sort_key_fn: Optional[Callable[[Any], Tuple[Any, ...]]] = None,
    ) -> None:
        self.data_source = data_source
        self.mini_repeat_count = max(1, int(mini_repeat_count))
        self.batch_size = max(1, int(batch_size))
        self.repeat_count = max(1, int(repeat_count))
        self.num_samples = max(0, int(len(data_source)))
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.sort_key_fn = sort_key_fn
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
        if callable(self.sort_key_fn):
            indices = sorted(indices, key=lambda index: self.sort_key_fn(self.data_source[index]))
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


def _raw_item_workload_sort_key(item: Any) -> Tuple[Any, ...]:
    if not isinstance(item, dict):
        return (0, 0, 0, 0, "")
    video_meta = dict(item.get("video_meta") or {})
    duration_sec = _safe_float(video_meta.get("duration_sec"), 0.0)
    total_frames = int(video_meta.get("total_frames", 0) or 0)
    allowed_tools = list(((item.get("tool_io") or {}).get("allowed_tools") or []))
    has_seek_evidence = int(any(str(tool_name or "").strip() == "seek_evidence" for tool_name in allowed_tools))
    qa_pair_count = len(item.get("qa_pairs") or []) if isinstance(item.get("qa_pairs"), list) else 0
    label = dict(item.get("label") or {})
    is_anomaly = int(bool(label.get("is_anomaly")))
    return (
        has_seek_evidence,
        int(duration_sec // 10.0),
        int(total_frames // 64),
        min(int(qa_pair_count), 8),
        is_anomaly,
        str(item.get("video_id") or ""),
    )


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


def _distributed_min_int(local_value: int, *, device: torch.device) -> int:
    local_min = int(local_value)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return local_min
    min_tensor = torch.tensor([local_min], dtype=torch.int64, device=device)
    torch.distributed.all_reduce(min_tensor, op=torch.distributed.ReduceOp.MIN)
    return int(min_tensor.item())


def _distributed_sum_float(local_value: float, *, device: torch.device) -> float:
    local_total = float(local_value)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return local_total
    total_tensor = torch.tensor([local_total], dtype=torch.float32, device=device)
    torch.distributed.all_reduce(total_tensor, op=torch.distributed.ReduceOp.SUM)
    return float(total_tensor.item())


def _distributed_max_int(local_value: int, *, device: torch.device) -> int:
    local_max = int(local_value)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return local_max
    max_tensor = torch.tensor([local_max], dtype=torch.int64, device=device)
    torch.distributed.all_reduce(max_tensor, op=torch.distributed.ReduceOp.MAX)
    return int(max_tensor.item())


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


_DEFAULT_SAMPLE_PARTITION_MULTIPLIERS: Dict[str, float] = {
    "anomaly": 1.0,
    "hard_normal": 1.0,
    "easy_normal": 1.0,
    "unknown": 1.0,
}


def _normalize_target_existence(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in {"anomaly", "normal"} else ""


def _rollout_target_existence(rollout: Dict[str, Any]) -> str:
    for key in ("scoring_target", "structured_target", "target", "reference_target"):
        payload = rollout.get(key)
        if isinstance(payload, dict):
            normalized = _normalize_target_existence(payload.get("existence"))
            if normalized:
                return normalized
    item = rollout.get("item")
    if isinstance(item, dict):
        for key in ("structured_target", "target"):
            payload = item.get(key)
            if isinstance(payload, dict):
                normalized = _normalize_target_existence(payload.get("existence"))
                if normalized:
                    return normalized
        label = item.get("label")
        if isinstance(label, dict) and "is_anomaly" in label:
            return "anomaly" if bool(label.get("is_anomaly")) else "normal"
    label = rollout.get("label")
    if isinstance(label, dict) and "is_anomaly" in label:
        return "anomaly" if bool(label.get("is_anomaly")) else "normal"
    return ""


def _normalize_sample_partition_multipliers(raw_value: Any) -> Dict[str, float]:
    normalized = dict(_DEFAULT_SAMPLE_PARTITION_MULTIPLIERS)
    if not isinstance(raw_value, dict):
        return normalized
    for key, value in raw_value.items():
        partition = str(key or "").strip().lower()
        if partition not in normalized:
            continue
        try:
            normalized[partition] = max(0.0, float(value))
        except Exception:
            continue
    return normalized


def _resolve_sample_partition(rollout: Dict[str, Any]) -> str:
    target_existence = _rollout_target_existence(rollout)
    if target_existence == "anomaly":
        return "anomaly"
    if target_existence == "normal":
        reward_summary = dict(rollout.get("reward_summary") or {})
        normal_case_type = str(
            reward_summary.get("normal_case_type") or ""
        ).strip().lower()
        if normal_case_type == "easy_normal":
            return "easy_normal"
        return "hard_normal"
    return "unknown"


def _resolve_sample_partition_multiplier(
    *,
    sample_partition: str,
    reward_summary: Dict[str, Any],
    sample_partition_multipliers: Optional[Dict[str, float]] = None,
) -> float:
    normalized_partition = str(sample_partition or "unknown").strip().lower() or "unknown"
    multipliers = _DEFAULT_SAMPLE_PARTITION_MULTIPLIERS
    if isinstance(sample_partition_multipliers, dict):
        multipliers = {**multipliers, **sample_partition_multipliers}
    multiplier = float(multipliers.get(normalized_partition, 1.0) or 1.0)
    if normalized_partition == "easy_normal":
        multiplier *= float(reward_summary.get("easy_normal_sample_loss_multiplier") or 1.0)
    return max(0.0, float(multiplier))


def _degrade_reward_summary_for_fecv_failure(
    reward_summary: Dict[str, Any],
    *,
    error_message: str = "",
) -> Dict[str, Any]:
    del error_message
    return copy.deepcopy(dict(reward_summary or {}))


def _replay_priority_from_experience(experience: Dict[str, Any]) -> float:
    eps = 1e-4
    episode_specs = list((experience or {}).get("episode_specs") or [])
    priorities: List[float] = []
    for episode_spec in episode_specs:
        advantage = abs(float((episode_spec or {}).get("advantage", 0.0) or 0.0))
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
        return dict(self.buffer[0])


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
        self.buffer.append(dict(experience or {}))
        self.advantages.append(priority)

    def sample(self) -> Dict[str, Any]:
        if not self.buffer:
            raise ValueError("Replay buffer is empty.")
        scaled = [max(1e-6, float(priority)) ** self.alpha for priority in self.advantages]
        selected = random.choices(range(len(self.buffer)), weights=scaled, k=1)[0]
        return dict(self.buffer[int(selected)])


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
        self.buffer.append(dict(experience or {}))
        self.weights.append(1.0)

    def sample(self) -> Dict[str, Any]:
        if not self.buffer:
            raise ValueError("Replay buffer is empty.")
        scaled = [max(1e-6, float(weight)) ** self.alpha for weight in self.weights]
        selected = random.choices(range(len(self.buffer)), weights=scaled, k=1)[0]
        return dict(self.buffer[int(selected)])


def get_replay_buffer(buffer_type: str, capacity: int, alpha: float = 1.0):
    normalized_type = str(buffer_type or "none").strip().lower()
    if normalized_type == "ssr":
        return SSRReplayBuffer(capacity, alpha=alpha)
    if normalized_type == "dapo":
        return DapoReplayBuffer(capacity, alpha=alpha)
    if normalized_type == "none":
        return None
    raise ValueError(f"Invalid replay buffer type: {buffer_type!r}")


def _named_tensor_non_finite_summary(name: str, tensor: torch.Tensor) -> Dict[str, Any]:
    value = tensor.detach()
    if bool(getattr(value, "is_sparse", False)):
        value = value.coalesce().values()
    summary: Dict[str, Any] = {
        "name": str(name or ""),
        "shape": tuple(int(dim) for dim in value.shape),
        "dtype": str(value.dtype).replace("torch.", ""),
        "device": str(value.device),
        "numel": int(value.numel()),
        "layout": str(value.layout),
    }
    if value.numel() <= 0:
        summary.update(
            {
                "all_finite": True,
                "nan_count": 0,
                "posinf_count": 0,
                "neginf_count": 0,
            }
        )
        return summary
    flat_value = value.reshape(-1)
    chunk_elements = max(1, min(int(flat_value.numel()), 4_194_304))
    nan_count = 0
    posinf_count = 0
    neginf_count = 0
    for offset in range(0, int(flat_value.numel()), int(chunk_elements)):
        chunk = flat_value.narrow(0, int(offset), min(int(chunk_elements), int(flat_value.numel()) - int(offset)))
        nan_count += int(torch.isnan(chunk).sum().item())
        posinf_count += int(torch.isposinf(chunk).sum().item())
        neginf_count += int(torch.isneginf(chunk).sum().item())
    summary.update(
        {
            "all_finite": bool((nan_count + posinf_count + neginf_count) == 0),
            "nan_count": int(nan_count),
            "posinf_count": int(posinf_count),
            "neginf_count": int(neginf_count),
        }
    )
    return summary


def _scan_named_tensors_for_non_finite(named_tensors: Sequence[Tuple[str, torch.Tensor]], *, max_entries: int = 8) -> Dict[str, Any]:
    checked_count = 0
    non_finite_count = 0
    entries: List[Dict[str, Any]] = []
    for name, tensor in list(named_tensors or []):
        if not isinstance(tensor, torch.Tensor):
            continue
        checked_count += 1
        value = tensor.detach()
        if bool(getattr(value, "is_sparse", False)):
            value = value.coalesce().values()
        flat_value = value.reshape(-1)
        chunk_elements = max(1, min(int(flat_value.numel()), 4_194_304))
        has_non_finite = False
        for offset in range(0, int(flat_value.numel()), int(chunk_elements)):
            chunk = flat_value.narrow(0, int(offset), min(int(chunk_elements), int(flat_value.numel()) - int(offset)))
            if not bool(torch.all(torch.isfinite(chunk)).item()):
                has_non_finite = True
                break
        if not has_non_finite:
            continue
        non_finite_count += 1
        if len(entries) < max(1, int(max_entries)):
            entries.append(_named_tensor_non_finite_summary(name=name, tensor=tensor))
    return {
        "checked_count": int(checked_count),
        "non_finite_count": int(non_finite_count),
        "entries": entries,
    }


def _iter_nested_named_tensors(prefix: str, value: Any):
    if isinstance(value, torch.Tensor):
        yield str(prefix or ""), value
        return
    if isinstance(value, dict):
        for key, nested_value in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_nested_named_tensors(child_prefix, nested_value)
        return
    if isinstance(value, (list, tuple)):
        for index, nested_value in enumerate(list(value)):
            child_prefix = f"{prefix}[{index}]"
            yield from _iter_nested_named_tensors(child_prefix, nested_value)


def _optimizer_param_group_summaries(optimizer: Any) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    param_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(param_groups, (list, tuple)):
        return summaries
    for group_index, group in enumerate(list(param_groups)):
        if not isinstance(group, dict):
            continue
        params = list(group.get("params") or [])
        summaries.append(
            {
                "group_index": int(group_index),
                "param_count": int(len(params)),
                "trainable_param_count": int(
                    sum(1 for param in params if bool(getattr(param, "requires_grad", False)))
                ),
                "lr": float(group.get("lr", 0.0) or 0.0),
                "weight_decay": float(group.get("weight_decay", 0.0) or 0.0),
            }
        )
    return summaries


def _flat_master_scan_payload_for_optimizer(optimizer: Any) -> Dict[str, Any]:
    fp32_partitions = []
    for index, tensor in enumerate(list(getattr(optimizer, "single_partition_of_fp32_groups", []) or [])):
        if isinstance(tensor, torch.Tensor):
            fp32_partitions.append((f"single_partition_of_fp32_groups[{index}]", tensor))

    fp32_partition_grads = []
    for index, tensor in enumerate(list(getattr(optimizer, "single_partition_of_fp32_groups", []) or [])):
        grad = getattr(tensor, "grad", None)
        if isinstance(grad, torch.Tensor):
            fp32_partition_grads.append((f"single_partition_of_fp32_groups[{index}].grad", grad))

    bit16_flat_groups = []
    for index, tensor in enumerate(list(getattr(optimizer, "bit16_groups_flat", []) or [])):
        if isinstance(tensor, torch.Tensor):
            bit16_flat_groups.append((f"bit16_groups_flat[{index}]", tensor))

    averaged_gradients = []
    raw_averaged_gradients = getattr(optimizer, "averaged_gradients", None)
    if isinstance(raw_averaged_gradients, dict):
        for key, value in sorted(raw_averaged_gradients.items(), key=lambda item: repr(item[0])):
            averaged_gradients.extend(
                list(_iter_nested_named_tensors(f"averaged_gradients[{key}]", value))
            )
    elif isinstance(raw_averaged_gradients, (list, tuple)):
        for index, value in enumerate(list(raw_averaged_gradients)):
            averaged_gradients.extend(
                list(_iter_nested_named_tensors(f"averaged_gradients[{index}]", value))
            )

    optimizer_state_tensors = []
    optimizer_state = getattr(optimizer, "state", None)
    if isinstance(optimizer_state, dict):
        fp32_group_lookup = {
            id(tensor): index
            for index, tensor in enumerate(list(getattr(optimizer, "single_partition_of_fp32_groups", []) or []))
            if isinstance(tensor, torch.Tensor)
        }
        for raw_key, raw_state in optimizer_state.items():
            if not isinstance(raw_state, dict):
                continue
            if isinstance(raw_key, torch.Tensor) and id(raw_key) in fp32_group_lookup:
                prefix = f"single_partition_of_fp32_groups[{fp32_group_lookup[id(raw_key)]}].state"
            else:
                prefix = f"optimizer_state[{repr(raw_key)}]"
            optimizer_state_tensors.extend(list(_iter_nested_named_tensors(prefix, raw_state)))

    sections = {
        "fp32_partitions": _scan_named_tensors_for_non_finite(fp32_partitions),
        "fp32_partition_grads": _scan_named_tensors_for_non_finite(fp32_partition_grads),
        "bit16_flat_groups": {
            "checked_count": int(len(bit16_flat_groups)),
            "non_finite_count": 0,
            "metadata_only": True,
            "entries": [
                {
                    "name": str(name or ""),
                    "shape": tuple(int(dim) for dim in tensor.shape),
                    "dtype": str(tensor.dtype).replace("torch.", ""),
                    "device": str(tensor.device),
                    "numel": int(tensor.numel()),
                }
                for name, tensor in bit16_flat_groups[:4]
                if isinstance(tensor, torch.Tensor)
            ],
        },
        "averaged_gradients": _scan_named_tensors_for_non_finite(averaged_gradients),
        "optimizer_state": _scan_named_tensors_for_non_finite(optimizer_state_tensors),
    }
    sections["any_non_finite"] = bool(
        any(int(section.get("non_finite_count", 0)) > 0 for section in sections.values() if isinstance(section, dict))
    )
    return sections


def _write_flat_master_forensics_dump(
    *,
    trainer: Any,
    optimizer: Any,
    stage: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    flat_master_scan = _flat_master_scan_payload_for_optimizer(optimizer)
    if not bool(flat_master_scan.get("any_non_finite")):
        return None
    runtime = distributed_runtime_from_env()
    global_rank = getattr(runtime, "global_rank", None)
    if global_rank is None:
        global_rank = getattr(runtime, "rank", 0)
    dump_path = trainer._non_finite_dump_dir() / (
        f"{utc_timestamp()}_{str(stage or 'optimizer_flat_master_state')}_"
        f"rank{int(getattr(runtime, 'local_rank', 0) or 0)}_"
        f"call{int(getattr(trainer, '_debug_last_compute_loss_call_index', 0))}.json"
    )
    payload = {
        "timestamp": utc_timestamp(),
        "stage": str(stage or ""),
        "compute_loss_call": int(getattr(trainer, "_debug_last_compute_loss_call_index", 0)),
        "local_prepared_batch_count": int(getattr(trainer, "_debug_last_compute_loss_rank_local_batch_count", 0)),
        "runtime": {
            "local_rank": int(getattr(runtime, "local_rank", 0) or 0),
            "global_rank": int(global_rank or 0),
            "world_size": int(getattr(runtime, "world_size", 1) or 1),
        },
        "trainer_state": {
            "global_step": int(getattr(getattr(trainer, "state", None), "global_step", 0) or 0),
            "epoch": float(getattr(getattr(trainer, "state", None), "epoch", 0.0) or 0.0),
        },
        "optimizer_class": type(optimizer).__name__,
        "optimizer_param_groups": _optimizer_param_group_summaries(optimizer),
        "flat_master_scan": flat_master_scan,
        "extra": dict(extra or {}),
    }
    write_json(dump_path, payload)
    runtime_log(
        (
            "trainer-native RL detected non-finite DeepSpeed flat master/state tensors: "
            f"stage={str(stage or '')} dump={dump_path}"
        ),
        runtime=runtime,
        main_process_only=False,
    )
    return dump_path


class _NativeRLOptimizerStepProxy:
    def __init__(self, optimizer: Any, *, trainer: Any):
        self._optimizer = optimizer
        self._trainer = trainer

    def _optimizer_param_group_summaries(self) -> List[Dict[str, Any]]:
        return _optimizer_param_group_summaries(self._optimizer)

    def _flat_master_scan_payload(self) -> Dict[str, Any]:
        return _flat_master_scan_payload_for_optimizer(self._optimizer)

    def _write_flat_master_forensics_dump(
        self,
        *,
        stage: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        return _write_flat_master_forensics_dump(
            trainer=self._trainer,
            optimizer=self._optimizer,
            stage=stage,
            extra=extra,
        )

    def step(self, *args, **kwargs):
        if bool(getattr(self._trainer, "_native_rl_skip_next_optimizer_step", False)):
            try:
                setattr(self._trainer.accelerator, "optimizer_step_was_skipped", True)
            except Exception:
                pass
            runtime_log(
                "RL optimizer step skipped: "
                f"reason={str(getattr(self._trainer, '_native_rl_last_skip_reason', '') or 'unknown')} "
                f"next_skip_count={int(getattr(self._trainer, '_optimizer_step_skips', 0)) + 1}",
                runtime=distributed_runtime_from_env(),
                main_process_only=False,
            )
            self._trainer._native_rl_skip_next_optimizer_step = False
            self._trainer._optimizer_step_skips += 1
            return None
        self._write_flat_master_forensics_dump(
            stage="pre_optimizer_step_flat_master_state",
            extra={"phase": "pre_step"},
        )
        try:
            setattr(self._trainer.accelerator, "optimizer_step_was_skipped", False)
        except Exception:
            pass
        try:
            result = self._optimizer.step(*args, **kwargs)
        except Exception as exc:
            self._write_flat_master_forensics_dump(
                stage="optimizer_step_exception_flat_master_state",
                extra={
                    "phase": "step_exception",
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                },
            )
            raise
        self._write_flat_master_forensics_dump(
            stage="post_optimizer_step_flat_master_state",
            extra={"phase": "post_step"},
        )
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
    ):
        self.runtime = runtime
        self.iteration_index = max(0, int(iteration_index))
        self.num_iterations = max(1, int(num_iterations))
        self.total_groups = max(0, int(total_groups))
        self.local_total_groups = 0
        self.num_generations = max(1, int(num_generations))
        self.compute_loss_microbatch_size = max(1, int(compute_loss_microbatch_size))
        self.rollout_use_generation_cache = bool(rollout_use_generation_cache)
        self._active_iteration_index = int(self.iteration_index)
        self.trainer_step = 0
        self.processed_groups = 0
        self.batch_index = 0
        self.last_video_id = ""
        self.last_stage = ""

    def set_total_groups(self, total_groups: int) -> None:
        self.total_groups = max(0, int(total_groups))

    def set_local_total_groups(self, local_total_groups: int) -> None:
        self.local_total_groups = max(0, int(local_total_groups))

    def set_trainer_step(self, trainer_step: int) -> None:
        self.trainer_step = max(0, int(trainer_step))

    def _reset_progress_counters(self) -> None:
        self.trainer_step = 0
        self.processed_groups = 0
        self.batch_index = 0
        self.last_video_id = ""
        self.last_stage = ""

    def _local_total_groups(self) -> int:
        if self.local_total_groups > 0:
            return int(self.local_total_groups)
        return _compute_rank_local_total_groups(int(self.total_groups), runtime=self.runtime)

    def _display_global_processed_groups(self, global_processed: int) -> int:
        processed = max(0, int(global_processed))
        if self.total_groups > 0:
            return min(processed, int(self.total_groups))
        return processed

    def _global_processed_groups(self) -> int:
        local_processed = max(0, int(self.processed_groups))
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return local_processed
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        payload = torch.tensor([local_processed], dtype=torch.int64, device=device)
        torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.SUM)
        return int(payload.item())

    def _maybe_reset_for_new_iteration(self) -> None:
        current_iteration_index = max(0, int(self.iteration_index))
        if current_iteration_index == int(getattr(self, "_active_iteration_index", current_iteration_index)):
            return
        self._active_iteration_index = current_iteration_index
        self._reset_progress_counters()

    def reset_for_iteration(
        self,
        *,
        iteration_index: int,
        num_iterations: int,
        total_groups: int,
    ) -> None:
        self.iteration_index = max(0, int(iteration_index))
        self.num_iterations = max(1, int(num_iterations))
        self.total_groups = max(0, int(total_groups))
        self._active_iteration_index = int(self.iteration_index)
        self._reset_progress_counters()

    def start_batch(self, *, num_items: int) -> None:
        del num_items
        self._maybe_reset_for_new_iteration()
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
        self._maybe_reset_for_new_iteration()
        self.processed_groups += 1
        self.last_video_id = str(video_id or "")
        local_total = self._local_total_groups()
        local_total_display = int(local_total) if local_total > 0 else "?"
        global_total_display = int(self.total_groups) if self.total_groups > 0 else "?"
        global_processed = self._display_global_processed_groups(self._global_processed_groups())
        runtime_log(
            (
                f"RL rank progress: iter={int(self.iteration_index) + 1}/{int(self.num_iterations)} "
                f"batch={int(self.batch_index)} "
                f"local_groups={int(self.processed_groups)}/{local_total_display} "
                f"global_groups={int(global_processed)}/{global_total_display}"
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


def _should_publish_rl_iteration_artifacts(
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


def _rl_iteration_checkpoint_strategy(
    *,
    publish_iteration_artifacts: bool,
    eval_start_iteration: int,
    eval_every_iterations: int,
) -> str:
    if publish_iteration_artifacts:
        return (
            "epoch_resume_inline_eval"
            f"_start_{max(1, int(eval_start_iteration))}"
            f"_every_{max(1, int(eval_every_iterations))}_iterations"
        )
    return "rolling_epoch_resume_continuation"


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
    checkpoint_strategy: str = "epoch_resume_only",
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
            self.checkpoint_strategy = str(checkpoint_strategy or "epoch_resume_only")
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
                                max_tool_message_frames=int(getattr(self.rollout_eval_config, "max_tool_message_frames", 0)),
                                max_total_video_frames=int(getattr(self.rollout_eval_config, "max_total_video_frames", 0)),
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
                progress.set_local_total_groups(_estimate_local_total_groups(trainer=trainer, args=args))
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


def _build_grad_norm_probe_callback(*, trainer: Any):
    """Per-step grad_norm / loss / batch / param_group probe. Bypasses logging_steps.

    Purpose: diagnose constant grad_norm=sqrt(2)=1.4142135381698608 observed
    in pipeline_20260421_184636.log. Confirms whether the value is:
      - a per-step reality (DeepSpeed stale sentinel), OR
      - 2 param groups each clipped to 1.0 -> norm([1.0, 1.0]) = sqrt(2), OR
      - a sparse logging artifact (only 2 samples at steps 10 and 20 matched coincidentally).

    Emits GRAD_PROBE lines every step, rank 0 only.
    """
    try:
        from transformers import TrainerCallback
    except Exception:
        class TrainerCallback:  # type: ignore[no-redef]
            pass

    def _probe_rank() -> int:
        runtime = distributed_runtime_from_env()
        rank = getattr(runtime, "rank", None)
        if rank is None:
            rank = getattr(trainer, "global_rank", None)
        if rank is None:
            trainer_args = getattr(trainer, "args", None)
            rank = getattr(trainer_args, "process_index", None)
        if rank is None:
            trainer_args = getattr(trainer, "args", None)
            rank = getattr(trainer_args, "local_rank", None)
        if rank is None:
            rank = getattr(trainer, "local_rank", 0)
        return int(rank or 0)

    class GradNormProbeCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            try:
                rank = _probe_rank()
                if rank != 0:
                    return control
                opt = getattr(trainer, "optimizer", None)
                deepspeed_engine = getattr(trainer, "deepspeed", None) or getattr(trainer, "model_wrapped", None)
                num_groups = -1
                group_summary = []
                if opt is not None and hasattr(opt, "param_groups"):
                    groups = list(opt.param_groups)
                    num_groups = len(groups)
                    for gi, g in enumerate(groups):
                        params_in_group = g.get("params", []) or []
                        trainable = sum(1 for p in params_in_group if getattr(p, "requires_grad", False))
                        group_summary.append(f"g{gi}[n={len(params_in_group)},trainable={trainable},lr={g.get('lr','?')},wd={g.get('weight_decay','?')}]")
                ds_type = type(deepspeed_engine).__name__ if deepspeed_engine is not None else "None"
                print(
                    f"GRAD_PROBE_INIT num_param_groups={num_groups} groups=[{', '.join(group_summary)}] "
                    f"deepspeed_engine_type={ds_type} max_grad_norm={getattr(args,'max_grad_norm','?')}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"GRAD_PROBE_INIT_ERR {type(exc).__name__}: {exc}", flush=True)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            del args, kwargs
            try:
                rank = _probe_rank()
                if rank != 0:
                    return control
                global_step = int(getattr(state, "global_step", 0) or 0)
                epoch = float(getattr(state, "epoch", 0.0) or 0.0)
                last_log = {}
                log_history = getattr(state, "log_history", None) or []
                if log_history:
                    last_log = dict(log_history[-1])
                loss_val = last_log.get("loss", None)
                lr_val = last_log.get("learning_rate", None)
                grad_norm_hf = last_log.get("grad_norm", None)
                grad_norm_ds = None
                deepspeed_engine = getattr(trainer, "deepspeed", None)
                if deepspeed_engine is not None and hasattr(deepspeed_engine, "get_global_grad_norm"):
                    try:
                        raw = deepspeed_engine.get_global_grad_norm()
                        grad_norm_ds = float(raw.item()) if hasattr(raw, "item") else float(raw) if raw is not None else None
                    except Exception as exc:  # noqa: BLE001
                        grad_norm_ds = f"ERR:{type(exc).__name__}"
                batch_len = None
                try:
                    last_batch = getattr(trainer, "_last_rl_batch_size", None)
                    batch_len = int(last_batch) if last_batch is not None else None
                except Exception:
                    batch_len = None
                print(
                    f"GRAD_PROBE step={global_step} epoch={epoch:.4f} "
                    f"grad_norm_hf={grad_norm_hf!r} grad_norm_ds={grad_norm_ds!r} "
                    f"loss={loss_val!r} lr={lr_val!r} last_batch_size={batch_len!r}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"GRAD_PROBE_ERR {type(exc).__name__}: {exc}", flush=True)
            return control

    return GradNormProbeCallback()


def _build_parameter_finite_probe_callback(*, trainer: Any):
    try:
        from transformers import TrainerCallback
    except Exception:
        class TrainerCallback:  # type: ignore[no-redef]
            pass

    class ParameterFiniteProbeCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            del args, state, kwargs
            model_ref = model if model is not None else getattr(trainer, "model", None)
            if model_ref is not None:
                trainer._assert_finite_trainable_parameters(
                    stage="train_begin_params",
                    model=model_ref,
                )
            return control

        def on_step_begin(self, args, state, control, model=None, **kwargs):
            del args, kwargs
            model_ref = model if model is not None else getattr(trainer, "model", None)
            if model_ref is not None:
                trainer._assert_finite_trainable_parameters(
                    stage="step_begin_params",
                    model=model_ref,
                    extra={
                        "global_step": int(getattr(state, "global_step", 0) or 0),
                        "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
                    },
                )
            return control

        def on_step_end(self, args, state, control, model=None, **kwargs):
            del args, kwargs
            model_ref = model if model is not None else getattr(trainer, "model", None)
            trainer._write_deepspeed_flat_master_forensics_dump(
                stage="post_optimizer_step_flat_master_state",
                extra={
                    "global_step": int(getattr(state, "global_step", 0) or 0),
                    "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
                    "phase": "callback_post_step",
                },
            )
            if model_ref is not None:
                trainer._assert_finite_trainable_parameters(
                    stage="post_optimizer_step_params",
                    model=model_ref,
                    extra={
                        "global_step": int(getattr(state, "global_step", 0) or 0),
                        "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
                    },
                )
            return control

    return ParameterFiniteProbeCallback()


def _is_episode_feature_from_feature(feature: Dict[str, Any]) -> bool:
    return isinstance(feature.get("messages"), list) and isinstance(feature.get("assistant_supervision"), list)


def _compute_group_relative_advantages(
    rollouts: Sequence[Dict[str, Any]],
    *,
    clip_value: Optional[float] = None,
    eps: float = 1e-6,
    sample_partition_multipliers: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    rewards = [float(((rollout.get("reward_summary") or {}).get("total_reward")) or 0.0) for rollout in rollouts]
    sample_partitions = [_resolve_sample_partition(rollout) for rollout in rollouts]
    mean_reward = sum(rewards) / float(len(rewards)) if rewards else 0.0
    variance = sum((reward - mean_reward) ** 2 for reward in rewards) / float(len(rewards)) if rewards else 0.0
    std_reward = math.sqrt(max(variance, 0.0))
    updated: List[Dict[str, Any]] = []
    for rollout, reward, sample_partition in zip(rollouts, rewards, sample_partitions):
        reward_summary = dict(rollout.get("reward_summary") or {})
        advantage = 0.0 if std_reward <= eps else (reward - mean_reward) / (std_reward + eps)
        advantage_source = "zero_advantage" if std_reward <= eps else "group_relative"
        if clip_value is not None:
            advantage = max(-float(clip_value), min(float(clip_value), float(advantage)))
        enriched = dict(rollout)
        enriched["group_reward"] = float(reward)
        enriched["group_reward_mean"] = float(mean_reward)
        enriched["group_reward_std"] = float(std_reward)
        enriched["group_advantage"] = round(float(advantage), 6)
        enriched["advantage_source"] = str(advantage_source)
        enriched["sample_partition"] = str(sample_partition)
        enriched["sample_partition_multiplier"] = round(
            float(
                _resolve_sample_partition_multiplier(
                    sample_partition=str(sample_partition),
                    reward_summary=reward_summary,
                    sample_partition_multipliers=sample_partition_multipliers,
                )
            ),
            6,
        )
        updated.append(enriched)
    return updated


def _log_rollout_reward_diagnostics(
    *,
    video_id: str,
    rollouts: Sequence[Dict[str, Any]],
    min_weight: float,
) -> None:
    if not rollouts:
        runtime_log(
            f"rl reward/advantage debug: video_id={video_id} rollout_count=0 min_weight={float(min_weight):.6f}",
            runtime=distributed_runtime_from_env(),
            main_process_only=False,
        )
        return
    reward_values = [round(float(((rollout.get("reward_summary") or {}).get("total_reward")) or 0.0), 6) for rollout in rollouts]
    advantage_values = [round(float(rollout.get("group_advantage", 0.0) or 0.0), 6) for rollout in rollouts]
    advantage_sources = [str(rollout.get("advantage_source") or "") for rollout in rollouts]
    sample_partitions = [str(rollout.get("sample_partition") or "") for rollout in rollouts]
    reward_mean = sum(reward_values) / float(len(reward_values))
    reward_variance = sum((value - reward_mean) ** 2 for value in reward_values) / float(len(reward_values))
    reward_std = math.sqrt(max(reward_variance, 0.0))
    zero_variance_group_count = 1 if reward_std <= 1e-6 else 0
    zero_variance_rollout_count = sum(1 for rollout in rollouts if bool(rollout.get("zero_variance_group")))
    zero_variance_skipped_count = zero_variance_rollout_count if zero_variance_group_count > 0 else 0
    filtered_count = sum(1 for value in advantage_values if abs(float(value)) < float(min_weight))
    zero_advantage_count = sum(1 for value in advantage_values if abs(float(value)) <= 1e-8)
    generation_ids = [int(rollout.get("generation_id", -1) or -1) for rollout in rollouts]
    runtime_log(
        "rl reward/advantage debug: "
        f"video_id={video_id} "
        f"rollout_count={len(rollouts)} "
        f"generation_ids={generation_ids} "
        f"min_weight={float(min_weight):.6f} "
        f"filtered_below_min_weight={int(filtered_count)} "
        f"zero_advantage_count={int(zero_advantage_count)} "
        f"zero_variance_group_count={int(zero_variance_group_count)} "
        f"zero_variance_rollout_count={int(zero_variance_rollout_count)} "
        f"zero_variance_skipped_count={int(zero_variance_skipped_count)} "
        f"reward_mean={float(reward_mean):.6f} "
        f"reward_std={float(reward_std):.6f} "
        f"reward_values={[f'{value:.6f}' for value in reward_values]} "
        f"advantages={[f'{value:.6f}' for value in advantage_values]} "
        f"advantage_sources={advantage_sources} "
        f"sample_partitions={sample_partitions}",
        runtime=distributed_runtime_from_env(),
        main_process_only=False,
    )
    component_rows = []
    for rollout in rollouts:
        reward_summary = dict(rollout.get("reward_summary") or {})
        components = dict(reward_summary.get("components") or {})
        weighted_components = dict(reward_summary.get("weighted_components") or {})
        advantage = float(rollout.get("group_advantage", 0.0) or 0.0)
        component_rows.append(
            {
                "generation_id": int(rollout.get("generation_id", -1) or -1),
                "total_reward": round(float(reward_summary.get("total_reward") or 0.0), 6),
                "group_advantage": round(float(advantage), 6),
                "below_min_weight": bool(abs(float(advantage)) < float(min_weight)),
                "accuracy_reward": round(float(components.get("accuracy_reward") or 0.0), 6),
                "weighted_accuracy_reward": round(float(weighted_components.get("accuracy_reward") or 0.0), 6),
                "protocol_finalize_reward": round(float(components.get("protocol_finalize_reward") or 0.0), 6),
                "weighted_protocol_finalize_reward": round(
                    float(weighted_components.get("protocol_finalize_reward") or 0.0),
                    6,
                ),
                "stage_necessity_reward": round(float(components.get("stage_necessity_reward") or 0.0), 6),
                "query_alignment_reward": round(float(components.get("query_alignment_reward") or 0.0), 6),
                "efficiency_reward": round(float(components.get("efficiency_reward") or 0.0), 6),
                "anomaly_false_normal_penalty": round(float(components.get("anomaly_false_normal_penalty") or 0.0), 6),
                "advantage_source": str(rollout.get("advantage_source") or ""),
                "sample_partition": str(rollout.get("sample_partition") or ""),
                "sample_partition_multiplier": round(
                    float(rollout.get("sample_partition_multiplier") or 1.0),
                    6,
                ),
                "normal_case_type": str(reward_summary.get("normal_case_type") or ""),
                "easy_normal_sample_loss_multiplier": round(
                    float(reward_summary.get("easy_normal_sample_loss_multiplier") or 1.0),
                    6,
                ),
                "sample_partition_type": str(rollout.get("sample_partition_type") or ""),
                "advantage_source": str(rollout.get("advantage_source") or "group_relative"),
                "normal_continuous_verifier_score": round(
                    float(reward_summary.get("normal_continuous_verifier_score") or 0.0),
                    6,
                ),
                "normal_verifier_primary_status": str(
                    reward_summary.get("normal_verifier_primary_status") or "unknown"
                ),
                "normal_verifier_next_tool": str(
                    reward_summary.get("normal_verifier_next_tool") or "unknown"
                ),
            }
        )
    runtime_log(
        f"rl reward components debug: video_id={video_id} rows={component_rows}",
        runtime=distributed_runtime_from_env(),
        main_process_only=False,
    )


def _flatten_rollout_to_episode_features(
    rollout: Dict[str, Any],
    *,
    min_abs_advantage: float = 0.0,
) -> List[Dict[str, Any]]:
    del rollout
    del min_abs_advantage
    raise ValueError(
        "Active native RL no longer uses intermediate rollout features; generation must build `episode_spec` "
        "entries directly from rollout results."
    )


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
        self.max_tool_message_frames = int(config.get("max_tool_message_frames", 0))
        self.max_total_video_frames = int(config.get("max_total_video_frames", 0))
        self.keep_recent_tool_image_messages = int(config["keep_recent_tool_image_messages"])
        self.keep_recent_text_messages = int(config["keep_recent_text_messages"])
        self.max_seq_length = int(config["max_seq_length"])
        self.policy_do_sample = bool(config["policy_do_sample"])
        self.policy_temperature = config["policy_temperature"]
        self.policy_top_p = config["policy_top_p"]
        self.policy_top_k = config["policy_top_k"]
        self.policy_repetition_penalty = config["policy_repetition_penalty"]
        self.rollout_use_generation_cache = bool(config["rollout_use_generation_cache"])
        self.compute_loss_microbatch_size = max(1, int(config["compute_loss_microbatch_size"]))
        self.steps_per_generation = max(1, int(config["steps_per_generation"]))
        self._generation_step_batch_size = max(1, int(config["per_device_train_batch_size"]))
        self._generation_batch_size = max(1, self._generation_step_batch_size * self.steps_per_generation)
        self.proposal_runtime = config.get("proposal_runtime")
        self.strict_feature_guided_proposal = bool(config.get("strict_feature_guided_proposal", False))
        self._buffered_generation_step_payloads: List[Dict[str, Any]] = []
        self._buffered_generation_batch_key: Optional[Tuple[Any, ...]] = None
        self.replay_buffer_enable = bool(config["replay_buffer_enable"])
        self.replay_buffer_type = str(config["replay_buffer_type"] or "none").strip().lower()
        self.replay_buffer_capacity = max(0, int(config["replay_buffer_capacity"]))
        self.replay_buffer_alpha = float(config["replay_buffer_alpha"])
        self.all_empty_policy = str(config["all_empty_policy"] or "true_skip").strip().lower()
        self.log_empty_batch_rank_summary = bool(config["log_empty_batch_rank_summary"])
        self.reward_version = str(config["reward_version"] or DEFAULT_RL_REWARD_VERSION).strip().lower()
        self.reward_config = dict(config["reward_config"] or {})
        self.reward_config.setdefault("reward_version", self.reward_version)
        self.reward_judge = build_open_ended_reward_judge(reward_config=self.reward_config)
        self._sample_partition_multipliers = _normalize_sample_partition_multipliers(
            self.reward_config.get("sample_partition_multipliers")
        )
        self._native_visual_tensor_dtype = (
            torch.bfloat16
            if bool(config.get("bf16"))
            else (torch.float16 if bool(config.get("fp16")) else None)
        )
        self._reference_model_device = None
        self._budgeting_stats = BudgetingStats()
        self._zero_response_dropped = 0
        self._materialize_fallback_batches = 0
        self._nonvisual_prepared_batch_noop_replacements = 0
        self._completion_only_grad_fallback_batches = 0
        self._ddp_global_empty_batch_skips = 0
        self._all_empty_batch_skips = 0
        self._effective_update_steps = 0
        self._optimizer_step_skips = 0
        self._replay_fill_batches = 0
        self._replay_fill_episode_specs = 0
        self._groups_all_zero_advantage = 0
        self._skipped_non_finite_old_policy_samples = 0
        self._skipped_non_finite_compute_samples = 0
        self._zero_variance_group_count = 0
        self._zero_variance_rollout_count = 0
        self._zero_variance_skipped_count = 0
        self._groups_filtered_by_min_weight = 0
        self._advantage_source_counts: Dict[str, int] = {}
        self._sample_partition_counts: Dict[str, int] = {}
        self._native_rl_skip_next_optimizer_step = False
        self._native_rl_last_skip_reason = ""
        self._debug_compute_loss_call_index = 0
        self._debug_last_compute_loss_call_index = 0
        self._debug_last_compute_loss_prepared_batch_count = 0
        self._debug_last_compute_loss_rank_local_batch_count = 0
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
        )
        super().__init__(*trainer_args, **trainer_kwargs)
        if self.reference_model is not None:
            self.reference_model.eval()
            for parameter in self.reference_model.parameters():
                parameter.requires_grad_(False)

    def _debug_prepared_batch_summary(self, prepared_batch: Dict[str, Any]) -> Dict[str, Any]:
        completion_ids = prepared_batch.get("completion_ids")
        completion_shape = None
        if isinstance(completion_ids, torch.Tensor):
            completion_shape = tuple(int(dim) for dim in completion_ids.shape)
        return {
            "merge_signature": repr(self._prepared_batch_merge_signature(prepared_batch)),
            "completion_ids_shape": completion_shape,
        }

    def _prepare_rollout_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        prepared = copy.deepcopy(dict(item or {}))
        if not bool(self.strict_feature_guided_proposal):
            return prepared
        multimodal_cache = prepared.get("multimodal_cache")
        if not isinstance(multimodal_cache, dict):
            raise ValueError("Strict RL seek_evidence requires multimodal_cache on every rollout item.")
        if multimodal_cache.get("embedding") is None:
            raise ValueError("Strict RL seek_evidence requires feature_cache on every rollout item.")
        if self.proposal_runtime is None:
            raise ValueError("Strict RL seek_evidence requires proposal_runtime before rollout generation.")
        multimodal_cache["proposal_runtime"] = self.proposal_runtime
        multimodal_cache["strict_feature_guided_proposal"] = True
        prepared["multimodal_cache"] = multimodal_cache
        return prepared

    @staticmethod
    def _tensor_debug_summary(value: Any) -> Dict[str, Any]:
        if not isinstance(value, torch.Tensor):
            return {"type": type(value).__name__}
        tensor = value.detach()
        summary: Dict[str, Any] = {
            "shape": tuple(int(dim) for dim in tensor.shape),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "device": str(tensor.device),
            "requires_grad": bool(getattr(value, "requires_grad", False)),
            "numel": int(tensor.numel()),
        }
        if tensor.numel() <= 0:
            summary.update(
                {
                    "all_finite": True,
                    "nan_count": 0,
                    "posinf_count": 0,
                    "neginf_count": 0,
                }
            )
            return summary
        tensor_f = tensor.to(dtype=torch.float32)
        finite_mask = torch.isfinite(tensor_f)
        summary.update(
            {
                "all_finite": bool(torch.all(finite_mask).item()),
                "nan_count": int(torch.isnan(tensor_f).sum().item()),
                "posinf_count": int(torch.isposinf(tensor_f).sum().item()),
                "neginf_count": int(torch.isneginf(tensor_f).sum().item()),
            }
        )
        if bool(torch.any(finite_mask)):
            finite_values = tensor_f.masked_select(finite_mask)
            summary.update(
                {
                    "min": float(finite_values.min().item()),
                    "max": float(finite_values.max().item()),
                    "mean": float(finite_values.mean().item()),
                }
            )
        return summary

    def _non_finite_dump_dir(self) -> Path:
        output_dir = str(getattr(getattr(self, "args", None), "output_dir", "") or "").strip()
        base_dir = Path(output_dir) if output_dir else Path.cwd()
        dump_dir = base_dir / "non_finite_dumps"
        dump_dir.mkdir(parents=True, exist_ok=True)
        return dump_dir

    def _raise_non_finite_training_error(
        self,
        *,
        stage: str,
        tensor_name: str,
        tensor_value: Any,
        batch: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        runtime = distributed_runtime_from_env()
        batch_summary = None
        if isinstance(batch, dict):
            batch_summary = {
                "merge_signature": repr(self._prepared_batch_merge_signature(batch)),
                "prompt_ids": self._tensor_debug_summary(batch.get("prompt_ids")),
                "completion_ids": self._tensor_debug_summary(batch.get("completion_ids")),
                "completion_mask": self._tensor_debug_summary(batch.get("completion_mask")),
                "token_loss_weight": self._tensor_debug_summary(batch.get("token_loss_weight")),
                "advantage": self._tensor_debug_summary(batch.get("advantage")),
                "sample_loss_multiplier": self._tensor_debug_summary(batch.get("sample_loss_multiplier")),
                "old_policy_token_log_probs": self._tensor_debug_summary(batch.get("old_policy_token_log_probs")),
                "multimodal_input_keys": sorted(self._episode_spec_multimodal_inputs(batch).keys()),
            }
        payload = {
            "timestamp": utc_timestamp(),
            "stage": str(stage or ""),
            "tensor_name": str(tensor_name or ""),
            "tensor_summary": self._tensor_debug_summary(tensor_value),
            "compute_loss_call": int(getattr(self, "_debug_last_compute_loss_call_index", 0)),
            "local_prepared_batch_count": int(getattr(self, "_debug_last_compute_loss_rank_local_batch_count", 0)),
            "runtime": {
                "local_rank": int(getattr(runtime, "local_rank", 0) or 0),
                "global_rank": int(getattr(runtime, "global_rank", 0) or 0),
                "world_size": int(getattr(runtime, "world_size", 1) or 1),
            },
            "batch_summary": batch_summary,
            "extra": dict(extra or {}),
        }
        dump_path = self._non_finite_dump_dir() / (
            f"{utc_timestamp()}_{str(stage or 'unknown')}_"
            f"rank{int(getattr(runtime, 'local_rank', 0) or 0)}_"
            f"call{int(getattr(self, '_debug_last_compute_loss_call_index', 0))}.json"
        )
        write_json(dump_path, payload)
        runtime_log(
            (
                "trainer-native RL detected non-finite tensor and aborted: "
                f"stage={str(stage or '')} tensor={str(tensor_name or '')} dump={dump_path}"
            ),
            runtime=runtime,
            main_process_only=False,
        )
        raise RuntimeError(
            "Active RL encountered a non-finite tensor and aborted to prevent poisoned training: "
            f"stage={str(stage or '')} tensor={str(tensor_name or '')} dump={dump_path}"
        )

    def _assert_finite_tensor(
        self,
        *,
        stage: str,
        tensor_name: str,
        tensor_value: Any,
        batch: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not isinstance(tensor_value, torch.Tensor):
            return
        if bool(torch.all(torch.isfinite(tensor_value.detach())).item()):
            return
        self._raise_non_finite_training_error(
            stage=stage,
            tensor_name=tensor_name,
            tensor_value=tensor_value,
            batch=batch,
            extra=extra,
        )

    def _iter_named_trainable_tensors(
        self,
        model: Any,
        *,
        tensor_kind: str,
    ):
        unwrapped_model = _unwrap_model(model)
        named_parameters = getattr(unwrapped_model, "named_parameters", None)
        if not callable(named_parameters):
            return
        for name, parameter in named_parameters():
            if not isinstance(parameter, torch.Tensor) or not bool(getattr(parameter, "requires_grad", False)):
                continue
            tensor_value = None
            if str(tensor_kind or "").strip() == "params":
                tensor_value = parameter.detach()
            elif str(tensor_kind or "").strip() == "grads":
                grad_value = getattr(parameter, "grad", None)
                if isinstance(grad_value, torch.Tensor):
                    tensor_value = grad_value.detach()
            if not isinstance(tensor_value, torch.Tensor):
                continue
            yield str(name or ""), tensor_value

    @staticmethod
    def _named_tensor_non_finite_summary(name: str, tensor: torch.Tensor) -> Dict[str, Any]:
        value = tensor.detach()
        if bool(getattr(value, "is_sparse", False)):
            value = value.coalesce().values()
        summary: Dict[str, Any] = {
            "name": str(name or ""),
            "shape": tuple(int(dim) for dim in value.shape),
            "dtype": str(value.dtype).replace("torch.", ""),
            "device": str(value.device),
            "numel": int(value.numel()),
            "layout": str(value.layout),
        }
        if value.numel() <= 0:
            summary.update(
                {
                    "all_finite": True,
                    "nan_count": 0,
                    "posinf_count": 0,
                    "neginf_count": 0,
                }
            )
            return summary
        finite_mask = torch.isfinite(value)
        summary.update(
            {
                "all_finite": bool(torch.all(finite_mask).item()),
                "nan_count": int(torch.isnan(value).sum().item()),
                "posinf_count": int(torch.isposinf(value).sum().item()),
                "neginf_count": int(torch.isneginf(value).sum().item()),
            }
        )
        return summary

    def _scan_non_finite_trainable_tensors(
        self,
        model: Any,
        *,
        tensor_kind: str,
        max_entries: int = 8,
    ) -> Dict[str, Any]:
        checked_count = 0
        non_finite_count = 0
        entries: List[Dict[str, Any]] = []
        for name, tensor in self._iter_named_trainable_tensors(model, tensor_kind=tensor_kind) or []:
            checked_count += 1
            if bool(torch.all(torch.isfinite(tensor)).item()):
                continue
            non_finite_count += 1
            if len(entries) < max(1, int(max_entries)):
                entries.append(self._named_tensor_non_finite_summary(name=name, tensor=tensor))
        return {
            "tensor_kind": str(tensor_kind or ""),
            "checked_count": int(checked_count),
            "non_finite_count": int(non_finite_count),
            "entries": entries,
        }

    def _raise_non_finite_trainable_tensors_error(
        self,
        *,
        stage: str,
        model: Any,
        tensor_kind: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        scan = self._scan_non_finite_trainable_tensors(model, tensor_kind=tensor_kind)
        if int(scan.get("non_finite_count", 0)) <= 0:
            return
        runtime = distributed_runtime_from_env()
        payload = {
            "timestamp": utc_timestamp(),
            "stage": str(stage or ""),
            "tensor_kind": str(tensor_kind or ""),
            "compute_loss_call": int(getattr(self, "_debug_last_compute_loss_call_index", 0)),
            "local_prepared_batch_count": int(getattr(self, "_debug_last_compute_loss_rank_local_batch_count", 0)),
            "runtime": {
                "local_rank": int(getattr(runtime, "local_rank", 0) or 0),
                "global_rank": int(getattr(runtime, "global_rank", 0) or 0),
                "world_size": int(getattr(runtime, "world_size", 1) or 1),
            },
            "trainer_state": {
                "global_step": int(getattr(getattr(self, "state", None), "global_step", 0) or 0),
                "epoch": float(getattr(getattr(self, "state", None), "epoch", 0.0) or 0.0),
            },
            "scan": scan,
            "extra": dict(extra or {}),
        }
        dump_path = self._non_finite_dump_dir() / (
            f"{utc_timestamp()}_{str(stage or 'non_finite_model_state')}_"
            f"rank{int(getattr(runtime, 'local_rank', 0) or 0)}_"
            f"call{int(getattr(self, '_debug_last_compute_loss_call_index', 0))}.json"
        )
        write_json(dump_path, payload)
        runtime_log(
            (
                "trainer-native RL detected non-finite trainable tensors and aborted: "
                f"stage={str(stage or '')} tensor_kind={str(tensor_kind or '')} dump={dump_path}"
            ),
            runtime=runtime,
            main_process_only=False,
        )
        raise RuntimeError(
            "Active RL detected non-finite trainable tensors and aborted to preserve forensics: "
            f"stage={str(stage or '')} tensor_kind={str(tensor_kind or '')} dump={dump_path}"
        )

    def _write_deepspeed_flat_master_forensics_dump(
        self,
        *,
        stage: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        candidates = []
        deepspeed_engine = getattr(self, "deepspeed", None)
        if deepspeed_engine is not None:
            candidates.append(deepspeed_engine)
        model_wrapped = getattr(self, "model_wrapped", None)
        if model_wrapped is not None and model_wrapped is not deepspeed_engine:
            candidates.append(model_wrapped)
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None:
            candidates.append(optimizer)

        for candidate in candidates:
            candidate_optimizer = getattr(candidate, "optimizer", None)
            if isinstance(candidate_optimizer, _NativeRLOptimizerStepProxy):
                candidate_optimizer = getattr(candidate_optimizer, "_optimizer", None)
            if candidate_optimizer is None:
                continue
            dump_path = _write_flat_master_forensics_dump(
                trainer=self,
                optimizer=candidate_optimizer,
                stage=stage,
                extra=extra,
            )
            if dump_path is not None:
                return dump_path
        return None

    def _assert_finite_trainable_parameters(
        self,
        *,
        stage: str,
        model: Any,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._raise_non_finite_trainable_tensors_error(
            stage=stage,
            model=model,
            tensor_kind="params",
            extra=extra,
        )

    def _assert_finite_trainable_gradients(
        self,
        *,
        stage: str,
        model: Any,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._raise_non_finite_trainable_tensors_error(
            stage=stage,
            model=model,
            tensor_kind="grads",
            extra=extra,
        )

    def _old_policy_prefill_entry_debug_rows(
        self,
        episode_entries: Optional[Sequence[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for row_index, entry in enumerate(list(episode_entries or [])):
            if not isinstance(entry, dict):
                continue
            debug_plan = dict(entry.get("episode_debug_metadata") or {})
            episode_spec = dict(entry.get("episode_spec") or {})
            completion_ids = episode_spec.get("completion_ids")
            prompt_ids = episode_spec.get("prompt_ids")
            rows.append(
                {
                    "row_index": int(row_index),
                    "video_id": str(entry.get("video_id") or ""),
                    "group_id": str(entry.get("group_id") or ""),
                    "generation_id": int(entry.get("generation_id", -1) or -1),
                    "sample_partition": str(entry.get("sample_partition") or ""),
                    "sample_partition_type": str(entry.get("sample_partition_type") or ""),
                    "advantage_source": str(entry.get("advantage_source") or ""),
                    "message_plan": copy.deepcopy(list(debug_plan.get("message_plan") or [])),
                    "assistant_supervision": copy.deepcopy(list(debug_plan.get("assistant_supervision") or [])),
                    "retained_image_provenance": copy.deepcopy(list(debug_plan.get("retained_image_provenance") or [])),
                    "retained_message_count": int(debug_plan.get("retained_message_count") or 0),
                    "prompt_ids_shape": (
                        tuple(int(dim) for dim in prompt_ids.shape) if isinstance(prompt_ids, torch.Tensor) else None
                    ),
                    "completion_ids_shape": (
                        tuple(int(dim) for dim in completion_ids.shape)
                        if isinstance(completion_ids, torch.Tensor)
                        else None
                    ),
                }
            )
        return rows

    def _summarize_old_policy_prefill_model_inputs(
        self,
        model_inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        summaries: Dict[str, Any] = {}
        for key in ("input_ids", "attention_mask"):
            if key in model_inputs:
                summaries[key] = self._tensor_debug_summary(model_inputs.get(key))
        for key, value in sorted(dict(model_inputs or {}).items()):
            key_name = str(key)
            if key_name in summaries:
                continue
            if self._is_visual_multimodal_tensor_key(key_name):
                summaries[key_name] = self._tensor_debug_summary(value)
        return summaries

    def _write_old_policy_prefill_debug_dump(
        self,
        *,
        stage: str,
        prepared_batch: Dict[str, Any],
        model_inputs: Dict[str, Any],
        episode_entries: Optional[Sequence[Dict[str, Any]]],
        error: Optional[BaseException] = None,
        tensor_name: str = "",
        tensor_value: Any = None,
    ) -> Path:
        runtime = distributed_runtime_from_env()
        payload: Dict[str, Any] = {
            "timestamp": utc_timestamp(),
            "stage": str(stage or ""),
            "tensor_name": str(tensor_name or ""),
            "compute_loss_call": int(getattr(self, "_debug_last_compute_loss_call_index", 0)),
            "local_prepared_batch_count": int(getattr(self, "_debug_last_compute_loss_rank_local_batch_count", 0)),
            "runtime": {
                "local_rank": int(getattr(runtime, "local_rank", 0) or 0),
                "global_rank": int(getattr(runtime, "global_rank", 0) or 0),
                "world_size": int(getattr(runtime, "world_size", 1) or 1),
            },
            "prepared_batch_summary": {
                "merge_signature": repr(self._prepared_batch_merge_signature(prepared_batch)),
                "prompt_ids": self._tensor_debug_summary(prepared_batch.get("prompt_ids")),
                "completion_ids": self._tensor_debug_summary(prepared_batch.get("completion_ids")),
                "completion_mask": self._tensor_debug_summary(prepared_batch.get("completion_mask")),
            },
            "model_input_summaries": self._summarize_old_policy_prefill_model_inputs(model_inputs),
            "episode_entries": self._old_policy_prefill_entry_debug_rows(episode_entries),
        }
        if isinstance(tensor_value, torch.Tensor):
            payload["tensor_summary"] = self._tensor_debug_summary(tensor_value)
        if error is not None:
            payload["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        dump_path = self._non_finite_dump_dir() / (
            f"{utc_timestamp()}_{str(stage or 'old_policy_prefill')}_"
            f"rank{int(getattr(runtime, 'local_rank', 0) or 0)}_"
            f"call{int(getattr(self, '_debug_last_compute_loss_call_index', 0))}.json"
        )
        write_json(dump_path, payload)
        return dump_path

    def _assert_finite_old_policy_prefill_inputs(
        self,
        *,
        prepared_batch: Dict[str, Any],
        model_inputs: Dict[str, Any],
        episode_entries: Optional[Sequence[Dict[str, Any]]],
    ) -> None:
        for key in ("input_ids", "attention_mask"):
            value = model_inputs.get(key)
            if not isinstance(value, torch.Tensor):
                continue
            if bool(torch.all(torch.isfinite(value.detach())).item()):
                continue
            dump_path = self._write_old_policy_prefill_debug_dump(
                stage="old_policy_prefill_inputs_non_finite",
                prepared_batch=prepared_batch,
                model_inputs=model_inputs,
                episode_entries=episode_entries,
                tensor_name=key,
                tensor_value=value,
            )
            raise RuntimeError(
                "Active RL old-policy prefill received a non-finite tensor input and aborted: "
                f"tensor={key} dump={dump_path}"
            )
        for key, value in sorted(dict(model_inputs or {}).items()):
            key_name = str(key)
            if key_name in {"input_ids", "attention_mask"}:
                continue
            if not self._is_visual_multimodal_tensor_key(key_name) or not isinstance(value, torch.Tensor):
                continue
            if bool(torch.all(torch.isfinite(value.detach())).item()):
                continue
            dump_path = self._write_old_policy_prefill_debug_dump(
                stage="old_policy_prefill_inputs_non_finite",
                prepared_batch=prepared_batch,
                model_inputs=model_inputs,
                episode_entries=episode_entries,
                tensor_name=key_name,
                tensor_value=value,
            )
            raise RuntimeError(
                "Active RL old-policy prefill received a non-finite multimodal tensor and aborted: "
                f"tensor={key_name} dump={dump_path}"
            )

    def _non_finite_skip_summary_path(self) -> Path:
        return self._non_finite_dump_dir() / "skipped_non_finite_samples.jsonl"

    @staticmethod
    def _is_skippable_sample_exception(exc: BaseException) -> bool:
        del exc
        return False

    @staticmethod
    def _skippable_sample_exception_reason(exc: BaseException) -> str:
        if isinstance(exc, torch.OutOfMemoryError):
            return "oom"
        message = str(exc or "").lower()
        if "non-finite" in message or "non finite" in message or "nan" in message or "inf" in message:
            return "non_finite"
        return type(exc).__name__

    def _write_skipped_sample_dump(
        self,
        *,
        stage: str,
        episode_entry: Dict[str, Any],
        error: BaseException,
        prepared_batch: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        runtime = distributed_runtime_from_env()
        payload: Dict[str, Any] = {
            "timestamp": utc_timestamp(),
            "stage": str(stage or ""),
            "reason": self._skippable_sample_exception_reason(error),
            "compute_loss_call": int(getattr(self, "_debug_last_compute_loss_call_index", 0)),
            "local_prepared_batch_count": int(getattr(self, "_debug_last_compute_loss_rank_local_batch_count", 0)),
            "runtime": {
                "local_rank": int(getattr(runtime, "local_rank", 0) or 0),
                "global_rank": int(getattr(runtime, "global_rank", 0) or 0),
                "world_size": int(getattr(runtime, "world_size", 1) or 1),
            },
            "episode_entry": (
                self._old_policy_prefill_entry_debug_rows([episode_entry])[0]
                if isinstance(episode_entry, dict)
                else {}
            ),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "extra": dict(extra or {}),
        }
        if isinstance(prepared_batch, dict):
            payload["prepared_batch_summary"] = {
                "merge_signature": repr(self._prepared_batch_merge_signature(prepared_batch)),
                "prompt_ids": self._tensor_debug_summary(prepared_batch.get("prompt_ids")),
                "completion_ids": self._tensor_debug_summary(prepared_batch.get("completion_ids")),
                "completion_mask": self._tensor_debug_summary(prepared_batch.get("completion_mask")),
                "token_loss_weight": self._tensor_debug_summary(prepared_batch.get("token_loss_weight")),
                "old_policy_token_log_probs": self._tensor_debug_summary(prepared_batch.get("old_policy_token_log_probs")),
                "multimodal_input_keys": sorted(self._episode_spec_multimodal_inputs(prepared_batch).keys()),
            }
        dump_path = self._non_finite_dump_dir() / (
            f"{utc_timestamp()}_{str(stage or 'skipped_sample')}_"
            f"rank{int(getattr(runtime, 'local_rank', 0) or 0)}_"
            f"call{int(getattr(self, '_debug_last_compute_loss_call_index', 0))}_"
            f"gen{int(episode_entry.get('generation_id', -1) or -1)}.json"
        )
        write_json(dump_path, payload)
        append_jsonl(self._non_finite_skip_summary_path(), payload)
        return dump_path

    def _classify_rollout_partition(self, rollout: Dict[str, Any]) -> str:
        return _resolve_sample_partition(rollout)

    def _partition_loss_multiplier(
        self,
        partition: str,
        *,
        reward_summary: Dict[str, Any],
    ) -> float:
        return _resolve_sample_partition_multiplier(
            sample_partition=str(partition),
            reward_summary=reward_summary,
            sample_partition_multipliers=getattr(self, "_sample_partition_multipliers", None),
        )

    def _apply_zero_variance_advantage_fallback(
        self,
        rollouts: Sequence[Dict[str, Any]],
        *,
        eps: float = 1e-6,
    ) -> List[Dict[str, Any]]:
        updated: List[Dict[str, Any]] = []
        if not rollouts:
            return updated
        group_std = float((rollouts[0] or {}).get("group_reward_std") or 0.0)
        zero_variance_group = group_std <= float(eps)
        for rollout in rollouts:
            enriched = dict(rollout)
            partition = str(enriched.get("sample_partition") or self._classify_rollout_partition(enriched))
            advantage = float(enriched.get("group_advantage") or 0.0)
            advantage_source = str(enriched.get("advantage_source") or "group_relative")
            if zero_variance_group:
                advantage = 0.0
                if partition == "easy_normal":
                    advantage_source = "zero_easy_normal"
                else:
                    advantage_source = "zero_variance_skip"
            enriched["group_advantage"] = round(float(advantage), 6)
            enriched["sample_partition"] = str(partition)
            enriched["sample_partition_type"] = str(partition)
            enriched["advantage_source"] = str(advantage_source)
            enriched["zero_variance_group"] = bool(zero_variance_group)
            updated.append(enriched)
        return updated

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
            _seed_worker = _build_seed_worker_init_fn(args=self.args)
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
            shuffle=True,
            seed=getattr(self.args, "seed", None),
            sort_key_fn=_raw_item_workload_sort_key,
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
            "reward_protocol_finalize": [],
            "reward_stage_necessity": [],
            "reward_query_alignment": [],
            "reward_efficiency": [],
        }

    def _new_runtime_stats(self) -> Dict[str, int]:
        return {
            "raw_local_episode_spec_count": 0,
            "raw_local_prepared_batch_count": 0,
            "raw_local_sample_count": 0,
            "groups_filtered_by_min_weight": 0,
            "groups_all_zero_advantage": 0,
            "zero_variance_group_count": 0,
            "zero_variance_rollout_count": 0,
            "zero_variance_skipped_count": 0,
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

    def _can_reuse_current_policy_as_old_logprobs(self) -> bool:
        steps_per_generation = int(getattr(self, "steps_per_generation", 1) or 1)
        explicit_setting = getattr(self, "allow_reuse_current_policy_as_old_logprobs", None)
        if explicit_setting is None:
            return steps_per_generation <= 1
        if not bool(explicit_setting):
            return False
        gradient_accumulation_steps = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)
        return steps_per_generation <= max(1, gradient_accumulation_steps)

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
        _log_rollout_reward_diagnostics(
            video_id=video_id,
            rollouts=scored_rollouts,
            min_weight=float(self.min_weight),
        )
        if scored_rollouts and bool((scored_rollouts[0] or {}).get("zero_variance_group")):
            runtime_stats["zero_variance_group_count"] += 1
            runtime_stats["zero_variance_rollout_count"] += int(len(scored_rollouts))
            runtime_stats["zero_variance_skipped_count"] += int(len(scored_rollouts))
        item_materialization_total = 0
        item_materialized_completed = 0
        for rollout in scored_rollouts:
            sample_partition = str(
                rollout.get("sample_partition")
                or rollout.get("sample_partition_type")
                or self._classify_rollout_partition(rollout)
            )
            advantage_source = str(rollout.get("advantage_source") or "group_relative")
            sample_partition_counts = getattr(self, "_sample_partition_counts", None)
            if isinstance(sample_partition_counts, dict):
                sample_partition_counts[sample_partition] = int(sample_partition_counts.get(sample_partition, 0)) + 1
            advantage_source_counts = getattr(self, "_advantage_source_counts", None)
            if isinstance(advantage_source_counts, dict):
                advantage_source_counts[advantage_source] = int(advantage_source_counts.get(advantage_source, 0)) + 1
            reward_summary = dict(rollout.get("reward_summary") or {})
            components = dict(reward_summary.get("components") or {})
            rollout_metrics["reward_total"].append(_safe_float(reward_summary.get("total_reward")))
            rollout_metrics["reward_accuracy"].append(_safe_float(components.get("accuracy_reward")))
            rollout_metrics["reward_protocol_finalize"].append(
                _safe_float(components.get("protocol_finalize_reward"))
            )
            rollout_metrics["reward_stage_necessity"].append(_safe_float(components.get("stage_necessity_reward")))
            rollout_metrics["reward_query_alignment"].append(_safe_float(components.get("query_alignment_reward")))
            rollout_metrics["reward_efficiency"].append(_safe_float(components.get("efficiency_reward")))
            rollout_advantage = abs(float(rollout.get("group_advantage", 0.0) or 0.0))
            if bool(rollout.get("zero_variance_group")):
                runtime_stats["groups_all_zero_advantage"] += 1
                continue
            if rollout_advantage < float(self.min_weight):
                runtime_stats["groups_filtered_by_min_weight"] += 1
                if rollout_advantage <= 0.0:
                    runtime_stats["groups_all_zero_advantage"] += 1
                continue
            item_materialization_total += 1
            if progress is not None:
                progress.extend_materialization_total(1)
            episode_spec_entry = self._build_episode_spec_entry_from_rollout(rollout)
            item_materialized_completed += 1
            if progress is not None:
                progress.advance_materialization(
                    video_id=str(rollout.get("video_id") or video_id),
                    completed=item_materialized_completed,
                    total=item_materialization_total,
                )
            if episode_spec_entry is not None:
                item_episode_specs.append(episode_spec_entry)
        if item_episode_specs and not self._can_reuse_current_policy_as_old_logprobs():
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
                self._zero_variance_group_count += int(runtime_stats.get("zero_variance_group_count", 0))
                self._zero_variance_rollout_count += int(runtime_stats.get("zero_variance_rollout_count", 0))
                self._zero_variance_skipped_count += int(runtime_stats.get("zero_variance_skipped_count", 0))
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

    def _build_rollout_policy(self, model: Any) -> QwenGenerationPolicy:
        return self._build_policy(
            model,
            use_generation_cache=self.rollout_use_generation_cache,
        )

    def _generate_scored_rollouts(
        self,
        item: Dict[str, Any],
        model: Any,
        *,
        progress: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        rollout_policy = self._build_rollout_policy(model)
        rollouts: List[Dict[str, Any]] = []
        video_id = str(item.get("video_id") or "")
        rollout_items = [self._prepare_rollout_item(item) for _ in range(self.num_generations)]
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
        # Batched semantic judge cache preload (2026-04-21): gather every
        # (question, reference, prediction) triple across this group's
        # rollouts and score them in ONE judge.score_batch() call. Results
        # land in the judge's in-memory cache so the per-rollout
        # score_rollout_trace() calls below hit cache instead of issuing
        # individual judge calls. Harmless best-effort warm-up: any failure
        # falls through to the existing per-item path. Requires scoring
        # metadata to be pre-attached; we setdefault it here so the main
        # loop below still works unchanged.
        _judge = getattr(self, "reward_judge", None)
        if _judge is not None and hasattr(_judge, "score_batch"):
            _flat_queries: List[Tuple[str, str, str]] = []
            for _rollout in generated_rollouts:
                if isinstance(item.get("structured_target"), dict):
                    _rollout.setdefault("scoring_target", copy.deepcopy(item["structured_target"]))
                if isinstance(item.get("qa_pairs"), list):
                    _rollout.setdefault("scoring_qa_pairs", copy.deepcopy(item.get("qa_pairs") or []))
                _evidence_preload = item.get("evidence") or {}
                if isinstance(_evidence_preload, dict) and isinstance(_evidence_preload.get("evidence_moments"), list):
                    _rollout.setdefault(
                        "scoring_evidence_moments",
                        copy.deepcopy(_evidence_preload.get("evidence_moments") or []),
                    )
                try:
                    _qmap = _collect_semantic_queries(_rollout, reward_version=self.reward_version)
                except Exception:
                    _qmap = {}
                for _triple in _qmap.values():
                    _flat_queries.append(_triple)
            if _flat_queries:
                try:
                    _judge.score_batch(_flat_queries)
                except Exception:
                    pass
        for generation_id, rollout in enumerate(generated_rollouts):
            if progress is not None:
                progress.advance_generation_stage(
                    video_id=str(rollout.get("video_id") or video_id),
                    generation_id=int(generation_id),
                    stage="rollout",
                )
            rollout["group_id"] = str(
                item.get("rl_instance_id")
                or item.get("video_id")
                or f"group_{generation_id}"
            )
            rollout["generation_id"] = int(generation_id)
            if item.get("source_video_id") is not None:
                rollout["source_video_id"] = str(item.get("source_video_id") or "")
            if isinstance(item.get("structured_target"), dict):
                rollout["scoring_target"] = copy.deepcopy(item["structured_target"])
            if isinstance(item.get("qa_pairs"), list):
                rollout["scoring_qa_pairs"] = copy.deepcopy(item.get("qa_pairs") or [])
            evidence = item.get("evidence") or {}
            if isinstance(evidence, dict) and isinstance(evidence.get("evidence_moments"), list):
                rollout["scoring_evidence_moments"] = copy.deepcopy(evidence.get("evidence_moments") or [])
            try:
                rollout["reward_summary"] = score_rollout_trace(
                    rollout,
                    reward_version=self.reward_version,
                    reward_config=self.reward_config,
                    llm_judge=self.reward_judge,
                )
            finally:
                if progress is not None:
                    progress.advance_generation_stage(
                        video_id=str(rollout.get("video_id") or video_id),
                        generation_id=int(generation_id),
                        stage="score",
                    )
            rollouts.append(rollout)
        rollouts = _compute_group_relative_advantages(
            rollouts,
            clip_value=self.advantage_clip,
            sample_partition_multipliers=getattr(self, "_sample_partition_multipliers", None),
        )
        return self._apply_zero_variance_advantage_fallback(rollouts)

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
            max_tool_message_frames=self.max_tool_message_frames,
            max_total_video_frames=self.max_total_video_frames,
            max_seq_length=self.max_seq_length,
            keep_recent_text_messages=self.keep_recent_text_messages,
        )

    def _build_episode_spec_entry_from_rollout(
        self,
        rollout: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        rollout_advantage = float(rollout.get("group_advantage", 0.0) or 0.0)
        existing_feature = rollout.get("_rl_episode_training_feature")
        if isinstance(existing_feature, dict) and _is_episode_feature_from_feature(existing_feature):
            message_feature = existing_feature
        else:
            message_feature = _build_episode_training_feature(
                result=rollout,
                require_message_supervision=True,
            )
        if not isinstance(message_feature, dict):
            raise ValueError(
                "Active native RL expected a message-only supervision feature for a rollout with valid turns."
            )
        if not _is_episode_feature_from_feature(message_feature):
            raise ValueError(
                "Active native RL requires message-only `messages` + `assistant_supervision` on each rollout."
            )
        if not list(message_feature.get("messages") or []) or not list(message_feature.get("assistant_supervision") or []):
            raise ValueError(
                "Active native RL requires non-empty `messages` and `assistant_supervision` on each rollout."
            )
        feature = dict(message_feature)
        reward_summary = dict(rollout.get("reward_summary") or {})
        partition = str(
            rollout.get("sample_partition")
            or rollout.get("sample_partition_type")
            or self._classify_rollout_partition(rollout)
        )
        partition_loss_multiplier = self._partition_loss_multiplier(
            partition,
            reward_summary=reward_summary,
        )
        effective_loss_multiplier = float(partition_loss_multiplier)
        feature["sample_weight"] = float(rollout_advantage)
        feature["advantage"] = float(rollout_advantage)
        feature["sample_loss_multiplier"] = float(effective_loss_multiplier)
        result = self._build_episode_spec(feature)
        self._budgeting_stats.record(result)
        if result.batch is None:
            runtime_log(
                "rl episode spec drop debug: "
                f"video_id={str(rollout.get('video_id') or '') or 'unknown'} "
                f"generation_id={int(rollout.get('generation_id', -1) or -1)} "
                f"group_advantage={float(rollout_advantage):.6f} "
                f"sample_partition_type={partition} "
                f"advantage_source={str(rollout.get('advantage_source') or 'group_relative')} "
                f"normal_case_type={str(reward_summary.get('normal_case_type') or '')} "
                f"sample_loss_multiplier={float(effective_loss_multiplier):.6f} "
                f"drop_reason={str(result.drop_reason or 'unknown')} "
                f"completion_token_count={int(getattr(result, 'completion_token_count', 0) or 0)} "
                f"has_messages={bool(isinstance(feature.get('messages'), list) and feature.get('messages'))} "
                f"message_count={int(len(list(feature.get('messages') or [])))} "
                f"assistant_supervision_count={int(len(list(feature.get('assistant_supervision') or [])))}",
                runtime=distributed_runtime_from_env(),
                main_process_only=False,
            )
            return None
        return {
            "episode_spec": result.batch,
            "video_id": str(rollout.get("video_id") or ""),
            "group_id": str(rollout.get("group_id") or rollout.get("video_id") or ""),
            "generation_id": int(rollout.get("generation_id", -1) or -1),
            "advantage": float(rollout_advantage),
            "sample_partition": partition,
            "sample_partition_type": partition,
            "sample_partition_multiplier": float(effective_loss_multiplier),
            "advantage_source": str(rollout.get("advantage_source") or "group_relative"),
            "reward_summary": copy.deepcopy(rollout.get("reward_summary") or {}),
            "episode_debug_metadata": copy.deepcopy(dict(result.cached_plan or {})),
        }

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
            "token_loss_weight",
            "sample_loss_multiplier",
            "multimodal_inputs",
            "_source_episode_entries",
        }

    def _episode_spec_multimodal_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if "multimodal_inputs" in batch:
            multimodal_inputs = batch.get("multimodal_inputs")
            if multimodal_inputs is None:
                return {}
            if isinstance(multimodal_inputs, dict):
                return multimodal_inputs
            if isinstance(multimodal_inputs, list):
                return self._collate_multimodal_input_samples(multimodal_inputs)
            raise ValueError("Prepared batch `multimodal_inputs` must be dict or list[dict].")
        return {
            key: value
            for key, value in batch.items()
            if key not in self._reserved_episode_spec_keys()
        }

    def _is_visual_multimodal_tensor_key(self, key_path: str) -> bool:
        key_name = str(key_path or "").split(".")[-1]
        return key_name in {
            "pixel_values",
            "pixel_values_videos",
            "image_grid_thw",
            "video_grid_thw",
            "second_per_grid_ts",
            "image_sizes",
        }

    def _multimodal_payload_has_visual_tensor(self, value: Any, *, key_path: str = "") -> bool:
        if isinstance(value, torch.Tensor):
            key_name = str(key_path or "").split(".")[-1]
            return key_name in {"pixel_values", "pixel_values_videos"} and int(value.numel()) > 0
        if isinstance(value, dict):
            return any(
                self._multimodal_payload_has_visual_tensor(
                    item,
                    key_path=f"{key_path}.{key}" if key_path else str(key),
                )
                for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(
                self._multimodal_payload_has_visual_tensor(
                    item,
                    key_path=f"{key_path}.{index}" if key_path else str(index),
                )
                for index, item in enumerate(value)
            )
        return False

    def _prepared_batch_has_visual_forward_payload(self, prepared_batch: Dict[str, Any]) -> bool:
        return self._multimodal_payload_has_visual_tensor(self._episode_spec_multimodal_inputs(prepared_batch))

    def _move_multimodal_payload_to_device(
        self,
        value: Any,
        *,
        device: torch.device,
        key_path: str = "",
    ) -> Any:
        if isinstance(value, torch.Tensor):
            target_dtype = value.dtype
            if (
                self._native_visual_tensor_dtype is not None
                and value.is_floating_point()
                and self._is_visual_multimodal_tensor_key(key_path)
                and value.dtype != self._native_visual_tensor_dtype
            ):
                target_dtype = self._native_visual_tensor_dtype
            if value.device != device or target_dtype != value.dtype:
                return value.to(device=device, dtype=target_dtype)
            return value
        if isinstance(value, dict):
            return {
                str(key): self._move_multimodal_payload_to_device(
                    item,
                    device=device,
                    key_path=f"{key_path}.{key}" if key_path else str(key),
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [
                self._move_multimodal_payload_to_device(item, device=device, key_path=key_path)
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                self._move_multimodal_payload_to_device(item, device=device, key_path=key_path)
                for item in value
            )
        return copy.deepcopy(value)

    def _move_episode_spec_to_device(
        self,
        episode_spec: Dict[str, Any],
        *,
        device: torch.device,
    ) -> Dict[str, Any]:
        prepared: Dict[str, Any] = {}
        reserved_keys = self._reserved_episode_spec_keys()
        for key, value in episode_spec.items():
            key_name = str(key)
            if key_name == "multimodal_inputs":
                prepared[key] = self._move_multimodal_payload_to_device(
                    value,
                    device=device,
                    key_path="multimodal_inputs",
                )
            elif key_name not in reserved_keys:
                if isinstance(value, torch.Tensor):
                    if (
                        self._native_visual_tensor_dtype is not None
                        and value.is_floating_point()
                        and self._is_visual_multimodal_tensor_key(key_name)
                        and value.dtype != self._native_visual_tensor_dtype
                    ):
                        prepared[key] = value.to(device=device, dtype=self._native_visual_tensor_dtype)
                    else:
                        prepared[key] = value if value.device == device else value.to(device=device)
                else:
                    prepared[key] = copy.deepcopy(value)
            elif isinstance(value, torch.Tensor):
                prepared[key] = value if value.device == device else value.to(device=device)
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
        cloned["_ddp_noop_padding"] = True
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

    def _pad_prepared_batches_to_distributed_max(
        self,
        prepared_batches: Sequence[Dict[str, Any]],
        *,
        device: torch.device,
        runtime_stats: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        padded_batches = [dict(batch) for batch in list(prepared_batches or [])]
        local_prepared_batch_count = int(len(padded_batches))
        min_prepared_batch_count = _distributed_min_int(local_prepared_batch_count, device=device)
        max_prepared_batch_count = _distributed_max_int(local_prepared_batch_count, device=device)
        runtime_stats["distributed_min_prepared_batch_count"] = int(min_prepared_batch_count)
        runtime_stats["distributed_max_prepared_batch_count"] = int(max_prepared_batch_count)
        runtime_log(
            "rl debug prepared batch padding: "
            f"raw_local_prepared_batch_count={int(local_prepared_batch_count)} "
            f"distributed_min_prepared_batch_count={int(min_prepared_batch_count)} "
            f"distributed_max_prepared_batch_count={int(max_prepared_batch_count)}",
            runtime=distributed_runtime_from_env(),
            main_process_only=False,
        )
        if int(max_prepared_batch_count) <= 0:
            return padded_batches
        padded_count = 0
        visual_replacement_count = 0
        for batch_position in range(int(max_prepared_batch_count)):
            has_local_batch = batch_position < int(local_prepared_batch_count)
            local_batch = padded_batches[batch_position] if has_local_batch else None
            local_has_visual_payload = bool(
                self._prepared_batch_has_visual_forward_payload(local_batch)
                if isinstance(local_batch, dict)
                else False
            )
            _, any_rank_has_visual_payload = _distributed_bool_consensus(
                local_has_visual_payload,
                device=device,
            )
            donor_source = None
            if isinstance(local_batch, dict) and (local_has_visual_payload or not any_rank_has_visual_payload):
                donor_source = self._prepared_batch_cpu_copy(local_batch)
            donor_prepared_batch = _distributed_first_available_object(
                donor_source,
                device=device,
            )
            if donor_prepared_batch is None:
                raise RuntimeError(
                    "Active native RL could not find a donor prepared batch for DDP noop padding: "
                    f"batch_position={int(batch_position)} local_prepared_batch_count={int(local_prepared_batch_count)} "
                    f"max_prepared_batch_count={int(max_prepared_batch_count)} "
                    f"any_rank_has_visual_payload={bool(any_rank_has_visual_payload)}"
                )
            if has_local_batch and (local_has_visual_payload or not any_rank_has_visual_payload):
                continue
            donor_prepared_batch = self._move_episode_spec_to_device(donor_prepared_batch, device=device)
            noop_batch = self._clone_prepared_batch_as_noop(donor_prepared_batch)
            if has_local_batch:
                padded_batches[batch_position] = noop_batch
                visual_replacement_count += 1
            else:
                padded_batches.append(noop_batch)
                padded_count += 1
        if padded_count > 0:
            runtime_stats["ddp_noop_padded_prepared_batches"] = int(
                runtime_stats.get("ddp_noop_padded_prepared_batches", 0)
            ) + int(padded_count)
        if visual_replacement_count > 0:
            runtime_stats["ddp_noop_replaced_nonvisual_prepared_batches"] = int(
                runtime_stats.get("ddp_noop_replaced_nonvisual_prepared_batches", 0)
            ) + int(visual_replacement_count)
            self._nonvisual_prepared_batch_noop_replacements += int(visual_replacement_count)
            runtime_log(
                "rl debug prepared batch visual branch padding: "
                f"replaced_nonvisual_prepared_batches={int(visual_replacement_count)} "
                f"raw_local_prepared_batch_count={int(local_prepared_batch_count)} "
                f"distributed_max_prepared_batch_count={int(max_prepared_batch_count)}",
                runtime=distributed_runtime_from_env(),
                main_process_only=False,
            )
        return padded_batches

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

    def _sample_loss_weight_summary(
        self,
        batch: Dict[str, Any],
        *,
        device: torch.device,
        sample_count: int,
    ) -> Tuple[torch.Tensor, int, float]:
        multiplier = self._sample_loss_multiplier(
            batch,
            device=device,
            sample_count=sample_count,
        )
        active_mask = multiplier > 0
        active_count = int(active_mask.sum().item())
        effective_weight_sum = float(multiplier.clamp_min(0.0).sum().item())
        return multiplier, active_count, effective_weight_sum

    def _sequence_pad_values(self) -> Dict[str, Tuple[Any, str]]:
        return {
            "prompt_ids": (self._pad_token_id(), "left"),
            "prompt_mask": (0, "left"),
            "completion_ids": (self._pad_token_id(), "right"),
            "completion_mask": (0, "right"),
            "old_policy_token_log_probs": (0.0, "right"),
            "token_loss_weight": (0.0, "right"),
        }

    def _prepared_batch_merge_signature_entry(
        self,
        key: str,
        value: Any,
        prepared_batch: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        if key == "multimodal_inputs":
            if not isinstance(value, dict):
                raise ValueError("Expected dict values for prepared batch key 'multimodal_inputs'.")
            return self._multimodal_inputs_signature(value)
        if key == "_source_episode_entries":
            return ("debug_source_entries",)
        if key == "old_policy_token_log_probs" and value is None:
            return ("reuse_current_policy_logprobs",)
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

    def _multimodal_inputs_signature(self, value: Any) -> Tuple[Any, ...]:
        if isinstance(value, torch.Tensor):
            if value.ndim <= 0:
                return ("tensor_scalar", str(value.dtype))
            return ("tensor_cat_dim0", str(value.dtype), int(value.ndim), tuple(value.shape[1:]))
        if isinstance(value, dict):
            return (
                "dict",
                tuple(
                    (str(key), self._multimodal_inputs_signature(item))
                    for key, item in sorted(value.items())
                ),
            )
        if isinstance(value, list):
            return ("list", len(value), tuple(self._multimodal_inputs_signature(item) for item in value))
        if isinstance(value, tuple):
            return ("tuple", len(value), tuple(self._multimodal_inputs_signature(item) for item in value))
        return ("python", type(value).__name__, copy.deepcopy(value))

    def _merge_multimodal_input_samples(self, values: Sequence[Any]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        for value in values:
            if isinstance(value, dict):
                merged.append(copy.deepcopy(value))
                continue
            if isinstance(value, list):
                for item in value:
                    if not isinstance(item, dict):
                        raise ValueError("Expected multimodal_inputs list items to be dicts.")
                    merged.append(copy.deepcopy(item))
                continue
            raise ValueError("Expected multimodal_inputs values to be dict or list[dict].")
        return merged

    def _collate_multimodal_input_samples(self, sample_inputs: Sequence[Any]) -> Dict[str, Any]:
        normalized_samples: List[Dict[str, Any]] = []
        for value in sample_inputs:
            if isinstance(value, dict):
                normalized_samples.append(value)
            else:
                raise ValueError("Expected `multimodal_inputs` samples to be dicts.")
        if not normalized_samples:
            return {}
        first_keys = set(normalized_samples[0].keys())
        for sample in normalized_samples[1:]:
            if set(sample.keys()) != first_keys:
                raise ValueError("Cannot collate multimodal_inputs with inconsistent key sets.")
        return {
            str(key): self._collate_multimodal_input_value([sample[key] for sample in normalized_samples])
            for key in sorted(first_keys)
        }

    def _collate_multimodal_input_value(self, values: Sequence[Any]) -> Any:
        first_value = values[0]
        if isinstance(first_value, torch.Tensor):
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise ValueError("Cannot collate multimodal_inputs with inconsistent tensor/value types.")
            if len(values) == 1 and first_value.ndim > 0:
                return first_value
            if first_value.ndim == 0:
                return torch.stack(list(values), dim=0)
            return torch.cat(list(values), dim=0)
        if isinstance(first_value, dict):
            if len(values) == 1:
                return first_value
            first_keys = set(first_value.keys())
            for value in values[1:]:
                if not isinstance(value, dict) or set(value.keys()) != first_keys:
                    raise ValueError("Cannot collate multimodal_inputs dicts with inconsistent keys.")
            return {
                str(key): self._collate_multimodal_input_value([value[key] for value in values])
                for key in sorted(first_keys)
            }
        if isinstance(first_value, list):
            if len(values) == 1:
                return first_value
            if not all(isinstance(value, list) and len(value) == len(first_value) for value in values):
                raise ValueError("Cannot collate multimodal_inputs lists with inconsistent lengths.")
            return [
                self._collate_multimodal_input_value([value[index] for value in values])
                for index in range(len(first_value))
            ]
        if isinstance(first_value, tuple):
            if len(values) == 1:
                return first_value
            if not all(isinstance(value, tuple) and len(value) == len(first_value) for value in values):
                raise ValueError("Cannot collate multimodal_inputs tuples with inconsistent lengths.")
            return tuple(
                self._collate_multimodal_input_value([value[index] for value in values])
                for index in range(len(first_value))
            )
        if not all(value == first_value for value in values[1:]):
            raise ValueError("Cannot collate multimodal_inputs python values with inconsistent contents.")
        return copy.deepcopy(first_value)

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
        episode_entries: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> torch.Tensor:
        multimodal_inputs = self._episode_spec_multimodal_inputs(prepared_batch)
        model_inputs, logits_to_keep = build_completion_only_model_inputs(
            prompt_ids=prepared_batch["prompt_ids"],
            prompt_mask=prepared_batch["prompt_mask"],
            completion_ids=prepared_batch["completion_ids"],
            completion_mask=prepared_batch["completion_mask"],
            multimodal_inputs=multimodal_inputs,
        )
        self._assert_finite_old_policy_prefill_inputs(
            prepared_batch=prepared_batch,
            model_inputs=model_inputs,
            episode_entries=episode_entries,
        )
        with torch.inference_mode():
            try:
                old_policy_token_log_probs, _ = compute_completion_only_token_log_probs_from_prepared_inputs(
                    model=model,
                    model_inputs=model_inputs,
                    completion_ids=prepared_batch["completion_ids"],
                    completion_mask=prepared_batch["completion_mask"],
                    logits_to_keep=logits_to_keep,
                    temperature=self.policy_temperature,
                    log_runtime_details=True,
                )
            except Exception as exc:
                dump_path = self._write_old_policy_prefill_debug_dump(
                    stage="old_policy_prefill_forward_exception",
                    prepared_batch=prepared_batch,
                    model_inputs=model_inputs,
                    episode_entries=episode_entries,
                    error=exc,
                )
                runtime_log(
                    (
                        "trainer-native RL old-policy prefill failed and dumped context: "
                        f"dump={dump_path} error={type(exc).__name__}: {exc}"
                    ),
                    runtime=distributed_runtime_from_env(),
                    main_process_only=False,
                )
                raise
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
        retained_episode_specs: List[Dict[str, Any]] = []
        for bucket in grouped_episode_specs:
            prepared_batches = [
                self._move_episode_spec_to_device(entry["episode_spec"], device=target_device)
                for entry in bucket
            ]
            if len(prepared_batches) == 1:
                try:
                    batched_log_probs = self._compute_old_policy_token_log_probs_for_prepared_batch(
                        model,
                        prepared_batch=prepared_batches[0],
                        episode_entries=bucket,
                    )
                except Exception as exc:
                    if not self._is_skippable_sample_exception(exc):
                        raise
                    dump_path = self._write_skipped_sample_dump(
                        stage="old_policy_prefill_skipped_sample",
                        episode_entry=bucket[0],
                        prepared_batch=prepared_batches[0],
                        error=exc,
                    )
                    runtime_log(
                        (
                            "trainer-native RL skipped a sample during old-policy prefill: "
                            f"video_id={str(bucket[0].get('video_id') or '') or 'unknown'} "
                            f"generation_id={int(bucket[0].get('generation_id', -1) or -1)} "
                            f"dump={dump_path}"
                        ),
                        runtime=distributed_runtime_from_env(),
                        main_process_only=False,
                    )
                    self._skipped_non_finite_old_policy_samples += 1
                    continue
            else:
                try:
                    merged_batch = self._merge_prepared_batches(prepared_batches)
                except ValueError as exc:
                    if not self._is_merge_fallback_error(exc):
                        raise
                    merged_batch = None
                if merged_batch is not None:
                    try:
                        batched_log_probs = self._compute_old_policy_token_log_probs_for_prepared_batch(
                            model,
                            prepared_batch=merged_batch,
                            episode_entries=bucket,
                        )
                    except Exception as exc:
                        if not self._is_skippable_sample_exception(exc):
                            raise
                        merged_batch = None
                if merged_batch is None:
                    recovered_entries: List[Dict[str, Any]] = []
                    for entry, prepared_batch in zip(bucket, prepared_batches):
                        try:
                            entry["old_policy_token_log_probs"] = self._compute_old_policy_token_log_probs_for_prepared_batch(
                                model,
                                prepared_batch=prepared_batch,
                                episode_entries=[entry],
                            )
                        except Exception as single_exc:
                            if not self._is_skippable_sample_exception(single_exc):
                                raise
                            dump_path = self._write_skipped_sample_dump(
                                stage="old_policy_prefill_skipped_sample",
                                episode_entry=entry,
                                prepared_batch=prepared_batch,
                                error=single_exc,
                            )
                            runtime_log(
                                (
                                    "trainer-native RL skipped a sample during old-policy prefill recovery: "
                                    f"video_id={str(entry.get('video_id') or '') or 'unknown'} "
                                    f"generation_id={int(entry.get('generation_id', -1) or -1)} "
                                    f"dump={dump_path}"
                                ),
                                runtime=distributed_runtime_from_env(),
                                main_process_only=False,
                            )
                            self._skipped_non_finite_old_policy_samples += 1
                            continue
                        recovered_entries.append(entry)
                    retained_episode_specs.extend(recovered_entries)
                    continue
            for row_index, entry in enumerate(bucket):
                completion_length = int(entry["episode_spec"]["completion_ids"].shape[-1])
                entry["old_policy_token_log_probs"] = (
                    batched_log_probs[row_index : row_index + 1, :completion_length].clone()
                )
                retained_episode_specs.append(entry)
        return retained_episode_specs

    def _materialize_episode_spec(
        self,
        episode_spec: Dict[str, Any],
        *,
        device: torch.device,
    ) -> Dict[str, Any]:
        cached_episode_spec = episode_spec.get("episode_spec")
        if not isinstance(cached_episode_spec, dict):
            raise ValueError("Active native RL episode_spec entries must contain materialized `episode_spec`.")
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
        prepared_batch["_source_episode_entries"] = [copy.deepcopy(dict(episode_spec or {}))]
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
            if key == "multimodal_inputs":
                merged[key] = self._merge_multimodal_input_samples(values)
                continue
            if key == "_source_episode_entries":
                merged_entries: List[Dict[str, Any]] = []
                for value in values:
                    if not isinstance(value, list):
                        raise ValueError("Expected _source_episode_entries values to be list[dict].")
                    for entry in value:
                        if not isinstance(entry, dict):
                            raise ValueError("Expected _source_episode_entries items to be dicts.")
                        merged_entries.append(copy.deepcopy(entry))
                merged[key] = merged_entries
                continue
            if key == "old_policy_token_log_probs" and all(value is None for value in values):
                merged[key] = None
                continue
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

    def _prepared_batch_source_entries(self, prepared_batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        source_entries = prepared_batch.get("_source_episode_entries")
        if not isinstance(source_entries, list):
            return []
        return [dict(entry) for entry in source_entries if isinstance(entry, dict)]

    def _recover_sample_losses_from_source_entries(
        self,
        *,
        model: Any,
        batch: Dict[str, Any],
        batch_error: BaseException,
    ) -> Tuple[Optional[torch.Tensor], int, float]:
        source_entries = self._prepared_batch_source_entries(batch)
        if len(source_entries) <= 1:
            raise batch_error
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")
        recovered_losses: List[torch.Tensor] = []
        recovered_trainable_samples = 0
        recovered_effective_weight_sum = 0.0
        skipped_count = 0
        for source_entry in source_entries:
            prepared_single = self._materialize_episode_spec(source_entry, device=target_device)
            try:
                sample_losses = self._compute_sample_losses_for_batch(
                    model=model,
                    batch=prepared_single,
                )
            except Exception as single_exc:
                if not self._is_skippable_sample_exception(single_exc):
                    raise
                dump_path = self._write_skipped_sample_dump(
                    stage="compute_loss_skipped_sample",
                    episode_entry=source_entry,
                    prepared_batch=prepared_single,
                    error=single_exc,
                    extra={
                        "recovery_from_batch_error": {
                            "type": type(batch_error).__name__,
                            "message": str(batch_error),
                        }
                    },
                )
                runtime_log(
                    (
                        "trainer-native RL skipped a sample after compute_loss batch failure: "
                        f"video_id={str(source_entry.get('video_id') or '') or 'unknown'} "
                        f"generation_id={int(source_entry.get('generation_id', -1) or -1)} "
                        f"dump={dump_path}"
                    ),
                    runtime=distributed_runtime_from_env(),
                    main_process_only=False,
                )
                self._skipped_non_finite_compute_samples += 1
                skipped_count += 1
                continue
            if sample_losses is None or sample_losses.numel() <= 0:
                continue
            (
                _sample_loss_multiplier,
                active_sample_count,
                effective_weight_sum,
            ) = self._sample_loss_weight_summary(
                prepared_single,
                device=sample_losses.device,
                sample_count=int(sample_losses.numel()),
            )
            recovered_trainable_samples += int(active_sample_count)
            recovered_effective_weight_sum += float(effective_weight_sum)
            recovered_losses.append(sample_losses.view(-1))
        if recovered_losses:
            runtime_log(
                "trainer-native RL recovered compute_loss batch by skipping bad samples: "
                f"kept={int(sum(int(loss.numel()) for loss in recovered_losses))} "
                f"skipped={int(skipped_count)}",
                runtime=distributed_runtime_from_env(),
                main_process_only=False,
            )
            return (
                torch.cat(recovered_losses, dim=0),
                int(recovered_trainable_samples),
                float(recovered_effective_weight_sum),
            )
        runtime_log(
            "trainer-native RL dropped an entire compute_loss batch after sample-level recovery failed: "
            f"batch_error={type(batch_error).__name__}: {batch_error}",
            runtime=distributed_runtime_from_env(),
            main_process_only=False,
        )
        return None, 0, 0.0

    def _prepare_advantages(self, batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
        advantages = batch.get("advantage")
        if advantages is None:
            raise ValueError("Trainer-native GRPO requires singular `advantage` for every episode batch.")
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
        self._assert_finite_trainable_parameters(
            stage="pre_policy_forward_params",
            model=model,
            extra={
                "prompt_ids_shape": tuple(int(dim) for dim in batch["prompt_ids"].shape),
                "completion_ids_shape": tuple(int(dim) for dim in batch["completion_ids"].shape),
            },
        )
        policy_token_log_probs, response_mask = compute_completion_only_token_log_probs_from_ids(
            model=model,
            prompt_ids=batch["prompt_ids"],
            prompt_mask=batch["prompt_mask"],
            completion_ids=batch["completion_ids"],
            completion_mask=batch["completion_mask"],
            multimodal_inputs=self._episode_spec_multimodal_inputs(batch),
            temperature=self.policy_temperature,
        )
        self._assert_finite_tensor(
            stage="policy_token_log_probs",
            tensor_name="policy_token_log_probs",
            tensor_value=policy_token_log_probs,
            batch=batch,
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
            if not self._can_reuse_current_policy_as_old_logprobs():
                raise RuntimeError(
                    "Active RL requires cached per-token old_policy_token_log_probs for this configuration; "
                    "current-policy reuse is disabled."
                )
            old_policy_token_log_probs = policy_token_log_probs.detach()
        else:
            old_policy_token_log_probs = old_policy_token_log_probs.to(
                policy_token_log_probs.device,
                dtype=torch.float32,
            ).detach()
            if old_policy_token_log_probs.ndim == 1:
                old_policy_token_log_probs = old_policy_token_log_probs.view(1, -1)
            if tuple(old_policy_token_log_probs.shape) != tuple(policy_token_log_probs.shape):
                raise ValueError(
                    "old_policy_token_log_probs must align with policy_token_log_probs shape: "
                    f"{tuple(old_policy_token_log_probs.shape)} vs {tuple(policy_token_log_probs.shape)}"
                )
        self._assert_finite_tensor(
            stage="old_policy_token_log_probs",
            tensor_name="old_policy_token_log_probs",
            tensor_value=old_policy_token_log_probs,
            batch=batch,
        )
        advantages = self._prepare_advantages(batch, policy_token_log_probs.device)
        coef_1 = torch.exp(policy_token_log_probs - old_policy_token_log_probs.detach())
        self._assert_finite_tensor(
            stage="ppo_ratio",
            tensor_name="coef_1",
            tensor_value=coef_1,
            batch=batch,
        )
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
                self._assert_finite_tensor(
                    stage="reference_token_log_probs",
                    tensor_name="reference_token_log_probs",
                    tensor_value=reference_token_log_probs,
                    batch=batch,
                )
                delta = reference_token_log_probs.to(policy_token_log_probs.device) - policy_token_log_probs
                per_token_kl = torch.exp(delta) - delta - 1.0
                per_token_loss = per_token_loss + per_token_loss.new_tensor(self.kl_beta) * per_token_kl
        self._assert_finite_tensor(
            stage="per_token_loss",
            tensor_name="per_token_loss",
            tensor_value=per_token_loss,
            batch=batch,
        )
        response_mask_f = response_mask.to(dtype=per_token_loss.dtype)
        token_loss_weight = batch.get("token_loss_weight")
        if token_loss_weight is None:
            weighted_response_mask = response_mask_f
            token_counts = response_mask_f.sum(dim=-1).clamp(min=1.0)
        else:
            if not isinstance(token_loss_weight, torch.Tensor):
                token_loss_weight = torch.tensor(
                    token_loss_weight,
                    dtype=torch.float32,
                    device=policy_token_log_probs.device,
                )
            token_loss_weight = token_loss_weight.to(device=policy_token_log_probs.device, dtype=torch.float32)
            if token_loss_weight.ndim == 1:
                token_loss_weight = token_loss_weight.view(1, -1)
            if tuple(token_loss_weight.shape) != tuple(policy_token_log_probs.shape):
                raise ValueError(
                    "token_loss_weight must align with policy_token_log_probs shape: "
                    f"{tuple(token_loss_weight.shape)} vs {tuple(policy_token_log_probs.shape)}"
                )
            weighted_response_mask = response_mask_f * token_loss_weight.clamp_min(0.0)
            if not bool(torch.any(weighted_response_mask > 0)):
                return None
            token_counts = weighted_response_mask.sum(dim=-1).clamp(min=1.0)
        sample_losses = (per_token_loss * weighted_response_mask).sum(dim=-1) / token_counts
        sample_losses = sample_losses * self._sample_loss_multiplier(
            batch,
            device=sample_losses.device,
            sample_count=int(sample_losses.shape[0]),
        )
        self._assert_finite_tensor(
            stage="sample_losses",
            tensor_name="sample_losses",
            tensor_value=sample_losses,
            batch=batch,
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
                "rl_nonvisual_prepared_batch_noop_replacements": int(self._nonvisual_prepared_batch_noop_replacements),
                "rl_completion_only_grad_fallback_batches": int(self._completion_only_grad_fallback_batches),
                "rl_ddp_global_empty_batch_skips": int(self._ddp_global_empty_batch_skips),
                "rl_all_empty_batch_skips": int(self._all_empty_batch_skips),
                "rl_effective_update_steps": int(self._effective_update_steps),
                "rl_optimizer_step_skips": int(self._optimizer_step_skips),
                "rl_replay_fill_batches": int(self._replay_fill_batches),
                "rl_replay_fill_episode_specs": int(self._replay_fill_episode_specs),
                "rl_groups_all_zero_advantage": int(self._groups_all_zero_advantage),
                "rl_skipped_non_finite_old_policy_samples": int(self._skipped_non_finite_old_policy_samples),
                "rl_skipped_non_finite_compute_samples": int(self._skipped_non_finite_compute_samples),
                "rl_zero_variance_group_count": int(getattr(self, "_zero_variance_group_count", 0)),
                "rl_zero_variance_rollout_count": int(getattr(self, "_zero_variance_rollout_count", 0)),
                "rl_zero_variance_skipped_count": int(getattr(self, "_zero_variance_skipped_count", 0)),
                "rl_groups_filtered_by_min_weight": int(self._groups_filtered_by_min_weight),
                "rl_compute_loss_microbatch_size_effective": int(self.compute_loss_microbatch_size),
            }
        )
        for key, value in sorted(dict(getattr(self, "_advantage_source_counts", {})).items()):
            metrics[f"rl_advantage_source_{str(key)}"] = int(value)
        for key, value in sorted(dict(getattr(self, "_sample_partition_counts", {})).items()):
            metrics[f"rl_sample_partition_{str(key)}"] = int(value)
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
        self._debug_compute_loss_call_index += 1
        compute_loss_call_index = int(self._debug_compute_loss_call_index)
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
        total_effective_weight = 0.0
        microbatch_size = max(1, int(self.compute_loss_microbatch_size))
        prepared_microbatches: List[Dict[str, Any]] = []
        for start_index in range(0, len(episode_specs), microbatch_size):
            chunk = episode_specs[start_index : start_index + microbatch_size]
            prepared_microbatches.extend(
                self._materialize_episode_spec_microbatch(chunk, device=target_device)
            )
        runtime_stats["raw_local_prepared_batch_count"] = int(len(prepared_microbatches))
        prepared_microbatches = self._pad_prepared_batches_to_distributed_max(
            prepared_microbatches,
            device=target_device,
            runtime_stats=runtime_stats,
        )
        local_prepared_batch_count = int(len(prepared_microbatches))
        self._debug_last_compute_loss_call_index = int(compute_loss_call_index)
        self._debug_last_compute_loss_prepared_batch_count = int(local_prepared_batch_count)
        self._debug_last_compute_loss_rank_local_batch_count = int(local_prepared_batch_count)
        global_prepared_batch_count = _distributed_sum_int(local_prepared_batch_count, device=target_device)
        all_ranks_have_prepared_batches, any_rank_has_prepared_batches = _distributed_bool_consensus(
            local_prepared_batch_count > 0,
            device=target_device,
        )
        if not any_rank_has_prepared_batches or global_prepared_batch_count <= 0:
            if self.all_empty_policy == "true_skip":
                self._mark_skip_next_optimizer_step(reason="all_empty_prepared_batches", all_empty=True)
            self._maybe_log_empty_batch_rank_summary(
                reason="all_empty_prepared_batches",
                runtime_stats=runtime_stats,
                trainable_samples=0,
            )
            return _zero_loss_from_model(model)

        for prepared_batch_index, batch in enumerate(prepared_microbatches, start=1):
            recovered_preweighted = False
            recovered_trainable_samples = 0
            recovered_effective_weight_sum = 0.0
            try:
                sample_losses = self._compute_sample_losses_for_batch(
                    model=model,
                    batch=batch,
                )
            except Exception as exc:
                if not self._is_skippable_sample_exception(exc):
                    raise
                source_entries = self._prepared_batch_source_entries(batch)
                if len(source_entries) <= 1:
                    episode_entry = source_entries[0] if source_entries else {}
                    dump_path = self._write_skipped_sample_dump(
                        stage="compute_loss_skipped_sample",
                        episode_entry=episode_entry,
                        prepared_batch=batch,
                        error=exc,
                    )
                    runtime_log(
                        (
                            "trainer-native RL skipped a sample after compute_loss failure: "
                            f"video_id={str(episode_entry.get('video_id') or '') or 'unknown'} "
                            f"generation_id={int(episode_entry.get('generation_id', -1) or -1)} "
                            f"dump={dump_path}"
                        ),
                        runtime=distributed_runtime_from_env(),
                        main_process_only=False,
                    )
                    self._skipped_non_finite_compute_samples += max(1, len(source_entries))
                    sample_losses = None
                else:
                    (
                        sample_losses,
                        recovered_trainable_samples,
                        recovered_effective_weight_sum,
                    ) = self._recover_sample_losses_from_source_entries(
                        model=model,
                        batch=batch,
                        batch_error=exc,
                    )
                    recovered_preweighted = sample_losses is not None
            if sample_losses is None or sample_losses.numel() <= 0:
                runtime_log(
                    "rl debug prepared batch end: "
                    f"compute_loss_call={compute_loss_call_index} "
                    f"prepared_batch_index={int(prepared_batch_index)}/{int(local_prepared_batch_count)} "
                    "status=empty_or_skipped",
                    runtime=distributed_runtime_from_env(),
                    main_process_only=False,
                )
                continue
            if recovered_preweighted:
                sample_loss_multiplier = torch.ones_like(sample_losses, dtype=torch.float32, device=sample_losses.device)
                active_sample_count = int(recovered_trainable_samples)
                effective_weight_sum = float(recovered_effective_weight_sum)
            else:
                (
                    sample_loss_multiplier,
                    active_sample_count,
                    effective_weight_sum,
                ) = self._sample_loss_weight_summary(
                    batch,
                    device=sample_losses.device,
                    sample_count=int(sample_losses.numel()),
                )
            total_samples += (
                int(recovered_trainable_samples)
                if recovered_preweighted
                else int(active_sample_count)
            )
            total_effective_weight += float(effective_weight_sum)
            batch_loss_sum = sample_losses.sum()
            self._assert_finite_tensor(
                stage="batch_loss_sum",
                tensor_name="batch_loss_sum",
                tensor_value=batch_loss_sum,
                batch=batch,
            )
            total_loss_sum = batch_loss_sum if total_loss_sum is None else total_loss_sum + batch_loss_sum
            runtime_log(
                "rl debug prepared batch end: "
                f"compute_loss_call={compute_loss_call_index} "
                f"prepared_batch_index={int(prepared_batch_index)}/{int(local_prepared_batch_count)} "
                f"trainable_samples={int(active_sample_count)} "
                f"effective_weight_sum={float(effective_weight_sum):.6f} "
                f"batch_loss_sum={float(batch_loss_sum.detach().float().item()):.6f}",
                runtime=distributed_runtime_from_env(),
                main_process_only=False,
            )
        runtime_stats["raw_local_sample_count"] = int(total_samples)
        global_total_effective_weight = _distributed_sum_float(float(total_effective_weight), device=target_device)
        if total_loss_sum is None or global_total_effective_weight <= 0.0:
            if self.all_empty_policy == "true_skip":
                self._mark_skip_next_optimizer_step(reason="all_empty_trainable_samples", all_empty=True)
            self._maybe_log_empty_batch_rank_summary(
                reason="all_empty_trainable_samples",
                runtime_stats=runtime_stats,
                trainable_samples=int(total_samples),
            )
            return _zero_loss_from_model(model)
        world_size = max(1, int(_distributed_world_size()))
        runtime_log(
            "rl debug trainer backward imminent: "
            f"compute_loss_call={compute_loss_call_index} "
            f"local_prepared_batch_count={int(local_prepared_batch_count)} "
            f"global_total_samples={int(total_samples)} "
            f"global_effective_weight={float(global_total_effective_weight):.6f}",
            runtime=distributed_runtime_from_env(),
            main_process_only=False,
        )
        final_loss = total_loss_sum * float(world_size) / float(global_total_effective_weight)
        self._assert_finite_tensor(
            stage="compute_loss_return",
            tensor_name="final_loss",
            tensor_value=final_loss,
            extra={
                "global_total_samples": int(total_samples),
                "global_effective_weight": float(global_total_effective_weight),
                "world_size": int(world_size),
            },
        )
        return final_loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        detached_loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        sync_gradients = bool(getattr(getattr(self, "accelerator", None), "sync_gradients", False))
        if sync_gradients:
            self._assert_finite_trainable_gradients(
                stage="post_training_step_gradients",
                model=model,
                extra={
                    "global_step": int(getattr(getattr(self, "state", None), "global_step", 0) or 0),
                    "epoch": float(getattr(getattr(self, "state", None), "epoch", 0.0) or 0.0),
                },
            )
        self._assert_finite_tensor(
            stage="training_step_detached_loss",
            tensor_name="detached_loss",
            tensor_value=detached_loss,
        )
        runtime_log(
            "rl debug trainer backward finished: "
            f"compute_loss_call={int(getattr(self, '_debug_last_compute_loss_call_index', 0))} "
            f"local_prepared_batch_count={int(getattr(self, '_debug_last_compute_loss_rank_local_batch_count', 0))} "
            f"reported_loss={float(detached_loss.detach().float().item()) if isinstance(detached_loss, torch.Tensor) else detached_loss}",
            runtime=distributed_runtime_from_env(),
            main_process_only=False,
        )
        return detached_loss


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
    save_only_model: bool = False,
    warmup_ratio: float = 0.03,
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
    max_tool_message_frames: int = 0,
    max_total_video_frames: int = 0,
    keep_recent_tool_image_messages: int = 0,
    keep_recent_text_messages: int = 0,
    max_seq_length: int = 0,
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
    proposal_runtime: Any = None,
    strict_feature_guided_proposal: bool = False,
    ddp_find_unused_parameters: bool = False,
    deepspeed: Optional[str] = None,
    save_strategy: str = "no",
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
    os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(int(gradient_accumulation_steps))
    effective_save_only_model = bool(save_only_model)
    if str(deepspeed or "").strip() and effective_save_only_model:
        runtime_log(
            "DeepSpeed RL resume requires full Trainer checkpoints; overriding save_only_model=false.",
            runtime=distributed_runtime_from_env(),
            main_process_only=True,
        )
        effective_save_only_model = False
    training_args_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": float(learning_rate),
        "num_train_epochs": float(num_train_epochs),
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "logging_steps": int(logging_steps),
        "save_steps": int(save_steps),
        "save_total_limit": int(save_total_limit),
        "save_only_model": bool(effective_save_only_model),
        "warmup_ratio": float(warmup_ratio),
        "weight_decay": float(weight_decay),
        "max_grad_norm": float(max_grad_norm),
        "bf16": bool(bf16),
        "fp16": bool(fp16),
        "remove_unused_columns": False,
        "report_to": [],
        "disable_tqdm": True,
        "save_strategy": str(save_strategy or "no"),
        "dataloader_num_workers": max(0, int(dataloader_num_workers)),
        "dataloader_persistent_workers": bool(effective_persistent_workers),
        "ddp_find_unused_parameters": bool(ddp_find_unused_parameters),
    }
    if str(deepspeed or "").strip():
        training_args_kwargs["deepspeed"] = str(deepspeed)
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
        "max_tool_message_frames": int(max_tool_message_frames),
        "max_total_video_frames": int(max_total_video_frames),
        "keep_recent_tool_image_messages": int(keep_recent_tool_image_messages),
        "keep_recent_text_messages": int(keep_recent_text_messages),
        "max_seq_length": int(max_seq_length),
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
        "proposal_runtime": proposal_runtime,
        "strict_feature_guided_proposal": bool(strict_feature_guided_proposal),
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
        "bf16": bool(bf16),
        "fp16": bool(fp16),
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
    trainer.add_callback(_build_grad_norm_probe_callback(trainer=trainer))
    trainer.add_callback(_build_parameter_finite_probe_callback(trainer=trainer))
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
    raise RuntimeError(
        "idea2_v3 RL has converged on saver_v3.cli.train_rl_ds -> train_saver_rl_trl.py; "
        "the legacy native GRPO backend is deprecated and unsupported."
    )
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
    saver_config = build_saver_config(args)
    if materialized_train_items_path:
        expected_protocol_signature = build_protocol_signature(
            config=saver_config,
            max_turns=int(
                getattr(args, "rollout_max_turns", DEFAULT_ROLLOUT_MAX_TURNS) or DEFAULT_ROLLOUT_MAX_TURNS
            ),
            policy_max_new_tokens=int(getattr(args, "policy_max_new_tokens", 0) or 0),
            teacher_role=DEFAULT_TEACHER_ROLE,
        )
        ensure_materialized_cache_metadata(
            materialized_train_items_path,
            expected_format=MATERIALIZED_RUNTIME_ITEMS_FORMAT,
            expected_source_path=args.data,
            expected_include_splits=include_splits_value,
            expected_config=saver_config,
            expected_protocol_signature=expected_protocol_signature,
            require_config_match=True,
            require_source=True,
        )
        dataset = MaterializedRuntimeItemDataset(
            materialized_train_items_path,
            include_splits=include_splits_value,
            config=saver_config,
            require_frame_cache=True,
            require_feature_cache=True,
            proposal_runtime=None,
            strict_feature_guided_proposal=False,
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
    if not materialized_train_items_path:
        dataset = SaverAgentDataset(
            args.data,
            data_root=args.data_root,
            config=config_builder(args),
            include_splits=include_splits_value,
            require_frame_cache=True,
            require_feature_cache=True,
            proposal_runtime=None,
            strict_feature_guided_proposal=False,
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
        iter_dir.mkdir(parents=True, exist_ok=True)
        publish_iteration_artifacts = _should_publish_rl_iteration_artifacts(
            int(iteration),
            eval_start_iteration=int(getattr(args, "rollout_eval_start_iteration", 1) or 1),
            eval_every_iterations=int(getattr(args, "rollout_eval_interval_iterations", 1) or 1),
        )
        checkpoint_dir = (
            iter_dir / "checkpoint"
            if publish_iteration_artifacts
            else output_dir / "_rolling_iteration_checkpoint" / "checkpoint"
        )
        try:
            indices = select_iteration_indices_fn(
                len(raw_records),
                args.rollout_count,
                args.rollout_start_index,
                iteration,
                seed=getattr(args, "seed", 42),
            )
        except TypeError:
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
            "publish_iteration_artifacts": bool(publish_iteration_artifacts),
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
        if hasattr(rollout_eval_config, "inline_rollout_eval"):
            rollout_eval_config.inline_rollout_eval = bool(
                getattr(args, "inline_rollout_eval", False) and publish_iteration_artifacts
            )
        rollout_eval_output_dir = (
            rollout_eval_output_root / f"iter_{int(iteration):03d}"
            if publish_iteration_artifacts
            else rollout_eval_output_root / "_rolling_iteration_eval"
        )
        checkpoint_strategy = _rl_iteration_checkpoint_strategy(
            publish_iteration_artifacts=publish_iteration_artifacts,
            eval_start_iteration=int(getattr(args, "rollout_eval_start_iteration", 1) or 1),
            eval_every_iterations=int(getattr(args, "rollout_eval_interval_iterations", 1) or 1),
        )
        rollout_eval_callback = _build_rl_authority_checkpoint_callback(
            processor=processor,
            rollout_eval_config=rollout_eval_config,
            rollout_eval_output_dir=rollout_eval_output_dir,
            iteration_index=int(iteration),
            checkpoint_strategy=checkpoint_strategy,
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
            save_only_model=bool(getattr(args, "save_only_model", False)),
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
            max_tool_message_frames=int(getattr(args, "max_tool_message_frames", 0) or 0),
            max_total_video_frames=int(getattr(args, "max_total_video_frames", 0) or 0),
            keep_recent_tool_image_messages=args.keep_recent_tool_image_messages,
            keep_recent_text_messages=args.keep_recent_text_messages,
            max_seq_length=args.max_seq_length,
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
            proposal_runtime=proposal_runtime,
            strict_feature_guided_proposal=bool(strict_feature_guided_proposal),
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
            checkpoint_strategy = str(getattr(rollout_eval_callback, "checkpoint_strategy", checkpoint_strategy))
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
