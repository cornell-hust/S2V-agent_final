from __future__ import annotations

import copy
import gc
import importlib.util
import json
import math
import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from saver_v3.core.counterfactual_verification import (
    run_counterfactual_verification,
    run_counterfactual_verification_batch,
)
from saver_v3.rl.grpo_trainer_env import (
    RepeatSampler,
    _NativeGRPOProgressReporter,
    _RawItemDataset,
    _build_native_grpo_progress_callback,
    _build_seed_worker_init_fn,
    _compute_group_relative_advantages,
    _degrade_reward_summary_for_fecv_failure,
    _distributed_bool_consensus,
    _distributed_first_available_object,
    _distributed_sum_int,
    _distributed_world_size,
    _safe_float,
    _truncate_error_message,
)
from saver_v3.core.reward import (
    DEFAULT_RL_REWARD_VERSION,
    build_open_ended_reward_judge,
    build_timesearch_reward_funcs,
    resolve_reward_component_weights,
)
from saver_v3.core.rollout import SaverRolloutRunner
from saver_v3.common.runtime import distributed_runtime_from_env, runtime_log
from saver_v3.sft.training import (
    BatchBuildResult,
    BudgetingStats,
    _build_rl_completion_episode_spec_from_feature,
    _unwrap_model,
    _zero_loss_from_model,
    build_completion_only_model_inputs,
    compute_completion_only_token_log_probs_from_prepared_inputs,
    compute_completion_only_token_log_probs_from_ids,
    load_qwen_model_and_processor,
)


try:
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss
except Exception:  # pragma: no cover - optional dependency
    LigerFusedLinearGRPOLoss = None

try:
    from trl.models import create_reference_model as trl_create_reference_model
    from trl.models import prepare_deepspeed as trl_prepare_deepspeed
except Exception:  # pragma: no cover - optional dependency
    trl_create_reference_model = None
    trl_prepare_deepspeed = None

try:
    from trl.models.utils import _ForwardRedirection as trl_forward_redirection
except Exception:  # pragma: no cover - optional dependency
    trl_forward_redirection = None


def _raw_item_collator(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return features


def _iter_chunked(values: Sequence[Any], chunk_size: int) -> List[List[Any]]:
    size = max(1, int(chunk_size))
    seq = list(values or [])
    return [seq[idx : idx + size] for idx in range(0, len(seq), size)]


_USE_CURRENT_POLICY_LOGPROBS_SENTINEL = "__use_current_policy_logprobs__"


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


def _training_phase_runtime_log(message: str) -> None:
    runtime_log(
        message,
        runtime=distributed_runtime_from_env(),
        main_process_only=True,
    )


def _distributed_rank() -> int:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    return int(torch.distributed.get_rank())


def _distributed_gather_object(local_object: Any) -> List[Any]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return [copy.deepcopy(local_object)]
    gathered_objects: List[Any] = [None for _ in range(max(1, int(_distributed_world_size())))]
    torch.distributed.all_gather_object(gathered_objects, local_object)
    return gathered_objects


def _episode_input_sample_count_for_debug(episode_input: Dict[str, Any]) -> int:
    for key in ("completion_ids", "prompt_ids", "completion_mask", "advantages"):
        value = episode_input.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 1:
            return max(1, int(value.shape[0]))
    return 1


def _summarize_episode_input_for_rank_debug(episode_input: Dict[str, Any]) -> Dict[str, Any]:
    prompt_ids = episode_input.get("prompt_ids")
    completion_ids = episode_input.get("completion_ids")
    multimodal_inputs = episode_input.get("multimodal_inputs")
    if isinstance(multimodal_inputs, list):
        multimodal_keys = sorted({str(key) for sample in multimodal_inputs if isinstance(sample, dict) for key in sample.keys()})
    elif isinstance(multimodal_inputs, dict):
        multimodal_keys = sorted(str(key) for key in multimodal_inputs.keys())
    else:
        multimodal_keys = []
    return {
        "samples": int(_episode_input_sample_count_for_debug(episode_input)),
        "prompt_tokens": int(prompt_ids.shape[-1]) if isinstance(prompt_ids, torch.Tensor) and prompt_ids.ndim >= 1 else -1,
        "completion_tokens": int(completion_ids.shape[-1]) if isinstance(completion_ids, torch.Tensor) and completion_ids.ndim >= 1 else -1,
        "multimodal_keys": multimodal_keys,
    }


def _summarize_microbatch_layout_for_rank_debug(episode_inputs: Sequence[Dict[str, Any]], trainer: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for batch_index, batch in enumerate(list(episode_inputs or []), start=1):
        microbatches = list(trainer._iter_loss_microbatches(batch))
        rows.append(
            {
                "batch_index": int(batch_index),
                "batch_summary": _summarize_episode_input_for_rank_debug(batch),
                "microbatch_count": int(len(microbatches)),
                "microbatch_sample_counts": [int(trainer._episode_input_sample_count(microbatch)) for microbatch in microbatches],
            }
        )
    return rows


def _summarize_episode_input_bucket_for_rank_debug(episode_inputs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [_summarize_episode_input_for_rank_debug(episode_input) for episode_input in list(episode_inputs or [])]
    return {
        "items": int(len(rows)),
        "samples": int(sum(int(row.get("samples") or 0) for row in rows)),
        "rows": rows,
    }


def _rollout_group_sample_count(rollout_group: Dict[str, Any]) -> int:
    sample_count = 0
    for episode_input in list(rollout_group.get("episode_inputs") or []):
        if isinstance(episode_input, dict):
            sample_count += int(_episode_input_sample_count_for_debug(episode_input))
    return int(sample_count)


def _flatten_rollout_groups_to_episode_inputs(rollout_groups: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    episode_inputs: List[Dict[str, Any]] = []
    for rollout_group in list(rollout_groups or []):
        for episode_input in list((rollout_group or {}).get("episode_inputs") or []):
            episode_inputs.append(copy.deepcopy(dict(episode_input or {})))
    return episode_inputs


def _summarize_rollout_group_for_rank_debug(rollout_group: Dict[str, Any]) -> Dict[str, Any]:
    episode_inputs = list((rollout_group or {}).get("episode_inputs") or [])
    first_episode_input = dict(episode_inputs[0] or {}) if episode_inputs else {}
    return {
        "video_id": str(rollout_group.get("video_id") or ""),
        "generation_id": int(rollout_group.get("generation_id") if rollout_group.get("generation_id") is not None else -1),
        "source_rank": int(rollout_group.get("source_rank") if rollout_group.get("source_rank") is not None else -1),
        "source_item_index": int(
            rollout_group.get("source_item_index") if rollout_group.get("source_item_index") is not None else -1
        ),
        "source_rollout_index": int(
            rollout_group.get("source_rollout_index") if rollout_group.get("source_rollout_index") is not None else -1
        ),
        "episode_input_count": int(len(episode_inputs)),
        "sample_count": int(_rollout_group_sample_count(rollout_group)),
        "first_batch_summary": _summarize_episode_input_for_rank_debug(first_episode_input) if first_episode_input else None,
    }


def _stable_balance_rollout_groups_across_ranks(
    gathered_rollout_groups: Sequence[Sequence[Dict[str, Any]]],
) -> List[List[Dict[str, Any]]]:
    rank_count = max(1, len(list(gathered_rollout_groups or [])))
    assigned_groups: List[List[Dict[str, Any]]] = [[] for _ in range(rank_count)]
    assigned_sample_counts: List[int] = [0 for _ in range(rank_count)]
    ordered_groups: List[Dict[str, Any]] = []
    for source_rank, groups in enumerate(list(gathered_rollout_groups or [])):
        for source_local_group_index, rollout_group in enumerate(list(groups or [])):
            normalized_group = copy.deepcopy(dict(rollout_group or {}))
            normalized_group["source_rank"] = int(source_rank)
            normalized_group["source_local_group_index"] = int(source_local_group_index)
            normalized_group["sample_count"] = int(_rollout_group_sample_count(normalized_group))
            ordered_groups.append(normalized_group)
    for rollout_group in ordered_groups:
        target_rank = min(range(rank_count), key=lambda rank: (assigned_sample_counts[rank], rank))
        assigned_groups[target_rank].append(rollout_group)
        assigned_sample_counts[target_rank] += int(rollout_group.get("sample_count") or 0)
    return assigned_groups


def _patch_flash_attention_packed_sequence_check_for_active_rl() -> bool:
    try:
        import transformers.modeling_flash_attention_utils as flash_attention_utils
    except Exception:
        return False
    current = getattr(flash_attention_utils, "_is_packed_sequence", None)
    if not callable(current):
        return False
    if bool(getattr(flash_attention_utils, "_idea2_v3_active_rl_force_non_packed", False)):
        return True

    def _idea2_v3_rl_never_packed_sequence(position_ids, batch_size):
        del position_ids, batch_size
        return False

    flash_attention_utils._is_packed_sequence = _idea2_v3_rl_never_packed_sequence
    flash_attention_utils._idea2_v3_active_rl_force_non_packed = True
    return True


def _deepspeed_config_uses_zero3(config_path: Optional[str]) -> bool:
    config_text = str(config_path or "").strip()
    if not config_text:
        return False
    try:
        payload = json.loads(Path(config_text).read_text(encoding="utf-8"))
    except Exception:
        return False
    zero_optimization = payload.get("zero_optimization") or {}
    try:
        return int(zero_optimization.get("stage", 0) or 0) == 3
    except Exception:
        return False


def _deepspeed_config_uses_zero3_param_offload(config_path: Optional[str]) -> bool:
    config_text = str(config_path or "").strip()
    if not config_text:
        return False
    try:
        payload = json.loads(Path(config_text).read_text(encoding="utf-8"))
    except Exception:
        return False
    zero_optimization = payload.get("zero_optimization") or {}
    try:
        if int(zero_optimization.get("stage", 0) or 0) != 3:
            return False
    except Exception:
        return False
    offload_param = zero_optimization.get("offload_param") or {}
    if not isinstance(offload_param, dict) or not offload_param:
        return False
    device = str(offload_param.get("device") or "").strip().lower()
    return device not in {"", "none"}


def _build_managed_reference_model_like_timesearch_r(
    *,
    trainer: Any,
    model: Any,
    trainer_init_model_path: str | Path,
    torch_dtype: str,
    attn_implementation: Optional[str],
    kl_beta: float,
    deepspeed: Optional[str],
) -> tuple[Any, str]:
    if float(kl_beta) <= 0.0:
        return None, "none"
    if bool(getattr(trainer, "is_deepspeed_enabled", False)):
        if _deepspeed_config_uses_zero3(deepspeed):
            if load_qwen_model_and_processor is None:
                raise ImportError("TimeSearch-R style ZeRO-3 reference loading requires load_qwen_model_and_processor.")
            reference_model, _ = load_qwen_model_and_processor(
                trainer_init_model_path,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation or None,
                gradient_checkpointing=False,
                use_lora=False,
            )
        else:
            if trl_create_reference_model is None:
                raise ImportError("TimeSearch-R style managed reference requires trl.models.create_reference_model.")
            reference_model = trl_create_reference_model(model)
        if trl_prepare_deepspeed is None:
            raise ImportError("TimeSearch-R style managed reference requires trl.models.prepare_deepspeed.")
        reference_model = trl_prepare_deepspeed(reference_model, trainer.accelerator)
        return reference_model, "deepspeed"
    if trl_create_reference_model is None:
        raise ImportError("TimeSearch-R style managed reference requires trl.models.create_reference_model.")
    reference_model = trl_create_reference_model(model)
    reference_model = trainer.accelerator.prepare_model(reference_model, evaluation_mode=True)
    return reference_model, "accelerate"


def _resolve_nested_attr(root: Any, dotted_path: str) -> Any:
    value = root
    for segment in [part for part in str(dotted_path or "").split(".") if part]:
        if value is None:
            return None
        value = getattr(value, segment, None)
    return value


def _resolve_liger_linear_head(unwrapped_model: Any) -> Tuple[Any, torch.Tensor, Any, str]:
    candidate_paths = [
        "lm_head",
        "language_model.lm_head",
        "module.lm_head",
        "module.language_model.lm_head",
        "base_model.lm_head",
        "base_model.language_model.lm_head",
        "base_model.model.lm_head",
        "model.lm_head",
    ]
    candidates: List[Tuple[str, Any]] = []
    for candidate_path in candidate_paths:
        candidates.append((candidate_path, _resolve_nested_attr(unwrapped_model, candidate_path)))
    get_output_embeddings = getattr(unwrapped_model, "get_output_embeddings", None)
    if callable(get_output_embeddings):
        try:
            candidates.append(("get_output_embeddings", get_output_embeddings()))
        except Exception:
            pass

    candidate_summaries: List[str] = []
    for resolved_path, head_module in candidates:
        if head_module is None:
            candidate_summaries.append(f"{resolved_path}:missing")
            continue
        weight = getattr(head_module, "weight", None)
        if isinstance(weight, (torch.Tensor, torch.nn.Parameter)):
            return head_module, weight, getattr(head_module, "bias", None), resolved_path
        candidate_summaries.append(f"{resolved_path}:no-weight")
    _training_phase_runtime_log(
        "rl liger linear head resolve failed: "
        f"model_type={type(unwrapped_model).__name__} "
        f"candidates={candidate_summaries}"
    )
    raise RuntimeError("Liger GRPO loss requires a valid linear output head with a 2D weight tensor.")


@contextmanager
def _gather_liger_linear_head_parameters(head_module: Any):
    params: List[torch.nn.Parameter] = []
    for value in (getattr(head_module, "weight", None), getattr(head_module, "bias", None)):
        if isinstance(value, torch.nn.Parameter):
            params.append(value)
    if not params:
        yield
        return
    needs_gather = any(
        hasattr(param, "ds_id")
        or hasattr(param, "ds_status")
        or int(param.numel()) == 0
        for param in params
    )
    if not needs_gather:
        yield
        return
    try:
        import deepspeed

        with deepspeed.zero.GatheredParameters(params, modifier_rank=None):
            yield
        return
    except Exception as exc:
        try:
            summaries = [
                f"{type(param).__name__}:shape={tuple(int(v) for v in tuple(getattr(param, 'shape', ()) or ()))}"
                for param in params
            ]
        except Exception:
            summaries = [type(param).__name__ for param in params]
        _training_phase_runtime_log(
            "rl liger linear head gather fallback: "
            f"reason={exc.__class__.__name__} "
            f"params={summaries}"
        )
        yield


@contextmanager
def _gather_liger_linear_head_parameters(head_module: Any):
    params: List[torch.nn.Parameter] = []
    for value in (getattr(head_module, "weight", None), getattr(head_module, "bias", None)):
        if isinstance(value, torch.nn.Parameter):
            params.append(value)
    if not params:
        yield
        return
    needs_gather = any(
        hasattr(param, "ds_id")
        or hasattr(param, "ds_status")
        or int(getattr(param, "numel", lambda: 0)()) == 0
        for param in params
    )
    if not needs_gather:
        yield
        return
    try:
        import deepspeed

        with deepspeed.zero.GatheredParameters(params, modifier_rank=None):
            yield
        return
    except Exception as exc:
        try:
            summaries = [
                f"{type(param).__name__}:shape={tuple(int(v) for v in tuple(getattr(param, 'shape', ()) or ()))}"
                for param in params
            ]
        except Exception:
            summaries = [type(param).__name__ for param in params]
        _training_phase_runtime_log(
            "rl liger linear head gather fallback: "
            f"reason={exc.__class__.__name__} "
            f"params={summaries}"
        )
        yield


def _build_episode_tensor_packs_from_rollout(
    rollout: Dict[str, Any],
    *,
    min_abs_advantage: float = 0.0,
) -> List[Dict[str, Any]]:
    def _copy_feature_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, dict):
            return {key: _copy_feature_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_copy_feature_value(item) for item in value]
        return copy.deepcopy(value)

    rollout_advantage = float(rollout.get("group_advantage", 0.0) or 0.0)
    if abs(rollout_advantage) < float(min_abs_advantage):
        return []
    episode_feature = rollout.get("_rl_episode_training_feature")
    if not isinstance(episode_feature, dict):
        return []
    turn_samples = list(episode_feature.get("episode_turn_samples") or [])
    if not turn_samples:
        legacy_pack = copy.deepcopy(episode_feature)
        legacy_pack["advantages"] = torch.tensor([float(rollout_advantage)], dtype=torch.float32)
        legacy_pack["reward_summary"] = copy.deepcopy(rollout.get("reward_summary") or {})
        legacy_pack["fecv_failed"] = bool(rollout.get("fecv_failed"))
        legacy_pack["fecv_failure_message"] = str(rollout.get("fecv_failure_message") or "")
        return [legacy_pack]
    reward_summary = copy.deepcopy(rollout.get("reward_summary") or {})
    fecv_failed = bool(rollout.get("fecv_failed"))
    fecv_failure_message = str(rollout.get("fecv_failure_message") or "")
    packs: List[Dict[str, Any]] = []
    for turn_sample in turn_samples:
        if not isinstance(turn_sample, dict):
            continue
        pack = {key: _copy_feature_value(value) for key, value in turn_sample.items()}
        pack["advantages"] = torch.tensor([float(rollout_advantage)], dtype=torch.float32)
        pack["reward_summary"] = copy.deepcopy(reward_summary)
        pack["fecv_failed"] = fecv_failed
        pack["fecv_failure_message"] = fecv_failure_message
        packs.append(pack)
    return packs


def _summarize_missing_episode_training_feature_rollout(
    rollout: Dict[str, Any],
) -> Dict[str, Any]:
    turns = [dict(turn) for turn in list(rollout.get("turns") or []) if isinstance(turn, dict)]
    invalid_attempts = [dict(turn) for turn in list(rollout.get("invalid_attempts") or []) if isinstance(turn, dict)]
    valid_turns = [turn for turn in turns if bool(turn.get("valid_action"))]
    traced_valid_turns = [turn for turn in valid_turns if isinstance(turn.get("_rl_token_trace"), dict)]
    first_invalid = invalid_attempts[0] if invalid_attempts else {}
    return {
        "turn_count": int(len(turns)),
        "valid_turn_count": int(len(valid_turns)),
        "traced_valid_turn_count": int(len(traced_valid_turns)),
        "invalid_attempt_count": int(len(invalid_attempts)),
        "terminated_reason": str(rollout.get("terminated_reason") or ""),
        "first_invalid_action": str(first_invalid.get("action") or ""),
        "first_invalid_tool_name": str(first_invalid.get("tool_name") or ""),
        "first_invalid_guardrail_reason": str(first_invalid.get("guardrail_reason") or ""),
        "first_invalid_response": _truncate_error_message(first_invalid.get("response") or "", max_chars=160),
    }


def _log_reward_advantage_distribution(
    *,
    video_id: str,
    rollouts: Sequence[Dict[str, Any]],
) -> None:
    reward_values = [
        round(float(((rollout.get("reward_summary") or {}).get("total_reward") or 0.0)), 6)
        for rollout in list(rollouts or [])
    ]
    advantage_values = [
        round(float(rollout.get("group_advantage", 0.0) or 0.0), 6)
        for rollout in list(rollouts or [])
    ]
    if not reward_values:
        runtime_log(
            f"rl reward/advantage debug: video_id={video_id} reward_count=0 advantage_count=0",
            runtime=distributed_runtime_from_env(),
            main_process_only=True,
        )
        return
    reward_min = min(reward_values)
    reward_max = max(reward_values)
    advantage_min = min(advantage_values) if advantage_values else 0.0
    advantage_max = max(advantage_values) if advantage_values else 0.0
    max_abs_advantage = max((abs(value) for value in advantage_values), default=0.0)
    unique_reward_count = len({float(value) for value in reward_values})
    unique_advantage_count = len({float(value) for value in advantage_values})
    reward_hist = [f"{value:.6f}" for value in reward_values]
    advantage_hist = [f"{value:.6f}" for value in advantage_values]
    runtime_log(
        "rl reward/advantage debug: "
        f"video_id={video_id} "
        f"reward_count={len(reward_values)} "
        f"reward_min={reward_min:.6f} "
        f"reward_max={reward_max:.6f} "
        f"unique_reward_count={unique_reward_count} "
        f"advantage_min={advantage_min:.6f} "
        f"advantage_max={advantage_max:.6f} "
        f"max_abs_advantage={max_abs_advantage:.6f} "
        f"unique_advantage_count={unique_advantage_count} "
        f"reward_values={reward_hist} "
        f"advantage_values={advantage_hist}",
        runtime=distributed_runtime_from_env(),
        main_process_only=True,
    )


def _log_rollout_reward_components(
    *,
    video_id: str,
    rollouts: Sequence[Dict[str, Any]],
) -> None:
    def _generation_id(rollout: Dict[str, Any]) -> int:
        value = rollout.get("generation_id")
        return -1 if value is None else int(value)

    component_rows = []
    for rollout in list(rollouts or []):
        reward_summary = dict(rollout.get("reward_summary") or {})
        components = dict(reward_summary.get("components") or {})
        component_rows.append(
            {
                "generation_id": _generation_id(rollout),
                "total_reward": round(float(reward_summary.get("total_reward") or 0.0), 6),
                "accuracy_reward": round(float(components.get("accuracy_reward") or 0.0), 6),
                "anomaly_false_normal_penalty": round(
                    float(components.get("anomaly_false_normal_penalty") or 0.0),
                    6,
                ),
                "fecv_evidence_faithfulness_reward": round(
                    float(components.get("fecv_evidence_faithfulness_reward") or 0.0),
                    6,
                ),
                "protocol_finalize_reward": round(float(components.get("protocol_finalize_reward") or 0.0), 6),
            }
        )
    runtime_log(
        f"rl reward components debug: video_id={video_id} rows={component_rows}",
        runtime=distributed_runtime_from_env(),
        main_process_only=True,
    )


def _log_rollout_sample_reward_details(
    *,
    video_id: str,
    rollouts: Sequence[Dict[str, Any]],
) -> None:
    def _generation_id(rollout: Dict[str, Any]) -> int:
        value = rollout.get("generation_id")
        return -1 if value is None else int(value)

    for rollout in list(rollouts or []):
        reward_summary = dict(rollout.get("reward_summary") or {})
        components = dict(reward_summary.get("components") or {})
        weighted_components = dict(reward_summary.get("weighted_components") or {})
        turns = [dict(turn) for turn in list(rollout.get("turns") or []) if isinstance(turn, dict)]
        invalid_attempts = [dict(turn) for turn in list(rollout.get("invalid_attempts") or []) if isinstance(turn, dict)]
        state = dict(rollout.get("state") or {})
        final_answer = rollout.get("final_answer")
        if not isinstance(final_answer, dict):
            final_answer = state.get("finalized_case")
        tool_sequence = [
            str(turn.get("tool_name") or turn.get("action") or "assistant")
            for turn in turns
        ]
        runtime_log(
            "rl sample reward debug: "
            f"video_id={video_id} "
            f"generation_id={_generation_id(rollout)} "
            f"total_reward={float(reward_summary.get('total_reward') or 0.0):.6f} "
            f"group_advantage={float(rollout.get('group_advantage') or 0.0):.6f} "
            f"below_min_weight={bool(rollout.get('below_min_weight'))} "
            f"turn_count={len(turns)} "
            f"invalid_attempt_count={len(invalid_attempts)} "
            f"terminated_reason={str(rollout.get('terminated_reason') or '')} "
            f"tool_sequence={tool_sequence} "
            f"final_answer={final_answer if isinstance(final_answer, dict) else None} "
            f"accuracy_reward={float(components.get('accuracy_reward') or 0.0):.6f} "
            f"weighted_accuracy_reward={float(weighted_components.get('accuracy_reward') or 0.0):.6f} "
            f"anomaly_false_normal_penalty={float(components.get('anomaly_false_normal_penalty') or 0.0):.6f} "
            f"weighted_anomaly_false_normal_penalty={float(weighted_components.get('anomaly_false_normal_penalty') or 0.0):.6f} "
            f"fecv_evidence_faithfulness_reward={float(components.get('fecv_evidence_faithfulness_reward') or 0.0):.6f} "
            f"weighted_fecv_evidence_faithfulness_reward={float(weighted_components.get('fecv_evidence_faithfulness_reward') or 0.0):.6f} "
            f"protocol_finalize_reward={float(components.get('protocol_finalize_reward') or 0.0):.6f} "
            f"weighted_protocol_finalize_reward={float(weighted_components.get('protocol_finalize_reward') or 0.0):.6f} "
            f"fecv_decision_sufficiency_reward={float(components.get('fecv_decision_sufficiency_reward') or 0.0):.6f} "
            f"fecv_specificity_reward={float(components.get('fecv_specificity_reward') or 0.0):.6f} "
            f"fecv_branch_profile={str(reward_summary.get('fecv_branch_profile') or '')} "
            f"fecv_profile_source={str(reward_summary.get('fecv_profile_source') or '')} "
            f"fecv_full_selected_available={bool(reward_summary.get('fecv_full_selected_available'))} "
            f"fecv_full_selected_parse_mode={str(reward_summary.get('fecv_full_selected_parse_mode') or '')} "
            f"fecv_full_selected_unavailable_reason={str(reward_summary.get('fecv_full_selected_unavailable_reason') or '')} "
            f"fecv_selection_resolution_source={str(reward_summary.get('fecv_selection_resolution_source') or '')} "
            f"fecv_recovered_from_trace={bool(reward_summary.get('fecv_recovered_from_trace'))} "
            f"fecv_selected_window_count={int(reward_summary.get('fecv_selected_window_count') or 0)} "
            f"fecv_selected_record_count={int(reward_summary.get('fecv_selected_record_count') or 0)} "
            f"fecv_selected_window_ids={list(reward_summary.get('fecv_selected_window_ids') or [])} "
            f"fecv_full_selected_window_ids={list(reward_summary.get('fecv_full_selected_window_ids') or [])} "
            f"fecv_selected_by_stage={dict(reward_summary.get('fecv_selected_by_stage') or {})} "
            f"fecv_hard_negative_reason={str(reward_summary.get('fecv_hard_negative_reason') or '')} "
            f"fecv_selected_support_score={float(reward_summary.get('fecv_selected_support_score') or 0.0):.6f} "
            f"fecv_trigger_necessity_delta={float(reward_summary.get('fecv_trigger_necessity_delta') or 0.0):.6f} "
            f"fecv_negative_resistance_delta={float(reward_summary.get('fecv_negative_resistance_delta') or 0.0):.6f} "
            f"fecv_normal_reward_mode={str(reward_summary.get('fecv_normal_reward_mode') or '')} "
            f"fecv_normal_evidence_tool_turn_count={int(reward_summary.get('fecv_normal_evidence_tool_turn_count') or 0)} "
            f"fecv_normal_search_restraint_score={float(reward_summary.get('fecv_normal_search_restraint_score') or 0.0):.6f} "
            f"fecv_normal_window_restraint_score={float(reward_summary.get('fecv_normal_window_restraint_score') or 0.0):.6f} "
            f"fecv_normal_verification_consistency_score={float(reward_summary.get('fecv_normal_verification_consistency_score') or 0.0):.6f} "
            f"fecv_normal_query_alignment_score={float(reward_summary.get('fecv_normal_query_alignment_score') or 0.0):.6f} "
            f"fecv_normal_restraint_reward={float(reward_summary.get('fecv_normal_restraint_reward') or 0.0):.6f} "
            f"latest_verifier_turn_present={bool(reward_summary.get('latest_verifier_turn_present'))} "
            f"verifier_source={str(reward_summary.get('verifier_source') or '')} "
            f"uses_reference_conditioned_verifier={bool(reward_summary.get('uses_reference_conditioned_verifier'))}",
            runtime=distributed_runtime_from_env(),
            main_process_only=True,
        )


def _log_rollout_trajectory_signatures(
    *,
    video_id: str,
    rollouts: Sequence[Dict[str, Any]],
) -> None:
    def _generation_id(rollout: Dict[str, Any]) -> int:
        value = rollout.get("generation_id")
        return -1 if value is None else int(value)

    signature_rows = []
    for rollout in list(rollouts or []):
        turns = [dict(turn) for turn in list(rollout.get("turns") or []) if isinstance(turn, dict)]
        state = dict(rollout.get("state") or {})
        final_answer = rollout.get("final_answer")
        if not isinstance(final_answer, dict):
            final_answer = state.get("finalized_case")
        tool_sequence = [
            str(turn.get("tool_name") or turn.get("action") or "assistant")
            for turn in turns
        ]
        signature_rows.append(
            {
                "generation_id": _generation_id(rollout),
                "terminated_reason": str(rollout.get("terminated_reason") or ""),
                "tool_sequence": tool_sequence,
                "selected_window_ids": [str(value) for value in list(state.get("active_evidence_window_ids") or [])],
                "final_answer": final_answer if isinstance(final_answer, dict) else None,
            }
        )
    runtime_log(
        f"rl trajectory signature debug: video_id={video_id} rows={signature_rows}",
        runtime=distributed_runtime_from_env(),
        main_process_only=True,
    )


def _debug_tensor_payload_summary(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": [int(dim) for dim in value.shape],
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {
            str(key): _debug_tensor_payload_summary(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_debug_tensor_payload_summary(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_debug_tensor_payload_summary(item) for item in value)
    return type(value).__name__


def _log_rollout_episode_visual_signatures(
    *,
    video_id: str,
    rollouts: Sequence[Dict[str, Any]],
) -> None:
    def _generation_id(rollout: Dict[str, Any]) -> int:
        value = rollout.get("generation_id")
        return -1 if value is None else int(value)

    visual_rows = []
    for rollout in list(rollouts or []):
        episode_feature = rollout.get("_rl_episode_training_feature") or {}
        turn_samples = list(episode_feature.get("episode_turn_samples") or [])
        for turn_sample in turn_samples:
            if not isinstance(turn_sample, dict):
                continue
            prompt_trace = dict(turn_sample.get("episode_prompt_trace") or {})
            multimodal_inputs = dict(prompt_trace.get("multimodal_inputs") or {})
            prompt_ids = prompt_trace.get("prompt_ids")
            completion_mask = turn_sample.get("completion_mask")
            visual_rows.append(
                {
                    "generation_id": _generation_id(rollout),
                    "turn_index": int(turn_sample.get("turn_index") or -1),
                    "turn_kind": str(turn_sample.get("turn_kind") or ""),
                    "tool_name": str(turn_sample.get("tool_name") or ""),
                    "prompt_token_count": int(prompt_ids.shape[-1]) if isinstance(prompt_ids, torch.Tensor) else -1,
                    "completion_token_count": int(completion_mask.sum().item()) if isinstance(completion_mask, torch.Tensor) else -1,
                    "multimodal_keys": sorted(str(key) for key in multimodal_inputs.keys()),
                    "multimodal_summary": _debug_tensor_payload_summary(multimodal_inputs),
                }
            )
    runtime_log(
        f"rl turn visual debug: video_id={video_id} rows={visual_rows}",
        runtime=distributed_runtime_from_env(),
        main_process_only=True,
    )


class TimesearchAlignedGRPOTrainerMixin:
    def __init__(
        self,
        *trainer_args: Any,
        aligned_grpo_config: Optional[Dict[str, Any]] = None,
        **trainer_kwargs: Any,
    ) -> None:
        config = dict(aligned_grpo_config or {})
        if not config:
            raise ValueError("aligned_grpo_config is required when constructing the dedicated GRPO trainer.")

        train_dataset = config.get("train_dataset")
        self.processor = config["processor"]
        self.reference_model = config.get("reference_model")
        self.use_lora_reference_disable_adapter = bool(config.get("use_lora_reference_disable_adapter", False))
        self.reference_model_mode = str(config.get("reference_model_mode") or "per_iteration_trainer_init")
        self.reference_model_source_path = str(config.get("reference_model_source_path") or "")
        self.reference_model_backend = str(config.get("reference_model_backend") or "none")
        self.kl_beta = float(config["kl_beta"])
        self.ppo_clip_epsilon = float(config["ppo_clip_epsilon"])
        self.rollout_runner = config["rollout_runner"]
        self.proposal_runtime = config.get("proposal_runtime")
        self.strict_feature_guided_proposal = bool(config.get("strict_feature_guided_proposal", False))
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
        self.rl_enable_reference_prefetch_cache = bool(config.get("rl_enable_reference_prefetch_cache", True))
        self.use_liger_loss = bool(config.get("use_liger_loss", False))
        self.use_liger_loss_requested = bool(self.use_liger_loss)
        self.use_liger_loss_effective = False
        self.steps_per_generation = max(1, int(config["steps_per_generation"]))
        self.rollout_stage_batch_size = max(1, int(config.get("rollout_stage_batch_size", 16)))
        self.fecv_stage_batch_size = max(1, int(config.get("fecv_stage_batch_size", 16)))
        self._generation_step_batch_size = max(1, int(config["per_device_train_batch_size"]))
        self._generation_batch_size = max(1, self._generation_step_batch_size * self.steps_per_generation)
        self._buffered_generation_step_payloads: List[Dict[str, Any]] = []
        self._buffered_generation_batch_key: Optional[Tuple[Any, ...]] = None
        self._recent_nonzero_advantage_payloads: List[Dict[str, Any]] = []
        self._recent_nonzero_advantage_payload_capacity = 8
        self.fecv_failure_policy = str(config["fecv_failure_policy"] or "degrade").strip().lower()
        self.log_empty_batch_rank_summary = bool(config["log_empty_batch_rank_summary"])
        self.reward_version = str(config["reward_version"] or DEFAULT_RL_REWARD_VERSION).strip().lower()
        self.reward_config = dict(config["reward_config"] or {})
        self.reward_config.setdefault("reward_version", self.reward_version)
        self.reward_judge = build_open_ended_reward_judge(reward_config=self.reward_config)
        self.reward_funcs = list(
            build_timesearch_reward_funcs(
                reward_config=self.reward_config,
                llm_judge=self.reward_judge,
            )
        )
        self.reward_func_names = [str(getattr(func, "__name__", f"reward_{index}")) for index, func in enumerate(self.reward_funcs)]
        self.reward_weights_map = resolve_reward_component_weights(
            reward_version=self.reward_version,
            reward_config=self.reward_config,
        )
        self.reward_weights = torch.tensor(
            [float(self.reward_weights_map.get(name, 0.0)) for name in self.reward_func_names],
            dtype=torch.float32,
        )
        self.policy_builder = config["policy_builder"]
        self._budgeting_stats = BudgetingStats()
        self._zero_response_dropped = 0
        self._materialize_fallback_batches = 0
        self._groups_all_zero_advantage = 0
        self._groups_filtered_by_min_weight = 0
        self._fecv_failure_count = 0
        self._fecv_degraded_rollout_count = 0
        self._ddp_noop_padded_episode_inputs = 0
        self._skip_empty_training_steps = 0
        self._zero_advantage_replay_uses = 0
        self._zero_advantage_replay_misses = 0
        self.liger_grpo_loss = None
        self._liger_runtime_probe_completed = False
        self._liger_runtime_disable_reason: Optional[str] = None
        self._liger_linear_head_path: str = ""
        self._liger_hidden_state_path: str = ""
        self._liger_runtime_logged = False
        self._forward_redirection = None
        self._liger_compiled = False
        self._flash_attention_packed_sequence_patch_applied = _patch_flash_attention_packed_sequence_check_for_active_rl()
        if self._flash_attention_packed_sequence_patch_applied:
            _training_phase_runtime_log(
                "rl flash attention packed-sequence patch enabled: force_non_packed_sequence=true"
            )
        if self.use_liger_loss:
            if LigerFusedLinearGRPOLoss is None or importlib.util.find_spec("liger_kernel") is None:
                raise ImportError("liger_kernel is required when use_liger_loss=True.")
            if trl_forward_redirection is None:
                raise ImportError("trl.models.utils._ForwardRedirection is required when use_liger_loss=True.")
            self._forward_redirection = trl_forward_redirection()
            self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
                beta=self.kl_beta,
                epsilon_low=self.ppo_clip_epsilon,
                epsilon_high=self.ppo_clip_epsilon,
                temperature=float(self.policy_temperature or 1.0),
                use_ref_model=self.kl_beta > 0.0,
                loss_type="grpo",
                max_completion_length=self.policy_max_new_tokens,
            )
            if hasattr(self.liger_grpo_loss, "compiled"):
                self.liger_grpo_loss.compiled = False
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

    def _ensure_liger_runtime_ready(self, model: Any) -> None:
        if bool(getattr(self, "_liger_runtime_probe_completed", False)) or not bool(getattr(self, "use_liger_loss", False)):
            return
        self._liger_runtime_probe_completed = True
        if self.liger_grpo_loss is None or self._forward_redirection is None:
            self._liger_runtime_disable_reason = "liger_runtime_unavailable"
            raise RuntimeError(
                "use_liger_loss=True requested, but the Liger runtime was not initialized correctly."
            )
        deepspeed_config_path = str(getattr(getattr(self, "args", None), "deepspeed", "") or "").strip()
        if _deepspeed_config_uses_zero3_param_offload(deepspeed_config_path):
            self._liger_runtime_disable_reason = "zero3_param_offload_incompatible_with_liger"
            raise RuntimeError(
                "idea2_v3 RL Liger path is incompatible with DeepSpeed ZeRO-3 parameter offload because it makes "
                "compute_loss forwards pathologically slow. Switch RL to configs/deepspeed/zero3_full_model.json "
                "or another non-offload config."
            )
        self.use_liger_loss_effective = True

    def _should_use_forward_redirection_for_liger(self) -> bool:
        deepspeed_config_path = str(getattr(getattr(self, "args", None), "deepspeed", "") or "").strip()
        return bool(self._forward_redirection is not None) and _deepspeed_config_uses_zero3(deepspeed_config_path)

    def get_train_dataloader(self):
        train_dataset = getattr(self, "train_dataset", None)
        if train_dataset is None:
            raise ValueError("Dedicated GRPO trainer requires a train_dataset.")
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
            shuffle=False,
            seed=getattr(self.args, "seed", None),
        )

    def _new_rollout_metric_lists(self) -> Dict[str, List[float]]:
        return {
            "reward_total": [],
            "reward_accuracy": [],
            "reward_fecv_evidence": [],
            "reward_protocol_finalize": [],
        }

    def _new_runtime_stats(self) -> Dict[str, int]:
        return {
            "raw_local_episode_input_count": 0,
            "raw_local_sample_count": 0,
            "local_fecv_failure_count": 0,
            "groups_filtered_by_min_weight": 0,
            "groups_all_zero_advantage": 0,
            "missing_episode_training_feature_count": 0,
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
            "episode_inputs": [],
            "rollout_metrics": {key: 0.0 for key in self._new_rollout_metric_lists().keys()},
            "budgeting_metrics": self.get_budget_drop_metrics(),
            "runtime_stats": self._new_runtime_stats(),
            "video_ids": [str(video_id or "") for video_id in (video_ids or [])],
        }

    def _build_policy(self, model: Any, *, use_generation_cache: bool) -> Any:
        return self.policy_builder(
            model=model,
            use_generation_cache=bool(use_generation_cache),
            step_resolver=lambda: int(getattr(getattr(self, "state", None), "global_step", 0) or 0),
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
        )

    def _build_rollout_policy(self, model: Any) -> Any:
        return self._build_policy(model, use_generation_cache=self.rollout_use_generation_cache)

    def _build_fecv_policy(self, model: Any) -> Any:
        return self._build_policy(model, use_generation_cache=self.fecv_use_generation_cache)

    def _effective_local_rollout_batch_size(self) -> int:
        return max(1, min(int(self.rollout_stage_batch_size), int(self.fecv_stage_batch_size)))

    def _assign_reward_summaries(
        self,
        rollouts: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rollout_list = list(rollouts or [])
        if not rollout_list:
            return []
        component_outputs: Dict[str, List[float]] = {}
        for reward_func, reward_name in zip(self.reward_funcs, self.reward_func_names):
            outputs = list(reward_func(rollout_traces=rollout_list))
            if len(outputs) != len(rollout_list):
                raise ValueError(
                    f"Reward function {reward_name!r} returned {len(outputs)} values for {len(rollout_list)} rollouts."
                )
            component_outputs[reward_name] = [
                float(value if value is not None else 0.0)
                for value in outputs
            ]
        weighted_names = [name for name in self.reward_func_names if float(self.reward_weights_map.get(name, 0.0)) != 0.0]
        enriched_rollouts: List[Dict[str, Any]] = []
        for index, rollout in enumerate(rollout_list):
            components = {
                reward_name: round(float(component_outputs[reward_name][index]), 6)
                for reward_name in self.reward_func_names
            }
            total_reward = 0.0
            for reward_name in weighted_names:
                total_reward += float(self.reward_weights_map.get(reward_name, 0.0)) * float(components.get(reward_name, 0.0))
            reward_summary = {
                "reward_version": str(self.reward_version),
                "total_reward": round(float(total_reward), 6),
                "components": dict(components),
                "weights": dict(self.reward_weights_map),
            }
            if bool(rollout.get("fecv_failed")):
                reward_summary = _degrade_reward_summary_for_fecv_failure(
                    reward_summary,
                    error_message=str(rollout.get("fecv_failure_message") or ""),
                )
                self._fecv_degraded_rollout_count += 1
            enriched_rollout = copy.deepcopy(rollout)
            enriched_rollout["reward_summary"] = reward_summary
            enriched_rollouts.append(enriched_rollout)
        return enriched_rollouts

    def _generate_scored_rollouts(
        self,
        item: Dict[str, Any],
        model: Any,
        *,
        progress: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        batched_rollouts = self._generate_scored_rollouts_batch(
            [item],
            model,
            progress=progress,
        )
        return batched_rollouts[0] if batched_rollouts else []

    def _apply_item_context_to_rollout(
        self,
        rollout: Dict[str, Any],
        *,
        item: Dict[str, Any],
        generation_id: int,
    ) -> Dict[str, Any]:
        rollout["group_id"] = str(
            item.get("rl_instance_id")
            or item.get("video_id")
            or f"group_{int(generation_id)}"
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
        return rollout

    def _prepare_rollout_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        prepared = copy.deepcopy(dict(item or {}))
        if not bool(getattr(self, "strict_feature_guided_proposal", False)):
            return prepared
        multimodal_cache = prepared.get("multimodal_cache")
        if not isinstance(multimodal_cache, dict):
            raise ValueError("Strict RL seek_evidence requires multimodal_cache on every rollout item.")
        if multimodal_cache.get("embedding") is None:
            raise ValueError("Strict RL seek_evidence requires feature_cache on every rollout item.")
        proposal_runtime = getattr(self, "proposal_runtime", None)
        if proposal_runtime is None:
            raise ValueError("Strict RL seek_evidence requires proposal_runtime before rollout generation.")
        multimodal_cache["proposal_runtime"] = proposal_runtime
        multimodal_cache["strict_feature_guided_proposal"] = True
        prepared["multimodal_cache"] = multimodal_cache
        return prepared

    def _generate_scored_rollouts_batch(
        self,
        items: Sequence[Dict[str, Any]],
        model: Any,
        *,
        progress: Optional[Any] = None,
    ) -> List[List[Dict[str, Any]]]:
        item_list = [dict(item) for item in list(items or [])]
        if not item_list:
            return []
        fecv_branch_profile = "online_core"
        rollout_policy = self._build_rollout_policy(model)
        runtime_log(
            "rl rollout policy debug: "
            f"policy_class={rollout_policy.__class__.__name__} "
            f"capture_rl_token_traces={bool(getattr(rollout_policy, 'capture_rl_token_traces', False))} "
            f"has_pop_traces={callable(getattr(rollout_policy, 'pop_last_rl_token_traces', None))} "
            f"vllm_runtime_enabled={bool(getattr(getattr(rollout_policy, 'vllm_runtime', None), 'enabled', False))} "
            f"use_generation_cache={bool(getattr(rollout_policy, 'use_generation_cache', False))}",
            runtime=distributed_runtime_from_env(),
            main_process_only=True,
        )
        verification_policy = self._build_fecv_policy(model)
        rollout_items: List[Dict[str, Any]] = []
        rollout_meta: List[Tuple[int, int]] = []
        for item_index, item in enumerate(item_list):
            prepared_rollout_item = self._prepare_rollout_item(item)
            for generation_id in range(self.num_generations):
                rollout_items.append(copy.deepcopy(prepared_rollout_item))
                rollout_meta.append((int(item_index), int(generation_id)))

        rollouts_by_item: List[List[Dict[str, Any]]] = [[] for _ in item_list]
        local_stage_batch_size = self._effective_local_rollout_batch_size()
        local_chunks = _iter_chunked(list(zip(rollout_meta, rollout_items)), local_stage_batch_size)
        total_local_chunks = len(local_chunks)
        for local_chunk_index, local_chunk in enumerate(local_chunks, start=1):
            chunk_meta = [entry[0] for entry in local_chunk]
            chunk_items = [entry[1] for entry in local_chunk]
            chunk_start_time = time.perf_counter()
            runtime_log(
                f"rl local chunk start: chunk={local_chunk_index}/{total_local_chunks} size={len(chunk_items)}",
                runtime=distributed_runtime_from_env(),
                main_process_only=True,
            )
            rollout_start_time = time.perf_counter()
            generated_rollouts = list(self.rollout_runner.run_episodes(chunk_items, rollout_policy))
            rollout_elapsed = time.perf_counter() - rollout_start_time

            fecv_batch_inputs: List[Dict[str, Any]] = []
            fecv_rollout_refs: List[Tuple[int, int, Dict[str, Any], Dict[str, Any]]] = []
            for (item_index, generation_id), rollout in zip(chunk_meta, generated_rollouts):
                item = item_list[item_index]
                rollout = self._apply_item_context_to_rollout(
                    rollout,
                    item=item,
                    generation_id=int(generation_id),
                )
                video_id = str(rollout.get("video_id") or item.get("video_id") or "")
                if progress is not None:
                    progress.advance_generation_stage(
                        video_id=video_id,
                        generation_id=int(generation_id),
                        stage="rollout",
                )
                fecv_batch_inputs.append(
                    {
                        "item": item,
                        "rollout": rollout,
                        "reference_record": item,
                    }
                )
                fecv_rollout_refs.append((int(item_index), int(generation_id), rollout, item))
            fecv_start_time = time.perf_counter()
            try:
                fecv_results = run_counterfactual_verification_batch(
                    verification_policy,
                    batch_inputs=fecv_batch_inputs,
                    max_images=self.counterfactual_max_images,
                    branch_profile=fecv_branch_profile,
                )
            except Exception as exc:
                if self.fecv_failure_policy == "fail":
                    raise
                fecv_results = [exc for _ in fecv_batch_inputs]
            fecv_elapsed = time.perf_counter() - fecv_start_time
            runtime_log(
                f"rl local chunk end: chunk={local_chunk_index}/{total_local_chunks} size={len(chunk_items)} rollout_elapsed_sec={rollout_elapsed:.3f} fecv_elapsed_sec={fecv_elapsed:.3f} elapsed_sec={time.perf_counter() - chunk_start_time:.3f}",
                runtime=distributed_runtime_from_env(),
                main_process_only=True,
            )
            for (item_index, generation_id, rollout, item), fecv_result in zip(fecv_rollout_refs, fecv_results):
                video_id = str(rollout.get("video_id") or item.get("video_id") or "")
                try:
                    if isinstance(fecv_result, Exception):
                        raise fecv_result
                    rollout.update(dict(fecv_result or {}))
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
                if progress is not None:
                    progress.advance_generation_stage(
                        video_id=video_id,
                        generation_id=int(generation_id),
                        stage="fecv",
                    )
                if bool(rollout.get("drop_due_to_fecv_failure")):
                    continue
                if progress is not None:
                    progress.advance_generation_stage(
                        video_id=video_id,
                        generation_id=int(generation_id),
                        stage="score",
                    )
                rollouts_by_item[item_index].append(rollout)

        grouped_scored_rollouts: List[List[Dict[str, Any]]] = []
        for item, item_rollouts in zip(item_list, rollouts_by_item):
            scored_rollouts = self._assign_reward_summaries(item_rollouts)
            advantaged_rollouts = _compute_group_relative_advantages(
                scored_rollouts,
                clip_value=self.advantage_clip,
            )
            _log_reward_advantage_distribution(
                video_id=str(item.get("video_id") or ""),
                rollouts=advantaged_rollouts,
            )
            _log_rollout_reward_components(
                video_id=str(item.get("video_id") or ""),
                rollouts=advantaged_rollouts,
            )
            _log_rollout_sample_reward_details(
                video_id=str(item.get("video_id") or ""),
                rollouts=advantaged_rollouts,
            )
            _log_rollout_trajectory_signatures(
                video_id=str(item.get("video_id") or ""),
                rollouts=advantaged_rollouts,
            )
            _log_rollout_episode_visual_signatures(
                video_id=str(item.get("video_id") or ""),
                rollouts=advantaged_rollouts,
            )
            grouped_scored_rollouts.append(advantaged_rollouts)
        return grouped_scored_rollouts

    def _build_generation_item_payload_from_rollouts(
        self,
        item: Dict[str, Any],
        scored_rollouts: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        video_id = str(item.get("video_id") or "")
        rollout_metrics = self._new_rollout_metric_lists()
        runtime_stats = self._new_runtime_stats()
        rollout_groups: List[Dict[str, Any]] = []
        num_episode_candidates_before_min_weight = 0
        num_episode_candidates_after_min_weight = 0
        num_invalid_action_turns = 0
        num_rollouts_dropped_by_min_weight = 0
        num_zero_response_after_budgeting = 0
        num_zero_prompt_after_budgeting = 0
        num_truncated_completion_after_budgeting = 0
        missing_feature_turn_count = 0
        missing_feature_valid_turn_count = 0
        missing_feature_traced_valid_turn_count = 0
        missing_feature_invalid_attempt_count = 0
        first_missing_feature_example = ""
        for rollout_index, rollout in enumerate(list(scored_rollouts or [])):
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
            for turn in list(rollout.get("turns") or []):
                if not bool(turn.get("valid_action")):
                    num_invalid_action_turns += 1
            rollout_advantage = abs(float(rollout.get("group_advantage", 0.0) or 0.0))
            if rollout_advantage < float(self.min_weight):
                runtime_stats["groups_filtered_by_min_weight"] += 1
                num_rollouts_dropped_by_min_weight += 1
                if rollout_advantage <= 0.0:
                    runtime_stats["groups_all_zero_advantage"] += 1
            raw_packs = _build_episode_tensor_packs_from_rollout(
                rollout,
                min_abs_advantage=0.0,
            )
            if raw_packs:
                num_episode_candidates_before_min_weight += len(raw_packs)
            else:
                runtime_stats["missing_episode_training_feature_count"] += 1
                missing_feature_summary = _summarize_missing_episode_training_feature_rollout(rollout)
                missing_feature_turn_count += int(missing_feature_summary["turn_count"])
                missing_feature_valid_turn_count += int(missing_feature_summary["valid_turn_count"])
                missing_feature_traced_valid_turn_count += int(missing_feature_summary["traced_valid_turn_count"])
                missing_feature_invalid_attempt_count += int(missing_feature_summary["invalid_attempt_count"])
                if not first_missing_feature_example:
                    first_missing_feature_example = (
                        f"terminated_reason={missing_feature_summary['terminated_reason'] or 'none'} "
                        f"turns={missing_feature_summary['turn_count']} "
                        f"valid_turns={missing_feature_summary['valid_turn_count']} "
                        f"traced_valid_turns={missing_feature_summary['traced_valid_turn_count']} "
                        f"invalid_attempts={missing_feature_summary['invalid_attempt_count']} "
                        f"first_invalid_action={missing_feature_summary['first_invalid_action'] or 'none'} "
                        f"first_invalid_tool={missing_feature_summary['first_invalid_tool_name'] or 'none'} "
                        f"guardrail={missing_feature_summary['first_invalid_guardrail_reason'] or 'none'} "
                        f"response={missing_feature_summary['first_invalid_response'] or 'none'}"
                    )
            if rollout_advantage < float(self.min_weight):
                continue
            rollout_episode_inputs: List[Dict[str, Any]] = []
            for pack in raw_packs:
                num_episode_candidates_after_min_weight += 1
                result = self._build_episode_input_from_feature(pack)
                self._budgeting_stats.record(result)
                if result.batch is None:
                    drop_reason = str(result.drop_reason or "")
                    if drop_reason == "zero_response_after_budgeting":
                        num_zero_response_after_budgeting += 1
                    elif drop_reason == "zero_prompt_after_budgeting":
                        num_zero_prompt_after_budgeting += 1
                    elif drop_reason == "truncated_completion_after_budgeting":
                        num_truncated_completion_after_budgeting += 1
                    self._zero_response_dropped += 1
                    continue
                rollout_episode_inputs.append(
                    dict(result.batch or {})
                )
            if rollout_episode_inputs:
                rollout_groups.append(
                    {
                        "video_id": video_id,
                        "group_id": str(rollout.get("group_id") or video_id),
                        "generation_id": int(rollout.get("generation_id") if rollout.get("generation_id") is not None else -1),
                        "source_item_index": None,
                        "source_rollout_index": int(rollout_index),
                        "episode_inputs": rollout_episode_inputs,
                        "sample_count": int(
                            sum(self._episode_input_sample_count(episode_input) for episode_input in rollout_episode_inputs)
                        ),
                    }
                )
        episode_inputs = _flatten_rollout_groups_to_episode_inputs(rollout_groups)
        runtime_stats["raw_local_episode_input_count"] = int(len(episode_inputs))
        runtime_log(
            "rl episode payload built: "
            f"video_id={video_id} "
            f"scored_rollouts={len(scored_rollouts)} "
            f"num_rollout_groups={len(rollout_groups)} "
            f"num_rollouts_dropped_by_min_weight={num_rollouts_dropped_by_min_weight} "
            f"num_episode_candidates_before_min_weight={num_episode_candidates_before_min_weight} "
            f"num_episode_candidates_after_min_weight={num_episode_candidates_after_min_weight} "
            f"num_missing_episode_training_feature={int(runtime_stats.get('missing_episode_training_feature_count', 0))} "
            f"missing_feature_turns={missing_feature_turn_count} "
            f"missing_feature_valid_turns={missing_feature_valid_turn_count} "
            f"missing_feature_traced_valid_turns={missing_feature_traced_valid_turn_count} "
            f"missing_feature_invalid_attempts={missing_feature_invalid_attempt_count} "
            f"num_episode_input_build_success={len(episode_inputs)} "
            f"num_zero_response_after_budgeting={num_zero_response_after_budgeting} "
            f"num_zero_prompt_after_budgeting={num_zero_prompt_after_budgeting} "
            f"num_truncated_completion_after_budgeting={num_truncated_completion_after_budgeting} "
            f"num_invalid_action_turns={num_invalid_action_turns}",
            runtime=distributed_runtime_from_env(),
            main_process_only=True,
        )
        if first_missing_feature_example:
            runtime_log(
                f"rl missing feature example: video_id={video_id} {first_missing_feature_example}",
                runtime=distributed_runtime_from_env(),
                main_process_only=True,
            )
        return {
            "video_id": video_id,
            "rollout_groups": rollout_groups,
            "episode_inputs": episode_inputs,
            "rollout_metric_values": rollout_metrics,
            "runtime_stats": runtime_stats,
        }

    def _build_generation_item_payload(
        self,
        item: Dict[str, Any],
        rollout_model: Any,
    ) -> Dict[str, Any]:
        scored_rollouts = self._generate_scored_rollouts(item, rollout_model, progress=None)
        return self._build_generation_item_payload_from_rollouts(item, scored_rollouts)

    def _build_episode_input_from_feature(
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

    def _move_episode_input_to_device(
        self,
        episode_input: Dict[str, Any],
        *,
        device: torch.device,
    ) -> Dict[str, Any]:
        return {
            str(key): self._move_nested_value_to_device(value, device=device)
            for key, value in episode_input.items()
        }

    def _move_nested_value_to_device(
        self,
        value: Any,
        *,
        device: torch.device,
    ) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device=device)
        if isinstance(value, dict):
            return {
                str(key): self._move_nested_value_to_device(item, device=device)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._move_nested_value_to_device(item, device=device) for item in value]
        if isinstance(value, tuple):
            return tuple(self._move_nested_value_to_device(item, device=device) for item in value)
        return copy.deepcopy(value)

    def _find_first_tensor_device(self, value: Any) -> Optional[torch.device]:
        if isinstance(value, torch.Tensor):
            return value.device
        if isinstance(value, dict):
            for item in value.values():
                found = self._find_first_tensor_device(item)
                if found is not None:
                    return found
            return None
        if isinstance(value, (list, tuple)):
            for item in value:
                found = self._find_first_tensor_device(item)
                if found is not None:
                    return found
            return None
        return None

    def _episode_input_sample_count(self, episode_input: Dict[str, Any]) -> int:
        for key in ("completion_ids", "prompt_ids", "completion_mask", "advantages"):
            value = episode_input.get(key)
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return max(1, int(value.shape[0]))
        return 1

    def _episode_input_cpu_copy(self, episode_input: Dict[str, Any]) -> Dict[str, Any]:
        return self._move_episode_input_to_device(episode_input, device=torch.device("cpu"))

    def _clone_episode_input_as_noop(self, episode_input: Dict[str, Any]) -> Dict[str, Any]:
        cloned = copy.deepcopy(dict(episode_input or {}))
        sample_count = self._episode_input_sample_count(cloned)
        first_tensor_device = next(
            (
                found
                for found in (self._find_first_tensor_device(value) for value in cloned.values())
                if found is not None
            ),
            torch.device("cpu"),
        )
        cloned["sample_loss_multiplier"] = torch.zeros(sample_count, dtype=torch.float32, device=first_tensor_device)
        if isinstance(cloned.get("advantages"), torch.Tensor):
            cloned["advantages"] = torch.zeros_like(cloned["advantages"], dtype=torch.float32)
        if isinstance(cloned.get("sample_weight"), torch.Tensor):
            cloned["sample_weight"] = torch.zeros_like(cloned["sample_weight"], dtype=torch.float32)
        return cloned

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
                "sample_loss_multiplier must align with the episode input sample count: "
                f"{tuple(multiplier.shape)} vs {int(sample_count)}"
            )
        return multiplier

    def _sample_weight(
        self,
        batch: Dict[str, Any],
        *,
        device: torch.device,
        sample_count: int,
    ) -> torch.Tensor:
        sample_weight = batch.get("sample_weight")
        if sample_weight is None:
            return torch.ones(sample_count, dtype=torch.float32, device=device)
        if not isinstance(sample_weight, torch.Tensor):
            sample_weight = torch.tensor(sample_weight, dtype=torch.float32, device=device)
        sample_weight = sample_weight.to(device=device, dtype=torch.float32).view(-1)
        if sample_weight.numel() == 1 and int(sample_count) != 1:
            sample_weight = sample_weight.expand(int(sample_count))
        if int(sample_weight.numel()) != int(sample_count):
            raise ValueError(
                "sample_weight must align with the episode input sample count: "
                f"{tuple(sample_weight.shape)} vs {int(sample_count)}"
            )
        return sample_weight

    def _effective_sample_weight(
        self,
        batch: Dict[str, Any],
        *,
        device: torch.device,
        sample_count: int,
    ) -> torch.Tensor:
        return self._sample_loss_multiplier(
            batch,
            device=device,
            sample_count=sample_count,
        ) * self._sample_weight(
            batch,
            device=device,
            sample_count=sample_count,
        )

    def _has_trainable_weight(self, episode_input: Dict[str, Any]) -> bool:
        sample_count = int(self._episode_input_sample_count(episode_input))
        if sample_count <= 0:
            return False
        effective_weight = self._effective_sample_weight(
            episode_input,
            device=torch.device("cpu"),
            sample_count=sample_count,
        )
        return bool(torch.any(effective_weight > 0))

    def _zero_reference_token_log_probs_for_episode(
        self,
        episode_input: Dict[str, Any],
    ) -> torch.Tensor:
        completion_ids = episode_input.get("completion_ids")
        if not isinstance(completion_ids, torch.Tensor):
            return torch.zeros((0, 0), dtype=torch.float32, device=torch.device("cpu"))
        if completion_ids.ndim == 1:
            return torch.zeros(
                (1, int(completion_ids.shape[0])),
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
        return torch.zeros(
            tuple(int(dim) for dim in completion_ids.shape),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

    def _compute_reference_token_log_probs_for_batch(
        self,
        reference_model: Any,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        prepared_microbatch = self._prepare_device_microbatch_for_completion_only(batch)
        with torch.inference_mode():
            reference_token_log_probs, _ = compute_completion_only_token_log_probs_from_prepared_inputs(
                model=reference_model,
                model_inputs=prepared_microbatch["model_inputs"],
                completion_ids=prepared_microbatch["completion_ids"],
                completion_mask=prepared_microbatch["completion_mask"],
                logits_to_keep=int(prepared_microbatch["logits_to_keep"]),
                temperature=self.policy_temperature,
                log_runtime_details=False,
            )
        return reference_token_log_probs.detach().to(dtype=torch.float32).cpu()

    def _iter_reference_prefetch_batches(
        self,
        episode_input: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        # Keep reference prefetch aligned with the active loss path: score each
        # merged episode batch in one forward instead of reintroducing tiny
        # multimodal microbatches during cache population.
        return [episode_input]

    def _prefetch_reference_log_probs(
        self,
        wrapped_model: Any,
        episode_inputs: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        del wrapped_model
        episodes = list(episode_inputs or [])
        if float(self.kl_beta) == 0.0:
            return episodes
        if self.reference_model is None:
            return episodes
        try:
            target_device = next(self.reference_model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")
        grouped_bucket_count = int(len(episodes))
        runtime_log(
            "rl reference prefetch start: "
            f"episode_inputs={len(episodes)} "
            f"target_device={target_device} "
            f"grouped_buckets={grouped_bucket_count}",
            runtime=distributed_runtime_from_env(),
            main_process_only=True,
        )
        for episode_input in episodes:
            if not self._has_trainable_weight(episode_input):
                episode_input["reference_token_log_probs"] = (
                    self._zero_reference_token_log_probs_for_episode(episode_input)
                )
                continue
            bucket_summary = _summarize_episode_input_bucket_for_rank_debug([episode_input])
            bucket_start = time.perf_counter()
            runtime_log(
                "rl reference prefetch bucket start: "
                f"samples={bucket_summary['samples']} "
                f"rows={bucket_summary['rows']}",
                runtime=distributed_runtime_from_env(),
                main_process_only=True,
            )
            sub_batches = self._iter_reference_prefetch_batches(episode_input)
            pieces: List[torch.Tensor] = []
            for sub_batch in sub_batches:
                device_sub_batch = self._move_episode_input_to_device(sub_batch, device=target_device)
                pieces.append(
                    self._compute_reference_token_log_probs_for_batch(
                        self.reference_model,
                        device_sub_batch,
                    )
                )
            if pieces:
                episode_input["reference_token_log_probs"] = torch.cat(pieces, dim=0)
            else:
                episode_input["reference_token_log_probs"] = (
                    self._zero_reference_token_log_probs_for_episode(episode_input)
                )
            runtime_log(
                "rl reference prefetch bucket end: "
                f"samples={bucket_summary['samples']} "
                f"elapsed_sec={time.perf_counter() - bucket_start:.3f}",
                runtime=distributed_runtime_from_env(),
                main_process_only=True,
            )
        return episodes

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

    def _sequence_pad_values(self) -> Dict[str, Tuple[Any, str]]:
        return {
            "prompt_ids": (self._pad_token_id(), "left"),
            "prompt_mask": (0, "left"),
            "completion_ids": (self._pad_token_id(), "right"),
            "completion_mask": (0, "right"),
            "old_policy_token_log_probs": (0.0, "right"),
        }

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
        max_seq_len = max(int(tensor.shape[-1]) for tensor in tensors)
        padded_tensors: List[torch.Tensor] = []
        for tensor in tensors:
            pad_width = max_seq_len - int(tensor.shape[-1])
            if pad_width <= 0:
                padded_tensors.append(tensor)
                continue
            pad_shape = list(tensor.shape)
            pad_shape[-1] = int(pad_width)
            pad_tensor = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
            if str(pad_side) == "right":
                padded_tensors.append(torch.cat([tensor, pad_tensor], dim=-1))
            else:
                padded_tensors.append(torch.cat([pad_tensor, tensor], dim=-1))
        return torch.cat(padded_tensors, dim=0)

    def _is_full_sequence_aligned_tensor(
        self,
        tensor: torch.Tensor,
        episode_input: Dict[str, Any],
    ) -> bool:
        prompt_ids = episode_input.get("prompt_ids")
        completion_ids = episode_input.get("completion_ids")
        if not isinstance(prompt_ids, torch.Tensor) or not isinstance(completion_ids, torch.Tensor):
            return False
        if tensor.ndim < 2 or prompt_ids.ndim != 2 or completion_ids.ndim != 2:
            return False
        batch_size = int(prompt_ids.shape[0])
        if int(completion_ids.shape[0]) != batch_size or int(tensor.shape[0]) != batch_size:
            return False
        expected_width = int(prompt_ids.shape[-1]) + int(completion_ids.shape[-1])
        return int(tensor.shape[-1]) == expected_width

    def _pad_full_sequence_aligned_and_concat(
        self,
        tensors: Sequence[torch.Tensor],
        episode_inputs: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        prompt_lengths = [int(batch["prompt_ids"].shape[-1]) for batch in episode_inputs]
        completion_lengths = [int(batch["completion_ids"].shape[-1]) for batch in episode_inputs]
        max_prompt_length = max(prompt_lengths)
        max_completion_length = max(completion_lengths)
        padded_tensors: List[torch.Tensor] = []
        for tensor, prompt_length, completion_length in zip(tensors, prompt_lengths, completion_lengths):
            left_pad = max_prompt_length - int(prompt_length)
            right_pad = max_completion_length - int(completion_length)
            if left_pad > 0:
                left_shape = list(tensor.shape)
                left_shape[-1] = int(left_pad)
                tensor = torch.cat(
                    [torch.zeros(left_shape, dtype=tensor.dtype, device=tensor.device), tensor],
                    dim=-1,
                )
            if right_pad > 0:
                right_shape = list(tensor.shape)
                right_shape[-1] = int(right_pad)
                tensor = torch.cat(
                    [tensor, torch.zeros(right_shape, dtype=tensor.dtype, device=tensor.device)],
                    dim=-1,
                )
            padded_tensors.append(tensor)
        return torch.cat(padded_tensors, dim=0)

    def _episode_input_merge_signature_entry(
        self,
        key: str,
        value: Any,
        episode_input: Dict[str, Any],
    ) -> Tuple[Any, ...]:
        if key == "multimodal_inputs":
            if not isinstance(value, dict):
                raise ValueError("Expected dict values for episode input key 'multimodal_inputs'.")
            return self._multimodal_inputs_signature(value)
        if key == "old_policy_token_log_probs" and value == _USE_CURRENT_POLICY_LOGPROBS_SENTINEL:
            return ("reuse_current_policy_logprobs",)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Expected tensor values for episode input key {key!r}.")
        if key in self._sequence_pad_values():
            return ("sequence_pad", str(value.dtype), tuple(value.shape[1:-1]))
        if value.ndim == 0:
            return ("stack_scalar", str(value.dtype))
        if self._is_full_sequence_aligned_tensor(value, episode_input):
            return ("full_sequence_aligned", str(value.dtype), tuple(value.shape[1:-1]))
        return ("concat", str(value.dtype), tuple(value.shape[1:]))

    def _episode_input_merge_signature(
        self,
        episode_input: Dict[str, Any],
    ) -> Tuple[Tuple[str, Tuple[Any, ...]], ...]:
        return tuple(
            (
                str(key),
                self._episode_input_merge_signature_entry(str(key), episode_input[key], episode_input),
            )
            for key in sorted(episode_input.keys())
        )

    def _multimodal_inputs_signature(self, value: Any) -> Tuple[Any, ...]:
        if isinstance(value, torch.Tensor):
            return ("tensor", str(value.dtype), tuple(value.shape))
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
        return ("python", type(value).__name__, repr(value))

    def _is_merge_fallback_error(self, exc: Exception) -> bool:
        error_message = str(exc)
        return (
            "Cannot merge episode inputs with inconsistent key presence" in error_message
            or "Unable to concatenate episode input key" in error_message
        )

    def _can_reuse_current_policy_as_old_logprobs(self) -> bool:
        steps_per_generation = int(getattr(self, "steps_per_generation", 1) or 1)
        if steps_per_generation <= 1:
            return True
        return (not bool(getattr(self, "use_liger_loss", False))) and steps_per_generation <= int(
            getattr(getattr(self, "args", None), "gradient_accumulation_steps", 1) or 1
        )

    def _resolve_liger_unwrapped_model(self, model: Any) -> Any:
        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None:
            unwrap_model = getattr(accelerator, "unwrap_model", None)
            if callable(unwrap_model):
                try:
                    unwrapped = unwrap_model(model)
                except Exception:
                    unwrapped = None
                if unwrapped is not None:
                    return unwrapped
        return _unwrap_model(model)

    def _log_liger_runtime_configuration_once(self) -> None:
        if bool(getattr(self, "_liger_runtime_logged", False)):
            return
        self._liger_runtime_logged = True
        _training_phase_runtime_log(
            "rl liger runtime ready: "
            f"use_liger_loss_requested={bool(getattr(self, 'use_liger_loss_requested', False))} "
            f"use_liger_loss_effective={bool(getattr(self, 'use_liger_loss_effective', False))} "
            f"liger_disable_reason={str(getattr(self, '_liger_runtime_disable_reason', None) or 'none')} "
            f"liger_compiled={bool(getattr(self, '_liger_compiled', False))} "
            f"liger_linear_head_path={str(getattr(self, '_liger_linear_head_path', '') or 'unknown')} "
            f"liger_hidden_state_path={str(getattr(self, '_liger_hidden_state_path', '') or 'unknown')}"
        )

    @staticmethod
    def _materialize_liger_tensor(value: Optional[torch.Tensor], *, device: torch.device, dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
        if value is None:
            return None
        tensor = value.to(device=device) if value.device != device else value
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor.clone()

    def _get_liger_last_hidden_state(
        self,
        *,
        unwrapped_model: Any,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        multimodal_inputs: Dict[str, Any],
        logits_to_keep: int,
    ) -> torch.Tensor:
        backbone = getattr(unwrapped_model, "model", None)
        if callable(backbone):
            try:
                backbone_outputs = backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **dict(multimodal_inputs or {}),
                )
                last_hidden_state = getattr(backbone_outputs, "last_hidden_state", None)
                if isinstance(last_hidden_state, torch.Tensor) and last_hidden_state.ndim == 3:
                    self._liger_hidden_state_path = "unwrapped_model.model.last_hidden_state"
                    return last_hidden_state[:, :-1, :][:, -int(logits_to_keep) :, :]
            except Exception as exc:
                _training_phase_runtime_log(
                    "rl liger hidden-state fallback: "
                    f"reason={exc.__class__.__name__}"
                )
        outputs = unwrapped_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **dict(multimodal_inputs or {}),
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("use_liger_loss=True requires model outputs to expose hidden_states.")
        self._liger_hidden_state_path = "model.output_hidden_states_fallback"
        return hidden_states[-1][:, :-1, :][:, -int(logits_to_keep) :, :]

    def _compute_old_policy_token_log_probs_for_episode_input(
        self,
        model: Any,
        *,
        episode_input: Dict[str, Any],
    ) -> torch.Tensor:
        prepared_microbatch = self._prepare_device_microbatch_for_completion_only(episode_input)
        with torch.inference_mode():
            old_policy_token_log_probs, _ = compute_completion_only_token_log_probs_from_prepared_inputs(
                model=model,
                model_inputs=prepared_microbatch["model_inputs"],
                completion_ids=prepared_microbatch["completion_ids"],
                completion_mask=prepared_microbatch["completion_mask"],
                logits_to_keep=int(prepared_microbatch["logits_to_keep"]),
                temperature=self.policy_temperature,
                log_runtime_details=False,
            )
        return old_policy_token_log_probs.detach().cpu()

    def _populate_old_policy_log_probs(
        self,
        model: Any,
        episode_inputs: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not episode_inputs:
            return []
        if self._can_reuse_current_policy_as_old_logprobs():
            populated: List[Dict[str, Any]] = []
            for episode_input in episode_inputs:
                entry = dict(episode_input or {})
                entry["old_policy_token_log_probs"] = _USE_CURRENT_POLICY_LOGPROBS_SENTINEL
                populated.append(entry)
            return populated
        pending_entries = []
        for episode_input in episode_inputs:
            entry = dict(episode_input or {})
            sample_count = int(self._episode_input_sample_count(entry))
            if sample_count > 0:
                effective_weight = self._effective_sample_weight(
                    entry,
                    device=torch.device("cpu"),
                    sample_count=sample_count,
                )
                if not bool(torch.any(effective_weight > 0)):
                    completion_ids = entry.get("completion_ids")
                    if isinstance(completion_ids, torch.Tensor):
                        entry["old_policy_token_log_probs"] = torch.zeros_like(
                            completion_ids,
                            dtype=torch.float32,
                        )
                        pending_entries.append({"episode_input": entry, "old_policy_token_log_probs": entry["old_policy_token_log_probs"]})
                        continue
            if not isinstance(entry.get("old_policy_token_log_probs"), torch.Tensor):
                entry.pop("old_policy_token_log_probs", None)
            pending_entries.append({"episode_input": entry})
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")
        grouped_episode_inputs = self._group_items_by_signature(
            pending_entries,
            signature_fn=lambda item: self._episode_input_merge_signature(item["episode_input"]),
        )
        runtime_log(
            "rl old_policy cache start: "
            f"episode_inputs={len(pending_entries)} "
            f"grouped_buckets={len(grouped_episode_inputs)} "
            f"target_device={target_device}",
            runtime=distributed_runtime_from_env(),
            main_process_only=True,
        )
        for bucket in grouped_episode_inputs:
            bucket_summary = _summarize_episode_input_bucket_for_rank_debug(
                [dict(entry.get("episode_input") or {}) for entry in bucket]
            )
            bucket_start = time.perf_counter()
            merge_mode = "single"
            runtime_log(
                "rl old_policy cache bucket start: "
                f"items={bucket_summary['items']} "
                f"samples={bucket_summary['samples']} "
                f"rows={bucket_summary['rows']}",
                runtime=distributed_runtime_from_env(),
                main_process_only=True,
            )
            device_episode_inputs = [
                self._move_episode_input_to_device(entry["episode_input"], device=target_device)
                for entry in bucket
            ]
            if len(device_episode_inputs) == 1:
                batched_log_probs = self._compute_old_policy_token_log_probs_for_episode_input(
                    model,
                    episode_input=device_episode_inputs[0],
                )
            else:
                try:
                    merged_episode_input = self._merge_episode_inputs(device_episode_inputs)
                    merge_mode = "merged"
                except ValueError as exc:
                    if not self._is_merge_fallback_error(exc):
                        raise
                    merge_mode = "fallback_per_episode"
                    for entry, episode_input in zip(bucket, device_episode_inputs):
                        entry["old_policy_token_log_probs"] = self._compute_old_policy_token_log_probs_for_episode_input(
                            model,
                            episode_input=episode_input,
                        )
                    runtime_log(
                        "rl old_policy cache bucket end: "
                        f"items={bucket_summary['items']} "
                        f"samples={bucket_summary['samples']} "
                        f"merge_mode={merge_mode} "
                        f"elapsed_sec={time.perf_counter() - bucket_start:.3f}",
                        runtime=distributed_runtime_from_env(),
                        main_process_only=True,
                    )
                    continue
                batched_log_probs = self._compute_old_policy_token_log_probs_for_episode_input(
                    model,
                    episode_input=merged_episode_input,
                )
            for row_index, entry in enumerate(bucket):
                completion_length = int(entry["episode_input"]["completion_ids"].shape[-1])
                entry["old_policy_token_log_probs"] = (
                    batched_log_probs[row_index : row_index + 1, :completion_length].clone()
                )
            runtime_log(
                "rl old_policy cache bucket end: "
                f"items={bucket_summary['items']} "
                f"samples={bucket_summary['samples']} "
                f"merge_mode={merge_mode} "
                f"elapsed_sec={time.perf_counter() - bucket_start:.3f}",
                runtime=distributed_runtime_from_env(),
                main_process_only=True,
            )
        populated: List[Dict[str, Any]] = []
        for entry in pending_entries:
            episode_input = dict(entry["episode_input"] or {})
            old_policy_token_log_probs = entry.get("old_policy_token_log_probs")
            if not isinstance(old_policy_token_log_probs, torch.Tensor):
                raise RuntimeError("Active RL failed to materialize old_policy_token_log_probs for an episode input.")
            episode_input["old_policy_token_log_probs"] = old_policy_token_log_probs
            populated.append(episode_input)
        return populated

    def _merge_episode_inputs(self, episode_inputs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        sequence_pad_values = self._sequence_pad_values()
        ordered_keys: List[str] = []
        for batch in episode_inputs:
            for key in batch.keys():
                if key not in ordered_keys:
                    ordered_keys.append(key)
        for key in ordered_keys:
            values = [batch[key] for batch in episode_inputs if key in batch]
            if len(values) != len(episode_inputs):
                raise ValueError(f"Cannot merge episode inputs with inconsistent key presence for {key!r}.")
            if key == "multimodal_inputs":
                merged[key] = self._merge_multimodal_input_samples(values)
                continue
            if key == "old_policy_token_log_probs" and all(
                value == _USE_CURRENT_POLICY_LOGPROBS_SENTINEL for value in values
            ):
                merged[key] = _USE_CURRENT_POLICY_LOGPROBS_SENTINEL
                continue
            if not all(isinstance(value, torch.Tensor) for value in values):
                raise ValueError(f"Expected tensor values for merged episode input key {key!r}.")
            if key in sequence_pad_values:
                pad_value, pad_side = sequence_pad_values[key]
                merged[key] = self._pad_and_concat(values, pad_value=pad_value, pad_side=pad_side)
                continue
            if values[0].ndim == 0:
                merged[key] = torch.stack(values, dim=0)
                continue
            if all(
                self._is_full_sequence_aligned_tensor(value, batch)
                for value, batch in zip(values, episode_inputs)
            ):
                merged[key] = self._pad_full_sequence_aligned_and_concat(values, episode_inputs)
                continue
            try:
                merged[key] = torch.cat(values, dim=0)
            except Exception as exc:
                raise ValueError(f"Unable to concatenate episode input key {key!r}.") from exc
        return merged

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

    def _episode_input_multimodal_inputs(self, episode_input: Dict[str, Any]) -> Dict[str, Any]:
        reserved = {
            "prompt_ids",
            "prompt_mask",
            "completion_ids",
            "completion_mask",
            "prompt_token_count",
            "completion_token_count",
            "advantages",
            "old_policy_token_log_probs",
            "sample_loss_multiplier",
            "sample_weight",
            "multimodal_inputs",
        }
        if "multimodal_inputs" in episode_input:
            multimodal_inputs = episode_input.get("multimodal_inputs")
            if multimodal_inputs is None:
                return {}
            if isinstance(multimodal_inputs, dict):
                return multimodal_inputs
            if isinstance(multimodal_inputs, list):
                return self._collate_multimodal_input_samples(multimodal_inputs)
            raise ValueError("Episode input `multimodal_inputs` must be dict or list[dict].")
        legacy_keys = [str(key) for key in episode_input.keys() if key not in reserved]
        if legacy_keys:
            raise ValueError(
                "Legacy active RL episode_input multimodal layout detected; regenerate materialized/runtime cache. "
                f"Offending top-level keys: {', '.join(sorted(legacy_keys))}"
            )
        return {}

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
            if first_value.ndim == 0:
                return torch.stack(list(values), dim=0)
            return torch.cat(list(values), dim=0)
        if isinstance(first_value, dict):
            first_keys = set(first_value.keys())
            for value in values[1:]:
                if not isinstance(value, dict) or set(value.keys()) != first_keys:
                    raise ValueError("Cannot collate multimodal_inputs dicts with inconsistent keys.")
            return {
                str(key): self._collate_multimodal_input_value([value[key] for value in values])
                for key in sorted(first_keys)
            }
        if isinstance(first_value, list):
            if not all(isinstance(value, list) and len(value) == len(first_value) for value in values):
                raise ValueError("Cannot collate multimodal_inputs lists with inconsistent lengths.")
            return [
                self._collate_multimodal_input_value([value[index] for value in values])
                for index in range(len(first_value))
            ]
        if isinstance(first_value, tuple):
            if not all(isinstance(value, tuple) and len(value) == len(first_value) for value in values):
                raise ValueError("Cannot collate multimodal_inputs tuples with inconsistent lengths.")
            return tuple(
                self._collate_multimodal_input_value([value[index] for value in values])
                for index in range(len(first_value))
            )
        if not all(value == first_value for value in values[1:]):
            raise ValueError("Cannot collate multimodal_inputs python values with inconsistent contents.")
        return copy.deepcopy(first_value)

    def _aggregate_generation_step_payload(
        self,
        item_payloads: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        local_rollout_groups: List[Dict[str, Any]] = []
        rollout_metric_values = self._new_rollout_metric_lists()
        runtime_stats = self._new_runtime_stats()
        video_ids: List[str] = []
        local_rank = int(_distributed_rank())
        for item_index, payload in enumerate(item_payloads):
            video_ids.append(str(payload.get("video_id") or ""))
            payload_rollout_groups = list(payload.get("rollout_groups") or [])
            if not payload_rollout_groups and payload.get("episode_inputs"):
                payload_rollout_groups = [
                    {
                        "video_id": str(payload.get("video_id") or ""),
                        "group_id": str(payload.get("video_id") or ""),
                        "generation_id": -1,
                        "source_item_index": int(item_index),
                        "source_rollout_index": 0,
                        "episode_inputs": [dict(episode_input or {}) for episode_input in list(payload.get("episode_inputs") or [])],
                    }
                ]
            for rollout_group in payload_rollout_groups:
                normalized_group = copy.deepcopy(dict(rollout_group or {}))
                normalized_group["video_id"] = str(normalized_group.get("video_id") or payload.get("video_id") or "")
                normalized_group["source_rank"] = int(local_rank)
                normalized_group["source_item_index"] = int(
                    normalized_group.get("source_item_index") if normalized_group.get("source_item_index") is not None else item_index
                )
                normalized_group["sample_count"] = int(_rollout_group_sample_count(normalized_group))
                local_rollout_groups.append(normalized_group)
            for key, values in dict(payload.get("rollout_metric_values") or {}).items():
                rollout_metric_values.setdefault(key, [])
                rollout_metric_values[key].extend([_safe_float(value) for value in (values or [])])
            for key, value in dict(payload.get("runtime_stats") or {}).items():
                runtime_stats[key] = int(runtime_stats.get(key, 0)) + int(value or 0)
        runtime_log(
            f"rl rank rollout_group summary: rows={[_summarize_rollout_group_for_rank_debug(group) for group in local_rollout_groups]}",
            runtime=distributed_runtime_from_env(),
            main_process_only=False,
        )
        gathered_rollout_groups = _distributed_gather_object(local_rollout_groups)
        balanced_rollout_groups = _stable_balance_rollout_groups_across_ranks(gathered_rollout_groups)
        local_assigned_rollout_groups = balanced_rollout_groups[local_rank] if local_rank < len(balanced_rollout_groups) else []
        runtime_log(
            "rl global rollout_group slice summary: "
            f"total_groups={sum(len(groups) for groups in balanced_rollout_groups)} "
            f"assigned_groups={len(local_assigned_rollout_groups)} "
            f"assigned_samples={sum(int(group.get('sample_count') or 0) for group in local_assigned_rollout_groups)} "
            f"rows={[_summarize_rollout_group_for_rank_debug(group) for group in local_assigned_rollout_groups]}",
            runtime=distributed_runtime_from_env(),
            main_process_only=False,
        )
        episode_inputs = _flatten_rollout_groups_to_episode_inputs(local_assigned_rollout_groups)
        aggregated_metrics = {
            key: (sum(values) / float(len(values)) if values else 0.0)
            for key, values in rollout_metric_values.items()
        }
        runtime_stats["raw_local_rollout_group_count_before_slice"] = int(len(local_rollout_groups))
        runtime_stats["raw_local_rollout_group_count_after_slice"] = int(len(local_assigned_rollout_groups))
        runtime_stats["raw_local_episode_input_count"] = int(len(episode_inputs))
        runtime_stats["raw_local_episode_input_count_after_aggregate"] = int(len(episode_inputs))
        return {
            "episode_inputs": episode_inputs,
            "rollout_metrics": aggregated_metrics,
            "budgeting_metrics": self.get_budget_drop_metrics(),
            "runtime_stats": runtime_stats,
            "video_ids": video_ids,
        }

    def _episode_input_has_nonzero_advantage(
        self,
        episode_input: Dict[str, Any],
    ) -> bool:
        advantages = episode_input.get("advantages")
        if advantages is None:
            return False
        if not isinstance(advantages, torch.Tensor):
            advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = advantages.to(dtype=torch.float32).view(-1)
        return bool(torch.any(torch.abs(advantages) > 1e-8))

    def _payload_has_nonzero_advantage(
        self,
        payload: Dict[str, Any],
    ) -> bool:
        episode_inputs = list(payload.get("episode_inputs") or [])
        return any(self._episode_input_has_nonzero_advantage(episode_input) for episode_input in episode_inputs)

    def _clone_generation_step_payload_for_replay(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        episode_inputs = [
            self._episode_input_cpu_copy(dict(episode_input or {}))
            for episode_input in list(payload.get("episode_inputs") or [])
        ]
        return {
            "episode_inputs": episode_inputs,
            "rollout_metrics": copy.deepcopy(dict(payload.get("rollout_metrics") or {})),
            "budgeting_metrics": copy.deepcopy(dict(payload.get("budgeting_metrics") or {})),
            "runtime_stats": copy.deepcopy(dict(payload.get("runtime_stats") or {})),
            "video_ids": [str(video_id or "") for video_id in list(payload.get("video_ids") or [])],
        }

    def _maybe_store_nonzero_advantage_payload(
        self,
        payload: Dict[str, Any],
    ) -> None:
        del payload
        return None

    def _maybe_replay_zero_advantage_payload(
        self,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not hasattr(self, "_zero_advantage_replay_uses"):
            self._zero_advantage_replay_uses = 0
        if not hasattr(self, "_zero_advantage_replay_misses"):
            self._zero_advantage_replay_misses = 0
        self._zero_advantage_replay_misses += 1
        return payload

    def _build_generation_step_payloads(
        self,
        generation_items: Sequence[Dict[str, Any]],
        rollout_model: Any,
    ) -> List[Dict[str, Any]]:
        item_payloads: List[Dict[str, Any]] = []
        with torch.inference_mode():
            grouped_rollouts = self._generate_scored_rollouts_batch(
                generation_items,
                rollout_model,
                progress=None,
            )
            for item, item_rollouts in zip(generation_items, grouped_rollouts):
                payload = self._build_generation_item_payload_from_rollouts(item, item_rollouts)
                item_payloads.append(payload)
                runtime_stats = dict(payload.get("runtime_stats") or {})
                self._groups_filtered_by_min_weight += int(runtime_stats.get("groups_filtered_by_min_weight", 0))
                self._groups_all_zero_advantage += int(runtime_stats.get("groups_all_zero_advantage", 0))
        step_payloads: List[Dict[str, Any]] = []
        step_size = max(1, int(self._generation_step_batch_size))
        for offset in range(0, len(item_payloads), step_size):
            payload = self._aggregate_generation_step_payload(item_payloads[offset : offset + step_size])
            episode_inputs = list(payload.get("episode_inputs") or [])
            if episode_inputs:
                if self._can_reuse_current_policy_as_old_logprobs():
                    payload["episode_inputs"] = [
                        {
                            **dict(episode_input),
                            "old_policy_token_log_probs": _USE_CURRENT_POLICY_LOGPROBS_SENTINEL,
                        }
                        if "old_policy_token_log_probs" not in dict(episode_input)
                        or dict(episode_input).get("old_policy_token_log_probs") is None
                        else dict(episode_input)
                        for episode_input in episode_inputs
                    ]
                else:
                    runtime_log(
                        "rl generation stage: before prefetch_old_policy_log_probs "
                        f"episode_inputs={len(episode_inputs)}",
                        runtime=distributed_runtime_from_env(),
                        main_process_only=True,
                    )
                    payload["episode_inputs"] = self._populate_old_policy_log_probs(
                        rollout_model,
                        episode_inputs,
                    )
                    runtime_log(
                        "rl generation stage: after prefetch_old_policy_log_probs "
                        f"episode_inputs={len(list(payload.get('episode_inputs') or []))}",
                        runtime=distributed_runtime_from_env(),
                        main_process_only=True,
                    )
            step_payloads.append(payload)
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

    def _resolve_episode_input_replay_config(self) -> Dict[str, Any]:
        config = getattr(self, "aligned_grpo_config", None)

        def _resolve(name: str, default: Any) -> Any:
            value = getattr(self, name, None)
            if value is not None:
                return value
            if config is not None:
                if isinstance(config, dict) and name in config:
                    return config.get(name, default)
                config_value = getattr(config, name, None)
                if config_value is not None:
                    return config_value
            args = getattr(self, "args", None)
            if args is not None:
                args_value = getattr(args, name, None)
                if args_value is not None:
                    return args_value
            return default

        return {
            "enabled": bool(_resolve("replay_buffer_enable", False)),
            "type": str(_resolve("replay_buffer_type", "ssr") or "ssr").strip().lower(),
            "capacity": max(0, int(_resolve("replay_buffer_capacity", 2) or 0)),
            "alpha": float(_resolve("replay_buffer_alpha", 1.0) or 1.0),
        }

    def _ensure_episode_input_replay_state(self) -> Dict[str, Any]:
        state = getattr(self, "_episode_input_replay_state", None)
        if isinstance(state, dict):
            return state
        state = {
            "buffer": [],
            "serial": 0,
        }
        self._episode_input_replay_state = state
        return state

    def _clone_episode_input_batch_for_replay(
        self,
        episode_inputs: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        return [
            self._episode_input_cpu_copy(episode_input)
            for episode_input in episode_inputs
        ]

    def _record_materialized_episode_inputs_for_replay(
        self,
        episode_inputs: Sequence[Dict[str, Any]],
    ) -> None:
        replay_config = self._resolve_episode_input_replay_config()
        if not replay_config["enabled"] or replay_config["capacity"] <= 0:
            return
        trainable_samples = int(self._count_local_trainable_samples(list(episode_inputs)))
        if trainable_samples <= 0:
            return
        state = self._ensure_episode_input_replay_state()
        state["serial"] = int(state.get("serial", 0)) + 1
        buffer_entries = list(state.get("buffer") or [])
        buffer_entries.append(
            {
                "episode_inputs": self._clone_episode_input_batch_for_replay(episode_inputs),
                "trainable_samples": trainable_samples,
                "serial": int(state["serial"]),
            }
        )
        capacity = int(replay_config["capacity"])
        if len(buffer_entries) > capacity:
            buffer_entries = buffer_entries[-capacity:]
        state["buffer"] = buffer_entries

    def _sample_materialized_episode_inputs_from_replay(self) -> Optional[List[Dict[str, Any]]]:
        replay_config = self._resolve_episode_input_replay_config()
        if not replay_config["enabled"] or replay_config["capacity"] <= 0:
            return None
        state = self._ensure_episode_input_replay_state()
        buffer_entries = list(state.get("buffer") or [])
        if not buffer_entries:
            return None
        replay_type = str(replay_config["type"] or "ssr").strip().lower()
        if replay_type == "dapo":
            alpha = float(replay_config["alpha"])
            selected_entry = max(
                buffer_entries,
                key=lambda entry: (
                    float(max(1, int(entry.get("trainable_samples", 0)))) ** alpha,
                    int(entry.get("serial", 0)),
                ),
            )
        else:
            selected_entry = buffer_entries[-1]
        return self._clone_episode_input_batch_for_replay(
            list(selected_entry.get("episode_inputs") or [])
        )

    def _maybe_fill_empty_episode_inputs_from_replay(
        self,
        inputs: Dict[str, Any],
        runtime_stats: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        episode_inputs = list(inputs.get("episode_inputs") or [])
        if episode_inputs and int(self._count_local_trainable_samples(episode_inputs)) > 0:
            return episode_inputs
        runtime_stats["replay_fill_attempts"] = int(runtime_stats.get("replay_fill_attempts", 0)) + 1
        replay_episode_inputs = self._sample_materialized_episode_inputs_from_replay()
        if not replay_episode_inputs:
            runtime_stats["replay_fill_misses"] = int(runtime_stats.get("replay_fill_misses", 0)) + 1
            return []
        runtime_stats["replay_fill_hits"] = int(runtime_stats.get("replay_fill_hits", 0)) + 1
        runtime_stats["replay_fill_samples"] = int(
            runtime_stats.get("replay_fill_samples", 0)
        ) + int(self._count_local_trainable_samples(replay_episode_inputs))
        inputs["episode_inputs"] = replay_episode_inputs
        inputs["runtime_stats"] = runtime_stats
        return replay_episode_inputs

    def _materialize_episode_inputs(
        self,
        episode_inputs: Sequence[Dict[str, Any]],
        *,
        device: torch.device,
    ) -> List[Dict[str, Any]]:
        del device
        materialized = [
            self._episode_input_cpu_copy(episode_input)
            for episode_input in episode_inputs
        ]
        grouped_batches = self._group_items_by_signature(
            materialized,
            signature_fn=self._episode_input_merge_signature,
        )
        merged_batches: List[Dict[str, Any]] = []
        for bucket in grouped_batches:
            if len(bucket) == 1:
                merged_batches.append(bucket[0])
                continue
            try:
                merged_batches.append(self._merge_episode_inputs(bucket))
            except ValueError as exc:
                if not self._is_merge_fallback_error(exc):
                    raise
                self._materialize_fallback_batches += 1
                merged_batches.extend(bucket)
        self._record_materialized_episode_inputs_for_replay(merged_batches)
        return merged_batches

    def _slice_episode_input_sample_range(
        self,
        episode_input: Dict[str, Any],
        *,
        start_index: int,
        end_index: int,
    ) -> Dict[str, Any]:
        sample_count = self._episode_input_sample_count(episode_input)
        sliced: Dict[str, Any] = {}
        for key, value in episode_input.items():
            if key == "multimodal_inputs":
                sliced[key] = self._slice_multimodal_input_samples(
                    value,
                    start_index=start_index,
                    end_index=end_index,
                    sample_count=sample_count,
                )
                continue
            if isinstance(value, torch.Tensor):
                if value.ndim == 0 or int(value.shape[0]) != int(sample_count):
                    sliced[key] = value
                else:
                    sliced[key] = value[int(start_index) : int(end_index)]
            else:
                sliced[key] = copy.deepcopy(value)
        return sliced

    def _slice_multimodal_input_samples(
        self,
        value: Any,
        *,
        start_index: int,
        end_index: int,
        sample_count: int,
    ) -> Any:
        if isinstance(value, list):
            if len(value) == int(sample_count):
                return copy.deepcopy(value[int(start_index) : int(end_index)])
            return copy.deepcopy(value)
        return copy.deepcopy(value)

    def _iter_loss_microbatches(
        self,
        episode_input: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        # Active RL now runs each merged episode batch in a single completion-only
        # forward to avoid the heavy latency penalty from repeated multimodal
        # microbatch slicing.
        return [episode_input]

    def _align_episode_inputs_across_ranks(
        self,
        episode_inputs: Sequence[Dict[str, Any]],
        *,
        device: torch.device,
        runtime_stats: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        local_batches = list(episode_inputs or [])
        local_count = int(len(local_batches))
        all_ranks_have_batches, any_rank_has_batches = _distributed_bool_consensus(
            local_count > 0,
            device=device,
        )
        if any_rank_has_batches and not all_ranks_have_batches:
            donor_episode_input = _distributed_first_available_object(
                self._episode_input_cpu_copy(local_batches[0]) if local_batches else None,
                device=device,
            )
            if local_count <= 0 and donor_episode_input is not None:
                local_batches = [self._clone_episode_input_as_noop(donor_episode_input)]
                runtime_stats["ddp_noop_padded_episode_inputs"] = 1
                self._ddp_noop_padded_episode_inputs += 1
        return local_batches

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
                f"episode_inputs={int(runtime_stats.get('raw_local_episode_input_count', 0))} "
                f"trainable_samples={int(trainable_samples)} "
                f"fecv_failures={int(runtime_stats.get('local_fecv_failure_count', 0))} "
                f"min_weight_drops={int(runtime_stats.get('groups_filtered_by_min_weight', 0))} "
                f"missing_episode_training_feature={int(runtime_stats.get('missing_episode_training_feature_count', 0))}"
            ),
            runtime=runtime,
            main_process_only=False,
        )

    def _count_local_trainable_samples(
        self,
        episode_inputs: Sequence[Dict[str, Any]],
    ) -> int:
        total = 0
        for batch in list(episode_inputs or []):
            if self._has_trainable_weight(batch):
                sample_count = int(self._episode_input_sample_count(batch))
                effective_weight = self._effective_sample_weight(
                    batch,
                    device=torch.device("cpu"),
                    sample_count=sample_count,
                )
                total += int((effective_weight > 0).sum().item())
        return int(total)

    def _maybe_skip_empty_training_step(
        self,
        model: Any,
        inputs: Dict[str, Any],
    ) -> Optional[torch.Tensor]:
        episode_inputs = list(inputs.get("episode_inputs") or [])
        runtime_stats = dict(inputs.get("runtime_stats") or {})
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")
        local_trainable_samples = int(self._count_local_trainable_samples(episode_inputs))
        global_trainable_samples = int(_distributed_sum_int(local_trainable_samples, device=target_device))
        if global_trainable_samples > 0:
            return None
        replay_episode_inputs = self._maybe_fill_empty_episode_inputs_from_replay(inputs, runtime_stats)
        if replay_episode_inputs:
            episode_inputs = list(replay_episode_inputs)
            local_trainable_samples = int(self._count_local_trainable_samples(episode_inputs))
            global_trainable_samples = int(_distributed_sum_int(local_trainable_samples, device=target_device))
            if global_trainable_samples > 0:
                return None
        reason = "all_empty_episode_inputs" if not episode_inputs else "all_empty_trainable_samples"
        self._maybe_log_empty_batch_rank_summary(
            reason=reason,
            runtime_stats=runtime_stats,
            trainable_samples=local_trainable_samples,
        )
        self._skip_empty_training_steps += 1
        return torch.zeros((), dtype=torch.float32, device=target_device)

    def _should_use_immediate_microbatch_backward(self) -> bool:
        return True

    def _backward_loss_scalar(
        self,
        loss: torch.Tensor,
        *,
        optimizer: Any,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        kwargs: Dict[str, Any] = {}
        args = getattr(self, "args", None)
        if isinstance(loss, torch.Tensor):
            if int(loss.numel()) != 1:
                _training_phase_runtime_log(
                    "rl loss scalarize debug: "
                    f"shape={tuple(int(v) for v in loss.shape)} "
                    f"numel={int(loss.numel())} "
                    f"requires_grad={bool(loss.requires_grad)} "
                    f"grad_fn={type(loss.grad_fn).__name__ if loss.grad_fn is not None else 'none'}"
                )
                loss = loss.sum()
            if loss.ndim != 0:
                loss = loss.reshape(())
        if int(getattr(args, "n_gpu", 1) or 1) > 1:
            loss = loss.mean()

        if bool(getattr(self, "use_apex", False)):
            from apex import amp

            _training_phase_runtime_log("rl backward start: backend=apex")
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            _training_phase_runtime_log("rl backward end: backend=apex")
            return loss.detach()

        model_accepts_loss_kwargs = bool(getattr(self, "model_accepts_loss_kwargs", False))
        compute_loss_func = getattr(self, "compute_loss_func", None)
        distributed_type = getattr(getattr(self, "accelerator", None), "distributed_type", None)
        is_deepspeed = str(distributed_type).lower().endswith("deepspeed")
        if (
            (not model_accepts_loss_kwargs or num_items_in_batch is None)
            and compute_loss_func is None
            and not is_deepspeed
        ):
            loss = loss / int(getattr(self, "current_gradient_accumulation_steps", 1) or 1)
        if is_deepspeed:
            kwargs["scale_wrt_gas"] = False
        _training_phase_runtime_log(
            "rl backward start: "
            f"backend={'deepspeed' if is_deepspeed else 'accelerate'}"
        )
        self.accelerator.backward(loss, **kwargs)
        _training_phase_runtime_log(
            "rl backward end: "
            f"backend={'deepspeed' if is_deepspeed else 'accelerate'}"
        )
        return loss.detach()

    def _compute_sample_losses_for_device_microbatch(
        self,
        model: Any,
        *,
        device_microbatch: Dict[str, Any],
        batch_index: int,
        batch_count: int,
        microbatch_index: int,
        microbatch_count: int,
    ) -> Optional[torch.Tensor]:
        liger_unwrapped_model = self._resolve_liger_unwrapped_model(model) if self.use_liger_loss else None
        if self.use_liger_loss:
            use_forward_redirection = self._should_use_forward_redirection_for_liger()
            dispatch_name = (
                "forward_redirection_zero3"
                if use_forward_redirection
                else "direct_unwrapped_model_with_head_gather"
            )
            _training_phase_runtime_log(
                "rl compute_loss liger path dispatch: "
                f"batch={batch_index}/{batch_count} "
                f"microbatch={microbatch_index}/{microbatch_count} "
                f"dispatch={dispatch_name}"
            )
            if use_forward_redirection:
                _training_phase_runtime_log(
                    "rl compute_loss liger forward_redirection enter: "
                    f"batch={batch_index}/{batch_count} "
                    f"microbatch={microbatch_index}/{microbatch_count}"
                )
                sample_losses = self._forward_redirection(
                    model,
                    liger_unwrapped_model,
                    self._compute_liger_loss_for_batch,
                    liger_unwrapped_model,
                    device_microbatch,
                )
                _training_phase_runtime_log(
                    "rl compute_loss liger forward_redirection exit: "
                    f"batch={batch_index}/{batch_count} "
                    f"microbatch={microbatch_index}/{microbatch_count}"
                )
                return sample_losses
            return self._compute_liger_loss_for_batch(
                unwrapped_model=liger_unwrapped_model,
                batch=device_microbatch,
            )
        return self._compute_sample_losses_for_batch(model=model, batch=device_microbatch)

    def _flatten_loss_microbatch_entries(
        self,
        episode_inputs: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        batch_count = int(len(episode_inputs))
        for batch_index, batch in enumerate(episode_inputs, start=1):
            microbatches = list(self._iter_loss_microbatches(batch))
            microbatch_count = int(len(microbatches))
            for microbatch_index, microbatch in enumerate(microbatches, start=1):
                entries.append(
                    {
                        "batch_index": int(batch_index),
                        "batch_count": batch_count,
                        "microbatch_index": int(microbatch_index),
                        "microbatch_count": microbatch_count,
                        "is_last_in_batch": bool(microbatch_index == microbatch_count),
                        "local_effective_weight_sum": None,
                        "local_active_samples": None,
                        "microbatch": microbatch,
                    }
                )
        return entries

    def _local_effective_weight_summary_for_microbatch(
        self,
        microbatch: Dict[str, Any],
    ) -> Tuple[float, int]:
        completion_mask = microbatch.get("completion_mask")
        if not isinstance(completion_mask, torch.Tensor) or not bool(torch.any(completion_mask.to(dtype=torch.bool))):
            return 0.0, 0
        sample_count = int(self._episode_input_sample_count(microbatch))
        effective_weight = self._effective_sample_weight(
            microbatch,
            device=torch.device("cpu"),
            sample_count=sample_count,
        ).to(dtype=torch.float32)
        return float(effective_weight.sum().item()), int((effective_weight > 0).sum().item())

    def _prepare_device_microbatch_for_completion_only(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt_ids = batch["prompt_ids"]
        prompt_mask = batch["prompt_mask"]
        completion_ids = batch["completion_ids"]
        completion_mask = batch["completion_mask"]
        multimodal_inputs = self._episode_input_multimodal_inputs(batch)
        model_inputs, logits_to_keep = build_completion_only_model_inputs(
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            completion_ids=completion_ids,
            completion_mask=completion_mask,
            multimodal_inputs=multimodal_inputs,
        )
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "multimodal_inputs": multimodal_inputs,
            "model_inputs": model_inputs,
            "logits_to_keep": int(logits_to_keep),
        }

    def _training_step_with_immediate_microbatch_backward(
        self,
        model: Any,
        inputs: Dict[str, Any],
        *,
        optimizer: Any,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        episode_inputs = list(inputs.get("episode_inputs") or [])
        runtime_stats = dict(inputs.get("runtime_stats") or {})
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")
        if not episode_inputs:
            episode_inputs = self._maybe_fill_empty_episode_inputs_from_replay(inputs, runtime_stats)
        if not episode_inputs:
            self._maybe_log_empty_batch_rank_summary(
                reason="all_empty_episode_inputs",
                runtime_stats=runtime_stats,
                trainable_samples=0,
            )
            zero_loss = _zero_loss_from_model(model)
            return self._backward_loss_scalar(
                zero_loss,
                optimizer=optimizer,
                num_items_in_batch=num_items_in_batch,
            )

        self._ensure_liger_runtime_ready(model)
        compute_loss_start = time.perf_counter()
        _training_phase_runtime_log(
            "rl compute_loss start: "
            f"episode_batches={len(episode_inputs)} "
            f"use_liger_loss_requested={bool(getattr(self, 'use_liger_loss_requested', False))} "
            f"use_liger_loss_effective={bool(getattr(self, 'use_liger_loss_effective', False))} "
            f"liger_disable_reason={str(getattr(self, '_liger_runtime_disable_reason', None) or 'none')} "
            f"loss_microbatch={int(getattr(self, 'compute_loss_microbatch_size', 1) or 1)} "
            "immediate_backward=true"
        )
        microbatch_entries = self._flatten_loss_microbatch_entries(episode_inputs)
        local_total_effective_weight = 0.0
        total_active_samples = 0
        for entry in microbatch_entries:
            weight_sum, active_samples = self._local_effective_weight_summary_for_microbatch(entry["microbatch"])
            entry["local_effective_weight_sum"] = float(weight_sum)
            entry["local_active_samples"] = int(active_samples)
            local_total_effective_weight += float(weight_sum)
            total_active_samples += int(active_samples)

        runtime_stats["raw_local_sample_count"] = int(total_active_samples)
        global_total_effective_weight = _distributed_sum_float(float(local_total_effective_weight), device=target_device)
        if global_total_effective_weight <= 0.0:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
            return self._backward_loss_scalar(
                loss,
                optimizer=optimizer,
                num_items_in_batch=num_items_in_batch,
            )

        world_size = max(1, int(_distributed_world_size()))
        max_microbatch_count = _distributed_max_int(len(microbatch_entries), device=target_device)
        padded_microbatch_backwards = max(0, int(max_microbatch_count) - int(len(microbatch_entries)))
        reported_loss: Optional[torch.Tensor] = None
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        batch_start_time = 0.0
        for microbatch_position in range(int(max_microbatch_count)):
            entry = microbatch_entries[microbatch_position] if microbatch_position < len(microbatch_entries) else None
            if entry is not None and int(entry["microbatch_index"]) == 1:
                _training_phase_runtime_log(
                    "rl compute_loss batch start: "
                    f"batch={int(entry['batch_index'])}/{int(entry['batch_count'])} "
                    f"samples={int(self._episode_input_sample_count(entry['microbatch']))} "
                    f"microbatches={int(entry['microbatch_count'])}"
                )
                batch_start_time = time.perf_counter()
            if entry is not None and int(entry["microbatch_count"]) > 1:
                _training_phase_runtime_log(
                    "rl compute_loss microbatch start: "
                    f"batch={int(entry['batch_index'])}/{int(entry['batch_count'])} "
                    f"microbatch={int(entry['microbatch_index'])}/{int(entry['microbatch_count'])} "
                    f"samples={int(self._episode_input_sample_count(entry['microbatch']))}"
                )

            if entry is None:
                step_loss = _zero_loss_from_model(model)
            else:
                local_effective_weight_sum = float(entry.get("local_effective_weight_sum") or 0.0)
                local_active_samples = int(entry.get("local_active_samples") or 0)
                if local_effective_weight_sum <= 0.0 or local_active_samples <= 0:
                    step_loss = _zero_loss_from_model(model)
                else:
                    device_microbatch = self._move_episode_input_to_device(entry["microbatch"], device=model_device)
                    with self.compute_loss_context_manager():
                        sample_losses = self._compute_sample_losses_for_device_microbatch(
                            model,
                            device_microbatch=device_microbatch,
                            batch_index=int(entry["batch_index"]),
                            batch_count=int(entry["batch_count"]),
                            microbatch_index=int(entry["microbatch_index"]),
                            microbatch_count=int(entry["microbatch_count"]),
                        )
                    if (
                        sample_losses is None
                        or sample_losses.numel() <= 0
                        or not bool(getattr(sample_losses, "requires_grad", False))
                    ):
                        step_loss = _zero_loss_from_model(model)
                    else:
                        step_loss = sample_losses.sum() * float(world_size) / float(global_total_effective_weight)

            detached_loss = self._backward_loss_scalar(
                step_loss,
                optimizer=optimizer,
                num_items_in_batch=num_items_in_batch,
            )
            reported_loss = detached_loss if reported_loss is None else reported_loss + detached_loss

            if entry is not None and bool(entry["is_last_in_batch"]):
                _training_phase_runtime_log(
                    "rl compute_loss batch end: "
                    f"batch={int(entry['batch_index'])}/{int(entry['batch_count'])} "
                    f"elapsed_sec={time.perf_counter() - batch_start_time:.3f}"
                )

        _training_phase_runtime_log(
            "rl compute_loss end: "
            f"episode_batches={len(episode_inputs)} "
            f"trainable_samples={int(total_active_samples)} "
            f"global_effective_weight={float(global_total_effective_weight):.3f} "
            f"padded_microbatch_backwards={int(padded_microbatch_backwards)} "
            f"elapsed_sec={time.perf_counter() - compute_loss_start:.3f}"
        )
        if reported_loss is None:
            return _zero_loss_from_model(model).detach()
        return reported_loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        cp_prepare = getattr(self, "_prepare_context_parallel_inputs", None)
        if callable(cp_prepare):
            cp_context, inputs = cp_prepare(model, inputs)
        else:
            cp_context, inputs = nullcontext, inputs

        with cp_context():
            model.train()
            optimizer = getattr(self, "optimizer", None)
            if optimizer is not None and hasattr(optimizer, "train") and callable(optimizer.train):
                optimizer.train()

            prepare_inputs = getattr(self, "_prepare_inputs", None)
            if callable(prepare_inputs):
                inputs = prepare_inputs(inputs)
            skipped_loss = self._maybe_skip_empty_training_step(model, inputs)
            if skipped_loss is not None:
                return skipped_loss.detach()

            if self._should_use_immediate_microbatch_backward():
                loss = self._training_step_with_immediate_microbatch_backward(
                    model,
                    inputs,
                    optimizer=optimizer,
                    num_items_in_batch=num_items_in_batch,
                )
            else:
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
                loss = self._backward_loss_scalar(
                    loss,
                    optimizer=optimizer,
                    num_items_in_batch=num_items_in_batch,
                )
            del inputs
            return loss.detach()

    def _prepare_advantages(self, batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
        advantages = batch.get("advantages")
        if advantages is None:
            raise ValueError("Dedicated GRPO trainer requires `advantages` in every episode input.")
        return advantages.to(device=device, dtype=torch.float32).view(-1)

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
        sample_count = int(response_mask.shape[0])
        effective_weight = self._effective_sample_weight(
            batch,
            device=response_mask.device,
            sample_count=sample_count,
        )
        if not bool(torch.any(effective_weight > 0)):
            return torch.zeros(sample_count, dtype=torch.float32, device=response_mask.device)
        prepared_microbatch = self._prepare_device_microbatch_for_completion_only(batch)
        policy_forward_start = time.perf_counter()
        _training_phase_runtime_log(
            "rl compute_loss policy forward start: "
            f"samples={sample_count} completion_tokens={int(batch['completion_ids'].shape[-1])}"
        )
        _training_phase_runtime_log(
            "rl compute_loss before completion_only_token_log_probs: "
            f"samples={sample_count} "
            f"prompt_tokens={int(batch['prompt_ids'].shape[-1])} "
            f"completion_tokens={int(batch['completion_ids'].shape[-1])}"
        )
        policy_token_log_probs, response_mask = compute_completion_only_token_log_probs_from_prepared_inputs(
            model=model,
            model_inputs=prepared_microbatch["model_inputs"],
            completion_ids=prepared_microbatch["completion_ids"],
            completion_mask=prepared_microbatch["completion_mask"],
            logits_to_keep=int(prepared_microbatch["logits_to_keep"]),
            temperature=self.policy_temperature,
            log_runtime_details=False,
        )
        _training_phase_runtime_log(
            "rl compute_loss after completion_only_token_log_probs: "
            f"samples={sample_count} "
            f"elapsed_sec={time.perf_counter() - policy_forward_start:.3f}"
        )
        _training_phase_runtime_log(
            "rl compute_loss policy forward end: "
            f"samples={sample_count} elapsed_sec={time.perf_counter() - policy_forward_start:.3f}"
        )
        if not bool(policy_token_log_probs.requires_grad):
            raise RuntimeError("Policy completion log-probs are detached in the dedicated GRPO path.")
        old_policy_token_log_probs = batch.get("old_policy_token_log_probs")
        if old_policy_token_log_probs == _USE_CURRENT_POLICY_LOGPROBS_SENTINEL:
            old_policy_token_log_probs = policy_token_log_probs.detach()
        if old_policy_token_log_probs is None:
            raise ValueError("Dedicated GRPO trainer requires old_policy_token_log_probs in every episode input.")
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
            cached_reference_token_log_probs = batch.get("reference_token_log_probs")
            reference_token_log_probs = None
            if isinstance(cached_reference_token_log_probs, torch.Tensor):
                reference_token_log_probs = cached_reference_token_log_probs.to(
                    device=policy_token_log_probs.device,
                    dtype=torch.float32,
                )
                if reference_token_log_probs.ndim == 1:
                    reference_token_log_probs = reference_token_log_probs.view(1, -1)
                if tuple(reference_token_log_probs.shape) != tuple(policy_token_log_probs.shape):
                    raise ValueError(
                        "reference_token_log_probs must align with policy_token_log_probs shape: "
                        f"{tuple(reference_token_log_probs.shape)} vs {tuple(policy_token_log_probs.shape)}"
                    )
                _training_phase_runtime_log(
                    "rl compute_loss reference kl cached_hit: "
                    f"samples={sample_count} "
                    f"completion_tokens={int(batch['completion_ids'].shape[-1])} "
                    f"elapsed_sec=0.000 source=prefetch"
                )
            elif self.reference_model is not None:
                reference_forward_start = time.perf_counter()
                _training_phase_runtime_log(
                    "rl compute_loss reference kl forward start: "
                    f"samples={sample_count} completion_tokens={int(batch['completion_ids'].shape[-1])}"
                )
                with torch.inference_mode():
                    reference_token_log_probs, _ = compute_completion_only_token_log_probs_from_prepared_inputs(
                        model=self.reference_model,
                        model_inputs=prepared_microbatch["model_inputs"],
                        completion_ids=prepared_microbatch["completion_ids"],
                        completion_mask=prepared_microbatch["completion_mask"],
                        logits_to_keep=int(prepared_microbatch["logits_to_keep"]),
                        temperature=self.policy_temperature,
                        log_runtime_details=False,
                    )
                _training_phase_runtime_log(
                    "rl compute_loss reference kl forward end: "
                    f"samples={sample_count} elapsed_sec={time.perf_counter() - reference_forward_start:.3f}"
                )
            elif self.use_lora_reference_disable_adapter:
                disable_context, reference_model = self._disable_adapter_context(model)
                reference_forward_start = time.perf_counter()
                _training_phase_runtime_log(
                    "rl compute_loss reference kl forward start: "
                    f"samples={sample_count} completion_tokens={int(batch['completion_ids'].shape[-1])}"
                )
                with torch.inference_mode():
                    with disable_context:
                        reference_token_log_probs, _ = compute_completion_only_token_log_probs_from_prepared_inputs(
                            model=reference_model,
                            model_inputs=prepared_microbatch["model_inputs"],
                            completion_ids=prepared_microbatch["completion_ids"],
                            completion_mask=prepared_microbatch["completion_mask"],
                            logits_to_keep=int(prepared_microbatch["logits_to_keep"]),
                            temperature=self.policy_temperature,
                            log_runtime_details=False,
                        )
                _training_phase_runtime_log(
                    "rl compute_loss reference kl forward end: "
                    f"samples={sample_count} elapsed_sec={time.perf_counter() - reference_forward_start:.3f}"
                )
            if reference_token_log_probs is not None:
                delta = reference_token_log_probs.to(policy_token_log_probs.device) - policy_token_log_probs
                per_token_kl = torch.exp(delta) - delta - 1.0
                per_token_loss = per_token_loss + per_token_loss.new_tensor(self.kl_beta) * per_token_kl
        response_mask_f = response_mask.to(dtype=per_token_loss.dtype)
        token_counts = response_mask_f.sum(dim=-1).clamp(min=1.0)
        sample_losses = (per_token_loss * response_mask_f).sum(dim=-1) / token_counts
        sample_losses = sample_losses * effective_weight.to(device=sample_losses.device, dtype=sample_losses.dtype)
        return sample_losses

    def _compute_liger_loss_for_batch(self, unwrapped_model: Any, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        if self.liger_grpo_loss is None:
            return self._compute_sample_losses_for_batch(model=unwrapped_model, batch=batch)
        prompt_ids = batch.get("prompt_ids")
        prompt_mask = batch.get("prompt_mask")
        completion_ids = batch.get("completion_ids")
        completion_mask = batch.get("completion_mask")
        if not all(isinstance(value, torch.Tensor) for value in (prompt_ids, prompt_mask, completion_ids, completion_mask)):
            return None
        if not bool(torch.any(completion_mask)):
            return None
        sample_count = int(completion_ids.shape[0])
        effective_weight = self._effective_sample_weight(
            batch,
            device=completion_ids.device,
            sample_count=sample_count,
        )
        if not bool(torch.any(effective_weight > 0)):
            return torch.zeros(sample_count, dtype=torch.float32, device=completion_ids.device)
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask.to(dtype=prompt_mask.dtype)], dim=1)
        multimodal_inputs = self._episode_input_multimodal_inputs(batch)
        logits_to_keep = int(completion_ids.shape[1])
        multimodal_summary = {}
        for key, value in dict(multimodal_inputs or {}).items():
            if isinstance(value, torch.Tensor):
                multimodal_summary[str(key)] = {
                    "shape": tuple(int(v) for v in value.shape),
                    "dtype": str(value.dtype).replace("torch.", ""),
                    "device": str(value.device),
                }
            elif isinstance(value, dict):
                multimodal_summary[str(key)] = {
                    str(sub_key): (
                        {
                            "shape": tuple(int(v) for v in sub_value.shape),
                            "dtype": str(sub_value.dtype).replace("torch.", ""),
                            "device": str(sub_value.device),
                        }
                        if isinstance(sub_value, torch.Tensor)
                        else type(sub_value).__name__
                    )
                    for sub_key, sub_value in value.items()
                }
            else:
                multimodal_summary[str(key)] = type(value).__name__
        liger_forward_start = time.perf_counter()
        _training_phase_runtime_log(
            "rl compute_loss liger forward start: "
            f"samples={sample_count} "
            f"prompt_tokens={int(prompt_ids.shape[-1])} "
            f"completion_tokens={int(completion_ids.shape[-1])} "
            f"total_tokens={int(input_ids.shape[-1])} "
            f"multimodal_keys={sorted(multimodal_summary.keys())} "
            f"multimodal_summary={multimodal_summary}"
        )
        last_hidden_state = self._get_liger_last_hidden_state(
            unwrapped_model=unwrapped_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            multimodal_inputs=multimodal_inputs,
            logits_to_keep=logits_to_keep,
        )
        _training_phase_runtime_log(
            "rl compute_loss liger forward end: "
            f"samples={sample_count} elapsed_sec={time.perf_counter() - liger_forward_start:.3f}"
        )
        self._log_liger_runtime_configuration_once()
        ref_per_token_logps = None
        if self.kl_beta > 0.0 and self.reference_model is not None:
            reference_forward_start = time.perf_counter()
            _training_phase_runtime_log(
                "rl compute_loss reference kl forward start: "
                f"samples={sample_count} completion_tokens={int(completion_ids.shape[-1])}"
            )
            with torch.inference_mode():
                ref_per_token_logps, _ = compute_completion_only_token_log_probs_from_ids(
                    model=self.reference_model,
                    prompt_ids=batch["prompt_ids"],
                    prompt_mask=batch["prompt_mask"],
                    completion_ids=batch["completion_ids"],
                    completion_mask=batch["completion_mask"],
                    multimodal_inputs=multimodal_inputs,
                    temperature=self.policy_temperature,
                )
            _training_phase_runtime_log(
                "rl compute_loss reference kl forward end: "
                f"samples={sample_count} elapsed_sec={time.perf_counter() - reference_forward_start:.3f}"
            )
        old_policy_token_log_probs = batch.get("old_policy_token_log_probs")
        if old_policy_token_log_probs == _USE_CURRENT_POLICY_LOGPROBS_SENTINEL:
            old_policy_token_log_probs = None
        completion_ids = self._materialize_liger_tensor(
            completion_ids,
            device=last_hidden_state.device,
            dtype=completion_ids.dtype if isinstance(completion_ids, torch.Tensor) else None,
        )
        completion_mask = self._materialize_liger_tensor(
            completion_mask,
            device=last_hidden_state.device,
            dtype=completion_mask.dtype if isinstance(completion_mask, torch.Tensor) else None,
        )
        last_hidden_state = self._materialize_liger_tensor(
            last_hidden_state,
            device=last_hidden_state.device,
            dtype=last_hidden_state.dtype,
        )
        lm_head, lm_head_weight, lm_head_bias, linear_head_path = _resolve_liger_linear_head(unwrapped_model)
        self._liger_linear_head_path = str(linear_head_path)
        ref_per_token_logps = self._materialize_liger_tensor(
            ref_per_token_logps,
            device=last_hidden_state.device,
            dtype=torch.float32,
        )
        old_policy_token_log_probs = self._materialize_liger_tensor(
            old_policy_token_log_probs,
            device=last_hidden_state.device,
            dtype=torch.float32,
        )
        liger_advantages = self._materialize_liger_tensor(
            self._prepare_advantages(batch, last_hidden_state.device),
            device=last_hidden_state.device,
            dtype=torch.float32,
        )
        with _gather_liger_linear_head_parameters(lm_head):
            lm_head_weight = getattr(lm_head, "weight", lm_head_weight)
            lm_head_bias = getattr(lm_head, "bias", lm_head_bias)
            try:
                weight_shape = tuple(int(value) for value in tuple(getattr(lm_head_weight, "shape", ()) or ()))
            except Exception:
                weight_shape = ()
            if len(weight_shape) != 2:
                _training_phase_runtime_log(
                    "rl liger linear head gathered but still invalid: "
                    f"path={linear_head_path} "
                    f"type={type(lm_head_weight).__name__} "
                    f"shape={weight_shape}"
                )
                raise RuntimeError("Liger GRPO loss requires a gathered 2D linear output head weight tensor.")
            loss, _ = self.liger_grpo_loss(
                _input=last_hidden_state,
                lin_weight=lm_head_weight,
                selected_token_ids=completion_ids,
                attention_mask=completion_mask,
                advantages=liger_advantages,
                bias=lm_head_bias,
                old_per_token_logps=old_policy_token_log_probs,
                ref_per_token_logps=ref_per_token_logps,
            )
        return loss.expand(sample_count) * effective_weight.to(device=last_hidden_state.device, dtype=torch.float32)

    def _prepare_inputs(self, inputs):
        if not isinstance(inputs, list):
            return super()._prepare_inputs(inputs)
        wrapped_model = self.model
        rollout_model = _unwrap_model(wrapped_model)
        progress = getattr(self, "_native_grpo_progress", None)
        was_training = bool(getattr(wrapped_model, "training", getattr(rollout_model, "training", False)))
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
                    generation_step_payloads = self._build_generation_step_payloads(inputs, rollout_model)
                finally:
                    if was_training and hasattr(wrapped_model, "train"):
                        wrapped_model.train()
                prepared = (
                    generation_step_payloads[0]
                    if generation_step_payloads
                    else self._empty_generation_step_payload(
                        video_ids=[str(item.get("video_id") or "") for item in inputs]
                    )
                )
            prepared = self._maybe_replay_zero_advantage_payload(prepared)
            for video_id in [str(video_id or "") for video_id in (prepared.get("video_ids") or [])]:
                if progress is not None:
                    progress.finish_item(video_id=video_id)
            runtime_stats = dict(prepared.get("runtime_stats") or {})
            episode_inputs = list(prepared.get("episode_inputs") or [])
            active_cpu_episode_inputs: List[Dict[str, Any]] = []
            inactive_cpu_episode_inputs: List[Dict[str, Any]] = []
            for episode_input in episode_inputs:
                if self._has_trainable_weight(episode_input):
                    active_cpu_episode_inputs.append(episode_input)
                else:
                    inactive_cpu_episode_inputs.append(episode_input)
            try:
                target_device = next(rollout_model.parameters()).device
            except StopIteration:
                target_device = torch.device("cpu")
            runtime_log(
                "rl prepare_inputs stage: before align_episode_inputs_across_ranks "
                f"episode_inputs={len(episode_inputs)} "
                f"active_cpu_episode_inputs={len(active_cpu_episode_inputs)} "
                f"inactive_cpu_episode_inputs={len(inactive_cpu_episode_inputs)}",
                runtime=distributed_runtime_from_env(),
                main_process_only=True,
            )
            episode_inputs = self._align_episode_inputs_across_ranks(
                active_cpu_episode_inputs,
                device=target_device,
                runtime_stats=runtime_stats,
            )
            runtime_log(
                "rl prepare_inputs stage: after align_episode_inputs_across_ranks "
                f"aligned_episode_inputs={len(episode_inputs)}",
                runtime=distributed_runtime_from_env(),
                main_process_only=True,
            )
            inactive_episode_inputs = list(inactive_cpu_episode_inputs)
            active_episode_inputs = []
            for episode_input in episode_inputs:
                if self._has_trainable_weight(episode_input):
                    active_episode_inputs.append(episode_input)
                else:
                    inactive_episode_inputs.append(episode_input)
            materialize_start = time.perf_counter()
            if active_episode_inputs:
                runtime_log(
                    "rl prepare_inputs stage: before materialize_episode_inputs "
                    f"active_episode_inputs={len(active_episode_inputs)}",
                    runtime=distributed_runtime_from_env(),
                    main_process_only=True,
                )
                active_episode_inputs = self._materialize_episode_inputs(active_episode_inputs, device=target_device)
                runtime_log(
                    "rl prepare_inputs stage: after materialize_episode_inputs "
                    f"active_episode_inputs={len(active_episode_inputs)}",
                    runtime=distributed_runtime_from_env(),
                    main_process_only=True,
                )
                if bool(getattr(self, "rl_enable_reference_prefetch_cache", True)):
                    prefetch_start = time.perf_counter()
                    _training_phase_runtime_log(
                        "rl prepare_inputs stage: before prefetch_reference_log_probs "
                        f"episode_inputs={len(active_episode_inputs)}"
                    )
                    active_episode_inputs = self._prefetch_reference_log_probs(
                        wrapped_model,
                        active_episode_inputs,
                    )
                    _training_phase_runtime_log(
                        "rl prepare_inputs stage: after prefetch_reference_log_probs "
                        f"episode_inputs={len(active_episode_inputs)} "
                        f"elapsed_sec={time.perf_counter() - prefetch_start:.3f}"
                    )
            episode_inputs = active_episode_inputs + inactive_episode_inputs
            runtime_stats["raw_local_episode_input_count"] = int(len(episode_inputs))
            runtime_log(
                "rl materialize end: "
                f"episode_inputs={len(episode_inputs)} "
                f"active_episode_inputs={len(active_episode_inputs)} "
                f"inactive_episode_inputs={len(inactive_episode_inputs)} "
                f"elapsed_sec={time.perf_counter() - materialize_start:.3f}",
                runtime=distributed_runtime_from_env(),
                main_process_only=True,
            )
            return {
                "episode_inputs": episode_inputs,
                "rollout_metrics": dict(prepared.get("rollout_metrics") or {}),
                "budgeting_metrics": dict(prepared.get("budgeting_metrics") or {}),
                "runtime_stats": runtime_stats,
            }
        finally:
            if progress is not None:
                progress.close_batch()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        if return_outputs:
            raise ValueError("Dedicated GRPO trainer does not support returning model outputs.")
        episode_inputs = list(inputs.get("episode_inputs") or [])
        runtime_stats = dict(inputs.get("runtime_stats") or {})
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")
        if not episode_inputs:
            self._maybe_log_empty_batch_rank_summary(
                reason="all_empty_episode_inputs",
                runtime_stats=runtime_stats,
                trainable_samples=0,
            )
            return _zero_loss_from_model(model)

        self._ensure_liger_runtime_ready(model)
        compute_loss_start = time.perf_counter()
        _training_phase_runtime_log(
            "rl compute_loss start: "
            f"episode_batches={len(episode_inputs)} "
            f"use_liger_loss_requested={bool(getattr(self, 'use_liger_loss_requested', False))} "
            f"use_liger_loss_effective={bool(getattr(self, 'use_liger_loss_effective', False))} "
            f"liger_disable_reason={str(getattr(self, '_liger_runtime_disable_reason', None) or 'none')} "
            f"loss_microbatch={int(getattr(self, 'compute_loss_microbatch_size', 1) or 1)}"
        )
        total_loss_sum = None
        total_active_samples = 0
        total_effective_weight = 0.0
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cpu")
        for batch_index, batch in enumerate(episode_inputs, start=1):
            microbatches = list(self._iter_loss_microbatches(batch))
            _training_phase_runtime_log(
                "rl compute_loss batch start: "
                f"batch={batch_index}/{len(episode_inputs)} "
                f"samples={int(self._episode_input_sample_count(batch))} "
                f"microbatches={len(microbatches)}"
            )
            batch_start = time.perf_counter()
            for microbatch_index, microbatch in enumerate(microbatches, start=1):
                if len(microbatches) > 1:
                    _training_phase_runtime_log(
                        "rl compute_loss microbatch start: "
                        f"batch={batch_index}/{len(episode_inputs)} "
                        f"microbatch={microbatch_index}/{len(microbatches)} "
                        f"samples={int(self._episode_input_sample_count(microbatch))}"
                    )
                device_microbatch = self._move_episode_input_to_device(microbatch, device=model_device)
                liger_unwrapped_model = self._resolve_liger_unwrapped_model(model) if self.use_liger_loss else None
                if self.use_liger_loss:
                    use_forward_redirection = self._should_use_forward_redirection_for_liger()
                    dispatch_name = (
                        "forward_redirection_zero3"
                        if use_forward_redirection
                        else "direct_unwrapped_model_with_head_gather"
                    )
                    _training_phase_runtime_log(
                        "rl compute_loss liger path dispatch: "
                        f"batch={batch_index}/{len(episode_inputs)} "
                        f"microbatch={microbatch_index}/{len(microbatches)} "
                        f"dispatch={dispatch_name}"
                    )
                    if use_forward_redirection:
                        _training_phase_runtime_log(
                            "rl compute_loss liger forward_redirection enter: "
                            f"batch={batch_index}/{len(episode_inputs)} "
                            f"microbatch={microbatch_index}/{len(microbatches)}"
                        )
                        sample_losses = self._forward_redirection(
                            model,
                            liger_unwrapped_model,
                            self._compute_liger_loss_for_batch,
                            liger_unwrapped_model,
                            device_microbatch,
                        )
                        _training_phase_runtime_log(
                            "rl compute_loss liger forward_redirection exit: "
                            f"batch={batch_index}/{len(episode_inputs)} "
                            f"microbatch={microbatch_index}/{len(microbatches)}"
                        )
                    else:
                        sample_losses = self._compute_liger_loss_for_batch(
                            unwrapped_model=liger_unwrapped_model,
                            batch=device_microbatch,
                        )
                else:
                    sample_losses = self._compute_sample_losses_for_batch(model=model, batch=device_microbatch)
                if sample_losses is None or sample_losses.numel() <= 0:
                    continue
                effective_weight = self._effective_sample_weight(
                    device_microbatch,
                    device=sample_losses.device,
                    sample_count=int(sample_losses.numel()),
                )
                total_active_samples += int((effective_weight > 0).sum().item())
                total_effective_weight += float(effective_weight.sum().item())
                batch_loss_sum = sample_losses.sum()
                total_loss_sum = batch_loss_sum if total_loss_sum is None else total_loss_sum + batch_loss_sum
            _training_phase_runtime_log(
                "rl compute_loss batch end: "
                f"batch={batch_index}/{len(episode_inputs)} "
                f"elapsed_sec={time.perf_counter() - batch_start:.3f}"
            )
        runtime_stats["raw_local_sample_count"] = int(total_active_samples)
        global_total_effective_weight = _distributed_sum_float(float(total_effective_weight), device=target_device)
        if total_loss_sum is None or global_total_effective_weight <= 0.0:
            self._maybe_log_empty_batch_rank_summary(
                reason="all_empty_trainable_samples",
                runtime_stats=runtime_stats,
                trainable_samples=int(total_active_samples),
            )
            return _zero_loss_from_model(model)
        world_size = max(1, int(_distributed_world_size()))
        _training_phase_runtime_log(
            "rl compute_loss end: "
            f"episode_batches={len(episode_inputs)} "
            f"trainable_samples={int(total_active_samples)} "
            f"global_effective_weight={float(global_total_effective_weight):.3f} "
            f"elapsed_sec={time.perf_counter() - compute_loss_start:.3f}"
        )
        return total_loss_sum * float(world_size) / float(global_total_effective_weight)

    def get_budget_drop_metrics(self) -> Dict[str, Any]:
        metrics = self._budgeting_stats.as_dict()
        metrics.update(
            {
                "rl_zero_response_dropped": int(self._zero_response_dropped),
                "rl_materialize_fallback_batches": int(self._materialize_fallback_batches),
                "rl_groups_all_zero_advantage": int(self._groups_all_zero_advantage),
                "rl_groups_filtered_by_min_weight": int(self._groups_filtered_by_min_weight),
                "rl_fecv_failure_count": int(self._fecv_failure_count),
                "rl_fecv_degraded_rollout_count": int(self._fecv_degraded_rollout_count),
                "rl_ddp_noop_padded_episode_inputs": int(self._ddp_noop_padded_episode_inputs),
                "rl_skipped_empty_training_steps": int(self._skip_empty_training_steps),
                "rl_zero_advantage_replay_uses": int(self._zero_advantage_replay_uses),
                "rl_zero_advantage_replay_misses": int(self._zero_advantage_replay_misses),
                "rl_zero_advantage_replay_cache_size": int(len(self._recent_nonzero_advantage_payloads)),
                "rl_compute_loss_microbatch_size_effective": int(self.compute_loss_microbatch_size),
            }
        )
        return metrics

    def get_budgeting_stats(self) -> BudgetingStats:
        stats = BudgetingStats()
        stats.merge(self._budgeting_stats)
        return stats


def create_timesearch_aligned_grpo_trainer(
    *,
    model: Any,
    processor: Any,
    train_items: Sequence[Dict[str, Any]],
    train_dataset: Any = None,
    output_dir: str | Path,
    trainer_init_model_path: str | Path,
    torch_dtype: str,
    attn_implementation: Optional[str],
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
    counterfactual_max_images: int = 12,
    policy_do_sample: bool = False,
    policy_temperature: Optional[float] = None,
    policy_top_p: Optional[float] = None,
    policy_top_k: Optional[int] = None,
    policy_repetition_penalty: Optional[float] = None,
    rollout_use_generation_cache: bool = True,
    fecv_use_generation_cache: bool = True,
    compute_loss_microbatch_size: int = 2,
    use_liger_loss: bool = False,
    iteration_index: int = 0,
    num_iterations: int = 1,
    rollout_eval_callback: Any = None,
    fecv_failure_policy: str = "degrade",
    log_empty_batch_rank_summary: bool = True,
    reward_version: str = DEFAULT_RL_REWARD_VERSION,
    reward_config: Optional[Dict[str, Any]] = None,
    steps_per_generation: int = 1,
    rollout_stage_batch_size: int = 16,
    fecv_stage_batch_size: int = 16,
    proposal_runtime: Any = None,
    strict_feature_guided_proposal: bool = False,
    policy_builder: Any = None,
    deepspeed: Optional[str] = None,
    save_strategy: str = "no",
) -> Any:
    try:
        from transformers import Trainer, TrainingArguments
    except Exception as exc:
        raise ImportError("Dedicated GRPO trainer requires the `transformers` package.") from exc

    if not callable(policy_builder):
        raise ValueError("policy_builder is required for the dedicated GRPO trainer.")

    dynamic_iteration_dataset = train_dataset is not None
    effective_persistent_workers = bool(dataloader_persistent_workers) and int(dataloader_num_workers) > 0
    effective_dataloader_num_workers = max(0, int(dataloader_num_workers))
    if dynamic_iteration_dataset and effective_dataloader_num_workers > 0:
        runtime_log(
            "continuous RL forcing dataloader_num_workers=0 because train dataset is replaced each epoch.",
            runtime=distributed_runtime_from_env(),
            main_process_only=True,
        )
        effective_dataloader_num_workers = 0
        effective_persistent_workers = False
    if effective_persistent_workers and float(num_train_epochs) <= 1.0:
        runtime_log(
            (
                "dedicated RL disabled dataloader_persistent_workers because "
                "num_train_epochs<=1 makes worker persistence across epochs ineffective and unstable "
                "for iteration-scoped Trainer recreation."
            ),
            runtime=distributed_runtime_from_env(),
            main_process_only=True,
        )
        effective_persistent_workers = False

    os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(int(gradient_accumulation_steps))
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
        "save_strategy": str(save_strategy),
        "dataloader_num_workers": int(effective_dataloader_num_workers),
        "dataloader_persistent_workers": bool(effective_persistent_workers),
        "ddp_find_unused_parameters": False,
    }
    if int(dataloader_num_workers) > 0 and int(dataloader_prefetch_factor) > 0:
        training_args_kwargs["dataloader_prefetch_factor"] = int(dataloader_prefetch_factor)
    if str(deepspeed or "").strip():
        training_args_kwargs["deepspeed"] = str(deepspeed)
    training_args = TrainingArguments(**training_args_kwargs)

    aligned_grpo_config = {
        "processor": processor,
        "train_dataset": train_dataset if train_dataset is not None else _RawItemDataset(train_items),
        "reference_model": None,
        "use_lora_reference_disable_adapter": False,
        "reference_model_mode": "per_iteration_trainer_init",
        "reference_model_source_path": str(trainer_init_model_path),
        "reference_model_backend": "none",
        "kl_beta": float(kl_beta),
        "ppo_clip_epsilon": float(ppo_clip_epsilon),
        "rollout_runner": rollout_runner,
        "proposal_runtime": proposal_runtime,
        "strict_feature_guided_proposal": bool(strict_feature_guided_proposal),
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
        "use_liger_loss": bool(use_liger_loss),
        "steps_per_generation": int(steps_per_generation),
        "rollout_stage_batch_size": int(rollout_stage_batch_size),
        "fecv_stage_batch_size": int(fecv_stage_batch_size),
        "per_device_train_batch_size": int(per_device_train_batch_size),
        "fecv_failure_policy": str(fecv_failure_policy),
        "log_empty_batch_rank_summary": bool(log_empty_batch_rank_summary),
        "reward_version": str(reward_version),
        "reward_config": dict(reward_config or {}),
        "iteration_index": int(iteration_index),
        "num_iterations": int(num_iterations),
        "policy_builder": policy_builder,
    }
    dedicated_trainer_class = type(
        "TimesearchAlignedGRPOTrainer",
        (TimesearchAlignedGRPOTrainerMixin, Trainer),
        {},
    )
    trainer = dedicated_trainer_class(
        model=model,
        args=training_args,
        train_dataset=aligned_grpo_config["train_dataset"],
        data_collator=_raw_item_collator,
        callbacks=[],
        aligned_grpo_config=aligned_grpo_config,
    )
    reference_model, reference_backend = _build_managed_reference_model_like_timesearch_r(
        trainer=trainer,
        model=model,
        trainer_init_model_path=str(trainer_init_model_path),
        torch_dtype=str(torch_dtype or "auto"),
        attn_implementation=attn_implementation,
        kl_beta=float(kl_beta),
        deepspeed=deepspeed,
    )
    trainer.reference_model = reference_model
    trainer.reference_model_mode = "per_iteration_trainer_init"
    trainer.reference_model_source_path = str(trainer_init_model_path)
    trainer.reference_model_backend = str(reference_backend)
    if trainer.reference_model is not None:
        trainer.reference_model.eval()
        for parameter in trainer.reference_model.parameters():
            parameter.requires_grad_(False)
        runtime_log(
            "rl reference model managed: "
            f"mode={trainer.reference_model_mode} "
            f"backend={trainer.reference_model_backend} "
            f"source_path={trainer.reference_model_source_path}",
            runtime=distributed_runtime_from_env(),
            main_process_only=True,
        )
    if rollout_eval_callback is not None:
        trainer.add_callback(rollout_eval_callback)
    trainer.add_callback(_build_native_grpo_progress_callback(trainer=trainer))
    return trainer
