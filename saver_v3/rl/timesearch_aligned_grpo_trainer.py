from __future__ import annotations

import copy
import gc
import math
from contextlib import nullcontext
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
    _compute_group_relative_advantages,
    _degrade_reward_summary_for_fecv_failure,
    _distributed_bool_consensus,
    _distributed_first_available_object,
    _distributed_sum_int,
    _distributed_world_size,
    _safe_float,
    _truncate_error_message,
    get_replay_buffer,
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
    compute_completion_only_token_log_probs_from_ids,
)


def _raw_item_collator(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return features


def _extract_turn_features_from_rollout(
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
        features.append(
            {
                "video_id": rollout.get("video_id"),
                "group_id": rollout.get("group_id"),
                "generation_id": rollout.get("generation_id"),
                "step_index": int(turn.get("step_index") or 0),
                "_rl_prompt_completion_native": True,
                "prompt_messages": copy.deepcopy(prompt_messages),
                "completion_text": target_response,
                "target_response": target_response,
                "sample_weight": float(rollout_advantage),
                "advantage": float(rollout_advantage),
                "target_action": turn.get("action"),
                "tool_name": turn.get("tool_name"),
            }
        )
    return features


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
        self.replay_buffer = get_replay_buffer(
            self.replay_buffer_type if self.replay_buffer_enable else "none",
            capacity=self.replay_buffer_capacity,
            alpha=self.replay_buffer_alpha,
        )
        self._budgeting_stats = BudgetingStats()
        self._zero_response_dropped = 0
        self._materialize_fallback_batches = 0
        self._replay_fill_batches = 0
        self._replay_fill_prepared_batches = 0
        self._groups_all_zero_advantage = 0
        self._groups_filtered_by_min_weight = 0
        self._fecv_failure_count = 0
        self._fecv_degraded_rollout_count = 0
        self._ddp_noop_padded_prepared_batches = 0
        self._reference_model_device = None
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

    def _new_rollout_metric_lists(self) -> Dict[str, List[float]]:
        return {
            "reward_total": [],
            "reward_accuracy": [],
            "reward_fecv_evidence": [],
            "reward_protocol_finalize": [],
        }

    def _new_runtime_stats(self) -> Dict[str, int]:
        return {
            "raw_local_prepared_batch_count": 0,
            "raw_local_sample_count": 0,
            "local_fecv_failure_count": 0,
            "groups_filtered_by_min_weight": 0,
            "groups_all_zero_advantage": 0,
            "replay_fill_batches": 0,
            "replay_fill_prepared_batches": 0,
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
            "prepared_batches": [],
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
        rollout["group_id"] = str(item.get("video_id") or f"group_{int(generation_id)}")
        rollout["generation_id"] = int(generation_id)
        if isinstance(item.get("structured_target"), dict):
            rollout["scoring_target"] = copy.deepcopy(item["structured_target"])
        if isinstance(item.get("qa_pairs"), list):
            rollout["scoring_qa_pairs"] = copy.deepcopy(item.get("qa_pairs") or [])
        evidence = item.get("evidence") or {}
        if isinstance(evidence, dict) and isinstance(evidence.get("evidence_moments"), list):
            rollout["scoring_evidence_moments"] = copy.deepcopy(evidence.get("evidence_moments") or [])
        return rollout

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
        rollout_policy = self._build_rollout_policy(model)
        verification_policy = self._build_fecv_policy(model)
        rollout_items: List[Dict[str, Any]] = []
        rollout_meta: List[Tuple[int, int]] = []
        for item_index, item in enumerate(item_list):
            for generation_id in range(self.num_generations):
                rollout_items.append(copy.deepcopy(item))
                rollout_meta.append((int(item_index), int(generation_id)))

        generated_rollouts = list(self.rollout_runner.run_episodes(rollout_items, rollout_policy, capture_prompt_messages=True))
        rollouts_by_item: List[List[Dict[str, Any]]] = [[] for _ in item_list]
        fecv_batch_inputs: List[Dict[str, Any]] = []
        fecv_rollout_refs: List[Tuple[int, int, Dict[str, Any], Dict[str, Any]]] = []
        for (item_index, generation_id), rollout in zip(rollout_meta, generated_rollouts):
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

        try:
            fecv_results = run_counterfactual_verification_batch(
                verification_policy,
                batch_inputs=fecv_batch_inputs,
                max_images=self.counterfactual_max_images,
                branch_profile="online_core",
            )
        except Exception as exc:
            if self.fecv_failure_policy == "fail":
                raise
            fecv_results = [exc for _ in fecv_batch_inputs]

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
        for item_rollouts in rollouts_by_item:
            scored_rollouts = self._assign_reward_summaries(item_rollouts)
            grouped_scored_rollouts.append(
                _compute_group_relative_advantages(
                    scored_rollouts,
                    clip_value=self.advantage_clip,
                )
            )
        return grouped_scored_rollouts

    def _build_generation_item_payload_from_rollouts(
        self,
        item: Dict[str, Any],
        scored_rollouts: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        video_id = str(item.get("video_id") or "")
        rollout_metrics = self._new_rollout_metric_lists()
        runtime_stats = self._new_runtime_stats()
        prepared_entries: List[Dict[str, Any]] = []
        for rollout in list(scored_rollouts or []):
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
            rollout_advantage = abs(float(rollout.get("group_advantage", 0.0) or 0.0))
            if rollout_advantage < float(self.min_weight):
                runtime_stats["groups_filtered_by_min_weight"] += 1
                if rollout_advantage <= 0.0:
                    runtime_stats["groups_all_zero_advantage"] += 1
            features = _extract_turn_features_from_rollout(
                rollout,
                min_abs_advantage=self.min_weight,
            )
            for feature in features:
                result = self._build_prepared_batch_from_feature(feature)
                self._budgeting_stats.record(result)
                if result.batch is None:
                    self._zero_response_dropped += 1
                    continue
                prepared_entries.append(
                    {
                        "feature": copy.deepcopy(feature),
                        "prepared_batch": result.batch,
                    }
                )
        if prepared_entries:
            self._populate_old_policy_token_log_probs(_unwrap_model(self.model), prepared_entries)
        prepared_batches: List[Dict[str, Any]] = []
        for entry in prepared_entries:
            prepared_batch = copy.deepcopy(dict(entry["prepared_batch"] or {}))
            old_policy_token_log_probs = entry.get("old_policy_token_log_probs")
            if isinstance(old_policy_token_log_probs, torch.Tensor):
                prepared_batch["old_policy_token_log_probs"] = old_policy_token_log_probs.clone()
            prepared_batches.append(prepared_batch)
        runtime_stats["raw_local_prepared_batch_count"] = int(len(prepared_batches))
        return {
            "video_id": video_id,
            "prepared_batches": prepared_batches,
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

    def _build_prepared_batch_from_feature(
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

    def _move_prepared_batch_to_device(
        self,
        prepared_batch: Dict[str, Any],
        *,
        device: torch.device,
    ) -> Dict[str, Any]:
        prepared: Dict[str, Any] = {}
        for key, value in prepared_batch.items():
            if isinstance(value, torch.Tensor):
                prepared[key] = value.to(device=device)
            else:
                prepared[key] = copy.deepcopy(value)
        return prepared

    def _prepared_batch_sample_count(self, prepared_batch: Dict[str, Any]) -> int:
        for key in ("completion_ids", "prompt_ids", "completion_mask", "sample_weight", "advantage"):
            value = prepared_batch.get(key)
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return max(1, int(value.shape[0]))
        return 1

    def _prepared_batch_cpu_copy(self, prepared_batch: Dict[str, Any]) -> Dict[str, Any]:
        return self._move_prepared_batch_to_device(prepared_batch, device=torch.device("cpu"))

    def _clone_prepared_batch_as_noop(self, prepared_batch: Dict[str, Any]) -> Dict[str, Any]:
        cloned = copy.deepcopy(dict(prepared_batch or {}))
        sample_count = self._prepared_batch_sample_count(cloned)
        first_tensor_device = next(
            (value.device for value in cloned.values() if isinstance(value, torch.Tensor)),
            torch.device("cpu"),
        )
        cloned["sample_loss_multiplier"] = torch.zeros(sample_count, dtype=torch.float32, device=first_tensor_device)
        if isinstance(cloned.get("sample_weight"), torch.Tensor):
            cloned["sample_weight"] = torch.zeros_like(cloned["sample_weight"], dtype=torch.float32)
        if isinstance(cloned.get("advantage"), torch.Tensor):
            cloned["advantage"] = torch.zeros_like(cloned["advantage"], dtype=torch.float32)
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
                "sample_loss_multiplier must align with the prepared batch sample count: "
                f"{tuple(multiplier.shape)} vs {int(sample_count)}"
            )
        return multiplier

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
        expected_width = int(prompt_ids.shape[-1]) + int(completion_ids.shape[-1])
        return int(tensor.shape[-1]) == expected_width

    def _pad_full_sequence_aligned_and_concat(
        self,
        tensors: Sequence[torch.Tensor],
        prepared_batches: Sequence[Dict[str, Any]],
    ) -> torch.Tensor:
        prompt_lengths = [int(batch["prompt_ids"].shape[-1]) for batch in prepared_batches]
        completion_lengths = [int(batch["completion_ids"].shape[-1]) for batch in prepared_batches]
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

    def _is_merge_fallback_error(self, exc: Exception) -> bool:
        error_message = str(exc)
        return (
            "Cannot merge prepared batches with inconsistent key presence" in error_message
            or "Unable to concatenate prepared batch key" in error_message
        )

    def _merge_prepared_batches(self, prepared_batches: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
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

    def _prepared_batch_multimodal_inputs(self, prepared_batch: Dict[str, Any]) -> Dict[str, Any]:
        reserved = {
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
        return {
            key: value
            for key, value in prepared_batch.items()
            if key not in reserved
        }

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
                multimodal_inputs=self._prepared_batch_multimodal_inputs(prepared_batch),
                temperature=self.policy_temperature,
            )
        return old_policy_token_log_probs.detach().cpu()

    def _populate_old_policy_token_log_probs(
        self,
        model: Any,
        prepared_entries: Sequence[Dict[str, Any]],
    ) -> None:
        if not prepared_entries:
            return
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")
        grouped_entries = self._group_items_by_signature(
            list(prepared_entries),
            signature_fn=lambda entry: self._prepared_batch_merge_signature(entry["prepared_batch"]),
        )
        for bucket in grouped_entries:
            prepared_batches = [
                self._move_prepared_batch_to_device(entry["prepared_batch"], device=target_device)
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
                completion_length = int(entry["prepared_batch"]["completion_ids"].shape[-1])
                entry["old_policy_token_log_probs"] = (
                    batched_log_probs[row_index : row_index + 1, :completion_length].clone()
                )

    def _add_prepared_batches_to_replay_buffer(self, prepared_batches: Sequence[Dict[str, Any]]) -> None:
        if self.replay_buffer is None or not prepared_batches:
            return
        cpu_batches = [self._prepared_batch_cpu_copy(prepared_batch) for prepared_batch in prepared_batches]
        self.replay_buffer.add({"prepared_batches": cpu_batches})

    def _sample_prepared_batches_from_replay_buffer(self) -> List[Dict[str, Any]]:
        if self.replay_buffer is None or len(self.replay_buffer) <= 0:
            return []
        sampled = self.replay_buffer.sample()
        return [dict(batch) for batch in list(sampled.get("prepared_batches") or [])]

    def _aggregate_generation_step_payload(
        self,
        item_payloads: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        prepared_batches: List[Dict[str, Any]] = []
        rollout_metric_values = self._new_rollout_metric_lists()
        runtime_stats = self._new_runtime_stats()
        video_ids: List[str] = []
        for payload in item_payloads:
            video_ids.append(str(payload.get("video_id") or ""))
            prepared_batches.extend(list(payload.get("prepared_batches") or []))
            for key, values in dict(payload.get("rollout_metric_values") or {}).items():
                rollout_metric_values.setdefault(key, [])
                rollout_metric_values[key].extend([_safe_float(value) for value in (values or [])])
            for key, value in dict(payload.get("runtime_stats") or {}).items():
                runtime_stats[key] = int(runtime_stats.get(key, 0)) + int(value or 0)
        if prepared_batches:
            self._add_prepared_batches_to_replay_buffer(prepared_batches)
        else:
            replay_prepared_batches = self._sample_prepared_batches_from_replay_buffer()
            if replay_prepared_batches:
                prepared_batches = replay_prepared_batches
                runtime_stats["replay_fill_batches"] = int(runtime_stats.get("replay_fill_batches", 0)) + 1
                runtime_stats["replay_fill_prepared_batches"] = int(len(replay_prepared_batches))
                self._replay_fill_batches += 1
                self._replay_fill_prepared_batches += int(len(replay_prepared_batches))
        aggregated_metrics = {
            key: (sum(values) / float(len(values)) if values else 0.0)
            for key, values in rollout_metric_values.items()
        }
        runtime_stats["raw_local_prepared_batch_count"] = int(len(prepared_batches))
        return {
            "prepared_batches": prepared_batches,
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

    def _materialize_prepared_batches(
        self,
        prepared_batches: Sequence[Dict[str, Any]],
        *,
        device: torch.device,
    ) -> List[Dict[str, Any]]:
        materialized = [
            self._move_prepared_batch_to_device(prepared_batch, device=device)
            for prepared_batch in prepared_batches
        ]
        grouped_batches = self._group_items_by_signature(
            materialized,
            signature_fn=self._prepared_batch_merge_signature,
        )
        merged_batches: List[Dict[str, Any]] = []
        for bucket in grouped_batches:
            for start_index in range(0, len(bucket), max(1, int(self.compute_loss_microbatch_size))):
                bucket_chunk = bucket[start_index : start_index + max(1, int(self.compute_loss_microbatch_size))]
                if len(bucket_chunk) == 1:
                    merged_batches.append(bucket_chunk[0])
                    continue
                try:
                    merged_batches.append(self._merge_prepared_batches(bucket_chunk))
                except ValueError as exc:
                    if not self._is_merge_fallback_error(exc):
                        raise
                    self._materialize_fallback_batches += 1
                    merged_batches.extend(bucket_chunk)
        return merged_batches

    def _slice_prepared_batch_sample_range(
        self,
        prepared_batch: Dict[str, Any],
        *,
        start_index: int,
        end_index: int,
    ) -> Dict[str, Any]:
        sample_count = self._prepared_batch_sample_count(prepared_batch)
        sliced: Dict[str, Any] = {}
        for key, value in prepared_batch.items():
            if isinstance(value, torch.Tensor):
                if value.ndim == 0 or int(value.shape[0]) != int(sample_count):
                    sliced[key] = value
                else:
                    sliced[key] = value[int(start_index) : int(end_index)]
            else:
                sliced[key] = copy.deepcopy(value)
        return sliced

    def _iter_loss_microbatches(
        self,
        prepared_batch: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        sample_count = self._prepared_batch_sample_count(prepared_batch)
        microbatch_size = max(1, int(self.compute_loss_microbatch_size))
        if sample_count <= microbatch_size:
            return [prepared_batch]
        if not all(
            (not isinstance(value, torch.Tensor))
            or value.ndim == 0
            or int(value.shape[0]) == int(sample_count)
            for value in prepared_batch.values()
        ):
            return [prepared_batch]
        return [
            self._slice_prepared_batch_sample_range(
                prepared_batch,
                start_index=start_index,
                end_index=min(sample_count, start_index + microbatch_size),
            )
            for start_index in range(0, sample_count, microbatch_size)
        ]

    def _align_prepared_batches_across_ranks(
        self,
        prepared_batches: Sequence[Dict[str, Any]],
        *,
        device: torch.device,
        runtime_stats: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        local_batches = list(prepared_batches or [])
        local_count = int(len(local_batches))
        all_ranks_have_batches, any_rank_has_batches = _distributed_bool_consensus(
            local_count > 0,
            device=device,
        )
        if any_rank_has_batches and not all_ranks_have_batches:
            donor_prepared_batch = _distributed_first_available_object(
                self._prepared_batch_cpu_copy(local_batches[0]) if local_batches else None,
                device=device,
            )
            if local_count <= 0 and donor_prepared_batch is not None:
                donor_device_batch = self._move_prepared_batch_to_device(donor_prepared_batch, device=device)
                local_batches = [self._clone_prepared_batch_as_noop(donor_device_batch)]
                runtime_stats["ddp_noop_padded_prepared_batches"] = 1
                self._ddp_noop_padded_prepared_batches += 1
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
                f"prepared_batches={int(runtime_stats.get('raw_local_prepared_batch_count', 0))} "
                f"trainable_samples={int(trainable_samples)} "
                f"fecv_failures={int(runtime_stats.get('local_fecv_failure_count', 0))} "
                f"min_weight_drops={int(runtime_stats.get('groups_filtered_by_min_weight', 0))} "
                f"replay_fills={int(runtime_stats.get('replay_fill_batches', 0))}"
            ),
            runtime=runtime,
            main_process_only=False,
        )

    def _ensure_reference_model_device(self, model: Any) -> None:
        if self.reference_model is None:
            return
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            return
        if self._reference_model_device != target_device:
            self.reference_model.to(target_device)
            self.reference_model.eval()
            self._reference_model_device = target_device

    def _prepare_advantages(self, batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
        advantages = batch.get("advantage")
        if advantages is None:
            advantages = batch.get("sample_weight")
        if advantages is None:
            raise ValueError("Dedicated GRPO trainer requires rollout advantages for every prepared batch.")
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
        policy_token_log_probs, response_mask = compute_completion_only_token_log_probs_from_ids(
            model=model,
            prompt_ids=batch["prompt_ids"],
            prompt_mask=batch["prompt_mask"],
            completion_ids=batch["completion_ids"],
            completion_mask=batch["completion_mask"],
            multimodal_inputs=self._prepared_batch_multimodal_inputs(batch),
            temperature=self.policy_temperature,
        )
        if not bool(policy_token_log_probs.requires_grad):
            raise RuntimeError("Policy completion log-probs are detached in the dedicated GRPO path.")
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
                        multimodal_inputs=self._prepared_batch_multimodal_inputs(batch),
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
                            multimodal_inputs=self._prepared_batch_multimodal_inputs(batch),
                            temperature=self.policy_temperature,
                        )
            if reference_token_log_probs is not None:
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
            for video_id in [str(video_id or "") for video_id in (prepared.get("video_ids") or [])]:
                if progress is not None:
                    progress.finish_item(video_id=video_id)
            runtime_stats = dict(prepared.get("runtime_stats") or {})
            prepared_batches = list(prepared.get("prepared_batches") or [])
            try:
                target_device = next(rollout_model.parameters()).device
            except StopIteration:
                target_device = torch.device("cpu")
            prepared_batches = [
                self._move_prepared_batch_to_device(prepared_batch, device=target_device)
                for prepared_batch in prepared_batches
            ]
            prepared_batches = self._align_prepared_batches_across_ranks(
                prepared_batches,
                device=target_device,
                runtime_stats=runtime_stats,
            )
            prepared_batches = self._materialize_prepared_batches(prepared_batches, device=target_device)
            runtime_stats["raw_local_prepared_batch_count"] = int(len(prepared_batches))
            return {
                "prepared_batches": prepared_batches,
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
        prepared_batches = list(inputs.get("prepared_batches") or [])
        runtime_stats = dict(inputs.get("runtime_stats") or {})
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")
        if not prepared_batches:
            self._maybe_log_empty_batch_rank_summary(
                reason="all_empty_prepared_batches",
                runtime_stats=runtime_stats,
                trainable_samples=0,
            )
            return _zero_loss_from_model(model)

        total_loss_sum = None
        total_samples = 0
        for batch in prepared_batches:
            for microbatch in self._iter_loss_microbatches(batch):
                sample_losses = self._compute_sample_losses_for_batch(model=model, batch=microbatch)
                if sample_losses is None or sample_losses.numel() <= 0:
                    continue
                sample_loss_multiplier = self._sample_loss_multiplier(
                    microbatch,
                    device=sample_losses.device,
                    sample_count=int(sample_losses.numel()),
                )
                total_samples += int((sample_loss_multiplier > 0).sum().item())
                batch_loss_sum = sample_losses.sum()
                total_loss_sum = batch_loss_sum if total_loss_sum is None else total_loss_sum + batch_loss_sum
        runtime_stats["raw_local_sample_count"] = int(total_samples)
        global_total_samples = _distributed_sum_int(int(total_samples), device=target_device)
        if total_loss_sum is None or global_total_samples <= 0:
            self._maybe_log_empty_batch_rank_summary(
                reason="all_empty_trainable_samples",
                runtime_stats=runtime_stats,
                trainable_samples=int(total_samples),
            )
            return _zero_loss_from_model(model)
        world_size = max(1, int(_distributed_world_size()))
        return total_loss_sum * float(world_size) / float(max(1, int(global_total_samples)))

    def get_budget_drop_metrics(self) -> Dict[str, Any]:
        metrics = self._budgeting_stats.as_dict()
        metrics.update(
            {
                "rl_zero_response_dropped": int(self._zero_response_dropped),
                "rl_materialize_fallback_batches": int(self._materialize_fallback_batches),
                "rl_replay_fill_batches": int(self._replay_fill_batches),
                "rl_replay_fill_prepared_batches": int(self._replay_fill_prepared_batches),
                "rl_groups_all_zero_advantage": int(self._groups_all_zero_advantage),
                "rl_groups_filtered_by_min_weight": int(self._groups_filtered_by_min_weight),
                "rl_fecv_failure_count": int(self._fecv_failure_count),
                "rl_fecv_degraded_rollout_count": int(self._fecv_degraded_rollout_count),
                "rl_ddp_noop_padded_prepared_batches": int(self._ddp_noop_padded_prepared_batches),
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
    reference_model: Any,
    use_lora_reference_disable_adapter: bool,
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
    iteration_index: int = 0,
    num_iterations: int = 1,
    rollout_eval_callback: Any = None,
    replay_buffer_enable: bool = True,
    replay_buffer_type: str = "ssr",
    replay_buffer_capacity: int = 16,
    replay_buffer_alpha: float = 1.0,
    fecv_failure_policy: str = "degrade",
    log_empty_batch_rank_summary: bool = True,
    reward_version: str = DEFAULT_RL_REWARD_VERSION,
    reward_config: Optional[Dict[str, Any]] = None,
    steps_per_generation: int = 1,
    policy_builder: Any = None,
    deepspeed: Optional[str] = None,
) -> Any:
    try:
        from transformers import Trainer, TrainingArguments
    except Exception as exc:
        raise ImportError("Dedicated GRPO trainer requires the `transformers` package.") from exc

    if not callable(policy_builder):
        raise ValueError("policy_builder is required for the dedicated GRPO trainer.")

    effective_persistent_workers = bool(dataloader_persistent_workers) and int(dataloader_num_workers) > 0
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
        "ddp_find_unused_parameters": False,
    }
    if int(dataloader_num_workers) > 0 and int(dataloader_prefetch_factor) > 0:
        training_args_kwargs["dataloader_prefetch_factor"] = int(dataloader_prefetch_factor)
    if str(deepspeed or "").strip():
        training_args_kwargs["deepspeed"] = str(deepspeed)
    training_args = TrainingArguments(**training_args_kwargs)

    aligned_grpo_config = {
        "processor": processor,
        "train_dataset": _RawItemDataset(train_items),
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
    if rollout_eval_callback is not None:
        trainer.add_callback(rollout_eval_callback)
    trainer.add_callback(_build_native_grpo_progress_callback(trainer=trainer))
    return trainer
