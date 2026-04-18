import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

import train_saver_rl

from saver_v3.rl import cli_shared
from saver_v3.rl.runtime import RLJobConfig, build_active_rl_trl_argv, run_rl_job
from saver_v3.core.rollout import SaverRolloutRunner, _build_episode_training_feature
from saver_v3.core.counterfactual_verification import run_counterfactual_verification_batch
from saver_v3.core.schema import SaverEnvironmentState
from saver_v3.sft.training import _build_rl_completion_episode_spec_from_feature
from saver_v3.model.vllm_generation import VllmQwenGenerationPolicy
from saver_v3.rl import timesearch_aligned_grpo_trainer as aligned_grpo_module
from saver_v3.rl import trl_grpo_trainer as trl_grpo_module
from saver_v3.rl.trl_grpo_trainer import MutableIterationDataset, TrlVllmGrpoRunner, _continuous_rl_args
from saver_v3.rl.timesearch_aligned_grpo_trainer import (
    TimesearchAlignedGRPOTrainerMixin,
    _USE_CURRENT_POLICY_LOGPROBS_SENTINEL,
    _build_managed_reference_model_like_timesearch_r,
    _resolve_liger_linear_head,
)
from saver_v3.rl.trl_grpo_trainer import _save_loadable_hf_authority_checkpoint


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class RLRuntimeTests(unittest.TestCase):
    def test_aggregate_generation_step_payload_balances_rollout_groups_across_ranks(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer._new_rollout_metric_lists = lambda: {"reward_total": []}
        trainer._new_runtime_stats = lambda: {}
        trainer.get_budget_drop_metrics = lambda: {"rl_zero_response_dropped": 0}
        trainer._aggregate_generation_step_payload = (
            TimesearchAlignedGRPOTrainerMixin._aggregate_generation_step_payload.__get__(trainer)
        )

        def _episode(token_id: int) -> dict:
            return {
                "prompt_ids": torch.tensor([[token_id]], dtype=torch.long),
                "prompt_mask": torch.tensor([[1]], dtype=torch.long),
                "completion_ids": torch.tensor([[token_id + 100]], dtype=torch.long),
                "completion_mask": torch.tensor([[1]], dtype=torch.bool),
                "advantages": torch.tensor([1.0], dtype=torch.float32),
            }

        local_payload = {
            "video_id": "local",
            "rollout_groups": [
                {
                    "video_id": "local",
                    "generation_id": 0,
                    "source_item_index": 0,
                    "source_rollout_index": 0,
                    "episode_inputs": [_episode(1), _episode(2), _episode(3), _episode(4)],
                }
            ],
            "rollout_metric_values": {"reward_total": [1.0]},
            "runtime_stats": {"foo": 1},
        }
        gathered_rollout_groups = [
            list(local_payload["rollout_groups"]),
            [
                {
                    "video_id": "remote_a",
                    "generation_id": 1,
                    "source_item_index": 0,
                    "source_rollout_index": 0,
                    "episode_inputs": [_episode(11), _episode(12), _episode(13), _episode(14)],
                },
                {
                    "video_id": "remote_b",
                    "generation_id": 2,
                    "source_item_index": 0,
                    "source_rollout_index": 1,
                    "episode_inputs": [_episode(21), _episode(22), _episode(23), _episode(24)],
                },
            ],
        ]
        logged_messages = []
        with mock.patch.object(aligned_grpo_module, "_distributed_rank", return_value=0), mock.patch.object(
            aligned_grpo_module,
            "_distributed_gather_object",
            return_value=gathered_rollout_groups,
        ), mock.patch.object(
            aligned_grpo_module,
            "runtime_log",
            side_effect=lambda message, **kwargs: logged_messages.append(str(message)),
        ):
            payload = trainer._aggregate_generation_step_payload([local_payload])

        self.assertEqual(payload["runtime_stats"]["raw_local_rollout_group_count_before_slice"], 1)
        self.assertEqual(payload["runtime_stats"]["raw_local_rollout_group_count_after_slice"], 2)
        self.assertEqual(len(payload["episode_inputs"]), 8)
        prompt_first_tokens = [int(batch["prompt_ids"][0, 0].item()) for batch in payload["episode_inputs"]]
        self.assertIn(1, prompt_first_tokens)
        self.assertIn(21, prompt_first_tokens)
        self.assertTrue(any("assigned_groups=2" in message for message in logged_messages))
        self.assertTrue(any("assigned_samples=8" in message for message in logged_messages))

    def test_aggregate_generation_step_payload_wraps_episode_inputs_without_rollout_groups(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer._new_rollout_metric_lists = lambda: {"reward_total": []}
        trainer._new_runtime_stats = lambda: {}
        trainer.get_budget_drop_metrics = lambda: {}
        trainer._aggregate_generation_step_payload = (
            TimesearchAlignedGRPOTrainerMixin._aggregate_generation_step_payload.__get__(trainer)
        )

        payload = {
            "video_id": "vid1",
            "episode_inputs": [
                {
                    "prompt_ids": torch.tensor([[1]], dtype=torch.long),
                    "prompt_mask": torch.tensor([[1]], dtype=torch.long),
                    "completion_ids": torch.tensor([[2]], dtype=torch.long),
                    "completion_mask": torch.tensor([[1]], dtype=torch.bool),
                    "advantages": torch.tensor([1.0], dtype=torch.float32),
                },
                {
                    "prompt_ids": torch.tensor([[3]], dtype=torch.long),
                    "prompt_mask": torch.tensor([[1]], dtype=torch.long),
                    "completion_ids": torch.tensor([[4]], dtype=torch.long),
                    "completion_mask": torch.tensor([[1]], dtype=torch.bool),
                    "advantages": torch.tensor([1.0], dtype=torch.float32),
                },
            ],
            "rollout_metric_values": {"reward_total": [0.5]},
            "runtime_stats": {"foo": 2},
        }

        with mock.patch.object(aligned_grpo_module, "_distributed_rank", return_value=0), mock.patch.object(
            aligned_grpo_module,
            "_distributed_gather_object",
            return_value=[[
                {
                    "video_id": "vid1",
                    "group_id": "vid1",
                    "generation_id": -1,
                    "source_item_index": 0,
                    "source_rollout_index": 0,
                    "episode_inputs": payload["episode_inputs"],
                }
            ]],
        ):
            aggregated = trainer._aggregate_generation_step_payload([payload])

        self.assertEqual(aggregated["runtime_stats"]["raw_local_rollout_group_count_before_slice"], 1)
        self.assertEqual(aggregated["runtime_stats"]["raw_local_rollout_group_count_after_slice"], 1)
        self.assertEqual(len(aggregated["episode_inputs"]), 2)
        self.assertEqual(aggregated["video_ids"], ["vid1"])

    def test_save_loadable_hf_authority_checkpoint_uses_accelerator_state_dict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            class TinyModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.tensor([1.0]))
                    self.saved = None

                def save_pretrained(self, path: str, state_dict=None):
                    self.saved = {"path": path, "state_dict": state_dict}

            model = TinyModel()
            accelerator = mock.Mock()
            accelerator.unwrap_model.return_value = model
            accelerator.get_state_dict.return_value = {"weight": torch.tensor([2.0])}
            trainer = mock.Mock()
            trainer.accelerator = accelerator
            trainer.model = object()
            trainer.model_wrapped = object()
            processor = mock.Mock()
            runtime = mock.Mock()
            runtime.is_main_process = True

            path = _save_loadable_hf_authority_checkpoint(
                trainer=trainer,
                processor=processor,
                checkpoint_root=tmp_path / "checkpoint",
                epoch_index=1,
                runtime=runtime,
            )

            accelerator.get_state_dict.assert_called_once_with(trainer.model_wrapped, unwrap=False)
            self.assertEqual(path, tmp_path / "checkpoint" / "authority_hf" / "epoch_001")
            self.assertEqual(model.saved["path"], str(path))
            self.assertTrue(torch.equal(model.saved["state_dict"]["weight"], torch.tensor([2.0])))
            processor.save_pretrained.assert_called_once_with(str(path))

    def test_repo_rl_yaml_default_min_weight_is_zero_for_diagnostics(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        payload = yaml.safe_load((repo_root / "configs/rl/qwen3_vl_8b_grpo_train.yaml").read_text(encoding="utf-8"))
        self.assertEqual(float(payload["optimization"]["min_weight"]), 0.0)

    def test_legacy_native_train_entrypoint_fails_fast(self) -> None:
        fake_args = mock.Mock()
        with mock.patch.object(train_saver_rl, "parse_args", return_value=fake_args):
            with self.assertRaisesRegex(RuntimeError, "legacy native GRPO backend is deprecated and unsupported"):
                train_saver_rl.main()

    def test_episode_training_feature_uses_full_assistant_trajectory(self) -> None:
        result = {
            "video_id": "vid1",
            "group_id": "vid1",
            "generation_id": 0,
            "turns": [
                {
                    "step_index": 1,
                    "action": "tool_call",
                    "tool_name": "scan_timeline",
                    "_rl_token_trace": {
                        "prompt_trace": {
                            "prompt_ids": torch.tensor([[1, 2]], dtype=torch.long),
                            "prompt_mask": torch.tensor([[1, 1]], dtype=torch.long),
                            "pixel_values": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
                        },
                        "completion_ids": torch.tensor([[3, 4]], dtype=torch.long),
                        "completion_mask": torch.tensor([[1, 1]], dtype=torch.bool),
                    },
                },
                {
                    "step_index": 2,
                    "action": "answer",
                    "tool_name": None,
                    "_rl_token_trace": {
                        "prompt_trace": {
                            "prompt_ids": torch.tensor([[1, 2, 7]], dtype=torch.long),
                            "prompt_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
                            "image_grid_thw": torch.tensor([[1, 2, 3]], dtype=torch.long),
                        },
                        "completion_ids": torch.tensor([[5, 6]], dtype=torch.long),
                        "completion_mask": torch.tensor([[1, 1]], dtype=torch.bool),
                    },
                },
            ],
        }
        feature = _build_episode_training_feature(result=result)
        self.assertIsNotNone(feature)
        self.assertIn("episode_turn_samples", feature)
        self.assertEqual(len(feature["episode_turn_samples"]), 2)
        first_turn, second_turn = feature["episode_turn_samples"]
        self.assertTrue(torch.equal(first_turn["episode_prompt_trace"]["prompt_ids"], torch.tensor([[1, 2]], dtype=torch.long)))
        self.assertTrue(torch.equal(second_turn["episode_prompt_trace"]["prompt_ids"], torch.tensor([[1, 2, 7]], dtype=torch.long)))
        self.assertIn("multimodal_inputs", first_turn["episode_prompt_trace"])
        self.assertIn("pixel_values", first_turn["episode_prompt_trace"]["multimodal_inputs"])
        self.assertIn("image_grid_thw", second_turn["episode_prompt_trace"]["multimodal_inputs"])
        self.assertEqual(first_turn["sample_weight"], 1.0)
        self.assertEqual(second_turn["sample_weight"], 2.0)

    def test_episode_training_feature_rejects_legacy_prompt_trace_keys(self) -> None:
        result = {
            "video_id": "vid1",
            "turns": [
                {
                    "step_index": 1,
                    "action": "tool_call",
                    "tool_name": "scan_timeline",
                    "_rl_token_trace": {
                        "prompt_trace": {
                            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
                        },
                        "completion_ids": torch.tensor([[3, 4]], dtype=torch.long),
                        "completion_mask": torch.tensor([[1, 1]], dtype=torch.bool),
                    },
                }
            ],
        }
        with self.assertRaisesRegex(ValueError, "prompt_trace must contain tensor `prompt_ids` and `prompt_mask`"):
            _build_episode_training_feature(result=result)

    def test_episode_training_feature_raises_when_valid_turns_have_no_traces(self) -> None:
        result = {
            "video_id": "vid1",
            "terminated_reason": "finalized",
            "turns": [
                {
                    "step_index": 1,
                    "action": "tool_call",
                    "tool_name": "scan_timeline",
                    "valid_action": True,
                },
                {
                    "step_index": 2,
                    "action": "tool_call",
                    "tool_name": "finalize_case",
                    "valid_action": True,
                },
            ],
        }
        with self.assertRaisesRegex(ValueError, "valid turns but no token traces were attached"):
            _build_episode_training_feature(result=result)

    def test_episode_feature_builds_episode_batch(self) -> None:
        processor = object()
        feature = {
            "episode_prompt_trace": {
                "prompt_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "prompt_mask": torch.tensor([[1, 1]], dtype=torch.long),
                "multimodal_inputs": {
                    "pixel_values": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
                },
            },
            "completion_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "completion_mask": torch.tensor([[1, 1]], dtype=torch.bool),
            "advantages": 1.0,
            "sample_weight": 2.0,
            "sample_loss_multiplier": 1.0,
        }
        result = _build_rl_completion_episode_spec_from_feature(processor, feature, max_seq_length=4096)
        self.assertIsNotNone(result.batch)
        self.assertIn("prompt_ids", result.batch)
        self.assertIn("completion_ids", result.batch)
        self.assertIn("completion_mask", result.batch)
        self.assertIn("old_policy_token_log_probs", result.batch)
        self.assertIn("advantages", result.batch)
        self.assertIn("sample_weight", result.batch)
        self.assertIn("sample_loss_multiplier", result.batch)
        self.assertIn("multimodal_inputs", result.batch)
        self.assertIn("pixel_values", result.batch["multimodal_inputs"])
        self.assertTrue(result.batch["completion_mask"].any().item())
        self.assertEqual(float(result.batch["sample_weight"].item()), 2.0)

    def test_episode_feature_requires_advantages(self) -> None:
        processor = object()
        feature = {
            "episode_prompt_trace": {
                "prompt_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "prompt_mask": torch.tensor([[1, 1]], dtype=torch.long),
                "multimodal_inputs": {},
            },
            "completion_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "completion_mask": torch.tensor([[1, 1]], dtype=torch.bool),
        }
        result = _build_rl_completion_episode_spec_from_feature(processor, feature, max_seq_length=4096)
        self.assertIsNone(result.batch)
        self.assertEqual(result.drop_reason, "missing_rollout_advantages")

    def test_episode_feature_rejects_legacy_flat_prompt_trace_multimodal_tensors(self) -> None:
        processor = object()
        feature = {
            "episode_prompt_trace": {
                "prompt_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "prompt_mask": torch.tensor([[1, 1]], dtype=torch.long),
                "pixel_values": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
            },
            "completion_ids": torch.tensor([[3, 4]], dtype=torch.long),
            "completion_mask": torch.tensor([[1, 1]], dtype=torch.bool),
            "advantages": 1.0,
        }
        with self.assertRaisesRegex(ValueError, "Legacy RL pure-pack prompt_trace format detected"):
            _build_rl_completion_episode_spec_from_feature(processor, feature, max_seq_length=4096)

    def test_episode_feature_rejects_legacy_prompt_completion_shape(self) -> None:
        processor = object()
        feature = {
            "prompt_messages": [{"role": "user", "content": [{"type": "text", "text": "q"}]}],
            "completion_text": "a",
            "advantages": 1.0,
        }
        result = _build_rl_completion_episode_spec_from_feature(processor, feature, max_seq_length=4096)
        self.assertIsNone(result.batch)
        self.assertEqual(result.drop_reason, "missing_pure_pack_materials")

    def test_old_logprob_reuse_condition_matches_timesearch_r_rule(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.num_iterations = 1
        trainer.steps_per_generation = 2
        trainer.args = type("Args", (), {"gradient_accumulation_steps": 4})()
        trainer.use_liger_loss = False
        self.assertTrue(trainer._can_reuse_current_policy_as_old_logprobs())

        trainer.steps_per_generation = 8
        self.assertFalse(trainer._can_reuse_current_policy_as_old_logprobs())

        trainer.num_iterations = 2
        trainer.steps_per_generation = 1
        self.assertTrue(trainer._can_reuse_current_policy_as_old_logprobs())

        trainer.use_liger_loss = True
        self.assertTrue(trainer._can_reuse_current_policy_as_old_logprobs())

        trainer.steps_per_generation = 2
        self.assertFalse(trainer._can_reuse_current_policy_as_old_logprobs())

    def test_populate_old_logprobs_uses_sentinel_when_reuse_is_safe(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.num_iterations = 1
        trainer.steps_per_generation = 1
        trainer.args = type("Args", (), {"gradient_accumulation_steps": 4})()
        trainer.use_liger_loss = False
        episode_inputs = [
            {
                "prompt_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "prompt_mask": torch.tensor([[1, 1]], dtype=torch.long),
                "completion_ids": torch.tensor([[3, 4]], dtype=torch.long),
                "completion_mask": torch.tensor([[1, 1]], dtype=torch.bool),
            }
        ]
        populated = trainer._populate_old_policy_log_probs(model=None, episode_inputs=episode_inputs)
        self.assertEqual(populated[0]["old_policy_token_log_probs"], _USE_CURRENT_POLICY_LOGPROBS_SENTINEL)

    def test_populate_old_logprobs_strips_none_placeholder_before_signature_grouping(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.num_iterations = 1
        trainer.steps_per_generation = 8
        trainer.args = type("Args", (), {"gradient_accumulation_steps": 4})()
        trainer.use_liger_loss = False
        trainer.processor = type("Processor", (), {"pad_token_id": 0, "eos_token_id": 0})()
        trainer._move_episode_input_to_device = lambda episode_input, device: episode_input

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        trainer._compute_old_policy_token_log_probs_for_episode_input = lambda model, episode_input: torch.tensor([[0.1, 0.2]], dtype=torch.float32)
        episode_inputs = [
            {
                "prompt_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "prompt_mask": torch.tensor([[1, 1]], dtype=torch.long),
                "completion_ids": torch.tensor([[3, 4]], dtype=torch.long),
                "completion_mask": torch.tensor([[1, 1]], dtype=torch.bool),
                "old_policy_token_log_probs": None,
            }
        ]
        populated = trainer._populate_old_policy_log_probs(model=TinyModel(), episode_inputs=episode_inputs)
        self.assertTrue(isinstance(populated[0]["old_policy_token_log_probs"], torch.Tensor))

    def test_effective_local_rollout_batch_size_uses_smaller_stage_limit(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.rollout_stage_batch_size = 16
        trainer.fecv_stage_batch_size = 4
        self.assertEqual(trainer._effective_local_rollout_batch_size(), 4)

        trainer.rollout_stage_batch_size = 3
        trainer.fecv_stage_batch_size = 8
        self.assertEqual(trainer._effective_local_rollout_batch_size(), 3)

    def test_rollout_preserves_rl_token_trace_for_episode_training_feature(self) -> None:
        class FakeAdapter:
            def build_initial_messages(self, item):
                return list(item["messages"])

            def build_assistant_message(self, text):
                return {"role": "assistant", "content": [{"type": "text", "text": text}]}

            def adapt_tool_observation(self, tool_message, multimodal_cache):
                del multimodal_cache
                return tool_message

        class FakeEnvironment:
            def execute_predictions(self, predictions, multimodal_cache_batch, state_batch, active_flags):
                del predictions, multimodal_cache_batch, active_flags
                return (
                    [{"role": "tool", "name": "scan_timeline", "content": [{"type": "text", "text": "ok"}]}],
                    [1],
                    [1],
                    [1],
                    [SaverEnvironmentState()],
                )

        class FakePolicy:
            def __init__(self):
                self._traces = [
                    {
                        "prompt_trace": {
                            "prompt_ids": torch.tensor([[1, 2]], dtype=torch.long),
                            "prompt_mask": torch.tensor([[1, 1]], dtype=torch.long),
                        },
                        "completion_ids": torch.tensor([[3, 4]], dtype=torch.long),
                        "completion_mask": torch.tensor([[1, 1]], dtype=torch.bool),
                    }
                ]
                self.seen_batch_sizes = []

            def generate_from_messages_batch(self, messages_batch):
                self.seen_batch_sizes.append(len(messages_batch))
                return ['<tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":1.0}}</tool_call>']

            def pop_last_rl_token_traces(self):
                traces = self._traces
                self._traces = None
                return traces

        runner = SaverRolloutRunner(environment=FakeEnvironment(), adapter=FakeAdapter(), max_turns=1)
        policy = FakePolicy()
        result = runner.run_episode(
            {
                "video_id": "vid-trace",
                "messages": [{"role": "user", "content": [{"type": "text", "text": "inspect"}]}],
                "multimodal_cache": {"preview_frames": [], "preview_timestamps": []},
            },
            policy,
        )

        self.assertEqual(policy.seen_batch_sizes, [1])
        self.assertIn("_rl_episode_training_feature", result)
        feature = result["_rl_episode_training_feature"]
        self.assertEqual(len(feature["episode_turn_samples"]), 1)
        self.assertTrue(torch.equal(feature["episode_turn_samples"][0]["completion_ids"], torch.tensor([[3, 4]])))
        self.assertNotIn("_rl_token_trace", result["turns"][0])
        self.assertNotIn("_prompt_messages", result["turns"][0])

    def test_training_step_skips_backward_when_empty_batch_is_globally_empty(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.accelerator = mock.Mock()
        trainer.use_liger_loss = False
        trainer._skip_empty_training_steps = 0
        trainer._last_logged_empty_batch = None
        trainer.compute_loss_context_manager = lambda: contextlib.nullcontext()
        trainer._prepare_inputs = lambda inputs: inputs
        trainer._maybe_skip_empty_training_step = mock.Mock(return_value=torch.tensor(0.0))

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        model = TinyModel()
        loss = trainer.training_step(model, {"episode_inputs": [], "runtime_stats": {}})

        trainer._maybe_skip_empty_training_step.assert_called_once()
        trainer.accelerator.backward.assert_not_called()
        self.assertEqual(float(loss.item()), 0.0)

    def test_training_step_does_not_manually_scale_loss_under_deepspeed(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.accelerator = mock.Mock()
        trainer.accelerator.distributed_type = "DEEPSPEED"
        trainer.use_liger_loss = False
        trainer.compute_loss_context_manager = lambda: contextlib.nullcontext()
        trainer._prepare_inputs = lambda inputs: inputs
        trainer._maybe_skip_empty_training_step = mock.Mock(return_value=None)
        trainer._ensure_liger_runtime_ready = lambda model: None
        trainer._iter_loss_microbatches = lambda batch: [batch]
        trainer._compute_sample_losses_for_batch = mock.Mock(return_value=torch.tensor([22.0], dtype=torch.float32, requires_grad=True))
        trainer._move_episode_input_to_device = lambda episode_input, device: episode_input
        trainer._episode_input_sample_count = TimesearchAlignedGRPOTrainerMixin._episode_input_sample_count.__get__(trainer)
        trainer._effective_sample_weight = TimesearchAlignedGRPOTrainerMixin._effective_sample_weight.__get__(trainer)
        trainer._sample_loss_multiplier = TimesearchAlignedGRPOTrainerMixin._sample_loss_multiplier.__get__(trainer)
        trainer._sample_weight = TimesearchAlignedGRPOTrainerMixin._sample_weight.__get__(trainer)
        trainer.args = type("Args", (), {"n_gpu": 1})()
        trainer.current_gradient_accumulation_steps = 22
        trainer.model_accepts_loss_kwargs = False
        trainer.compute_loss_func = None

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        model = TinyModel()
        batch = {
            "prompt_ids": torch.tensor([[1]], dtype=torch.long),
            "prompt_mask": torch.tensor([[1]], dtype=torch.long),
            "completion_ids": torch.tensor([[2]], dtype=torch.long),
            "completion_mask": torch.tensor([[1]], dtype=torch.bool),
            "sample_loss_multiplier": torch.tensor([1.0], dtype=torch.float32),
            "sample_weight": torch.tensor([1.0], dtype=torch.float32),
        }
        loss = trainer.training_step(model, {"episode_inputs": [batch], "runtime_stats": {}})

        trainer.accelerator.backward.assert_called_once()
        backward_loss = trainer.accelerator.backward.call_args.args[0]
        backward_kwargs = trainer.accelerator.backward.call_args.kwargs
        self.assertAlmostEqual(float(backward_loss.item()), 22.0, places=6)
        self.assertEqual(backward_kwargs.get("scale_wrt_gas"), False)
        self.assertAlmostEqual(float(loss.item()), 22.0, places=6)

    def test_resolve_liger_unwrapped_model_prefers_accelerator_unwrap(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.accelerator = mock.Mock()
        unwrapped = object()
        trainer.accelerator.unwrap_model.return_value = unwrapped
        resolved = trainer._resolve_liger_unwrapped_model(object())
        self.assertIs(resolved, unwrapped)

    def test_resolve_liger_linear_head_uses_output_embeddings_when_top_level_lm_head_missing(self) -> None:
        class OutputHead:
            def __init__(self) -> None:
                self.weight = torch.ones((5, 4), dtype=torch.float32)
                self.bias = torch.zeros((5,), dtype=torch.float32)

        output_head = OutputHead()
        unwrapped = type(
            "Unwrapped",
            (),
            {
                "get_output_embeddings": lambda self: output_head,
            },
        )()

        resolved_head, resolved_weight, resolved_bias, _ = _resolve_liger_linear_head(unwrapped)

        self.assertIs(resolved_head, output_head)
        self.assertTrue(torch.equal(resolved_weight, output_head.weight))
        self.assertTrue(torch.equal(resolved_bias, output_head.bias))

    def test_compute_liger_loss_uses_output_embeddings_head_when_top_level_lm_head_missing(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.liger_grpo_loss = mock.Mock(return_value=(torch.tensor(3.0, dtype=torch.float32), None))
        trainer.accelerator = mock.Mock()
        trainer.policy_temperature = None
        trainer.kl_beta = 0.0
        trainer.reference_model = None
        trainer._episode_input_multimodal_inputs = lambda batch: {}
        trainer._prepare_advantages = lambda batch, device: torch.ones((1,), dtype=torch.float32, device=device)
        output_head = type(
            "OutputHead",
            (),
            {
                "weight": torch.ones((5, 4), dtype=torch.float32),
                "bias": None,
            },
        )()
        backbone = mock.Mock(return_value=None)
        trainer.accelerator.unwrap_model.return_value = type(
            "Unwrapped",
            (),
            {
                "model": backbone,
                "get_output_embeddings": lambda self: output_head,
            },
        )()

        class BackboneOutputs:
            last_hidden_state = torch.zeros((1, 2, 4), dtype=torch.float32)

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))

            def forward(self, **kwargs):
                raise AssertionError("compute_liger_loss should use the unwrapped backbone path")

        backbone.return_value = BackboneOutputs()

        fake_model = FakeModel()
        batch = {
            "prompt_ids": torch.tensor([[1]], dtype=torch.long),
            "prompt_mask": torch.tensor([[1]], dtype=torch.long),
            "completion_ids": torch.tensor([[2]], dtype=torch.long),
            "completion_mask": torch.tensor([[1]], dtype=torch.bool),
        }
        trainer._compute_liger_loss_for_batch(
            trainer.accelerator.unwrap_model.return_value,
            batch,
        )

        trainer.liger_grpo_loss.assert_called_once()
        self.assertTrue(
            torch.equal(
                trainer.liger_grpo_loss.call_args.kwargs["lin_weight"],
                output_head.weight,
            )
        )

    def test_compute_liger_loss_applies_multiplier_and_sample_weight(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.policy_temperature = None
        trainer.kl_beta = 0.0
        trainer.reference_model = None
        trainer._episode_input_multimodal_inputs = lambda batch: {}
        trainer._prepare_advantages = lambda batch, device: torch.ones((2,), dtype=torch.float32, device=device)
        trainer.liger_grpo_loss = mock.Mock(return_value=(torch.tensor(3.0, dtype=torch.float32), None))

        class BackboneOutputs:
            last_hidden_state = torch.zeros((2, 3, 4), dtype=torch.float32)

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))
                self.lm_head = type("LMHead", (), {"weight": torch.ones((5, 4), dtype=torch.float32), "bias": None})()
                self.model = mock.Mock(return_value=BackboneOutputs())

        trainer.accelerator = mock.Mock()
        fake_model = FakeModel()
        trainer.accelerator.unwrap_model.return_value = fake_model
        batch = {
            "prompt_ids": torch.tensor([[1], [1]], dtype=torch.long),
            "prompt_mask": torch.tensor([[1], [1]], dtype=torch.long),
            "completion_ids": torch.tensor([[2, 3], [2, 3]], dtype=torch.long),
            "completion_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.bool),
            "sample_loss_multiplier": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "sample_weight": torch.tensor([5.0, 2.0], dtype=torch.float32),
        }
        loss = trainer._compute_liger_loss_for_batch(fake_model, batch)
        self.assertTrue(torch.equal(loss, torch.tensor([0.0, 6.0], dtype=torch.float32)))

    def test_compute_loss_does_not_disable_liger_before_batch_compute(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.use_liger_loss = True
        trainer.use_liger_loss_requested = True
        trainer.use_liger_loss_effective = False
        trainer.compute_loss_microbatch_size = 2
        trainer._liger_runtime_probe_completed = False
        trainer._liger_runtime_disable_reason = None
        trainer.liger_grpo_loss = object()
        trainer._forward_redirection = mock.Mock(side_effect=lambda model, unwrapped_model, func, *args: func(*args))
        trainer._maybe_log_empty_batch_rank_summary = mock.Mock()
        trainer._iter_loss_microbatches = lambda batch: [batch, batch]
        trainer._compute_liger_loss_for_batch = mock.Mock(
            side_effect=[
                torch.tensor([1.0], dtype=torch.float32),
                torch.tensor([2.0], dtype=torch.float32),
                torch.tensor([1.0], dtype=torch.float32),
                torch.tensor([2.0], dtype=torch.float32),
            ]
        )
        trainer._compute_sample_losses_for_batch = mock.Mock(return_value=torch.tensor([2.0], dtype=torch.float32))
        trainer._effective_sample_weight = TimesearchAlignedGRPOTrainerMixin._effective_sample_weight.__get__(trainer)
        trainer._sample_loss_multiplier = TimesearchAlignedGRPOTrainerMixin._sample_loss_multiplier.__get__(trainer)
        trainer._sample_weight = TimesearchAlignedGRPOTrainerMixin._sample_weight.__get__(trainer)
        trainer._resolve_liger_unwrapped_model = TimesearchAlignedGRPOTrainerMixin._resolve_liger_unwrapped_model.__get__(trainer)
        trainer._ensure_liger_runtime_ready = TimesearchAlignedGRPOTrainerMixin._ensure_liger_runtime_ready.__get__(trainer)
        trainer.compute_loss = TimesearchAlignedGRPOTrainerMixin.compute_loss.__get__(trainer)
        trainer._episode_input_sample_count = TimesearchAlignedGRPOTrainerMixin._episode_input_sample_count.__get__(trainer)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        model = TinyModel()
        trainer.accelerator = mock.Mock()
        trainer.accelerator.unwrap_model.return_value = model
        batch = {
            "prompt_ids": torch.tensor([[1]], dtype=torch.long),
            "prompt_mask": torch.tensor([[1]], dtype=torch.long),
            "completion_ids": torch.tensor([[2]], dtype=torch.long),
            "completion_mask": torch.tensor([[1]], dtype=torch.bool),
            "sample_loss_multiplier": torch.tensor([1.0], dtype=torch.float32),
            "sample_weight": torch.tensor([1.0], dtype=torch.float32),
        }
        loss = trainer.compute_loss(model, {"episode_inputs": [dict(batch), dict(batch)], "runtime_stats": {}})

        self.assertAlmostEqual(float(loss.item()), 1.5, places=6)
        self.assertTrue(trainer.use_liger_loss)
        self.assertTrue(trainer.use_liger_loss_effective)
        self.assertIsNone(trainer._liger_runtime_disable_reason)
        self.assertEqual(trainer._forward_redirection.call_count, 4)
        self.assertEqual(trainer._compute_liger_loss_for_batch.call_count, 4)
        trainer._compute_sample_losses_for_batch.assert_not_called()

    def test_compute_loss_normalizes_by_effective_weight(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.use_liger_loss = False
        trainer._maybe_log_empty_batch_rank_summary = mock.Mock()
        trainer._iter_loss_microbatches = lambda batch: [batch]
        trainer._compute_sample_losses_for_batch = mock.Mock(
            return_value=torch.tensor([0.0, 6.0], dtype=torch.float32)
        )

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        model = TinyModel()
        batch = {
            "prompt_ids": torch.tensor([[1], [1]], dtype=torch.long),
            "prompt_mask": torch.tensor([[1], [1]], dtype=torch.long),
            "completion_ids": torch.tensor([[2], [2]], dtype=torch.long),
            "completion_mask": torch.tensor([[1], [1]], dtype=torch.bool),
            "sample_loss_multiplier": torch.tensor([0.0, 1.0], dtype=torch.float32),
            "sample_weight": torch.tensor([5.0, 2.0], dtype=torch.float32),
        }
        loss = trainer.compute_loss(model, {"episode_inputs": [batch], "runtime_stats": {}})
        self.assertAlmostEqual(float(loss.item()), 3.0, places=6)

    def test_compute_loss_logs_progress_per_batch(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.use_liger_loss = False
        trainer.compute_loss_microbatch_size = 2
        trainer._maybe_log_empty_batch_rank_summary = mock.Mock()
        trainer._episode_input_sample_count = TimesearchAlignedGRPOTrainerMixin._episode_input_sample_count.__get__(trainer)
        trainer._iter_loss_microbatches = lambda batch: [batch]
        trainer._compute_sample_losses_for_batch = mock.Mock(
            side_effect=[
                torch.tensor([1.0], dtype=torch.float32),
                torch.tensor([2.0], dtype=torch.float32),
            ]
        )

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        model = TinyModel()
        batch = {
            "prompt_ids": torch.tensor([[1]], dtype=torch.long),
            "prompt_mask": torch.tensor([[1]], dtype=torch.long),
            "completion_ids": torch.tensor([[2]], dtype=torch.long),
            "completion_mask": torch.tensor([[1]], dtype=torch.bool),
            "sample_loss_multiplier": torch.tensor([1.0], dtype=torch.float32),
            "sample_weight": torch.tensor([1.0], dtype=torch.float32),
        }
        logged_messages = []
        with mock.patch.object(aligned_grpo_module, "runtime_log", side_effect=lambda message, **kwargs: logged_messages.append(str(message))):
            loss = trainer.compute_loss(model, {"episode_inputs": [dict(batch), dict(batch)], "runtime_stats": {}})

        self.assertAlmostEqual(float(loss.item()), 1.5, places=6)
        self.assertTrue(any("rl compute_loss start:" in message for message in logged_messages))
        self.assertTrue(any("rl compute_loss batch start: batch=1/2" in message for message in logged_messages))
        self.assertTrue(any("rl compute_loss batch start: batch=2/2" in message for message in logged_messages))
        self.assertTrue(any("rl compute_loss end:" in message for message in logged_messages))

    def test_training_step_logs_backward_start_end(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.accelerator = mock.Mock()
        trainer.accelerator.distributed_type = "DEEPSPEED"
        trainer.use_liger_loss = False
        trainer.compute_loss_context_manager = lambda: contextlib.nullcontext()
        trainer._prepare_inputs = lambda inputs: inputs
        trainer._maybe_skip_empty_training_step = mock.Mock(return_value=None)
        trainer._ensure_liger_runtime_ready = lambda model: None
        trainer._iter_loss_microbatches = lambda batch: [batch]
        trainer._compute_sample_losses_for_batch = mock.Mock(return_value=torch.tensor([22.0], dtype=torch.float32, requires_grad=True))
        trainer._move_episode_input_to_device = lambda episode_input, device: episode_input
        trainer._episode_input_sample_count = TimesearchAlignedGRPOTrainerMixin._episode_input_sample_count.__get__(trainer)
        trainer._effective_sample_weight = TimesearchAlignedGRPOTrainerMixin._effective_sample_weight.__get__(trainer)
        trainer._sample_loss_multiplier = TimesearchAlignedGRPOTrainerMixin._sample_loss_multiplier.__get__(trainer)
        trainer._sample_weight = TimesearchAlignedGRPOTrainerMixin._sample_weight.__get__(trainer)
        trainer.args = type("Args", (), {"n_gpu": 1})()
        trainer.current_gradient_accumulation_steps = 1
        trainer.model_accepts_loss_kwargs = False
        trainer.compute_loss_func = None

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        model = TinyModel()
        batch = {
            "prompt_ids": torch.tensor([[1]], dtype=torch.long),
            "prompt_mask": torch.tensor([[1]], dtype=torch.long),
            "completion_ids": torch.tensor([[2]], dtype=torch.long),
            "completion_mask": torch.tensor([[1]], dtype=torch.bool),
            "sample_loss_multiplier": torch.tensor([1.0], dtype=torch.float32),
            "sample_weight": torch.tensor([1.0], dtype=torch.float32),
        }
        logged_messages = []
        with mock.patch.object(aligned_grpo_module, "runtime_log", side_effect=lambda message, **kwargs: logged_messages.append(str(message))):
            trainer.training_step(model, {"episode_inputs": [batch], "runtime_stats": {}})

        self.assertTrue(any("rl backward start:" in message for message in logged_messages))
        self.assertTrue(any("rl backward end:" in message for message in logged_messages))

    def test_training_step_immediate_microbatch_backward_pads_to_shared_max_count(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.accelerator = mock.Mock()
        trainer.accelerator.distributed_type = "DEEPSPEED"
        trainer.use_liger_loss = False
        trainer.compute_loss_context_manager = lambda: contextlib.nullcontext()
        trainer._prepare_inputs = lambda inputs: inputs
        trainer._maybe_skip_empty_training_step = mock.Mock(return_value=None)
        trainer._ensure_liger_runtime_ready = lambda model: None
        trainer._iter_loss_microbatches = lambda batch: [batch]
        trainer._compute_sample_losses_for_batch = mock.Mock(return_value=torch.tensor([5.0], dtype=torch.float32, requires_grad=True))
        trainer._move_episode_input_to_device = lambda episode_input, device: episode_input
        trainer._episode_input_sample_count = TimesearchAlignedGRPOTrainerMixin._episode_input_sample_count.__get__(trainer)
        trainer._effective_sample_weight = TimesearchAlignedGRPOTrainerMixin._effective_sample_weight.__get__(trainer)
        trainer._sample_loss_multiplier = TimesearchAlignedGRPOTrainerMixin._sample_loss_multiplier.__get__(trainer)
        trainer._sample_weight = TimesearchAlignedGRPOTrainerMixin._sample_weight.__get__(trainer)
        trainer.args = type("Args", (), {"n_gpu": 1})()
        trainer.current_gradient_accumulation_steps = 1
        trainer.model_accepts_loss_kwargs = False
        trainer.compute_loss_func = None

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        model = TinyModel()
        batch = {
            "prompt_ids": torch.tensor([[1]], dtype=torch.long),
            "prompt_mask": torch.tensor([[1]], dtype=torch.long),
            "completion_ids": torch.tensor([[2]], dtype=torch.long),
            "completion_mask": torch.tensor([[1]], dtype=torch.bool),
            "sample_loss_multiplier": torch.tensor([1.0], dtype=torch.float32),
            "sample_weight": torch.tensor([1.0], dtype=torch.float32),
        }
        with mock.patch.object(aligned_grpo_module, "_distributed_world_size", return_value=2), mock.patch.object(
            aligned_grpo_module,
            "_distributed_sum_float",
            return_value=2.0,
        ), mock.patch.object(
            aligned_grpo_module,
            "_distributed_max_int",
            return_value=2,
        ):
            loss = trainer.training_step(model, {"episode_inputs": [batch], "runtime_stats": {}})

        self.assertEqual(trainer.accelerator.backward.call_count, 2)
        first_backward_loss = trainer.accelerator.backward.call_args_list[0].args[0]
        second_backward_loss = trainer.accelerator.backward.call_args_list[1].args[0]
        self.assertAlmostEqual(float(first_backward_loss.item()), 5.0, places=6)
        self.assertAlmostEqual(float(second_backward_loss.item()), 0.0, places=6)
        self.assertAlmostEqual(float(loss.item()), 5.0, places=6)

    def test_training_step_immediate_microbatch_backward_skips_inactive_microbatch_forward(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.accelerator = mock.Mock()
        trainer.accelerator.distributed_type = "DEEPSPEED"
        trainer.use_liger_loss = False
        trainer.compute_loss_context_manager = lambda: contextlib.nullcontext()
        trainer._prepare_inputs = lambda inputs: inputs
        trainer._maybe_skip_empty_training_step = mock.Mock(return_value=None)
        trainer._ensure_liger_runtime_ready = lambda model: None
        trainer._iter_loss_microbatches = lambda batch: [batch]
        trainer._compute_sample_losses_for_batch = mock.Mock(side_effect=AssertionError("inactive microbatch should bypass loss compute"))
        move_calls = []
        trainer._move_episode_input_to_device = lambda episode_input, device: move_calls.append((episode_input, device)) or episode_input
        trainer._episode_input_sample_count = TimesearchAlignedGRPOTrainerMixin._episode_input_sample_count.__get__(trainer)
        trainer._effective_sample_weight = TimesearchAlignedGRPOTrainerMixin._effective_sample_weight.__get__(trainer)
        trainer._sample_loss_multiplier = TimesearchAlignedGRPOTrainerMixin._sample_loss_multiplier.__get__(trainer)
        trainer._sample_weight = TimesearchAlignedGRPOTrainerMixin._sample_weight.__get__(trainer)
        trainer.args = type("Args", (), {"n_gpu": 1})()
        trainer.current_gradient_accumulation_steps = 1
        trainer.model_accepts_loss_kwargs = False
        trainer.compute_loss_func = None

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        model = TinyModel()
        batch = {
            "prompt_ids": torch.tensor([[1]], dtype=torch.long),
            "prompt_mask": torch.tensor([[1]], dtype=torch.long),
            "completion_ids": torch.tensor([[2]], dtype=torch.long),
            "completion_mask": torch.tensor([[1]], dtype=torch.bool),
            "sample_loss_multiplier": torch.tensor([0.0], dtype=torch.float32),
            "sample_weight": torch.tensor([0.0], dtype=torch.float32),
        }
        with mock.patch.object(aligned_grpo_module, "_distributed_world_size", return_value=1), mock.patch.object(
            aligned_grpo_module,
            "_distributed_sum_float",
            return_value=1.0,
        ), mock.patch.object(
            aligned_grpo_module,
            "_distributed_max_int",
            return_value=1,
        ):
            loss = trainer.training_step(model, {"episode_inputs": [batch], "runtime_stats": {}})

        trainer.accelerator.backward.assert_called_once()
        backward_loss = trainer.accelerator.backward.call_args.args[0]
        self.assertAlmostEqual(float(backward_loss.item()), 0.0, places=6)
        self.assertEqual(move_calls, [])
        trainer._compute_sample_losses_for_batch.assert_not_called()
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_compute_liger_loss_logs_liger_and_reference_forward(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.policy_temperature = None
        trainer.kl_beta = 0.01
        trainer.reference_model = object()
        trainer._episode_input_multimodal_inputs = lambda batch: {}
        trainer._prepare_advantages = lambda batch, device: torch.ones((1,), dtype=torch.float32, device=device)
        trainer.liger_grpo_loss = mock.Mock(return_value=(torch.tensor(3.0, dtype=torch.float32), None))

        class BackboneOutputs:
            last_hidden_state = torch.zeros((1, 2, 4), dtype=torch.float32)

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))
                self.lm_head = type("LMHead", (), {"weight": torch.ones((5, 4), dtype=torch.float32), "bias": None})()
                self.model = mock.Mock(return_value=BackboneOutputs())

            def forward(self, **kwargs):
                raise AssertionError("Liger path should use unwrapped_model.model directly")

        trainer.accelerator = mock.Mock()
        fake_model = FakeModel()
        trainer.accelerator.unwrap_model.return_value = fake_model
        batch = {
            "prompt_ids": torch.tensor([[1]], dtype=torch.long),
            "prompt_mask": torch.tensor([[1]], dtype=torch.long),
            "completion_ids": torch.tensor([[2]], dtype=torch.long),
            "completion_mask": torch.tensor([[1]], dtype=torch.bool),
            "sample_loss_multiplier": torch.tensor([1.0], dtype=torch.float32),
            "sample_weight": torch.tensor([1.0], dtype=torch.float32),
        }
        logged_messages = []
        with mock.patch.object(
            aligned_grpo_module,
            "compute_completion_only_token_log_probs_from_ids",
            return_value=(torch.zeros((1, 1), dtype=torch.float32), torch.ones((1, 1), dtype=torch.bool)),
        ), mock.patch.object(aligned_grpo_module, "runtime_log", side_effect=lambda message, **kwargs: logged_messages.append(str(message))):
            trainer._compute_liger_loss_for_batch(fake_model, batch)

        self.assertTrue(any("rl compute_loss liger forward start:" in message for message in logged_messages))
        self.assertTrue(any("rl compute_loss liger forward end:" in message for message in logged_messages))
        self.assertTrue(any("rl compute_loss reference kl forward start:" in message for message in logged_messages))
        self.assertTrue(any("rl compute_loss reference kl forward end:" in message for message in logged_messages))

    def test_compute_sample_losses_reuses_prepared_multimodal_inputs_for_policy_and_reference(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.policy_temperature = None
        trainer.kl_beta = 0.01
        trainer.reference_model = object()
        trainer.ppo_clip_epsilon = 0.2
        trainer._prepare_advantages = lambda batch, device: torch.ones((2,), dtype=torch.float32, device=device)
        trainer._episode_input_multimodal_inputs = mock.Mock(
            return_value={
                "pixel_values": torch.ones((2, 3), dtype=torch.float32),
                "image_grid_thw": torch.ones((2, 3), dtype=torch.long),
            }
        )

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        batch = {
            "prompt_ids": torch.tensor([[1, 2], [1, 2]], dtype=torch.long),
            "prompt_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.long),
            "completion_ids": torch.tensor([[3, 4], [3, 4]], dtype=torch.long),
            "completion_mask": torch.tensor([[1, 1], [1, 1]], dtype=torch.bool),
            "old_policy_token_log_probs": torch.zeros((2, 2), dtype=torch.float32),
            "sample_loss_multiplier": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "sample_weight": torch.tensor([1.0, 1.0], dtype=torch.float32),
        }
        policy_logps = torch.zeros((2, 2), dtype=torch.float32, requires_grad=True)
        reference_logps = torch.zeros((2, 2), dtype=torch.float32)
        response_mask = torch.ones((2, 2), dtype=torch.bool)
        with mock.patch.object(
            aligned_grpo_module,
            "compute_completion_only_token_log_probs_from_prepared_inputs",
            side_effect=[(policy_logps, response_mask), (reference_logps, response_mask)],
        ) as prepared_helper:
            sample_losses = trainer._compute_sample_losses_for_batch(model=TinyModel(), batch=batch)

        self.assertEqual(trainer._episode_input_multimodal_inputs.call_count, 1)
        self.assertEqual(prepared_helper.call_count, 2)
        first_call = prepared_helper.call_args_list[0].kwargs
        second_call = prepared_helper.call_args_list[1].kwargs
        self.assertIs(first_call["model_inputs"], second_call["model_inputs"])
        self.assertFalse(first_call["log_runtime_details"])
        self.assertFalse(second_call["log_runtime_details"])
        self.assertEqual(tuple(sample_losses.shape), (2,))

    def test_counterfactual_verification_batch_allows_null_policy_for_structured_oracle(self) -> None:
        result = run_counterfactual_verification_batch(
            None,
            batch_inputs=[
                {
                    "item": {"structured_target": {"existence": "normal"}},
                    "rollout": {"state": {}, "turns": []},
                    "reference_record": {"structured_target": {"existence": "normal"}},
                }
            ],
            branch_profile="structured_oracle_v1",
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["counterfactual_profile_source"], "structured_oracle_v1")

    def test_counterfactual_verification_batch_rejects_null_policy_for_non_oracle(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a non-null policy"):
            run_counterfactual_verification_batch(
                None,
                batch_inputs=[
                    {
                        "item": {},
                        "rollout": {},
                        "reference_record": {},
                    }
                ],
                branch_profile="full",
            )

    def test_generate_scored_rollouts_batch_skips_verification_policy_for_structured_oracle(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.num_generations = 1
        trainer.counterfactual_max_images = 12
        trainer.fecv_failure_policy = "degrade"
        trainer.rollout_stage_batch_size = 1
        trainer.fecv_stage_batch_size = 1
        trainer.advantage_clip = 3.0
        trainer._fecv_failure_count = 0
        trainer._apply_item_context_to_rollout = TimesearchAlignedGRPOTrainerMixin._apply_item_context_to_rollout.__get__(trainer)
        trainer._effective_local_rollout_batch_size = TimesearchAlignedGRPOTrainerMixin._effective_local_rollout_batch_size.__get__(trainer)
        trainer._build_rollout_policy = lambda model: "rollout-policy"
        trainer._build_fecv_policy = mock.Mock(side_effect=AssertionError("should not build verification policy"))
        trainer.rollout_runner = mock.Mock()
        trainer.rollout_runner.run_episodes.return_value = [
            {"video_id": "vid1", "state": {}, "turns": [], "_rl_episode_training_feature": {"episode_turn_samples": []}}
        ]
        trainer._assign_reward_summaries = lambda rollouts: list(rollouts)
        item = {
            "video_id": "vid1",
            "structured_target": {"existence": "normal"},
            "qa_pairs": [],
            "evidence": {"evidence_moments": []},
        }

        with mock.patch.object(
            aligned_grpo_module,
            "run_counterfactual_verification_batch",
            return_value=[{"counterfactual_profile_source": "structured_oracle_v1"}],
        ) as patched_fecv, mock.patch.object(
            aligned_grpo_module,
            "_compute_group_relative_advantages",
            side_effect=lambda rollouts, clip_value: list(rollouts),
        ):
            grouped = trainer._generate_scored_rollouts_batch([item], model=object(), progress=None)

        trainer._build_fecv_policy.assert_not_called()
        patched_fecv.assert_called_once()
        self.assertIsNone(patched_fecv.call_args.args[0])
        self.assertEqual(len(grouped), 1)

    def test_generate_scored_rollouts_batch_injects_proposal_runtime_for_strict_seek_evidence(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.num_generations = 1
        trainer.counterfactual_max_images = 12
        trainer.fecv_failure_policy = "degrade"
        trainer.rollout_stage_batch_size = 1
        trainer.fecv_stage_batch_size = 1
        trainer.advantage_clip = 3.0
        trainer._fecv_failure_count = 0
        trainer.strict_feature_guided_proposal = True
        trainer.proposal_runtime = "proposal-runtime"
        trainer._apply_item_context_to_rollout = TimesearchAlignedGRPOTrainerMixin._apply_item_context_to_rollout.__get__(trainer)
        trainer._effective_local_rollout_batch_size = TimesearchAlignedGRPOTrainerMixin._effective_local_rollout_batch_size.__get__(trainer)
        trainer._prepare_rollout_item = TimesearchAlignedGRPOTrainerMixin._prepare_rollout_item.__get__(trainer)
        trainer._build_rollout_policy = lambda model: "rollout-policy"
        trainer._build_fecv_policy = mock.Mock(side_effect=AssertionError("should not build verification policy"))
        trainer._assign_reward_summaries = lambda rollouts: list(rollouts)

        captured_chunk_items = []

        def _run_episodes(chunk_items, rollout_policy):
            captured_chunk_items.append(chunk_items)
            return [{"video_id": "vid1", "state": {}, "turns": [], "_rl_episode_training_feature": {"episode_turn_samples": []}}]

        trainer.rollout_runner = mock.Mock()
        trainer.rollout_runner.run_episodes.side_effect = _run_episodes
        item = {
            "video_id": "vid1",
            "structured_target": {"existence": "normal"},
            "qa_pairs": [],
            "evidence": {"evidence_moments": []},
            "multimodal_cache": {"embedding": {"embeddings": [1.0]}},
        }

        with mock.patch.object(
            aligned_grpo_module,
            "run_counterfactual_verification_batch",
            return_value=[{"counterfactual_profile_source": "structured_oracle_v1"}],
        ), mock.patch.object(
            aligned_grpo_module,
            "_compute_group_relative_advantages",
            side_effect=lambda rollouts, clip_value: list(rollouts),
        ):
            trainer._generate_scored_rollouts_batch([item], model=object(), progress=None)

        injected_cache = captured_chunk_items[0][0]["multimodal_cache"]
        self.assertTrue(injected_cache["strict_feature_guided_proposal"])
        self.assertEqual(injected_cache["proposal_runtime"], "proposal-runtime")

    def test_generate_scored_rollouts_batch_rejects_missing_proposal_runtime_for_strict_seek_evidence(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.strict_feature_guided_proposal = True
        trainer.proposal_runtime = None
        trainer._prepare_rollout_item = TimesearchAlignedGRPOTrainerMixin._prepare_rollout_item.__get__(trainer)

        item = {
            "video_id": "vid1",
            "multimodal_cache": {"embedding": {"embeddings": [1.0]}},
        }

        with self.assertRaisesRegex(ValueError, "requires proposal_runtime"):
            trainer._prepare_rollout_item(item)

    def test_generation_payload_calls_pack_builder_once_per_rollout(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer._new_rollout_metric_lists = TimesearchAlignedGRPOTrainerMixin._new_rollout_metric_lists.__get__(trainer)
        trainer._new_runtime_stats = TimesearchAlignedGRPOTrainerMixin._new_runtime_stats.__get__(trainer)
        trainer.min_weight = 0.0
        trainer._budgeting_stats = mock.Mock()
        trainer._budgeting_stats.record = mock.Mock()
        trainer._zero_response_dropped = 0
        trainer._build_episode_input_from_feature = mock.Mock(
            return_value=type("BuildResult", (), {"batch": {"prompt_ids": torch.tensor([[1]])}, "drop_reason": None})()
        )
        trainer._safe_float = None
        rollout = {
            "group_advantage": 1.0,
            "reward_summary": {"total_reward": 1.0, "components": {}},
            "turns": [],
            "_rl_episode_training_feature": {
                "episode_turn_samples": [
                    {
                        "episode_prompt_trace": {
                            "prompt_ids": torch.tensor([[1]], dtype=torch.long),
                            "prompt_mask": torch.tensor([[1]], dtype=torch.long),
                            "multimodal_inputs": {},
                        },
                        "completion_ids": torch.tensor([[2]], dtype=torch.long),
                        "completion_mask": torch.tensor([[1]], dtype=torch.bool),
                        "sample_weight": 1.0,
                    }
                ]
            },
        }

        with mock.patch.object(
            aligned_grpo_module,
            "_build_episode_tensor_packs_from_rollout",
            wraps=aligned_grpo_module._build_episode_tensor_packs_from_rollout,
        ) as patched_pack_builder:
            payload = trainer._build_generation_item_payload_from_rollouts({"video_id": "vid1"}, [rollout])

        self.assertEqual(patched_pack_builder.call_count, 1)
        self.assertEqual(len(payload["rollout_groups"]), 1)
        self.assertEqual(payload["rollout_groups"][0]["sample_count"], 1)
        self.assertEqual(len(payload["episode_inputs"]), 1)

    def test_build_generation_step_payloads_prefetches_old_logprobs(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer._generation_step_batch_size = 1
        trainer.steps_per_generation = 2
        trainer.args = type("Args", (), {"gradient_accumulation_steps": 1})()
        trainer.use_liger_loss = True
        trainer._groups_filtered_by_min_weight = 0
        trainer._groups_all_zero_advantage = 0
        trainer._aggregate_generation_step_payload = TimesearchAlignedGRPOTrainerMixin._aggregate_generation_step_payload.__get__(trainer)
        trainer.get_budget_drop_metrics = lambda: {}
        trainer._populate_old_policy_log_probs = mock.Mock(
            side_effect=lambda model, episode_inputs: [
                {**dict(episode_input), "old_policy_token_log_probs": torch.tensor([[0.1]], dtype=torch.float32)}
                for episode_input in episode_inputs
            ]
        )
        trainer._generate_scored_rollouts_batch = mock.Mock(return_value=[[{"video_id": "vid1"}]])
        trainer._build_generation_item_payload_from_rollouts = mock.Mock(
            return_value={
                "video_id": "vid1",
                "episode_inputs": [
                    {
                        "prompt_ids": torch.tensor([[1]], dtype=torch.long),
                        "prompt_mask": torch.tensor([[1]], dtype=torch.long),
                        "completion_ids": torch.tensor([[2]], dtype=torch.long),
                        "completion_mask": torch.tensor([[1]], dtype=torch.bool),
                        "advantages": torch.tensor([1.0], dtype=torch.float32),
                    }
                ],
                "rollout_metric_values": {},
                "runtime_stats": {},
            }
        )
        payloads = trainer._build_generation_step_payloads([{"video_id": "vid1"}], rollout_model=object())
        trainer._populate_old_policy_log_probs.assert_called_once()
        self.assertTrue(isinstance(payloads[0]["episode_inputs"][0]["old_policy_token_log_probs"], torch.Tensor))

    def test_build_generation_step_payloads_uses_sentinel_when_reuse_is_safe(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer._generation_step_batch_size = 1
        trainer.steps_per_generation = 1
        trainer.args = type("Args", (), {"gradient_accumulation_steps": 4})()
        trainer.use_liger_loss = True
        trainer._groups_filtered_by_min_weight = 0
        trainer._groups_all_zero_advantage = 0
        trainer._aggregate_generation_step_payload = TimesearchAlignedGRPOTrainerMixin._aggregate_generation_step_payload.__get__(trainer)
        trainer.get_budget_drop_metrics = lambda: {}
        trainer._populate_old_policy_log_probs = mock.Mock(side_effect=AssertionError("prefetch should be skipped when reuse is safe"))
        trainer._generate_scored_rollouts_batch = mock.Mock(return_value=[[{"video_id": "vid1"}]])
        trainer._build_generation_item_payload_from_rollouts = mock.Mock(
            return_value={
                "video_id": "vid1",
                "episode_inputs": [
                    {
                        "prompt_ids": torch.tensor([[1]], dtype=torch.long),
                        "prompt_mask": torch.tensor([[1]], dtype=torch.long),
                        "completion_ids": torch.tensor([[2]], dtype=torch.long),
                        "completion_mask": torch.tensor([[1]], dtype=torch.bool),
                        "advantages": torch.tensor([1.0], dtype=torch.float32),
                    }
                ],
                "rollout_metric_values": {},
                "runtime_stats": {},
            }
        )
        payloads = trainer._build_generation_step_payloads([{"video_id": "vid1"}], rollout_model=object())
        trainer._populate_old_policy_log_probs.assert_not_called()
        self.assertEqual(payloads[0]["episode_inputs"][0]["old_policy_token_log_probs"], _USE_CURRENT_POLICY_LOGPROBS_SENTINEL)

    def test_prepare_inputs_skips_old_logprob_population_for_zero_weight_batches(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.model = mock.Mock()
        trainer.model.training = True
        trainer.model.train = mock.Mock()
        trainer.model.eval = mock.Mock()
        trainer._native_grpo_progress = None
        trainer._active_generation_progress = None
        trainer._buffered_generation_step_payloads = []
        trainer._buffered_generation_batch_key = None
        trainer._build_generation_batch_key = lambda items: ("batch",)
        trainer._empty_generation_step_payload = lambda video_ids=None: {"episode_inputs": [], "runtime_stats": {}, "rollout_metrics": {}, "budgeting_metrics": {}, "video_ids": video_ids or []}
        trainer._move_episode_input_to_device = lambda episode_input, device: episode_input
        trainer._align_episode_inputs_across_ranks = lambda episode_inputs, device, runtime_stats: list(episode_inputs)
        trainer._materialize_episode_inputs = mock.Mock(side_effect=AssertionError("inactive batches should bypass materialization"))
        trainer._populate_old_policy_log_probs = mock.Mock(side_effect=AssertionError("inactive batches should bypass old logprobs"))
        trainer._has_trainable_weight = TimesearchAlignedGRPOTrainerMixin._has_trainable_weight.__get__(trainer)
        trainer._effective_sample_weight = TimesearchAlignedGRPOTrainerMixin._effective_sample_weight.__get__(trainer)
        trainer._sample_loss_multiplier = TimesearchAlignedGRPOTrainerMixin._sample_loss_multiplier.__get__(trainer)
        trainer._sample_weight = TimesearchAlignedGRPOTrainerMixin._sample_weight.__get__(trainer)
        trainer._episode_input_sample_count = TimesearchAlignedGRPOTrainerMixin._episode_input_sample_count.__get__(trainer)

        class TinyRolloutModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        rollout_model = TinyRolloutModel()
        with mock.patch.object(aligned_grpo_module, "_unwrap_model", return_value=rollout_model):
            trainer._pop_or_generate_generation_step_payload = mock.Mock(
                return_value={
                    "episode_inputs": [
                        {
                            "prompt_ids": torch.tensor([[1]], dtype=torch.long),
                            "prompt_mask": torch.tensor([[1]], dtype=torch.long),
                            "completion_ids": torch.tensor([[2]], dtype=torch.long),
                            "completion_mask": torch.tensor([[1]], dtype=torch.bool),
                            "sample_loss_multiplier": torch.tensor([0.0], dtype=torch.float32),
                            "sample_weight": torch.tensor([0.0], dtype=torch.float32),
                        }
                    ],
                    "runtime_stats": {},
                    "rollout_metrics": {},
                    "budgeting_metrics": {},
                    "video_ids": ["vid1"],
                }
            )
            prepared = trainer._prepare_inputs([{"video_id": "vid1"}])

        self.assertEqual(len(prepared["episode_inputs"]), 1)
        self.assertNotIn("old_policy_token_log_probs", prepared["episode_inputs"][0])
        trainer._populate_old_policy_log_probs.assert_not_called()
        trainer._materialize_episode_inputs.assert_not_called()

    def test_prepare_inputs_keeps_episode_batches_on_cpu_until_loss_microbatch(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.model = mock.Mock()
        trainer.model.training = True
        trainer.model.train = mock.Mock()
        trainer.model.eval = mock.Mock()
        trainer._native_grpo_progress = None
        trainer._active_generation_progress = None
        trainer._buffered_generation_step_payloads = []
        trainer._buffered_generation_batch_key = None
        trainer._build_generation_batch_key = lambda items: ("batch",)
        trainer._empty_generation_step_payload = lambda video_ids=None: {"episode_inputs": [], "runtime_stats": {}, "rollout_metrics": {}, "budgeting_metrics": {}, "video_ids": video_ids or []}
        trainer._align_episode_inputs_across_ranks = lambda episode_inputs, device, runtime_stats: list(episode_inputs)
        trainer._materialize_episode_inputs = mock.Mock(side_effect=lambda episode_inputs, device: list(episode_inputs))
        trainer._populate_old_policy_log_probs = mock.Mock(side_effect=lambda model, episode_inputs: list(episode_inputs))
        trainer._has_trainable_weight = TimesearchAlignedGRPOTrainerMixin._has_trainable_weight.__get__(trainer)
        trainer._effective_sample_weight = TimesearchAlignedGRPOTrainerMixin._effective_sample_weight.__get__(trainer)
        trainer._sample_loss_multiplier = TimesearchAlignedGRPOTrainerMixin._sample_loss_multiplier.__get__(trainer)
        trainer._sample_weight = TimesearchAlignedGRPOTrainerMixin._sample_weight.__get__(trainer)
        trainer._episode_input_sample_count = TimesearchAlignedGRPOTrainerMixin._episode_input_sample_count.__get__(trainer)

        move_calls = []

        def _record_move(episode_input, device):
            move_calls.append((dict(episode_input), device))
            return episode_input

        trainer._move_episode_input_to_device = _record_move

        class TinyRolloutModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        rollout_model = TinyRolloutModel()
        with mock.patch.object(aligned_grpo_module, "_unwrap_model", return_value=rollout_model):
            trainer._pop_or_generate_generation_step_payload = mock.Mock(
                return_value={
                    "episode_inputs": [
                        {
                            "prompt_ids": torch.tensor([[1]], dtype=torch.long),
                            "prompt_mask": torch.tensor([[1]], dtype=torch.long),
                            "completion_ids": torch.tensor([[2]], dtype=torch.long),
                            "completion_mask": torch.tensor([[1]], dtype=torch.bool),
                            "sample_loss_multiplier": torch.tensor([0.0], dtype=torch.float32),
                            "sample_weight": torch.tensor([0.0], dtype=torch.float32),
                        },
                        {
                            "prompt_ids": torch.tensor([[3]], dtype=torch.long),
                            "prompt_mask": torch.tensor([[1]], dtype=torch.long),
                            "completion_ids": torch.tensor([[4]], dtype=torch.long),
                            "completion_mask": torch.tensor([[1]], dtype=torch.bool),
                            "sample_loss_multiplier": torch.tensor([1.0], dtype=torch.float32),
                            "sample_weight": torch.tensor([1.0], dtype=torch.float32),
                        },
                    ],
                    "runtime_stats": {},
                    "rollout_metrics": {},
                    "budgeting_metrics": {},
                    "video_ids": ["vid1"],
                }
            )
            prepared = trainer._prepare_inputs([{"video_id": "vid1"}])

        self.assertEqual(len(prepared["episode_inputs"]), 2)
        self.assertEqual(len(move_calls), 0)

    def test_prepare_inputs_does_not_recompute_old_logprobs_for_active_batches(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.model = mock.Mock()
        trainer.model.training = True
        trainer.model.train = mock.Mock()
        trainer.model.eval = mock.Mock()
        trainer._native_grpo_progress = None
        trainer._active_generation_progress = None
        trainer._buffered_generation_step_payloads = []
        trainer._buffered_generation_batch_key = None
        trainer._build_generation_batch_key = lambda items: ("batch",)
        trainer._empty_generation_step_payload = lambda video_ids=None: {"episode_inputs": [], "runtime_stats": {}, "rollout_metrics": {}, "budgeting_metrics": {}, "video_ids": video_ids or []}
        trainer._align_episode_inputs_across_ranks = lambda episode_inputs, device, runtime_stats: list(episode_inputs)
        trainer._materialize_episode_inputs = mock.Mock(side_effect=lambda episode_inputs, device: list(episode_inputs))
        trainer._populate_old_policy_log_probs = mock.Mock(side_effect=AssertionError("should not be called from prepare_inputs"))
        trainer._has_trainable_weight = TimesearchAlignedGRPOTrainerMixin._has_trainable_weight.__get__(trainer)
        trainer._effective_sample_weight = TimesearchAlignedGRPOTrainerMixin._effective_sample_weight.__get__(trainer)
        trainer._sample_loss_multiplier = TimesearchAlignedGRPOTrainerMixin._sample_loss_multiplier.__get__(trainer)
        trainer._sample_weight = TimesearchAlignedGRPOTrainerMixin._sample_weight.__get__(trainer)
        trainer._episode_input_sample_count = TimesearchAlignedGRPOTrainerMixin._episode_input_sample_count.__get__(trainer)
        trainer._move_episode_input_to_device = lambda episode_input, device: episode_input

        class TinyRolloutModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.0]))

        rollout_model = TinyRolloutModel()
        with mock.patch.object(aligned_grpo_module, "_unwrap_model", return_value=rollout_model):
            trainer._pop_or_generate_generation_step_payload = mock.Mock(
                return_value={
                    "episode_inputs": [
                        {
                            "prompt_ids": torch.tensor([[3]], dtype=torch.long),
                            "prompt_mask": torch.tensor([[1]], dtype=torch.long),
                            "completion_ids": torch.tensor([[4]], dtype=torch.long),
                            "completion_mask": torch.tensor([[1]], dtype=torch.bool),
                            "sample_loss_multiplier": torch.tensor([1.0], dtype=torch.float32),
                            "sample_weight": torch.tensor([1.0], dtype=torch.float32),
                            "old_policy_token_log_probs": torch.tensor([[0.1]], dtype=torch.float32),
                        },
                    ],
                    "runtime_stats": {},
                    "rollout_metrics": {},
                    "budgeting_metrics": {},
                    "video_ids": ["vid1"],
                }
            )
            prepared = trainer._prepare_inputs([{"video_id": "vid1"}])

        self.assertEqual(len(prepared["episode_inputs"]), 1)
        trainer._populate_old_policy_log_probs.assert_not_called()

    def test_materialize_episode_inputs_merges_old_logprob_sentinel_batches(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.compute_loss_microbatch_size = 8
        trainer.processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_token_id=0, eos_token_id=0))
        trainer._materialize_fallback_batches = 0
        trainer._move_episode_input_to_device = lambda episode_input, device: dict(episode_input)
        trainer._group_items_by_signature = TimesearchAlignedGRPOTrainerMixin._group_items_by_signature.__get__(trainer)
        trainer._episode_input_merge_signature = TimesearchAlignedGRPOTrainerMixin._episode_input_merge_signature.__get__(trainer)
        trainer._episode_input_merge_signature_entry = (
            TimesearchAlignedGRPOTrainerMixin._episode_input_merge_signature_entry.__get__(trainer)
        )
        trainer._merge_episode_inputs = TimesearchAlignedGRPOTrainerMixin._merge_episode_inputs.__get__(trainer)
        trainer._sequence_pad_values = TimesearchAlignedGRPOTrainerMixin._sequence_pad_values.__get__(trainer)
        trainer._pad_token_id = TimesearchAlignedGRPOTrainerMixin._pad_token_id.__get__(trainer)
        trainer._pad_and_concat = TimesearchAlignedGRPOTrainerMixin._pad_and_concat.__get__(trainer)
        trainer._is_full_sequence_aligned_tensor = (
            TimesearchAlignedGRPOTrainerMixin._is_full_sequence_aligned_tensor.__get__(trainer)
        )
        trainer._pad_full_sequence_aligned_and_concat = (
            TimesearchAlignedGRPOTrainerMixin._pad_full_sequence_aligned_and_concat.__get__(trainer)
        )
        trainer._materialize_episode_inputs = TimesearchAlignedGRPOTrainerMixin._materialize_episode_inputs.__get__(
            trainer
        )
        trainer._is_merge_fallback_error = TimesearchAlignedGRPOTrainerMixin._is_merge_fallback_error.__get__(
            trainer
        )

        materialized = trainer._materialize_episode_inputs(
            [
                {
                    "prompt_ids": torch.tensor([[1]], dtype=torch.long),
                    "prompt_mask": torch.tensor([[1]], dtype=torch.long),
                    "completion_ids": torch.tensor([[2]], dtype=torch.long),
                    "completion_mask": torch.tensor([[1]], dtype=torch.bool),
                    "advantages": torch.tensor([1.0], dtype=torch.float32),
                    "old_policy_token_log_probs": _USE_CURRENT_POLICY_LOGPROBS_SENTINEL,
                },
                {
                    "prompt_ids": torch.tensor([[3]], dtype=torch.long),
                    "prompt_mask": torch.tensor([[1]], dtype=torch.long),
                    "completion_ids": torch.tensor([[4]], dtype=torch.long),
                    "completion_mask": torch.tensor([[1]], dtype=torch.bool),
                    "advantages": torch.tensor([0.5], dtype=torch.float32),
                    "old_policy_token_log_probs": _USE_CURRENT_POLICY_LOGPROBS_SENTINEL,
                },
            ],
            device=torch.device("cpu"),
        )

        self.assertEqual(len(materialized), 1)
        self.assertEqual(materialized[0]["old_policy_token_log_probs"], _USE_CURRENT_POLICY_LOGPROBS_SENTINEL)
        self.assertEqual(materialized[0]["prompt_ids"].shape[0], 2)
        self.assertEqual(materialized[0]["completion_ids"].shape[0], 2)

    def test_materialize_episode_inputs_merges_full_signature_bucket_before_loss_microbatching(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.compute_loss_microbatch_size = 2
        trainer.processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(pad_token_id=0, eos_token_id=0))
        trainer._materialize_fallback_batches = 0
        trainer._move_episode_input_to_device = lambda episode_input, device: dict(episode_input)
        trainer._group_items_by_signature = TimesearchAlignedGRPOTrainerMixin._group_items_by_signature.__get__(trainer)
        trainer._episode_input_merge_signature = TimesearchAlignedGRPOTrainerMixin._episode_input_merge_signature.__get__(trainer)
        trainer._episode_input_merge_signature_entry = (
            TimesearchAlignedGRPOTrainerMixin._episode_input_merge_signature_entry.__get__(trainer)
        )
        trainer._merge_episode_inputs = TimesearchAlignedGRPOTrainerMixin._merge_episode_inputs.__get__(trainer)
        trainer._sequence_pad_values = TimesearchAlignedGRPOTrainerMixin._sequence_pad_values.__get__(trainer)
        trainer._pad_token_id = TimesearchAlignedGRPOTrainerMixin._pad_token_id.__get__(trainer)
        trainer._pad_and_concat = TimesearchAlignedGRPOTrainerMixin._pad_and_concat.__get__(trainer)
        trainer._is_full_sequence_aligned_tensor = (
            TimesearchAlignedGRPOTrainerMixin._is_full_sequence_aligned_tensor.__get__(trainer)
        )
        trainer._pad_full_sequence_aligned_and_concat = (
            TimesearchAlignedGRPOTrainerMixin._pad_full_sequence_aligned_and_concat.__get__(trainer)
        )
        trainer._materialize_episode_inputs = TimesearchAlignedGRPOTrainerMixin._materialize_episode_inputs.__get__(
            trainer
        )
        trainer._is_merge_fallback_error = TimesearchAlignedGRPOTrainerMixin._is_merge_fallback_error.__get__(
            trainer
        )

        materialized = trainer._materialize_episode_inputs(
            [
                {
                    "prompt_ids": torch.tensor([[1]], dtype=torch.long),
                    "prompt_mask": torch.tensor([[1]], dtype=torch.long),
                    "completion_ids": torch.tensor([[2]], dtype=torch.long),
                    "completion_mask": torch.tensor([[1]], dtype=torch.bool),
                    "advantages": torch.tensor([0.1], dtype=torch.float32),
                    "old_policy_token_log_probs": torch.tensor([[0.1]], dtype=torch.float32),
                    "multimodal_inputs": {
                        "pixel_values": torch.tensor([[0.1, 0.2]], dtype=torch.float32),
                    },
                }
                for _ in range(5)
            ],
            device=torch.device("cpu"),
        )

        self.assertEqual(len(materialized), 1)
        self.assertEqual(materialized[0]["prompt_ids"].shape[0], 5)
        self.assertEqual(materialized[0]["completion_ids"].shape[0], 5)
        self.assertEqual(materialized[0]["advantages"].shape[0], 5)
        self.assertEqual(materialized[0]["old_policy_token_log_probs"].shape[0], 5)
        self.assertEqual(len(materialized[0]["multimodal_inputs"]), 5)

    def test_iter_loss_microbatches_slices_nested_multimodal_inputs(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer.compute_loss_microbatch_size = 2
        trainer._episode_input_sample_count = TimesearchAlignedGRPOTrainerMixin._episode_input_sample_count.__get__(trainer)
        trainer._slice_episode_input_sample_range = (
            TimesearchAlignedGRPOTrainerMixin._slice_episode_input_sample_range.__get__(trainer)
        )
        trainer._slice_multimodal_input_samples = (
            TimesearchAlignedGRPOTrainerMixin._slice_multimodal_input_samples.__get__(trainer)
        )
        trainer._iter_loss_microbatches = TimesearchAlignedGRPOTrainerMixin._iter_loss_microbatches.__get__(trainer)

        microbatches = trainer._iter_loss_microbatches(
            {
                "prompt_ids": torch.tensor([[1], [2], [3], [4]], dtype=torch.long),
                "prompt_mask": torch.tensor([[1], [1], [1], [1]], dtype=torch.long),
                "completion_ids": torch.tensor([[5], [6], [7], [8]], dtype=torch.long),
                "completion_mask": torch.tensor([[1], [1], [1], [1]], dtype=torch.bool),
                "advantages": torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
                "multimodal_inputs": [
                    {"pixel_values": torch.tensor([[0.1, 0.2]], dtype=torch.float32)}
                    for _ in range(4)
                ],
            }
        )

        self.assertEqual(len(microbatches), 2)
        self.assertEqual(microbatches[0]["prompt_ids"].shape[0], 2)
        self.assertEqual(len(microbatches[0]["multimodal_inputs"]), 2)
        self.assertEqual(microbatches[1]["prompt_ids"].shape[0], 2)
        self.assertEqual(len(microbatches[1]["multimodal_inputs"]), 2)

    def test_zero_advantage_payload_no_longer_replays_recent_payload(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer._recent_nonzero_advantage_payloads = []
        trainer._recent_nonzero_advantage_payload_capacity = 8
        trainer._zero_advantage_replay_uses = 0
        trainer._zero_advantage_replay_misses = 0
        trainer._move_episode_input_to_device = lambda episode_input, device: episode_input
        trainer._episode_input_cpu_copy = TimesearchAlignedGRPOTrainerMixin._episode_input_cpu_copy.__get__(trainer)
        trainer._payload_has_nonzero_advantage = TimesearchAlignedGRPOTrainerMixin._payload_has_nonzero_advantage.__get__(trainer)
        trainer._episode_input_has_nonzero_advantage = TimesearchAlignedGRPOTrainerMixin._episode_input_has_nonzero_advantage.__get__(trainer)
        trainer._clone_generation_step_payload_for_replay = TimesearchAlignedGRPOTrainerMixin._clone_generation_step_payload_for_replay.__get__(trainer)
        trainer._maybe_store_nonzero_advantage_payload = TimesearchAlignedGRPOTrainerMixin._maybe_store_nonzero_advantage_payload.__get__(trainer)
        trainer._maybe_replay_zero_advantage_payload = TimesearchAlignedGRPOTrainerMixin._maybe_replay_zero_advantage_payload.__get__(trainer)

        nonzero_payload = {
            "episode_inputs": [
                {
                    "prompt_ids": torch.tensor([[1]], dtype=torch.long),
                    "prompt_mask": torch.tensor([[1]], dtype=torch.long),
                    "completion_ids": torch.tensor([[2]], dtype=torch.long),
                    "completion_mask": torch.tensor([[1]], dtype=torch.bool),
                    "advantages": torch.tensor([0.5], dtype=torch.float32),
                }
            ],
            "runtime_stats": {},
            "rollout_metrics": {"reward_total": 1.0},
            "budgeting_metrics": {},
            "video_ids": ["cached_vid"],
        }
        trainer._maybe_store_nonzero_advantage_payload(nonzero_payload)

        zero_payload = {
            "episode_inputs": [
                {
                    "prompt_ids": torch.tensor([[3]], dtype=torch.long),
                    "prompt_mask": torch.tensor([[1]], dtype=torch.long),
                    "completion_ids": torch.tensor([[4]], dtype=torch.long),
                    "completion_mask": torch.tensor([[1]], dtype=torch.bool),
                    "advantages": torch.tensor([0.0], dtype=torch.float32),
                }
            ],
            "runtime_stats": {},
            "rollout_metrics": {"reward_total": 0.0},
            "budgeting_metrics": {},
            "video_ids": ["zero_vid"],
        }

        replayed = trainer._maybe_replay_zero_advantage_payload(zero_payload)

        self.assertIs(replayed, zero_payload)
        self.assertEqual(replayed["video_ids"], ["zero_vid"])
        self.assertEqual(float(replayed["episode_inputs"][0]["advantages"].item()), 0.0)
        self.assertEqual(trainer._zero_advantage_replay_uses, 0)
        self.assertEqual(trainer._zero_advantage_replay_misses, 1)

    def test_zero_advantage_payload_without_cache_returns_original_payload(self) -> None:
        trainer = object.__new__(TimesearchAlignedGRPOTrainerMixin)
        trainer._recent_nonzero_advantage_payloads = []
        trainer._recent_nonzero_advantage_payload_capacity = 8
        trainer._zero_advantage_replay_uses = 0
        trainer._zero_advantage_replay_misses = 0
        trainer._move_episode_input_to_device = lambda episode_input, device: episode_input
        trainer._episode_input_cpu_copy = TimesearchAlignedGRPOTrainerMixin._episode_input_cpu_copy.__get__(trainer)
        trainer._payload_has_nonzero_advantage = TimesearchAlignedGRPOTrainerMixin._payload_has_nonzero_advantage.__get__(trainer)
        trainer._episode_input_has_nonzero_advantage = TimesearchAlignedGRPOTrainerMixin._episode_input_has_nonzero_advantage.__get__(trainer)
        trainer._clone_generation_step_payload_for_replay = TimesearchAlignedGRPOTrainerMixin._clone_generation_step_payload_for_replay.__get__(trainer)
        trainer._maybe_replay_zero_advantage_payload = TimesearchAlignedGRPOTrainerMixin._maybe_replay_zero_advantage_payload.__get__(trainer)

        zero_payload = {
            "episode_inputs": [
                {
                    "prompt_ids": torch.tensor([[3]], dtype=torch.long),
                    "prompt_mask": torch.tensor([[1]], dtype=torch.long),
                    "completion_ids": torch.tensor([[4]], dtype=torch.long),
                    "completion_mask": torch.tensor([[1]], dtype=torch.bool),
                    "advantages": torch.tensor([0.0], dtype=torch.float32),
                }
            ],
            "runtime_stats": {},
            "rollout_metrics": {"reward_total": 0.0},
            "budgeting_metrics": {},
            "video_ids": ["zero_vid"],
        }

        replayed = trainer._maybe_replay_zero_advantage_payload(zero_payload)

        self.assertIs(replayed, zero_payload)
        self.assertEqual(trainer._zero_advantage_replay_uses, 0)
        self.assertEqual(trainer._zero_advantage_replay_misses, 1)

    def test_trl_runner_build_dataset_requires_materialized_runtime_cache_path(self) -> None:
        runner = object.__new__(TrlVllmGrpoRunner)
        runner.args = types.SimpleNamespace(
            materialized_train_items_path="",
            require_materialized_runtime_cache=True,
            include_splits="train",
            data="/tmp/train.jsonl",
            data_root="/tmp/data",
        )
        runner.config_builder = lambda args: {"dummy": True}
        with self.assertRaisesRegex(ValueError, "Active RL requires --materialized-train-items-path"):
            runner._build_dataset()

    def test_trl_runner_requires_proposal_model_path_when_seek_evidence_is_enabled(self) -> None:
        runner = object.__new__(TrlVllmGrpoRunner)
        runner.args = types.SimpleNamespace(
            proposal_model_path="",
            proposal_torch_dtype="auto",
            proposal_device="",
        )
        runner.runtime = types.SimpleNamespace()
        runner.raw_records = [
            {
                "tool_io": {
                    "allowed_tools": ["scan_timeline", "seek_evidence", "verify_hypothesis", "finalize_case"],
                }
            }
        ]

        with self.assertRaisesRegex(ValueError, "requires proposal_model_path"):
            runner._resolve_training_proposal_support()

    def test_trl_runner_loads_proposal_runtime_when_seek_evidence_is_enabled(self) -> None:
        runner = object.__new__(TrlVllmGrpoRunner)
        runner.args = types.SimpleNamespace(
            proposal_model_path="/models/siglip",
            proposal_torch_dtype="float16",
            proposal_device="cuda:3",
        )
        runner.runtime = types.SimpleNamespace()
        runner.raw_records = [
            {
                "tool_io": {
                    "allowed_tools": ["scan_timeline", "seek_evidence", "verify_hypothesis", "finalize_case"],
                }
            }
        ]

        with mock.patch.object(
            trl_grpo_module,
            "_load_training_proposal_runtime",
            return_value="proposal-runtime",
        ) as proposal_loader:
            strict, runtime = runner._resolve_training_proposal_support()

        self.assertTrue(strict)
        self.assertEqual(runtime, "proposal-runtime")
        proposal_loader.assert_called_once_with(
            proposal_model_path="/models/siglip",
            proposal_torch_dtype="float16",
            proposal_device="cuda:3",
            runtime=runner.runtime,
        )

    def test_mutable_iteration_dataset_replaces_items(self) -> None:
        dataset = MutableIterationDataset([{"video_id": "a"}, {"video_id": "b"}])
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]["video_id"], "a")
        dataset.replace_items([{"video_id": "c"}])
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0]["video_id"], "c")

    def test_continuous_rl_args_promotes_num_train_epochs_to_num_iterations(self) -> None:
        args = types.SimpleNamespace(num_iterations=7, num_train_epochs=1.0, use_liger_loss=True)
        continuous = _continuous_rl_args(args)
        self.assertEqual(float(continuous.num_train_epochs), 7.0)
        self.assertTrue(continuous.use_liger_loss)

    def test_vllm_policy_sampling_uses_single_batched_request(self) -> None:
        policy = object.__new__(VllmQwenGenerationPolicy)
        runtime = types.SimpleNamespace(
            enabled=True,
            args=types.SimpleNamespace(seed=7),
            runtime=types.SimpleNamespace(rank=0),
            build_sampling_params=mock.Mock(return_value={"seed": 7}),
            generate_completion_ids=mock.Mock(return_value=[[1, 2], [3, 4]]),
        )
        policy.vllm_runtime = runtime
        policy.source_model = None
        policy.step_resolver = lambda: 0
        policy.guided_decoding_regex = ""
        policy.remote_lora_request = None
        policy.capture_rl_token_traces = False
        policy._last_rl_token_traces = None
        policy.use_generation_cache = True
        policy.do_sample = True
        policy.temperature = 0.8
        policy.top_p = 0.95
        policy.top_k = 50
        policy.repetition_penalty = 1.02
        policy.max_new_tokens = 32
        policy.max_seq_length = 0
        policy.processor = types.SimpleNamespace(
            batch_decode=lambda ids, **kwargs: ['{"a":1}', '{"b":2}'],
        )
        policy.prepare_messages = lambda messages: messages
        policy._build_vllm_prompt_payload = lambda prepared_messages: (prepared_messages, "prompt")
        policy._extract_vision_inputs = lambda prepared_messages: ([], [])

        outputs = policy.generate_from_messages_batch([[{"role": "user", "content": []}], [{"role": "user", "content": []}]])

        self.assertEqual(outputs, ['{"a":1}', '{"b":2}'])
        runtime.generate_completion_ids.assert_called_once()
        request_inputs = runtime.generate_completion_ids.call_args.kwargs.get("request_inputs")
        if request_inputs is None:
            request_inputs = runtime.generate_completion_ids.call_args.args[0]
        self.assertEqual(len(request_inputs), 2)

    def test_rl_job_config_maps_external_launcher_aliases_to_colocate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "rl.yaml"
            model_config_path = tmp_path / "model.yaml"
            attention_config_path = tmp_path / "attn.yaml"
            rollout_config_path = tmp_path / "rollout.yaml"
            deepspeed_config_path = tmp_path / "zero3.json"
            deepspeed_config_path.write_text("{}\n", encoding="utf-8")

            _write_yaml(
                rollout_config_path,
                {
                    "engine": "vllm_local_rank",
                    "server": {
                        "launcher": "external_launcher",
                        "tensor_parallel_size": 2,
                        "gpu_memory_utilization": 0.82,
                    },
                    "client": {"guided_decoding_regex": "^ok$"},
                },
            )
            _write_yaml(
                model_config_path,
                {
                    "base_model": "/models/qwen3-vl-8b-Instruct",
                    "torch_dtype": "bfloat16",
                    "attn_implementation": "flash_attention_3",
                    "gradient_checkpointing": True,
                    "sequence": {"max_length": 4096},
                    "vision": {"max_images_per_sample": 12, "max_image_side": 720},
                },
            )
            _write_yaml(attention_config_path, {"policy_name": "fa3_only"})
            _write_yaml(
                config_path,
                {
                    "run_name": "rl-test",
                    "output_dir": str(tmp_path / "artifacts"),
                    "policy_init_from": "/checkpoints/sft/final_hf_model",
                    "rollout_backend": "vllm",
                    "rollout_config": str(rollout_config_path),
                    "data": {
                        "train_manifest": "/data/rl_train.jsonl",
                        "eval_manifest": "/data/rl_eval.jsonl",
                        "data_root": "/data/videos",
                        "eval_data_root": "/data/eval_videos",
                        "include_splits": "train",
                        "eval_include_splits": "val",
                    },
                    "optimization": {
                        "num_iterations": 3,
                        "num_train_epochs": 1.5,
                        "per_device_batch_size": 2,
                        "gradient_accumulation_steps": 4,
                        "min_weight": 0.0,
                        "advantage_clip": 2.5,
                        "ppo_clip_epsilon": 0.15,
                        "kl_beta": 0.02,
                        "rl_steps_per_generation": 4,
                        "rollout_stage_batch_size": 12,
                        "fecv_stage_batch_size": 10,
                        "rollout_count": 6,
                        "num_generations": 3,
                        "rollout_max_turns": 9,
                        "policy_do_sample": True,
                        "policy_temperature": 0.8,
                        "policy_top_p": 0.9,
                        "policy_top_k": 32,
                        "policy_repetition_penalty": 1.05,
                    },
                    "distributed": {"bf16": True, "fp16": False},
                    "rewards": {
                        "reward_version": "timesearch_v2",
                        "accuracy_reward_weight": 1.0,
                        "fecv_evidence_faithfulness_reward_weight": 0.5,
                    },
                },
            )

            job = RLJobConfig.from_files(
                config_path=str(config_path),
                model_config_path=str(model_config_path),
                attention_config_path=str(attention_config_path),
                deepspeed_config_path=str(deepspeed_config_path),
            )

            self.assertEqual(job.vllm_mode, "colocate")
            self.assertEqual(job.vllm_tensor_parallel_size, 2)
            self.assertEqual(job.min_weight, 0.0)
            self.assertEqual(job.advantage_clip, 2.5)
            self.assertEqual(job.ppo_clip_epsilon, 0.15)
            self.assertEqual(job.kl_beta, 0.02)
            self.assertEqual(job.rollout_stage_batch_size, 12)
            self.assertEqual(job.fecv_stage_batch_size, 10)
            self.assertTrue(job.policy_do_sample)
            self.assertEqual(job.policy_temperature, 0.8)
            self.assertEqual(job.policy_top_p, 0.9)
            self.assertEqual(job.policy_top_k, 32)
            self.assertEqual(job.policy_repetition_penalty, 1.05)
            self.assertEqual(job.vllm_guided_decoding_regex, "^ok$")
            self.assertEqual(job.deepspeed_config_path, str(deepspeed_config_path))
            self.assertEqual(job.reward_config["weights"]["accuracy_reward"], 1.0)
            self.assertEqual(job.reward_config["weights"]["fecv_evidence_faithfulness_reward"], 0.5)


    def test_rl_job_config_parses_stage_chunk_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "rl.yaml"
            model_config_path = tmp_path / "model.yaml"
            attention_config_path = tmp_path / "attn.yaml"
            rollout_config_path = tmp_path / "rollout.yaml"

            _write_yaml(rollout_config_path, {"server": {"launcher": "external_launcher"}})
            _write_yaml(model_config_path, {
                "base_model": "/models/qwen3-vl-8b-Instruct",
                "attn_implementation": "flash_attention_3",
            })
            _write_yaml(attention_config_path, {"policy_name": "fa3_only"})
            _write_yaml(config_path, {
                "policy_init_from": "/checkpoints/sft/final_hf_model",
                "rollout_backend": "vllm",
                "rollout_config": str(rollout_config_path),
                "data": {"train_manifest": "/data/rl_train.jsonl"},
                "optimization": {
                    "rollout_stage_batch_size": 6,
                    "fecv_stage_batch_size": 5,
                },
            })

            job = RLJobConfig.from_files(
                config_path=str(config_path),
                model_config_path=str(model_config_path),
                attention_config_path=str(attention_config_path),
            )
            self.assertEqual(job.rollout_stage_batch_size, 6)
            self.assertEqual(job.fecv_stage_batch_size, 5)

    def test_rl_job_config_defaults_keep_recent_tool_image_messages_to_three(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "rl.yaml"
            model_config_path = tmp_path / "model.yaml"
            attention_config_path = tmp_path / "attn.yaml"
            rollout_config_path = tmp_path / "rollout.yaml"

            _write_yaml(rollout_config_path, {"server": {"launcher": "external_launcher"}})
            _write_yaml(
                model_config_path,
                {
                    "base_model": "/models/qwen3-vl-8b-Instruct",
                    "attn_implementation": "flash_attention_3",
                },
            )
            _write_yaml(attention_config_path, {"policy_name": "fa3_only"})
            _write_yaml(
                config_path,
                {
                    "policy_init_from": "/checkpoints/sft/final_hf_model",
                    "rollout_backend": "vllm",
                    "rollout_config": str(rollout_config_path),
                    "data": {"train_manifest": "/data/rl_train.jsonl"},
                    "optimization": {},
                },
            )

            job = RLJobConfig.from_files(
                config_path=str(config_path),
                model_config_path=str(model_config_path),
                attention_config_path=str(attention_config_path),
            )

            self.assertEqual(job.keep_recent_tool_image_messages, 3)

    def test_rl_job_config_parses_vllm_max_num_seqs_controls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "rl.yaml"
            model_config_path = tmp_path / "model.yaml"
            attention_config_path = tmp_path / "attn.yaml"
            rollout_config_path = tmp_path / "rollout.yaml"

            _write_yaml(rollout_config_path, {"server": {"launcher": "external_launcher"}})
            _write_yaml(model_config_path, {
                "base_model": "/models/qwen3-vl-8b-Instruct",
                "attn_implementation": "flash_attention_3",
            })
            _write_yaml(attention_config_path, {"policy_name": "fa3_only"})
            _write_yaml(config_path, {
                "policy_init_from": "/checkpoints/sft/final_hf_model",
                "rollout_backend": "vllm",
                "rollout_config": str(rollout_config_path),
                "data": {"train_manifest": "/data/rl_train.jsonl"},
                "optimization": {
                    "vllm_max_num_seqs": 4,
                    "vllm_fallback_max_num_seqs": 2,
                },
            })

            job = RLJobConfig.from_files(
                config_path=str(config_path),
                model_config_path=str(model_config_path),
                attention_config_path=str(attention_config_path),
            )
            self.assertEqual(job.rl_steps_per_generation, 4)
            self.assertEqual(job.vllm_max_num_seqs, 4)
            self.assertEqual(job.vllm_fallback_max_num_seqs, 2)

    def test_rl_job_config_rejects_unknown_reward_weight_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "rl.yaml"
            model_config_path = tmp_path / "model.yaml"
            attention_config_path = tmp_path / "attn.yaml"
            rollout_config_path = tmp_path / "rollout.yaml"

            _write_yaml(rollout_config_path, {"server": {"launcher": "external_launcher"}})
            _write_yaml(
                model_config_path,
                {
                    "base_model": "/models/qwen3-vl-8b-Instruct",
                    "attn_implementation": "flash_attention_3",
                },
            )
            _write_yaml(attention_config_path, {"policy_name": "fa3_only"})
            _write_yaml(
                config_path,
                {
                    "policy_init_from": "/checkpoints/sft/final_hf_model",
                    "rollout_backend": "vllm",
                    "rollout_config": str(rollout_config_path),
                    "data": {"train_manifest": "/data/rl_train.jsonl"},
                    "rewards": {"anomaly_span_recall_weight": 1.0},
                },
            )

            with self.assertRaisesRegex(ValueError, "Unsupported RL reward weight keys"):
                RLJobConfig.from_files(
                    config_path=str(config_path),
                    model_config_path=str(model_config_path),
                    attention_config_path=str(attention_config_path),
                )

    def test_build_active_rl_trl_argv_includes_required_vllm_and_deepspeed_flags(self) -> None:
        job = RLJobConfig(
            run_name="rl-argv-test",
            output_dir="/tmp/out",
            train_manifest="/tmp/train.jsonl",
            eval_manifest="/tmp/eval.jsonl",
            data_root="/tmp/videos",
            eval_data_root="/tmp/eval_videos",
            include_splits="train",
            eval_include_splits="val",
            policy_init_from="/tmp/policy",
            reference_model="",
            base_model="/tmp/base",
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_3",
            gradient_checkpointing=True,
            rollout_backend="vllm",
            rollout_config="/tmp/rollout.yaml",
            deepspeed_config_path="/tmp/zero3.json",
            reward_config={"weights": {"temporal_miou_weight": 1.0}},
            num_iterations=7,
            num_train_epochs=1.0,
            learning_rate=5e-7,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            min_weight=0.0,
            advantage_clip=2.5,
            ppo_clip_epsilon=0.15,
            kl_beta=0.02,
            rl_steps_per_generation=4,
            rollout_count=8,
            num_generations=4,
            rollout_max_turns=12,
            policy_do_sample=True,
            policy_temperature=0.8,
            policy_top_p=0.9,
            policy_top_k=32,
            policy_repetition_penalty=1.05,
            vllm_mode="colocate",
            vllm_tensor_parallel_size=1,
            vllm_gpu_memory_utilization=0.9,
            vllm_max_num_seqs=4,
            vllm_fallback_max_num_seqs=2,
            vllm_guided_decoding_regex="^json$",
        )

        argv = build_active_rl_trl_argv(job)
        def _assert_flag_value(flag: str, expected: str) -> None:
            self.assertIn(flag, argv)
            index = argv.index(flag)
            self.assertLess(index + 1, len(argv))
            self.assertEqual(argv[index + 1], expected)

        _assert_flag_value("--model-path", "/tmp/policy")
        self.assertNotIn("--reference-model-path", argv)
        _assert_flag_value("--deepspeed", "/tmp/zero3.json")
        _assert_flag_value("--vllm-guided-decoding-regex", "^json$")
        _assert_flag_value("--min-weight", "0.0")
        _assert_flag_value("--advantage-clip", "2.5")
        _assert_flag_value("--ppo-clip-epsilon", "0.15")
        _assert_flag_value("--kl-beta", "0.02")
        _assert_flag_value("--policy-temperature", "0.8")
        _assert_flag_value("--policy-top-p", "0.9")
        _assert_flag_value("--policy-top-k", "32")
        _assert_flag_value("--policy-repetition-penalty", "1.05")
        _assert_flag_value("--rl-steps-per-generation", "4")
        _assert_flag_value("--vllm-max-num-seqs", "4")
        _assert_flag_value("--vllm-fallback-max-num-seqs", "2")
        reward_json = argv[argv.index("--rl-reward-config-json") + 1]
        self.assertEqual(json.loads(reward_json)["weights"]["temporal_miou_weight"], 1.0)
        self.assertIn("--gradient-checkpointing", argv)
        self.assertIn("--policy-do-sample", argv)
        self.assertIn("--bf16", argv)

    def test_build_active_rl_trl_argv_omits_open_ended_judge_flags(self) -> None:
        job = RLJobConfig(
            run_name="rl-argv-default-judge",
            output_dir="/tmp/out",
            train_manifest="/tmp/train.jsonl",
            eval_manifest=None,
            data_root="/tmp/videos",
            eval_data_root="/tmp/videos",
            include_splits="train",
            eval_include_splits="",
            policy_init_from="/tmp/policy",
            reference_model="/tmp/reference",
            base_model="/tmp/base",
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_3",
            gradient_checkpointing=False,
            rollout_backend="vllm",
            rollout_config="/tmp/rollout.yaml",
            deepspeed_config_path="/tmp/zero3.json",
            reward_config={"weights": {"temporal_miou_weight": 1.0}},
        )

        argv = build_active_rl_trl_argv(job)
        self.assertNotIn("--rl-open-ended-judge-enabled", argv)
        self.assertNotIn("--rl-open-ended-judge-base-url", argv)
        self.assertNotIn("--rl-open-ended-judge-model", argv)
        self.assertNotIn("--rl-open-ended-judge-cache-path", argv)
        self.assertNotIn("--rl-open-ended-judge-timeout-sec", argv)
        reward_json = argv[argv.index("--rl-reward-config-json") + 1]
        self.assertEqual(json.loads(reward_json)["weights"]["temporal_miou_weight"], 1.0)

    def test_rl_job_config_rejects_removed_open_ended_judge_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "rl.yaml"
            model_config_path = tmp_path / "model.yaml"
            attention_config_path = tmp_path / "attn.yaml"
            rollout_config_path = tmp_path / "rollout.yaml"

            _write_yaml(rollout_config_path, {"server": {"launcher": "external_launcher"}})
            _write_yaml(model_config_path, {
                "base_model": "/models/qwen3-vl-8b-Instruct",
                "attn_implementation": "flash_attention_3",
            })
            _write_yaml(attention_config_path, {"policy_name": "fa3_only"})
            _write_yaml(config_path, {
                "policy_init_from": "/checkpoints/sft/final_hf_model",
                "rollout_backend": "vllm",
                "rollout_config": str(rollout_config_path),
                "data": {"train_manifest": "/data/rl_train.jsonl"},
                "rewards": {
                    "open_ended_judge_enabled": True,
                },
            })

            with self.assertRaisesRegex(ValueError, "external LLM judging was removed from the RL reward path"):
                RLJobConfig.from_files(
                    config_path=str(config_path),
                    model_config_path=str(model_config_path),
                    attention_config_path=str(attention_config_path),
                )

    def test_rl_job_config_defaults_reference_model_to_empty_for_managed_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "rl.yaml"
            model_config_path = tmp_path / "model.yaml"
            attention_config_path = tmp_path / "attn.yaml"
            rollout_config_path = tmp_path / "rollout.yaml"

            _write_yaml(rollout_config_path, {"server": {"launcher": "external_launcher"}})
            _write_yaml(model_config_path, {
                "base_model": "/models/qwen3-vl-8b-Instruct",
                "attn_implementation": "flash_attention_3",
            })
            _write_yaml(attention_config_path, {"policy_name": "fa3_only"})
            _write_yaml(config_path, {
                "policy_init_from": "/checkpoints/sft/final_hf_model",
                "rollout_backend": "vllm",
                "rollout_config": str(rollout_config_path),
                "data": {"train_manifest": "/data/rl_train.jsonl"},
            })

            job = RLJobConfig.from_files(
                config_path=str(config_path),
                model_config_path=str(model_config_path),
                attention_config_path=str(attention_config_path),
            )
            self.assertEqual(job.reference_model, "")

    def test_rl_job_config_rejects_explicit_reference_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "rl.yaml"
            model_config_path = tmp_path / "model.yaml"
            attention_config_path = tmp_path / "attn.yaml"
            rollout_config_path = tmp_path / "rollout.yaml"

            _write_yaml(rollout_config_path, {"server": {"launcher": "external_launcher"}})
            _write_yaml(model_config_path, {
                "base_model": "/models/qwen3-vl-8b-Instruct",
                "attn_implementation": "flash_attention_3",
            })
            _write_yaml(attention_config_path, {"policy_name": "fa3_only"})
            _write_yaml(config_path, {
                "policy_init_from": "/checkpoints/sft/final_hf_model",
                "reference_model": "/models/reference",
                "rollout_backend": "vllm",
                "rollout_config": str(rollout_config_path),
                "data": {"train_manifest": "/data/rl_train.jsonl"},
            })

            with self.assertRaisesRegex(ValueError, "reference_model"):
                RLJobConfig.from_files(
                    config_path=str(config_path),
                    model_config_path=str(model_config_path),
                    attention_config_path=str(attention_config_path),
                )

    def test_select_iteration_indices_uses_seeded_permutation_chunks(self) -> None:
        first = cli_shared.select_iteration_indices(dataset_size=10, rollout_count=4, start_index=0, iteration=0, seed=42)
        second = cli_shared.select_iteration_indices(dataset_size=10, rollout_count=4, start_index=0, iteration=1, seed=42)
        replay = cli_shared.select_iteration_indices(dataset_size=10, rollout_count=4, start_index=0, iteration=0, seed=42)

        self.assertEqual(first, replay)
        self.assertNotEqual(first, [0, 1, 2, 3])
        self.assertEqual(len(first), 4)
        self.assertEqual(len(second), 4)
        self.assertTrue(set(first).isdisjoint(set(second)))

    def test_run_rl_job_writes_launch_manifest_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "artifacts"
        job = RLJobConfig(
            run_name="rl-run-test",
            output_dir=str(output_dir),
            train_manifest="/tmp/train.jsonl",
            eval_manifest=None,
            data_root="/tmp/videos",
            eval_data_root="/tmp/videos",
            include_splits="train",
            eval_include_splits="",
            policy_init_from="/tmp/policy",
            reference_model="",
            base_model="/tmp/base",
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_3",
            gradient_checkpointing=False,
            rollout_backend="vllm",
            rollout_config="/tmp/rollout.yaml",
            deepspeed_config_path="/tmp/zero3.json",
        )

        with mock.patch("saver_v3.rl.runtime.ensure_fa3_training_ready") as ensure_mock, mock.patch(
            "saver_v3.rl.runtime.legacy_train_saver_rl_trl.main",
            return_value={"latest_checkpoint": "/tmp/checkpoint-final"},
        ) as main_mock:
            result = run_rl_job(job)

            ensure_mock.assert_called_once_with(require_gpu=True)
            main_mock.assert_called_once()
            self.assertEqual(result.latest_checkpoint, "/tmp/checkpoint-final")
            self.assertTrue((output_dir / "rl_launch_manifest.json").exists())
            self.assertTrue((output_dir / "rl_summary.json").exists())

        payload = json.loads((output_dir / "rl_launch_manifest.json").read_text(encoding="utf-8"))
        argv = payload["argv"]
        argv_pairs = dict(zip(argv[::2], argv[1::2]))
        self.assertTrue(payload["episode_grpo_pure_pack"])
        self.assertNotIn("--reference-model-path", argv_pairs)
        self.assertEqual(argv_pairs["--vllm-max-num-seqs"], "4")
        self.assertEqual(argv_pairs["--vllm-fallback-max-num-seqs"], "2")

    def test_active_rl_shared_parser_parses_rl_reward_config_json(self) -> None:
        args = cli_shared.parse_active_rl_args(
            [
                "--output-dir",
                "/tmp/out",
                "--data",
                "/tmp/train.jsonl",
                "--model-path",
                "/tmp/policy",
                "--rl-reward-config-json",
                json.dumps({"weights": {"accuracy_reward": 1.25}}),
            ],
            description="test",
        )

        self.assertEqual(args.rl_reward_config["weights"]["accuracy_reward"], 1.25)

    def test_active_rl_shared_parser_rejects_lora_mode_in_v3(self) -> None:
        with self.assertRaisesRegex(ValueError, "full-model RL only"):
            cli_shared.parse_active_rl_args(
                [
                    "--output-dir",
                    "/tmp/out",
                    "--data",
                    "/tmp/train.jsonl",
                    "--model-path",
                    "/tmp/policy",
                    "--lora",
                ],
                description="test",
            )

    def test_active_rl_shared_parser_rejects_adapter_only_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter_dir = Path(tmp_dir) / "adapter_ckpt"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "adapter-only checkpoint"):
                cli_shared.parse_active_rl_args(
                    [
                        "--output-dir",
                        "/tmp/out",
                        "--data",
                        "/tmp/train.jsonl",
                        "--model-path",
                        str(adapter_dir),
                    ],
                    description="test",
                )

    def test_active_rl_shared_parser_rejects_removed_replay_buffer_flag(self) -> None:
        with self.assertRaisesRegex(SystemExit, "removed from active RL"):
            cli_shared.parse_active_rl_args(
                [
                    "--output-dir",
                    "/tmp/out",
                    "--data",
                    "/tmp/train.jsonl",
                    "--rl-replay-buffer-enable",
                    "true",
                ],
                description="test",
            )

    def test_active_rl_shared_parser_rejects_removed_open_ended_judge_flag(self) -> None:
        with self.assertRaises(SystemExit):
            cli_shared.parse_active_rl_args(
                [
                    "--output-dir",
                    "/tmp/out",
                    "--data",
                    "/tmp/train.jsonl",
                    "--rl-open-ended-judge-enabled",
                    "true",
                ],
                description="test",
            )

    def test_run_policy_inference_vllm_is_deprecated(self) -> None:
        from saver_v3.cli import run_policy_inference_vllm

        with mock.patch("sys.argv", ["run_policy_inference_vllm", "--config", "/tmp/unused.yaml"]):
            with self.assertRaisesRegex(SystemExit, "run_policy_rollout_vllm.py"):
                run_policy_inference_vllm.main()

    def test_train_saver_rl_trl_parser_rejects_removed_vllm_mode_flag(self) -> None:
        import train_saver_rl_trl

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaisesRegex(SystemExit, "removed"):
                train_saver_rl_trl.parse_args(
                    [
                        "--output-dir",
                        "/tmp/out",
                        "--data",
                        "/tmp/train.jsonl",
                        "--vllm-mode",
                        "server",
                    ]
                )

    def test_train_saver_rl_trl_parser_rejects_removed_vllm_server_flag(self) -> None:
        import train_saver_rl_trl

        with self.assertRaisesRegex(SystemExit, "removed"):
            train_saver_rl_trl.parse_args(
                [
                    "--output-dir",
                    "/tmp/out",
                    "--data",
                    "/tmp/train.jsonl",
                    "--vllm-server-host",
                    "127.0.0.1",
                ]
            )


    def test_train_saver_rl_trl_parser_accepts_vllm_max_num_seqs_controls(self) -> None:
        import train_saver_rl_trl

        args = train_saver_rl_trl.parse_args(
            [
                "--output-dir",
                "/tmp/out",
                "--data",
                "/tmp/train.jsonl",
                "--vllm-max-num-seqs",
                "4",
                "--vllm-fallback-max-num-seqs",
                "2",
            ]
        )
        self.assertEqual(args.vllm_max_num_seqs, 4)
        self.assertEqual(args.vllm_fallback_max_num_seqs, 2)

    def test_train_saver_rl_trl_parser_accepts_use_liger_loss(self) -> None:
        import train_saver_rl_trl

        args = train_saver_rl_trl.parse_args(
            [
                "--output-dir",
                "/tmp/out",
                "--data",
                "/tmp/train.jsonl",
                "--use-liger-loss",
                "true",
            ]
        )
        self.assertTrue(args.use_liger_loss)
        self.assertTrue(args.use_vllm)
        self.assertEqual(args.vllm_mode, "colocate")

    def test_train_saver_rl_trl_parser_defaults_keep_recent_tool_image_messages_to_three(self) -> None:
        import train_saver_rl_trl

        args = train_saver_rl_trl.parse_args(
            [
                "--output-dir",
                "/tmp/out",
                "--data",
                "/tmp/train.jsonl",
            ]
        )

        self.assertEqual(args.keep_recent_tool_image_messages, 3)

    def test_active_rl_shared_parser_rejects_reference_model_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference-model-path"):
            cli_shared.parse_active_rl_args(
                [
                    "--output-dir",
                    "/tmp/out",
                    "--data",
                    "/tmp/train.jsonl",
                    "--model-path",
                    "/tmp/policy",
                    "--reference-model-path",
                    "/tmp/reference",
                ],
                description="test",
            )

    def test_rl_job_config_parses_materialized_runtime_cache_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "rl.yaml"
            model_config_path = tmp_path / "model.yaml"
            attention_config_path = tmp_path / "attn.yaml"
            rollout_config_path = tmp_path / "rollout.yaml"

            _write_yaml(rollout_config_path, {"server": {"launcher": "external_launcher"}})
            _write_yaml(model_config_path, {
                "base_model": "/models/qwen3-vl-8b-Instruct",
                "attn_implementation": "flash_attention_3",
            })
            _write_yaml(attention_config_path, {"policy_name": "fa3_only"})
            _write_yaml(config_path, {
                "policy_init_from": "/checkpoints/sft/final_hf_model",
                "rollout_backend": "vllm",
                "rollout_config": str(rollout_config_path),
                "data": {
                    "train_manifest": "/data/rl_train.jsonl",
                    "eval_manifest": "/data/rl_eval.jsonl",
                    "materialized_train_items_path": "/data/rl_train.materialized.jsonl",
                    "materialized_eval_items_path": "/data/rl_eval.materialized.jsonl",
                    "require_materialized_runtime_cache": True,
                },
            })

            job = RLJobConfig.from_files(
                config_path=str(config_path),
                model_config_path=str(model_config_path),
                attention_config_path=str(attention_config_path),
            )
            self.assertEqual(job.materialized_train_items_path, "/data/rl_train.materialized.jsonl")
            self.assertEqual(job.materialized_eval_items_path, "/data/rl_eval.materialized.jsonl")
            self.assertTrue(job.require_materialized_runtime_cache)

    def test_build_active_rl_trl_argv_includes_materialized_runtime_cache_flags(self) -> None:
        job = RLJobConfig(
            run_name="rl-materialized",
            output_dir="/tmp/out",
            train_manifest="/tmp/train.jsonl",
            eval_manifest="/tmp/eval.jsonl",
            materialized_train_items_path="/tmp/train.materialized.jsonl",
            materialized_eval_items_path="/tmp/eval.materialized.jsonl",
            require_materialized_runtime_cache=True,
            data_root="/tmp/videos",
            eval_data_root="/tmp/eval_videos",
            include_splits="train",
            eval_include_splits="val",
            policy_init_from="/tmp/policy",
            reference_model="",
            base_model="/tmp/base",
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_3",
            gradient_checkpointing=True,
            rollout_backend="vllm",
            rollout_config="/tmp/rollout.yaml",
            deepspeed_config_path="/tmp/zero3.json",
        )
        argv = build_active_rl_trl_argv(job)
        argv_pairs = dict(zip(argv[::2], argv[1::2]))
        self.assertEqual(argv_pairs["--materialized-train-items-path"], "/tmp/train.materialized.jsonl")
        self.assertEqual(argv_pairs["--materialized-eval-items-path"], "/tmp/eval.materialized.jsonl")
        self.assertEqual(argv_pairs["--require-materialized-runtime-cache"], "true")

    def test_build_managed_reference_model_uses_prepare_deepspeed_for_zero3(self) -> None:
        trainer = types.SimpleNamespace(
            is_deepspeed_enabled=True,
            accelerator=object(),
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "zero3.json"
            config_path.write_text(json.dumps({"zero_optimization": {"stage": 3}}), encoding="utf-8")
            with mock.patch.object(
                aligned_grpo_module,
                "load_qwen_model_and_processor",
                return_value=("ref-model", None),
            ) as load_mock, mock.patch.object(
                aligned_grpo_module,
                "trl_prepare_deepspeed",
                return_value="managed-ref-model",
            ) as prepare_mock:
                reference_model, backend = _build_managed_reference_model_like_timesearch_r(
                    trainer=trainer,
                    model=object(),
                    trainer_init_model_path="/tmp/current",
                    torch_dtype="bfloat16",
                    attn_implementation="flash_attention_3",
                    kl_beta=0.01,
                    deepspeed=str(config_path),
                )
        load_mock.assert_called_once()
        prepare_mock.assert_called_once_with("ref-model", trainer.accelerator)
        self.assertEqual(reference_model, "managed-ref-model")
        self.assertEqual(backend, "deepspeed")

    def test_build_managed_reference_model_uses_accelerator_prepare_model_off_zero3(self) -> None:
        accelerator = mock.Mock()
        accelerator.prepare_model.return_value = "accelerate-ref-model"
        trainer = types.SimpleNamespace(
            is_deepspeed_enabled=False,
            accelerator=accelerator,
        )
        with mock.patch.object(
            aligned_grpo_module,
            "trl_create_reference_model",
            return_value="ref-model",
        ) as create_mock:
            reference_model, backend = _build_managed_reference_model_like_timesearch_r(
                trainer=trainer,
                model="policy-model",
                trainer_init_model_path="/tmp/current",
                torch_dtype="bfloat16",
                attn_implementation="flash_attention_3",
                kl_beta=0.01,
                deepspeed="",
            )
        create_mock.assert_called_once_with("policy-model")
        accelerator.prepare_model.assert_called_once_with("ref-model", evaluation_mode=True)
        self.assertEqual(reference_model, "accelerate-ref-model")
        self.assertEqual(backend, "accelerate")


if __name__ == "__main__":
    unittest.main()


class ProposalGpuPathTests(unittest.TestCase):
    def test_encode_query_text_entries_preserves_tensor_device(self) -> None:
        import torch
        from saver_v3.core.proposal import _encode_query_text_entries

        class FakeProposalRuntime:
            def __init__(self):
                self.device = "cpu"
            def encode_texts(self, texts):
                return torch.ones((len(texts), 4), dtype=torch.float32)

        entries = [{"text": "a", "weight": 1.0}, {"text": "b", "weight": 0.5}]
        encoded = _encode_query_text_entries(FakeProposalRuntime(), entries)
        self.assertEqual(len(encoded), 2)
        self.assertIsInstance(encoded[0]["embedding"], torch.Tensor)
        self.assertEqual(str(encoded[0]["embedding"].device), "cpu")

    def test_feature_guided_frame_proposal_returns_python_metadata(self) -> None:
        import torch
        from saver_v3.core.proposal import feature_guided_frame_proposal

        class FakeProposalRuntime:
            def __init__(self):
                self.device = "cpu"
            def encode_texts(self, texts):
                base = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
                return base.repeat(len(texts), 1)

        feature_cache = {
            "embeddings": torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.9, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.9, 0.1, 0.0],
            ], dtype=torch.float32),
            "normalized": False,
        }
        metadata = feature_guided_frame_proposal(
            feature_cache=feature_cache,
            proposal_runtime=FakeProposalRuntime(),
            query="event",
            query_package=None,
            role="trigger",
            search_anchor=None,
            start_sec=0.0,
            end_sec=3.0,
            fps=1.0,
            num_frames=2,
            top_k_candidates=2,
            candidate_merge_gap_sec=1.0,
            query_source="model",
        )
        self.assertIsInstance(metadata["selected_frame_indices"], list)
        self.assertTrue(all(isinstance(v, int) for v in metadata["selected_frame_indices"]))
        self.assertIsInstance(metadata["proposal_candidate_frame_scores"], list)
