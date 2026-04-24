from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from saver_v3.rl import grpo_trainer_env as env_mod
from saver_v3.sft import training as training_mod


def _make_old_policy_prefill_trainer(tmp_path):
    trainer = object.__new__(env_mod._NativeGRPOTrainerMixin)
    trainer.args = SimpleNamespace(output_dir=str(tmp_path), gradient_accumulation_steps=16)
    trainer.steps_per_generation = 1
    trainer.policy_temperature = None
    trainer.processor = SimpleNamespace(pad_token_id=0, eos_token_id=0)
    trainer._native_visual_tensor_dtype = None
    trainer._debug_last_compute_loss_call_index = 5
    trainer._debug_last_compute_loss_rank_local_batch_count = 1
    trainer._skipped_non_finite_old_policy_samples = 0
    trainer._skipped_non_finite_compute_samples = 0
    trainer._prepared_batch_merge_signature = lambda batch: ("unit_test_signature", tuple(sorted(batch.keys())))
    return trainer


def test_assert_finite_tensor_aborts_and_writes_diagnostic(tmp_path):
    trainer = object.__new__(env_mod._NativeGRPOTrainerMixin)
    trainer.args = SimpleNamespace(output_dir=str(tmp_path))
    trainer._debug_last_compute_loss_call_index = 5
    trainer._debug_last_compute_loss_rank_local_batch_count = 2

    with pytest.raises(RuntimeError, match="non-finite tensor"):
        trainer._assert_finite_tensor(
            stage="unit_test_non_finite",
            tensor_name="sample_losses",
            tensor_value=torch.tensor([1.0, float("nan")], dtype=torch.float32),
            batch=None,
        )

    dump_dir = tmp_path / "non_finite_dumps"
    dump_files = list(dump_dir.glob("*.json"))
    assert dump_files


def test_completion_only_helper_reports_non_finite_output_logits():
    class _FakeModel(torch.nn.Module):
        def forward(self, input_ids=None, attention_mask=None, logits_to_keep=None, **kwargs):
            del attention_mask, logits_to_keep, kwargs
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros((batch_size, seq_len, 8), dtype=torch.float32)
            logits[:, :, 0] = float("nan")
            return SimpleNamespace(logits=logits)

    with pytest.raises(RuntimeError, match="outputs\\.logits"):
        env_mod.compute_completion_only_token_log_probs_from_ids(
            model=_FakeModel(),
            prompt_ids=torch.tensor([[1, 2]], dtype=torch.long),
            prompt_mask=torch.ones((1, 2), dtype=torch.long),
            completion_ids=torch.tensor([[3, 4, 5]], dtype=torch.long),
            completion_mask=torch.ones((1, 3), dtype=torch.bool),
            multimodal_inputs=None,
        )


def test_completion_only_helper_reports_non_finite_log_probs(monkeypatch: pytest.MonkeyPatch):
    class _FakeModel(torch.nn.Module):
        def forward(self, input_ids=None, attention_mask=None, logits_to_keep=None, **kwargs):
            del attention_mask, logits_to_keep, kwargs
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros((batch_size, seq_len, 8), dtype=torch.float32)
            return SimpleNamespace(logits=logits)

    original_log_softmax = training_mod.F.log_softmax

    def _fake_log_softmax(*args, **kwargs):
        result = original_log_softmax(*args, **kwargs)
        result[..., 0] = float("nan")
        return result

    monkeypatch.setattr(training_mod.F, "log_softmax", _fake_log_softmax)

    with pytest.raises(RuntimeError, match="log_probs"):
        env_mod.compute_completion_only_token_log_probs_from_ids(
            model=_FakeModel(),
            prompt_ids=torch.tensor([[1, 2]], dtype=torch.long),
            prompt_mask=torch.ones((1, 2), dtype=torch.long),
            completion_ids=torch.tensor([[3, 4, 5]], dtype=torch.long),
            completion_mask=torch.ones((1, 3), dtype=torch.bool),
            multimodal_inputs=None,
        )


def test_current_policy_reuse_defaults_to_enabled_for_single_step_generation():
    trainer = object.__new__(env_mod._NativeGRPOTrainerMixin)
    trainer.args = SimpleNamespace(gradient_accumulation_steps=16)
    trainer.steps_per_generation = 1
    assert trainer._can_reuse_current_policy_as_old_logprobs() is True

    trainer.allow_reuse_current_policy_as_old_logprobs = False
    assert trainer._can_reuse_current_policy_as_old_logprobs() is False

    trainer.allow_reuse_current_policy_as_old_logprobs = True
    assert trainer._can_reuse_current_policy_as_old_logprobs() is True


def test_current_policy_reuse_requires_explicit_opt_in_for_multi_step_generation():
    trainer = object.__new__(env_mod._NativeGRPOTrainerMixin)
    trainer.args = SimpleNamespace(gradient_accumulation_steps=16)
    trainer.steps_per_generation = 2
    assert trainer._can_reuse_current_policy_as_old_logprobs() is False

    trainer.allow_reuse_current_policy_as_old_logprobs = True
    assert trainer._can_reuse_current_policy_as_old_logprobs() is True


def test_compute_sample_losses_requires_cached_old_policy_when_reuse_disabled(tmp_path, monkeypatch: pytest.MonkeyPatch):
    trainer = object.__new__(env_mod._NativeGRPOTrainerMixin)
    trainer.args = SimpleNamespace(output_dir=str(tmp_path), gradient_accumulation_steps=16)
    trainer.steps_per_generation = 1
    trainer.allow_reuse_current_policy_as_old_logprobs = False
    trainer.policy_temperature = None
    trainer.ppo_clip_epsilon = 0.2
    trainer.kl_beta = 0.0
    trainer.reference_model = None
    trainer.use_lora_reference_disable_adapter = False
    trainer._debug_last_compute_loss_call_index = 1
    trainer._debug_last_compute_loss_rank_local_batch_count = 1

    def _fake_log_probs(*args, **kwargs):
        del args, kwargs
        policy = torch.zeros((1, 3), dtype=torch.float32, requires_grad=True)
        response_mask = torch.ones((1, 3), dtype=torch.bool)
        return policy, response_mask

    monkeypatch.setattr(env_mod, "compute_completion_only_token_log_probs_from_ids", _fake_log_probs)

    batch = {
        "prompt_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "prompt_mask": torch.ones((1, 2), dtype=torch.long),
        "completion_ids": torch.tensor([[3, 4, 5]], dtype=torch.long),
        "completion_mask": torch.ones((1, 3), dtype=torch.bool),
        "advantage": torch.tensor([1.0], dtype=torch.float32),
        "sample_loss_multiplier": torch.tensor([1.0], dtype=torch.float32),
        "token_loss_weight": torch.ones((1, 3), dtype=torch.float32),
    }

    with pytest.raises(RuntimeError, match="cached per-token old_policy_token_log_probs"):
        trainer._compute_sample_losses_for_batch(model=object(), batch=batch)


def test_old_policy_prefill_dumps_non_finite_visual_inputs(tmp_path):
    trainer = _make_old_policy_prefill_trainer(tmp_path)
    prepared_batch = {
        "prompt_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "prompt_mask": torch.ones((1, 2), dtype=torch.long),
        "completion_ids": torch.tensor([[3, 4, 5]], dtype=torch.long),
        "completion_mask": torch.ones((1, 3), dtype=torch.bool),
        "pixel_values": torch.tensor([[float("nan"), 1.0]], dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.long),
    }
    episode_entries = [
        {
            "video_id": "vid-a",
            "group_id": "group-a",
            "generation_id": 2,
            "sample_partition": "anomaly",
            "sample_partition_type": "anomaly",
            "advantage_source": "group_relative",
            "episode_spec": {
                "prompt_ids": prepared_batch["prompt_ids"],
                "completion_ids": prepared_batch["completion_ids"],
            },
            "episode_debug_metadata": {
                "message_plan": [{"message_index": 1, "content_indices": [0, 1]}],
                "assistant_supervision": [{"assistant_message_index": 2}],
                "retained_image_provenance": [{"message_index": 1, "content_index": 1, "provenance": {"video_path": "/tmp/a.mp4"}}],
                "retained_message_count": 2,
            },
        }
    ]

    with pytest.raises(RuntimeError, match="old-policy prefill received a non-finite multimodal tensor"):
        trainer._compute_old_policy_token_log_probs_for_prepared_batch(
            model=object(),
            prepared_batch=prepared_batch,
            episode_entries=episode_entries,
        )

    dump_files = sorted((tmp_path / "non_finite_dumps").glob("*.json"))
    assert dump_files
    payload = json.loads(dump_files[-1].read_text())
    assert payload["stage"] == "old_policy_prefill_inputs_non_finite"
    assert payload["tensor_name"] == "pixel_values"
    assert payload["model_input_summaries"]["pixel_values"]["nan_count"] == 1
    assert payload["episode_entries"][0]["video_id"] == "vid-a"
    assert payload["episode_entries"][0]["generation_id"] == 2
    assert payload["episode_entries"][0]["message_plan"] == [{"message_index": 1, "content_indices": [0, 1]}]
    assert payload["episode_entries"][0]["retained_image_provenance"] == [
        {"message_index": 1, "content_index": 1, "provenance": {"video_path": "/tmp/a.mp4"}}
    ]


def test_old_policy_prefill_dumps_context_on_forward_exception(tmp_path):
    trainer = _make_old_policy_prefill_trainer(tmp_path)

    class _NaNLogitModel(torch.nn.Module):
        def forward(self, input_ids=None, attention_mask=None, logits_to_keep=None, **kwargs):
            del attention_mask, kwargs
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros((batch_size, int(logits_to_keep), 8), dtype=torch.float32)
            logits[:, :, 0] = float("nan")
            return SimpleNamespace(logits=logits)

    prepared_batch = {
        "prompt_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "prompt_mask": torch.ones((1, 2), dtype=torch.long),
        "completion_ids": torch.tensor([[3, 4, 5]], dtype=torch.long),
        "completion_mask": torch.ones((1, 3), dtype=torch.bool),
        "pixel_values": torch.ones((2, 2), dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.long),
    }
    episode_entries = [
        {
            "video_id": "vid-b",
            "group_id": "group-b",
            "generation_id": 7,
            "sample_partition": "anomaly",
            "sample_partition_type": "anomaly",
            "advantage_source": "group_relative",
            "episode_spec": {
                "prompt_ids": prepared_batch["prompt_ids"],
                "completion_ids": prepared_batch["completion_ids"],
            },
            "episode_debug_metadata": {
                "message_plan": [{"message_index": 4, "content_indices": [0]}],
                "assistant_supervision": [{"assistant_message_index": 5}],
                "retained_image_provenance": [{"message_index": 4, "content_index": 0, "provenance": {"video_path": "/tmp/b.mp4"}}],
                "retained_message_count": 1,
            },
        }
    ]

    with pytest.raises(RuntimeError, match="outputs\\.logits"):
        trainer._compute_old_policy_token_log_probs_for_prepared_batch(
            model=_NaNLogitModel(),
            prepared_batch=prepared_batch,
            episode_entries=episode_entries,
        )

    dump_files = sorted((tmp_path / "non_finite_dumps").glob("*.json"))
    assert dump_files
    payload = json.loads(dump_files[-1].read_text())
    assert payload["stage"] == "old_policy_prefill_forward_exception"
    assert payload["error"]["type"] == "RuntimeError"
    assert payload["model_input_summaries"]["pixel_values"]["all_finite"] is True
    assert payload["episode_entries"][0]["video_id"] == "vid-b"
    assert payload["episode_entries"][0]["generation_id"] == 7
    assert payload["episode_entries"][0]["message_plan"] == [{"message_index": 4, "content_indices": [0]}]


def test_old_policy_prefill_fail_fast_on_bad_sample(tmp_path):
    trainer = _make_old_policy_prefill_trainer(tmp_path)

    class _SelectiveNaNModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, input_ids=None, attention_mask=None, logits_to_keep=None, **kwargs):
            del attention_mask, kwargs
            batch_size = int(input_ids.shape[0])
            logits = torch.zeros((batch_size, int(logits_to_keep), 8), dtype=torch.float32)
            if batch_size > 1 or bool(torch.any(input_ids[:, 0] == 99)):
                logits[:, :, 0] = float("nan")
            return SimpleNamespace(logits=logits + self.anchor.reshape(1, 1, 1) * 0.0)

    def _entry(first_prompt_token: int, generation_id: int) -> dict:
        return {
            "video_id": "vid-c",
            "group_id": "group-c",
            "generation_id": generation_id,
            "sample_partition": "anomaly",
            "sample_partition_type": "anomaly",
            "advantage_source": "group_relative",
            "episode_spec": {
                "prompt_ids": torch.tensor([[first_prompt_token, 2]], dtype=torch.long),
                "prompt_mask": torch.ones((1, 2), dtype=torch.long),
                "completion_ids": torch.tensor([[3, 4, 5]], dtype=torch.long),
                "completion_mask": torch.ones((1, 3), dtype=torch.bool),
                "sample_loss_multiplier": torch.tensor([1.0], dtype=torch.float32),
                "advantage": torch.tensor([1.0], dtype=torch.float32),
                "token_loss_weight": torch.ones((1, 3), dtype=torch.float32),
                "pixel_values": torch.ones((2, 2), dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.long),
            },
            "episode_debug_metadata": {
                "message_plan": [{"message_index": generation_id, "content_indices": [0]}],
                "assistant_supervision": [{"assistant_message_index": generation_id + 1}],
                "retained_image_provenance": [],
                "retained_message_count": 1,
            },
        }

    bad_entry = _entry(99, 1)
    good_entry = _entry(1, 2)
    with pytest.raises(RuntimeError, match="non-finite tensor: outputs.logits"):
        trainer._populate_old_policy_log_probs(
            _SelectiveNaNModel(),
            [bad_entry, good_entry],
        )

    assert trainer._skipped_non_finite_old_policy_samples == 0
    dump_files = sorted((tmp_path / "non_finite_dumps").glob("*old_policy_prefill_skipped_sample*.json"))
    assert not dump_files


def test_recover_sample_losses_from_source_entries_fail_fast_on_bad_sample(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    trainer = _make_old_policy_prefill_trainer(tmp_path)

    def _entry(first_prompt_token: int, generation_id: int) -> dict:
        return {
            "video_id": "vid-d",
            "group_id": "group-d",
            "generation_id": generation_id,
            "sample_partition": "anomaly",
            "sample_partition_type": "anomaly",
            "advantage_source": "group_relative",
            "episode_spec": {
                "prompt_ids": torch.tensor([[first_prompt_token, 2]], dtype=torch.long),
                "prompt_mask": torch.ones((1, 2), dtype=torch.long),
                "completion_ids": torch.tensor([[3, 4, 5]], dtype=torch.long),
                "completion_mask": torch.ones((1, 3), dtype=torch.bool),
                "sample_loss_multiplier": torch.tensor([1.0], dtype=torch.float32),
                "advantage": torch.tensor([1.0], dtype=torch.float32),
                "old_policy_token_log_probs": torch.zeros((1, 3), dtype=torch.float32),
                "token_loss_weight": torch.ones((1, 3), dtype=torch.float32),
                "pixel_values": torch.ones((2, 2), dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.long),
            },
            "episode_debug_metadata": {
                "message_plan": [{"message_index": generation_id, "content_indices": [0]}],
                "assistant_supervision": [{"assistant_message_index": generation_id + 1}],
                "retained_image_provenance": [],
                "retained_message_count": 1,
            },
        }

    def _fake_compute_sample_losses_for_batch(self, *, model, batch):
        del model
        first_prompt_token = int(batch["prompt_ids"][0, 0].item())
        if first_prompt_token == 99:
            raise RuntimeError("completion-only forward produced a non-finite tensor: outputs.logits")
        return torch.tensor([0.25], dtype=torch.float32)

    monkeypatch.setattr(
        env_mod._NativeGRPOTrainerMixin,
        "_compute_sample_losses_for_batch",
        _fake_compute_sample_losses_for_batch,
    )

    class _DeviceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

    source_entries = [_entry(99, 1), _entry(1, 2)]
    batch = {
        "prompt_ids": torch.tensor([[99, 2], [1, 2]], dtype=torch.long),
        "prompt_mask": torch.ones((2, 2), dtype=torch.long),
        "completion_ids": torch.tensor([[3, 4, 5], [3, 4, 5]], dtype=torch.long),
        "completion_mask": torch.ones((2, 3), dtype=torch.bool),
        "token_loss_weight": torch.ones((2, 3), dtype=torch.float32),
        "_source_episode_entries": source_entries,
        "pixel_values": torch.ones((4, 2), dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, 1, 2], [1, 1, 2]], dtype=torch.long),
    }

    with pytest.raises(RuntimeError, match="non-finite tensor: outputs.logits"):
        trainer._recover_sample_losses_from_source_entries(
            model=_DeviceModel(),
            batch=batch,
            batch_error=RuntimeError("completion-only forward produced a non-finite tensor: outputs.logits"),
        )

    assert trainer._skipped_non_finite_compute_samples == 0
    dump_files = sorted((tmp_path / "non_finite_dumps").glob("*compute_loss_skipped_sample*.json"))
    assert not dump_files
