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
    trainer.state = SimpleNamespace(global_step=1, epoch=1.0)
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


def test_assert_finite_trainable_parameters_aborts_and_writes_diagnostic(tmp_path):
    trainer = _make_old_policy_prefill_trainer(tmp_path)

    class _BadParamModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.good = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))
            self.bad = torch.nn.Parameter(torch.tensor([float("nan")], dtype=torch.float32))

    with pytest.raises(RuntimeError, match="non-finite trainable tensors"):
        trainer._assert_finite_trainable_parameters(
            stage="unit_test_params",
            model=_BadParamModel(),
        )

    dump_files = sorted((tmp_path / "non_finite_dumps").glob("*unit_test_params*.json"))
    assert dump_files
    payload = json.loads(dump_files[-1].read_text())
    assert payload["tensor_kind"] == "params"
    assert payload["scan"]["non_finite_count"] == 1
    assert payload["scan"]["entries"][0]["name"] == "bad"
    assert payload["scan"]["entries"][0]["nan_count"] == 1


def test_assert_finite_trainable_gradients_aborts_and_writes_diagnostic(tmp_path):
    trainer = _make_old_policy_prefill_trainer(tmp_path)
    model = torch.nn.Linear(2, 1)
    model.weight.grad = torch.zeros_like(model.weight)
    model.bias.grad = torch.full_like(model.bias, float("nan"))

    with pytest.raises(RuntimeError, match="non-finite trainable tensors"):
        trainer._assert_finite_trainable_gradients(
            stage="unit_test_grads",
            model=model,
        )

    dump_files = sorted((tmp_path / "non_finite_dumps").glob("*unit_test_grads*.json"))
    assert dump_files
    payload = json.loads(dump_files[-1].read_text())
    assert payload["tensor_kind"] == "grads"
    assert payload["scan"]["non_finite_count"] == 1
    assert payload["scan"]["entries"][0]["name"].endswith("bias")
    assert payload["scan"]["entries"][0]["nan_count"] == 1


def test_optimizer_step_proxy_dumps_flat_master_forensics_when_post_step_non_finite(tmp_path):
    trainer = _make_old_policy_prefill_trainer(tmp_path)
    trainer.accelerator = SimpleNamespace()
    trainer._native_rl_skip_next_optimizer_step = False
    trainer._native_rl_last_skip_reason = ""
    trainer._optimizer_step_skips = 0
    trainer._effective_update_steps = 0

    flat_param = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))

    class _FakeOptimizer:
        def __init__(self):
            self.single_partition_of_fp32_groups = [flat_param]
            self.bit16_groups_flat = [torch.zeros(4, dtype=torch.bfloat16)]
            self.averaged_gradients = {0: torch.zeros(4, dtype=torch.float32)}
            self.state = {
                flat_param: {
                    "exp_avg": torch.zeros(4, dtype=torch.float32),
                    "exp_avg_sq": torch.zeros(4, dtype=torch.float32),
                }
            }

        def step(self, *args, **kwargs):
            del args, kwargs
            flat_param.data.fill_(float("nan"))
            self.state[flat_param]["exp_avg_sq"].fill_(float("nan"))
            return "stepped"

    proxy = env_mod._NativeRLOptimizerStepProxy(_FakeOptimizer(), trainer=trainer)

    assert proxy.step() == "stepped"

    dump_files = sorted((tmp_path / "non_finite_dumps").glob("*post_optimizer_step_flat_master_state*.json"))
    assert dump_files
    payload = json.loads(dump_files[-1].read_text())
    assert payload["optimizer_class"] == "_FakeOptimizer"
    assert payload["flat_master_scan"]["fp32_partitions"]["non_finite_count"] == 1
    assert payload["flat_master_scan"]["optimizer_state"]["non_finite_count"] == 1
    state_entry_names = [entry["name"] for entry in payload["flat_master_scan"]["optimizer_state"]["entries"]]
    assert any(name.endswith("exp_avg_sq") for name in state_entry_names)


def test_trainer_dumps_deepspeed_flat_master_forensics(tmp_path):
    trainer = _make_old_policy_prefill_trainer(tmp_path)
    trainer.accelerator = SimpleNamespace()

    flat_param = torch.nn.Parameter(torch.full((4,), float("nan"), dtype=torch.float32))

    class _FakeOptimizer:
        def __init__(self):
            self.single_partition_of_fp32_groups = [flat_param]
            self.bit16_groups_flat = [torch.zeros(4, dtype=torch.bfloat16)]
            self.averaged_gradients = {0: torch.zeros(4, dtype=torch.float32)}
            self.state = {
                flat_param: {
                    "exp_avg": torch.zeros(4, dtype=torch.float32),
                    "exp_avg_sq": torch.full((4,), float("nan"), dtype=torch.float32),
                }
            }

    trainer.deepspeed = SimpleNamespace(optimizer=_FakeOptimizer())

    dump_path = trainer._write_deepspeed_flat_master_forensics_dump(
        stage="unit_test_deepspeed_post_step",
        extra={"marker": "unit"},
    )

    assert dump_path is not None
    payload = json.loads(dump_path.read_text())
    assert payload["optimizer_class"] == "_FakeOptimizer"
    assert payload["flat_master_scan"]["fp32_partitions"]["non_finite_count"] == 1
    assert payload["flat_master_scan"]["optimizer_state"]["non_finite_count"] == 1
    assert payload["extra"]["marker"] == "unit"


def test_completion_only_helper_allows_non_finite_output_logits():
    class _FakeModel(torch.nn.Module):
        def forward(self, input_ids=None, attention_mask=None, logits_to_keep=None, **kwargs):
            del attention_mask, logits_to_keep, kwargs
            batch_size, seq_len = input_ids.shape
            logits = torch.zeros((batch_size, seq_len, 8), dtype=torch.float32)
            logits[:, :, 0] = float("nan")
            return SimpleNamespace(logits=logits)

    token_log_probs, response_mask = env_mod.compute_completion_only_token_log_probs_from_ids(
        model=_FakeModel(),
        prompt_ids=torch.tensor([[1, 2]], dtype=torch.long),
        prompt_mask=torch.ones((1, 2), dtype=torch.long),
        completion_ids=torch.tensor([[3, 4, 5]], dtype=torch.long),
        completion_mask=torch.ones((1, 3), dtype=torch.bool),
        multimodal_inputs=None,
    )

    assert response_mask.dtype == torch.bool
    assert torch.isnan(token_log_probs).any()


def test_completion_only_helper_avoids_full_vocab_log_softmax(monkeypatch: pytest.MonkeyPatch):
    class _FakeModel(torch.nn.Module):
        def forward(self, input_ids=None, attention_mask=None, logits_to_keep=None, **kwargs):
            del attention_mask, logits_to_keep, kwargs
            batch_size, seq_len = input_ids.shape
            logits = torch.arange(batch_size * seq_len * 8, dtype=torch.float32).view(batch_size, seq_len, 8)
            return SimpleNamespace(logits=logits)

    original_log_softmax = training_mod.F.log_softmax

    def _fail_log_softmax(*args, **kwargs):
        del args, kwargs
        raise AssertionError("completion-only helper should not materialize full-vocab log_softmax")

    monkeypatch.setattr(training_mod.F, "log_softmax", _fail_log_softmax)

    token_log_probs, response_mask = env_mod.compute_completion_only_token_log_probs_from_ids(
        model=_FakeModel(),
        prompt_ids=torch.tensor([[1, 2]], dtype=torch.long),
        prompt_mask=torch.ones((1, 2), dtype=torch.long),
        completion_ids=torch.tensor([[3, 4, 5]], dtype=torch.long),
        completion_mask=torch.ones((1, 3), dtype=torch.bool),
        multimodal_inputs=None,
    )

    full_logits = torch.arange(1 * 5 * 8, dtype=torch.float32).view(1, 5, 8)
    expected_shift_logits = full_logits[:, -4:, :][:, :-1, :]
    expected_log_probs = original_log_softmax(expected_shift_logits, dim=-1)
    expected = torch.gather(
        expected_log_probs,
        dim=-1,
        index=torch.tensor([[3, 4, 5]], dtype=torch.long).unsqueeze(-1),
    ).squeeze(-1)
    assert response_mask.dtype == torch.bool
    torch.testing.assert_close(token_log_probs, expected)


def test_grad_probe_logs_only_on_true_rank_zero(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    trainer = SimpleNamespace(
        local_rank=0,
        args=SimpleNamespace(max_grad_norm=1.0, local_rank=0, process_index=0),
        optimizer=None,
        deepspeed=None,
        model_wrapped=None,
        _last_rl_batch_size=None,
    )
    callback = env_mod._build_grad_norm_probe_callback(trainer=trainer)
    control = SimpleNamespace()
    state = SimpleNamespace(global_step=1, epoch=1.0, log_history=[])

    monkeypatch.setattr(
        env_mod,
        "distributed_runtime_from_env",
        lambda env=None: SimpleNamespace(rank=2, world_size=3, local_rank=2),
    )
    callback.on_train_begin(trainer.args, state, control)
    callback.on_step_end(trainer.args, state, control)
    assert capsys.readouterr().out == ""

    monkeypatch.setattr(
        env_mod,
        "distributed_runtime_from_env",
        lambda env=None: SimpleNamespace(rank=0, world_size=3, local_rank=0),
    )
    callback.on_train_begin(trainer.args, state, control)
    callback.on_step_end(trainer.args, state, control)
    captured = capsys.readouterr().out
    assert "GRAD_PROBE_INIT" in captured
    assert "GRAD_PROBE step=1" in captured


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


def test_compute_loss_normalizes_by_effective_sample_weight(monkeypatch: pytest.MonkeyPatch):
    trainer = object.__new__(env_mod._NativeGRPOTrainerMixin)
    trainer.args = SimpleNamespace(output_dir="/tmp/unit-test", gradient_accumulation_steps=16)
    trainer._debug_compute_loss_call_index = 0
    trainer._debug_last_compute_loss_call_index = 0
    trainer._debug_last_compute_loss_prepared_batch_count = 0
    trainer._debug_last_compute_loss_rank_local_batch_count = 0
    trainer.compute_loss_microbatch_size = 4
    trainer.all_empty_policy = "keep_zero_loss"
    trainer._mark_skip_next_optimizer_step = lambda **kwargs: None
    trainer._maybe_log_empty_batch_rank_summary = lambda **kwargs: None
    trainer._debug_prepared_batch_summary = lambda batch: {"keys": sorted(batch.keys())}
    trainer._prepared_batch_cpu_copy = lambda batch: batch
    trainer._clone_prepared_batch_as_noop = lambda batch: batch
    trainer._move_episode_spec_to_device = lambda batch, device: batch
    trainer._materialize_episode_spec_microbatch = lambda specs, device: list(specs)
    trainer._is_skippable_sample_exception = lambda exc: False
    trainer._prepared_batch_source_entries = lambda batch: []
    trainer._assert_finite_tensor = lambda **kwargs: None

    monkeypatch.setattr(env_mod, "_distributed_sum_int", lambda value, *, device: int(value))
    monkeypatch.setattr(env_mod, "_distributed_sum_float", lambda value, *, device: float(value))
    monkeypatch.setattr(env_mod, "_distributed_bool_consensus", lambda local_value, *, device: (bool(local_value), bool(local_value)))
    monkeypatch.setattr(env_mod, "_distributed_first_available_object", lambda local_object, *, device=None: local_object)
    monkeypatch.setattr(env_mod, "_distributed_world_size", lambda: 1)

    weighted_sample_losses = torch.tensor([0.2, 1.0], dtype=torch.float32)
    trainer._compute_sample_losses_for_batch = lambda model, batch: weighted_sample_losses

    model = torch.nn.Linear(1, 1, bias=False)
    batch = {
        "sample_loss_multiplier": torch.tensor([0.2, 1.0], dtype=torch.float32),
        "completion_ids": torch.tensor([[1], [2]], dtype=torch.long),
    }

    loss = trainer.compute_loss(model, {"episode_specs": [batch]})

    assert float(loss.item()) == pytest.approx(1.0, rel=1e-6, abs=1e-6)


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


def test_episode_spec_multimodal_inputs_excludes_source_entries():
    trainer = object.__new__(env_mod._NativeGRPOTrainerMixin)
    batch = {
        "prompt_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "prompt_mask": torch.ones((1, 2), dtype=torch.long),
        "completion_ids": torch.tensor([[3, 4, 5]], dtype=torch.long),
        "completion_mask": torch.ones((1, 3), dtype=torch.bool),
        "_source_episode_entries": [{"video_id": "vid-x"}],
        "pixel_values": torch.ones((2, 2), dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, 1, 2]], dtype=torch.long),
    }

    multimodal_inputs = trainer._episode_spec_multimodal_inputs(batch)

    assert "_source_episode_entries" not in multimodal_inputs
    assert set(multimodal_inputs.keys()) == {"pixel_values", "image_grid_thw"}


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


def test_old_policy_prefill_allows_non_finite_forward_outputs_without_dump(tmp_path):
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

    token_log_probs = trainer._compute_old_policy_token_log_probs_for_prepared_batch(
        model=_NaNLogitModel(),
        prepared_batch=prepared_batch,
        episode_entries=episode_entries,
    )

    dump_files = sorted((tmp_path / "non_finite_dumps").glob("*.json"))
    assert torch.isnan(token_log_probs).any()
    assert not dump_files


def test_old_policy_prefill_keeps_non_finite_sample_when_forward_succeeds(tmp_path):
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
    retained_entries = trainer._populate_old_policy_log_probs(
        _SelectiveNaNModel(),
        [bad_entry, good_entry],
    )

    assert len(retained_entries) == 2
    assert torch.isnan(retained_entries[0]["old_policy_token_log_probs"]).any()
    assert torch.isnan(retained_entries[1]["old_policy_token_log_probs"]).any()
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


def test_pad_prepared_batches_to_distributed_max_adds_noop_donor_batch(monkeypatch: pytest.MonkeyPatch):
    trainer = object.__new__(env_mod._NativeGRPOTrainerMixin)
    trainer._native_visual_tensor_dtype = None
    for method_name in (
        "_reserved_episode_spec_keys",
        "_is_visual_multimodal_tensor_key",
        "_move_multimodal_payload_to_device",
        "_move_episode_spec_to_device",
        "_prepared_batch_sample_count",
        "_clone_prepared_batch_as_noop",
        "_prepared_batch_cpu_copy",
        "_pad_prepared_batches_to_distributed_max",
    ):
        setattr(
            trainer,
            method_name,
            getattr(env_mod._NativeGRPOTrainerMixin, method_name).__get__(trainer, env_mod._NativeGRPOTrainerMixin),
        )

    local_batch = {
        "prompt_ids": torch.tensor([[11, 12]], dtype=torch.long),
        "prompt_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "completion_ids": torch.tensor([[21, 22]], dtype=torch.long),
        "completion_mask": torch.tensor([[1, 1]], dtype=torch.bool),
        "sample_weight": torch.ones(1, dtype=torch.float32),
        "advantage": torch.ones(1, dtype=torch.float32),
        "sample_loss_multiplier": torch.ones(1, dtype=torch.float32),
        "multimodal_inputs": {
            "pixel_values": torch.ones((4, 8), dtype=torch.bfloat16),
            "image_grid_thw": torch.ones((1, 3), dtype=torch.int64),
        },
    }
    donor_text_only_batch = {
        "prompt_ids": torch.tensor([[31, 32, 33]], dtype=torch.long),
        "prompt_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "completion_ids": torch.tensor([[41, 42, 43, 44]], dtype=torch.long),
        "completion_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.bool),
        "sample_weight": torch.ones(1, dtype=torch.float32),
        "advantage": torch.full((1,), 2.0, dtype=torch.float32),
        "sample_loss_multiplier": torch.ones(1, dtype=torch.float32),
        "multimodal_inputs": {},
    }

    monkeypatch.setattr(env_mod, "_distributed_max_int", lambda local_value, device: 2)
    monkeypatch.setattr(env_mod, "_distributed_min_int", lambda local_value, device: 1)
    monkeypatch.setattr(
        env_mod,
        "_distributed_first_available_object",
        lambda local_object, device=None: donor_text_only_batch if local_object is None else local_object,
    )

    runtime_stats = {}
    padded = trainer._pad_prepared_batches_to_distributed_max(
        [local_batch],
        device=torch.device("cpu"),
        runtime_stats=runtime_stats,
    )

    assert len(padded) == 2
    assert runtime_stats["ddp_noop_padded_prepared_batches"] == 1
    assert torch.equal(padded[0]["sample_weight"], local_batch["sample_weight"])
    assert padded[1]["multimodal_inputs"] == {}
    assert torch.equal(
        padded[1]["sample_loss_multiplier"],
        torch.zeros_like(donor_text_only_batch["sample_loss_multiplier"], dtype=torch.float32),
    )
    assert torch.equal(
        padded[1]["sample_weight"],
        torch.zeros_like(donor_text_only_batch["sample_weight"], dtype=torch.float32),
    )
    assert torch.equal(
        padded[1]["advantage"],
        torch.zeros_like(donor_text_only_batch["advantage"], dtype=torch.float32),
    )


def test_pad_prepared_batches_to_distributed_max_uses_symmetric_donor_gather(monkeypatch: pytest.MonkeyPatch):
    trainer = object.__new__(env_mod._NativeGRPOTrainerMixin)
    trainer._native_visual_tensor_dtype = None
    for method_name in (
        "_reserved_episode_spec_keys",
        "_is_visual_multimodal_tensor_key",
        "_move_multimodal_payload_to_device",
        "_move_episode_spec_to_device",
        "_prepared_batch_sample_count",
        "_clone_prepared_batch_as_noop",
        "_prepared_batch_cpu_copy",
        "_pad_prepared_batches_to_distributed_max",
    ):
        setattr(
            trainer,
            method_name,
            getattr(env_mod._NativeGRPOTrainerMixin, method_name).__get__(trainer, env_mod._NativeGRPOTrainerMixin),
        )

    local_batches = [
        {
            "prompt_ids": torch.tensor([[11, 12]], dtype=torch.long),
            "prompt_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "completion_ids": torch.tensor([[21, 22]], dtype=torch.long),
            "completion_mask": torch.tensor([[1, 1]], dtype=torch.bool),
            "sample_weight": torch.ones(1, dtype=torch.float32),
            "advantage": torch.ones(1, dtype=torch.float32),
            "sample_loss_multiplier": torch.ones(1, dtype=torch.float32),
            "multimodal_inputs": {},
        },
        {
            "prompt_ids": torch.tensor([[31, 32]], dtype=torch.long),
            "prompt_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "completion_ids": torch.tensor([[41, 42]], dtype=torch.long),
            "completion_mask": torch.tensor([[1, 1]], dtype=torch.bool),
            "sample_weight": torch.ones(1, dtype=torch.float32),
            "advantage": torch.ones(1, dtype=torch.float32),
            "sample_loss_multiplier": torch.ones(1, dtype=torch.float32),
            "multimodal_inputs": {},
        },
    ]

    gathered_positions = []
    monkeypatch.setattr(env_mod, "_distributed_max_int", lambda local_value, device: 2)
    monkeypatch.setattr(env_mod, "_distributed_min_int", lambda local_value, device: 1)

    def _fake_first_available(local_object, device=None):
        del device
        gathered_positions.append(local_object)
        return local_object

    monkeypatch.setattr(env_mod, "_distributed_first_available_object", _fake_first_available)

    runtime_stats = {}
    padded = trainer._pad_prepared_batches_to_distributed_max(
        local_batches,
        device=torch.device("cpu"),
        runtime_stats=runtime_stats,
    )

    assert len(padded) == 2
    assert len(gathered_positions) == 2
    assert "ddp_noop_padded_prepared_batches" not in runtime_stats
    assert runtime_stats["distributed_min_prepared_batch_count"] == 1
    assert runtime_stats["distributed_max_prepared_batch_count"] == 2


def test_pad_prepared_batches_replaces_nonvisual_local_batch_with_visual_noop(
    monkeypatch: pytest.MonkeyPatch,
):
    trainer = object.__new__(env_mod._NativeGRPOTrainerMixin)
    trainer._native_visual_tensor_dtype = None
    trainer._nonvisual_prepared_batch_noop_replacements = 0
    for method_name in (
        "_reserved_episode_spec_keys",
        "_is_visual_multimodal_tensor_key",
        "_move_multimodal_payload_to_device",
        "_move_episode_spec_to_device",
        "_prepared_batch_sample_count",
        "_clone_prepared_batch_as_noop",
        "_prepared_batch_cpu_copy",
        "_pad_prepared_batches_to_distributed_max",
    ):
        setattr(
            trainer,
            method_name,
            getattr(env_mod._NativeGRPOTrainerMixin, method_name).__get__(trainer, env_mod._NativeGRPOTrainerMixin),
        )

    local_text_only_batch = {
        "prompt_ids": torch.tensor([[11, 12]], dtype=torch.long),
        "prompt_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "completion_ids": torch.tensor([[21, 22]], dtype=torch.long),
        "completion_mask": torch.tensor([[1, 1]], dtype=torch.bool),
        "sample_weight": torch.ones(1, dtype=torch.float32),
        "advantage": torch.ones(1, dtype=torch.float32),
        "sample_loss_multiplier": torch.ones(1, dtype=torch.float32),
        "multimodal_inputs": {},
    }
    donor_visual_batch = {
        "prompt_ids": torch.tensor([[31, 32, 33]], dtype=torch.long),
        "prompt_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "completion_ids": torch.tensor([[41, 42, 43, 44]], dtype=torch.long),
        "completion_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.bool),
        "sample_weight": torch.ones(1, dtype=torch.float32),
        "advantage": torch.full((1,), 2.0, dtype=torch.float32),
        "sample_loss_multiplier": torch.ones(1, dtype=torch.float32),
        "multimodal_inputs": {
            "pixel_values": torch.ones((4, 8), dtype=torch.bfloat16),
            "image_grid_thw": torch.ones((1, 3), dtype=torch.int64),
        },
    }

    monkeypatch.setattr(env_mod, "_distributed_max_int", lambda local_value, device: 1)
    monkeypatch.setattr(env_mod, "_distributed_min_int", lambda local_value, device: 1)
    monkeypatch.setattr(env_mod, "_distributed_bool_consensus", lambda local_value, device: (False, True))
    monkeypatch.setattr(
        env_mod,
        "_distributed_first_available_object",
        lambda local_object, device=None: donor_visual_batch if local_object is None else local_object,
    )

    runtime_stats = {}
    padded = trainer._pad_prepared_batches_to_distributed_max(
        [local_text_only_batch],
        device=torch.device("cpu"),
        runtime_stats=runtime_stats,
    )

    assert len(padded) == 1
    assert runtime_stats["ddp_noop_replaced_nonvisual_prepared_batches"] == 1
    assert trainer._nonvisual_prepared_batch_noop_replacements == 1
    assert "pixel_values" in padded[0]["multimodal_inputs"]
    assert torch.equal(
        padded[0]["sample_loss_multiplier"],
        torch.zeros_like(donor_visual_batch["sample_loss_multiplier"], dtype=torch.float32),
    )
    assert torch.equal(
        padded[0]["sample_weight"],
        torch.zeros_like(donor_visual_batch["sample_weight"], dtype=torch.float32),
    )
    assert torch.equal(
        padded[0]["advantage"],
        torch.zeros_like(donor_visual_batch["advantage"], dtype=torch.float32),
    )
