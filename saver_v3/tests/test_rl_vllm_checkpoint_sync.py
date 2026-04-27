from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from saver_v3.metrics.evaluation import RolloutEvaluationConfig, _build_rollout_eval_metadata
from saver_v3.model import vllm_generation
from saver_v3.rl import trl_grpo_trainer as rl_trainer


def test_external_launcher_runtime_syncs_online_weights():
    calls = []

    class FakeDriverModel:
        def load_weights(self, weights):
            calls.extend(list(weights))

    runtime = object.__new__(vllm_generation._VllmExternalLauncherRuntime)
    runtime.enabled = True
    runtime.llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            model_executor=SimpleNamespace(
                driver_worker=SimpleNamespace(
                    model_runner=SimpleNamespace(model=FakeDriverModel())
                )
            )
        )
    )
    runtime._last_loaded_step = None
    runtime.weights_synced_step = None

    source_model = SimpleNamespace(
        named_parameters=lambda: [("w", torch.nn.Parameter(torch.tensor([1.0])))]
    )

    runtime.ensure_weights_synced(source_model, global_step=10)

    assert [(name, value.detach().cpu().tolist()) for name, value in calls] == [("w", [1.0])]
    assert runtime._last_loaded_step == 10
    assert runtime.weights_synced_step == 10


def test_external_launcher_runtime_skips_duplicate_sync_step():
    calls = []

    class FakeDriverModel:
        def load_weights(self, weights):
            calls.extend(list(weights))

    runtime = object.__new__(vllm_generation._VllmExternalLauncherRuntime)
    runtime.enabled = True
    runtime.llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            model_executor=SimpleNamespace(
                driver_worker=SimpleNamespace(
                    model_runner=SimpleNamespace(model=FakeDriverModel())
                )
            )
        )
    )
    runtime._last_loaded_step = 10
    runtime.weights_synced_step = 10
    source_model = SimpleNamespace(
        named_parameters=lambda: [("w", torch.nn.Parameter(torch.tensor([2.0])))]
    )

    runtime.ensure_weights_synced(source_model, global_step=10)

    assert calls == []


def test_rl_vllm_runtime_requires_sync_capable_runtime(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def _fake_create_vllm_runtime(*, args, runtime, model_path, prefer_direct_local_rank_runtime=False):
        captured["prefer_direct_local_rank_runtime"] = prefer_direct_local_rank_runtime
        captured["model_path"] = str(model_path)
        return SimpleNamespace(
            supports_weight_sync=False,
            base_model_path=str(model_path),
            mode="external_launcher",
        )

    monkeypatch.setattr(
        rl_trainer.shared_vllm_generation,
        "create_vllm_runtime",
        _fake_create_vllm_runtime,
    )

    args = SimpleNamespace(
        use_vllm=True,
        vllm_mode="colocate",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.35,
        vllm_max_num_seqs=4,
        vllm_fallback_max_num_seqs=2,
    )
    runtime = SimpleNamespace(is_distributed=True, local_rank=0, rank=0, world_size=1)

    with pytest.raises(RuntimeError, match="does not support online weight sync"):
        rl_trainer.create_vllm_runtime(
            args=args,
            runtime=runtime,
            model_path="/models/sft/checkpoint-8",
            require_weight_sync=True,
        )

    assert captured["prefer_direct_local_rank_runtime"] is True
    assert captured["model_path"] == "/models/sft/checkpoint-8"


def test_inline_eval_sync_is_observable_in_rollout_metadata():
    calls = []

    class FakeRuntime:
        supports_weight_sync = True
        base_model_path = "/artifacts/rl/checkpoint-40"
        mode = "colocate"
        _last_loaded_step = None
        weights_synced_step = None

        def ensure_weights_synced(self, source_model, *, global_step):
            calls.append((source_model, int(global_step)))
            self._last_loaded_step = int(global_step)
            self.weights_synced_step = int(global_step)

    runtime = FakeRuntime()
    source_model = object()
    policy = SimpleNamespace(vllm_runtime=runtime, source_model=source_model)
    state = SimpleNamespace(global_step=40)
    eval_config = RolloutEvaluationConfig(data_path="/data/eval.jsonl")

    synced_step = rl_trainer._sync_inline_vllm_policy_for_eval(policy, state=state)
    rl_trainer._annotate_rollout_eval_vllm_provenance(
        eval_config,
        current_model_path="/artifacts/rl/checkpoint-40",
        loadable_authority_checkpoint="/artifacts/rl/checkpoint-40",
        policy=policy,
        weights_synced_step=synced_step,
    )
    metadata = _build_rollout_eval_metadata(eval_config)

    assert calls == [(source_model, 40)]
    assert metadata["current_model_path"] == "/artifacts/rl/checkpoint-40"
    assert metadata["loadable_authority_checkpoint"] == "/artifacts/rl/checkpoint-40"
    assert metadata["runtime_type"] == "FakeRuntime"
    assert metadata["vllm_runtime_base_model_path"] == "/artifacts/rl/checkpoint-40"
    assert metadata["vllm_runtime_supports_weight_sync"] is True
    assert metadata["weights_synced_step"] == 40


def test_rollout_eval_metadata_includes_checkpoint_provenance_fields():
    eval_config = RolloutEvaluationConfig(
        data_path="/data/eval.jsonl",
        current_model_path="/artifacts/rl/checkpoint-20",
        loadable_authority_checkpoint="/artifacts/rl/checkpoint-20",
        runtime_type="_VllmColocateRuntime",
        vllm_runtime_type="_VllmColocateRuntime",
        vllm_runtime_base_model_path="/artifacts/rl/checkpoint-20",
        vllm_runtime_supports_weight_sync=True,
        weights_synced_step=20,
    )

    metadata = _build_rollout_eval_metadata(eval_config)

    assert metadata["current_model_path"] == "/artifacts/rl/checkpoint-20"
    assert metadata["loadable_authority_checkpoint"] == "/artifacts/rl/checkpoint-20"
    assert metadata["runtime_type"] == "_VllmColocateRuntime"
    assert metadata["vllm_runtime_type"] == "_VllmColocateRuntime"
    assert metadata["vllm_runtime_supports_weight_sync"] is True
    assert metadata["weights_synced_step"] == 20


def test_close_persistent_vllm_runtime_detaches_reward_judge():
    events = []
    old_engine = object()

    class FakeJudge:
        def __init__(self):
            self.engine = old_engine

        def attach_local_vllm(self, engine):
            events.append(("attach", engine))
            self.engine = engine

    class FakeRuntime:
        def close(self):
            events.append(("close", None))

    runner = object.__new__(rl_trainer.TrlVllmGrpoRunner)
    runner.persistent_vllm_runtime = FakeRuntime()
    runner.inline_policy_factory = object()
    trainer = SimpleNamespace(_vllm_runtime=runner.persistent_vllm_runtime, reward_judge=FakeJudge())

    runner._close_persistent_vllm_runtime(trainer=trainer)

    assert trainer._vllm_runtime is None
    assert trainer.reward_judge.engine is None
    assert events == [("attach", None), ("close", None)]
    assert runner.persistent_vllm_runtime is None
    assert runner.inline_policy_factory is None


def test_cleanup_inline_rollout_eval_policy_releases_large_refs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rl_trainer, "_release_inline_eval_cuda_cache", lambda: None)
    runtime = object()
    model = object()
    policy = SimpleNamespace(
        _prepared_messages_cache=[{"image": object()}],
        _prepared_messages_cache_signatures=[("role", "content")],
        _prepared_messages_cache_source_id=123,
        _prepared_messages_cache_len=1,
        _last_rl_token_traces=[{"ids": torch.tensor([1])}],
        vllm_runtime=runtime,
        source_model=model,
        model=model,
    )

    rl_trainer._cleanup_inline_rollout_eval_policy(policy)

    assert policy._prepared_messages_cache == []
    assert policy._prepared_messages_cache_signatures == []
    assert policy._prepared_messages_cache_source_id is None
    assert policy._prepared_messages_cache_len == 0
    assert policy._last_rl_token_traces is None
    assert policy.vllm_runtime is None
    assert policy.source_model is None
    assert policy.model is None


def test_continuous_save_uses_global_step_checkpoint_not_stale_last_checkpoint(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip("transformers")
    import transformers.trainer_utils as trainer_utils

    output_dir = tmp_path / "rl"
    stale_checkpoint = output_dir / "checkpoint-10"
    current_checkpoint = output_dir / "checkpoint-20"
    stale_checkpoint.mkdir(parents=True)
    current_checkpoint.mkdir(parents=True)

    monkeypatch.setattr(trainer_utils, "get_last_checkpoint", lambda _: str(stale_checkpoint))

    saved_roots = []

    def _fake_save_loadable_hf_authority_checkpoint(*, trainer, processor, checkpoint_root, epoch_index, runtime):
        del trainer, processor, runtime
        saved_roots.append(Path(checkpoint_root))
        del epoch_index
        loadable_dir = Path(checkpoint_root)
        (loadable_dir / "config.json").write_text("{}", encoding="utf-8")
        return loadable_dir

    monkeypatch.setattr(
        rl_trainer,
        "_save_loadable_hf_authority_checkpoint",
        _fake_save_loadable_hf_authority_checkpoint,
    )

    runtime = SimpleNamespace(
        rank=0,
        world_size=1,
        local_rank=0,
        is_distributed=False,
        is_main_process=True,
    )
    owner_output_dir = tmp_path / "owner"
    logs_dir = tmp_path / "logs"
    owner_output_dir.mkdir()
    logs_dir.mkdir()
    owner = SimpleNamespace(
        args=SimpleNamespace(num_generations=4),
        runtime=runtime,
        current_model_path="/models/sft/checkpoint-8",
        reference_model_source_path="/models/sft/checkpoint-8",
        reference_model_backend="none",
        use_liger_loss_requested=False,
        use_liger_loss_effective=False,
        output_dir=owner_output_dir,
        resolved_log_dir=logs_dir,
        rollout_eval_output_root=tmp_path / "eval",
        persistent_vllm_runtime=None,
        config_builder=lambda args: SimpleNamespace(),
        eval_config_builder=lambda **kwargs: SimpleNamespace(inline_rollout_eval=False),
        _build_iteration_summary=lambda *, iteration, items: {"iteration": int(iteration), "items": items},
    )
    mutable_dataset = SimpleNamespace(snapshot_items=lambda: [])
    trainer = SimpleNamespace(
        reference_model_source_path="/models/sft/checkpoint-8",
        reference_model_backend="none",
        use_liger_loss_requested=False,
        use_liger_loss_effective=False,
        get_budgeting_stats=lambda: None,
        get_budget_drop_metrics=lambda: {},
    )

    callback = rl_trainer._build_continuous_iteration_callback(
        owner=owner,
        mutable_dataset=mutable_dataset,
        trainer=trainer,
        processor=SimpleNamespace(),
        eval_start_iteration=1,
        eval_every_iterations=1,
    )

    callback.on_save(
        SimpleNamespace(output_dir=str(output_dir)),
        SimpleNamespace(global_step=20, epoch=20.0),
        SimpleNamespace(),
        model=object(),
    )

    assert saved_roots == [current_checkpoint]
    assert callback.last_saved_checkpoint == current_checkpoint
    assert owner.current_model_path == str(current_checkpoint)
