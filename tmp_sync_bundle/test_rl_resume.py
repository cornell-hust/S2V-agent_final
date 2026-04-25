import json
from pathlib import Path
from types import SimpleNamespace

from saver_v3.rl.resume import load_trainer_resume_state, resolve_rl_training_resume_state
from saver_v3.rl.runtime import (
    RLJobConfig,
    _resolve_liger_compatible_deepspeed_config,
    build_active_rl_trl_argv,
)
from saver_v3.rl.trl_grpo_trainer import TrlVllmGrpoRunner, _rollout_eval_iteration_dir_name


def test_load_trainer_resume_state_reads_epoch_as_next_iteration(tmp_path: Path):
    checkpoint_dir = tmp_path / "checkpoint-40"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "trainer_state.json").write_text(
        json.dumps({"epoch": 40.0, "global_step": 400}),
        encoding="utf-8",
    )

    state = load_trainer_resume_state(checkpoint_dir, source="unit-test")

    assert state.checkpoint_path == str(checkpoint_dir.resolve())
    assert state.resume_iteration == 40
    assert state.completed_iteration == 39
    assert state.global_step == 400
    assert state.source == "unit-test"


def test_resolve_rl_training_resume_state_prefers_highest_iteration(tmp_path: Path):
    rl_dir = tmp_path / "rl"
    rl_dir.mkdir()
    checkpoint_40 = rl_dir / "checkpoint-400"
    checkpoint_80 = rl_dir / "checkpoint-800"
    checkpoint_40.mkdir()
    checkpoint_80.mkdir()
    (checkpoint_40 / "trainer_state.json").write_text(
        json.dumps({"epoch": 40.0, "global_step": 400}),
        encoding="utf-8",
    )
    (checkpoint_80 / "trainer_state.json").write_text(
        json.dumps({"epoch": 80.0, "global_step": 800}),
        encoding="utf-8",
    )
    iter_039 = rl_dir / "iter_039"
    iter_079 = rl_dir / "iter_079"
    iter_039.mkdir()
    iter_079.mkdir()
    (iter_039 / "summary.json").write_text(
        json.dumps({"iteration": 39, "latest_checkpoint": str(checkpoint_40)}),
        encoding="utf-8",
    )
    (iter_079 / "summary.json").write_text(
        json.dumps({"iteration": 79, "latest_checkpoint": str(checkpoint_80)}),
        encoding="utf-8",
    )

    state = resolve_rl_training_resume_state(rl_dir)

    assert state is not None
    assert state.resume_iteration == 80
    assert state.completed_iteration == 79
    assert state.checkpoint_path == str(checkpoint_80.resolve())


def test_build_active_rl_trl_argv_includes_resume_checkpoint():
    job = RLJobConfig(
        run_name="resume-test",
        output_dir="artifacts/rl_resume",
        rollout_eval_output_dir="artifacts/eval",
        train_manifest="/data/train.jsonl",
        eval_manifest="/data/eval.jsonl",
        data_root="/data",
        eval_data_root="/data",
        include_splits="train",
        eval_include_splits="test",
        policy_init_from="/models/sft",
        reference_model="",
        base_model="/models/base",
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_3",
        gradient_checkpointing=True,
        rollout_backend="vllm",
        rollout_config="configs/rollout_eval/vllm_qwen3_vl_8b_rl_lowmem.yaml",
        deepspeed_config_path=None,
        resume_from_checkpoint="/artifacts/rl/checkpoint-400",
    )

    argv = build_active_rl_trl_argv(job)

    assert "--resume-from-checkpoint" in argv
    assert argv[argv.index("--resume-from-checkpoint") + 1] == "/artifacts/rl/checkpoint-400"
    assert "--rollout-eval-output-dir" in argv
    assert argv[argv.index("--rollout-eval-output-dir") + 1] == "artifacts/eval"


def test_rollout_eval_iteration_dir_name_is_one_based():
    assert _rollout_eval_iteration_dir_name(0) == "rl_iter_001"
    assert _rollout_eval_iteration_dir_name(39) == "rl_iter_040"


def test_resolve_liger_compatible_deepspeed_config_preserves_offload_when_liger_disabled(tmp_path: Path):
    offload_config = tmp_path / "zero3_offload_rl.json"
    offload_config.write_text(
        json.dumps(
            {
                "zero_optimization": {
                    "stage": 3,
                    "offload_param": {"device": "cpu"},
                }
            }
        ),
        encoding="utf-8",
    )
    fallback = tmp_path / "zero2_rl.json"
    fallback.write_text(json.dumps({"zero_optimization": {"stage": 2}}), encoding="utf-8")

    resolved, switched, reason = _resolve_liger_compatible_deepspeed_config(
        str(offload_config),
        use_liger_loss=False,
    )

    assert resolved == str(offload_config.resolve())
    assert switched is False
    assert reason == ""


def test_rl_job_from_files_keeps_offload_config_when_active_rl_disables_liger(tmp_path: Path):
    rollout_config_path = tmp_path / "rollout.yaml"
    rollout_config_path.write_text("semantic_metrics:\n  enabled: false\n", encoding="utf-8")
    offload_config_path = tmp_path / "zero3_offload_rl.json"
    offload_config_path.write_text(
        json.dumps(
            {
                "zero_optimization": {
                    "stage": 3,
                    "offload_param": {"device": "cpu"},
                }
            }
        ),
        encoding="utf-8",
    )
    fallback_config_path = tmp_path / "zero2_rl.json"
    fallback_config_path.write_text(json.dumps({"zero_optimization": {"stage": 2}}), encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_name: test",
                "output_dir: artifacts/test_rl",
                f"rollout_config: {rollout_config_path}",
                "policy_init_from: /models/checkpoint",
                "rollout_backend: vllm",
                "data:",
                "  train_manifest: /data/train.jsonl",
                "  eval_manifest: /data/eval.jsonl",
                "  data_root: /data",
                "rewards:",
                "  reward_version: timesearch_v4",
            ]
        ),
        encoding="utf-8",
    )
    model_config_path = tmp_path / "model.yaml"
    model_config_path.write_text(
        "\n".join(
            [
                "base_model: /models/base",
                "torch_dtype: bfloat16",
                "attn_implementation: flash_attention_3",
                "sequence:",
                "  max_length: 8192",
                "vision:",
                "  max_images_per_sample: 28",
                "  max_image_side: 640",
                "  max_image_pixels: 0",
            ]
        ),
        encoding="utf-8",
    )
    attention_config_path = tmp_path / "attention.yaml"
    attention_config_path.write_text("policy_name: fa3_only\n", encoding="utf-8")

    job = RLJobConfig.from_files(
        config_path=str(config_path),
        model_config_path=str(model_config_path),
        attention_config_path=str(attention_config_path),
        deepspeed_config_path=str(offload_config_path),
    )

    assert job.requested_deepspeed_config_path == str(offload_config_path.resolve())
    assert job.deepspeed_config_path == str(offload_config_path.resolve())
    assert job.liger_deepspeed_auto_switched is False
    assert job.liger_deepspeed_switch_reason == ""


def test_resolve_liger_compatible_deepspeed_config_prefers_zero3_full_model_for_liger(tmp_path: Path):
    offload_config = tmp_path / "zero3_offload_rl.json"
    offload_config.write_text(
        json.dumps(
            {
                "zero_optimization": {
                    "stage": 3,
                    "offload_param": {"device": "cpu"},
                }
            }
        ),
        encoding="utf-8",
    )
    zero3_full_model = tmp_path / "zero3_full_model.json"
    zero3_full_model.write_text(json.dumps({"zero_optimization": {"stage": 3}}), encoding="utf-8")

    resolved, switched, reason = _resolve_liger_compatible_deepspeed_config(
        str(offload_config),
        use_liger_loss=True,
    )

    assert resolved == str(zero3_full_model.resolve())
    assert switched is True
    assert "zero3_full_model.json" in reason


def test_rl_shell_entrypoints_default_to_zero3_full_model():
    repo_root = Path(__file__).resolve().parents[2]
    pipeline_script = (repo_root / "scripts/run_full_pipeline.sh").read_text(encoding="utf-8")
    rl_script = (repo_root / "scripts/train_rl_qwen3_vl_8b_ds8.sh").read_text(encoding="utf-8")
    rl_yaml = (repo_root / "configs/rl/qwen3_vl_8b_grpo_train.yaml").read_text(encoding="utf-8")

    assert 'RL_DEEPSPEED_CONFIG="${RL_DEEPSPEED_CONFIG:-${ROOT_DIR}/configs/deepspeed/zero3_full_model.json}"' in pipeline_script
    assert 'DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${ROOT_DIR}/configs/deepspeed/zero3_full_model.json}"' in rl_script
    assert 'NPROC_PER_NODE="${NPROC_PER_NODE:-8}"' in rl_script
    assert 'RL_GRPO_ROLLOUT_COUNT="${RL_GRPO_ROLLOUT_COUNT:-8}"' in pipeline_script
    assert "deepspeed_config: configs/deepspeed/zero3_full_model.json" in rl_yaml
    assert "  rollout_count: 8" in rl_yaml
    assert "  gradient_accumulation_steps: 4" in rl_yaml
    assert "  nproc_per_node: 8" in rl_yaml
    assert "num_train_epochs:" not in rl_yaml


def test_grpo_trainer_env_no_longer_emits_prepared_batch_start_debug_log():
    repo_root = Path(__file__).resolve().parents[2]
    trainer_env_source = (repo_root / "saver_v3/rl/grpo_trainer_env.py").read_text(encoding="utf-8")

    assert "rl debug prepared batch start:" not in trainer_env_source
    assert "rl debug prepared batch layout:" not in trainer_env_source


def test_rl_job_from_files_rejects_non_default_num_train_epochs(tmp_path: Path):
    rollout_config_path = tmp_path / "rollout.yaml"
    rollout_config_path.write_text("semantic_metrics:\n  enabled: false\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_name: test",
                "output_dir: artifacts/test_rl",
                f"rollout_config: {rollout_config_path}",
                "policy_init_from: /models/checkpoint",
                "rollout_backend: vllm",
                "optimization:",
                "  num_iterations: 12",
                "  num_train_epochs: 2.0",
                "data:",
                "  train_manifest: /data/train.jsonl",
                "  eval_manifest: /data/eval.jsonl",
                "  data_root: /data",
                "rewards:",
                "  reward_version: timesearch_v4",
            ]
        ),
        encoding="utf-8",
    )
    model_config_path = tmp_path / "model.yaml"
    model_config_path.write_text(
        "\n".join(
            [
                "base_model: /models/base",
                "torch_dtype: bfloat16",
                "attn_implementation: flash_attention_3",
                "sequence:",
                "  max_length: 8192",
                "vision:",
                "  max_images_per_sample: 28",
                "  max_image_side: 640",
                "  max_image_pixels: 0",
            ]
        ),
        encoding="utf-8",
    )
    attention_config_path = tmp_path / "attention.yaml"
    attention_config_path.write_text("policy_name: fa3_only\n", encoding="utf-8")

    try:
        RLJobConfig.from_files(
            config_path=str(config_path),
            model_config_path=str(model_config_path),
            attention_config_path=str(attention_config_path),
        )
    except ValueError as exc:
        assert "num_train_epochs" in str(exc)
    else:
        raise AssertionError("Expected non-default optimization.num_train_epochs to be rejected")


def test_trl_runner_reuses_materialized_train_dataset_without_rebuilding(tmp_path: Path, monkeypatch):
    dataset_inits: list[str] = []

    class _FakeDataset:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            dataset_inits.append("init")
            self.records = [
                {
                    "video_id": "vid-1",
                    "video_path": "/data/vid-1.mp4",
                    "split": "train",
                }
            ]

        def __getitem__(self, index: int):
            return dict(self.records[index])

        def __len__(self) -> int:
            return len(self.records)

    monkeypatch.setattr(
        "saver_v3.rl.trl_grpo_trainer.ensure_materialized_cache_metadata",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "saver_v3.rl.trl_grpo_trainer.build_protocol_signature",
        lambda *args, **kwargs: "unit-test-protocol",
    )
    monkeypatch.setattr(
        "saver_v3.rl.trl_grpo_trainer.MaterializedRuntimeItemDataset",
        _FakeDataset,
    )

    args = SimpleNamespace(
        output_dir=str(tmp_path / "rl"),
        rollout_eval_output_dir=str(tmp_path / "eval"),
        materialized_train_items_path=str(tmp_path / "train.materialized.jsonl"),
        include_splits="train",
        data="/data/train.jsonl",
        require_materialized_runtime_cache=True,
        rollout_max_turns=10,
        policy_max_new_tokens=1024,
        resume_from_checkpoint="",
        model_path="/models/policy",
        use_liger_loss=False,
        num_iterations=2,
        rollout_count=1,
        rollout_start_index=0,
        num_generations=4,
        seed=42,
    )
    runtime = SimpleNamespace(is_main_process=True)

    runner = TrlVllmGrpoRunner(
        args=args,
        runtime=runtime,
        log_dir=str(tmp_path / "logs"),
        config_builder=lambda current_args: {"output_dir": current_args.output_dir},
        eval_config_builder=lambda **kwargs: None,
        reference_model_resolver=lambda *args, **kwargs: None,
        select_iteration_indices_fn=lambda raw_record_count, rollout_count, rollout_start_index, iteration_index, seed=42, records=None: [0],
    )

    assert len(dataset_inits) == 1
    assert runner.dataset is not None
    assert len(runner.raw_records) == 1
