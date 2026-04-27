import json
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from saver_v3.data.config import SaverAgentConfig
from saver_v3.inference.fixed_baseline_eval import FixedBaselineEvalConfig
from saver_v3.inference.rollout_eval import StepRolloutEvalConfig, _build_vllm_args
from saver_v3.metrics import evaluation as evaluation_mod
from saver_v3.metrics.evaluation import RolloutEvaluationConfig, run_rollout_evaluation
from saver_v3.metrics.semantic_metrics import _evaluate_bertscore
from saver_v3.model.vllm_generation import (
    _construct_vllm_llm_with_max_model_len_retry,
    _extract_vllm_estimated_max_model_len,
)
from saver_v3.rl.cli_shared import build_active_rl_arg_parser, build_rollout_eval_config
from saver_v3.rl.grpo_trainer_env import (
    _NativeGRPOProgressReporter,
    _build_seed_worker_init_fn,
    _compute_rank_local_total_groups,
    create_native_grpo_trainer,
)
from saver_v3.rl.trl_grpo_trainer import create_trl_vllm_grpo_trainer
from saver_v3.rl.runtime import RLJobConfig, build_active_rl_trl_argv
from saver_v3.sft import training as sft_training_mod


def _write_train_manifest(tmp_path: Path, *, count: int = 1, split: str = "train") -> Path:
    path = tmp_path / "train.jsonl"
    rows = [
        {
            "video_id": f"video-{index}",
            "video_path": f"/data/video-{index}.mp4",
            "split": split,
        }
        for index in range(int(count))
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_build_rollout_eval_config_inherits_semantic_settings(tmp_path: Path):
    parser = build_active_rl_arg_parser(description="test parser")
    args = parser.parse_args(
        [
            "--data",
            str(tmp_path / "train.jsonl"),
            "--output-dir",
            str(tmp_path / "output"),
            "--eval-data",
            str(tmp_path / "eval.jsonl"),
            "--materialized-eval-items-path",
            str(tmp_path / "eval.materialized_items_v3.jsonl"),
            "--require-materialized-runtime-cache",
            "true",
        ]
    )
    args.eval_enable_semantic_metrics = False
    args.eval_semantic_metrics = "qa_accuracy,bertscore"
    args.eval_semantic_judge_model = "judge-model"
    args.eval_bertscore_model_path = "/models/roberta-large"

    config = build_rollout_eval_config(
        args=args,
        current_model_path="/models/policy",
        reference_model_path="/models/reference",
        config=SaverAgentConfig(),
    )

    assert config is not None
    assert config.enable_semantic_metrics is False
    assert config.semantic_metrics == "qa_accuracy,bertscore"
    assert config.semantic_judge_model == "judge-model"
    assert config.semantic_bertscore_model_path == "/models/roberta-large"
    assert config.materialized_items_path == str(tmp_path / "eval.materialized_items_v3.jsonl")
    assert config.require_materialized_cache is True


def test_rl_job_runtime_forwards_rollout_semantic_settings(tmp_path: Path):
    rollout_config_path = tmp_path / "rollout.yaml"
    rollout_config_path.write_text(
        "\n".join(
            [
                "semantic_metrics:",
                "  enabled: false",
                "  metrics:",
                "    - qa_accuracy",
                "    - bertscore",
                "  bertscore_model_path: /models/roberta-large",
            ]
        ),
        encoding="utf-8",
    )
    train_manifest_path = _write_train_manifest(tmp_path)
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
                f"  train_manifest: {train_manifest_path}",
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
    )
    argv = build_active_rl_trl_argv(job)

    assert job.eval_enable_semantic_metrics is False
    assert job.eval_semantic_metrics == ["qa_accuracy", "bertscore"]
    assert job.eval_bertscore_model_path == "/models/roberta-large"
    assert "--eval-enable-semantic-metrics" in argv
    assert argv[argv.index("--eval-enable-semantic-metrics") + 1] == "false"
    assert argv[argv.index("--eval-semantic-metrics") + 1] == "qa_accuracy,bertscore"
    assert argv[argv.index("--eval-bertscore-model-path") + 1] == "/models/roberta-large"


def test_rl_job_runtime_uses_throughput_oriented_dataloader_defaults(tmp_path: Path):
    rollout_config_path = tmp_path / "rollout.yaml"
    rollout_config_path.write_text("semantic_metrics:\n  enabled: false\n", encoding="utf-8")
    train_manifest_path = _write_train_manifest(tmp_path)
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
                f"  train_manifest: {train_manifest_path}",
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
    )
    argv = build_active_rl_trl_argv(job)

    assert job.dataloader_num_workers == 4
    assert job.dataloader_prefetch_factor == 2
    assert job.dataloader_persistent_workers is False
    assert argv[argv.index("--dataloader-num-workers") + 1] == "4"
    assert argv[argv.index("--dataloader-prefetch-factor") + 1] == "2"
    assert "--dataloader-persistent-workers" not in argv


def test_evaluate_bertscore_uses_explicit_model_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "saver_v3.metrics.semantic_metrics._collect_summary_strings",
        lambda rollouts, reference_data: (["pred"], ["ref"]),
    )

    summary = _evaluate_bertscore(
        [{"video_id": "video-1"}],
        reference_data=SimpleNamespace(),
        bertscore_model_path="/models/roberta-large",
    )

    assert summary["available"] is False
    assert summary["skipped_reason"] == "BERTScore is temporarily disabled in this repository."
    assert summary["model_type"] == "/models/roberta-large"


def test_resolve_training_proposal_device_defaults_to_local_rank(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sft_training_mod.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sft_training_mod.torch.cuda, "device_count", lambda: 8)

    device = sft_training_mod._resolve_training_proposal_device(
        "",
        runtime=SimpleNamespace(local_rank=5, world_size=8),
    )

    assert device == "cuda:5"


def test_resolve_training_proposal_device_warns_on_explicit_single_cuda(monkeypatch: pytest.MonkeyPatch):
    warnings = []
    monkeypatch.setattr(sft_training_mod, "runtime_log", lambda message, **kwargs: warnings.append(str(message)))

    device = sft_training_mod._resolve_training_proposal_device(
        "cuda:0",
        runtime=SimpleNamespace(local_rank=5, world_size=8),
    )

    assert device == "cuda:0"
    assert any("pins every rank's SigLIP encoder to the same CUDA device" in message for message in warnings)


def test_resolve_eval_proposal_device_defaults_to_local_rank(monkeypatch: pytest.MonkeyPatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 8,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    device = evaluation_mod._resolve_proposal_device(
        "",
        runtime=SimpleNamespace(local_rank=6, world_size=8),
    )

    assert device == "cuda:6"


def test_resolve_eval_proposal_device_warns_on_explicit_single_cuda(monkeypatch: pytest.MonkeyPatch):
    warnings = []
    monkeypatch.setattr(evaluation_mod, "runtime_log", lambda message, **kwargs: warnings.append(str(message)))

    device = evaluation_mod._resolve_proposal_device(
        "cuda:0",
        runtime=SimpleNamespace(local_rank=6, world_size=8),
    )

    assert device == "cuda:0"
    assert any("pins every rank's SigLIP encoder to the same CUDA device" in message for message in warnings)


def test_native_grpo_progress_reporter_uses_explicit_local_total_groups():
    reporter = _NativeGRPOProgressReporter(
        runtime=SimpleNamespace(world_size=8, rank=6),
        iteration_index=0,
        num_iterations=80,
        total_groups=12,
        num_generations=4,
    )

    reporter.set_local_total_groups(2)

    assert reporter._local_total_groups() == 2
    assert reporter._display_global_processed_groups(16) == 12


def test_compute_rank_local_total_groups_matches_rank_shard_distribution():
    assert _compute_rank_local_total_groups(8, runtime=SimpleNamespace(world_size=8, rank=0)) == 1
    assert _compute_rank_local_total_groups(8, runtime=SimpleNamespace(world_size=8, rank=7)) == 1
    assert _compute_rank_local_total_groups(12, runtime=SimpleNamespace(world_size=8, rank=0)) == 2
    assert _compute_rank_local_total_groups(12, runtime=SimpleNamespace(world_size=8, rank=6)) == 1


def test_build_seed_worker_init_fn_supports_new_transformers_signature(monkeypatch: pytest.MonkeyPatch):
    trainer_utils_mod = type(sys)("transformers.trainer_utils")
    calls = []

    def fake_seed_worker(worker_id, num_workers, rank):
        calls.append((worker_id, num_workers, rank))

    trainer_utils_mod.seed_worker = fake_seed_worker
    transformers_mod = type(sys)("transformers")
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "transformers.trainer_utils", trainer_utils_mod)

    worker_init_fn = _build_seed_worker_init_fn(
        args=SimpleNamespace(dataloader_num_workers=4, process_index=3, seed=42, data_seed=None)
    )

    assert worker_init_fn is not None
    worker_init_fn(1)
    assert calls == [(1, 4, 3)]


def test_build_seed_worker_init_fn_preserves_legacy_single_argument_signature(monkeypatch: pytest.MonkeyPatch):
    trainer_utils_mod = type(sys)("transformers.trainer_utils")
    calls = []

    def fake_seed_worker(worker_id):
        calls.append(worker_id)

    trainer_utils_mod.seed_worker = fake_seed_worker
    transformers_mod = type(sys)("transformers")
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "transformers.trainer_utils", trainer_utils_mod)

    worker_init_fn = _build_seed_worker_init_fn(
        args=SimpleNamespace(dataloader_num_workers=4, process_index=3, seed=42, data_seed=None)
    )

    assert worker_init_fn is fake_seed_worker
    worker_init_fn(2)
    assert calls == [2]


def test_fixed_baseline_eval_config_reads_bertscore_model_path():
    config = FixedBaselineEvalConfig.from_mapping(
        {
            "base_model": "/models/qwen",
            "io": {
                "data_path": "/data/eval.jsonl",
                "output_dir": "artifacts/fixed_eval",
            },
            "evaluation": {
                "enable_semantic_metrics": True,
                "semantic_metrics": ["qa_accuracy", "bertscore"],
                "bertscore_model_path": "/models/roberta-large",
            },
        }
    )

    assert config.enable_semantic_metrics is True
    assert config.semantic_metrics == ["qa_accuracy", "bertscore"]
    assert config.semantic_bertscore_model_path == "/models/roberta-large"


def test_default_sft_rollout_eval_config_enables_qa_accuracy():
    config_path = Path(__file__).resolve().parents[2] / "configs" / "rollout_eval" / "vllm_qwen3_vl_8b.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    semantic_metrics = dict(payload.get("semantic_metrics") or {})

    assert semantic_metrics.get("enabled") is True
    assert "qa_accuracy" in list(semantic_metrics.get("metrics") or [])


def test_run_rollout_evaluation_writes_qa_accuracy_overall(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    class _FakeDataset:
        def __init__(self):
            self.records = [{"video_id": "vid-1"}]

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            del idx
            return {"video_id": "vid-1", "messages": []}

        def format_frame_cache_status(self, *, prefix: str = "rollout eval frame cache", max_examples: int = 5):
            del max_examples
            return f"{prefix}: ok"

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def run_episodes(self, batch_items, policy):
            del policy
            return [{"video_id": str(item.get("video_id") or ""), "messages": []} for item in batch_items]

    runtime = SimpleNamespace(rank=0, world_size=1, is_main_process=True, is_distributed=False, local_rank=0)

    monkeypatch.setattr(evaluation_mod, "init_torch_distributed", lambda runtime: None)
    monkeypatch.setattr(
        evaluation_mod,
        "resolve_shard_spec",
        lambda runtime: SimpleNamespace(num_shards=1, shard_index=0, is_sharded=False),
    )
    monkeypatch.setattr(evaluation_mod, "distributed_barrier", lambda runtime: None)
    monkeypatch.setattr(evaluation_mod, "runtime_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(evaluation_mod, "SaverAgentDataset", lambda *args, **kwargs: _FakeDataset())
    monkeypatch.setattr(evaluation_mod, "SaverRolloutRunner", _FakeRunner)
    monkeypatch.setattr(evaluation_mod, "_cleanup_cuda_cache", lambda **kwargs: None)
    monkeypatch.setattr(evaluation_mod, "_load_proposal_runtime", lambda **kwargs: None)
    monkeypatch.setattr(evaluation_mod, "_load_verifier_runtime", lambda **kwargs: None)
    monkeypatch.setattr(evaluation_mod, "_clear_stale_json_shards", lambda path: 0)
    monkeypatch.setattr(evaluation_mod, "_clear_rollout_eval_sync_files", lambda eval_root: 0)
    monkeypatch.setattr(evaluation_mod, "_append_per_video_rollout_record", lambda **kwargs: None)
    monkeypatch.setattr(evaluation_mod, "score_rollout_records", lambda rollouts, **kwargs: [dict(rollout) for rollout in rollouts])
    monkeypatch.setattr(
        evaluation_mod,
        "summarize_saver_metrics",
        lambda merged_scored_records, reference_data, include_diagnostic_summary=False: {"existence_accuracy": 1.0},
    )
    monkeypatch.setattr(
        evaluation_mod,
        "evaluate_semantic_rollouts",
        lambda *args, **kwargs: {"qa_accuracy_overall": 0.75, "qa_accuracy_coverage": 1.0, "qa_accuracy_fields": {}},
    )
    monkeypatch.setattr(evaluation_mod, "ReferenceDataProvider", lambda data_path, data_root="": SimpleNamespace(records=[]))
    monkeypatch.setattr(
        evaluation_mod,
        "_wait_for_current_scored_records",
        lambda scored_shard_dir, failure_path, runtime: [{"video_id": "vid-1"}],
    )

    eval_config = RolloutEvaluationConfig(
        data_path=str(tmp_path / "eval.jsonl"),
        data_root=str(tmp_path),
        rollout_max_turns=1,
        saver_config=SaverAgentConfig(),
        enable_semantic_replay=False,
        enable_semantic_metrics=True,
        semantic_metrics=["qa_accuracy"],
    )

    summary = run_rollout_evaluation(
        policy=SimpleNamespace(),
        eval_config=eval_config,
        output_dir=tmp_path / "out",
        epoch_index=1,
        runtime=runtime,
    )

    semantic_metrics_path = tmp_path / "out" / "rollout_eval" / "epoch_001" / "semantic_metrics.json"
    metrics_path = tmp_path / "out" / "rollout_eval" / "epoch_001" / "metrics.json"

    assert summary is not None
    assert semantic_metrics_path.exists()
    assert metrics_path.exists()
    assert json.loads(semantic_metrics_path.read_text(encoding="utf-8"))["qa_accuracy_overall"] == pytest.approx(0.75)
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["semantic_metrics"]["qa_accuracy_overall"] == pytest.approx(0.75)


def test_rollout_eval_build_vllm_args_preserves_prompt_budget_fields():
    config = StepRolloutEvalConfig(
        base_model="/models/qwen",
        data_path="/data/eval.jsonl",
        output_dir="artifacts/eval",
        policy_max_new_tokens=512,
        max_total_images=28,
        max_seq_length=8192,
        keep_recent_text_messages=20,
        keep_recent_tool_image_messages=3,
        max_image_side=640,
        max_image_pixels=0,
    )

    args = _build_vllm_args(config)

    assert args.max_seq_length == 8192
    assert args.policy_max_new_tokens == 512
    assert args.max_new_tokens == 512
    assert args.max_total_images == 28


def test_extract_vllm_estimated_max_model_len_from_error():
    exc = ValueError(
        "Based on the available memory, the estimated maximum model length is 12256. "
        "Try increasing gpu_memory_utilization."
    )

    assert _extract_vllm_estimated_max_model_len(exc) == 12256


def test_construct_vllm_llm_with_max_model_len_retry(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_llm_ctor(**kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise ValueError(
                "To serve at least one request ... estimated maximum model length is 12256."
            )
        return {"ok": True, "max_model_len": kwargs["max_model_len"]}

    runtime = SimpleNamespace(is_main_process=True)
    monkeypatch.setattr("saver_v3.model.vllm_generation.runtime_log", lambda *args, **kwargs: None)
    result = _construct_vllm_llm_with_max_model_len_retry(
        llm_ctor=fake_llm_ctor,
        llm_kwargs={"max_model_len": 16384, "model": "dummy"},
        runtime=runtime,
        context_label="unit_test_vllm_retry",
    )

    assert result == {"ok": True, "max_model_len": 12256}
    assert [call["max_model_len"] for call in calls] == [16384, 12256]




def test_create_native_grpo_trainer_syncs_accelerate_gradient_accumulation_steps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured = {}

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            captured["kwargs"] = dict(kwargs)
            captured["env"] = __import__("os").environ.get("ACCELERATE_GRADIENT_ACCUMULATION_STEPS")

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            self.args = kwargs.get("args")
            self.train_dataset = kwargs.get("train_dataset")
            self.data_collator = kwargs.get("data_collator")
            self.model = kwargs.get("model")
            self.callbacks = []

        def add_callback(self, callback):
            self.callbacks.append(callback)

    fake_transformers = SimpleNamespace(Trainer=FakeTrainer, TrainingArguments=FakeTrainingArguments)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.delenv("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", raising=False)

    trainer = create_native_grpo_trainer(
        model=object(),
        processor=object(),
        train_dataset=[],
        output_dir=tmp_path / "out",
        learning_rate=1e-5,
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        logging_steps=1,
        save_steps=10,
        save_total_limit=1,
        warmup_ratio=0.0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=False,
        old_policy_model=None,
        reference_model=None,
        kl_beta=0.0,
        ppo_clip_epsilon=0.2,
        rollout_runner=object(),
        num_generations=4,
        min_weight=0.01,
        advantage_clip=3.0,
        policy_max_new_tokens=1024,
        max_image_side=640,
        max_image_pixels=0,
        max_total_images=28,
        keep_recent_tool_image_messages=3,
        keep_recent_text_messages=20,
        max_seq_length=8192,
        policy_do_sample=True,
        policy_temperature=0.8,
        policy_top_p=0.95,
        policy_top_k=50,
        policy_repetition_penalty=1.02,
        save_strategy="no",
    )

    assert trainer is not None
    assert captured["kwargs"]["gradient_accumulation_steps"] == 16
    assert captured["env"] == "16"


def test_create_trl_vllm_grpo_trainer_keeps_ddp_find_unused_parameters_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    captured = {}

    def fake_create_native_grpo_trainer(**kwargs):
        captured.update(kwargs)
        return {"trainer": "ok"}

    monkeypatch.setattr("saver_v3.rl.trl_grpo_trainer.create_native_grpo_trainer", fake_create_native_grpo_trainer)
    monkeypatch.setattr("saver_v3.rl.trl_grpo_trainer.SaverRolloutRunner", lambda **kwargs: {"runner": kwargs})

    args = SimpleNamespace(
        learning_rate=1e-5,
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        logging_steps=1,
        save_steps=10,
        save_total_limit=1,
        warmup_ratio=0.0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=False,
        kl_beta=0.0,
        ppo_clip_epsilon=0.2,
        rollout_max_turns=10,
        num_generations=1,
        min_weight=0.0,
        advantage_clip=5.0,
        policy_max_new_tokens=8192,
        max_image_side=640,
        max_image_pixels=0,
        max_total_images=28,
        max_tool_message_frames=0,
        max_total_video_frames=0,
        keep_recent_tool_image_messages=3,
        keep_recent_text_messages=20,
        max_seq_length=8192,
        policy_do_sample=False,
        policy_temperature=0.0,
        policy_top_p=1.0,
        policy_top_k=-1,
        policy_repetition_penalty=1.0,
        rl_rollout_use_cache=True,
        rl_compute_loss_microbatch_size=1,
        rl_steps_per_generation=1,
        rl_log_empty_batch_rank_summary=True,
        rl_reward_version="timesearch_v4",
        rl_reward_config={},
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_3",
        deepspeed="",
    )

    result = create_trl_vllm_grpo_trainer(
        args=args,
        model=object(),
        processor=object(),
        trainer_init_model_path=str(tmp_path / "checkpoint"),
        train_items=[],
        train_dataset=[],
        checkpoint_dir=tmp_path / "out",
        iteration_index=0,
        num_iterations=1,
        config=SaverAgentConfig(),
        rollout_eval_callback=None,
        vllm_runtime=None,
        proposal_runtime=None,
        strict_feature_guided_proposal=False,
        save_strategy="no",
    )

    assert result == {"trainer": "ok"}
    assert captured["ddp_find_unused_parameters"] is False
def test_create_native_grpo_trainer_signature_removes_counterfactual_max_images():
    signature = inspect.signature(create_native_grpo_trainer)

    assert "counterfactual_max_images" not in signature.parameters


def test_create_trl_vllm_grpo_trainer_does_not_forward_counterfactual_max_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    captured = {}

    def fake_create_native_grpo_trainer(**kwargs):
        captured.update(kwargs)
        return {"trainer": "ok"}

    monkeypatch.setattr("saver_v3.rl.trl_grpo_trainer.create_native_grpo_trainer", fake_create_native_grpo_trainer)
    monkeypatch.setattr("saver_v3.rl.trl_grpo_trainer.SaverRolloutRunner", lambda **kwargs: {"runner": kwargs})

    args = SimpleNamespace(
        learning_rate=1e-5,
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=1,
        save_steps=10,
        save_total_limit=1,
        warmup_ratio=0.0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        dataloader_num_workers=0,
        dataloader_prefetch_factor=2,
        dataloader_persistent_workers=False,
        kl_beta=0.0,
        ppo_clip_epsilon=0.2,
        rollout_max_turns=10,
        num_generations=1,
        min_weight=0.0,
        advantage_clip=5.0,
        policy_max_new_tokens=8192,
        max_image_side=640,
        max_image_pixels=0,
        max_total_images=28,
        max_tool_message_frames=0,
        max_total_video_frames=0,
        keep_recent_tool_image_messages=3,
        keep_recent_text_messages=20,
        max_seq_length=8192,
        policy_do_sample=False,
        policy_temperature=0.0,
        policy_top_p=1.0,
        policy_top_k=-1,
        policy_repetition_penalty=1.0,
        rl_rollout_use_cache=True,
        rl_compute_loss_microbatch_size=1,
        rl_steps_per_generation=1,
        rl_log_empty_batch_rank_summary=True,
        rl_reward_version="timesearch_v4",
        rl_reward_config={},
        torch_dtype="bfloat16",
        attn_implementation="flash_attention_3",
        deepspeed="",
    )

    result = create_trl_vllm_grpo_trainer(
        args=args,
        model=object(),
        processor=object(),
        trainer_init_model_path=str(tmp_path / "checkpoint"),
        train_items=[],
        train_dataset=[],
        checkpoint_dir=tmp_path / "out",
        iteration_index=0,
        num_iterations=1,
        config=SaverAgentConfig(),
        rollout_eval_callback=None,
        vllm_runtime=None,
        proposal_runtime=None,
        strict_feature_guided_proposal=False,
        save_strategy="no",
    )

    assert result == {"trainer": "ok"}
    assert "counterfactual_max_images" not in captured
    assert captured["proposal_runtime"] is None
    assert captured["strict_feature_guided_proposal"] is False
