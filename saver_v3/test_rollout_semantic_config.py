import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from saver_v3.data.config import SaverAgentConfig
from saver_v3.inference.fixed_baseline_eval import FixedBaselineEvalConfig
from saver_v3.inference.rollout_eval import StepRolloutEvalConfig, _build_vllm_args
from saver_v3.metrics.semantic_metrics import _evaluate_bertscore
from saver_v3.model.vllm_generation import (
    _construct_vllm_llm_with_max_model_len_retry,
    _extract_vllm_estimated_max_model_len,
)
from saver_v3.rl.cli_shared import build_active_rl_arg_parser, build_rollout_eval_config
from saver_v3.rl.runtime import RLJobConfig, build_active_rl_trl_argv


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
    )
    argv = build_active_rl_trl_argv(job)

    assert job.eval_enable_semantic_metrics is False
    assert job.eval_semantic_metrics == ["qa_accuracy", "bertscore"]
    assert job.eval_bertscore_model_path == "/models/roberta-large"
    assert "--eval-enable-semantic-metrics" in argv
    assert argv[argv.index("--eval-enable-semantic-metrics") + 1] == "false"
    assert argv[argv.index("--eval-semantic-metrics") + 1] == "qa_accuracy,bertscore"
    assert argv[argv.index("--eval-bertscore-model-path") + 1] == "/models/roberta-large"


def test_evaluate_bertscore_uses_explicit_model_path(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_bert_score(predictions, references, **kwargs):
        captured["predictions"] = list(predictions)
        captured["references"] = list(references)
        captured["kwargs"] = dict(kwargs)
        tensor = SimpleNamespace(mean=lambda: SimpleNamespace(item=lambda: 0.5))
        return tensor, tensor, tensor

    monkeypatch.setitem(sys.modules, "bert_score", SimpleNamespace(score=fake_bert_score))
    monkeypatch.setattr(
        "saver_v3.metrics.semantic_metrics._collect_summary_strings",
        lambda rollouts, reference_data: (["pred"], ["ref"]),
    )

    summary = _evaluate_bertscore(
        [{"video_id": "video-1"}],
        reference_data=SimpleNamespace(),
        bertscore_model_path="/models/roberta-large",
    )

    assert summary["available"] is True
    assert summary["model_type"] == "/models/roberta-large"
    assert captured["kwargs"]["model_type"] == "/models/roberta-large"
    assert captured["kwargs"]["lang"] == "en"


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


def test_construct_vllm_llm_with_max_model_len_retry():
    calls = []

    def fake_llm_ctor(**kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise ValueError(
                "To serve at least one request ... estimated maximum model length is 12256."
            )
        return {"ok": True, "max_model_len": kwargs["max_model_len"]}

    runtime = SimpleNamespace()
    result = _construct_vllm_llm_with_max_model_len_retry(
        llm_ctor=fake_llm_ctor,
        llm_kwargs={"max_model_len": 16384, "model": "dummy"},
        runtime=runtime,
        context_label="unit_test_vllm_retry",
    )

    assert result == {"ok": True, "max_model_len": 12256}
    assert [call["max_model_len"] for call in calls] == [16384, 12256]
