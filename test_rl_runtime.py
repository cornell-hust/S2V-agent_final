import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

import train_saver_rl

from saver_v3.rl.runtime import RLJobConfig, build_legacy_rl_trl_argv, run_rl_job
from saver_v3.core.rollout import _build_episode_training_feature
from saver_v3.sft.training import _build_rl_completion_episode_spec_from_feature
from saver_v3.model.qwen_policy import load_auto_processor_with_compat


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class RLRuntimeTests(unittest.TestCase):
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
                {"_rl_token_trace": {"prompt_ids": torch.tensor([[1, 2]]), "prompt_mask": torch.tensor([[1, 1]]), "completion_ids": torch.tensor([[3, 4]]), "completion_mask": torch.tensor([[1, 1]])}},
                {"_rl_token_trace": {"prompt_ids": torch.tensor([[1, 2]]), "prompt_mask": torch.tensor([[1, 1]]), "completion_ids": torch.tensor([[5, 6]]), "completion_mask": torch.tensor([[1, 1]])}},
            ],
        }
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "<tool_call>a</tool_call>"}]},
            {"role": "tool", "content": [{"type": "text", "text": "obs"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "<answer>b</answer>"}]},
        ]
        feature = _build_episode_training_feature(result=result, messages=messages)
        self.assertIsNotNone(feature)
        self.assertIn("episode_prompt_trace", feature)
        self.assertIn("episode_assistant_traces", feature)
        self.assertEqual(len(feature["episode_assistant_traces"]), 2)

    def test_episode_feature_builds_episode_batch(self) -> None:
        processor = load_auto_processor_with_compat("/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/qwen3-vl-8b-Instruct")
        feature = {
            "episode_prompt_trace": {
                "prompt_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "prompt_mask": torch.tensor([[1, 1]], dtype=torch.long),
            },
            "episode_assistant_traces": [
                {
                    "completion_ids": torch.tensor([[3, 4]], dtype=torch.long),
                    "completion_mask": torch.tensor([[1, 1]], dtype=torch.bool),
                },
                {
                    "completion_ids": torch.tensor([[5, 6]], dtype=torch.long),
                    "completion_mask": torch.tensor([[1, 1]], dtype=torch.bool),
                },
            ],
            "advantages": 1.0,
        }
        result = _build_rl_completion_episode_spec_from_feature(processor, feature, max_seq_length=4096)
        self.assertIsNotNone(result.batch)
        self.assertIn("prompt_ids", result.batch)
        self.assertIn("completion_ids", result.batch)
        self.assertIn("completion_mask", result.batch)
        self.assertIn("old_policy_token_log_probs", result.batch)
        self.assertIn("advantages", result.batch)
        self.assertTrue(result.batch["completion_mask"].any().item())

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
                    "reference_model": "/models/qwen3-vl-8b-Instruct",
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
                        "rl_steps_per_generation": 4,
                        "rollout_stage_batch_size": 12,
                        "fecv_stage_batch_size": 10,
                        "rollout_count": 6,
                        "num_generations": 3,
                        "rollout_max_turns": 9,
                    },
                    "distributed": {"bf16": True, "fp16": False},
                    "rewards": {
                        "reward_version": "timesearch_v2",
                        "accuracy_reward_weight": 1.0,
                        "fecv_evidence_faithfulness_reward_weight": 0.5,
                        "open_ended_judge_enabled": True,
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
            self.assertEqual(job.rollout_stage_batch_size, 12)
            self.assertEqual(job.fecv_stage_batch_size, 10)
            self.assertEqual(job.vllm_guided_decoding_regex, "^ok$")
            self.assertEqual(job.deepspeed_config_path, str(deepspeed_config_path))
            self.assertEqual(job.reward_config["weights"]["accuracy_reward"], 1.0)
            self.assertEqual(job.reward_config["weights"]["fecv_evidence_faithfulness_reward"], 0.5)
            self.assertTrue(job.reward_config["open_ended_judge_enabled"])


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

    def test_build_legacy_rl_trl_argv_includes_required_vllm_and_deepspeed_flags(self) -> None:
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
            reference_model="/tmp/reference",
            base_model="/tmp/base",
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_3",
            gradient_checkpointing=True,
            rollout_backend="vllm",
            rollout_config="/tmp/rollout.yaml",
            deepspeed_config_path="/tmp/zero3.json",
            reward_config={"weights": {"temporal_miou_weight": 1.0}, "open_ended_judge_enabled": False},
            num_iterations=7,
            num_train_epochs=1.0,
            learning_rate=5e-7,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            rl_steps_per_generation=4,
            rollout_count=8,
            num_generations=4,
            rollout_max_turns=12,
            vllm_mode="colocate",
            vllm_tensor_parallel_size=1,
            vllm_gpu_memory_utilization=0.9,
            vllm_max_num_seqs=4,
            vllm_fallback_max_num_seqs=2,
            vllm_guided_decoding_regex="^json$",
        )

        argv = build_legacy_rl_trl_argv(job)
        argv_pairs = list(zip(argv[::2], argv[1::2]))

        self.assertIn(("--model-path", "/tmp/policy"), argv_pairs)
        self.assertIn(("--reference-model-path", "/tmp/reference"), argv_pairs)
        self.assertIn(("--deepspeed", "/tmp/zero3.json"), argv_pairs)
        self.assertIn(("--vllm-mode", "colocate"), argv_pairs)
        self.assertIn(("--vllm-guided-decoding-regex", "^json$"), argv_pairs)
        self.assertIn(("--rl-steps-per-generation", "4"), argv_pairs)
        self.assertIn(("--vllm-max-num-seqs", "4"), argv_pairs)
        self.assertIn(("--vllm-fallback-max-num-seqs", "2"), argv_pairs)
        reward_json = dict(argv_pairs)["--rl-reward-config-json"]
        self.assertEqual(json.loads(reward_json)["weights"]["temporal_miou_weight"], 1.0)
        self.assertIn("--gradient-checkpointing", argv)
        self.assertIn("--bf16", argv)

    def test_build_legacy_rl_trl_argv_defaults_open_ended_judge_to_true(self) -> None:
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

        argv = build_legacy_rl_trl_argv(job)
        argv_pairs = dict(zip(argv[::2], argv[1::2]))

        self.assertEqual(argv_pairs["--rl-open-ended-judge-enabled"], "true")
        self.assertEqual(json.loads(argv_pairs["--rl-reward-config-json"])["weights"]["temporal_miou_weight"], 1.0)

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
                reference_model="/tmp/reference",
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
            self.assertEqual(argv_pairs["--vllm-max-num-seqs"], "4")
            self.assertEqual(argv_pairs["--vllm-fallback-max-num-seqs"], "2")

    def test_train_saver_rl_parses_rl_reward_config_json(self) -> None:
        args = train_saver_rl.parse_args(
            [
                "--output-dir",
                "/tmp/out",
                "--data",
                "/tmp/train.jsonl",
                "--model-path",
                "/tmp/policy",
                "--rl-reward-config-json",
                json.dumps({"weights": {"accuracy_reward": 1.25}}),
            ]
        )

        self.assertEqual(args.rl_reward_config["weights"]["accuracy_reward"], 1.25)

    def test_train_saver_rl_rejects_lora_mode_in_v3(self) -> None:
        with self.assertRaisesRegex(ValueError, "full-model RL only"):
            train_saver_rl.parse_args(
                [
                    "--output-dir",
                    "/tmp/out",
                    "--data",
                    "/tmp/train.jsonl",
                    "--model-path",
                    "/tmp/policy",
                    "--lora",
                ]
            )

    def test_train_saver_rl_rejects_adapter_only_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            adapter_dir = Path(tmp_dir) / "adapter_ckpt"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "adapter-only checkpoint"):
                train_saver_rl.parse_args(
                    [
                        "--output-dir",
                        "/tmp/out",
                        "--data",
                        "/tmp/train.jsonl",
                        "--model-path",
                        str(adapter_dir),
                    ]
                )

    def test_run_policy_inference_vllm_is_deprecated(self) -> None:
        from saver_v3.cli import run_policy_inference_vllm

        with mock.patch("sys.argv", ["run_policy_inference_vllm", "--config", "/tmp/unused.yaml"]):
            with self.assertRaisesRegex(SystemExit, "run_policy_rollout_vllm.py"):
                run_policy_inference_vllm.main()

    def test_train_saver_rl_trl_parser_rejects_server_mode(self) -> None:
        import train_saver_rl_trl

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
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

    def test_build_legacy_rl_trl_argv_includes_materialized_runtime_cache_flags(self) -> None:
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
            reference_model="/tmp/reference",
            base_model="/tmp/base",
            torch_dtype="bfloat16",
            attn_implementation="flash_attention_3",
            gradient_checkpointing=True,
            rollout_backend="vllm",
            rollout_config="/tmp/rollout.yaml",
            deepspeed_config_path="/tmp/zero3.json",
        )
        argv = build_legacy_rl_trl_argv(job)
        argv_pairs = dict(zip(argv[::2], argv[1::2]))
        self.assertEqual(argv_pairs["--materialized-train-items-path"], "/tmp/train.materialized.jsonl")
        self.assertEqual(argv_pairs["--materialized-eval-items-path"], "/tmp/eval.materialized.jsonl")
        self.assertEqual(argv_pairs["--require-materialized-runtime-cache"], "true")


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
