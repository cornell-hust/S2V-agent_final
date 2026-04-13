import tempfile
import unittest

import yaml
from pathlib import Path
from unittest import mock

from saver_v3.sft.runtime import SFTJobConfig, run_sft_job


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class SFTRuntimeTests(unittest.TestCase):
    def test_run_sft_job_delegates_to_standard_episode_sft(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "sft"
            job = SFTJobConfig(
                run_name="sft-standard-wrapper",
                output_dir=str(output_dir),
                prepared_data_path="/tmp/prepared.compact_trace_v2.jsonl",
                include_splits="train",
                num_workers=2,
                dataloader_prefetch_factor=4,
                dataloader_persistent_workers=True,
                epochs=2.0,
                learning_rate=1e-5,
                weight_decay=0.01,
                warmup_ratio=0.03,
                lr_scheduler_type="cosine",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                max_grad_norm=1.0,
                log_every_n_steps=10,
                save_every_n_steps=100,
                save_total_limit=2,
                report_to=["wandb"],
                seed=123,
                ddp_find_unused_parameters=False,
                bf16=True,
                fp16=False,
                model_path="/models/qwen3-vl-8b-Instruct",
                torch_dtype="bfloat16",
                gradient_checkpointing=True,
                attn_implementation="flash_attention_3",
                max_seq_length=8192,
                max_total_images=28,
                max_image_side=640,
                max_image_pixels=0,
                keep_recent_text_messages=20,
                keep_recent_tool_image_messages=0,
                trust_remote_code=True,
                train_mode="full",
                deepspeed_config_path="/tmp/zero3.json",
                use_sample_weights=True,
            )

            with mock.patch("saver_v3.sft.runtime.ensure_fa3_training_ready") as ensure_mock, mock.patch(
                "saver_v3.sft.runtime.ensure_prepared_sft_metadata",
                return_value={"schema_version": 3, "prepared_format": "compact_trace_v2"},
            ) as metadata_mock, mock.patch(
                "saver_v3.sft.runtime.run_standard_sft",
                return_value={"num_examples": 7, "train_loss": 0.25, "output_dir": str(output_dir)},
            ) as standard_sft_mock:
                result = run_sft_job(job)

            ensure_mock.assert_called_once_with(require_gpu=True)
            metadata_mock.assert_called_once()
            metadata_kwargs = metadata_mock.call_args.kwargs
            self.assertEqual(metadata_mock.call_args.args[0], "/tmp/prepared.compact_trace_v2.jsonl")
            self.assertTrue(metadata_kwargs["require_config_match"])
            self.assertIsNotNone(metadata_kwargs["config"])
            standard_sft_mock.assert_called_once()
            kwargs = standard_sft_mock.call_args.kwargs
            self.assertEqual(kwargs["prepared_data_path"], "/tmp/prepared.compact_trace_v2.jsonl")
            self.assertEqual(kwargs["include_splits"], "train")
            self.assertEqual(kwargs["model_path"], "/models/qwen3-vl-8b-Instruct")
            self.assertFalse(kwargs["use_lora"])
            self.assertEqual(kwargs["attn_implementation"], "flash_attention_3")
            self.assertEqual(kwargs["deepspeed"], "/tmp/zero3.json")
            self.assertEqual(kwargs["lr_scheduler_type"], "cosine")
            self.assertEqual(kwargs["max_seq_length"], 8192)
            self.assertTrue(kwargs["use_sample_weights"])
            self.assertIsNotNone(kwargs["saver_config"])
            self.assertEqual(result.num_train_examples, 7)
            self.assertTrue((output_dir / "sft_summary.json").exists())

    def test_from_files_preserves_prepared_saver_config_and_sample_weight_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "sft.yaml"
            model_config_path = tmp_path / "model.yaml"
            attention_config_path = tmp_path / "attention.yaml"

            _write_yaml(attention_config_path, {"policy_name": "fa3_only"})
            _write_yaml(
                model_config_path,
                {
                    "base_model": "/models/qwen3-vl-8b-Instruct",
                    "torch_dtype": "bfloat16",
                    "attn_implementation": "flash_attention_3",
                    "gradient_checkpointing": True,
                    "train_mode": "full",
                    "sequence": {"max_length": 8192},
                    "vision": {"max_images_per_sample": 28, "max_image_side": 640},
                },
            )
            _write_yaml(
                config_path,
                {
                    "run_name": "sft-config-test",
                    "output_dir": str(tmp_path / "out"),
                    "data": {"prepared_data_path": "/data/sft_train.compact_trace_v2.jsonl"},
                    "optimization": {"use_sample_weights": True},
                    "preview": {"num_preview_frames": 12, "preview_sampling_fps": 1.5},
                    "prompt": {
                        "initial_user_template": "Case: {public_case_id}",
                        "preview_instruction": "Inspect preview.",
                        "tool_response_template": "Use tools.",
                    },
                    "rollout_trace": {
                        "record_observation_content": False,
                        "record_state_deltas": True,
                        "record_message_history": False,
                    },
                },
            )

            job = SFTJobConfig.from_files(
                config_path=str(config_path),
                model_config_path=str(model_config_path),
                attention_config_path=str(attention_config_path),
            )

            self.assertTrue(job.use_sample_weights)
            self.assertEqual(job.saver_config["preview"]["num_preview_frames"], 12)
            self.assertEqual(job.saver_config["preview"]["preview_sampling_fps"], 1.5)
            self.assertEqual(job.saver_config["prompt"]["preview_instruction"], "Inspect preview.")
            self.assertFalse(job.saver_config["rollout_trace"]["record_message_history"])

    def test_from_files_allows_disabling_sample_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "sft.yaml"
            model_config_path = tmp_path / "model.yaml"
            attention_config_path = tmp_path / "attention.yaml"

            _write_yaml(attention_config_path, {"policy_name": "fa3_only"})
            _write_yaml(
                model_config_path,
                {
                    "base_model": "/models/qwen3-vl-8b-Instruct",
                    "torch_dtype": "bfloat16",
                    "attn_implementation": "flash_attention_3",
                    "gradient_checkpointing": True,
                    "train_mode": "full",
                    "sequence": {"max_length": 8192},
                    "vision": {"max_images_per_sample": 28, "max_image_side": 640},
                },
            )
            _write_yaml(
                config_path,
                {
                    "run_name": "sft-config-test-no-weights",
                    "output_dir": str(tmp_path / "out"),
                    "data": {"prepared_data_path": "/data/sft_train.compact_trace_v2.jsonl"},
                    "optimization": {"use_sample_weights": False},
                },
            )

            job = SFTJobConfig.from_files(
                config_path=str(config_path),
                model_config_path=str(model_config_path),
                attention_config_path=str(attention_config_path),
            )

            self.assertFalse(job.use_sample_weights)

    def test_from_files_parses_materialized_cache_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "sft.yaml"
            model_config_path = tmp_path / "model.yaml"
            attention_config_path = tmp_path / "attention.yaml"

            _write_yaml(attention_config_path, {"policy_name": "fa3_only"})
            _write_yaml(
                model_config_path,
                {
                    "base_model": "/models/qwen3-vl-8b-Instruct",
                    "torch_dtype": "bfloat16",
                    "attn_implementation": "flash_attention_3",
                    "gradient_checkpointing": True,
                    "train_mode": "full",
                    "sequence": {"max_length": 8192},
                    "vision": {"max_images_per_sample": 28, "max_image_side": 640},
                },
            )
            _write_yaml(
                config_path,
                {
                    "run_name": "sft-config-test-materialized",
                    "output_dir": str(tmp_path / "out"),
                    "data": {
                        "prepared_data_path": "/data/sft_train.compact_trace_v2.jsonl",
                        "materialized_messages_path": "/data/sft_train.materialized.jsonl",
                        "require_materialized_cache": True,
                    },
                },
            )

            job = SFTJobConfig.from_files(
                config_path=str(config_path),
                model_config_path=str(model_config_path),
                attention_config_path=str(attention_config_path),
            )

            self.assertEqual(job.materialized_messages_path, "/data/sft_train.materialized.jsonl")
            self.assertTrue(job.require_materialized_cache)

    def test_run_sft_job_validates_and_passes_materialized_cache_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "sft"
            job = SFTJobConfig(
                run_name="sft-materialized-wrapper",
                output_dir=str(output_dir),
                prepared_data_path="/tmp/prepared.compact_trace_v2.jsonl",
                include_splits="train",
                num_workers=0,
                dataloader_prefetch_factor=0,
                dataloader_persistent_workers=False,
                epochs=1.0,
                learning_rate=1e-5,
                weight_decay=0.0,
                warmup_ratio=0.0,
                lr_scheduler_type="cosine",
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                max_grad_norm=1.0,
                log_every_n_steps=10,
                save_every_n_steps=100,
                save_total_limit=2,
                report_to=[],
                seed=123,
                ddp_find_unused_parameters=False,
                bf16=True,
                fp16=False,
                model_path="/models/qwen3-vl-8b-Instruct",
                torch_dtype="bfloat16",
                gradient_checkpointing=True,
                attn_implementation="flash_attention_3",
                max_seq_length=8192,
                max_total_images=28,
                max_image_side=640,
                max_image_pixels=0,
                keep_recent_text_messages=20,
                keep_recent_tool_image_messages=0,
                trust_remote_code=True,
                train_mode="full",
                materialized_messages_path="/tmp/materialized.messages.jsonl",
                require_materialized_cache=True,
            )

            with mock.patch("saver_v3.sft.runtime.ensure_fa3_training_ready") as ensure_mock, mock.patch(
                "saver_v3.sft.runtime.ensure_prepared_sft_metadata",
                return_value={"schema_version": 3, "prepared_format": "compact_trace_v2"},
            ) as metadata_mock, mock.patch(
                "saver_v3.sft.runtime.ensure_materialized_cache_metadata",
                return_value={"schema_version": 1, "materialized_format": "materialized_sft_messages_v1"},
            ) as materialized_metadata_mock, mock.patch(
                "saver_v3.sft.runtime.run_standard_sft",
                return_value={"num_examples": 3, "train_loss": 0.1, "output_dir": str(output_dir)},
            ) as standard_sft_mock:
                result = run_sft_job(job)

            ensure_mock.assert_called_once_with(require_gpu=True)
            metadata_mock.assert_called_once()
            self.assertEqual(metadata_mock.call_args.args[0], "/tmp/prepared.compact_trace_v2.jsonl")
            materialized_metadata_mock.assert_called_once()
            self.assertEqual(materialized_metadata_mock.call_args.args[0], "/tmp/materialized.messages.jsonl")
            self.assertEqual(materialized_metadata_mock.call_args.kwargs["expected_source_path"], "/tmp/prepared.compact_trace_v2.jsonl")
            kwargs = standard_sft_mock.call_args.kwargs
            self.assertEqual(kwargs["materialized_messages_path"], "/tmp/materialized.messages.jsonl")
            self.assertTrue(kwargs["require_materialized_cache"])
            self.assertEqual(result.num_train_examples, 3)


if __name__ == "__main__":
    unittest.main()
