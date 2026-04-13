import sys
import tempfile
import types
import unittest

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import saver_v3.sft.training as training
from saver_v3.data.prepared_metadata import PREPARED_SFT_FORMAT


class _FakeSFTConfig:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = dict(kwargs)
        self.kwargs = dict(kwargs)


class _FakeSFTTrainer:
    instances = []
    next_world_process_zero = False

    def __init__(self, *args, **kwargs):
        self.model = kwargs.get("model")
        self.args = kwargs.get("args")
        self.train_dataset = kwargs.get("train_dataset")
        self.data_collator = kwargs.get("data_collator")
        self.processing_class = kwargs.get("processing_class")
        self.callbacks = kwargs.get("callbacks")
        self.state = SimpleNamespace(log_history=[], global_step=0, epoch=0.0, max_steps=0, num_train_epochs=0)
        self.resume_from_checkpoint = None
        self.saved_model_path = None
        self.world_process_zero = bool(type(self).next_world_process_zero)
        type(self).instances.append(self)

    def train(self, resume_from_checkpoint=None):
        self.resume_from_checkpoint = resume_from_checkpoint
        return SimpleNamespace(training_loss=0.125)

    def is_world_process_zero(self):
        return bool(self.world_process_zero)

    def save_model(self, output_dir):
        self.saved_model_path = str(output_dir)


class _FakeMaterializedMessagesSFTDataset:
    init_kwargs = None

    def __init__(self, materialized_messages_path, include_splits=None, config=None):
        type(self).init_kwargs = {
            "materialized_messages_path": materialized_messages_path,
            "include_splits": include_splits,
            "config": config,
        }
        self.rows = [
            {
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "question"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
                ],
                "sample_weight": 1.0,
                "video_id": "video-1",
            }
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class SFTTrainingPhaseBTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeSFTConfig.last_kwargs = None
        _FakeSFTTrainer.instances = []
        _FakeSFTTrainer.next_world_process_zero = False
        _FakeMaterializedMessagesSFTDataset.init_kwargs = None

    def _fake_runtime(self):
        return SimpleNamespace(is_main_process=False, world_size=1, local_rank=0)

    def test_run_standard_sft_uses_materialized_messages_dataset_when_available(self) -> None:
        fake_trl = types.ModuleType("trl")
        fake_trl.SFTConfig = _FakeSFTConfig
        fake_trl.SFTTrainer = _FakeSFTTrainer

        fake_materialized_cache = types.ModuleType("saver_v3.data.materialized_cache")
        fake_materialized_cache.MaterializedMessagesSFTDataset = _FakeMaterializedMessagesSFTDataset

        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch.dict(
            sys.modules,
            {
                "trl": fake_trl,
                "saver_v3.data.materialized_cache": fake_materialized_cache,
            },
        ), mock.patch.object(
            training, "distributed_runtime_from_env", return_value=self._fake_runtime()
        ), mock.patch.object(
            training, "distributed_barrier"
        ), mock.patch.object(
            training, "load_qwen_model_and_processor", return_value=(object(), SimpleNamespace())
        ), mock.patch.object(
            training, "ensure_materialized_cache_metadata", return_value={"materialized_format": "materialized_sft_messages_v1"}
        ) as metadata_mock, mock.patch.object(
            training, "runtime_log"
        ) as runtime_log_mock, mock.patch.object(
            training, "_load_prepared_jsonl_rows", side_effect=AssertionError("legacy loader should not run in materialized mode")
        ), mock.patch.object(
            training, "_load_training_proposal_runtime", side_effect=AssertionError("proposal runtime should not load in materialized mode")
        ):
            result = training.run_standard_sft(
                prepared_data_path="/tmp/prepared.compact_trace_v2.jsonl",
                materialized_messages_path="/tmp/materialized.messages.jsonl",
                require_materialized_cache=True,
                include_splits="train",
                model_path="/models/qwen3-vl-8b-Instruct",
                output_dir=Path(tmp_dir) / "out",
                saver_config={"source": "test-config"},
            )

        metadata_mock.assert_called_once()
        self.assertEqual(metadata_mock.call_args.args[0], "/tmp/materialized.messages.jsonl")
        self.assertEqual(
            _FakeMaterializedMessagesSFTDataset.init_kwargs["materialized_messages_path"],
            "/tmp/materialized.messages.jsonl",
        )
        self.assertEqual(_FakeMaterializedMessagesSFTDataset.init_kwargs["include_splits"], "train")
        self.assertIsInstance(_FakeSFTTrainer.instances[-1].train_dataset, _FakeMaterializedMessagesSFTDataset)
        self.assertEqual(result["num_examples"], 1)
        logged_messages = [str(call.args[0]) for call in runtime_log_mock.call_args_list if call.args]
        self.assertTrue(any("using materialized SFT messages cache" in message for message in logged_messages))

    def test_run_standard_sft_forces_workers_off_for_cuda_proposal_runtime(self) -> None:
        fake_trl = types.ModuleType("trl")
        fake_trl.SFTConfig = _FakeSFTConfig
        fake_trl.SFTTrainer = _FakeSFTTrainer
        examples = [
            {
                "prepared_format": PREPARED_SFT_FORMAT,
                "video_id": "video-1",
                "video_path": "/tmp/video.mp4",
                "split": "train",
                "oracle_trajectory": [{"tool": "seek_evidence", "sample_weight": 1.0}],
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch.dict(sys.modules, {"trl": fake_trl}), mock.patch.object(
            training, "distributed_runtime_from_env", return_value=self._fake_runtime()
        ), mock.patch.object(
            training, "distributed_barrier"
        ), mock.patch.object(
            training, "load_qwen_model_and_processor", return_value=(object(), SimpleNamespace())
        ), mock.patch.object(
            training, "_load_prepared_jsonl_rows", return_value=examples
        ), mock.patch.object(
            training, "summarize_example_frame_cache_status", return_value={"num_missing_frame_cache": 0}
        ), mock.patch.object(
            training, "summarize_example_feature_cache_status", return_value={"num_missing_feature_cache": 0}
        ), mock.patch.object(
            training, "format_example_frame_cache_status", return_value="frame-cache-ok"
        ), mock.patch.object(
            training, "format_example_feature_cache_status", return_value="feature-cache-ok"
        ), mock.patch.object(
            training, "_load_training_proposal_runtime", return_value=object()
        ) as proposal_runtime_mock, mock.patch.object(
            training, "runtime_log"
        ) as runtime_log_mock:
            result = training.run_standard_sft(
                prepared_data_path="/tmp/prepared.compact_trace_v2.jsonl",
                include_splits="train",
                model_path="/models/qwen3-vl-8b-Instruct",
                output_dir=Path(tmp_dir) / "out",
                proposal_model_path="/models/proposal",
                proposal_device="cuda:0",
                dataloader_num_workers=4,
                dataloader_prefetch_factor=8,
                dataloader_persistent_workers=True,
            )

        proposal_runtime_mock.assert_called_once()
        self.assertEqual(_FakeSFTConfig.last_kwargs["dataloader_num_workers"], 0)
        self.assertFalse(_FakeSFTConfig.last_kwargs["dataloader_persistent_workers"])
        self.assertNotIn("dataloader_prefetch_factor", _FakeSFTConfig.last_kwargs)
        self.assertEqual(result["num_examples"], 1)
        logged_messages = [str(call.args[0]) for call in runtime_log_mock.call_args_list if call.args]
        self.assertTrue(any("forcing SFT dataloader workers off" in message for message in logged_messages))

    def test_run_standard_sft_logs_train_return_save_and_barrier_stages(self) -> None:
        fake_trl = types.ModuleType("trl")
        fake_trl.SFTConfig = _FakeSFTConfig
        fake_trl.SFTTrainer = _FakeSFTTrainer

        fake_materialized_cache = types.ModuleType("saver_v3.data.materialized_cache")
        fake_materialized_cache.MaterializedMessagesSFTDataset = _FakeMaterializedMessagesSFTDataset

        fake_processor = mock.Mock()
        fake_runtime = SimpleNamespace(is_main_process=True, world_size=1, local_rank=0)
        _FakeSFTTrainer.next_world_process_zero = True

        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch.dict(
            sys.modules,
            {
                "trl": fake_trl,
                "saver_v3.data.materialized_cache": fake_materialized_cache,
            },
        ), mock.patch.object(
            training, "distributed_runtime_from_env", return_value=fake_runtime
        ), mock.patch.object(
            training, "distributed_barrier"
        ) as barrier_mock, mock.patch.object(
            training, "load_qwen_model_and_processor", return_value=(object(), fake_processor)
        ), mock.patch.object(
            training, "ensure_materialized_cache_metadata", return_value={"materialized_format": "materialized_sft_messages_v1"}
        ), mock.patch.object(
            training, "runtime_log"
        ) as runtime_log_mock:
            result = training.run_standard_sft(
                prepared_data_path="/tmp/prepared.compact_trace_v2.jsonl",
                materialized_messages_path="/tmp/materialized.messages.jsonl",
                require_materialized_cache=True,
                include_splits="train",
                model_path="/models/qwen3-vl-8b-Instruct",
                output_dir=Path(tmp_dir) / "out",
                saver_config={"source": "test-config"},
            )

        logged_messages = [str(call.args[0]) for call in runtime_log_mock.call_args_list if call.args]
        self.assertTrue(any("SFTTrainer.train() returned" in message for message in logged_messages))
        self.assertTrue(any("saving final SFT model to" in message for message in logged_messages))
        self.assertTrue(any("final SFT model save complete" in message for message in logged_messages))
        self.assertTrue(any("entering final SFT distributed barrier" in message for message in logged_messages))
        self.assertTrue(any("passed final SFT distributed barrier" in message for message in logged_messages))
        barrier_mock.assert_called_once()
        self.assertEqual(result["num_examples"], 1)

    def test_run_standard_sft_reuses_latest_checkpoint_instead_of_top_level_full_save(self) -> None:
        fake_trl = types.ModuleType("trl")
        fake_trl.SFTConfig = _FakeSFTConfig
        fake_trl.SFTTrainer = _FakeSFTTrainer

        fake_materialized_cache = types.ModuleType("saver_v3.data.materialized_cache")
        fake_materialized_cache.MaterializedMessagesSFTDataset = _FakeMaterializedMessagesSFTDataset

        fake_processor = mock.Mock()
        fake_runtime = SimpleNamespace(is_main_process=True, world_size=1, local_rank=0)
        _FakeSFTTrainer.next_world_process_zero = True

        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch.dict(
            sys.modules,
            {
                "trl": fake_trl,
                "saver_v3.data.materialized_cache": fake_materialized_cache,
            },
        ), mock.patch.object(
            training, "distributed_runtime_from_env", return_value=fake_runtime
        ), mock.patch.object(
            training, "distributed_barrier"
        ), mock.patch.object(
            training, "load_qwen_model_and_processor", return_value=(object(), fake_processor)
        ), mock.patch.object(
            training, "ensure_materialized_cache_metadata", return_value={"materialized_format": "materialized_sft_messages_v1"}
        ), mock.patch.object(
            training, "runtime_log"
        ) as runtime_log_mock:
            output_dir = Path(tmp_dir) / "out"
            checkpoint_dir = output_dir / "checkpoint-6"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
            (checkpoint_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            result = training.run_standard_sft(
                prepared_data_path="/tmp/prepared.compact_trace_v2.jsonl",
                materialized_messages_path="/tmp/materialized.messages.jsonl",
                require_materialized_cache=True,
                include_splits="train",
                model_path="/models/qwen3-vl-8b-Instruct",
                output_dir=output_dir,
                saver_config={"source": "test-config"},
            )

        trainer = _FakeSFTTrainer.instances[-1]
        self.assertIsNone(trainer.saved_model_path)
        self.assertEqual(result["latest_checkpoint"], str(checkpoint_dir))
        self.assertEqual(result["authoritative_model_path"], str(checkpoint_dir))
        logged_messages = [str(call.args[0]) for call in runtime_log_mock.call_args_list if call.args]
        self.assertTrue(any("reusing latest SFT checkpoint instead of saving duplicate top-level full model" in message for message in logged_messages))

    def test_save_sft_epoch_resume_checkpoint_reuses_latest_trainer_checkpoint(self) -> None:
        model = mock.Mock()
        processor = mock.Mock()
        runtime = SimpleNamespace(is_main_process=True, world_size=1, local_rank=0, rank=0)

        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch.object(
            training, "distributed_barrier"
        ) as barrier_mock, mock.patch.object(
            training, "runtime_log"
        ) as runtime_log_mock:
            output_dir = Path(tmp_dir) / "out"
            checkpoint_dir = output_dir / "checkpoint-6"
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "config.json").write_text("{}", encoding="utf-8")
            (checkpoint_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

            resolved = training.save_sft_epoch_resume_checkpoint(
                model=model,
                processor=processor,
                output_dir=output_dir,
                epoch_index=1,
                runtime=runtime,
            )

        self.assertEqual(resolved, checkpoint_dir.resolve())
        self.assertFalse(model.save_pretrained.called)
        self.assertFalse(processor.save_pretrained.called)
        barrier_mock.assert_called_once()
        logged_messages = [str(call.args[0]) for call in runtime_log_mock.call_args_list if call.args]
        self.assertTrue(any("reusing latest trainer checkpoint for epoch-end rollout-eval recovery" in message for message in logged_messages))


if __name__ == "__main__":
    unittest.main()
