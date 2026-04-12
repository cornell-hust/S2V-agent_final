import tempfile
import unittest
from pathlib import Path
from unittest import mock

from saver_v3.sft.runtime import SFTJobConfig, run_sft_job


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
            )

            with mock.patch("saver_v3.sft.runtime.ensure_fa3_training_ready") as ensure_mock, mock.patch(
                "saver_v3.sft.runtime.run_standard_sft",
                return_value={"num_examples": 7, "train_loss": 0.25, "output_dir": str(output_dir)},
            ) as standard_sft_mock:
                result = run_sft_job(job)

            ensure_mock.assert_called_once_with(require_gpu=True)
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
            self.assertEqual(result.num_train_examples, 7)
            self.assertTrue((output_dir / "sft_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
