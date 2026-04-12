import unittest

from saver_v3.inference.policy_rollout import PolicyRolloutConfig
from saver_v3.inference.rollout_eval import StepRolloutEvalConfig


class VllmEntrypointConfigTests(unittest.TestCase):
    def test_policy_rollout_config_requires_raw_data_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "io.data_path"):
            PolicyRolloutConfig.from_mapping(
                {
                    "base_model": "/models/qwen3-vl-8b-Instruct",
                    "io": {"input_manifest": "/tmp/legacy.jsonl", "output_path": "/tmp/out.jsonl"},
                }
            )

    def test_policy_rollout_config_parses_raw_data_contract(self) -> None:
        config = PolicyRolloutConfig.from_mapping(
            {
                "base_model": "/models/qwen3-vl-8b-Instruct",
                "server": {"mode": "colocate", "tensor_parallel_size": 1},
                "client": {"max_tokens": 384},
                "io": {
                    "data_path": "/tmp/raw_eval.jsonl",
                    "output_path": "/tmp/out.jsonl",
                    "include_splits": "val",
                    "count": 12,
                },
            }
        )

        self.assertEqual(config.data_path, "/tmp/raw_eval.jsonl")
        self.assertEqual(config.output_path, "/tmp/out.jsonl")
        self.assertEqual(config.include_splits, "val")
        self.assertEqual(config.count, 12)
        self.assertEqual(config.max_new_tokens, 384)

    def test_rollout_eval_config_requires_raw_data_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "io.data_path"):
            StepRolloutEvalConfig.from_mapping(
                {
                    "base_model": "/models/qwen3-vl-8b-Instruct",
                    "io": {"input_manifest": "/tmp/legacy.jsonl", "output_dir": "/tmp/eval"},
                }
            )

    def test_rollout_eval_config_parses_raw_data_contract(self) -> None:
        config = StepRolloutEvalConfig.from_mapping(
            {
                "base_model": "/models/qwen3-vl-8b-Instruct",
                "client": {"max_tokens": 256, "max_total_images": 24},
                "io": {
                    "data_path": "/tmp/raw_eval.jsonl",
                    "output_dir": "/tmp/eval",
                    "include_splits": "val",
                    "max_records": 8,
                },
                "evaluation": {"epoch_index": 3, "max_turns": 10},
            }
        )

        self.assertEqual(config.data_path, "/tmp/raw_eval.jsonl")
        self.assertEqual(config.output_dir, "/tmp/eval")
        self.assertEqual(config.max_records, 8)
        self.assertEqual(config.epoch_index, 3)
        self.assertEqual(config.max_turns, 10)
        self.assertEqual(config.policy_max_new_tokens, 256)
        self.assertEqual(config.max_total_images, 24)


if __name__ == "__main__":
    unittest.main()
