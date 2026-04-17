import sys
import unittest
from unittest import mock

import train_saver_rl

from saver_v3.cli import train_rl_ds, train_sft_ds
from saver_v3.rl import cli_shared


class DeepSpeedCLIArgTests(unittest.TestCase):
    def test_train_sft_cli_accepts_local_rank_argument(self) -> None:
        argv = [
            "train_sft_ds.py",
            "--local_rank=3",
            "--config",
            "cfg.yaml",
            "--model-config",
            "model.yaml",
            "--attention-config",
            "attn.yaml",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = train_sft_ds.parse_args()
        self.assertEqual(args.local_rank, 3)

    def test_train_rl_cli_accepts_hyphenated_local_rank_argument(self) -> None:
        argv = [
            "train_rl_ds.py",
            "--local-rank",
            "5",
            "--config",
            "cfg.yaml",
            "--model-config",
            "model.yaml",
            "--attention-config",
            "attn.yaml",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = train_rl_ds.parse_args()
        self.assertEqual(args.local_rank, 5)

    def test_active_rl_shared_parser_accepts_timesearch_v3_reward_version(self) -> None:
        args = cli_shared.parse_active_rl_args(
            [
                "--output-dir",
                "/tmp/out",
                "--rl-reward-version",
                "timesearch_v3",
            ],
            description="test",
        )
        self.assertEqual(args.rl_reward_version, "timesearch_v3")

    def test_legacy_train_rl_parser_rejects_removed_replay_flag(self) -> None:
        with self.assertRaisesRegex(SystemExit, "removed from active RL"):
            train_saver_rl.parse_args(
                [
                    "--output-dir",
                    "/tmp/out",
                    "--rl-replay-buffer-enable",
                    "true",
                ]
            )


if __name__ == "__main__":
    unittest.main()
