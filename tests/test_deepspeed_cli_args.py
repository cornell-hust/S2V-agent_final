import sys
import unittest
from unittest import mock

from saver_v3.cli import train_rl_ds, train_sft_ds


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


if __name__ == "__main__":
    unittest.main()
