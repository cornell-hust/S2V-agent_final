#!/usr/bin/env python3
from __future__ import annotations

from typing import List, Optional

from saver_v3.rl import cli_shared


def parse_args(argv: Optional[List[str]] = None):
    return cli_shared.parse_active_rl_args(
        argv,
        description="Deprecated legacy RL entrypoint. Use saver_v3.cli.train_rl_ds instead.",
    )


def main() -> None:
    raise RuntimeError(
        "idea2_v3 RL has converged on saver_v3.cli.train_rl_ds -> train_saver_rl_trl.py; "
        "the legacy native GRPO backend is deprecated and unsupported. "
        "Use saver_v3.cli.train_rl_ds or scripts/train_rl_qwen3_vl_8b_ds8.sh instead."
    )


if __name__ == "__main__":
    main()
