from __future__ import annotations

import argparse
import json

from saver_v3.inference.policy_rollout import PolicyRolloutConfig, run_policy_rollout_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-rollout policy inference with the official raw SAVER + vLLM stack.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PolicyRolloutConfig.from_yaml(args.config)
    result = run_policy_rollout_job(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
