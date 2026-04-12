from __future__ import annotations

import argparse
import json

from saver_v3.inference.rollout_eval import StepRolloutEvalConfig, run_step_rollout_eval_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-rollout SFT evaluation with local-rank vLLM policy inference.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StepRolloutEvalConfig.from_yaml(args.config)
    result = run_step_rollout_eval_job(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
