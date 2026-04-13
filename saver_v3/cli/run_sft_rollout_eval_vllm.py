from __future__ import annotations

from saver_v3.cli._suppress_warnings import suppress_third_party_warnings
suppress_third_party_warnings()

import argparse
import json

from saver_v3.cli.common import apply_config_overrides, load_yaml_mapping
from saver_v3.inference.rollout_eval import StepRolloutEvalConfig, run_step_rollout_eval_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full-rollout SFT evaluation with local-rank vLLM policy inference.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Optional dotted.key=value override applied to the rollout-eval YAML config before launch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StepRolloutEvalConfig.from_mapping(apply_config_overrides(load_yaml_mapping(args.config), args.override))
    result = run_step_rollout_eval_job(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
