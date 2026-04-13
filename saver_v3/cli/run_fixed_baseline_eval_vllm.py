from __future__ import annotations

import argparse
import json

from saver_v3.cli.common import apply_config_overrides, load_yaml_mapping
from saver_v3.inference.fixed_baseline_eval import FixedBaselineEvalConfig, run_fixed_baseline_eval_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run direct fixed-observation Qwen3-VL baseline evaluation with local-rank HF inference under torchrun.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = load_yaml_mapping(args.config)
    if args.override:
        mapping = apply_config_overrides(mapping, args.override)
    config = FixedBaselineEvalConfig.from_mapping(mapping)
    result = run_fixed_baseline_eval_job(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
