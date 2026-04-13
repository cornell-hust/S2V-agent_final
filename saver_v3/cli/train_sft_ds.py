from __future__ import annotations

from saver_v3.cli._suppress_warnings import suppress_third_party_warnings
suppress_third_party_warnings()

import argparse
import json

from saver_v3.sft.runtime import SFTJobConfig, run_sft_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3-VL-8B full-model SFT under DeepSpeed/torchrun.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--attention-config", required=True)
    parser.add_argument("--deepspeed-config", default="")
    parser.add_argument("--local_rank", "--local-rank", dest="local_rank", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Optional dotted.key=value override applied to the main YAML config before launch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job = SFTJobConfig.from_files(
        config_path=args.config,
        model_config_path=args.model_config,
        attention_config_path=args.attention_config,
        deepspeed_config_path=args.deepspeed_config or None,
        config_overrides=args.override,
    )
    result = run_sft_job(job)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
