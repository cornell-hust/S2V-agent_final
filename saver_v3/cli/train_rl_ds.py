from __future__ import annotations

import argparse
import json

from saver_v3.rl.runtime import RLJobConfig, run_rl_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAVER RL with the TRL + vLLM GRPO route under DeepSpeed/torchrun.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--attention-config", required=True)
    parser.add_argument("--deepspeed-config", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job = RLJobConfig.from_files(
        config_path=args.config,
        model_config_path=args.model_config,
        attention_config_path=args.attention_config,
        deepspeed_config_path=args.deepspeed_config or None,
    )
    result = run_rl_job(job)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
