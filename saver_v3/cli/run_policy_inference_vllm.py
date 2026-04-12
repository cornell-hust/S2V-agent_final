from __future__ import annotations

import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL policy inference with local-rank vLLM engines.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    raise SystemExit(
        "run_policy_inference_vllm.py has been retired because it used the broken lightweight step-message path. "
        "Use saver_v3/cli/run_policy_rollout_vllm.py so inference goes through the official raw-SAVER rollout stack."
    )


if __name__ == "__main__":
    main()
