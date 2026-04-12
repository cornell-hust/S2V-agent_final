from __future__ import annotations

import argparse
import json

from saver_v3.data.prepare_sft_manifest import PrepareSFTManifestConfig, run_prepare_sft_manifest_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare compact_trace_v2 SFT manifest from raw SAVER JSONL.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PrepareSFTManifestConfig.from_yaml(args.config)
    result = run_prepare_sft_manifest_job(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
