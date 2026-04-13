from __future__ import annotations

import argparse
import json

from saver_v3.cli.common import apply_config_overrides, load_yaml_mapping
from saver_v3.data.prepare_sft_manifest import PrepareSFTManifestConfig, run_prepare_sft_manifest_job


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare compact_trace_v2 SFT manifest from raw SAVER JSONL.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PrepareSFTManifestConfig.from_mapping(
        apply_config_overrides(load_yaml_mapping(args.config), args.override),
        source_anchor=args.config,
    )
    result = run_prepare_sft_manifest_job(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
