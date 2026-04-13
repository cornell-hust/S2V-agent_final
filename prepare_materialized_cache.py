from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from data_utils.jsonl import iter_jsonl, write_jsonl
from saver_v3.cli.common import load_json_mapping, load_yaml_mapping, write_json
from saver_v3.data.config import SaverAgentConfig, saver_config_from_mapping
from saver_v3.data.materialized_cache import (
    MATERIALIZED_RUNTIME_ITEMS_FORMAT,
    MATERIALIZED_SFT_MESSAGES_FORMAT,
    build_jsonl_provenance,
    build_runtime_materialized_rows,
    build_sft_materialized_rows,
    write_materialized_cache_metadata,
)
from saver_v3.data.prepared_loader import iter_prepared_rows


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Phase A materialized cache JSONL files.")
    parser.add_argument("--mode", choices=("sft", "runtime"), required=True)
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--include-splits", default="", help="Optional comma-separated split filter.")
    parser.add_argument("--config", default="", help="Optional saver config YAML/JSON.")
    parser.add_argument("--model-config", default="", help="Optional model-config YAML/JSON stored in metadata.")
    parser.add_argument("--data-root", default="", help="Optional runtime data root for path resolution.")
    parser.add_argument("--proposal-model-path", default="", help="Optional proposal model path for SFT replay.")
    parser.add_argument("--proposal-torch-dtype", default="auto")
    parser.add_argument("--proposal-device", default="")
    parser.add_argument("--overwrite-existing", action="store_true")
    return parser.parse_args(argv)


def _load_mapping(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    resolved = Path(path)
    if resolved.suffix.lower() == ".json":
        return load_json_mapping(resolved)
    return load_yaml_mapping(resolved)


def _load_saver_config(path: str | Path | None) -> SaverAgentConfig:
    if not path:
        return SaverAgentConfig()
    mapping = _load_mapping(path)
    saver_mapping: Mapping[str, Any]
    if isinstance(mapping.get("saver_config"), Mapping):
        saver_mapping = mapping.get("saver_config") or {}
    else:
        saver_mapping = mapping
    return saver_config_from_mapping(
        saver_mapping,
        saver_config_source=mapping.get("saver_config_source"),
        source_anchor=path,
    )


def _load_model_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    return _load_mapping(path)


def _build_proposal_runtime(args: argparse.Namespace) -> Any:
    if not args.proposal_model_path:
        return None
    from saver_v3.core.proposal import SiglipFeatureEncoder

    device = str(args.proposal_device or "").strip()
    if not device:
        try:
            import torch
        except Exception:
            device = "cpu"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    return SiglipFeatureEncoder.from_pretrained(
        args.proposal_model_path,
        torch_dtype=args.proposal_torch_dtype,
        device=device,
    )


def _iter_selected_prepared_rows(input_path: str | Path, include_splits: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_prepared_rows(input_path):
        if include_splits and str(row.get("split") or "") not in include_splits:
            continue
        rows.append(row)
    return rows


def _iter_selected_runtime_rows(input_path: str | Path, include_splits: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(input_path, skip_invalid_lines=False):
        if include_splits and str(row.get("split") or "") not in include_splits:
            continue
        rows.append(dict(row))
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if output_path.exists() and not args.overwrite_existing:
        raise FileExistsError(f"Refusing to overwrite existing output without --overwrite-existing: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    include_splits = {split for split in (item.strip() for item in str(args.include_splits or "").split(",")) if split}
    saver_config = _load_saver_config(args.config)
    model_config = _load_model_config(args.model_config)

    if args.mode == "sft":
        rows = _iter_selected_prepared_rows(input_path, include_splits)
        proposal_runtime = _build_proposal_runtime(args)
        materialized_rows = build_sft_materialized_rows(
            rows,
            config=saver_config,
            proposal_runtime=proposal_runtime,
            strict=False,
        )
        materialized_format = MATERIALIZED_SFT_MESSAGES_FORMAT
    else:
        rows = _iter_selected_runtime_rows(input_path, include_splits)
        materialized_rows = build_runtime_materialized_rows(
            rows,
            config=saver_config,
            data_root=args.data_root,
        )
        materialized_format = MATERIALIZED_RUNTIME_ITEMS_FORMAT

    write_jsonl(materialized_rows, output_path)
    metadata_path = write_materialized_cache_metadata(
        output_path,
        materialized_format=materialized_format,
        config=saver_config,
        model_config=model_config,
        extra_fields={
            "mode": args.mode,
            "num_records": int(len(materialized_rows)),
            "include_splits": sorted(include_splits),
            "source_jsonl": build_jsonl_provenance(input_path, include_splits=sorted(include_splits)),
            "input_path": str(input_path),
            "data_root": str(args.data_root or ""),
            "proposal_model_path": str(args.proposal_model_path or ""),
            "proposal_torch_dtype": str(args.proposal_torch_dtype or ""),
            "proposal_device": str(args.proposal_device or ""),
        },
    )
    summary = {
        "mode": args.mode,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "num_records": int(len(materialized_rows)),
        "materialized_format": materialized_format,
    }
    write_json(summary, output_path.with_suffix(output_path.suffix + ".summary.json"))
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
