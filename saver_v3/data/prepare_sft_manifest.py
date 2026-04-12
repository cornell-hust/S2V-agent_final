from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from data_utils.jsonl import iter_jsonl, write_jsonl
from split_utils import parse_include_splits

from saver_v3.data.config import PreviewConfig, SaverAgentConfig
from saver_v3.data.dataset import SaverRecordItemBuilder
from saver_v3.data.prepared_metadata import write_prepared_sft_metadata
from saver_v3.data.training_data import build_compact_trace_sft_record
from saver_v3.cli.common import load_yaml_mapping, write_json
from saver_v3.common import distributed_runtime_from_env, runtime_log


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return value or {}


@dataclass
class PrepareSFTManifestConfig:
    input_data_path: str
    output_path: str
    data_root: str = ""
    include_splits: str = ""
    num_preview_frames: int = 8
    preview_sampling_fps: float | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PrepareSFTManifestConfig":
        io_cfg = dict(_mapping(mapping.get("io")))
        preview_cfg = dict(_mapping(mapping.get("preview")))
        return cls(
            input_data_path=str(io_cfg.get("input_data_path") or "").strip(),
            output_path=str(io_cfg.get("output_path") or "").strip(),
            data_root=str(io_cfg.get("data_root") or "").strip(),
            include_splits=str(io_cfg.get("include_splits") or "").strip(),
            num_preview_frames=int(preview_cfg.get("num_preview_frames", 8) or 8),
            preview_sampling_fps=(
                None
                if preview_cfg.get("preview_sampling_fps") is None
                else float(preview_cfg.get("preview_sampling_fps") or 0.0)
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PrepareSFTManifestConfig":
        return cls.from_mapping(load_yaml_mapping(path))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_prepare_sft_manifest_job(config: PrepareSFTManifestConfig) -> Dict[str, Any]:
    if not config.input_data_path:
        raise ValueError("prepare_sft_manifest requires input_data_path")
    if not config.output_path:
        raise ValueError("prepare_sft_manifest requires output_path")

    runtime = distributed_runtime_from_env()
    saver_config = SaverAgentConfig(
        preview=PreviewConfig(
            num_preview_frames=int(config.num_preview_frames),
            preview_sampling_fps=config.preview_sampling_fps,
            max_preview_frames=int(config.num_preview_frames),
        )
    )
    record_builder = SaverRecordItemBuilder(
        data_root=config.data_root,
        config=saver_config,
        require_frame_cache=False,
        require_feature_cache=False,
        load_frame_cache=False,
        load_feature_cache=False,
    )
    include_splits = set(parse_include_splits(config.include_splits) or [])
    prepared_rows: list[dict[str, Any]] = []

    runtime_log(
        (
            "prepare_sft_manifest startup: "
            f"input={config.input_data_path} output={config.output_path} "
            f"include_splits={sorted(include_splits) if include_splits else 'all'}"
        ),
        runtime=runtime,
        main_process_only=True,
    )

    for row in iter_jsonl(config.input_data_path, skip_invalid_lines=False):
        if include_splits and str(row.get("split") or "") not in include_splits:
            continue
        item = record_builder.build_item(dict(row))
        prepared_rows.append(build_compact_trace_sft_record(item=item, record=dict(row), config=saver_config))

    output_path = Path(config.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(prepared_rows, output_path)
    metadata_path = write_prepared_sft_metadata(
        output_path,
        config=saver_config,
        extra_fields={
            "input_data_path": str(config.input_data_path),
            "num_records": int(len(prepared_rows)),
            "include_splits": sorted(include_splits),
        },
    )
    summary = {
        "input_data_path": str(config.input_data_path),
        "output_path": str(output_path),
        "metadata_path": str(metadata_path),
        "num_records": int(len(prepared_rows)),
        "prepared_format": "compact_trace_v2",
    }
    write_json(summary, output_path.with_suffix(output_path.suffix + ".summary.json"))
    return summary
