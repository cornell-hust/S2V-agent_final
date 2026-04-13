from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence

from split_utils import parse_include_splits

from saver_v3.data.config import PreviewConfig, PromptConfig, RolloutTraceConfig, SaverAgentConfig


PREPARED_SFT_METADATA_SCHEMA_VERSION = 3
PREPARED_SFT_FORMAT = "compact_trace_v2"
PREPARED_SFT_PROVENANCE_MODE = "hybrid"


def prepared_sft_metadata_path(prepared_data_path: str | Path) -> Path:
    return Path(str(prepared_data_path) + ".meta.json")


def _normalize_include_splits(include_splits: str | Sequence[str] | None) -> list[str]:
    return sorted({str(item).strip() for item in parse_include_splits(include_splits) or [] if str(item).strip()})


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_jsonl_provenance(jsonl_path: str | Path, *, include_splits: str | Sequence[str] | None = None) -> Dict[str, Any]:
    resolved_path = Path(jsonl_path).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Cannot build provenance for missing JSONL source: {resolved_path}")

    normalized_splits = _normalize_include_splits(include_splits)
    include_split_set = set(normalized_splits)
    hasher = hashlib.sha256()
    num_records = 0
    with resolved_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {resolved_path}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected dict JSONL row on line {line_number} of {resolved_path}")
            if include_split_set and str(payload.get("split") or "").strip() not in include_split_set:
                continue
            hasher.update(_canonical_json_bytes(payload))
            hasher.update(b"\n")
            num_records += 1

    stat = resolved_path.stat()
    return {
        "path": str(jsonl_path),
        "resolved_path": str(resolved_path),
        "include_splits": normalized_splits,
        "file_size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "num_records": int(num_records),
        "sha256": hasher.hexdigest(),
        "validation_mode": PREPARED_SFT_PROVENANCE_MODE,
    }


def build_prepared_sft_metadata(*, config: SaverAgentConfig) -> Dict[str, Any]:
    snapshot = config.to_dict()
    return {
        "schema_version": int(PREPARED_SFT_METADATA_SCHEMA_VERSION),
        "prepared_format": str(PREPARED_SFT_FORMAT),
        "preview": snapshot.get("preview", {}),
        "prompt": snapshot.get("prompt", {}),
        "rollout_trace": snapshot.get("rollout_trace", {}),
    }


def config_from_prepared_sft_metadata(metadata: Dict[str, Any]) -> SaverAgentConfig:
    payload = dict(metadata or {})
    preview_payload = payload.get("preview") or {}
    prompt_payload = payload.get("prompt") or {}
    rollout_trace_payload = payload.get("rollout_trace") or {}
    return SaverAgentConfig(
        preview=PreviewConfig(**dict(preview_payload)),
        prompt=PromptConfig(**dict(prompt_payload)),
        rollout_trace=RolloutTraceConfig(**dict(rollout_trace_payload)),
    )


def load_prepared_sft_metadata(prepared_data_path: str | Path) -> Dict[str, Any]:
    metadata_path = prepared_sft_metadata_path(prepared_data_path)
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_prepared_sft_metadata(
    prepared_data_path: str | Path,
    *,
    config: SaverAgentConfig,
    extra_fields: Dict[str, Any] | None = None,
) -> Path:
    metadata = build_prepared_sft_metadata(config=config)
    if extra_fields:
        metadata.update(dict(extra_fields))
    metadata_path = prepared_sft_metadata_path(prepared_data_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def _validate_provenance_block(
    *,
    metadata: Dict[str, Any],
    field_name: str,
    expected_path: str | Path | None = None,
    expected_include_splits: str | Sequence[str] | None = None,
) -> None:
    block = metadata.get(field_name)
    if not isinstance(block, dict) or not block:
        raise ValueError(
            f"Prepared SFT metadata is missing required `{field_name}` provenance. Regenerate the prepared JSONL before continuing."
        )

    recorded_splits = _normalize_include_splits(block.get("include_splits"))
    expected_splits = _normalize_include_splits(expected_include_splits)
    if expected_include_splits is not None and recorded_splits != expected_splits:
        raise ValueError(
            f"Prepared SFT metadata `{field_name}.include_splits` mismatch: found {recorded_splits}, expected {expected_splits}. "
            "Regenerate the prepared JSONL before continuing."
        )

    recorded_resolved_path = str(block.get("resolved_path") or "").strip()
    if expected_path is not None:
        expected_resolved_path = str(Path(expected_path).expanduser().resolve())
        if recorded_resolved_path and recorded_resolved_path != expected_resolved_path:
            raise ValueError(
                f"Prepared SFT metadata `{field_name}.resolved_path` mismatch: found {recorded_resolved_path}, "
                f"expected {expected_resolved_path}. Regenerate the prepared JSONL before continuing."
            )
        actual_path = Path(expected_resolved_path)
    else:
        actual_path = Path(recorded_resolved_path or str(block.get("path") or "")).expanduser().resolve()

    if not actual_path.exists():
        raise ValueError(
            f"Prepared SFT metadata `{field_name}` points to a missing file: {actual_path}. "
            "Regenerate the prepared JSONL before continuing."
        )

    stat = actual_path.stat()
    recorded_size = int(block.get("file_size", -1) or -1)
    recorded_mtime_ns = int(block.get("mtime_ns", -1) or -1)
    if int(stat.st_size) == recorded_size and int(stat.st_mtime_ns) == recorded_mtime_ns:
        return

    rebuilt = build_jsonl_provenance(actual_path, include_splits=recorded_splits)
    if rebuilt.get("sha256") != str(block.get("sha256") or "") or int(rebuilt.get("num_records", -1)) != int(block.get("num_records", -1) or -1):
        raise ValueError(
            f"Prepared SFT metadata `{field_name}` is stale relative to {actual_path}. "
            "Regenerate the prepared JSONL before continuing."
        )


def ensure_prepared_sft_metadata(
    prepared_data_path: str | Path,
    *,
    config: SaverAgentConfig | None = None,
    require_config_match: bool = False,
    expected_source_runtime_path: str | Path | None = None,
    expected_source_runtime_include_splits: str | Sequence[str] | None = None,
    require_source_runtime: bool = False,
    expected_source_prepared_path: str | Path | None = None,
    expected_source_prepared_include_splits: str | Sequence[str] | None = None,
    require_source_prepared: bool = False,
    require_teacher_annotated: bool = False,
    require_teacher_rollout_primary_materialized: bool = False,
) -> Dict[str, Any]:
    metadata_path = prepared_sft_metadata_path(prepared_data_path)
    metadata = load_prepared_sft_metadata(prepared_data_path)
    if not metadata:
        raise ValueError(
            f"Prepared SFT metadata is missing or unreadable: {metadata_path}. "
            "Regenerate the prepared JSONL before continuing."
        )

    schema_version = int(metadata.get("schema_version", 0) or 0)
    if schema_version != int(PREPARED_SFT_METADATA_SCHEMA_VERSION):
        raise ValueError(
            f"Prepared SFT metadata schema mismatch for {prepared_data_path}: "
            f"found {schema_version}, expected {PREPARED_SFT_METADATA_SCHEMA_VERSION}. "
            "Regenerate the prepared JSONL before continuing."
        )
    prepared_format = str(metadata.get("prepared_format") or "").strip()
    if prepared_format != str(PREPARED_SFT_FORMAT):
        raise ValueError(
            f"Prepared SFT metadata format mismatch for {prepared_data_path}: "
            f"found {prepared_format or '(missing)'}, expected {PREPARED_SFT_FORMAT}. "
            "Regenerate the prepared JSONL before continuing."
        )

    if require_config_match and config is not None:
        expected = build_prepared_sft_metadata(config=config)
        if (
            metadata.get("preview") != expected.get("preview")
            or metadata.get("prompt") != expected.get("prompt")
            or metadata.get("rollout_trace") != expected.get("rollout_trace")
        ):
            raise ValueError(
                f"Prepared SFT metadata does not match the current preview/prompt/rollout_trace config for {prepared_data_path}. "
                "Regenerate the prepared JSONL with the same preview/prompt/rollout_trace settings used for training and evaluation."
            )

    if require_source_runtime or expected_source_runtime_path is not None:
        _validate_provenance_block(
            metadata=metadata,
            field_name="source_runtime",
            expected_path=expected_source_runtime_path,
            expected_include_splits=expected_source_runtime_include_splits,
        )
    if require_source_prepared or expected_source_prepared_path is not None:
        _validate_provenance_block(
            metadata=metadata,
            field_name="source_prepared",
            expected_path=expected_source_prepared_path,
            expected_include_splits=expected_source_prepared_include_splits,
        )
    if require_teacher_annotated and not bool(metadata.get("teacher_annotated")):
        raise ValueError(
            f"Prepared SFT metadata for {prepared_data_path} must set teacher_annotated=true. "
            "Regenerate the teacher prepared JSONL before continuing."
        )
    if require_teacher_rollout_primary_materialized and not bool(metadata.get("teacher_rollout_primary_materialized")):
        raise ValueError(
            f"Prepared SFT metadata for {prepared_data_path} must set teacher_rollout_primary_materialized=true. "
            "Regenerate the teacher prepared JSONL before continuing."
        )
    return metadata
