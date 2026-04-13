from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from split_utils import parse_include_splits

from saver_v3.data.prepared_schema import validate_compact_trace_row


MATERIALIZED_CACHE_METADATA_SCHEMA_VERSION = 1
MATERIALIZED_SFT_MESSAGES_FORMAT = "materialized_sft_messages_v1"
MATERIALIZED_RUNTIME_ITEMS_FORMAT = "materialized_runtime_items_v1"


class MaterializedCacheError(ValueError):
    """Raised when materialized cache rows or sidecars violate the Phase A contract."""


replay_compact_trace_messages = None
SaverRecordItemBuilder = None


def _normalize_include_splits(include_splits: str | Sequence[str] | None) -> list[str]:
    return sorted({str(item).strip() for item in parse_include_splits(include_splits) or [] if str(item).strip()})


def canonical_json_bytes(payload: Any) -> bytes:
    """Return stable UTF-8 JSON bytes for provenance hashes."""

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_json_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def materialized_cache_metadata_path(cache_path: str | Path) -> Path:
    return Path(str(cache_path) + ".meta.json")


def build_jsonl_provenance(
    jsonl_path: str | Path,
    *,
    include_splits: str | Sequence[str] | None = None,
) -> dict[str, Any]:
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
            hasher.update(canonical_json_bytes(payload))
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
        "validation_mode": "canonical_jsonl",
    }


def _config_snapshot(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "to_dict"):
        payload = config.to_dict()
        return copy.deepcopy(payload) if isinstance(payload, dict) else {}
    if isinstance(config, Mapping):
        return copy.deepcopy(dict(config))
    return {}


def build_materialized_cache_metadata(
    *,
    materialized_format: str,
    config: Any = None,
    model_config: Mapping[str, Any] | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_version": int(MATERIALIZED_CACHE_METADATA_SCHEMA_VERSION),
        "materialized_format": str(materialized_format),
    }
    config_payload = _config_snapshot(config)
    if config_payload:
        metadata["config"] = config_payload
    if model_config:
        metadata["model_config"] = copy.deepcopy(dict(model_config))
    if extra_fields:
        metadata.update(copy.deepcopy(dict(extra_fields)))
    return metadata


def load_materialized_cache_metadata(cache_path: str | Path) -> dict[str, Any]:
    metadata_path = materialized_cache_metadata_path(cache_path)
    if not metadata_path.exists():
        return {}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_materialized_cache_metadata(
    cache_path: str | Path,
    *,
    materialized_format: str,
    config: Any = None,
    model_config: Mapping[str, Any] | None = None,
    source_path: str | Path | None = None,
    include_splits: str | Sequence[str] | None = None,
    source_field_name: str = "source_jsonl",
    extra_fields: Mapping[str, Any] | None = None,
) -> Path:
    merged_extra_fields = dict(extra_fields or {})
    if source_path is not None:
        merged_extra_fields[source_field_name] = build_jsonl_provenance(
            source_path,
            include_splits=include_splits,
        )
    metadata = build_materialized_cache_metadata(
        materialized_format=materialized_format,
        config=config,
        model_config=model_config,
        extra_fields=merged_extra_fields,
    )
    metadata_path = materialized_cache_metadata_path(cache_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def _validate_jsonl_provenance_block(
    *,
    metadata: Mapping[str, Any],
    field_name: str,
    expected_path: str | Path | None = None,
    expected_include_splits: str | Sequence[str] | None = None,
) -> None:
    block = metadata.get(field_name)
    if not isinstance(block, Mapping) or not block:
        raise ValueError(f"Materialized cache metadata is missing required `{field_name}` provenance.")

    recorded_splits = _normalize_include_splits(block.get("include_splits"))
    expected_splits = _normalize_include_splits(expected_include_splits)
    if expected_include_splits is not None and recorded_splits != expected_splits:
        raise ValueError(
            f"Materialized cache metadata `{field_name}.include_splits` mismatch: "
            f"found {recorded_splits}, expected {expected_splits}."
        )

    recorded_resolved_path = str(block.get("resolved_path") or "").strip()
    if expected_path is not None:
        actual_path = Path(expected_path).expanduser().resolve()
        if recorded_resolved_path and recorded_resolved_path != str(actual_path):
            raise ValueError(
                f"Materialized cache metadata `{field_name}.resolved_path` mismatch: "
                f"found {recorded_resolved_path}, expected {actual_path}."
            )
    else:
        actual_path = Path(recorded_resolved_path or str(block.get("path") or "")).expanduser().resolve()

    if not actual_path.exists():
        raise ValueError(f"Materialized cache metadata `{field_name}` points to a missing file: {actual_path}.")

    rebuilt = build_jsonl_provenance(actual_path, include_splits=recorded_splits)
    if rebuilt.get("sha256") != str(block.get("sha256") or "") or int(rebuilt.get("num_records", -1)) != int(
        block.get("num_records", -1) or -1
    ):
        raise ValueError(f"Materialized cache metadata `{field_name}` is stale relative to {actual_path}.")


def ensure_materialized_cache_metadata(
    cache_path: str | Path,
    *,
    expected_format: str | None = None,
    expected_source_path: str | Path | None = None,
    expected_include_splits: str | Sequence[str] | None = None,
    require_source: bool = False,
    source_field_name: str = "source_jsonl",
) -> dict[str, Any]:
    metadata_path = materialized_cache_metadata_path(cache_path)
    metadata = load_materialized_cache_metadata(cache_path)
    if not metadata:
        raise ValueError(f"Materialized cache metadata is missing or unreadable: {metadata_path}.")

    schema_version = int(metadata.get("schema_version", 0) or 0)
    if schema_version != int(MATERIALIZED_CACHE_METADATA_SCHEMA_VERSION):
        raise ValueError(
            f"Materialized cache metadata schema mismatch for {cache_path}: "
            f"found {schema_version}, expected {MATERIALIZED_CACHE_METADATA_SCHEMA_VERSION}."
        )
    materialized_format = str(metadata.get("materialized_format") or "").strip()
    if expected_format is not None and materialized_format != str(expected_format):
        raise ValueError(
            f"Materialized cache metadata format mismatch for {cache_path}: "
            f"found {materialized_format or '(missing)'}, expected {expected_format}."
        )

    if require_source or expected_source_path is not None:
        _validate_jsonl_provenance_block(
            metadata=metadata,
            field_name=source_field_name,
            expected_path=expected_source_path,
            expected_include_splits=expected_include_splits,
        )
    return dict(metadata)


def _require_format(row: Mapping[str, Any], expected_format: str) -> None:
    materialized_format = str(row.get("materialized_format") or row.get("prepared_format") or "").strip()
    if materialized_format != expected_format:
        raise MaterializedCacheError(
            f"Unsupported materialized_format `{materialized_format or '(missing)'}`. Expected `{expected_format}`."
        )


def _validate_messages(messages: Any, *, field_name: str = "messages") -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise MaterializedCacheError(f"Materialized row is missing required non-empty list field `{field_name}`.")
    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            raise MaterializedCacheError(f"{field_name}[{index}] must be a JSON object.")
        role = str(message.get("role") or "").strip()
        if not role:
            raise MaterializedCacheError(f"{field_name}[{index}] is missing required field `role`.")
        if "content" not in message:
            raise MaterializedCacheError(f"{field_name}[{index}] is missing required field `content`.")
        normalized.append(copy.deepcopy(message))
    return normalized


def _coerce_finite_float(value: Any, *, field_name: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise MaterializedCacheError(f"`{field_name}` must be numeric.") from exc
    if not math.isfinite(result):
        raise MaterializedCacheError(f"`{field_name}` must be finite.")
    return result


def validate_materialized_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise MaterializedCacheError("Materialized SFT rows must be JSON objects.")
    _require_format(row, MATERIALIZED_SFT_MESSAGES_FORMAT)

    normalized = copy.deepcopy(row)
    normalized["schema_version"] = int(
        normalized.get("schema_version", MATERIALIZED_CACHE_METADATA_SCHEMA_VERSION) or MATERIALIZED_CACHE_METADATA_SCHEMA_VERSION
    )
    normalized["materialized_format"] = MATERIALIZED_SFT_MESSAGES_FORMAT
    normalized.pop("prepared_format", None)
    normalized["messages"] = _validate_messages(normalized.get("messages"))
    normalized["sample_weight"] = _coerce_finite_float(normalized.get("sample_weight", 1.0), field_name="sample_weight")
    return normalized


def validate_materialized_runtime_item_row(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise MaterializedCacheError("Materialized runtime item rows must be JSON objects.")
    _require_format(row, MATERIALIZED_RUNTIME_ITEMS_FORMAT)

    normalized = copy.deepcopy(row)
    normalized["schema_version"] = int(
        normalized.get("schema_version", MATERIALIZED_CACHE_METADATA_SCHEMA_VERSION) or MATERIALIZED_CACHE_METADATA_SCHEMA_VERSION
    )
    normalized["materialized_format"] = MATERIALIZED_RUNTIME_ITEMS_FORMAT
    normalized.pop("prepared_format", None)
    if not isinstance(normalized.get("record"), dict):
        raise MaterializedCacheError("Materialized runtime row is missing required object field `record`.")
    if not isinstance(normalized.get("multimodal_cache"), dict):
        raise MaterializedCacheError("Materialized runtime row is missing required object field `multimodal_cache`.")
    normalized["messages"] = _validate_messages(normalized.get("messages"))
    return normalized


def _iter_validated_jsonl_rows(
    path: str | Path,
    *,
    validator: Any,
) -> Iterator[dict[str, Any]]:
    resolved_path = Path(path)
    with resolved_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MaterializedCacheError(f"Invalid JSON on line {line_number} of {resolved_path}: {exc}") from exc
            try:
                yield validator(payload)
            except MaterializedCacheError as exc:
                raise MaterializedCacheError(f"Invalid materialized row on line {line_number} of {resolved_path}: {exc}") from exc


def iter_materialized_sft_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    return _iter_validated_jsonl_rows(path, validator=validate_materialized_sft_row)


def load_materialized_sft_rows(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_materialized_sft_rows(path))


def iter_materialized_runtime_item_rows(path: str | Path) -> Iterator[dict[str, Any]]:
    return _iter_validated_jsonl_rows(path, validator=validate_materialized_runtime_item_row)


def load_materialized_runtime_item_rows(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_materialized_runtime_item_rows(path))


def _replay_compact_trace_messages(
    row: dict[str, Any],
    *,
    config: Any = None,
    proposal_runtime: Any = None,
    strict: bool = False,
) -> list[dict[str, Any]]:
    replay_fn = replay_compact_trace_messages
    if replay_fn is None:
        from saver_v3.sft.training import replay_compact_trace_messages as replay_fn

    return replay_fn(
        row,
        config=config,
        proposal_runtime=proposal_runtime,
        strict_feature_guided_proposal=bool(strict),
    )


def _episode_sample_weight(row: Mapping[str, Any]) -> float:
    if row.get("sample_weight") is not None:
        return _coerce_finite_float(row.get("sample_weight"), field_name="sample_weight")
    steps = list(row.get("oracle_trajectory") or [])
    step_weights = [float(step.get("sample_weight", 1.0)) for step in steps if isinstance(step, Mapping) and step.get("sample_weight") is not None]
    if not step_weights:
        return 1.0
    return float(sum(step_weights) / len(step_weights))


def build_sft_materialized_rows(
    rows: Iterable[dict[str, Any]],
    *,
    config: Any = None,
    proposal_runtime: Any = None,
    strict: bool = False,
) -> list[dict[str, Any]]:
    materialized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized = validate_compact_trace_row(row)
        messages = _replay_compact_trace_messages(
            normalized,
            config=config,
            proposal_runtime=proposal_runtime,
            strict=bool(strict),
        )
        materialized = {
            "schema_version": int(MATERIALIZED_CACHE_METADATA_SCHEMA_VERSION),
            "materialized_format": MATERIALIZED_SFT_MESSAGES_FORMAT,
            "source_prepared_format": normalized.get("prepared_format"),
            "source_row_sha256": canonical_json_hash(normalized),
            "video_id": normalized.get("video_id"),
            "split": normalized.get("split"),
            "source": normalized.get("source"),
            "messages": _sanitize_messages(
                messages,
                video_path=str(normalized.get("video_path") or ""),
            ),
            "sample_weight": _episode_sample_weight(normalized),
        }
        materialized_rows.append(validate_materialized_sft_row(materialized))
    return materialized_rows


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None or dtype is not None:
        shape_list = [int(dim) for dim in list(shape or [])]
        return {"tensor_ref": None, "shape": shape_list, "dtype": str(dtype or "")}
    try:
        json.dumps(value)
    except TypeError:
        return {"python_object_ref": type(value).__name__, "repr": repr(value)}
    return value


def _image_ref_from_item(item: Mapping[str, Any], *, video_path: str) -> dict[str, Any]:
    image_ref = copy.deepcopy(dict(item.get("image_ref") or {}))
    if video_path and not str(image_ref.get("video_path") or "").strip():
        image_ref["video_path"] = video_path
    if item.get("sampled_frame_index") is not None and image_ref.get("sampled_frame_index") is None:
        image_ref["sampled_frame_index"] = int(item["sampled_frame_index"])
    if item.get("raw_frame_index") is not None and image_ref.get("raw_frame_index") is None:
        image_ref["raw_frame_index"] = int(item["raw_frame_index"])
    if item.get("timestamp_sec") is not None and image_ref.get("timestamp_sec") is None:
        image_ref["timestamp_sec"] = float(item["timestamp_sec"])
    return image_ref


def _sanitize_message_item(item: Any, *, video_path: str) -> Any:
    if not isinstance(item, Mapping):
        return _json_safe(item)
    item_type = str(item.get("type") or "").strip()
    if item_type == "image" and ("image" in item or "image_ref" in item):
        sanitized_item = {
            key: _json_safe(value)
            for key, value in dict(item).items()
            if key not in {"image", "sampled_frame_index", "raw_frame_index", "timestamp_sec", "image_ref"}
        }
        sanitized_item["type"] = "image"
        sanitized_item["image_ref"] = _image_ref_from_item(item, video_path=video_path)
        return sanitized_item
    return {str(key): _json_safe(value) for key, value in dict(item).items()}


def _sanitize_messages(messages: Any, *, video_path: str) -> list[dict[str, Any]]:
    sanitized_messages: list[dict[str, Any]] = []
    for message in list(messages or []):
        if not isinstance(message, Mapping):
            raise MaterializedCacheError("Materialized runtime messages must be JSON objects.")
        sanitized_message = {
            key: _json_safe(value)
            for key, value in dict(message).items()
            if key != "content"
        }
        sanitized_message["content"] = [
            _sanitize_message_item(item, video_path=video_path)
            for item in list(message.get("content") or [])
        ]
        sanitized_messages.append(sanitized_message)
    return sanitized_messages


def _sanitize_multimodal_cache(cache: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = {str(key): _json_safe(value) for key, value in dict(cache).items()}
    for heavy_key in ("video", "embedding", "preview_frames", "proposal_runtime"):
        if heavy_key in sanitized:
            sanitized[heavy_key] = None
    video_path = str(sanitized.get("video_path") or "").strip()
    if video_path:
        sanitized.setdefault("frame_cache_path", str(Path(str(video_path) + ".frame_cache")))
        sanitized.setdefault("feature_cache_path", str(Path(str(video_path) + ".feature_cache")))
    return sanitized


def build_runtime_materialized_rows(
    rows: Iterable[dict[str, Any]],
    *,
    config: Any = None,
    data_root: str | Path | None = None,
    strict: bool = False,
) -> list[dict[str, Any]]:
    record_builder_cls = SaverRecordItemBuilder
    if record_builder_cls is None:
        from saver_v3.data.dataset import SaverRecordItemBuilder as record_builder_cls

    record_builder = record_builder_cls(
        data_root=data_root or "",
        config=copy.deepcopy(config) if config is not None else None,
        require_frame_cache=False,
        require_feature_cache=False,
        load_frame_cache=False,
        load_feature_cache=False,
    )
    materialized_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            record = copy.deepcopy(dict(row))
            item = record_builder.build_item(record)
            multimodal_cache = _sanitize_multimodal_cache(item.get("multimodal_cache") or {})
            video_path = str(multimodal_cache.get("video_path") or item.get("video") or record.get("video_path") or "")
            materialized = {
                "schema_version": int(MATERIALIZED_CACHE_METADATA_SCHEMA_VERSION),
                "materialized_format": MATERIALIZED_RUNTIME_ITEMS_FORMAT,
                "source_row_sha256": canonical_json_hash(record),
                "video_id": item.get("video_id", record.get("video_id")),
                "split": item.get("split", record.get("split")),
                "source": item.get("source", record.get("source")),
                "video": str(item.get("video") or ""),
                "record": _json_safe(record),
                "messages": _sanitize_messages(item.get("messages") or [], video_path=video_path),
                "multimodal_cache": multimodal_cache,
            }
            materialized_rows.append(validate_materialized_runtime_item_row(materialized))
        except Exception:
            if strict:
                raise
    return materialized_rows


class MaterializedMessagesSFTDataset:
    def __init__(
        self,
        materialized_messages_path: str | Path,
        *,
        include_splits: str | Sequence[str] | None = None,
        config: Any = None,
    ):
        del config
        include_split_set = set(_normalize_include_splits(include_splits))
        self.rows = [
            row
            for row in load_materialized_sft_rows(materialized_messages_path)
            if not include_split_set or str(row.get("split") or "").strip() in include_split_set
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = copy.deepcopy(self.rows[int(idx)])
        return {
            "messages": row.get("messages") or [],
            "sample_weight": float(row.get("sample_weight", 1.0) or 1.0),
            "video_id": row.get("video_id"),
            "split": row.get("split"),
        }


class MaterializedRuntimeItemDataset:
    def __init__(
        self,
        materialized_items_path: str | Path,
        *,
        include_splits: str | Sequence[str] | None = None,
        require_frame_cache: bool = True,
        require_feature_cache: bool = True,
        proposal_runtime: Any = None,
        strict_feature_guided_proposal: bool = False,
    ):
        include_split_set = set(_normalize_include_splits(include_splits))
        self.items = [
            row
            for row in load_materialized_runtime_item_rows(materialized_items_path)
            if not include_split_set or str(row.get("split") or "").strip() in include_split_set
        ]
        self.records = [copy.deepcopy(dict(row.get("record") or {})) for row in self.items]
        self.require_frame_cache = bool(require_frame_cache)
        self.require_feature_cache = bool(require_feature_cache)
        self.proposal_runtime = proposal_runtime
        self.strict_feature_guided_proposal = bool(strict_feature_guided_proposal)

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _load_frame_cache(cache_path: str | Path) -> tuple[Any, str]:
        from pathlib import Path as _Path
        import torch

        resolved = _Path(cache_path)
        if not str(resolved) or not resolved.exists():
            return None, "missing"
        try:
            payload = torch.load(resolved)
        except Exception:
            return None, "load_error"
        if not isinstance(payload, Mapping) or "frame_tensor" not in payload or "fps" not in payload:
            return None, "invalid"
        return payload, "loaded"

    @staticmethod
    def _load_feature_cache(cache_path: str | Path, *, fps: float | None = None, frame_indices: Sequence[int] | None = None) -> Any:
        from pathlib import Path as _Path
        import torch
        from saver_v3.core.proposal import coerce_feature_cache_payload

        resolved = _Path(cache_path)
        if not str(resolved) or not resolved.exists():
            return None
        try:
            payload = torch.load(resolved)
        except Exception:
            return None
        return coerce_feature_cache_payload(payload, fps=fps, frame_indices=frame_indices)

    def _rehydrate_multimodal_cache(self, cache_payload: Mapping[str, Any]) -> dict[str, Any]:
        cache = copy.deepcopy(dict(cache_payload or {}))
        frame_cache_path = str(cache.get("frame_cache_path") or "")
        feature_cache_path = str(cache.get("feature_cache_path") or "")
        fps = float(cache.get("fps") or 1.0)
        frame_indices = [int(index) for index in list(cache.get("frame_indices") or [])]

        frame_cache, frame_cache_status = self._load_frame_cache(frame_cache_path)
        if self.require_frame_cache and frame_cache is None:
            raise MaterializedCacheError(
                f"Materialized runtime item requires frame_cache but it is unavailable: path={frame_cache_path} status={frame_cache_status}"
            )
        if frame_cache is not None:
            cache["video"] = frame_cache.get("frame_tensor")
            cache["fps"] = float(frame_cache.get("fps") or fps)
            if not frame_indices:
                cache["frame_indices"] = [int(index) for index in list(frame_cache.get("frame_indices") or [])]
                frame_indices = [int(index) for index in list(cache.get("frame_indices") or [])]
        else:
            cache["video"] = None

        feature_cache = self._load_feature_cache(feature_cache_path, fps=float(cache.get("fps") or fps), frame_indices=frame_indices or None)
        if self.require_feature_cache and feature_cache is None:
            raise MaterializedCacheError(
                f"Materialized runtime item requires feature_cache but it is unavailable: path={feature_cache_path}"
            )
        cache["embedding"] = feature_cache
        cache["proposal_runtime"] = self.proposal_runtime
        cache["strict_feature_guided_proposal"] = bool(self.strict_feature_guided_proposal)
        cache.setdefault("preview_frames", None)
        cache.setdefault("preview_timestamps", [])
        cache.setdefault("preview_frame_indices", [])
        return cache

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = copy.deepcopy(self.items[int(idx)])
        record = copy.deepcopy(dict(row.get("record") or {}))
        multimodal_cache = self._rehydrate_multimodal_cache(row.get("multimodal_cache") or {})
        item = copy.deepcopy(record)
        item["video"] = str(row.get("video") or multimodal_cache.get("video_path") or record.get("video_path") or "")
        item["messages"] = copy.deepcopy(list(row.get("messages") or []))
        item["multimodal_cache"] = multimodal_cache
        item.setdefault("video_id", row.get("video_id", record.get("video_id")))
        item.setdefault("split", row.get("split", record.get("split")))
        item.setdefault("source", row.get("source", record.get("source")))
        return item
