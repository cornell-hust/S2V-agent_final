from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from saver_v3.data.config import PreviewConfig, PromptConfig, RolloutTraceConfig, SaverAgentConfig


PREPARED_SFT_METADATA_SCHEMA_VERSION = 2
PREPARED_SFT_FORMAT = "compact_trace_v2"


def prepared_sft_metadata_path(prepared_data_path: str | Path) -> Path:
    return Path(str(prepared_data_path) + ".meta.json")


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


def ensure_prepared_sft_metadata(
    prepared_data_path: str | Path,
    *,
    config: SaverAgentConfig | None = None,
    require_config_match: bool = False,
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
    return metadata
