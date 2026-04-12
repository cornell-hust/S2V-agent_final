"""Checkpoint manifest helpers for self-describing full-model training artifacts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


CHECKPOINT_MANIFEST_FILENAME = "checkpoint_manifest.json"
CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class CheckpointManifest:
    checkpoint_dir: str
    base_model: str
    global_step: int
    schema_version: int = CHECKPOINT_MANIFEST_SCHEMA_VERSION
    created_at: str = field(default_factory=_utcnow_iso)
    epoch: Optional[float] = None
    source_stage: str = ""
    attn_implementation: str = "flash_attention_3"
    files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["global_step"] = int(self.global_step)
        if self.epoch is not None:
            payload["epoch"] = float(self.epoch)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CheckpointManifest":
        return cls(
            checkpoint_dir=str(payload.get("checkpoint_dir") or "").strip(),
            base_model=str(payload.get("base_model") or payload.get("base_model_path") or "").strip(),
            global_step=int(payload.get("global_step", 0) or 0),
            schema_version=int(payload.get("schema_version", CHECKPOINT_MANIFEST_SCHEMA_VERSION) or CHECKPOINT_MANIFEST_SCHEMA_VERSION),
            created_at=str(payload.get("created_at") or _utcnow_iso()),
            epoch=float(payload["epoch"]) if payload.get("epoch") is not None else None,
            source_stage=str(payload.get("source_stage") or "").strip(),
            attn_implementation=str(payload.get("attn_implementation") or "flash_attention_3").strip() or "flash_attention_3",
            files=[str(item) for item in list(payload.get("files") or [])],
            metadata=dict(payload.get("metadata") or {}),
            metrics=dict(payload.get("metrics") or {}),
        )


def checkpoint_manifest_path(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return path
    return path / CHECKPOINT_MANIFEST_FILENAME


def write_checkpoint_manifest(
    manifest: CheckpointManifest,
    path: str | Path | None = None,
) -> Path:
    manifest_path = checkpoint_manifest_path(path or manifest.checkpoint_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def load_checkpoint_manifest(path: str | Path) -> CheckpointManifest:
    manifest_path = checkpoint_manifest_path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint manifest at {manifest_path} does not contain a JSON object.")
    return CheckpointManifest.from_dict(payload)


def resolve_latest_checkpoint_manifest(root_dir: str | Path) -> CheckpointManifest | None:
    root_dir = Path(root_dir)
    manifest_paths = sorted(root_dir.rglob(CHECKPOINT_MANIFEST_FILENAME))
    manifests: list[CheckpointManifest] = []
    for manifest_path in manifest_paths:
        try:
            manifests.append(load_checkpoint_manifest(manifest_path))
        except Exception:
            continue
    if not manifests:
        return None
    manifests.sort(key=lambda item: (int(item.global_step), str(item.created_at), str(item.checkpoint_dir)))
    return manifests[-1]
