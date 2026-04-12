from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from saver_v3.common.checkpoint_manifest import CheckpointManifest, write_checkpoint_manifest
from saver_v3.model.trainability import build_trainability_report


@dataclass(frozen=True)
class TrainingCheckpointPaths:
    root: Path
    ds_checkpoint_dir: Path
    hf_model_dir: Path
    manifest_path: Path


def build_checkpoint_paths(output_dir: str | Path, checkpoint_name: str) -> TrainingCheckpointPaths:
    root = Path(output_dir) / str(checkpoint_name)
    return TrainingCheckpointPaths(
        root=root,
        ds_checkpoint_dir=root / "ds_checkpoint",
        hf_model_dir=root / "hf_model",
        manifest_path=root / "checkpoint_manifest.json",
    )


def write_training_checkpoint_manifest(
    *,
    output_dir: str | Path,
    checkpoint_name: str,
    base_model_path: str,
    source_stage: str,
    model: Any,
    epoch: int = 0,
    global_step: int = 0,
    metadata: Dict[str, Any] | None = None,
) -> TrainingCheckpointPaths:
    paths = build_checkpoint_paths(output_dir, checkpoint_name)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.ds_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths.hf_model_dir.mkdir(parents=True, exist_ok=True)
    report = build_trainability_report(model)
    manifest = CheckpointManifest(
        checkpoint_dir=str(paths.root),
        base_model=str(base_model_path),
        source_stage=str(source_stage),
        epoch=float(epoch),
        global_step=int(global_step),
        files=[
            str(paths.ds_checkpoint_dir.relative_to(paths.root)),
            str(paths.hf_model_dir.relative_to(paths.root)),
        ],
        metadata={**dict(metadata or {}), "trainability_report": report},
    )
    write_checkpoint_manifest(manifest, paths.manifest_path)
    return paths
