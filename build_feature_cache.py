#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from saver_v3.data.dataset import SaverAgentDataset
from saver_v3.core.proposal import (
    SiglipFeatureEncoder,
    coerce_encoder_feature_tensor,
    compute_frame_cache_signature,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build .feature_cache files next to source videos using frame caches and an image-text encoder."
    )
    parser.add_argument("--data", required=True, help="Path to SAVER agent/oracle JSONL data.")
    parser.add_argument("--data-root", default="", help="Root path used to resolve relative video paths.")
    parser.add_argument("--include-splits", default="", help="Optional comma-separated split whitelist, e.g. train or train,val.")
    parser.add_argument("--model-path", default="", help="Local SigLIP/CLIP-compatible feature model path.")
    parser.add_argument("--torch-dtype", default="auto", help="Torch dtype for the feature encoder.")
    parser.add_argument("--device", default="cpu", help="Feature encoder device, e.g. cpu or cuda:0.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .feature_cache files.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N records. First and last are always logged.")
    parser.add_argument("--summary-output", default="", help="Optional path to write a JSON summary.")
    return parser.parse_args(argv)


def _should_log_progress(completed: int, total: int, progress_every: int) -> bool:
    if completed <= 0:
        return False
    if completed == 1 or completed == total:
        return True
    return int(progress_every) > 0 and completed % int(progress_every) == 0


def _print_progress(message: str) -> None:
    tqdm.write(message)


def _feature_cache_path(video_path: Path) -> Path:
    return Path(str(video_path) + ".feature_cache")


def build_feature_caches(
    *,
    data_path: str | Path,
    data_root: str | Path = "",
    include_splits: Optional[str] = None,
    encoder: Any = None,
    model_path: str = "",
    torch_dtype: str = "auto",
    device: str = "cpu",
    overwrite: bool = False,
    progress_every: int = 25,
) -> Dict[str, Any]:
    if encoder is None:
        if not model_path:
            raise ValueError("Either encoder or model_path must be provided to build feature caches.")
        encoder = SiglipFeatureEncoder.from_pretrained(model_path, torch_dtype=torch_dtype, device=device)

    dataset = SaverAgentDataset(
        data_path,
        data_root=data_root,
        include_splits=include_splits,
    )

    summary: Dict[str, Any] = {
        "data_path": str(data_path),
        "data_root": str(data_root),
        "include_splits": include_splits or "",
        "num_records": len(dataset.records),
        "written": 0,
        "skipped_existing": 0,
        "missing_videos": 0,
        "missing_frame_cache": 0,
        "encode_failures": 0,
        "failures": [],
    }

    total_records = len(dataset.records)
    for index, record in enumerate(
        tqdm(dataset.records, total=total_records, desc="Building Feature Caches", unit="video"),
        start=1
    ):
        video_path = dataset._resolve_video_path(record["video_path"])
        feature_cache_path = _feature_cache_path(video_path)

        if feature_cache_path.exists() and not overwrite:
            summary["skipped_existing"] += 1
            continue

        if not video_path.exists():
            summary["missing_videos"] += 1
            summary["failures"].append(
                {
                    "video_id": record.get("video_id", ""),
                    "video_path": str(video_path),
                    "reason": "missing_video",
                }
            )
            continue

        frame_cache, _ = dataset._load_frame_cache(video_path)
        if frame_cache is None:
            summary["missing_frame_cache"] += 1
            summary["failures"].append(
                {
                    "video_id": record.get("video_id", ""),
                    "video_path": str(video_path),
                    "reason": "missing_frame_cache",
                }
            )
            continue

        try:
            frame_tensor = frame_cache["frame_tensor"]
            embeddings = coerce_encoder_feature_tensor(
                encoder.encode_images(frame_tensor),
                preferred_keys=("image_embeds", "pooler_output", "last_hidden_state", "hidden_states"),
            )
            payload = {
                "version": "saver_feature_cache_v1",
                "model_name": getattr(encoder, "model_name", None) or getattr(encoder, "__class__", type(encoder)).__name__,
                "fps": float(frame_cache.get("fps") or 0.0),
                "frame_indices": [int(value) for value in frame_cache.get("frame_indices") or []],
                "timestamps_sec": [
                    round(float(index_value) / max(float(frame_cache.get("fps") or 1.0), 1e-6), 6)
                    for index_value in (frame_cache.get("frame_indices") or [])
                ],
                "embeddings": embeddings.detach().cpu(),
                "embedding_dim": int(embeddings.shape[-1]) if embeddings.ndim >= 2 else 1,
                "normalized": True,
                "frame_cache_signature": compute_frame_cache_signature(
                    fps=float(frame_cache.get("fps") or 0.0),
                    frame_indices=[int(value) for value in frame_cache.get("frame_indices") or []],
                    num_frames=int(frame_tensor.shape[0]),
                ),
            }
            feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, feature_cache_path)
            summary["written"] += 1
            if _should_log_progress(index, total_records, progress_every):
                _print_progress(
                    f"[{index}/{total_records}] wrote feature cache: video_id={record.get('video_id', '')} path={feature_cache_path}"
                )
        except Exception as exc:
            summary["encode_failures"] += 1
            summary["failures"].append(
                {
                    "video_id": record.get("video_id", ""),
                    "video_path": str(video_path),
                    "reason": "encode_failure",
                    "detail": str(exc),
                }
            )

    summary["succeeded"] = int(summary["written"])
    summary["failed"] = int(summary["missing_videos"]) + int(summary["missing_frame_cache"]) + int(summary["encode_failures"])
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    summary = build_feature_caches(
        data_path=args.data,
        data_root=args.data_root,
        include_splits=args.include_splits,
        model_path=args.model_path,
        torch_dtype=args.torch_dtype,
        device=args.device,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )
    if args.summary_output:
        output_path = Path(args.summary_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
