#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from saver_v3.data.dataset import DEFAULT_CACHE_VIDEO_FPS, DEFAULT_MAX_CACHE_FRAMES, SaverAgentDataset


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build .frame_cache files next to source videos so SAVER training can avoid repeated raw-video decoding."
    )
    parser.add_argument("--data", required=True, help="Path to SAVER agent/oracle JSONL data.")
    parser.add_argument("--data-root", default="", help="Root path used to resolve relative video paths.")
    parser.add_argument("--include-splits", default="", help="Optional comma-separated split whitelist, e.g. train or train,val.")
    parser.add_argument(
        "--cache-video-fps",
        type=float,
        default=DEFAULT_CACHE_VIDEO_FPS,
        help="Target sampling fps used when building frame_cache files.",
    )
    parser.add_argument(
        "--max-cache-frames",
        type=int,
        default=DEFAULT_MAX_CACHE_FRAMES,
        help="Maximum number of cached frames per video.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .frame_cache files.")
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
    print(message, flush=True)


def _build_dataset(
    *,
    data_path: str | Path,
    data_root: str | Path = "",
    include_splits: Optional[str] = None,
    cache_video_fps: float = DEFAULT_CACHE_VIDEO_FPS,
    max_cache_frames: int = DEFAULT_MAX_CACHE_FRAMES,
) -> SaverAgentDataset:
    return SaverAgentDataset(
        data_path,
        data_root=data_root,
        include_splits=include_splits,
        cache_video_fps=cache_video_fps,
        max_cache_frames=max_cache_frames,
    )


def _cache_path_for_video(video_path: Path) -> Path:
    return Path(str(video_path) + ".frame_cache")


def _save_frame_cache(cache_path: Path, payload: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)


def build_frame_caches(
    *,
    data_path: str | Path,
    data_root: str | Path = "",
    include_splits: Optional[str] = None,
    cache_video_fps: float = DEFAULT_CACHE_VIDEO_FPS,
    max_cache_frames: int = DEFAULT_MAX_CACHE_FRAMES,
    overwrite: bool = False,
    progress_every: int = 25,
) -> Dict[str, Any]:
    dataset = _build_dataset(
        data_path=data_path,
        data_root=data_root,
        include_splits=include_splits,
        cache_video_fps=cache_video_fps,
        max_cache_frames=max_cache_frames,
    )

    summary: Dict[str, Any] = {
        "data_path": str(data_path),
        "data_root": str(data_root),
        "include_splits": include_splits or "",
        "cache_video_fps": float(cache_video_fps),
        "max_cache_frames": int(max_cache_frames),
        "num_records": len(dataset.records),
        "written": 0,
        "skipped_existing": 0,
        "missing_videos": 0,
        "decode_failures": 0,
        "failures": [],
    }

    total_records = len(dataset.records)
    for index, record in enumerate(dataset.records, start=1):
        video_path = dataset._resolve_video_path(record["video_path"])
        cache_path = _cache_path_for_video(video_path)

        if cache_path.exists() and not overwrite:
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
            if _should_log_progress(index, total_records, progress_every):
                _print_progress(
                    f"[{index}/{total_records}] missing video: video_id={record.get('video_id', '')} path={video_path}"
                )
            continue

        frame_cache = dataset._maybe_sample_video_frames(video_path, record)
        if frame_cache is None:
            summary["decode_failures"] += 1
            summary["failures"].append(
                {
                    "video_id": record.get("video_id", ""),
                    "video_path": str(video_path),
                    "reason": "decode_failure",
                }
            )
            if _should_log_progress(index, total_records, progress_every):
                _print_progress(
                    f"[{index}/{total_records}] decode failure: video_id={record.get('video_id', '')} path={video_path}"
                )
            continue

        payload = {
            "version": "saver_frame_cache_v1",
            "frame_tensor": frame_cache["frame_tensor"].cpu(),
            "frame_indices": list(frame_cache.get("frame_indices") or []),
            "fps": float(frame_cache.get("fps") or 0.0),
            "timestamps_sec": [
                round(float(index_value) / max(float(frame_cache.get("fps") or 1.0), 1e-6), 6)
                for index_value in (frame_cache.get("frame_indices") or [])
            ],
        }
        _save_frame_cache(cache_path, payload)
        summary["written"] += 1
        if _should_log_progress(index, total_records, progress_every):
            _print_progress(
                f"[{index}/{total_records}] wrote cache: video_id={record.get('video_id', '')} frames={len(payload['frame_indices'])} path={cache_path}"
            )

    summary["succeeded"] = int(summary["written"])
    summary["failed"] = int(summary["missing_videos"]) + int(summary["decode_failures"])
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    summary = build_frame_caches(
        data_path=args.data,
        data_root=args.data_root,
        include_splits=args.include_splits,
        cache_video_fps=args.cache_video_fps,
        max_cache_frames=args.max_cache_frames,
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
