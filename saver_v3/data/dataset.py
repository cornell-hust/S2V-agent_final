from __future__ import annotations

import copy
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from split_utils import filter_records_by_split, parse_include_splits

from saver_v3.data.config import PreviewConfig, SaverAgentConfig
from saver_v3.core.proposal import coerce_feature_cache_payload
from saver_v3.core.prompts import build_system_prompt, build_user_prompt
from saver_v3.core.tool_registry import get_tool_schemas


DEFAULT_CACHE_VIDEO_FPS = 2.0
DEFAULT_MAX_CACHE_FRAMES = 256
DEFAULT_NUM_PREVIEW_FRAMES = 8


def _frame_cache_path(video_path: Path) -> Path:
    return Path(str(video_path) + ".frame_cache")


def _print_cache_warning(message: str) -> None:
    print(f"[cache-warning] {message}", flush=True)


def _jsonl_decode_error_message(path: Path, line_number: int, line: str, exc: Exception) -> str:
    preview = line.strip().replace("\t", " ")
    if len(preview) > 240:
        preview = preview[:240] + "..."
    return f"Invalid JSONL at {path}:{line_number}: {exc}. Line preview: {preview}"


def _load_jsonl(path: Path, *, skip_invalid_lines: bool = False) -> List[Dict]:
    rows: List[Dict] = []
    invalid_messages: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                message = _jsonl_decode_error_message(path, line_number, line, exc)
                if not skip_invalid_lines:
                    raise ValueError(message) from exc
                invalid_messages.append(message)
    if invalid_messages:
        warnings.warn(
            f"Skipped {len(invalid_messages)} invalid JSONL lines while loading {path}. "
            f"First error: {invalid_messages[0]}",
            RuntimeWarning,
        )
    return rows


class SaverAgentDataset(torch.utils.data.Dataset):
    """Lightweight SAVER dataset wrapper with TimeSearch-style message fields.

    This class intentionally keeps visual loading conservative. If ``.frame_cache``
    or ``.feature_cache`` files exist next to the resolved video path, it loads
    them. Otherwise it tries a lightweight decord sampling fallback so the
    tool/environment skeleton can still return real frames on datasets that do
    not yet have precomputed caches.
    """

    def __init__(
        self,
        data_path: str | Path,
        data_root: str | Path = "",
        *,
        extra_video_roots: Optional[Sequence[str | Path]] = None,
        cache_video_fps: float = DEFAULT_CACHE_VIDEO_FPS,
        max_cache_frames: int = DEFAULT_MAX_CACHE_FRAMES,
        num_preview_frames: int = DEFAULT_NUM_PREVIEW_FRAMES,
        preview_sampling_fps: float | None = None,
        skip_invalid_jsonl_lines: bool = False,
        config: SaverAgentConfig | None = None,
        include_splits: Optional[Sequence[str] | str] = None,
        require_frame_cache: bool = False,
        require_feature_cache: bool = False,
        load_frame_cache: bool = True,
        load_feature_cache: bool = True,
        proposal_runtime: Any = None,
        strict_feature_guided_proposal: bool = False,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.data_root = Path(data_root) if data_root else Path()
        self.include_splits = parse_include_splits(include_splits)
        self.records = filter_records_by_split(
            _load_jsonl(self.data_path, skip_invalid_lines=skip_invalid_jsonl_lines),
            self.include_splits,
        )
        self.tool_schemas = get_tool_schemas()
        self.tool_names = [tool["function"]["name"] for tool in self.tool_schemas]
        self.extra_video_roots = [Path(path) for path in (extra_video_roots or [])]
        self.cache_video_fps = float(cache_video_fps)
        self.max_cache_frames = int(max_cache_frames)
        self.require_frame_cache = bool(require_frame_cache)
        self.require_feature_cache = bool(require_feature_cache)
        self.load_frame_cache = bool(load_frame_cache)
        self.load_feature_cache = bool(load_feature_cache)
        if self.require_frame_cache and not self.load_frame_cache:
            raise ValueError("require_frame_cache=True is incompatible with load_frame_cache=False")
        if self.require_feature_cache and not self.load_feature_cache:
            raise ValueError("require_feature_cache=True is incompatible with load_feature_cache=False")
        self.proposal_runtime = proposal_runtime
        self.strict_feature_guided_proposal = bool(strict_feature_guided_proposal)
        self._resolved_video_paths = [self._resolve_video_path(record["video_path"]) for record in self.records]
        self._logged_cache_fallbacks: set[tuple[str, str]] = set()
        self.config = copy.deepcopy(config) if config is not None else SaverAgentConfig(
            preview=PreviewConfig(
                num_preview_frames=int(num_preview_frames),
                preview_sampling_fps=preview_sampling_fps,
                max_preview_frames=int(num_preview_frames),
            )
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        record = copy.deepcopy(self.records[idx])
        video_path = self._resolved_video_paths[idx]
        record["video"] = str(video_path)
        record["multimodal_cache"] = self._build_multimodal_cache(record)
        record["messages"] = self._build_messages(record, record["multimodal_cache"])
        return record

    def summarize_frame_cache_status(self, *, max_examples: int = 5) -> Dict[str, Any]:
        num_cached_videos = 0
        num_missing_frame_cache = 0
        num_missing_video_files = 0
        missing_examples: List[Dict[str, str]] = []

        for record, video_path in zip(self.records, self._resolved_video_paths):
            cache_path = _frame_cache_path(video_path)
            if cache_path.exists():
                num_cached_videos += 1
            else:
                num_missing_frame_cache += 1
                if len(missing_examples) < max(0, int(max_examples)):
                    missing_examples.append(
                        {
                            "video_id": str(record.get("video_id") or ""),
                            "video_path": str(video_path),
                            "cache_path": str(cache_path),
                        }
                    )
            if not video_path.exists():
                num_missing_video_files += 1

        return {
            "num_records": len(self.records),
            "num_cached_videos": num_cached_videos,
            "num_missing_frame_cache": num_missing_frame_cache,
            "num_missing_video_files": num_missing_video_files,
            "missing_examples": missing_examples,
        }

    def format_frame_cache_status(self, *, prefix: str = "frame cache", max_examples: int = 5) -> str:
        summary = self.summarize_frame_cache_status(max_examples=max_examples)
        num_records = int(summary["num_records"])
        num_cached_videos = int(summary["num_cached_videos"])
        num_missing_frame_cache = int(summary["num_missing_frame_cache"])
        num_missing_video_files = int(summary["num_missing_video_files"])
        message = (
            f"{prefix}: cached={num_cached_videos}/{num_records} "
            f"missing_frame_cache={num_missing_frame_cache} "
            f"missing_video_files={num_missing_video_files}"
        )
        missing_examples = list(summary.get("missing_examples") or [])
        if missing_examples:
            preview = "; ".join(
                (
                    f"video_id={item.get('video_id') or '(unknown)'} "
                    f"cache_path={item.get('cache_path') or ''} "
                    f"video_path={item.get('video_path') or ''}"
                )
                for item in missing_examples
            )
            message += f" missing_examples=[{preview}]"
        return message

    def _build_messages(self, record: Dict, multimodal_cache: Dict) -> List[Dict]:
        tool_io = record.get("tool_io") or {}
        tool_schemas = get_tool_schemas(finalize_case_schema=tool_io.get("finalize_case_schema"))
        system_prompt = build_system_prompt(tool_schemas)
        user_prompt = build_user_prompt(
            record,
            preview_available=bool(multimodal_cache.get("preview_frames") is not None),
            prompt_config=self.config.prompt,
        )
        user_content = self._build_user_content(user_prompt, multimodal_cache)
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

    def _build_multimodal_cache(self, record: Dict) -> Dict:
        video_path = Path(record["video"])
        frame_cache = None
        cache_status = "disabled"
        fps = float((record.get("video_meta") or {}).get("fps") or 1.0)
        frames = None
        frame_indices: List[int] = []
        if self.load_frame_cache:
            frame_cache, cache_status = self._load_frame_cache(video_path)
            if frame_cache is None:
                if self.require_frame_cache:
                    raise ValueError(
                        f"Video-level frame_cache is required but unavailable: video_path={video_path} cache_status={cache_status}"
                    )
                self._warn_frame_cache_fallback(record=record, video_path=video_path, cache_status=cache_status)
                frame_cache = self._maybe_sample_video_frames(video_path, record)
            fps = frame_cache["fps"] if frame_cache else fps
            frames = frame_cache["frame_tensor"] if frame_cache else None
            frame_indices = list(frame_cache.get("frame_indices") or []) if frame_cache else []
        feature_cache = None
        if self.load_feature_cache:
            feature_cache = self._maybe_load_feature_cache(
                video_path,
                fps=float(fps),
                frame_indices=frame_indices or None,
            )
            if self.require_feature_cache and feature_cache is None:
                raise ValueError(f"Video-level feature_cache is required but unavailable: video_path={video_path}")
            if feature_cache is not None:
                fps = float(feature_cache.get("fps") or fps)
                if not frame_indices:
                    frame_indices = [int(index) for index in list(feature_cache.get("frame_indices") or [])]
        preview_frames, preview_timestamps, preview_frame_indices = (
            self._build_preview(frames, float(fps))
            if frames is not None
            else (None, [], [])
        )
        return {
            "video": frames,
            "embedding": feature_cache,
            "fps": float(fps),
            "duration": float(record["video_meta"]["duration_sec"]),
            "question": record["agent_task"]["task_prompt"],
            "structured_target": record.get("structured_target", {}),
            "tool_io": record.get("tool_io", {}),
            "video_path": str(video_path),
            "video_meta": record.get("video_meta", {}),
            "frame_indices": frame_indices,
            "preview_frames": preview_frames,
            "preview_timestamps": preview_timestamps,
            "preview_frame_indices": preview_frame_indices,
            "config_snapshot": self.config.to_dict(),
            "proposal_runtime": self.proposal_runtime,
            "strict_feature_guided_proposal": self.strict_feature_guided_proposal,
        }

    @staticmethod
    def _load_frame_cache(video_path: Path) -> tuple[Optional[Dict], str]:
        cache_path = _frame_cache_path(video_path)
        if not cache_path.exists():
            return None, "missing"
        try:
            cache = torch.load(cache_path)
        except Exception:
            return None, "load_error"
        if "frame_tensor" not in cache or "fps" not in cache:
            return None, "invalid"
        return cache, "loaded"

    @staticmethod
    def _maybe_load_feature_cache(video_path: Path, *, fps: Optional[float] = None, frame_indices: Optional[Sequence[int]] = None):
        cache_path = Path(str(video_path) + ".feature_cache")
        if not cache_path.exists():
            return None
        try:
            payload = torch.load(cache_path)
        except Exception:
            return None
        return coerce_feature_cache_payload(payload, fps=fps, frame_indices=frame_indices)

    def _resolve_video_path(self, raw_video_path: str | Path) -> Path:
        raw_path = Path(raw_video_path)
        if raw_path.is_absolute() and raw_path.exists():
            return raw_path

        relative_variants: List[Path] = [raw_path]
        if raw_path.parts and raw_path.parts[0] in {"data", "datasets"} and len(raw_path.parts) > 1:
            relative_variants.append(Path(*raw_path.parts[1:]))

        candidates: List[Path] = []
        for relative_path in relative_variants:
            if relative_path.is_absolute():
                candidates.append(relative_path)
                continue
            for root in self._candidate_video_roots():
                candidate = root / relative_path
                if candidate not in candidates:
                    candidates.append(candidate)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        if candidates:
            return candidates[0]
        return raw_path

    def _candidate_video_roots(self) -> List[Path]:
        candidates: List[Path] = []
        if self.data_root:
            candidates.extend(
                [
                    self.data_root,
                    self.data_root / "data",
                    self.data_root / "datasets",
                    self.data_root / "Wmh" / "datasets",
                    self.data_root / "Wmh" / "datasets" / "MSDA",
                ]
            )
        candidates.append(self.data_path.parent)
        candidates.extend(self.extra_video_roots)

        unique_candidates: List[Path] = []
        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique_candidates.append(candidate)
        return unique_candidates

    def _maybe_sample_video_frames(self, video_path: Path, record: Dict) -> Optional[Dict]:
        if not video_path.exists():
            return None

        try:
            from decord import VideoReader, cpu
        except Exception:
            return None

        try:
            video_reader = VideoReader(str(video_path), ctx=cpu(0))
        except Exception:
            return None

        video_meta = record.get("video_meta", {})
        native_fps = float(video_reader.get_avg_fps() or 0.0)
        if native_fps <= 0:
            native_fps = float(video_meta.get("fps") or 1.0)
        total_frames = len(video_reader)
        duration = float(video_meta.get("duration_sec") or 0.0)
        if duration <= 0 and total_frames > 0 and native_fps > 0:
            duration = total_frames / native_fps

        frame_indices = self._build_sample_indices(
            total_frames=total_frames,
            duration=duration,
            native_fps=native_fps,
        )
        if not frame_indices:
            return None

        try:
            sampled_frames = video_reader.get_batch(frame_indices).asnumpy()
        except Exception:
            return None
        if len(sampled_frames) == 0:
            return None

        frame_tensor = torch.from_numpy(sampled_frames).permute(0, 3, 1, 2).contiguous()
        sampled_fps = len(sampled_frames) / duration if duration > 0 else native_fps
        return {
            "frame_tensor": frame_tensor,
            "frame_indices": frame_indices,
            "fps": float(sampled_fps if sampled_fps > 0 else 1.0),
        }

    def _warn_frame_cache_fallback(
        self,
        *,
        record: Dict[str, Any],
        video_path: Path,
        cache_status: str,
    ) -> None:
        warning_key = (str(video_path), str(cache_status))
        if warning_key in self._logged_cache_fallbacks:
            return
        self._logged_cache_fallbacks.add(warning_key)
        _print_cache_warning(
            f"video_id={record.get('video_id') or '(unknown)'} cache_status={cache_status} "
            f"cache_path={_frame_cache_path(video_path)} video_path={video_path} "
            "falling back to raw video decode."
        )

    def _build_sample_indices(self, *, total_frames: int, duration: float, native_fps: float) -> List[int]:
        if total_frames <= 0:
            return []

        target_fps = native_fps if native_fps <= 0 else min(self.cache_video_fps, native_fps)
        if duration > 0 and target_fps > 0:
            target_count = int(math.ceil(duration * target_fps))
        else:
            target_count = total_frames
        target_count = max(1, min(target_count, total_frames, self.max_cache_frames))

        indices = np.round(np.linspace(0, total_frames - 1, target_count)).astype(int).tolist()
        return list(dict.fromkeys(indices))

    def _build_preview(
        self,
        frames: Optional[torch.Tensor],
        fps: float,
    ) -> tuple[Optional[torch.Tensor], List[float], List[int]]:
        if frames is None or len(frames) == 0:
            return None, [], []
        preview_count = self._resolve_preview_count(len(frames), fps)
        indices = np.round(np.linspace(0, len(frames) - 1, preview_count)).astype(int).tolist()
        indices = list(dict.fromkeys(indices))
        index_tensor = torch.tensor(indices, dtype=torch.long)
        preview_frames = frames.index_select(0, index_tensor)
        preview_timestamps = [round(float(index) / max(fps, 1e-6), 6) for index in indices]
        return preview_frames, preview_timestamps, indices

    def _resolve_preview_count(self, total_frames: int, fps: float) -> int:
        preview_cfg = self.config.preview
        max_preview_frames = max(1, int(preview_cfg.max_preview_frames))
        configured_frames = max(1, int(preview_cfg.num_preview_frames))
        preview_count = min(total_frames, configured_frames, max_preview_frames)
        if preview_cfg.preview_sampling_fps is not None and fps > 0:
            duration = total_frames / fps
            sampled_count = max(1, int(math.ceil(duration * float(preview_cfg.preview_sampling_fps))))
            preview_count = min(preview_count, sampled_count)
        return max(1, preview_count)

    @staticmethod
    def _build_user_content(user_prompt: str, multimodal_cache: Dict) -> List[Dict]:
        preview_frames = multimodal_cache.get("preview_frames")
        preview_timestamps = multimodal_cache.get("preview_timestamps") or []
        preview_frame_indices = multimodal_cache.get("preview_frame_indices") or []
        frame_indices = multimodal_cache.get("frame_indices") or []
        if preview_frames is None or len(preview_timestamps) == 0:
            return [{"type": "text", "text": user_prompt}]

        content: List[Dict] = []
        for timestamp, frame, sampled_frame_index in zip(preview_timestamps, preview_frames, preview_frame_indices):
            content.append({"type": "text", "text": f"{timestamp:.3f}s"})
            image_item = {
                "type": "image",
                "image": frame,
                "sampled_frame_index": int(sampled_frame_index),
                "timestamp_sec": float(timestamp),
            }
            if 0 <= int(sampled_frame_index) < len(frame_indices):
                image_item["raw_frame_index"] = int(frame_indices[int(sampled_frame_index)])
            content.append(image_item)
        content.append({"type": "text", "text": user_prompt})
        return content


class SaverRecordItemBuilder:
    """Build a single SAVER runtime item lazily from an in-memory record."""

    def __init__(
        self,
        data_root: str | Path = "",
        *,
        extra_video_roots: Optional[Sequence[str | Path]] = None,
        cache_video_fps: float = DEFAULT_CACHE_VIDEO_FPS,
        max_cache_frames: int = DEFAULT_MAX_CACHE_FRAMES,
        num_preview_frames: int = DEFAULT_NUM_PREVIEW_FRAMES,
        preview_sampling_fps: float | None = None,
        config: SaverAgentConfig | None = None,
        require_frame_cache: bool = False,
        require_feature_cache: bool = False,
        load_frame_cache: bool = True,
        load_feature_cache: bool = True,
        proposal_runtime: Any = None,
        strict_feature_guided_proposal: bool = False,
    ):
        self.data_root = Path(data_root) if data_root else Path()
        self.extra_video_roots = [Path(path) for path in (extra_video_roots or [])]
        self.cache_video_fps = float(cache_video_fps)
        self.max_cache_frames = int(max_cache_frames)
        self.require_frame_cache = bool(require_frame_cache)
        self.require_feature_cache = bool(require_feature_cache)
        self.load_frame_cache = bool(load_frame_cache)
        self.load_feature_cache = bool(load_feature_cache)
        if self.require_frame_cache and not self.load_frame_cache:
            raise ValueError("require_frame_cache=True is incompatible with load_frame_cache=False")
        if self.require_feature_cache and not self.load_feature_cache:
            raise ValueError("require_feature_cache=True is incompatible with load_feature_cache=False")
        self.proposal_runtime = proposal_runtime
        self.strict_feature_guided_proposal = bool(strict_feature_guided_proposal)
        self._logged_cache_fallbacks: set[tuple[str, str]] = set()
        self.data_path = Path(".")
        self.config = copy.deepcopy(config) if config is not None else SaverAgentConfig(
            preview=PreviewConfig(
                num_preview_frames=int(num_preview_frames),
                preview_sampling_fps=preview_sampling_fps,
                max_preview_frames=int(num_preview_frames),
            )
        )

    def build_item(self, record: Dict[str, Any]) -> Dict[str, Any]:
        built = copy.deepcopy(record)
        video_path = self._resolve_video_path(str(built.get("video_path") or ""))
        built["video"] = str(video_path)
        built["multimodal_cache"] = self._build_multimodal_cache(built)
        built["messages"] = self._build_messages(built, built["multimodal_cache"])
        return built

    def _resolve_video_path(self, raw_video_path: str | Path) -> Path:
        return SaverAgentDataset._resolve_video_path(self, raw_video_path)

    def _candidate_video_roots(self) -> List[Path]:
        return SaverAgentDataset._candidate_video_roots(self)

    def _build_messages(self, record: Dict, multimodal_cache: Dict) -> List[Dict]:
        return SaverAgentDataset._build_messages(self, record, multimodal_cache)

    @staticmethod
    def _build_user_content(user_prompt: str, multimodal_cache: Dict) -> List[Dict]:
        return SaverAgentDataset._build_user_content(user_prompt, multimodal_cache)

    def _build_multimodal_cache(self, record: Dict) -> Dict:
        return SaverAgentDataset._build_multimodal_cache(self, record)

    def _warn_frame_cache_fallback(
        self,
        *,
        record: Dict[str, Any],
        video_path: Path,
        cache_status: str,
    ) -> None:
        SaverAgentDataset._warn_frame_cache_fallback(
            self,
            record=record,
            video_path=video_path,
            cache_status=cache_status,
        )

    def _maybe_sample_video_frames(self, video_path: Path, record: Dict) -> Optional[Dict]:
        return SaverAgentDataset._maybe_sample_video_frames(self, video_path, record)

    def _build_sample_indices(self, *, total_frames: int, duration: float, native_fps: float) -> List[int]:
        return SaverAgentDataset._build_sample_indices(
            self,
            total_frames=total_frames,
            duration=duration,
            native_fps=native_fps,
        )

    def _build_preview(
        self,
        frames: Optional[torch.Tensor],
        fps: float,
    ) -> tuple[Optional[torch.Tensor], List[float], List[int]]:
        return SaverAgentDataset._build_preview(self, frames, fps)

    def _resolve_preview_count(self, total_frames: int, fps: float) -> int:
        return SaverAgentDataset._resolve_preview_count(self, total_frames, fps)

    @staticmethod
    def _load_frame_cache(video_path: Path) -> tuple[Optional[Dict], str]:
        return SaverAgentDataset._load_frame_cache(video_path)

    @staticmethod
    def _maybe_load_feature_cache(video_path: Path, *, fps: Optional[float] = None, frame_indices: Optional[Sequence[int]] = None):
        return SaverAgentDataset._maybe_load_feature_cache(video_path, fps=fps, frame_indices=frame_indices)
