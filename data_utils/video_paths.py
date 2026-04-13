from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence


def frame_cache_path(video_path: str | Path) -> Path:
    return Path(str(video_path) + ".frame_cache")


def feature_cache_path(video_path: str | Path) -> Path:
    return Path(str(video_path) + ".feature_cache")


def candidate_video_roots(
    *,
    data_root: str | Path = "",
    data_path_parent: str | Path | None = None,
    extra_video_roots: Sequence[str | Path] | None = None,
) -> List[Path]:
    candidates: List[Path] = []
    if data_root:
        root = Path(data_root)
        candidates.extend(
            [
                root,
                root / "data",
                root / "datasets",
                root / "Wmh" / "datasets",
                root / "Wmh" / "datasets" / "MSDA",
            ]
        )
    if data_path_parent is not None:
        candidates.append(Path(data_path_parent))
    candidates.extend(Path(path) for path in (extra_video_roots or []))

    unique_candidates: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)
    return unique_candidates


def resolve_video_path(
    raw_video_path: str | Path,
    *,
    data_root: str | Path = "",
    data_path_parent: str | Path | None = None,
    extra_video_roots: Sequence[str | Path] | None = None,
) -> Path:
    raw_path = Path(raw_video_path)
    if raw_path.is_absolute() and raw_path.exists():
        return raw_path

    relative_variants: List[Path] = [raw_path]
    if raw_path.parts and raw_path.parts[0] in {"data", "datasets"} and len(raw_path.parts) > 1:
        relative_variants.append(Path(*raw_path.parts[1:]))

    candidates: List[Path] = []
    roots = candidate_video_roots(
        data_root=data_root,
        data_path_parent=data_path_parent,
        extra_video_roots=extra_video_roots,
    )
    for relative_path in relative_variants:
        if relative_path.is_absolute():
            candidates.append(relative_path)
            continue
        for root in roots:
            candidate = root / relative_path
            if candidate not in candidates:
                candidates.append(candidate)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    if candidates:
        return candidates[0]
    return raw_path


def existing_video_paths(paths: Iterable[str | Path]) -> List[Path]:
    return [Path(path) for path in paths if Path(path).exists()]
