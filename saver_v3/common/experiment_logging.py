from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_experiment_log_dir(
    explicit_log_dir: str | Path = "",
    *,
    output_dir: str | Path = "",
    fallback_paths: Optional[Sequence[str | Path]] = None,
) -> Optional[Path]:
    explicit_value = str(explicit_log_dir or "").strip()
    if explicit_value:
        return Path(explicit_value)

    output_value = str(output_dir or "").strip()
    if output_value:
        return Path(output_value) / "logs"

    for candidate in fallback_paths or ():
        candidate_value = str(candidate or "").strip()
        if not candidate_value:
            continue
        candidate_path = Path(candidate_value)
        if candidate_path.suffix:
            return candidate_path.parent / "logs"
        return candidate_path / "logs"
    return None


def ensure_log_dir(log_dir: str | Path | None) -> Optional[Path]:
    if log_dir is None:
        return None
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
