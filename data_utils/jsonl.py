from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List


def jsonl_decode_error_message(path: Path, line_number: int, line: str, exc: Exception) -> str:
    preview = line.strip().replace("\t", " ")
    if len(preview) > 240:
        preview = preview[:240] + "..."
    return f"Invalid JSONL at {path}:{line_number}: {exc}. Line preview: {preview}"


def iter_jsonl(path: str | Path, *, skip_invalid_lines: bool = False) -> Iterator[Dict[str, Any]]:
    resolved_path = Path(path)
    invalid_messages: List[str] = []
    with resolved_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                message = jsonl_decode_error_message(resolved_path, line_number, line, exc)
                if not skip_invalid_lines:
                    raise ValueError(message) from exc
                invalid_messages.append(message)
                continue
            if isinstance(payload, dict):
                yield payload
            elif not skip_invalid_lines:
                raise ValueError(f"Invalid JSONL at {resolved_path}:{line_number}: expected object, got {type(payload).__name__}.")
    if invalid_messages:
        warnings.warn(
            f"Skipped {len(invalid_messages)} invalid JSONL lines while loading {resolved_path}. "
            f"First error: {invalid_messages[0]}",
            RuntimeWarning,
        )


def load_jsonl(path: str | Path, *, skip_invalid_lines: bool = False) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path, skip_invalid_lines=skip_invalid_lines))


def write_jsonl(
    records: Iterable[Dict[str, Any]],
    path: str | Path,
    *,
    progress_callback: Callable[[int], None] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for index, record in enumerate(records, start=1):
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            if progress_callback is not None:
                progress_callback(int(index))
    return output_path
