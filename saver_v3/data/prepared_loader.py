"""JSONL loader for validated prepared compact-trace rows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from saver_v3.data.prepared_schema import PreparedDataError, validate_prepared_row


def iter_prepared_rows(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PreparedDataError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            try:
                yield validate_prepared_row(payload)
            except PreparedDataError as exc:
                raise PreparedDataError(f"Invalid prepared row on line {line_number} of {path}: {exc}") from exc


def load_prepared_rows(path: str | Path) -> list[dict]:
    return list(iter_prepared_rows(path))
