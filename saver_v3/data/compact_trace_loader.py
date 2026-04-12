from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Sequence

from data_utils.jsonl import iter_jsonl

from .prepared_schema import validate_compact_trace_row


def _normalize_include_splits(include_splits: str | Sequence[str] | None) -> set[str]:
    if include_splits is None:
        return set()
    if isinstance(include_splits, str):
        values = [value.strip() for value in include_splits.split(",") if value.strip()]
    else:
        values = [str(value).strip() for value in include_splits if str(value).strip()]
    return set(values)


def iter_compact_trace_rows(path: str, *, include_splits: str | Sequence[str] | None = None) -> Iterator[Dict]:
    include_split_set = _normalize_include_splits(include_splits)
    for row in iter_jsonl(path, skip_invalid_lines=False):
        normalized = validate_compact_trace_row(row)
        if include_split_set and str(normalized.get("split") or "") not in include_split_set:
            continue
        yield normalized


def load_compact_trace_rows(path: str, *, include_splits: str | Sequence[str] | None = None) -> List[Dict]:
    return list(iter_compact_trace_rows(path, include_splits=include_splits))
