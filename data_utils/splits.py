from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence


def parse_include_splits(include_splits: Optional[str | Sequence[str]]) -> Optional[List[str]]:
    if include_splits is None:
        return None

    raw_values: List[str] = []
    if isinstance(include_splits, str):
        raw_values = include_splits.split(",")
    else:
        for value in include_splits:
            if value is None:
                continue
            if isinstance(value, str):
                raw_values.extend(value.split(","))
            else:
                raw_values.append(str(value))

    normalized: List[str] = []
    seen = set()
    for value in raw_values:
        split_name = str(value).strip()
        if not split_name or split_name in seen:
            continue
        seen.add(split_name)
        normalized.append(split_name)
    return normalized or None


def filter_records_by_split(
    records: Iterable[Dict[str, Any]],
    include_splits: Optional[str | Sequence[str]],
) -> List[Dict[str, Any]]:
    allowed_splits = parse_include_splits(include_splits)
    materialized_records = list(records)
    if not allowed_splits:
        return materialized_records

    allowed = set(allowed_splits)
    return [record for record in materialized_records if str(record.get("split") or "").strip() in allowed]
