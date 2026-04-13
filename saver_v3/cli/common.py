"""Shared CLI helpers for SAVER v3 entrypoints."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return REPO_ROOT


def resolve_path(path_value: str | Path | None, *, anchor: str | Path | None = None) -> Path | None:
    if path_value is None:
        return None
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    anchor_path = Path(anchor).expanduser().resolve() if anchor is not None else Path.cwd().resolve()
    search_roots = [anchor_path, REPO_ROOT, Path.cwd().resolve()]
    for root in search_roots:
        resolved = (root / candidate).resolve()
        if resolved.exists():
            return resolved
    return (anchor_path / candidate).resolve()


def ensure_mapping(payload: Any, *, path: Path) -> Mapping[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected mapping payload in {path}, got {type(payload).__name__}.")
    return payload


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"Missing YAML config: {path}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return dict(ensure_mapping(payload, path=resolved))


def _parse_override_value(raw_value: str) -> Any:
    text = str(raw_value)
    if not text:
        return ""
    try:
        return json.loads(text)
    except Exception:
        return text


def apply_config_overrides(mapping: Mapping[str, Any], overrides: Sequence[str] | None) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(mapping))
    for raw_item in list(overrides or []):
        item = str(raw_item or "").strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --override item {item!r}: expected dotted.key=value")
        dotted_key, raw_value = item.split("=", 1)
        path = [segment.strip() for segment in dotted_key.split(".") if segment.strip()]
        if not path:
            raise ValueError(f"Invalid --override item {item!r}: empty override path")
        cursor: dict[str, Any] = resolved
        for segment in path[:-1]:
            next_value = cursor.get(segment)
            if not isinstance(next_value, Mapping):
                next_value = {}
                cursor[segment] = next_value
            cursor = cursor[segment]
        cursor[path[-1]] = _parse_override_value(raw_value)
    return resolved


def load_json_mapping(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"Missing JSON config: {path}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return dict(ensure_mapping(payload, path=resolved))


def ensure_parent_dir(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def write_json(payload: Any, path: str | Path) -> Path:
    resolved = ensure_parent_dir(path)
    with resolved.open("w", encoding="utf-8") as handle:
        handle.write(json_dumps(payload) + "\n")
    return resolved
