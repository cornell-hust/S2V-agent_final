from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SaverEnvironmentState:
    visited_windows: List[Dict[str, Any]] = field(default_factory=list)
    evidence_ledger: List[Dict[str, Any]] = field(default_factory=list)
    verification_records: List[Dict[str, Any]] = field(default_factory=list)
    finalized_case: Optional[Dict[str, Any]] = None
    finalized_semantic_answer: Optional[Dict[str, Any]] = None
    last_claim: Optional[Dict[str, Any]] = None
    active_evidence_window_ids: List[str] = field(default_factory=list)
    verifier_cache: List[Dict[str, Any]] = field(default_factory=list)
    next_evidence_id: int = 1
    next_window_id: int = 1


def validate_required_fields(payload: Dict[str, Any], schema: Optional[Dict[str, Any]]) -> None:
    if not schema:
        return
    required = schema.get("required", [])
    missing = [field_name for field_name in required if field_name not in payload]
    if missing:
        raise ValueError(f"Missing required finalize_case fields: {missing}")
    properties = schema.get("properties") or {}
    for field_name, field_schema in properties.items():
        if field_name not in payload:
            continue
        _validate_schema_value(payload[field_name], field_schema, field_name)


def _validate_schema_value(value: Any, schema: Dict[str, Any], field_name: str) -> None:
    if not schema:
        return
    if "oneOf" in schema:
        errors: List[str] = []
        for option in schema.get("oneOf") or []:
            try:
                _validate_schema_value(value, option, field_name)
                return
            except ValueError as exc:
                errors.append(str(exc))
        raise ValueError(errors[0] if errors else f"Invalid value for {field_name!r}.")

    expected_type = schema.get("type")
    if expected_type == "null":
        if value is not None:
            raise ValueError(f"{field_name} must be null.")
        return
    if expected_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string.")
    elif expected_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"{field_name} must be a number.")
    elif expected_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{field_name} must be an integer.")
    elif expected_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"{field_name} must be an array.")
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(value) < int(min_items):
            raise ValueError(f"{field_name} must contain at least {int(min_items)} items.")
        if max_items is not None and len(value) > int(max_items):
            raise ValueError(f"{field_name} must contain at most {int(max_items)} items.")
        item_schema = schema.get("items") or {}
        for index, item in enumerate(value):
            _validate_schema_value(item, item_schema, f"{field_name}[{index}]")
    elif expected_type == "object":
        if not isinstance(value, dict):
            raise ValueError(f"{field_name} must be an object.")

    enum_values = schema.get("enum")
    if enum_values is not None and value not in enum_values:
        raise ValueError(f"{field_name} must be one of {list(enum_values)}, got {value!r}.")
