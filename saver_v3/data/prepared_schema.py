"""Schema validation for compact-trace prepared rows used by full-model training."""

from __future__ import annotations

import copy
from typing import Any, Iterable

from saver_v3.data.protocol_signature import extract_protocol_signature
from saver_v3.data.runtime_contract import ensure_removed_fields_absent, validate_finalize_case_schema_removed_fields


PREPARED_SCHEMA_VERSION = 5
PREPARED_FORMAT = "compact_trace_v5"
PREPARED_SFT_FORMAT = PREPARED_FORMAT
LEGACY_PREPARED_FORMATS = ()
_ALLOWED_FORMATS = {PREPARED_FORMAT, *LEGACY_PREPARED_FORMATS}


class PreparedDataError(ValueError):
    """Raised when prepared rows do not match the expected compact-trace contract."""


def is_compact_trace_row(row: dict[str, Any]) -> bool:
    return isinstance(row, dict) and str(row.get("prepared_format") or "").strip() in _ALLOWED_FORMATS


def _require_non_empty_string(row: dict[str, Any], key: str) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise PreparedDataError(f"Prepared row is missing required field `{key}`.")
    return value


def _validate_trajectory(trajectory: Iterable[Any]) -> list[dict[str, Any]]:
    validated_steps: list[dict[str, Any]] = []
    for step_index, raw_step in enumerate(trajectory, start=1):
        if not isinstance(raw_step, dict):
            raise PreparedDataError(f"oracle_trajectory[{step_index}] must be a JSON object.")
        tool_name = str(raw_step.get("tool") or "").strip()
        if not tool_name:
            raise PreparedDataError(f"oracle_trajectory[{step_index}] is missing a tool name.")
        arguments = raw_step.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise PreparedDataError(f"oracle_trajectory[{step_index}].arguments must be a JSON object.")
        step = copy.deepcopy(raw_step)
        step["tool"] = tool_name
        step["arguments"] = copy.deepcopy(arguments)
        if tool_name == "verify_hypothesis":
            ensure_removed_fields_absent(
                dict(step["arguments"].get("claim") or {}),
                context=f"oracle_trajectory[{step_index}].arguments.claim",
            )
        elif tool_name == "finalize_case":
            ensure_removed_fields_absent(
                step["arguments"],
                context=f"oracle_trajectory[{step_index}].arguments",
            )
        validated_steps.append(step)
    return validated_steps


def validate_prepared_row(row: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(row, dict):
        raise PreparedDataError("Prepared rows must be JSON objects.")

    normalized = copy.deepcopy(row)
    normalized["schema_version"] = int(normalized.get("schema_version", PREPARED_SCHEMA_VERSION) or PREPARED_SCHEMA_VERSION)
    prepared_format = str(normalized.get("prepared_format") or "").strip()
    if prepared_format not in _ALLOWED_FORMATS:
        raise PreparedDataError(
            f"Unsupported prepared_format `{prepared_format or '(missing)'}`. Expected one of {sorted(_ALLOWED_FORMATS)}."
        )
    normalized["prepared_format"] = prepared_format
    normalized["video_id"] = str(normalized.get("video_id") or normalized.get("id") or "").strip() or "unknown"
    normalized["video_path"] = _require_non_empty_string(normalized, "video_path")
    protocol_signature = extract_protocol_signature(normalized)
    if not protocol_signature.get("protocol_version"):
        raise PreparedDataError("Prepared row is missing required `protocol_signature`.")
    normalized["protocol_signature"] = protocol_signature
    if "structured_target" in normalized and normalized["structured_target"] is not None:
        normalized["structured_target"] = ensure_removed_fields_absent(
            normalized.get("structured_target"),
            context="prepared.structured_target",
        )
    if "oracle_final_decision" in normalized and normalized["oracle_final_decision"] is not None:
        normalized["oracle_final_decision"] = ensure_removed_fields_absent(
            normalized.get("oracle_final_decision"),
            context="prepared.oracle_final_decision",
        )
    tool_io = normalized.get("tool_io")
    if isinstance(tool_io, dict):
        validate_finalize_case_schema_removed_fields(
            dict(tool_io.get("finalize_case_schema") or {}),
            context="prepared.tool_io.finalize_case_schema",
        )
    search_supervision = normalized.get("search_supervision")
    if isinstance(search_supervision, dict):
        ensure_removed_fields_absent(
            dict(search_supervision.get("finalize_policy") or {}),
            context="prepared.search_supervision.finalize_policy",
        )

    trajectory = normalized.get("oracle_trajectory")
    if not isinstance(trajectory, list):
        raise PreparedDataError("Prepared row is missing required list field `oracle_trajectory`.")
    normalized["oracle_trajectory"] = _validate_trajectory(trajectory)

    if "agent_task" in normalized and normalized["agent_task"] is not None and not isinstance(normalized["agent_task"], dict):
        raise PreparedDataError("agent_task must be a JSON object when present.")
    if "qa_pairs" in normalized and normalized["qa_pairs"] is not None and not isinstance(normalized["qa_pairs"], list):
        raise PreparedDataError("qa_pairs must be a list when present.")

    return normalized


def validate_compact_trace_row(row: dict[str, Any]) -> dict[str, Any]:
    return validate_prepared_row(row)
