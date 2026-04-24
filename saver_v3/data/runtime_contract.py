from __future__ import annotations

import copy
from typing import Any, Dict

from saver_v3.data.protocol_signature import build_protocol_signature, ensure_protocol_signature_matches, extract_protocol_signature

RUNTIME_CONTRACT_VERSION = 5
REMOVED_RUNTIME_FIELDS = (
    "severity",
    "counterfactual_type",
    "counterfactual_faithfulness",
    "precursor_interval_sec",
    "earliest_actionable_sec",
)


def ensure_removed_fields_absent(payload: Dict[str, Any] | None, *, context: str) -> Dict[str, Any]:
    normalized = copy.deepcopy(dict(payload or {}))
    present = [field_name for field_name in REMOVED_RUNTIME_FIELDS if field_name in normalized]
    if present:
        raise ValueError(
            f"{context} still uses removed contract field(s): {', '.join(sorted(present))}. "
            "Regenerate the artifact with the active v5 contract."
        )
    return normalized


def drop_removed_fields(payload: Dict[str, Any] | None) -> Dict[str, Any]:
    normalized = copy.deepcopy(dict(payload or {}))
    for field_name in REMOVED_RUNTIME_FIELDS:
        normalized.pop(field_name, None)
    return normalized


def validate_finalize_case_schema_removed_fields(
    schema: Dict[str, Any] | None,
    *,
    context: str,
) -> Dict[str, Any]:
    finalize_schema = copy.deepcopy(dict(schema or {}))
    ensure_removed_fields_absent(
        dict(finalize_schema.get("properties") or {}),
        context=f"{context}.properties",
    )
    required = [
        str(field_name).strip()
        for field_name in list(finalize_schema.get("required") or [])
        if str(field_name).strip()
    ]
    removed_required = [field_name for field_name in required if field_name in REMOVED_RUNTIME_FIELDS]
    if removed_required:
        raise ValueError(
            f"{context}.required still contains removed field(s): "
            + ", ".join(sorted(removed_required))
        )
    return finalize_schema


def validate_runtime_record_contract(row: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(row, dict):
        raise ValueError("Runtime record must be a JSON object.")
    normalized = copy.deepcopy(row)
    version = int(normalized.get("runtime_contract_version", 0) or 0)
    if version != int(RUNTIME_CONTRACT_VERSION):
        raise ValueError(
            f"Unsupported runtime_contract_version {version or '(missing)'}. "
            f"Expected {RUNTIME_CONTRACT_VERSION} for the active v5 runtime contract."
        )
    protocol_signature = extract_protocol_signature(normalized)
    if not protocol_signature.get("protocol_version"):
        raise ValueError(
            "Runtime record is missing required `protocol_signature`. "
            "Regenerate the artifact with the active v5 contract."
        )
    ensure_protocol_signature_matches(
        actual_signature=protocol_signature,
        expected_signature=build_protocol_signature(),
        context="Runtime record",
    )
    normalized["protocol_signature"] = protocol_signature
    ensure_removed_fields_absent(normalized.get("structured_target"), context="structured_target")
    tool_io = dict(normalized.get("tool_io") or {})
    validate_finalize_case_schema_removed_fields(
        dict(tool_io.get("finalize_case_schema") or {}),
        context="tool_io.finalize_case_schema",
    )
    search_supervision = dict(normalized.get("search_supervision") or {})
    ensure_removed_fields_absent(
        dict(search_supervision.get("finalize_policy") or {}),
        context="search_supervision.finalize_policy",
    )
    oracle_sft = dict(normalized.get("oracle_sft") or {})
    ensure_removed_fields_absent(
        dict(oracle_sft.get("final_decision") or {}),
        context="oracle_sft.final_decision",
    )
    return normalized
