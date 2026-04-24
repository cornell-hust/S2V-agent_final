from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping

from saver_v3.data.config import (
    DEFAULT_POLICY_MAX_NEW_TOKENS,
    DEFAULT_ROLLOUT_MAX_TURNS,
    SaverAgentConfig,
    saver_config_from_mapping,
)


PROTOCOL_SIGNATURE_SCHEMA_VERSION = 1
ACTIVE_PROTOCOL_VERSION = "seek_v5"
ACTIVE_MAIN_ROLLOUT_TERMINAL_MODE = "finalize_case_only"
ACTIVE_VERIFIER_CONTRACT = "next_tool_only"
ACTIVE_GENERATOR_REVISION = "2026-04-23-active-contract-v5"
DEFAULT_TEACHER_ROLE = "none"
TEACHER_ROLE_AUXILIARY = "teacher_rollout_primary_auxiliary"


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _resolved_config(config: Any) -> SaverAgentConfig:
    if config is None:
        return SaverAgentConfig()
    if isinstance(config, SaverAgentConfig):
        return copy.deepcopy(config)
    if isinstance(config, Mapping):
        return saver_config_from_mapping(config)
    if hasattr(config, "to_dict"):
        payload = config.to_dict()
        if isinstance(payload, Mapping):
            return saver_config_from_mapping(payload)
    return SaverAgentConfig()


def _generator_fingerprint_payload(*, config: SaverAgentConfig) -> dict[str, Any]:
    return {
        "protocol_version": ACTIVE_PROTOCOL_VERSION,
        "main_rollout_terminal_mode": ACTIVE_MAIN_ROLLOUT_TERMINAL_MODE,
        "verifier_contract": ACTIVE_VERIFIER_CONTRACT,
        "generator_revision": ACTIVE_GENERATOR_REVISION,
        "initial_observation_mode": str(config.initial_observation.mode or ""),
        "scan_num_frames": int(config.initial_observation.scan_num_frames or 0),
        "scan_purpose": str(config.initial_observation.scan_purpose or ""),
        "protect_initial_scan_from_visual_budget": bool(config.initial_observation.protect_from_visual_budget),
        "error_on_initial_scan_seq_prune": bool(config.initial_observation.error_on_seq_prune),
    }


def build_generator_fingerprint(*, config: Any = None) -> str:
    resolved = _resolved_config(config)
    return hashlib.sha256(_canonical_json_bytes(_generator_fingerprint_payload(config=resolved))).hexdigest()


def build_protocol_signature(
    *,
    config: Any = None,
    max_turns: int = DEFAULT_ROLLOUT_MAX_TURNS,
    policy_max_new_tokens: int = DEFAULT_POLICY_MAX_NEW_TOKENS,
    teacher_role: str = DEFAULT_TEACHER_ROLE,
) -> dict[str, Any]:
    resolved = _resolved_config(config)
    return {
        "signature_schema_version": int(PROTOCOL_SIGNATURE_SCHEMA_VERSION),
        "protocol_version": ACTIVE_PROTOCOL_VERSION,
        "initial_observation_mode": str(resolved.initial_observation.mode or "").strip(),
        "main_rollout_terminal_mode": ACTIVE_MAIN_ROLLOUT_TERMINAL_MODE,
        "verifier_contract": ACTIVE_VERIFIER_CONTRACT,
        "teacher_role": str(teacher_role or DEFAULT_TEACHER_ROLE).strip() or DEFAULT_TEACHER_ROLE,
        "max_turns": int(max_turns),
        "policy_max_new_tokens": int(policy_max_new_tokens),
        "generator_revision": ACTIVE_GENERATOR_REVISION,
        "generator_fingerprint": build_generator_fingerprint(config=resolved),
    }


def normalize_protocol_signature(signature: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(signature or {})
    return {
        "signature_schema_version": int(
            payload.get("signature_schema_version", PROTOCOL_SIGNATURE_SCHEMA_VERSION)
            or PROTOCOL_SIGNATURE_SCHEMA_VERSION
        ),
        "protocol_version": str(payload.get("protocol_version") or "").strip(),
        "initial_observation_mode": str(payload.get("initial_observation_mode") or "").strip(),
        "main_rollout_terminal_mode": str(payload.get("main_rollout_terminal_mode") or "").strip(),
        "verifier_contract": str(payload.get("verifier_contract") or "").strip(),
        "teacher_role": str(payload.get("teacher_role") or DEFAULT_TEACHER_ROLE).strip() or DEFAULT_TEACHER_ROLE,
        "max_turns": int(payload.get("max_turns", 0) or 0),
        "policy_max_new_tokens": int(payload.get("policy_max_new_tokens", 0) or 0),
        "generator_revision": str(payload.get("generator_revision") or "").strip(),
        "generator_fingerprint": str(payload.get("generator_fingerprint") or "").strip(),
    }


def extract_protocol_signature(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    signature = dict((payload or {}).get("protocol_signature") or {})
    return normalize_protocol_signature(signature)


def infer_teacher_role_from_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    default: str = DEFAULT_TEACHER_ROLE,
) -> str:
    payload = dict(metadata or {})
    explicit_signature = extract_protocol_signature(payload)
    explicit_role = str(explicit_signature.get("teacher_role") or "").strip()
    if explicit_role:
        return explicit_role
    if bool(payload.get("teacher_annotated")) or bool(payload.get("teacher_rollout_primary_materialized")):
        return TEACHER_ROLE_AUXILIARY

    def _path_contains_teacher_tag(value: Any) -> bool:
        return "teacher_rollout_primary" in str(value or "").strip()

    if _path_contains_teacher_tag(payload.get("path")) or _path_contains_teacher_tag(payload.get("input_path")):
        return TEACHER_ROLE_AUXILIARY
    for field_name in ("source_prepared", "source_jsonl", "source_runtime"):
        field_payload = payload.get(field_name)
        if not isinstance(field_payload, Mapping):
            continue
        if _path_contains_teacher_tag(field_payload.get("path")) or _path_contains_teacher_tag(field_payload.get("resolved_path")):
            return TEACHER_ROLE_AUXILIARY
    return str(default or DEFAULT_TEACHER_ROLE).strip() or DEFAULT_TEACHER_ROLE


def ensure_protocol_signature_matches(
    *,
    actual_signature: Mapping[str, Any] | None,
    expected_signature: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    actual = normalize_protocol_signature(actual_signature)
    expected = normalize_protocol_signature(expected_signature)
    if actual != expected:
        raise ValueError(
            f"{context} protocol signature mismatch. "
            f"expected={json.dumps(expected, ensure_ascii=False, sort_keys=True)} "
            f"actual={json.dumps(actual, ensure_ascii=False, sort_keys=True)}. "
            "Regenerate the artifact or align the official config before continuing."
        )
    return actual
