from __future__ import annotations

from copy import deepcopy

import pytest

pytest.importorskip("torch")

from convert_to_saver_agent import FINALIZE_CASE_SCHEMA, convert_record
from saver_v3.core.self_verification import (
    POLICY_SELF_VERIFICATION_CLAIM_KEYS,
    build_self_verification_tool_schema,
)
from saver_v3.data.protocol_signature import build_protocol_signature
from saver_v3.data.runtime_contract import RUNTIME_CONTRACT_VERSION, validate_runtime_record_contract


def _raw_anomaly_record(*, include_precursor_moment: bool) -> dict:
    evidence_moments = []
    if include_precursor_moment:
        evidence_moments.append(
            {
                "moment_id": "ev0",
                "role": "precursor",
                "description": "A person approaches the machine before ignition.",
                "start_frame": 1,
                "end_frame": 9,
            }
        )
    evidence_moments.extend(
        [
            {
                "moment_id": "ev1",
                "role": "trigger",
                "description": "Flames appear near the machine.",
                "start_frame": 10,
                "end_frame": 12,
            },
            {
                "moment_id": "ev2",
                "role": "confirmation",
                "description": "Fire persists after the initial ignition.",
                "start_frame": 13,
                "end_frame": 16,
            },
        ]
    )
    return {
        "video_id": "vid-fire-1",
        "file_name": "vid-fire-1.mp4",
        "video_path": "data/fire/vid-fire-1.mp4",
        "source_dataset": "MSAD",
        "source_split": "anomaly",
        "split": "train",
        "frame_index_base": 1,
        "video_meta": {
            "fps": 1.0,
            "width": 640,
            "height": 360,
            "total_frames": 30,
            "duration_sec": 30.0,
        },
        "scene": {"scenario": "warehouse"},
        "key_objects": ["machine", "worker"],
        "label": {"is_anomaly": True, "category": "fire", "severity": 2, "hard_normal": False},
        "temporal": {
            "anomaly_interval_frames": [10, 20],
            "precursor_interval_frames": [1, 9],
            "earliest_alert_frame": 10,
        },
        "evidence": {"evidence_moments": evidence_moments},
        "language": {
            "summary": "A fire starts in the warehouse.",
            "rationale": "The machine ignites and continues burning.",
        },
        "qa_pairs": [
            {
                "type": "precursor_temporal",
                "question": "When do precursor cues appear?",
                "answer": "Precursor cues appear from frame 1 to frame 9.",
            }
        ],
        "provenance": {},
        "qwen_preannotation": {},
    }


def test_active_finalize_and_verifier_schemas_drop_removed_fields():
    finalize_properties = dict(FINALIZE_CASE_SCHEMA.get("properties") or {})
    finalize_required = list(FINALIZE_CASE_SCHEMA.get("required") or [])
    verify_claim_properties = dict(
        (((build_self_verification_tool_schema().get("properties") or {}).get("claim") or {}).get("properties") or {})
    )

    assert "precursor_interval_sec" not in finalize_properties
    assert "earliest_actionable_sec" not in finalize_properties
    assert "precursor_interval_sec" not in finalize_required
    assert "earliest_actionable_sec" not in verify_claim_properties
    assert POLICY_SELF_VERIFICATION_CLAIM_KEYS == ("existence", "category")


def test_agent_train_conversion_omits_removed_fields_and_stops_interval_only_precursor_promotion():
    converted = convert_record(
        _raw_anomaly_record(include_precursor_moment=False),
        mode="agent_train",
    )

    structured_target = dict(converted.get("structured_target") or {})
    finalize_schema = dict(((converted.get("tool_io") or {}).get("finalize_case_schema")) or {})
    finalize_properties = dict(finalize_schema.get("properties") or {})
    finalize_required = list(finalize_schema.get("required") or [])
    finalize_policy = dict(((converted.get("search_supervision") or {}).get("finalize_policy")) or {})
    event_chain_target = dict(structured_target.get("event_chain_target") or {})

    assert "precursor_interval_sec" not in structured_target
    assert "earliest_actionable_sec" not in structured_target
    assert "precursor_interval_sec" not in finalize_properties
    assert "earliest_actionable_sec" not in finalize_properties
    assert "precursor_interval_sec" not in finalize_required
    assert "earliest_actionable_sec" not in finalize_policy
    assert "precursor" not in list(event_chain_target.get("required_stages") or [])


def test_oracle_sft_conversion_omits_removed_fields_from_verify_claim_and_final_decision():
    converted = convert_record(
        _raw_anomaly_record(include_precursor_moment=False),
        mode="oracle_sft",
    )

    oracle_sft = dict(converted.get("oracle_sft") or {})
    trajectory = list(oracle_sft.get("trajectory") or [])
    verify_step = next(step for step in trajectory if str(step.get("tool") or "") == "verify_hypothesis")
    finalize_step = next(step for step in trajectory if str(step.get("tool") or "") == "finalize_case")
    verify_claim = dict((verify_step.get("arguments") or {}).get("claim") or {})
    final_decision = dict(oracle_sft.get("final_decision") or {})
    finalize_arguments = dict((finalize_step.get("arguments") or {}) or {})

    assert "earliest_actionable_sec" not in verify_claim
    assert "precursor_interval_sec" not in final_decision
    assert "earliest_actionable_sec" not in final_decision
    assert "precursor_interval_sec" not in finalize_arguments
    assert "earliest_actionable_sec" not in finalize_arguments


def test_runtime_contract_rejects_removed_fields_in_active_structured_target_and_finalize_schema():
    row = {
        "runtime_contract_version": int(RUNTIME_CONTRACT_VERSION),
        "protocol_signature": build_protocol_signature(),
        "video_id": "vid1",
        "video_path": "/tmp/example.mp4",
        "structured_target": {
            "existence": "normal",
            "category": "normal",
            "precursor_interval_sec": None,
        },
        "tool_io": {
            "finalize_case_schema": {
                "type": "object",
                "properties": {
                    "existence": {"type": "string"},
                    "category": {"type": "string"},
                    "precursor_interval_sec": {"oneOf": [{"type": "null"}]},
                },
                "required": ["existence", "category", "precursor_interval_sec"],
            }
        },
    }

    with pytest.raises(ValueError, match="removed contract field"):
        validate_runtime_record_contract(row)


def test_runtime_contract_rejects_removed_fields_in_search_supervision_finalize_policy():
    row = {
        "runtime_contract_version": int(RUNTIME_CONTRACT_VERSION),
        "protocol_signature": build_protocol_signature(),
        "video_id": "vid1",
        "video_path": "/tmp/example.mp4",
        "structured_target": {"existence": "normal", "category": "normal"},
        "tool_io": {
            "finalize_case_schema": deepcopy(FINALIZE_CASE_SCHEMA),
        },
        "search_supervision": {
            "finalize_policy": {
                "earliest_actionable_sec": 1.0,
                "minimal_sufficient_moment_ids": [],
                "minimal_sufficient_stage_order": [],
            }
        },
    }

    with pytest.raises(ValueError, match="search_supervision.finalize_policy"):
        validate_runtime_record_contract(row)
