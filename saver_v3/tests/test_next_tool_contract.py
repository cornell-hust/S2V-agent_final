from __future__ import annotations

import pytest

from saver_v3.core.self_verification import (
    build_policy_self_verification_payload,
    coerce_self_verification_claim_payload,
    parse_self_verification_payload,
)
from saver_v3.teacher.teacher_judge import normalize_teacher_judge_result

try:
    import torch  # noqa: F401
except Exception:
    tools_mod = None
else:
    from saver_v3.core import tools as tools_mod


def test_build_policy_self_verification_payload_uses_next_tool():
    payload = build_policy_self_verification_payload(
        {
            "verification_mode": "stage_check",
            "verification_decision": "sufficient",
            "next_tool": "finalize_case",
            "selected_window_ids": ["w0001"],
            "sufficiency_score": 0.9,
            "necessity_score": 0.7,
            "finalize_readiness_score": 1.0,
        }
    )

    assert payload["next_tool"] == "finalize_case"
    assert "recommended_action" not in payload
    assert "counterfactual_faithfulness" not in payload


def test_parse_self_verification_payload_rejects_legacy_recommended_action():
    with pytest.raises(ValueError, match="recommended_action"):
        parse_self_verification_payload(
            {
                "verification_mode": "final_check",
                "verification_decision": "insufficient",
                "recommended_action": "continue_search",
                "selected_window_ids": ["w0001"],
                "sufficiency_score": 0.3,
                "necessity_score": 0.3,
                "finalize_readiness_score": 0.2,
            }
        )


def test_parse_self_verification_payload_rejects_legacy_verification_mode():
    with pytest.raises(ValueError, match="Legacy verification_mode"):
        parse_self_verification_payload(
            {
                "verification_mode": "full_keep_drop",
                "verification_decision": "insufficient",
                "next_tool": "seek_evidence",
                "selected_window_ids": ["w0001"],
                "sufficiency_score": 0.3,
                "necessity_score": 0.3,
                "finalize_readiness_score": 0.2,
            }
        )


def test_parse_self_verification_payload_coerces_json_string_claim():
    parsed = parse_self_verification_payload(
        {
            "verification_mode": "stage_check",
            "claim": '{"existence":"anomaly","category":"assault"}',
            "selected_window_ids": ["w0001"],
            "verification_decision": "sufficient",
            "next_tool": "finalize_case",
            "sufficiency_score": 0.9,
            "necessity_score": 0.8,
            "finalize_readiness_score": 1.0,
        }
    )

    assert parsed["claim"] == {"existence": "anomaly", "category": "assault"}


def test_coerce_self_verification_claim_payload_drops_unparseable_string():
    claim = coerce_self_verification_claim_payload("not-json")

    assert claim == {}


def test_coerce_self_verification_claim_payload_accepts_generic_anomaly_category():
    claim = coerce_self_verification_claim_payload(
        {"existence": "anomaly", "category": "anomaly"},
    )

    assert claim == {"existence": "anomaly", "category": "anomaly"}


def test_coerce_self_verification_claim_payload_uses_fallback_for_non_dict_payload():
    claim = coerce_self_verification_claim_payload(
        "not-json",
        fallback_claim={"existence": "anomaly", "category": "assault"},
    )

    assert claim == {"existence": "anomaly", "category": "assault"}


def test_normalize_verification_arguments_coerces_json_string_claim():
    if tools_mod is None:
        pytest.skip("torch required for tools import")
    normalized = tools_mod._normalize_verification_arguments(
        {
            "verification_mode": "stage_check",
            "claim": '{"existence":"anomaly","category":"assault"}',
        }
    )

    assert normalized["claim"] == {"existence": "anomaly", "category": "assault"}


def test_normalize_verification_arguments_keeps_generic_anomaly_category():
    if tools_mod is None:
        pytest.skip("torch required for tools import")
    normalized = tools_mod._normalize_verification_arguments(
        {
            "verification_mode": "stage_check",
            "claim": {"existence": "anomaly", "category": "anomaly"},
        },
        fallback_claim={"existence": "anomaly", "category": "robbery"},
    )

    assert normalized["claim"] == {"existence": "anomaly", "category": "anomaly"}


def test_teacher_judge_rationale_sanitizes_legacy_action_words():
    normalized = normalize_teacher_judge_result(
        {
            "teacher_judge_decision": "insufficient",
            "teacher_judge_scores": {
                "sufficiency": 0.3,
                "necessity": 0.3,
                "finalize_readiness": 0.0,
            },
            "teacher_judge_rationale": (
                "The policy should continue_search now, refine_evidence, and should finalize later."
            ),
        }
    )

    rationale = normalized["teacher_judge_rationale"]
    assert "continue_search" not in rationale
    assert "refine_evidence" not in rationale
    assert "seek_evidence" in rationale
    assert "finalize_case" in rationale
