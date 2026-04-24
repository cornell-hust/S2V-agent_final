from __future__ import annotations

from types import SimpleNamespace

import pytest

from saver_v3.core.event_chain import extract_stage_annotation_from_turn
from saver_v3.data.training_data import _compute_turn_level_advantages
from saver_v3.metrics.legacy_metrics import summarize_saver_metrics


def test_event_chain_annotation_ignores_legacy_recommended_action():
    annotation = extract_stage_annotation_from_turn(
        {
            "covered_stages": ["trigger"],
            "missing_required_stages": [],
            "recommended_action": "finalize_case",
        }
    )

    assert annotation["covered_stages"] == ["trigger"]
    assert annotation["next_tool"] == ""


def test_turn_level_advantages_ignore_legacy_finalize_recommendation():
    advantages = _compute_turn_level_advantages(
        {
            "group_advantage": 0.0,
            "turns": [
                {
                    "tool_name": "verify_hypothesis",
                    "step_index": 1,
                    "valid_action": True,
                    "verifier_recommended_action": "finalize_case",
                },
                {
                    "tool_name": "seek_evidence",
                    "step_index": 2,
                    "valid_action": True,
                    "parsed_tool_call": {"arguments": {"query": "new clue"}},
                    "state_delta": {"new_visited_windows": [{"window_id": "w0001"}]},
                },
            ],
        },
        gamma=1.0,
        alpha=1.0,
        search_bonus=1.0,
        evidence_bonus=0.0,
        finalize_bonus=0.0,
        invalid_penalty=0.0,
    )

    assert advantages[1]["turn_credit"] == pytest.approx(1.0, abs=1e-6)


def test_legacy_metrics_ignore_legacy_finalize_recommendation():
    reference_data = SimpleNamespace(
        by_video_id={
            "vid-1": {
                "structured_target": {"existence": "normal", "category": "normal"},
                "video_meta": {"duration_sec": 10.0},
            }
        }
    )
    record = {
        "video_id": "vid-1",
        "num_turns": 1,
        "turns": [
            {
                "tool_name": "verify_hypothesis",
                "step_index": 1,
                "valid_action": True,
                "verifier_recommended_action": "finalize_case",
            }
        ],
        "state": {
            "finalized_case": {"existence": "normal", "category": "normal"},
            "visited_windows": [],
        },
        "final_answer": {"existence": "normal", "category": "normal"},
    }

    summary = summarize_saver_metrics([record], reference_data=reference_data)

    assert summary["verify_finalize_followthrough_rate"] == pytest.approx(0.0, abs=1e-6)
