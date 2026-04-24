import sys
from types import ModuleType

import pytest

try:
    import torch  # noqa: F401
except Exception:
    fake_torch = ModuleType("torch")
    fake_torch.Tensor = type("Tensor", (), {})
    fake_torch.dtype = type("dtype", (), {})
    fake_torch.device = lambda *_args, **_kwargs: "cpu"
    fake_torch.float32 = "float32"
    fake_torch.float16 = "float16"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.long = "long"
    sys.modules["torch"] = fake_torch

from saver_v3.core import reward as reward_mod


def _base_rollout(*, target, final_answer):
    return {
        "video_id": "vid-1",
        "structured_target": dict(target),
        "final_answer": dict(final_answer),
        "turns": [
            {"tool_name": "verify_hypothesis", "step_index": 1},
            {"tool_name": "finalize_case", "step_index": 2},
        ],
        "state": {"finalized_case": dict(final_answer)},
    }


def test_timesearch_v4_penalizes_anomaly_predicted_as_normal():
    rollout = _base_rollout(
        target={
            "existence": "anomaly",
            "category": "assault",
            "anomaly_interval_sec": [0.0, 2.0],
        },
        final_answer={
            "existence": "normal",
            "category": "normal",
        },
    )

    result = reward_mod._score_rollout_trace_timesearch(rollout, reward_version="timesearch_v4")

    assert result["components"]["anomaly_false_normal_penalty"] == -1.0
    assert result["weighted_components"]["anomaly_false_normal_penalty"] == -1.25
    assert result["accuracy_reward"] == 0.0
    assert result["total_reward"] < 0.0


def test_timesearch_v4_keeps_normal_case_unpenalized():
    rollout = _base_rollout(
        target={
            "existence": "normal",
            "category": "normal",
        },
        final_answer={
            "existence": "normal",
            "category": "normal",
        },
    )

    result = reward_mod._score_rollout_trace_timesearch(rollout, reward_version="timesearch_v4")

    assert result["components"]["anomaly_false_normal_penalty"] == 0.0
    assert result["weighted_components"]["anomaly_false_normal_penalty"] == 0.0
    assert result["accuracy_reward"] == 1.0
    assert result["total_reward"] > 0.0


def test_build_timesearch_reward_funcs_exposes_penalty_to_trainer_path():
    rollout = _base_rollout(
        target={
            "existence": "anomaly",
            "category": "assault",
            "anomaly_interval_sec": [0.0, 2.0],
        },
        final_answer={
            "existence": "normal",
            "category": "normal",
        },
    )

    funcs = reward_mod.build_timesearch_reward_funcs(reward_version="timesearch_v4")
    outputs = {
        func.__name__: float(func(rollout_traces=[rollout])[0])
        for func in funcs
    }
    weights = reward_mod.resolve_reward_component_weights(reward_version="timesearch_v4")
    total = sum(float(weights.get(name, 0.0)) * value for name, value in outputs.items())

    assert "anomaly_false_normal_penalty" in outputs
    assert outputs["anomaly_false_normal_penalty"] == -1.0
    assert total < 0.0


def test_normal_verifier_score_ignores_legacy_recommended_action():
    rollout = {
        "offline_verifier": {
            "primary_status": "complete",
            "recommended_action": "finalize_case",
        }
    }

    details = reward_mod._normal_continuous_verifier_score_details(rollout)

    assert details["next_tool"] == "unknown"
    assert reward_mod._normal_verification_consistency_score(rollout) == pytest.approx(0.6, abs=1e-6)


def test_stage_necessity_reward_ignores_malformed_string_claim():
    rollout = {
        "video_id": "vid-malformed",
        "structured_target": {
            "existence": "anomaly",
            "category": "assault",
            "event_chain_target": {"required_stages": ["trigger"]},
        },
        "final_answer": {
            "existence": "anomaly",
            "category": "assault",
        },
        "turns": [
            {
                "tool_name": "verify_hypothesis",
                "step_index": 1,
                "parsed_tool_call": {
                    "arguments": {
                        "claim": "not-a-dict",
                        "covered_stages": ["trigger"],
                        "verification_decision": "sufficient",
                        "next_tool": "finalize_case",
                    }
                },
            },
            {"tool_name": "finalize_case", "step_index": 2},
        ],
        "state": {"finalized_case": {"existence": "anomaly", "category": "assault"}},
    }

    result = reward_mod._score_rollout_trace_timesearch(rollout, reward_version="timesearch_v4")

    assert isinstance(result["components"]["stage_necessity_reward"], float)
