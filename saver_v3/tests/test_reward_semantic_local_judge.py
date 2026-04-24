from __future__ import annotations

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


class _FakeJudge:
    def __init__(self) -> None:
        self.batch_calls: list[list[tuple[str, str, str]]] = []
        self.single_calls: list[tuple[str, str, str]] = []

    def score_batch(self, queries):
        materialized = [(str(q), str(r), str(p)) for q, r, p in list(queries or [])]
        self.batch_calls.append(materialized)
        return [1.0 for _ in materialized]

    def score(self, *, question: str, reference: str, prediction: str) -> float:
        self.single_calls.append((str(question), str(reference), str(prediction)))
        return 1.0


def _semantic_rollout():
    return {
        "video_id": "fire-1",
        "video_meta": {"fps": 1.0},
        "structured_target": {
            "existence": "anomaly",
            "category": "fire",
            "anomaly_interval_sec": [1.0, 2.0],
            "summary": "A fire starts in the machine room.",
            "rationale": "Flames appear near the machine and continue burning.",
        },
        "scoring_qa_pairs": [
            {
                "type": "trigger_evidence",
                "question": "What visible evidence first makes the anomaly actionable?",
                "answer": "Flames erupt near the machine.",
            }
        ],
        "scoring_evidence_moments": [
            {"moment_id": "m1", "role": "precursor", "description": "Smoke appears before ignition."},
            {"moment_id": "m2", "role": "trigger", "description": "Flames erupt near the machine."},
            {"moment_id": "m3", "role": "confirmation", "description": "The fire keeps burning."},
        ],
        "semantic_answer": {
            "decision": {
                "existence": "anomaly",
                "category": "fire",
                "anomaly_interval_sec": [1.0, 2.0],
            },
            "summary": "A fire starts in the machine room.",
            "rationale": "Flames appear near the machine and continue burning.",
            "event_chain_summary": {
                "precursor": "Smoke appears before ignition.",
                "trigger": "Flames erupt near the machine.",
                "confirmation": "The fire keeps burning.",
            },
            "qa_focus_answers": {
                "existence": "Yes, there is a fire.",
                "category": "The anomaly is fire.",
                "temporal": "The anomaly occurs from 1.0s to 2.0s.",
            },
        },
        "final_answer": {
            "existence": "anomaly",
            "category": "fire",
            "anomaly_interval_sec": [1.0, 2.0],
        },
        "turns": [
            {"tool_name": "verify_hypothesis", "verifier_next_tool": "finalize_case"},
            {"tool_name": "finalize_case"},
        ],
        "state": {
            "finalized_case": {
                "existence": "anomaly",
                "category": "fire",
                "anomaly_interval_sec": [1.0, 2.0],
            }
        },
    }


def test_collect_semantic_queries_uses_targets_and_semantic_answer():
    rollout = _semantic_rollout()

    queries = reward_mod._collect_semantic_queries(rollout, reward_version="timesearch_v4")

    assert "summary" in queries
    assert "trigger_evidence" in queries
    assert "event_chain_summary.trigger" in queries


def test_score_rollout_trace_timesearch_counts_open_ended_semantic_reward():
    rollout = _semantic_rollout()
    judge = _FakeJudge()

    result = reward_mod._score_rollout_trace_timesearch(
        rollout,
        reward_version="timesearch_v4",
        llm_judge=judge,
    )

    assert result["accuracy_reward"] == pytest.approx(1.0, abs=1e-6)
    assert result["accuracy_question_count"] >= 6
    assert result["accuracy_by_family"]["open_ended"] == pytest.approx(1.0, abs=1e-6)
    assert judge.single_calls


def test_score_rollout_trace_timesearch_counts_missing_open_ended_answer_as_zero():
    rollout = _semantic_rollout()
    judge = _FakeJudge()
    rollout["semantic_answer"]["event_chain_summary"]["confirmation"] = ""

    result = reward_mod._score_rollout_trace_timesearch(
        rollout,
        reward_version="timesearch_v4",
        llm_judge=judge,
    )

    assert result["accuracy_question_count"] == 8
    assert result["accuracy_by_family"]["open_ended"] == pytest.approx(0.8, abs=1e-6)
    assert result["accuracy_reward"] == pytest.approx(7.0 / 8.0, abs=1e-6)
    assert len(judge.single_calls) == 4


def test_build_timesearch_reward_funcs_batches_semantic_queries_through_local_judge():
    rollout = _semantic_rollout()
    judge = _FakeJudge()

    funcs = reward_mod.build_timesearch_reward_funcs(
        reward_version="timesearch_v4",
        llm_judge=judge,
    )
    outputs = {
        func.__name__: float(func(rollout_traces=[rollout])[0])
        for func in funcs
    }

    assert outputs["accuracy_reward"] == pytest.approx(1.0, abs=1e-6)
    assert judge.batch_calls
    assert len(judge.batch_calls[0]) >= 3
