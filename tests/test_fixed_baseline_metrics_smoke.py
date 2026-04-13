import json
import tempfile
import unittest
from pathlib import Path

from saver_v3.inference.fixed_baseline_eval import adapt_fixed_baseline_prediction_to_rollout
from saver_v3.metrics.legacy_metrics import summarize_saver_metrics
from saver_v3.metrics.offline_scoring import ReferenceDataProvider
from saver_v3.metrics.semantic_metrics import evaluate_semantic_rollouts


def _runtime_row() -> dict:
    return {
        "video_id": "video-metric-001",
        "file_name": "video-metric-001.mp4",
        "source_dataset": "MSAD",
        "split": "test",
        "video_path": "/tmp/video-metric-001.mp4",
        "video_meta": {"duration_sec": 6.0, "fps": 2.0},
        "label": {"is_anomaly": True, "category": "assault", "severity": 4, "hard_normal": False},
        "temporal": {
            "anomaly_interval_sec": [1.0, 4.0],
            "precursor_interval_sec": [0.0, 1.0],
            "earliest_actionable_sec": 1.0,
        },
        "structured_target": {
            "existence": "anomaly",
            "category": "assault",
            "severity": 4,
            "anomaly_interval_sec": [1.0, 4.0],
            "precursor_interval_sec": [0.0, 1.0],
            "counterfactual_type": "remove_actor_interaction",
            "covered_stages": ["precursor", "trigger", "confirmation"],
            "stage_selected_moment_ids": {
                "precursor": ["ev1"],
                "trigger": ["ev2"],
                "confirmation": ["ev3"],
            },
        },
        "tool_io": {
            "oracle_windows_sec": [
                {"moment_id": "ev1", "role": "precursor", "window": [0.0, 1.0]},
                {"moment_id": "ev2", "role": "trigger", "window": [1.0, 2.0]},
                {"moment_id": "ev3", "role": "confirmation", "window": [2.0, 4.0]},
            ]
        },
        "evidence": {
            "evidence_moments": [
                {"moment_id": "ev1", "role": "precursor", "description": "Approach"},
                {"moment_id": "ev2", "role": "trigger", "description": "Attack"},
                {"moment_id": "ev3", "role": "confirmation", "description": "Victim remains down"},
            ]
        },
        "qa_pairs": [
            {"type": "existence", "answer": "Yes, there is an assault occurring."},
            {"type": "category", "answer": "The anomaly is assault."},
            {"type": "temporal", "answer": "The anomaly occurs from 1.000s to 4.000s."},
        ],
    }


class FixedBaselineMetricsSmokeTests(unittest.TestCase):
    def test_metrics_and_semantic_eval_accept_adapted_direct_baseline_record(self) -> None:
        row = _runtime_row()
        prediction = {
            "decision": {
                "existence": "anomaly",
                "category": "assault",
                "severity": 4,
                "anomaly_interval_sec": [1.0, 4.0],
                "precursor_interval_sec": [0.0, 1.0],
                "counterfactual_type": "remove_actor_interaction",
            },
            "semantic_answer": {
                "decision": {"existence": "anomaly", "category": "assault"},
                "summary": "An assault occurs.",
                "rationale": "Visible attack in preview frames.",
                "event_chain_summary": {
                    "precursor": "Approach.",
                    "trigger": "Attack.",
                    "confirmation": "Victim remains down.",
                },
                "qa_focus_answers": {
                    "existence": "Yes, there is an assault occurring.",
                    "category": "The anomaly is assault.",
                    "temporal": "The anomaly occurs from 1.000s to 4.000s.",
                },
            },
            "evidence_topk": [
                {"rank": 1, "start_sec": 0.0, "end_sec": 1.0, "role": "precursor", "description": "Approach"},
                {"rank": 2, "start_sec": 1.0, "end_sec": 2.0, "role": "trigger", "description": "Attack"},
                {"rank": 3, "start_sec": 2.0, "end_sec": 4.0, "role": "confirmation", "description": "Victim remains down"},
            ],
        }
        rollout_record = adapt_fixed_baseline_prediction_to_rollout(
            row,
            prediction,
            raw_response_text=json.dumps(prediction),
            parse_ok=True,
            parse_error=None,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = Path(tmp_dir) / "runtime.jsonl"
            data_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            reference_data = ReferenceDataProvider(data_path=data_path)
            summary = summarize_saver_metrics([rollout_record], reference_data=reference_data)
            semantic = evaluate_semantic_rollouts([rollout_record], data_path=data_path, metrics=["qa_accuracy"])

        self.assertIn("existence_accuracy", summary)
        self.assertIn("temporal_miou", summary)
        self.assertIn("event_chain_f1", summary)
        self.assertIn("evidence_f1_at_3", summary)
        self.assertIn("qa_accuracy_overall", semantic)


if __name__ == "__main__":
    unittest.main()
