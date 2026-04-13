import unittest

from saver_v3.inference.fixed_baseline_eval import adapt_fixed_baseline_prediction_to_rollout


def _sample_record() -> dict:
    return {
        "video_id": "video-001",
        "file_name": "video-001.mp4",
        "source_dataset": "MSAD",
        "split": "test",
        "video_meta": {"duration_sec": 6.0},
    }


class FixedBaselineAdapterTests(unittest.TestCase):
    def test_adapter_emits_scorer_friendly_record_without_fake_turns(self) -> None:
        record = adapt_fixed_baseline_prediction_to_rollout(
            _sample_record(),
            {
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
                    "rationale": "Visible attack.",
                    "event_chain_summary": {
                        "precursor": "Approach.",
                        "trigger": "Attack.",
                        "confirmation": "Victim remains down.",
                    },
                    "qa_focus_answers": {
                        "existence": "Yes, there is an anomaly.",
                        "category": "The anomaly is assault.",
                        "temporal": "The anomaly occurs from 1.0s to 4.0s.",
                    },
                },
                "evidence_topk": [
                    {"rank": 1, "start_sec": 0.0, "end_sec": 1.0, "role": "precursor", "description": "Approach"},
                    {"rank": 2, "start_sec": 1.0, "end_sec": 2.0, "role": "trigger", "description": "Attack"},
                ],
            },
            raw_response_text='{"ok": true}',
            parse_ok=True,
            parse_error=None,
        )

        self.assertEqual(record["turns"], [])
        self.assertEqual(record["num_turns"], 0)
        self.assertNotIn("finalized_case", record["state"])
        self.assertEqual(record["state"]["active_evidence_window_ids"], ["baseline_ev1", "baseline_ev2"])
        self.assertEqual(record["final_answer"]["covered_stages"], ["precursor", "trigger"])
        self.assertEqual(record["final_answer"]["stage_selected_moment_ids"]["precursor"], ["baseline_ev1"])


if __name__ == "__main__":
    unittest.main()
