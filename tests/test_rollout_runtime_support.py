import unittest

from saver_v3.data.runtime_items import build_runtime_item_from_compact_trace_row
from saver_v3.core.environment import SaverVideoInteraction
from saver_v3.core.schema import SaverEnvironmentState


def _sample_row() -> dict:
    return {
        "schema_version": 1,
        "prepared_format": "compact_trace_v2",
        "video_id": "video-rt-001",
        "video_path": "/tmp/video-rt-001.mp4",
        "split": "train",
        "source": "oracle_sft_compact_trace_v2",
        "video_meta": {"duration_sec": 12.0, "fps": 2.0},
        "agent_task": {
            "task_prompt": "Inspect the clip, gather evidence, and finalize the anomaly decision.",
            "success_criteria": ["Use tools before finalizing."],
        },
        "structured_target": {"existence": "anomaly", "category": "people_falling"},
        "tool_io": {
            "finalize_case_schema": {
                "type": "object",
                "properties": {
                    "existence": {"type": "string", "enum": ["normal", "anomaly"]},
                    "category": {"type": "string"},
                },
                "required": ["existence", "category"],
            }
        },
        "oracle_trajectory": [],
        "oracle_final_decision": {"existence": "anomaly", "category": "people_falling"},
    }


class RolloutRuntimeSupportTests(unittest.TestCase):
    def test_build_runtime_item_from_compact_trace_row_builds_messages_and_multimodal_cache(self) -> None:
        item = build_runtime_item_from_compact_trace_row(_sample_row())

        self.assertEqual(item["video_id"], "video-rt-001")
        self.assertEqual(len(item["messages"]), 2)
        self.assertEqual(item["messages"][0]["role"], "system")
        self.assertEqual(item["messages"][1]["role"], "user")
        self.assertEqual(item["messages"][1]["content"][0]["type"], "video")
        self.assertEqual(item["messages"][1]["content"][0]["video"], "/tmp/video-rt-001.mp4")
        self.assertIn("Inspect the clip", item["messages"][1]["content"][1]["text"])

        multimodal_cache = item["multimodal_cache"]
        self.assertEqual(multimodal_cache["video_path"], "/tmp/video-rt-001.mp4")
        self.assertEqual(multimodal_cache["duration"], 12.0)
        self.assertEqual(multimodal_cache["fps"], 2.0)
        self.assertEqual(multimodal_cache["question"], _sample_row()["agent_task"]["task_prompt"])
        self.assertEqual(
            multimodal_cache["tool_io"]["allowed_tools"],
            ["scan_timeline", "seek_evidence", "verify_hypothesis", "finalize_case"],
        )
        self.assertEqual(len(multimodal_cache["tool_io"]["function_schemas"]), 4)
        self.assertIn("summary", multimodal_cache["tool_io"]["finalize_case_schema"]["properties"])

    def test_saver_video_interaction_executes_minimal_tool_chain_without_gpu(self) -> None:
        item = build_runtime_item_from_compact_trace_row(_sample_row())
        environment = SaverVideoInteraction()
        multimodal_cache = item["multimodal_cache"]
        state = SaverEnvironmentState()

        observations, dones, valid_actions, is_search, next_states = environment.execute_predictions(
            [
                '<tool_call>{"name":"scan_timeline","arguments":{"start_sec":0.0,"end_sec":6.0,"num_frames":3}}</tool_call>'
            ],
            [multimodal_cache],
            [state],
            [True],
        )
        state = next_states[0]

        self.assertEqual(observations[0]["name"], "scan_timeline")
        self.assertEqual(dones, [0])
        self.assertEqual(valid_actions, [1])
        self.assertEqual(is_search, [1])
        self.assertEqual(len(state.visited_windows), 1)

        observations, dones, valid_actions, is_search, next_states = environment.execute_predictions(
            [
                '<tool_call>{"name":"seek_evidence","arguments":{"query":"person slipping near the aisle","role":"trigger","start_sec":2.0,"end_sec":8.0,"num_frames":4}}</tool_call>'
            ],
            [multimodal_cache],
            [state],
            [True],
        )
        state = next_states[0]
        evidence_window_id = state.evidence_ledger[0]["window_id"]

        self.assertEqual(observations[0]["name"], "seek_evidence")
        self.assertEqual(valid_actions, [1])
        self.assertEqual(is_search, [1])
        self.assertEqual(len(state.evidence_ledger), 1)
        self.assertEqual(
            state.evidence_ledger[0]["metadata"]["proposal_fallback_reason"],
            "seek_evidence_uniform_fallback",
        )

        observations, dones, valid_actions, is_search, next_states = environment.execute_predictions(
            [
                "".join(
                    [
                        '<tool_call>{"name":"verify_hypothesis","arguments":{',
                        '"verification_mode":"final_check",',
                        '"claim":{"existence":"anomaly","category":"people_falling"},',
                        f'"selected_window_ids":["{evidence_window_id}"],',
                        '"covered_stages":["trigger","confirmation"],',
                        '"missing_required_stages":[],',
                        '"verification_decision":"sufficient",',
                        '"recommended_action":"finalize",',
                        '"sufficiency_score":0.95,',
                        '"necessity_score":0.8,',
                        '"finalize_readiness_score":0.97,',
                        '"counterfactual_faithfulness":0.88',
                        '}}</tool_call>',
                    ]
                )
            ],
            [multimodal_cache],
            [state],
            [True],
        )
        state = next_states[0]

        self.assertEqual(observations[0]["name"], "verify_hypothesis")
        self.assertEqual(valid_actions, [1])
        self.assertEqual(is_search, [1])
        self.assertEqual(state.active_evidence_window_ids, [evidence_window_id])
        self.assertEqual(state.verification_records[0]["recommended_action"], "finalize")

        observations, dones, valid_actions, is_search, next_states = environment.execute_predictions(
            [
                '<tool_call>{"name":"finalize_case","arguments":{"existence":"anomaly","category":"fall","summary":"A person falls in the aisle."}}</tool_call>'
            ],
            [multimodal_cache],
            [state],
            [True],
        )
        state = next_states[0]

        self.assertEqual(observations[0]["name"], "finalize_case")
        self.assertEqual(valid_actions, [1])
        self.assertEqual(is_search, [1])
        self.assertEqual(state.finalized_case, {"existence": "anomaly", "category": "people_falling"})
        self.assertEqual(state.finalized_semantic_answer["summary"], "A person falls in the aisle.")


if __name__ == "__main__":
    unittest.main()
