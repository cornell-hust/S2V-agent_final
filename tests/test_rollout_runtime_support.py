import unittest

from saver_v3.data.runtime_items import build_runtime_item_from_compact_trace_row
from saver_v3.core.environment import SaverVideoInteraction
from saver_v3.core.schema import SaverEnvironmentState
from saver_v3.core.tool_registry import execute_tool_call


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
        ledger_entry = state.evidence_ledger[0]
        fallback_reason = (
            ((ledger_entry.get("metadata") or {}).get("proposal_fallback_reason"))
            or ledger_entry.get("proposal_fallback_reason")
        )
        self.assertIn(fallback_reason, {"seek_evidence_uniform_fallback", "missing_feature_cache"})

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

    def test_finalize_case_normalizes_runtime_ids_to_gt_moment_ids(self) -> None:
        state = SaverEnvironmentState()
        state.evidence_ledger = [
            {
                "window_id": "w0002",
                "evidence_id": "e0002",
                "moment_id": "m1",
                "role": "trigger",
                "start_sec": 4.9,
                "end_sec": 8.9,
            },
            {
                "window_id": "w0003",
                "evidence_id": "e0003",
                "moment_id": None,
                "role": "confirmation",
                "start_sec": 8.8,
                "end_sec": 12.1,
            },
        ]
        state.active_evidence_window_ids = ["w0002", "w0003"]
        multimodal_cache = {
            "structured_target": {
                "existence": "anomaly",
                "category": "assault",
                "event_chain_target": {
                    "stage_to_moment_ids": {
                        "trigger": ["ev2"],
                        "confirmation": ["ev4"],
                    }
                },
            },
            "tool_io": {
                "oracle_windows_sec": [
                    {"moment_id": "ev2", "role": "trigger", "window": [5.0, 9.0]},
                    {"moment_id": "ev4", "role": "confirmation", "window": [9.0, 12.0]},
                ],
                "finalize_case_schema": {
                    "type": "object",
                    "properties": {
                        "existence": {"type": "string"},
                        "category": {"type": "string"},
                        "evidence_moment_ids": {"type": "array", "items": {"type": "string"}},
                        "stage_selected_moment_ids": {"type": "object"},
                    },
                    "required": ["existence", "category", "evidence_moment_ids", "stage_selected_moment_ids"],
                },
            },
        }

        _, state, _ = execute_tool_call(
            "finalize_case",
            {
                "existence": "anomaly",
                "category": "assault",
                "evidence_moment_ids": ["m1", "e0002", "e0003"],
                "stage_selected_moment_ids": {
                    "trigger": ["m1", "e0002"],
                    "confirmation": ["e0003"],
                },
            },
            multimodal_cache,
            state,
        )

        self.assertEqual(state.finalized_case["evidence_moment_ids"], ["ev2", "ev4"])
        self.assertEqual(state.finalized_case["stage_selected_moment_ids"], {"trigger": ["ev2"], "confirmation": ["ev4"]})
        self.assertEqual(state.finalized_case["covered_stages"], ["trigger", "confirmation"])


if __name__ == "__main__":
    unittest.main()



class RolloutEvalMaterializedConfigTests(unittest.TestCase):
    def test_step_rollout_eval_config_parses_materialized_items(self) -> None:
        from saver_v3.inference.rollout_eval import StepRolloutEvalConfig

        config = StepRolloutEvalConfig.from_mapping({
            "base_model": "/models/policy",
            "io": {
                "data_path": "/data/runtime_eval.jsonl",
                "data_root": "/data",
                "materialized_items_path": "/data/runtime_eval.materialized.jsonl",
                "require_materialized_cache": True,
                "output_dir": "/tmp/out",
                "include_splits": "test",
            },
        })
        self.assertEqual(config.materialized_items_path, "/data/runtime_eval.materialized.jsonl")
        self.assertTrue(config.require_materialized_cache)
