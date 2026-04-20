import copy
import inspect
import unittest
from unittest import mock

from saver_v3.core.counterfactual_verification import (
    _resolve_counterfactual_branch_order,
    resolve_selected_window_ids_for_fecv,
    run_counterfactual_verification_batch,
)
from saver_v3.core.reward import (
    _compute_accuracy_breakdown,
    _score_rollout_trace_timesearch,
    _timesearch_fecv_diagnostics,
    _timesearch_fecv_reward,
    build_timesearch_reward_funcs,
)
from saver_v3.rl import timesearch_aligned_grpo_trainer


class _ReplayBatchPolicy:
    def __init__(self, response_batches):
        self.response_batches = [list(batch) for batch in response_batches]
        self.calls = []

    def generate_from_messages_batch(self, messages_batch):
        recorded_batch = []
        for messages in list(messages_batch or []):
            recorded_batch.append([dict(message) for message in list(messages or [])])
        self.calls.append(recorded_batch)
        if not self.response_batches:
            raise AssertionError("No prepared replay batch responses remain.")
        responses = list(self.response_batches.pop(0))
        if len(responses) != len(recorded_batch):
            raise AssertionError(
                f"Prepared replay batch size {len(responses)} does not match request batch size {len(recorded_batch)}."
            )
        return responses


class CounterfactualOnlineCoreTests(unittest.TestCase):
    def test_normal_skip_fecv_diagnostics_use_restraint_reward(self) -> None:
        target = {"existence": "normal", "category": "normal"}
        profile = {
            "counterfactual_profile_source": "normal_skip_v1",
            "selection_metadata": {
                "normalized_branch_profile": "normal_skip_v1",
                "selected_window_count": 0,
                "selected_record_count": 0,
                "selected_by_stage": {"precursor": [], "trigger": [], "confirmation": []},
                "full_selected_parse_mode": "skipped_normal",
                "full_selected_unavailable_reason": "normal_sample_skipped",
            },
            "stage_packages": {
                "selected_window_ids": [],
                "selected_by_stage": {"precursor": [], "trigger": [], "confirmation": []},
            },
        }
        rollout_trace = {
            "structured_target": dict(target),
            "final_answer": dict(target),
            "state": {"finalized_case": dict(target)},
            "turns": [{"tool_name": "scan_timeline"}],
        }
        diagnostics = _timesearch_fecv_diagnostics(
            profile,
            target=target,
            rollout_trace=rollout_trace,
        )
        self.assertEqual(diagnostics["branch_profile"], "normal_skip_v1")
        self.assertEqual(diagnostics["normal_reward_mode"], "restraint_v4")
        self.assertEqual(diagnostics["normal_case_type"], "easy_normal")
        self.assertEqual(diagnostics["easy_normal_sample_loss_multiplier"], 0.2)
        self.assertEqual(diagnostics["normal_evidence_tool_turn_count"], 1)
        self.assertEqual(diagnostics["normal_restraint_reward"], 1.0)
        self.assertEqual(diagnostics["evidence_faithfulness_reward"], 1.0)
        self.assertEqual(diagnostics["normal_search_restraint_score"], 1.0)
        self.assertEqual(diagnostics["normal_window_restraint_score"], 1.0)
        self.assertEqual(diagnostics["normal_verification_consistency_score"], 1.0)
        self.assertEqual(diagnostics["normal_query_alignment_score"], 1.0)
        self.assertEqual(diagnostics["normal_continuous_verifier_score"], 0.5)
        self.assertEqual(diagnostics["normal_verifier_primary_status"], "unknown")
        self.assertEqual(diagnostics["normal_verifier_recommended_action"], "unknown")
        self.assertEqual(diagnostics["normal_verifier_base_status_score"], 0.5)
        self.assertEqual(diagnostics["normal_verifier_action_offset"], 0.0)
        self.assertEqual(diagnostics["normal_continuous_verifier_score_before_action"], 0.5)
        self.assertEqual(diagnostics["normal_continuous_verifier_score_after_action"], 0.5)
        self.assertEqual(diagnostics["normal_provenance_score"], 0.4)
        self.assertEqual(diagnostics["normal_provenance_source_bucket"], "active_evidence_or_unknown_or_normal_sample_skipped")
        self.assertAlmostEqual(diagnostics["normal_grounded_local_score"], 0.82, places=6)

    def test_normal_skip_scored_rollout_penalizes_excess_search(self) -> None:
        target = {"existence": "normal", "category": "normal"}
        light_profile = {
            "counterfactual_profile_source": "normal_skip_v1",
            "selection_metadata": {
                "normalized_branch_profile": "normal_skip_v1",
                "selected_window_count": 0,
                "selected_record_count": 0,
                "selected_by_stage": {"precursor": [], "trigger": [], "confirmation": []},
                "full_selected_parse_mode": "skipped_normal",
                "full_selected_unavailable_reason": "normal_sample_skipped",
            },
            "stage_packages": {
                "selected_window_ids": [],
                "selected_by_stage": {"precursor": [], "trigger": [], "confirmation": []},
            },
        }
        heavy_profile = {
            "counterfactual_profile_source": "normal_skip_v1",
            "selection_metadata": {
                "normalized_branch_profile": "normal_skip_v1",
                "selected_window_count": 3,
                "selected_record_count": 3,
                "selected_by_stage": {"precursor": [], "trigger": [], "confirmation": []},
                "full_selected_parse_mode": "skipped_normal",
                "full_selected_unavailable_reason": "normal_sample_skipped",
            },
            "stage_packages": {
                "selected_window_ids": ["w1", "w2", "w3"],
                "selected_by_stage": {"precursor": [], "trigger": [], "confirmation": []},
            },
        }
        light_rollout = {
            "structured_target": dict(target),
            "final_answer": dict(target),
            "state": {"finalized_case": dict(target)},
            "counterfactual_profile": light_profile,
            "turns": [{"tool_name": "scan_timeline"}],
        }
        heavy_rollout = {
            "structured_target": dict(target),
            "final_answer": dict(target),
            "state": {"finalized_case": dict(target)},
            "counterfactual_profile": heavy_profile,
            "turns": [
                {"tool_name": "scan_timeline"},
                {
                    "tool_name": "verify_hypothesis",
                    "verifier_recommended_action": "continue_search",
                    "verifier_failure_reasons": ["selected_evidence_not_sufficient"],
                },
                {"tool_name": "scan_timeline"},
            ],
        }
        light_score = _score_rollout_trace_timesearch(light_rollout, reward_version="timesearch_v3")
        heavy_score = _score_rollout_trace_timesearch(heavy_rollout, reward_version="timesearch_v3")
        self.assertEqual(light_score["components"]["fecv_evidence_faithfulness_reward"], 1.0)
        self.assertEqual(light_score["fecv_normal_case_type"], "easy_normal")
        self.assertEqual(light_score["fecv_easy_normal_sample_loss_multiplier"], 0.2)
        self.assertEqual(heavy_score["fecv_normal_case_type"], "suspicious_normal")
        self.assertEqual(heavy_score["fecv_easy_normal_sample_loss_multiplier"], 1.0)
        self.assertEqual(heavy_score["fecv_normal_verifier_primary_status"], "unknown")
        self.assertEqual(heavy_score["fecv_normal_verifier_recommended_action"], "continue_search")
        self.assertEqual(heavy_score["fecv_normal_verifier_base_status_score"], 0.5)
        self.assertEqual(heavy_score["fecv_normal_verifier_action_offset"], -0.3)
        self.assertEqual(heavy_score["fecv_normal_continuous_verifier_score_before_action"], 0.5)
        self.assertEqual(heavy_score["fecv_normal_continuous_verifier_score_after_action"], 0.2)
        self.assertAlmostEqual(heavy_score["components"]["fecv_evidence_faithfulness_reward"], 0.4515, places=6)
        self.assertGreater(light_score["total_reward"], heavy_score["total_reward"])

    def test_normal_skip_diagnostics_still_track_trigger_stage_selection(self) -> None:
        target = {"existence": "normal", "category": "normal"}
        no_stage_profile = {
            "counterfactual_profile_source": "normal_skip_v1",
            "selection_metadata": {
                "normalized_branch_profile": "normal_skip_v1",
                "selected_window_count": 1,
                "selected_record_count": 1,
                "selected_by_stage": {"precursor": [], "trigger": [], "confirmation": []},
                "full_selected_parse_mode": "skipped_normal",
                "full_selected_unavailable_reason": "normal_sample_skipped",
            },
            "stage_packages": {
                "selected_window_ids": ["w1"],
                "selected_by_stage": {"precursor": [], "trigger": [], "confirmation": []},
            },
        }
        trigger_profile = copy.deepcopy(no_stage_profile)
        trigger_profile["selection_metadata"]["selected_by_stage"] = {"precursor": [], "trigger": ["w1"], "confirmation": []}
        trigger_profile["stage_packages"]["selected_by_stage"] = {"precursor": [], "trigger": ["w1"], "confirmation": []}
        base_rollout = {
            "structured_target": dict(target),
            "final_answer": dict(target),
            "state": {"finalized_case": dict(target)},
            "turns": [{"tool_name": "scan_timeline"}],
        }
        no_stage_rollout = dict(base_rollout)
        no_stage_rollout["counterfactual_profile"] = no_stage_profile
        trigger_rollout = dict(base_rollout)
        trigger_rollout["counterfactual_profile"] = trigger_profile
        no_stage_diag = _timesearch_fecv_diagnostics(
            no_stage_profile,
            target=target,
            rollout_trace=no_stage_rollout,
        )
        trigger_diag = _timesearch_fecv_diagnostics(
            trigger_profile,
            target=target,
            rollout_trace=trigger_rollout,
        )
        self.assertGreater(
            no_stage_diag["normal_window_restraint_score"],
            trigger_diag["normal_window_restraint_score"],
        )
        self.assertGreater(
            no_stage_diag["normal_restraint_reward"],
            trigger_diag["normal_restraint_reward"],
        )

    def test_online_core_branch_order_includes_minimal_subset_and_skips_hard_negative(self) -> None:
        branch_order = _resolve_counterfactual_branch_order(
            "online_core",
            stage_requirements={},
        )
        self.assertEqual(branch_order, ["full_selected", "minimal_subset", "drop_trigger"])

    def test_online_core_fecv_reward_includes_parsimony_but_skips_negative_resistance(self) -> None:
        profile = {
            "counterfactual_profile_source": "online_core",
            "selection_metadata": {"normalized_branch_profile": "online_core"},
            "branch_field_matrix": {
                "full_selected": {
                    "available": True,
                    "window_ids": ["w1", "w2"],
                    "fields": {
                        "existence": {"score": 1.0},
                        "category": {"score": 1.0},
                        "trigger": {"score": 1.0},
                    },
                },
                "minimal_subset": {
                    "available": True,
                    "window_ids": ["w1"],
                    "fields": {},
                },
            },
            "branch_delta_matrix": {
                "drop_trigger": {
                    "fields": {
                        "existence": 1.0,
                        "category": 1.0,
                    }
                },
                "hard_negative_swap": {
                    "fields": {
                        "existence": 1.0,
                        "category": 1.0,
                    }
                },
            },
        }
        target = {"existence": "anomaly", "category": "fall"}
        reward = _timesearch_fecv_reward(profile, target=target)
        self.assertEqual(reward, 0.9375)

    def test_online_core_fecv_reward_unchanged_when_rollout_trace_is_present(self) -> None:
        profile = {
            "counterfactual_profile_source": "online_core",
            "selection_metadata": {"normalized_branch_profile": "online_core"},
            "branch_field_matrix": {
                "full_selected": {
                    "available": True,
                    "window_ids": ["w1", "w2"],
                    "fields": {
                        "existence": {"score": 1.0},
                        "category": {"score": 1.0},
                        "trigger": {"score": 1.0},
                    },
                },
                "minimal_subset": {
                    "available": True,
                    "window_ids": ["w1"],
                    "fields": {},
                },
            },
            "branch_delta_matrix": {
                "drop_trigger": {
                    "fields": {
                        "existence": 1.0,
                        "category": 1.0,
                    }
                },
                "hard_negative_swap": {
                    "fields": {
                        "existence": 1.0,
                        "category": 1.0,
                    }
                },
            },
        }
        target = {"existence": "anomaly", "category": "fall"}
        reward = _timesearch_fecv_reward(
            profile,
            target=target,
            rollout_trace={"turns": [{"tool_name": "scan_timeline"}]},
        )
        self.assertEqual(reward, 0.9375)

    def test_normal_skip_query_alignment_score_reduces_reward_when_seek_alignment_is_poor(self) -> None:
        target = {"existence": "normal", "category": "normal"}
        profile = {
            "counterfactual_profile_source": "normal_skip_v1",
            "selection_metadata": {
                "normalized_branch_profile": "normal_skip_v1",
                "selected_window_count": 0,
                "selected_record_count": 0,
                "selected_by_stage": {"precursor": [], "trigger": [], "confirmation": []},
                "full_selected_parse_mode": "skipped_normal",
                "full_selected_unavailable_reason": "normal_sample_skipped",
            },
            "stage_packages": {
                "selected_window_ids": [],
                "selected_by_stage": {"precursor": [], "trigger": [], "confirmation": []},
            },
        }
        rollout_trace = {
            "structured_target": dict(target),
            "final_answer": dict(target),
            "state": {"finalized_case": dict(target)},
            "counterfactual_profile": profile,
            "turns": [{"tool_name": "seek_evidence"}],
        }
        with mock.patch(
            "saver_v3.core.reward._query_alignment_reward",
            return_value=-1.0,
        ):
            diagnostics = _timesearch_fecv_diagnostics(profile, target=target, rollout_trace=rollout_trace)
        self.assertEqual(diagnostics["normal_query_alignment_score"], 0.0)
        self.assertEqual(diagnostics["normal_case_type"], "suspicious_normal")
        self.assertAlmostEqual(diagnostics["normal_restraint_reward"], 0.496, places=6)

    def test_normal_skip_best_effort_selection_raises_grounded_local_provenance(self) -> None:
        target = {"existence": "normal", "category": "normal"}
        profile = {
            "counterfactual_profile_source": "normal_skip_v1",
            "selection_metadata": {
                "normalized_branch_profile": "normal_skip_v1",
                "selected_window_count": 1,
                "selected_record_count": 1,
                "selected_by_stage": {"precursor": [], "trigger": [], "confirmation": []},
                "selection_resolution_source": "verification_record_best_effort_window_ids",
                "full_selected_parse_mode": "skipped_normal",
                "full_selected_unavailable_reason": "normal_sample_skipped",
            },
            "stage_packages": {
                "selected_window_ids": ["w0009"],
                "selected_by_stage": {"precursor": [], "trigger": [], "confirmation": []},
            },
        }
        rollout_trace = {
            "structured_target": dict(target),
            "final_answer": dict(target),
            "state": {"finalized_case": dict(target)},
            "counterfactual_profile": profile,
            "turns": [{"tool_name": "scan_timeline"}, {"tool_name": "seek_evidence"}],
        }
        diagnostics = _timesearch_fecv_diagnostics(profile, target=target, rollout_trace=rollout_trace)
        self.assertEqual(diagnostics["normal_case_type"], "suspicious_normal")
        self.assertEqual(diagnostics["normal_provenance_score"], 0.7)
        self.assertEqual(diagnostics["normal_provenance_source_bucket"], "best_effort_or_evidence_anchor")
        self.assertAlmostEqual(diagnostics["normal_grounded_local_score"], 0.805, places=6)

    def test_normal_skip_continuous_verifier_score_uses_action_offsets(self) -> None:
        target = {"existence": "normal", "category": "normal"}
        profile = {
            "counterfactual_profile_source": "normal_skip_v1",
            "selection_metadata": {
                "normalized_branch_profile": "normal_skip_v1",
                "selected_window_count": 1,
                "selected_record_count": 1,
                "selected_by_stage": {"precursor": [], "trigger": [], "confirmation": []},
                "selection_resolution_source": "normal_sample_skipped",
                "full_selected_parse_mode": "skipped_normal",
                "full_selected_unavailable_reason": "normal_sample_skipped",
            },
            "stage_packages": {
                "selected_window_ids": ["w1"],
                "selected_by_stage": {"precursor": [], "trigger": [], "confirmation": []},
            },
        }
        rollout_trace = {
            "structured_target": dict(target),
            "final_answer": dict(target),
            "state": {"finalized_case": dict(target)},
            "counterfactual_profile": profile,
            "turns": [
                {
                    "tool_name": "verify_hypothesis",
                    "verifier_primary_status": "complete",
                    "verifier_recommended_action": "continue_search",
                }
            ],
        }
        diagnostics = _timesearch_fecv_diagnostics(profile, target=target, rollout_trace=rollout_trace)
        self.assertEqual(diagnostics["normal_verifier_primary_status"], "complete")
        self.assertEqual(diagnostics["normal_verifier_recommended_action"], "continue_search")
        self.assertEqual(diagnostics["normal_verifier_base_status_score"], 1.0)
        self.assertEqual(diagnostics["normal_verifier_action_offset"], -0.3)
        self.assertEqual(diagnostics["normal_continuous_verifier_score_before_action"], 1.0)
        self.assertEqual(diagnostics["normal_continuous_verifier_score_after_action"], 0.7)
        self.assertEqual(diagnostics["normal_continuous_verifier_score"], 0.7)
        self.assertAlmostEqual(diagnostics["normal_restraint_reward"], 0.5965, places=6)

    def test_resolve_selected_window_ids_for_fecv_prefers_verification_record_recovery(self) -> None:
        rollout = {
            "state": {
                "active_evidence_window_ids": [],
                "verification_records": [
                    {
                        "verified_window_ids": [],
                        "best_effort_window_ids": ["w0009"],
                    }
                ],
            },
            "turns": [],
        }
        resolved = resolve_selected_window_ids_for_fecv(rollout)
        self.assertEqual(resolved["selected_window_ids"], ["w0009"])
        self.assertEqual(resolved["selection_resolution_source"], "verification_record_best_effort_window_ids")
        self.assertTrue(resolved["recovered_from_trace"])

    def test_anomaly_counterfactual_verification_recovers_selected_windows_from_verification_records(self) -> None:
        class _AlwaysAnswerPolicy:
            def generate_from_messages_batch(self, messages_batch):
                return ['<answer>{"decision":{"existence":"anomaly","category":"fall"}}</answer>' for _ in list(messages_batch or [])]

        policy = _AlwaysAnswerPolicy()
        result = run_counterfactual_verification_batch(
            policy,
            batch_inputs=[
                {
                    "item": {"video_id": "recover_case", "multimodal_cache": {"question": "Does an anomaly exist?"}},
                    "rollout": {
                        "video_id": "recover_case",
                        "state": {
                            "active_evidence_window_ids": [],
                            "verification_records": [
                                {
                                    "verified_window_ids": [],
                                    "best_effort_window_ids": ["w0009"],
                                }
                            ],
                            "evidence_ledger": [
                                {
                                    "window_id": "w0009",
                                    "role": "trigger",
                                    "description": "A person falls to the ground.",
                                    "selected_frame_indices": [],
                                    "selected_timestamps": [],
                                }
                            ],
                        },
                        "turns": [],
                    },
                    "reference_record": {
                        "structured_target": {
                            "existence": "anomaly",
                            "category": "fall",
                            "event_chain_target": {"required_stages": ["trigger"]},
                        }
                    },
                }
            ],
            branch_profile="online_core",
        )
        profile = result[0]["counterfactual_profile"]
        metadata = dict(profile.get("selection_metadata") or {})
        self.assertEqual(metadata.get("selection_resolution_source"), "verification_record_best_effort_window_ids")
        self.assertTrue(bool(metadata.get("recovered_from_trace")))
        self.assertEqual(profile["stage_packages"]["selected_window_ids"], ["w0009"])
        self.assertTrue(bool((profile.get("branch_field_matrix") or {}).get("full_selected", {}).get("available")))

    def test_anomaly_empty_selection_contract_failure_forces_protocol_penalty(self) -> None:
        result = run_counterfactual_verification_batch(
            object(),
            batch_inputs=[
                {
                    "item": {"video_id": "empty_selection_case"},
                    "rollout": {
                        "video_id": "empty_selection_case",
                        "state": {
                            "active_evidence_window_ids": [],
                            "verification_records": [],
                            "evidence_ledger": [],
                        },
                        "turns": [
                            {"tool_name": "finalize_case", "step_index": 1},
                        ],
                        "final_answer": {"existence": "anomaly", "category": "fall"},
                    },
                    "reference_record": {
                        "structured_target": {
                            "existence": "anomaly",
                            "category": "fall",
                            "event_chain_target": {"required_stages": ["trigger"]},
                        }
                    },
                }
            ],
            branch_profile="online_core",
        )
        profile = result[0]["counterfactual_profile"]
        metadata = dict(profile.get("selection_metadata") or {})
        self.assertEqual(metadata.get("selection_resolution_source"), "recovery_failed")
        self.assertFalse(bool(metadata.get("recovered_from_trace")))
        self.assertEqual(metadata.get("full_selected_unavailable_reason"), "contract_violation_empty_selection")

        rollout_trace = {
            "structured_target": {"existence": "anomaly", "category": "fall"},
            "final_answer": {"existence": "anomaly", "category": "fall"},
            "state": {"finalized_case": {"existence": "anomaly", "category": "fall"}},
            "turns": [{"tool_name": "finalize_case", "step_index": 1}],
            "counterfactual_profile": profile,
        }
        reward_summary = _score_rollout_trace_timesearch(rollout_trace, reward_version="timesearch_v3")
        self.assertEqual(reward_summary["components"]["protocol_finalize_reward"], -1.0)

    def test_online_core_fecv_diagnostics_expose_driving_terms(self) -> None:
        profile = {
            "counterfactual_profile_source": "online_core",
            "selection_metadata": {"normalized_branch_profile": "online_core"},
            "branch_field_matrix": {
                "full_selected": {
                    "available": True,
                    "window_ids": ["w1", "w2"],
                    "fields": {
                        "existence": {"score": 1.0},
                        "category": {"score": 0.5},
                        "trigger": {"score": 0.25},
                    },
                },
                "minimal_subset": {
                    "available": True,
                    "window_ids": ["w1"],
                    "fields": {},
                },
            },
            "branch_delta_matrix": {
                "drop_trigger": {
                    "fields": {
                        "existence": 0.6,
                        "category": 0.2,
                    }
                },
            },
        }
        target = {
            "existence": "anomaly",
            "category": "fall",
            "event_chain_target": {"required_stages": ["trigger"]},
        }
        diagnostics = _timesearch_fecv_diagnostics(profile, target=target)
        self.assertEqual(diagnostics["branch_profile"], "online_core")
        self.assertTrue(diagnostics["full_selected_available"])
        self.assertEqual(diagnostics["decision_field_scores"], {"existence": 1.0, "category": 0.5})
        self.assertEqual(diagnostics["required_stage_scores"], {"trigger": 0.25})
        self.assertAlmostEqual(diagnostics["selected_support_score"], 0.6, places=6)
        self.assertAlmostEqual(diagnostics["trigger_necessity_delta"], 0.6, places=6)
        self.assertAlmostEqual(diagnostics["minimal_subset_parsimony_bonus"], 0.5, places=6)
        self.assertAlmostEqual(diagnostics["evidence_faithfulness_reward"], 0.5875, places=6)

    def test_online_core_minimal_subset_uses_compact_decision_scaffold(self) -> None:
        policy = _ReplayBatchPolicy(
            [
                ['<answer>{"decision":{"existence":"anomaly","category":"fall"}}</answer>'],
                ['<answer>{"decision":{"existence":"anomaly","category":"fall"}}</answer>'],
                ['<answer>{"decision":{"existence":"anomaly","category":"fall"}}</answer>'],
            ]
        )
        result = run_counterfactual_verification_batch(
            policy,
            batch_inputs=[
                {
                    "item": {"video_id": "sample", "multimodal_cache": {"question": "Does an anomaly exist?"}},
                    "rollout": {
                        "video_id": "sample",
                        "state": {
                            "active_evidence_window_ids": ["w0002"],
                            "evidence_ledger": [
                                {
                                    "window_id": "w0002",
                                    "role": "trigger",
                                    "description": "A person falls to the ground.",
                                    "selected_frame_indices": [],
                                    "selected_timestamps": [],
                                }
                            ],
                        },
                        "turns": [],
                    },
                    "reference_record": {
                        "structured_target": {
                            "existence": "anomaly",
                            "category": "fall",
                            "event_chain_target": {"required_stages": ["trigger"]},
                        }
                    },
                }
            ],
            branch_profile="online_core",
        )
        self.assertGreaterEqual(len(policy.calls), 2)
        self.assertIn("minimal_subset", result[0]["counterfactual_branches"])
        second_call_messages = policy.calls[1][0]
        user_text = ""
        for message in second_call_messages:
            if message.get("role") != "user":
                continue
            for item in message.get("content", []):
                if isinstance(item, dict) and item.get("type") == "text":
                    user_text += str(item.get("text") or "")
        self.assertIn('"decision"', user_text)
        self.assertNotIn('"summary"', user_text)
        self.assertNotIn('"qa_focus_answers"', user_text)

    def test_full_profile_fecv_reward_keeps_parsimony_and_negative_resistance(self) -> None:
        profile = {
            "counterfactual_profile_source": "full",
            "selection_metadata": {"normalized_branch_profile": "full"},
            "branch_field_matrix": {
                "full_selected": {
                    "available": True,
                    "window_ids": ["w1", "w2", "w3", "w4"],
                    "fields": {
                        "existence": {"score": 1.0},
                        "category": {"score": 1.0},
                        "trigger": {"score": 1.0},
                    },
                },
                "minimal_subset": {
                    "available": True,
                    "window_ids": ["w1", "w2"],
                    "fields": {},
                },
            },
            "branch_delta_matrix": {
                "drop_trigger": {
                    "fields": {
                        "existence": 1.0,
                        "category": 1.0,
                    }
                },
                "hard_negative_swap": {
                    "fields": {
                        "existence": 1.0,
                        "category": 1.0,
                    }
                },
            },
        }
        target = {"existence": "anomaly", "category": "fall"}
        reward = _timesearch_fecv_reward(profile, target=target)
        self.assertEqual(reward, 0.95)

    def test_structured_oracle_profile_reward_uses_three_term_weighting(self) -> None:
        profile = {
            "counterfactual_profile_source": "structured_oracle_v1",
            "selection_metadata": {"normalized_branch_profile": "structured_oracle_v1"},
            "summary": {
                "oracle_selected_support_score": 1.0,
                "oracle_required_stage_coverage_score": 0.5,
                "oracle_drop_trigger_necessity_score": 0.5,
            },
        }
        reward = _timesearch_fecv_reward(profile, target={"existence": "anomaly", "category": "fall"})
        self.assertEqual(reward, 0.8)

    def test_structured_oracle_profile_builds_scores_from_selected_windows(self) -> None:
        batch_input = {
            "item": {
                "structured_target": {
                    "existence": "anomaly",
                    "category": "fall",
                    "anomaly_interval_sec": [4.0, 6.0],
                    "event_chain_target": {
                        "required_stages": ["trigger", "confirmation"],
                        "stage_to_moment_ids": {
                            "trigger": ["m_trigger"],
                            "confirmation": ["m_confirm"],
                        },
                    },
                },
                "evidence": {
                    "evidence_moments": [
                        {"moment_id": "m_trigger", "role": "trigger", "start_sec": 4.0, "end_sec": 6.0},
                        {"moment_id": "m_confirm", "role": "confirmation", "start_sec": 7.0, "end_sec": 8.0},
                    ]
                },
            },
            "rollout": {
                "final_answer": {
                    "existence": "anomaly",
                    "category": "fall",
                    "anomaly_interval_sec": [4.0, 6.0],
                },
                "state": {
                    "active_evidence_window_ids": ["w_trigger", "w_confirm"],
                    "evidence_ledger": [
                        {"window_id": "w_trigger", "start_sec": 4.0, "end_sec": 6.0},
                        {"window_id": "w_confirm", "start_sec": 7.0, "end_sec": 8.0},
                    ],
                },
                "turns": [],
            },
            "reference_record": {
                "structured_target": {
                    "existence": "anomaly",
                    "category": "fall",
                    "anomaly_interval_sec": [4.0, 6.0],
                    "event_chain_target": {
                        "required_stages": ["trigger", "confirmation"],
                        "stage_to_moment_ids": {
                            "trigger": ["m_trigger"],
                            "confirmation": ["m_confirm"],
                        },
                    },
                },
                "evidence": {
                    "evidence_moments": [
                        {"moment_id": "m_trigger", "role": "trigger", "start_sec": 4.0, "end_sec": 6.0},
                        {"moment_id": "m_confirm", "role": "confirmation", "start_sec": 7.0, "end_sec": 8.0},
                    ]
                },
            },
        }
        result = run_counterfactual_verification_batch(
            object(),
            batch_inputs=[batch_input],
            branch_profile="structured_oracle_v1",
        )[0]
        summary = result["counterfactual_profile"]["summary"]
        self.assertEqual(result["counterfactual_profile_source"], "structured_oracle_v1")
        self.assertEqual(summary["oracle_selected_support_score"], 1.0)
        self.assertEqual(summary["oracle_required_stage_coverage_score"], 1.0)
        self.assertEqual(summary["oracle_drop_trigger_necessity_score"], 1.0)

    def test_structured_oracle_profile_can_omit_counterfactual_type_support_for_active_rl(self) -> None:
        batch_input = {
            "item": {
                "structured_target": {
                    "existence": "anomaly",
                    "category": "fall",
                    "anomaly_interval_sec": [4.0, 6.0],
                },
                "evidence": {"evidence_moments": []},
            },
            "rollout": {
                "final_answer": {
                    "existence": "anomaly",
                    "category": "fall",
                    "anomaly_interval_sec": [4.0, 6.0],
                },
                "state": {"active_evidence_window_ids": []},
                "turns": [],
            },
            "reference_record": {
                "structured_target": {
                    "existence": "anomaly",
                    "category": "fall",
                    "anomaly_interval_sec": [4.0, 6.0],
                },
                "evidence": {"evidence_moments": []},
            },
        }
        result = run_counterfactual_verification_batch(
            object(),
            batch_inputs=[batch_input],
            branch_profile="structured_oracle_v1",
            include_counterfactual_type=False,
        )[0]
        summary = result["counterfactual_profile"]["summary"]
        self.assertNotIn("counterfactual_type_supported", summary)

    def test_active_rl_uses_structured_oracle_profile(self) -> None:
        src = inspect.getsource(timesearch_aligned_grpo_trainer.TimesearchAlignedGRPOTrainerMixin._generate_scored_rollouts_batch)
        self.assertIn('fecv_branch_profile = "structured_oracle_v1"', src)
        self.assertIn("include_counterfactual_type=False", src)

    def test_timesearch_v3_accuracy_reward_ignores_open_ended_text_fields(self) -> None:
        rollout_trace = {
            "structured_target": {
                "existence": "anomaly",
                "category": "fall",
                "anomaly_interval_sec": [4.0, 6.0],
                "summary": "target summary",
                "event_chain_summary": {
                    "precursor": "person walking",
                    "trigger": "person falls",
                    "confirmation": "person remains on ground",
                },
            },
            "qa_pairs": [
                {"type": "trigger_evidence", "question": "What visible evidence first makes the anomaly actionable?", "answer": "person falls"},
            ],
            "semantic_answer": {
                "decision": {
                    "existence": "normal",
                    "category": "normal",
                    "anomaly_interval_sec": [0.0, 1.0],
                },
                "summary": "target summary",
                "event_chain_summary": {
                    "precursor": "person walking",
                    "trigger": "person falls",
                    "confirmation": "person remains on ground",
                },
                "qa_focus_answers": {},
            },
        }
        result = _compute_accuracy_breakdown(rollout_trace, reward_version="timesearch_v3")
        self.assertEqual(result["accuracy_by_family"]["open_ended"], 0.0)
        self.assertEqual(result["accuracy_question_count"], 2)

    def test_active_rl_accuracy_reward_ignores_severity_and_counterfactual_type(self) -> None:
        rollout_trace = {
            "structured_target": {
                "existence": "anomaly",
                "category": "fall",
                "severity": 4,
                "counterfactual_type": "remove_actor_interaction",
                "anomaly_interval_sec": [4.0, 6.0],
            },
            "semantic_answer": {
                "decision": {
                    "existence": "anomaly",
                    "category": "fall",
                    "severity": 1,
                    "counterfactual_type": "none",
                    "anomaly_interval_sec": [4.0, 6.0],
                },
                "summary": "",
                "event_chain_summary": {},
                "qa_focus_answers": {},
            },
        }
        reward_func = build_timesearch_reward_funcs(
            reward_version="timesearch_v3",
            include_aux_decision_fields=False,
        )[0]
        reward_value = reward_func(rollout_traces=[rollout_trace])[0]
        breakdown = _compute_accuracy_breakdown(
            rollout_trace,
            reward_version="timesearch_v3",
            include_aux_decision_fields=False,
        )
        self.assertEqual(reward_value, 1.0)
        self.assertNotIn("severity", breakdown["accuracy_by_type"])
        self.assertNotIn("counterfactual", breakdown["accuracy_by_type"])


if __name__ == "__main__":
    unittest.main()
