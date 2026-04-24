from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import pytest

from saver_v3.core.environment import invalid_answer_message
from saver_v3.core.rollout import ReplayPolicy, SaverRolloutRunner
from saver_v3.core.schema import SaverEnvironmentState
from saver_v3.core.tools import _resolve_selected_window_ids
from saver_v3.data.config import SaverAgentConfig
from saver_v3.data.materialized_cache import (
    MATERIALIZED_RUNTIME_ITEMS_FORMAT,
    ensure_materialized_cache_metadata,
    write_materialized_cache_metadata,
)
from saver_v3.data.prepared_metadata import (
    ensure_prepared_sft_metadata,
    load_prepared_sft_metadata,
    write_prepared_sft_metadata,
)
from saver_v3.data.protocol_signature import (
    TEACHER_ROLE_AUXILIARY,
    build_protocol_signature,
    infer_teacher_role_from_metadata,
)
from saver_v3.data.runtime_contract import validate_runtime_record_contract
from saver_v3.model.qwen_policy import QwenGenerationPolicy


def _rollout_item() -> dict:
    return {
        "video_id": "vid1",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "Use tools only."}]},
            {"role": "user", "content": [{"type": "text", "text": "Inspect and finalize the case."}]},
        ],
        "multimodal_cache": {
            "video_path": "/tmp/example.mp4",
            "duration": 8.0,
            "fps": 1.0,
            "tool_io": {
                "allowed_tools": ["scan_timeline", "verify_hypothesis", "finalize_case"],
                "initial_scan_window_sec": [0.0, 8.0],
                "finalize_case_schema": {
                    "type": "object",
                    "properties": {
                        "existence": {"type": "string"},
                        "category": {"type": "string"},
                    },
                    "required": ["existence", "category"],
                },
            },
        },
    }


class _DummyModel:
    def generate(self, **kwargs):
        del kwargs
        return [[1]]


class _DummyProcessor:
    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)

    def batch_decode(self, generated_ids_trimmed, *, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        del generated_ids_trimmed, skip_special_tokens, clean_up_tokenization_spaces
        return list(self.outputs)


def test_runtime_contract_rejects_missing_protocol_signature():
    row = {
        "runtime_contract_version": 4,
        "video_id": "vid1",
        "video_path": "/tmp/example.mp4",
        "structured_target": {"existence": "normal", "category": "normal"},
        "tool_io": {
            "finalize_case_schema": {
                "type": "object",
                "properties": {
                    "existence": {"type": "string"},
                    "category": {"type": "string"},
                },
                "required": ["existence", "category"],
            }
        },
    }

    with pytest.raises(ValueError, match="protocol_signature"):
        validate_runtime_record_contract(row)


def test_prepared_metadata_protocol_signature_mismatch_fails(tmp_path: Path):
    prepared_path = tmp_path / "prepared.jsonl"
    prepared_path.write_text("{}\n", encoding="utf-8")
    write_prepared_sft_metadata(prepared_path, config=SaverAgentConfig())

    with pytest.raises(ValueError, match="protocol signature mismatch"):
        ensure_prepared_sft_metadata(
            prepared_path,
            config=SaverAgentConfig(),
            expected_protocol_signature=build_protocol_signature(max_turns=99),
        )


def test_materialized_cache_protocol_signature_mismatch_fails(tmp_path: Path):
    source_path = tmp_path / "runtime.jsonl"
    source_path.write_text('{"video_id":"vid1","split":"train"}\n', encoding="utf-8")
    cache_path = tmp_path / "runtime.materialized_items_v4.jsonl"
    cache_path.write_text("{}\n", encoding="utf-8")
    write_materialized_cache_metadata(
        cache_path,
        materialized_format=MATERIALIZED_RUNTIME_ITEMS_FORMAT,
        config=SaverAgentConfig(),
        protocol_signature=build_protocol_signature(max_turns=10, policy_max_new_tokens=512),
        source_path=source_path,
    )

    with pytest.raises(ValueError, match="protocol signature mismatch"):
        ensure_materialized_cache_metadata(
            cache_path,
            expected_format=MATERIALIZED_RUNTIME_ITEMS_FORMAT,
            expected_source_path=source_path,
            expected_config=SaverAgentConfig(),
            expected_protocol_signature=build_protocol_signature(),
            require_config_match=True,
            require_source=True,
        )


def test_teacher_prepared_metadata_infers_auxiliary_teacher_role(tmp_path: Path):
    prepared_path = tmp_path / "teacher_prepared.jsonl"
    prepared_path.write_text("{}\n", encoding="utf-8")
    write_prepared_sft_metadata(
        prepared_path,
        config=SaverAgentConfig(),
        extra_fields={
            "teacher_annotated": True,
            "teacher_rollout_primary_materialized": True,
            "protocol_signature": build_protocol_signature(teacher_role=TEACHER_ROLE_AUXILIARY),
        },
    )

    metadata = load_prepared_sft_metadata(prepared_path)
    inferred_teacher_role = infer_teacher_role_from_metadata(metadata)

    assert inferred_teacher_role == TEACHER_ROLE_AUXILIARY
    ensure_prepared_sft_metadata(
        prepared_path,
        config=SaverAgentConfig(),
        expected_protocol_signature=build_protocol_signature(teacher_role=inferred_teacher_role),
    )


def test_main_rollout_rejects_answer_and_terminates_only_via_finalize_case():
    runner = SaverRolloutRunner(max_turns=3, config=SaverAgentConfig())
    policy = ReplayPolicy(
        [
            '<answer>{"decision":{"existence":"normal","category":"normal"}}</answer>',
            '<tool_call>{"name":"finalize_case","arguments":{"existence":"normal","category":"normal"}}</tool_call>',
        ]
    )

    result = runner.run_episode(_rollout_item(), policy)

    assert result["terminated_reason"] == "finalized"
    assert result["num_turns"] == 1
    assert result["num_invalid_attempts"] == 1
    assert result["invalid_attempts"][0]["action"] == "invalid_answer"
    assert result["final_answer"] == {"existence": "normal", "category": "normal"}
    assert result["final_answer_source"] == "finalize_case"


def test_invalid_answer_message_retries_with_tool_call_only():
    prompt_text = invalid_answer_message()["content"][0]["text"]

    assert "<tool_call>" in prompt_text
    assert "finalize_case" in prompt_text
    assert "<answer>" not in prompt_text


def test_selected_window_resolution_only_auto_heals_single_candidate():
    state = SaverEnvironmentState(
        evidence_ledger=[
            {"window_id": "w0007", "evidence_id": "e7"},
            {"window_id": "w0008", "evidence_id": "e8"},
        ]
    )

    unresolved = _resolve_selected_window_ids(
        state,
        selected_window_ids=[],
        selected_evidence_ids=[],
        selected_evidence_moment_ids=[],
        candidate_window_ids=[],
    )
    assert unresolved["resolved_window_ids"] == []
    assert unresolved["auto_heal_applied"] is False

    resolved = _resolve_selected_window_ids(
        state,
        selected_window_ids=[],
        selected_evidence_ids=[],
        selected_evidence_moment_ids=[],
        candidate_window_ids=["w0008"],
    )
    assert resolved["resolved_window_ids"] == ["w0008"]
    assert resolved["auto_heal_applied"] is True
    assert resolved["selection_resolution_source"] == "auto_heal_single_candidate_window"


def test_qwen_policy_batch_generation_applies_verify_compaction():
    raw_output = (
        '<think>inspect</think><tool_call>{"name":"verify_hypothesis","arguments":'
        '{"verification_mode":"stage_check","claim":{"existence":"anomaly","category":"assault"},'
        '"query":"long query","rationale":"too long","selected_window_ids":["w0001"],'
        '"candidate_window_ids":["w0001"],"verification_decision":"sufficient","next_tool":"finalize_case",'
        '"sufficiency_score":0.9,"necessity_score":0.8,"finalize_readiness_score":1.0,'
        '"covered_stages":["trigger"],"missing_required_stages":[],"stage_selected_moment_ids":{"trigger":["m1"]}}}'
        "</tool_call>"
    )
    policy = object.__new__(QwenGenerationPolicy)
    policy.model = _DummyModel()
    policy.processor = _DummyProcessor([raw_output])
    policy.prepare_messages = lambda messages: messages
    policy._build_inputs_batch = lambda prepared_messages_batch: {}
    policy._build_inputs = lambda prepared_messages: {}
    policy._move_to_model_device = lambda inputs: inputs
    policy._generation_kwargs = lambda: {}
    policy._temporary_generation_cache = lambda: nullcontext()
    policy._trim_generated_ids = lambda inputs, output_ids: output_ids

    batch_output = QwenGenerationPolicy.generate_from_messages_batch(policy, [[{"role": "user", "content": []}]])[0]
    single_output = QwenGenerationPolicy.generate_from_messages(policy, [{"role": "user", "content": []}])

    assert batch_output == single_output
    assert '"next_tool":"finalize_case"' in batch_output
    assert '"query":' not in batch_output
    assert '"rationale":' not in batch_output
