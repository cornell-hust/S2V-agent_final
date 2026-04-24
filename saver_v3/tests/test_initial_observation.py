from __future__ import annotations

from pathlib import Path

import pytest
import torch

from saver_v3.common.message_budget import apply_message_budget
from saver_v3.core.initial_observation import mark_initial_global_scan_message
from saver_v3.core.rollout import SaverRolloutRunner
from saver_v3.core.schema import SaverEnvironmentState
from saver_v3.core.tool_registry import execute_tool_call
from saver_v3.data.config import InitialObservationConfig, PreviewConfig, PromptConfig, SaverAgentConfig
from saver_v3.data.dataset import SaverRecordItemBuilder
from saver_v3.data.materialized_cache import (
    MATERIALIZED_RUNTIME_ITEMS_FORMAT,
    ensure_materialized_cache_metadata,
    write_materialized_cache_metadata,
)
import saver_v3.data.materialized_cache as materialized_cache_module
from saver_v3.data.prepared_metadata import ensure_prepared_sft_metadata, write_prepared_sft_metadata
from saver_v3.model.qwen_policy import QwenGenerationPolicy
from saver_v3.sft import training as sft_training


def _base_record() -> dict:
    return {
        "video_id": "vid1",
        "split": "train",
        "video_path": "/tmp/example.mp4",
        "video_meta": {"duration_sec": 8.0, "fps": 1.0},
        "scene": {"scenario": "garage"},
        "agent_task": {
            "task_prompt": "Determine whether the video contains an actionable anomaly.",
            "success_criteria": ["Search before you verify and finalize."],
        },
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
    }


def _image_items(message: dict) -> list[dict]:
    return [
        item
        for item in list(message.get("content") or [])
        if isinstance(item, dict) and item.get("type") == "image"
    ]


def test_record_builder_omits_preview_images_for_explicit_first_scan():
    builder = SaverRecordItemBuilder(
        config=SaverAgentConfig(
            preview=PreviewConfig(num_preview_frames=8, max_preview_frames=8),
            initial_observation=InitialObservationConfig(mode="explicit_first_scan"),
        ),
        load_frame_cache=False,
        load_feature_cache=False,
    )

    item = builder.build_item(_base_record())

    user_content = item["messages"][1]["content"]
    assert [entry["type"] for entry in user_content] == ["text"]
    assert "scan_timeline" in user_content[0]["text"]
    assert "global_overview" in user_content[0]["text"]
    assert item["multimodal_cache"]["preview_frames"] is None
    assert item["multimodal_cache"]["preview_timestamps"] == []


def test_execute_tool_call_marks_canonical_initial_global_scan():
    multimodal_cache = {
        "video": torch.zeros((8, 3, 4, 4), dtype=torch.float32),
        "fps": 1.0,
        "duration": 8.0,
        "frame_indices": list(range(8)),
        "video_path": "/tmp/example.mp4",
        "tool_io": {"initial_scan_window_sec": [0.0, 8.0]},
        "config_snapshot": SaverAgentConfig().to_dict(),
    }

    tool_message, _ = execute_tool_call(
        {
            "function": {
                "name": "scan_timeline",
                "arguments": {
                    "start_sec": 0.0,
                    "end_sec": 8.0,
                    "num_frames": 8,
                    "purpose": "global_overview",
                },
            }
        },
        multimodal_cache,
        SaverEnvironmentState(),
    )

    assert tool_message["initial_global_scan"] is True
    assert tool_message["protect_initial_global_scan"] is True
    assert tool_message["error_on_initial_scan_seq_prune"] is True
    assert len(_image_items(tool_message)) == 8


def test_materialized_cache_metadata_rejects_preview_semantics(tmp_path: Path):
    source_path = tmp_path / "runtime.jsonl"
    source_path.write_text('{"video_id":"vid1","split":"train"}\n', encoding="utf-8")
    cache_path = tmp_path / "runtime.materialized_items_v4.jsonl"
    cache_path.write_text("{}\n", encoding="utf-8")

    write_materialized_cache_metadata(
        cache_path,
        materialized_format=MATERIALIZED_RUNTIME_ITEMS_FORMAT,
        config=SaverAgentConfig(initial_observation=InitialObservationConfig(mode="preview")),
        source_path=source_path,
    )

    with pytest.raises(ValueError, match="config mismatch"):
        ensure_materialized_cache_metadata(
            cache_path,
            expected_format=MATERIALIZED_RUNTIME_ITEMS_FORMAT,
            expected_source_path=source_path,
            expected_config=SaverAgentConfig(),
            require_config_match=True,
            require_source=True,
        )


def test_materialized_cache_metadata_ignores_preview_only_config_for_explicit_first_scan(tmp_path: Path):
    source_path = tmp_path / "runtime.jsonl"
    source_path.write_text('{"video_id":"vid1","split":"train"}\n', encoding="utf-8")
    cache_path = tmp_path / "runtime.materialized_items_v4.jsonl"
    cache_path.write_text("{}\n", encoding="utf-8")

    write_materialized_cache_metadata(
        cache_path,
        materialized_format=MATERIALIZED_RUNTIME_ITEMS_FORMAT,
        config=SaverAgentConfig(
            preview=PreviewConfig(num_preview_frames=28, max_preview_frames=28),
            initial_observation=InitialObservationConfig(mode="explicit_first_scan"),
            prompt=PromptConfig(preview_instruction="legacy preview wording should be ignored"),
        ),
        source_path=source_path,
    )

    ensure_materialized_cache_metadata(
        cache_path,
        expected_format=MATERIALIZED_RUNTIME_ITEMS_FORMAT,
        expected_source_path=source_path,
        expected_config=SaverAgentConfig(
            preview=PreviewConfig(num_preview_frames=8, max_preview_frames=8),
            initial_observation=InitialObservationConfig(mode="explicit_first_scan"),
            prompt=PromptConfig(preview_instruction="different preview-only wording"),
        ),
        require_config_match=True,
        require_source=True,
    )


def test_materialized_cache_metadata_uses_stat_fast_path_when_source_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source_path = tmp_path / "runtime.jsonl"
    source_path.write_text('{"video_id":"vid1","split":"train"}\n', encoding="utf-8")
    cache_path = tmp_path / "runtime.materialized_items_v4.jsonl"
    cache_path.write_text("{}\n", encoding="utf-8")

    write_materialized_cache_metadata(
        cache_path,
        materialized_format=MATERIALIZED_RUNTIME_ITEMS_FORMAT,
        config=SaverAgentConfig(initial_observation=InitialObservationConfig(mode="explicit_first_scan")),
        source_path=source_path,
        include_splits="train",
    )

    def _unexpected_rehash(*args, **kwargs):
        raise AssertionError("build_jsonl_provenance should not run when source stat is unchanged")

    monkeypatch.setattr(materialized_cache_module, "build_jsonl_provenance", _unexpected_rehash)

    ensure_materialized_cache_metadata(
        cache_path,
        expected_format=MATERIALIZED_RUNTIME_ITEMS_FORMAT,
        expected_source_path=source_path,
        expected_include_splits="train",
        expected_config=SaverAgentConfig(initial_observation=InitialObservationConfig(mode="explicit_first_scan")),
        require_config_match=True,
        require_source=True,
    )


def test_prepared_metadata_ignores_preview_only_config_for_explicit_first_scan(tmp_path: Path):
    prepared_path = tmp_path / "prepared.jsonl"
    prepared_path.write_text("{}\n", encoding="utf-8")

    write_prepared_sft_metadata(
        prepared_path,
        config=SaverAgentConfig(
            preview=PreviewConfig(num_preview_frames=28, max_preview_frames=28),
            initial_observation=InitialObservationConfig(mode="explicit_first_scan"),
            prompt=PromptConfig(preview_instruction="legacy preview wording should be ignored"),
        ),
    )

    ensure_prepared_sft_metadata(
        prepared_path,
        config=SaverAgentConfig(
            preview=PreviewConfig(num_preview_frames=8, max_preview_frames=8),
            initial_observation=InitialObservationConfig(mode="explicit_first_scan"),
            prompt=PromptConfig(preview_instruction="different preview-only wording"),
        ),
        require_config_match=True,
    )


def test_materialized_cache_metadata_ignores_prompt_trailing_newlines(tmp_path: Path):
    source_path = tmp_path / "runtime.jsonl"
    source_path.write_text('{"video_id":"vid1","split":"train"}\n', encoding="utf-8")
    cache_path = tmp_path / "runtime.materialized_items_v3.jsonl"
    cache_path.write_text("{}\n", encoding="utf-8")

    write_materialized_cache_metadata(
        cache_path,
        materialized_format=MATERIALIZED_RUNTIME_ITEMS_FORMAT,
        config=SaverAgentConfig(
            initial_observation=InitialObservationConfig(mode="explicit_first_scan"),
            prompt=PromptConfig(
                initial_user_template="Line A\nLine B\n",
                explicit_first_scan_instruction="Scan first.\n",
                tool_response_template="Tool text.\n",
            ),
        ),
        source_path=source_path,
        include_splits="train",
    )

    ensure_materialized_cache_metadata(
        cache_path,
        expected_format=MATERIALIZED_RUNTIME_ITEMS_FORMAT,
        expected_source_path=source_path,
        expected_include_splits="train",
        expected_config=SaverAgentConfig(
            initial_observation=InitialObservationConfig(mode="explicit_first_scan"),
            prompt=PromptConfig(
                initial_user_template="Line A\nLine B",
                explicit_first_scan_instruction="Scan first.",
                tool_response_template="Tool text.",
            ),
        ),
        require_config_match=True,
        require_source=True,
    )


def test_message_budget_preserves_full_canonical_initial_scan():
    protected_scan = mark_initial_global_scan_message(
        {
            "role": "tool",
            "name": "scan_timeline",
            "content": [
                item
                for index in range(8)
                for item in (
                    {"type": "text", "text": f"{float(index):.3f}s"},
                    {"type": "image", "image": None, "timestamp_sec": float(index)},
                )
            ]
            + [{"type": "text", "text": "overview"}],
        },
        config=SaverAgentConfig(),
    )
    later_scan = {
        "role": "tool",
        "name": "seek_evidence",
        "content": [
            {"type": "text", "text": "9.000s"},
            {"type": "image", "image": None, "timestamp_sec": 9.0},
            {"type": "text", "text": "10.000s"},
            {"type": "image", "image": None, "timestamp_sec": 10.0},
            {"type": "text", "text": "later evidence"},
        ],
    }
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {"role": "user", "content": [{"type": "text", "text": "user"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "<tool_call>...</tool_call>"}]},
        protected_scan,
        {"role": "assistant", "content": [{"type": "text", "text": "<tool_call>...</tool_call>"}]},
        later_scan,
    ]

    prepared = apply_message_budget(
        messages,
        keep_recent_text_messages=1,
        keep_recent_tool_image_messages=1,
        max_total_images=2,
        max_tool_message_frames=2,
        max_total_video_frames=2,
    )

    protected_messages = [message for message in prepared if message.get("initial_global_scan")]
    assert len(protected_messages) == 1
    assert len(_image_items(protected_messages[0])) == 8
    assert sum(len(_image_items(message)) for message in prepared) == 8


def test_message_budget_prunes_oldest_tool_frames_incrementally_instead_of_dropping_whole_messages():
    protected_scan = mark_initial_global_scan_message(
        {
            "role": "tool",
            "name": "scan_timeline",
            "content": [
                item
                for index in range(8)
                for item in (
                    {"type": "text", "text": f"{float(index):.3f}s"},
                    {"type": "image", "image": None, "timestamp_sec": float(index)},
                )
            ]
            + [{"type": "text", "text": "overview"}],
        },
        config=SaverAgentConfig(),
    )
    old_tool_1 = {
        "role": "tool",
        "name": "seek_evidence",
        "content": [
            {"type": "text", "text": "8.000s"},
            {"type": "image", "image": None, "timestamp_sec": 8.0},
            {"type": "text", "text": "9.000s"},
            {"type": "image", "image": None, "timestamp_sec": 9.0},
            {"type": "text", "text": "older evidence 1"},
        ],
    }
    old_tool_2 = {
        "role": "tool",
        "name": "seek_evidence",
        "content": [
            {"type": "text", "text": "10.000s"},
            {"type": "image", "image": None, "timestamp_sec": 10.0},
            {"type": "text", "text": "11.000s"},
            {"type": "image", "image": None, "timestamp_sec": 11.0},
            {"type": "text", "text": "older evidence 2"},
        ],
    }
    recent_tool = {
        "role": "tool",
        "name": "seek_evidence",
        "content": [
            {"type": "text", "text": "12.000s"},
            {"type": "image", "image": None, "timestamp_sec": 12.0},
            {"type": "text", "text": "13.000s"},
            {"type": "image", "image": None, "timestamp_sec": 13.0},
            {"type": "text", "text": "recent evidence"},
        ],
    }
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {"role": "user", "content": [{"type": "text", "text": "user"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "<tool_call>...</tool_call>"}]},
        protected_scan,
        {"role": "assistant", "content": [{"type": "text", "text": "<tool_call>...</tool_call>"}]},
        old_tool_1,
        {"role": "assistant", "content": [{"type": "text", "text": "<tool_call>...</tool_call>"}]},
        old_tool_2,
        {"role": "assistant", "content": [{"type": "text", "text": "<tool_call>...</tool_call>"}]},
        recent_tool,
    ]

    prepared = apply_message_budget(
        messages,
        keep_recent_text_messages=10,
        keep_recent_tool_image_messages=2,
        max_total_images=11,
        max_tool_message_frames=0,
        max_total_video_frames=0,
    )

    tool_messages = [message for message in prepared if message.get("role") == "tool"]
    tool_image_counts = [len(_image_items(message)) for message in tool_messages]
    assert tool_image_counts == [8, 0, 1, 2]


def test_qwen_policy_refuses_to_prune_protected_initial_scan_for_max_seq():
    policy = object.__new__(QwenGenerationPolicy)
    policy._build_inputs_exact = lambda messages: {"input_ids": list(range(32))}
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {"role": "user", "content": [{"type": "text", "text": "user"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "<tool_call>...</tool_call>"}]},
        mark_initial_global_scan_message(
            {
                "role": "tool",
                "name": "scan_timeline",
                "content": [
                    {"type": "text", "text": "0.000s"},
                    {"type": "image", "image": None, "timestamp_sec": 0.0},
                    {"type": "text", "text": "overview"},
                ],
            },
            config=SaverAgentConfig(),
        ),
    ]

    with pytest.raises(ValueError, match="canonical initial global scan"):
        QwenGenerationPolicy._fit_prepared_messages_to_max_length(policy, messages, max_length=1)


def test_sft_episode_budget_refuses_to_prune_protected_initial_scan(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sft_training, "_build_chat_text", lambda *args, **kwargs: "episode")
    monkeypatch.setattr(
        sft_training,
        "_tokenize_chat",
        lambda *args, **kwargs: {"input_ids": list(range(32))},
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {"role": "user", "content": [{"type": "text", "text": "user"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "<tool_call>...</tool_call>"}]},
        mark_initial_global_scan_message(
            {
                "role": "tool",
                "name": "scan_timeline",
                "content": [
                    {"type": "text", "text": "0.000s"},
                    {"type": "image", "image": None, "timestamp_sec": 0.0},
                    {"type": "text", "text": "overview"},
                ],
            },
            config=SaverAgentConfig(),
        ),
    ]

    with pytest.raises(ValueError, match="canonical initial global scan"):
        sft_training._fit_episode_messages_to_budget(object(), messages, max_seq_length=1)


def test_sft_batch_episode_budget_refuses_to_prune_protected_initial_scan(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(sft_training, "_build_chat_text", lambda *args, **kwargs: "episode")
    monkeypatch.setattr(
        sft_training,
        "_tokenize_chat_batch",
        lambda *args, **kwargs: {
            "input_ids": torch.zeros((1, 32), dtype=torch.long),
            "attention_mask": torch.ones((1, 32), dtype=torch.long),
        },
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {"role": "user", "content": [{"type": "text", "text": "user"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "<tool_call>...</tool_call>"}]},
        mark_initial_global_scan_message(
            {
                "role": "tool",
                "name": "scan_timeline",
                "content": [
                    {"type": "text", "text": "0.000s"},
                    {"type": "image", "image": None, "timestamp_sec": 0.0},
                    {"type": "text", "text": "overview"},
                ],
            },
            config=SaverAgentConfig(),
        ),
    ]

    with pytest.raises(ValueError, match="canonical initial global scan"):
        sft_training._fit_batch_episode_messages_to_budget(object(), [messages], max_seq_length=1)


def test_rollout_builds_initial_scan_trace_and_preview_alias_from_turns():
    turns = [
        {
            "step_index": 1,
            "tool_name": "scan_timeline",
            "initial_global_scan": True,
            "parsed_tool_call": {
                "name": "scan_timeline",
                "arguments": {"start_sec": 0.0, "end_sec": 8.0, "purpose": "global_overview"},
            },
            "tool_timestamps": ["0.000s", "1.000s"],
            "tool_image_count": 8,
            "new_visited_windows": [
                {
                    "window_id": "w0001",
                    "start_sec": 0.0,
                    "end_sec": 8.0,
                    "selected_timestamps": [0.0, 1.0],
                    "selected_frame_count": 8,
                }
            ],
        }
    ]

    initial_scan_trace = SaverRolloutRunner._build_initial_scan_trace(turns)
    preview_trace = SaverRolloutRunner._build_preview_trace(turns)

    assert initial_scan_trace["window_id"] == "w0001"
    assert initial_scan_trace["purpose"] == "global_overview"
    assert initial_scan_trace["frame_count"] == 8
    assert preview_trace == {
        "preview_frame_count": 8,
        "preview_timestamps": [0.0, 1.0],
    }
