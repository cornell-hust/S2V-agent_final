from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from saver_v3.sft import training as training_mod


def test_trim_episode_feature_messages_to_last_supervised_assistant_drops_trailing_history():
    feature = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "scan"}]},
            {"role": "tool", "name": "scan_timeline", "content": [{"type": "text", "text": "obs"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "seek"}]},
            {"role": "tool", "name": "seek_evidence", "content": [{"type": "text", "text": "tail"}]},
        ],
        "assistant_supervision": [
            {"assistant_message_index": 1},
            {"assistant_message_index": 3},
        ],
    }

    trimmed = training_mod._trim_episode_feature_messages_to_last_supervised_assistant(feature)

    assert len(trimmed["messages"]) == 4
    assert [msg["role"] for msg in trimmed["messages"]] == ["system", "assistant", "tool", "assistant"]
    assert trimmed["assistant_supervision"] == feature["assistant_supervision"]
    assert len(feature["messages"]) == 5


def test_trim_episode_feature_messages_to_last_supervised_assistant_noops_without_trailing_history():
    feature = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "final"}]},
        ],
        "assistant_supervision": [{"assistant_message_index": 1}],
    }

    trimmed = training_mod._trim_episode_feature_messages_to_last_supervised_assistant(feature)

    assert trimmed is feature


def test_message_only_episode_spec_trims_only_trailing_zero_weight_tokens(monkeypatch: pytest.MonkeyPatch):
    batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long),
        "attention_mask": torch.ones((1, 10), dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100, -100, 4, 5, -100, 7, 8, -100]], dtype=torch.long),
        "token_advantages": torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
    }

    def fake_build_episode_batch_from_feature(*_args, **_kwargs):
        return training_mod.BatchBuildResult(
            batch=batch,
            cached_plan=None,
            completion_token_count=6,
            drop_reason=None,
            budgeting_attempted=False,
            is_episode_feature=True,
        )

    monkeypatch.setattr(training_mod, "_build_episode_batch_from_feature", fake_build_episode_batch_from_feature)

    result = training_mod._build_message_only_completion_episode_spec_from_feature(
        processor=object(),
        feature={"advantage": 1.0, "sample_weight": 1.0},
    )

    episode_spec = dict(result.batch or {})
    assert tuple(episode_spec["prompt_ids"].shape) == (1, 4)
    assert tuple(episode_spec["completion_ids"].shape) == (1, 5)
    assert episode_spec["completion_ids"].tolist() == [[4, 5, 6, 7, 8]]
    assert episode_spec["token_loss_weight"].tolist() == [[1.0, 1.0, 0.0, 1.0, 1.0]]
    assert int(episode_spec["completion_token_count"].item()) == 5


def test_message_only_episode_spec_skips_tail_trim_for_multimodal_batches(monkeypatch: pytest.MonkeyPatch):
    batch = {
        "input_ids": torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long),
        "attention_mask": torch.ones((1, 10), dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100, -100, 4, 5, -100, 7, 8, -100]], dtype=torch.long),
        "token_advantages": torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]], dtype=torch.float32),
        "pixel_values": torch.ones((16, 1536), dtype=torch.float32),
        "image_grid_thw": torch.tensor([[1, 4, 4]], dtype=torch.long),
    }

    def fake_build_episode_batch_from_feature(*_args, **_kwargs):
        return training_mod.BatchBuildResult(
            batch=batch,
            cached_plan=None,
            completion_token_count=6,
            drop_reason=None,
            budgeting_attempted=False,
            is_episode_feature=True,
        )

    monkeypatch.setattr(training_mod, "_build_episode_batch_from_feature", fake_build_episode_batch_from_feature)

    result = training_mod._build_message_only_completion_episode_spec_from_feature(
        processor=object(),
        feature={"advantage": 1.0, "sample_weight": 1.0},
    )

    episode_spec = dict(result.batch or {})
    assert tuple(episode_spec["prompt_ids"].shape) == (1, 4)
    assert tuple(episode_spec["completion_ids"].shape) == (1, 6)
    assert episode_spec["completion_ids"].tolist() == [[4, 5, 6, 7, 8, 9]]
    assert int(episode_spec["completion_token_count"].item()) == 6


def test_rl_completion_episode_spec_trims_messages_before_generic_builder(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_build_message_only_completion_episode_spec_from_feature(processor, feature, **kwargs):
        del processor, kwargs
        captured["messages"] = list(feature["messages"])
        captured["assistant_supervision"] = list(feature["assistant_supervision"])
        return training_mod.BatchBuildResult(
            batch={"completion_ids": torch.tensor([[1, 2]], dtype=torch.long)},
            cached_plan=None,
            completion_token_count=2,
            drop_reason=None,
            budgeting_attempted=False,
            is_episode_feature=True,
        )

    monkeypatch.setattr(
        training_mod,
        "_build_message_only_completion_episode_spec_from_feature",
        fake_build_message_only_completion_episode_spec_from_feature,
    )

    feature = {
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "scan"}]},
            {"role": "tool", "name": "scan_timeline", "content": [{"type": "text", "text": "obs"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "seek"}]},
            {
                "role": "tool",
                "name": "seek_evidence",
                "content": [
                    {"type": "text", "text": "obs"},
                    {"type": "image", "image": torch.ones((2, 2), dtype=torch.float32)},
                ],
            },
        ],
        "assistant_supervision": [
            {"assistant_message_index": 1},
            {"assistant_message_index": 3},
        ],
        "advantage": 1.0,
    }

    training_mod._build_rl_completion_episode_spec_from_feature(
        processor=object(),
        feature=feature,
    )

    assert [msg["role"] for msg in captured["messages"]] == ["system", "assistant", "tool", "assistant"]
    assert captured["assistant_supervision"] == feature["assistant_supervision"]


def test_episode_feature_debug_metadata_captures_retained_message_plan_and_image_provenance():
    fitted_messages = [
        {
            "role": "system",
            "_cache_message_index": 0,
            "content": [
                {
                    "type": "text",
                    "text": "sys",
                    "_cache_content_index": 0,
                }
            ],
        },
        {
            "role": "tool",
            "name": "scan_timeline",
            "_cache_message_index": 3,
            "content": [
                {
                    "type": "text",
                    "text": "1.000s",
                    "_cache_content_index": 0,
                },
                {
                    "type": "image",
                    "image": torch.ones((2, 2), dtype=torch.float32),
                    "image_ref": {
                        "video_path": "/tmp/example.mp4",
                        "sampled_frame_index": 4,
                        "raw_frame_index": 8,
                        "timestamp_sec": 1.0,
                    },
                    "_cache_content_index": 1,
                    "_visual_provenance": {
                        "kind": "image_ref",
                        "video_path": "/tmp/example.mp4",
                        "sampled_frame_index": 4,
                        "raw_frame_index": 8,
                        "timestamp_sec": 1.0,
                    },
                    "timestamp_sec": 1.0,
                },
            ],
        },
    ]
    assistant_supervision = [
        {
            "assistant_message_index": 2,
            "turn_index": 1,
            "turn_kind": "scan_timeline",
            "tool_name": "scan_timeline",
            "loss_weight": 2.0,
        }
    ]

    metadata = training_mod._build_episode_feature_debug_metadata(
        fitted_messages=fitted_messages,
        assistant_supervision=assistant_supervision,
    )

    assert metadata["message_plan"] == [
        {"message_index": 0, "content_indices": [0]},
        {"message_index": 3, "content_indices": [0, 1]},
    ]
    assert metadata["assistant_supervision"] == [
        {
            "assistant_message_index": 2,
            "turn_index": 1,
            "turn_kind": "scan_timeline",
            "tool_name": "scan_timeline",
            "loss_weight": 2.0,
        }
    ]
    assert metadata["retained_message_count"] == 2
    assert metadata["retained_image_provenance"] == [
        {
            "message_index": 3,
            "content_index": 1,
            "provenance": {
                "kind": "image_ref",
                "raw_frame_index": 8,
                "sampled_frame_index": 4,
                "timestamp_sec": 1.0,
                "video_path": "/tmp/example.mp4",
            },
            "timestamp_sec": 1.0,
        }
    ]
