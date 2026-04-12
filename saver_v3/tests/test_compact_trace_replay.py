from saver_v3.data.compact_trace_replay import build_initial_messages, expand_compact_trace_row_to_sft_steps


def test_expand_compact_trace_row_to_sft_steps_builds_one_example_per_tool_call():
    row = {
        "video_id": "vid1",
        "split": "train",
        "video_path": "/tmp/example.mp4",
        "agent_task": {"task_prompt": "Inspect the video."},
        "tool_io": {"allowed_tools": ["scan_timeline", "finalize_case"]},
        "oracle_trajectory": [
            {"tool": "scan_timeline", "arguments": {"start_sec": 0.0, "end_sec": 2.0}},
            {"tool": "finalize_case", "arguments": {"existence": "normal", "category": "normal"}},
        ],
    }
    steps = expand_compact_trace_row_to_sft_steps(row)
    assert len(steps) == 2
    assert steps[0].target_action == "tool_call"
    assert "<tool_call>" in steps[0].target_response
    assert steps[1].tool_name == "finalize_case"


def test_build_initial_messages_contains_video_and_prompt():
    row = {
        "video_path": "/tmp/example.mp4",
        "agent_task": {"task_prompt": "Inspect the clip."},
        "tool_io": {"allowed_tools": ["scan_timeline"]},
    }
    messages = build_initial_messages(row)
    assert messages[1]["content"][0]["type"] == "video"
    assert "Inspect the clip" in messages[1]["content"][1]["text"]
