from saver_v3.data.prepared_schema import PREPARED_SFT_FORMAT, PreparedDataError, validate_compact_trace_row


def test_validate_compact_trace_row_accepts_minimal_valid_payload():
    row = {
        "prepared_format": PREPARED_SFT_FORMAT,
        "video_path": "/tmp/example.mp4",
        "oracle_trajectory": [],
        "agent_task": {},
    }
    validate_compact_trace_row(row)


def test_validate_compact_trace_row_rejects_missing_video_path():
    row = {
        "prepared_format": PREPARED_SFT_FORMAT,
        "video_path": "",
        "oracle_trajectory": [],
        "agent_task": {},
    }
    try:
        validate_compact_trace_row(row)
    except ValueError as exc:
        assert "video_path" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_compact_trace_row_rejects_compact_trace_v3():
    row = {
        "prepared_format": "compact_trace_v3",
        "video_path": "/tmp/example.mp4",
        "oracle_trajectory": [],
        "agent_task": {},
    }
    try:
        validate_compact_trace_row(row)
    except PreparedDataError as exc:
        assert "prepared_format" in str(exc)
    else:
        raise AssertionError("expected PreparedDataError")
