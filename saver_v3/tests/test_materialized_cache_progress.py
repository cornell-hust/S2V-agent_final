from __future__ import annotations

from pathlib import Path

from data_utils.jsonl import write_jsonl
from saver_v3.data import materialized_cache as materialized_cache_mod
from saver_v3.data.protocol_signature import build_protocol_signature


def test_build_sft_materialized_rows_reports_progress(monkeypatch):
    progress_steps: list[int] = []

    monkeypatch.setattr(
        materialized_cache_mod,
        "validate_compact_trace_row",
        lambda row: dict(row, prepared_format="compact_trace_v5"),
    )
    monkeypatch.setattr(
        materialized_cache_mod,
        "_replay_compact_trace_messages",
        lambda *args, **kwargs: [{"role": "user", "content": [{"type": "text", "text": "ok"}]}],
    )

    rows = [
        {
            "runtime_contract_version": 5,
            "protocol_signature": build_protocol_signature(),
            "structured_target": {"existence": "normal", "category": "normal"},
            "tool_io": {"finalize_case_schema": {"type": "object", "properties": {}, "required": []}},
            "video_id": "vid1",
            "split": "train",
            "source": "msad",
            "video_path": "/tmp/1.mp4",
        },
        {
            "runtime_contract_version": 5,
            "protocol_signature": build_protocol_signature(),
            "structured_target": {"existence": "normal", "category": "normal"},
            "tool_io": {"finalize_case_schema": {"type": "object", "properties": {}, "required": []}},
            "video_id": "vid2",
            "split": "train",
            "source": "msad",
            "video_path": "/tmp/2.mp4",
        },
    ]

    materialized = materialized_cache_mod.build_sft_materialized_rows(
        rows,
        progress_callback=progress_steps.append,
    )

    assert len(materialized) == 2
    assert progress_steps == [1, 2]


def test_build_runtime_materialized_rows_reports_progress(monkeypatch):
    progress_steps: list[int] = []

    class _FakeRecordBuilder:
        def __init__(self, *args, **kwargs):
            pass

        def build_item(self, record):
            return {
                "video_id": record.get("video_id"),
                "split": record.get("split"),
                "source": record.get("source"),
                "video": record.get("video_path"),
                "messages": [{"role": "user", "content": [{"type": "text", "text": "ok"}]}],
                "multimodal_cache": {"video_path": record.get("video_path")},
            }

    monkeypatch.setattr(materialized_cache_mod, "SaverRecordItemBuilder", _FakeRecordBuilder)

    rows = [
        {
            "runtime_contract_version": 5,
            "protocol_signature": build_protocol_signature(),
            "structured_target": {"existence": "normal", "category": "normal"},
            "tool_io": {"finalize_case_schema": {"type": "object", "properties": {}, "required": []}},
            "video_id": "vid1",
            "split": "train",
            "source": "msad",
            "video_path": "/tmp/1.mp4",
        },
        {
            "runtime_contract_version": 5,
            "protocol_signature": build_protocol_signature(),
            "structured_target": {"existence": "normal", "category": "normal"},
            "tool_io": {"finalize_case_schema": {"type": "object", "properties": {}, "required": []}},
            "video_id": "vid2",
            "split": "train",
            "source": "msad",
            "video_path": "/tmp/2.mp4",
        },
    ]

    materialized = materialized_cache_mod.build_runtime_materialized_rows(
        rows,
        progress_callback=progress_steps.append,
    )

    assert len(materialized) == 2
    assert progress_steps == [1, 2]


def test_write_jsonl_reports_progress(tmp_path: Path):
    progress_steps: list[int] = []
    output_path = tmp_path / "rows.jsonl"

    write_jsonl(
        [{"value": 1}, {"value": 2}, {"value": 3}],
        output_path,
        progress_callback=progress_steps.append,
    )

    assert output_path.exists()
    assert progress_steps == [1, 2, 3]
