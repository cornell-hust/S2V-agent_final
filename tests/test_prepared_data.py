import json
import tempfile
import unittest
from pathlib import Path

from saver_v3.data.compact_trace import compact_trace_row_to_runtime_record, replay_compact_trace_messages
from saver_v3.data.compact_trace_loader import load_compact_trace_rows
from saver_v3.data.prepared_loader import load_prepared_rows
from saver_v3.data.prepared_schema import PreparedDataError, validate_prepared_row


def _sample_row() -> dict:
    return {
        "schema_version": 1,
        "prepared_format": "compact_trace_v2",
        "video_id": "video-001",
        "video_path": "/data/videos/video-001.mp4",
        "split": "train",
        "source": "oracle_sft_compact_trace_v2",
        "agent_task": {"task_prompt": "Inspect the clip and explain the anomaly."},
        "structured_target": {"existence": "anomaly", "category": "fall"},
        "qa_pairs": [{"question": "What happened?", "answer": "A person fell."}],
        "oracle_trajectory": [
            {
                "tool": "seek_evidence",
                "arguments": {"query": "falling down"},
                "observation": {"status": "ok", "hits": [12, 13]},
            },
            {
                "tool": "finalize_case",
                "arguments": {"summary": "A person falls near the doorway."},
                "observation": {"status": "done"},
            },
        ],
        "oracle_final_decision": {"summary": "A person falls near the doorway."},
    }


class PreparedDataTests(unittest.TestCase):
    def test_validate_prepared_row_accepts_compact_trace_v2(self) -> None:
        row = validate_prepared_row(_sample_row())

        self.assertEqual(row["video_id"], "video-001")
        self.assertEqual(row["oracle_trajectory"][0]["tool"], "seek_evidence")

    def test_validate_prepared_row_rejects_compact_trace_v3(self) -> None:
        row = _sample_row()
        row["prepared_format"] = "compact_trace_v3"

        with self.assertRaises(PreparedDataError):
            validate_prepared_row(row)

    def test_validate_prepared_row_rejects_missing_trajectory(self) -> None:
        row = _sample_row()
        row.pop("oracle_trajectory")

        with self.assertRaises(PreparedDataError):
            validate_prepared_row(row)

    def test_load_prepared_rows_reads_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prepared_path = Path(tmp_dir) / "prepared.jsonl"
            prepared_path.write_text(json.dumps(_sample_row()) + "\n", encoding="utf-8")

            rows = load_prepared_rows(prepared_path)

            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["video_path"].endswith("video-001.mp4"))

    def test_load_compact_trace_rows_returns_normalized_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            prepared_path = Path(tmp_dir) / "prepared.jsonl"
            row = _sample_row()
            row.pop("video_id")
            row["id"] = "video-from-id"
            prepared_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

            rows = load_compact_trace_rows(prepared_path)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["video_id"], "video-from-id")
            self.assertEqual(rows[0]["prepared_format"], "compact_trace_v2")

    def test_replay_compact_trace_messages_stops_before_requested_step(self) -> None:
        row = _sample_row()

        messages = replay_compact_trace_messages(row, stop_before_step_index=2)
        record = compact_trace_row_to_runtime_record(row)

        self.assertEqual(record["oracle_sft"]["trajectory"][0]["tool"], "seek_evidence")
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertIn("<tool_call>", messages[2]["content"][0]["text"])
        self.assertEqual(messages[3]["role"], "tool")
        self.assertEqual(messages[3]["name"], "seek_evidence")


if __name__ == "__main__":
    unittest.main()
