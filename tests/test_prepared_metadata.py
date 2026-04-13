import json
import tempfile
import unittest
from pathlib import Path

from saver_v3.data.config import SaverAgentConfig
from saver_v3.data.prepared_metadata import (
    build_jsonl_provenance,
    ensure_prepared_sft_metadata,
    write_prepared_sft_metadata,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class PreparedMetadataTests(unittest.TestCase):
    def test_ensure_prepared_sft_metadata_validates_runtime_provenance_and_detects_stale_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            runtime_path = tmp_path / "runtime.jsonl"
            prepared_path = tmp_path / "prepared.jsonl"
            _write_jsonl(runtime_path, [{"split": "train", "video_id": "v1"}])
            _write_jsonl(prepared_path, [{"prepared_format": "compact_trace_v2"}])
            write_prepared_sft_metadata(
                prepared_path,
                config=SaverAgentConfig(),
                extra_fields={
                    "source_runtime": build_jsonl_provenance(runtime_path, include_splits="train"),
                },
            )

            ensure_prepared_sft_metadata(
                prepared_path,
                expected_source_runtime_path=runtime_path,
                expected_source_runtime_include_splits="train",
                require_source_runtime=True,
            )

            _write_jsonl(runtime_path, [{"split": "train", "video_id": "v2"}, {"split": "train", "video_id": "v3"}])
            with self.assertRaisesRegex(ValueError, "source_runtime"):
                ensure_prepared_sft_metadata(
                    prepared_path,
                    expected_source_runtime_path=runtime_path,
                    expected_source_runtime_include_splits="train",
                    require_source_runtime=True,
                )

    def test_ensure_prepared_sft_metadata_enforces_teacher_flags_and_source_prepared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            base_prepared_path = tmp_path / "base.compact_trace_v2.jsonl"
            teacher_prepared_path = tmp_path / "teacher.compact_trace_v2.jsonl"
            _write_jsonl(base_prepared_path, [{"prepared_format": "compact_trace_v2"}])
            _write_jsonl(teacher_prepared_path, [{"prepared_format": "compact_trace_v2"}])
            write_prepared_sft_metadata(
                teacher_prepared_path,
                config=SaverAgentConfig(),
                extra_fields={
                    "source_prepared": build_jsonl_provenance(base_prepared_path, include_splits="train"),
                },
            )

            with self.assertRaisesRegex(ValueError, "teacher_annotated"):
                ensure_prepared_sft_metadata(
                    teacher_prepared_path,
                    expected_source_prepared_path=base_prepared_path,
                    expected_source_prepared_include_splits="train",
                    require_source_prepared=True,
                    require_teacher_annotated=True,
                    require_teacher_rollout_primary_materialized=True,
                )


if __name__ == "__main__":
    unittest.main()
