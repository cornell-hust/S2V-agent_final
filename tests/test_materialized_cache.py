import importlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from saver_v3.data.config import SaverAgentConfig
from saver_v3.data.prepared_schema import PREPARED_SFT_FORMAT

from saver_v3.data.materialized_cache import (
    MATERIALIZED_RUNTIME_ITEMS_FORMAT,
    MATERIALIZED_SFT_MESSAGES_FORMAT,
    build_runtime_materialized_rows,
    build_sft_materialized_rows,
    ensure_materialized_cache_metadata,
    load_materialized_sft_rows,
    validate_materialized_runtime_item_row,
    validate_materialized_sft_row,
    write_materialized_cache_metadata,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _compact_trace_row() -> dict:
    return {
        "schema_version": 1,
        "prepared_format": PREPARED_SFT_FORMAT,
        "video_id": "video-001",
        "video_path": "/tmp/video-001.mp4",
        "split": "train",
        "source": "oracle_sft_compact_trace_v2",
        "agent_task": {"task_prompt": "Inspect the clip."},
        "oracle_trajectory": [
            {"tool": "seek_evidence", "arguments": {"query": "falling"}, "sample_weight": 2.0},
            {"tool": "finalize_case", "arguments": {"summary": "A fall occurs."}, "sample_weight": 1.0},
        ],
    }


class MaterializedCacheTests(unittest.TestCase):
    def test_validate_materialized_sft_row_and_load_helpers_accept_valid_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            row = validate_materialized_sft_row(
                {
                    "schema_version": 1,
                    "materialized_format": MATERIALIZED_SFT_MESSAGES_FORMAT,
                    "video_id": "video-001",
                    "split": "train",
                    "sample_weight": 1.25,
                    "messages": [
                        {"role": "system", "content": [{"type": "text", "text": "You are SAVER."}]},
                        {"role": "user", "content": [{"type": "text", "text": "Inspect the clip."}]},
                    ],
                }
            )
            self.assertEqual(row["materialized_format"], MATERIALIZED_SFT_MESSAGES_FORMAT)
            self.assertEqual(row["sample_weight"], 1.25)

            path = Path(tmp_dir) / "materialized_sft.jsonl"
            _write_jsonl(path, [row])
            loaded = load_materialized_sft_rows(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["video_id"], "video-001")

    def test_validate_materialized_runtime_item_row_rejects_missing_record(self) -> None:
        with self.assertRaisesRegex(ValueError, "record"):
            validate_materialized_runtime_item_row(
                {
                    "schema_version": 1,
                    "materialized_format": MATERIALIZED_RUNTIME_ITEMS_FORMAT,
                    "video_id": "video-rt-001",
                    "messages": [],
                    "multimodal_cache": {},
                }
            )

    def test_ensure_materialized_cache_metadata_detects_stale_source_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            source_path = tmp_path / "source.jsonl"
            output_path = tmp_path / "materialized.jsonl"
            _write_jsonl(source_path, [{"split": "train", "video_id": "video-001"}])
            _write_jsonl(output_path, [{"materialized_format": MATERIALIZED_SFT_MESSAGES_FORMAT}])

            write_materialized_cache_metadata(
                output_path,
                materialized_format=MATERIALIZED_SFT_MESSAGES_FORMAT,
                config=SaverAgentConfig(),
                source_path=source_path,
                include_splits="train",
            )

            ensure_materialized_cache_metadata(
                output_path,
                expected_format=MATERIALIZED_SFT_MESSAGES_FORMAT,
                expected_source_path=source_path,
                expected_include_splits="train",
            )

            _write_jsonl(source_path, [{"split": "train", "video_id": "video-002"}])
            with self.assertRaisesRegex(ValueError, "source_jsonl"):
                ensure_materialized_cache_metadata(
                    output_path,
                    expected_format=MATERIALIZED_SFT_MESSAGES_FORMAT,
                    expected_source_path=source_path,
                    expected_include_splits="train",
                )

    def test_build_sft_materialized_rows_uses_replay_messages_and_episode_weight(self) -> None:
        with mock.patch(
            "saver_v3.data.materialized_cache.replay_compact_trace_messages",
            return_value=[
                {"role": "system", "content": [{"type": "text", "text": "sys"}]},
                {"role": "user", "content": [{"type": "text", "text": "usr"}]},
            ],
        ) as replay_mock:
            rows = build_sft_materialized_rows([_compact_trace_row()], config=SaverAgentConfig(), strict=True)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["materialized_format"], MATERIALIZED_SFT_MESSAGES_FORMAT)
        self.assertAlmostEqual(rows[0]["sample_weight"], 1.5)
        replay_mock.assert_called_once()


    def test_build_sft_materialized_rows_strips_inline_tensors_to_image_refs(self) -> None:
        import torch

        with mock.patch(
            "saver_v3.data.materialized_cache.replay_compact_trace_messages",
            return_value=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": torch.zeros(3, 2, 2),
                            "sampled_frame_index": 1,
                            "raw_frame_index": 7,
                            "timestamp_sec": 3.5,
                        },
                        {"type": "text", "text": "Inspect."},
                    ],
                }
            ],
        ):
            rows = build_sft_materialized_rows([_compact_trace_row()], config=SaverAgentConfig(), strict=True)

        json.dumps(rows[0], ensure_ascii=False)
        image_item = rows[0]["messages"][0]["content"][0]
        self.assertEqual(image_item["type"], "image")
        self.assertNotIn("image", image_item)
        self.assertEqual(image_item["image_ref"]["video_path"], "/tmp/video-001.mp4")
        self.assertEqual(image_item["image_ref"]["sampled_frame_index"], 1)
        self.assertEqual(image_item["image_ref"]["raw_frame_index"], 7)

    def test_build_runtime_materialized_rows_strips_tensor_heavy_fields_into_refs(self) -> None:
        runtime_row = {
            "video_id": "video-rt-001",
            "video_path": "/tmp/video-rt-001.mp4",
            "split": "train",
            "video_meta": {"duration_sec": 10.0, "fps": 2.0},
            "agent_task": {"task_prompt": "Inspect runtime row."},
        }
        built_item = {
            "video": runtime_row["video_path"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": object(),
                            "sampled_frame_index": 0,
                            "raw_frame_index": 4,
                            "timestamp_sec": 2.0,
                        },
                        {"type": "text", "text": "Inspect runtime row."},
                    ],
                }
            ],
            "multimodal_cache": {
                "video": object(),
                "embedding": object(),
                "fps": 2.0,
                "duration": 10.0,
                "question": "Inspect runtime row.",
                "structured_target": {},
                "tool_io": {},
                "video_path": runtime_row["video_path"],
                "video_meta": runtime_row["video_meta"],
                "frame_indices": [4],
                "preview_frames": object(),
                "preview_timestamps": [2.0],
                "preview_frame_indices": [0],
                "proposal_runtime": object(),
                "strict_feature_guided_proposal": False,
            },
        }

        builder_instance = mock.Mock()
        builder_instance.build_item.return_value = built_item
        with mock.patch("saver_v3.data.materialized_cache.SaverRecordItemBuilder", return_value=builder_instance):
            rows = build_runtime_materialized_rows([runtime_row], config=SaverAgentConfig(), data_root="", strict=True)

        self.assertEqual(len(rows), 1)
        normalized = rows[0]
        self.assertEqual(normalized["materialized_format"], MATERIALIZED_RUNTIME_ITEMS_FORMAT)
        image_item = normalized["messages"][0]["content"][0]
        self.assertEqual(image_item["type"], "image")
        self.assertNotIn("image", image_item)
        self.assertEqual(image_item["image_ref"]["video_path"], runtime_row["video_path"])
        cache = normalized["multimodal_cache"]
        self.assertIsNone(cache["video"])
        self.assertIsNone(cache["embedding"])
        self.assertTrue(cache["frame_cache_path"].endswith(".frame_cache"))
        self.assertTrue(cache["feature_cache_path"].endswith(".feature_cache"))

    def test_prepare_materialized_cache_module_is_importable(self) -> None:
        module = importlib.import_module("prepare_materialized_cache")
        self.assertTrue(callable(getattr(module, "main", None)))


if __name__ == "__main__":
    unittest.main()
