import tempfile
import unittest
from pathlib import Path

from saver_v3.common.checkpoint_manifest import (
    CheckpointManifest,
    load_checkpoint_manifest,
    resolve_latest_checkpoint_manifest,
    write_checkpoint_manifest,
)


class CheckpointManifestTests(unittest.TestCase):
    def test_checkpoint_manifest_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_dir = Path(tmp_dir) / "checkpoint-000010"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "model.safetensors").write_text("weights", encoding="utf-8")

            manifest = CheckpointManifest(
                checkpoint_dir=str(checkpoint_dir),
                base_model="Qwen/Qwen3-VL-8B-Instruct",
                global_step=10,
                epoch=1.25,
                files=["model.safetensors"],
                metadata={"run_name": "unit-test"},
            )
            manifest_path = write_checkpoint_manifest(manifest)
            loaded = load_checkpoint_manifest(manifest_path)

            self.assertEqual(loaded.base_model, manifest.base_model)
            self.assertEqual(loaded.global_step, 10)
            self.assertEqual(loaded.files, ["model.safetensors"])
            self.assertEqual(loaded.metadata["run_name"], "unit-test")

    def test_resolve_latest_checkpoint_manifest_prefers_highest_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for step in (10, 20):
                checkpoint_dir = root / f"checkpoint-{step:06d}"
                checkpoint_dir.mkdir()
                write_checkpoint_manifest(
                    CheckpointManifest(
                        checkpoint_dir=str(checkpoint_dir),
                        base_model="Qwen/Qwen3-VL-8B-Instruct",
                        global_step=step,
                    )
                )

            latest = resolve_latest_checkpoint_manifest(root)

            self.assertIsNotNone(latest)
            self.assertEqual(latest.global_step, 20)


if __name__ == "__main__":
    unittest.main()
