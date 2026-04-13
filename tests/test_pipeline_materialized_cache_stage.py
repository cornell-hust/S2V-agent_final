import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_full_pipeline.sh"


class PipelineMaterializedCacheStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT_PATH.read_text(encoding="utf-8")

    def test_pipeline_declares_materialized_cache_stage_and_paths(self) -> None:
        self.assertIn("Stage 1c: Materialized Cache Resolve", self.text)
        self.assertIn("SFT_MATERIALIZED_FILE", self.text)
        self.assertIn("RUNTIME_TRAIN_ITEMS_FILE", self.text)
        self.assertIn("RUNTIME_EVAL_ITEMS_FILE", self.text)

    def test_pipeline_passes_materialized_cache_overrides(self) -> None:
        self.assertIn('data.materialized_messages_path=${SFT_MATERIALIZED_FILE}', self.text)
        self.assertIn('io.materialized_items_path=${RUNTIME_EVAL_ITEMS_FILE}', self.text)
        self.assertIn('data.materialized_train_items_path=${RUNTIME_TRAIN_ITEMS_FILE}', self.text)
        self.assertIn('data.materialized_eval_items_path=${RUNTIME_EVAL_ITEMS_FILE}', self.text)


if __name__ == "__main__":
    unittest.main()
