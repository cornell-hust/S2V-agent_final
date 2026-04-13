import unittest
from pathlib import Path


TRAINING_PATH = Path(__file__).resolve().parents[1] / "saver_v3" / "sft" / "training.py"


class SFTProgressConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = TRAINING_PATH.read_text(encoding="utf-8")

    def test_sft_uses_transformers_builtin_tqdm(self) -> None:
        self.assertNotIn('"disable_tqdm": True', self.text)
        self.assertEqual(self.text.count('"disable_tqdm": False'), 2)

    def test_sft_does_not_register_custom_epoch_progress_callback(self) -> None:
        self.assertNotIn('trainer.add_callback(_build_epoch_progress_callback(trainer=trainer))', self.text)
        self.assertNotIn('def _build_epoch_progress_callback(', self.text)


if __name__ == "__main__":
    unittest.main()
