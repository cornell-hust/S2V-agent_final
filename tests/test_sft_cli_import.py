import importlib
import unittest


class SFTCLIImportTests(unittest.TestCase):
    def test_train_sft_cli_module_is_importable(self) -> None:
        module = importlib.import_module("saver_v3.cli.train_sft_ds")
        self.assertTrue(callable(getattr(module, "main", None)))


if __name__ == "__main__":
    unittest.main()
