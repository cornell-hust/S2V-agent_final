import re
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_full_pipeline.sh"


class RunFullPipelineSFTCheckpointGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT_PATH.read_text(encoding="utf-8")

    def test_script_defines_sft_checkpoint_completeness_helper(self) -> None:
        self.assertIn("is_complete_sft_epoch_output", self.text)
        self.assertIn("resolve_sft_epoch_model_path", self.text)
        self.assertIn("sft_summary.json", self.text)
        self.assertIn("authoritative_model_path", self.text)

    def test_training_skip_gate_is_not_directory_only(self) -> None:
        old_gate = 'if [[ -d "${EPOCH_OUT}" ]]; then\n    skip "SFT epoch ${epoch} output already exists'
        self.assertNotIn(old_gate, self.text)
        self.assertIn('if is_complete_sft_epoch_output "${EPOCH_OUT}"; then', self.text)

    def test_stale_epoch_dir_is_removed_before_retraining(self) -> None:
        self.assertIn('stale/incomplete', self.text)
        self.assertIn('rm -rf "${EPOCH_OUT}"', self.text)

    def test_eval_validates_epoch_checkpoint_before_torchrun(self) -> None:
        pattern = re.compile(r'require_valid_sft_epoch_output\s+"\$\{EPOCH_OUT\}".*?torchrun', re.S)
        self.assertRegex(self.text, pattern)

    def test_pipeline_uses_resolved_checkpoint_path_for_resume_eval_and_rl(self) -> None:
        self.assertIn('EPOCH_MODEL_PATH="$(resolve_sft_epoch_model_path "${EPOCH_OUT}")"', self.text)
        self.assertIn('PREV_EPOCH_MODEL_PATH="$(resolve_sft_epoch_model_path "${PREV_EPOCH_OUT}")"', self.text)
        self.assertIn('LAST_SFT_CKPT="$(resolve_sft_epoch_model_path "${LAST_SFT_EPOCH_DIR}")"', self.text)
        self.assertIn('base_model=${EPOCH_MODEL_PATH}', self.text)
        self.assertIn('resume_from_checkpoint=${PREV_EPOCH_MODEL_PATH}', self.text)


if __name__ == "__main__":
    unittest.main()
