import tempfile
import unittest
from pathlib import Path

import yaml

from saver_v3.data.prepare_sft_manifest import PrepareSFTManifestConfig


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


class PrepareSFTManifestConfigTests(unittest.TestCase):
    def test_prepare_config_loads_prompt_and_rollout_trace_from_sft_config_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            sft_config_path = tmp_path / "sft.yaml"
            prepare_config_path = tmp_path / "prepare.yaml"
            _write_yaml(
                sft_config_path,
                {
                    "preview": {"num_preview_frames": 12, "preview_sampling_fps": 1.5, "max_preview_frames": 12},
                    "prompt": {
                        "initial_user_template": "Case: {public_case_id}\nTask: {task_prompt}\n",
                        "preview_instruction": "Inspect ordered frames.\n",
                        "tool_response_template": "Selected frames: {timestamps}.\nUse tools.\n",
                    },
                    "rollout_trace": {
                        "record_observation_content": True,
                        "record_state_deltas": False,
                        "record_counterfactual_trace": False,
                        "record_message_history": True,
                    },
                },
            )
            _write_yaml(
                prepare_config_path,
                {
                    "saver_config_source": str(sft_config_path),
                    "io": {
                        "input_data_path": str(tmp_path / "runtime.jsonl"),
                        "output_path": str(tmp_path / "prepared.jsonl"),
                        "include_splits": "train",
                    },
                },
            )

            config = PrepareSFTManifestConfig.from_yaml(prepare_config_path)

            self.assertEqual(config.saver_config["preview"]["num_preview_frames"], 12)
            self.assertEqual(config.saver_config["prompt"]["initial_user_template"], "Case: {public_case_id}\nTask: {task_prompt}\n")
            self.assertEqual(config.saver_config["prompt"]["preview_instruction"], "Inspect ordered frames.\n")
            self.assertFalse(config.saver_config["rollout_trace"]["record_state_deltas"])


if __name__ == "__main__":
    unittest.main()
