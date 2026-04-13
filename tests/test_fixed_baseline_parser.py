import unittest

from saver_v3.inference.fixed_baseline_eval import (
    normalize_fixed_baseline_prediction,
    parse_fixed_baseline_response_text,
)


class FixedBaselineParserTests(unittest.TestCase):
    def test_parse_strict_json_object_accepts_plain_json(self) -> None:
        payload, error = parse_fixed_baseline_response_text('{"decision": {"existence": "anomaly"}}')
        self.assertIsNone(error)
        self.assertEqual(payload["decision"]["existence"], "anomaly")

    def test_parse_strict_json_object_rejects_fenced_output(self) -> None:
        payload, error = parse_fixed_baseline_response_text('```json\n{"decision": {"existence": "anomaly"}}\n```')
        self.assertIsNone(payload)
        self.assertEqual(error, "fenced_output")

    def test_normalize_prediction_clamps_and_dedupes_evidence(self) -> None:
        normalized = normalize_fixed_baseline_prediction(
            {
                "decision": {
                    "existence": "anomaly",
                    "category": "Assault",
                    "severity": 4,
                    "anomaly_interval_sec": [5.0, 2.0],
                    "precursor_interval_sec": [-1.0, 1.0],
                },
                "evidence_topk": [
                    {"rank": 2, "start_sec": 1.0, "end_sec": 2.0, "role": "trigger", "description": "hit"},
                    {"rank": 1, "start_sec": 0.0, "end_sec": 1.0, "role": "precursor", "description": "approach"},
                    {"rank": 1, "start_sec": 0.0, "end_sec": 1.0, "role": "precursor", "description": "duplicate"},
                    {"rank": 3, "start_sec": 2.0, "end_sec": 8.0, "role": "confirmation", "description": "after"},
                ],
            },
            duration_sec=4.0,
            evidence_top_k=3,
        )
        self.assertEqual(normalized["decision"]["category"], "assault")
        self.assertEqual(normalized["decision"]["anomaly_interval_sec"], [2.0, 4.0])
        self.assertEqual(normalized["decision"]["precursor_interval_sec"], [0.0, 1.0])
        self.assertEqual([item["role"] for item in normalized["evidence_topk"]], ["precursor", "trigger", "confirmation"])


if __name__ == "__main__":
    unittest.main()


from saver_v3.inference.fixed_baseline_eval import FixedBaselineEvalConfig, run_fixed_baseline_eval_job


class FixedBaselineConfigValidationTests(unittest.TestCase):
    def test_config_parses_generation_cache_flag(self) -> None:
        config = FixedBaselineEvalConfig.from_mapping({
            "base_model": "/tmp/model",
            "torch_dtype": "bfloat16",
            "attn_implementation": "flash_attention_3",
            "client": {"batch_size": 1, "use_generation_cache": True},
            "io": {"data_path": "/tmp/data.jsonl", "output_dir": "/tmp/out"},
        })
        self.assertTrue(config.use_generation_cache)

    def test_run_rejects_batch_size_greater_than_one(self) -> None:
        config = FixedBaselineEvalConfig(
            base_model="/tmp/model",
            data_path="/tmp/data.jsonl",
            output_dir="/tmp/out",
            batch_size=2,
        )
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            run_fixed_baseline_eval_job(config)

    def test_run_rejects_non_fa3_attention(self) -> None:
        config = FixedBaselineEvalConfig(
            base_model="/tmp/model",
            data_path="/tmp/data.jsonl",
            output_dir="/tmp/out",
            attn_implementation="sdpa",
        )
        with self.assertRaisesRegex(ValueError, "flash_attention_3"):
            run_fixed_baseline_eval_job(config)
