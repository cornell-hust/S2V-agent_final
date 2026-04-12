import unittest

from saver_v3.common.runtime_env import distributed_runtime_from_env, resolve_runtime_env


class RuntimeEnvTests(unittest.TestCase):
    def test_distributed_runtime_from_env_clamps_invalid_values(self) -> None:
        runtime = distributed_runtime_from_env(
            {
                "WORLD_SIZE": "8",
                "RANK": "99",
                "LOCAL_RANK": "-3",
            }
        )

        self.assertEqual(runtime.world_size, 8)
        self.assertEqual(runtime.rank, 0)
        self.assertEqual(runtime.local_rank, 0)
        self.assertTrue(runtime.is_distributed)

    def test_resolve_runtime_env_reads_owned_defaults(self) -> None:
        env = resolve_runtime_env(
            {
                "HF_HOME": "/tmp/hf",
                "HF_TOKEN": "token-123",
                "SAVER_V3_DATA_ROOT": "/data/root",
                "SAVER_V3_OUTPUT_ROOT": "/outputs/root",
                "SAVER_V3_ATTN_BACKEND": "fa2",
                "SAVER_V3_DISABLE_FA3": "1",
                "SAVER_V3_DISABLE_FA2": "0",
                "WORLD_SIZE": "2",
                "RANK": "1",
                "LOCAL_RANK": "1",
            }
        )

        self.assertEqual(env.hf_home, "/tmp/hf")
        self.assertEqual(env.hf_token, "token-123")
        self.assertEqual(env.data_root, "/data/root")
        self.assertEqual(env.output_root, "/outputs/root")
        self.assertEqual(env.attention_backend_override, "fa2")
        self.assertTrue(env.disable_fa3)
        self.assertFalse(env.disable_fa2)
        self.assertEqual(env.runtime.rank, 1)


if __name__ == "__main__":
    unittest.main()
