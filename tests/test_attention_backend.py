import unittest

from saver_v3.common.fa3_guard import (
    AttentionBackendUnavailableError,
    ensure_fa3_training_ready,
    resolve_attention_backend,
)


class AttentionBackendTests(unittest.TestCase):
    def test_resolve_attention_backend_accepts_fa3_on_hopper(self) -> None:
        decision = resolve_attention_backend(
            cuda_device_capabilities=[(9, 0)],
            module_available=lambda name: name in {"flash_attn", "flash_attn_interface"},
            env={"SAVER_V3_ATTN_BACKEND": "auto"},
        )

        self.assertEqual(decision.backend, "fa3")
        self.assertEqual(decision.attn_implementation, "flash_attention_3")
        self.assertEqual(decision.compute_capability_major, 9)

    def test_non_hopper_gpu_raises(self) -> None:
        with self.assertRaises(AttentionBackendUnavailableError):
            ensure_fa3_training_ready(
                cuda_device_capabilities=[(8, 0)],
                module_available=lambda name: name in {"flash_attn", "flash_attn_interface"},
            )

    def test_missing_fa3_module_raises(self) -> None:
        with self.assertRaises(AttentionBackendUnavailableError):
            resolve_attention_backend(
                cuda_device_capabilities=[(9, 0)],
                module_available=lambda name: False,
                env={"SAVER_V3_ATTN_BACKEND": "fa3"},
            )
