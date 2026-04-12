import unittest

from saver_v3.model.trainability import assert_full_model_trainable, build_trainability_report


class FakeParameter:
    def __init__(self, count: int, *, requires_grad: bool) -> None:
        self._count = count
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._count


class FakeModel:
    def __init__(self) -> None:
        self._named_parameters = [
            ("language_model.layers.0.weight", FakeParameter(10, requires_grad=True)),
            ("vision_tower.blocks.0.weight", FakeParameter(5, requires_grad=False)),
            ("multimodal_projector.linear.weight", FakeParameter(3, requires_grad=True)),
            ("lm_head.weight", FakeParameter(2, requires_grad=True)),
        ]

    def named_parameters(self):
        return list(self._named_parameters)


class TrainabilityTests(unittest.TestCase):
    def test_build_trainability_report_groups_parameters(self) -> None:
        report = build_trainability_report(FakeModel())

        self.assertEqual(report["total_parameters"], 20)
        self.assertEqual(report["trainable_parameters"], 15)
        self.assertEqual(report["frozen_parameters"], 5)
        self.assertEqual(report["by_group"]["vision"]["frozen_parameters"], 5)
        self.assertEqual(report["by_group"]["projector"]["trainable_parameters"], 3)

    def test_assert_full_model_trainable_detects_frozen_parameters(self) -> None:
        with self.assertRaises(ValueError):
            assert_full_model_trainable(FakeModel())


if __name__ == "__main__":
    unittest.main()
