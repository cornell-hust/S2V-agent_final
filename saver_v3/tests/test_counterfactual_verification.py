import sys
from types import ModuleType

try:
    import torch  # noqa: F401
except Exception:
    fake_torch = ModuleType("torch")
    fake_torch.Tensor = type("Tensor", (), {})
    fake_torch.dtype = type("dtype", (), {})
    fake_torch.device = lambda *_args, **_kwargs: "cpu"
    fake_torch.float32 = "float32"
    fake_torch.long = "long"
    sys.modules["torch"] = fake_torch

from saver_v3.core.counterfactual_verification import _stage_for_entry, _stage_lookup_from_target, _stage_window_ids


def test_stage_for_entry_maps_peak_and_evidence_roles():
    assert _stage_for_entry({"role": "peak"}) == "trigger"
    assert _stage_for_entry({"role": "evidence"}) == "precursor"


def test_stage_for_entry_falls_back_to_moment_stage_lookup():
    stage_by_moment_id = {"oracle_precursor": "precursor", "ev_trigger": "trigger"}

    assert _stage_for_entry({"moment_id": "oracle_precursor"}, stage_by_moment_id=stage_by_moment_id) == "precursor"
    assert _stage_for_entry({"moment_id": "ev_trigger"}, stage_by_moment_id=stage_by_moment_id) == "trigger"


def test_stage_window_ids_uses_role_and_moment_fallbacks():
    stage_by_moment_id = _stage_lookup_from_target(
        {
            "event_chain_target": {
                "stage_to_moment_ids": {
                    "precursor": ["oracle_precursor"],
                    "trigger": ["ev2"],
                    "confirmation": ["ev4"],
                }
            }
        }
    )
    records = [
        {"window_id": "w0002", "role": "evidence", "moment_id": "ev1"},
        {"window_id": "w0003", "role": None, "moment_id": "oracle_precursor"},
        {"window_id": "w0004", "role": "peak", "moment_id": "ev2"},
        {"window_id": "w0005", "role": None, "moment_id": "ev4"},
    ]

    assert _stage_window_ids(records, stage_by_moment_id=stage_by_moment_id) == {
        "precursor": ["w0002", "w0003"],
        "trigger": ["w0004"],
        "confirmation": ["w0005"],
    }
