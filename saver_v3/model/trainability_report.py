from __future__ import annotations

from typing import Any, Dict


def _bucket_for_parameter_name(name: str) -> str:
    lowered = str(name).lower()
    if any(token in lowered for token in ("vision", "visual", "image", "patch_embed", "merger")):
        return "vision"
    return "language"


def build_trainability_report(model: Any) -> Dict[str, Any]:
    report = {
        "vision_total_params": 0,
        "vision_trainable_params": 0,
        "language_total_params": 0,
        "language_trainable_params": 0,
    }
    for name, parameter in model.named_parameters():
        bucket = _bucket_for_parameter_name(name)
        count = int(parameter.numel())
        report[f"{bucket}_total_params"] += count
        if bool(getattr(parameter, "requires_grad", False)):
            report[f"{bucket}_trainable_params"] += count
    report["full_model_trainable"] = (
        report["vision_trainable_params"] > 0 and report["language_trainable_params"] > 0
    )
    return report


def collect_grad_norm_report(model: Any) -> Dict[str, float]:
    result = {"vision_grad_norm": 0.0, "language_grad_norm": 0.0}
    for name, parameter in model.named_parameters():
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        bucket = _bucket_for_parameter_name(name)
        result[f"{bucket}_grad_norm"] += float(grad.detach().norm().item())
    return result
