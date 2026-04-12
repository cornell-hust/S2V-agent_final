"""Trainability reporting for full-model Qwen3-VL training."""

from __future__ import annotations

from typing import Any


def _parameter_count(parameter: Any) -> int:
    numel = getattr(parameter, "numel", None)
    if callable(numel):
        return int(numel())
    return int(numel or 0)


def _group_name(parameter_name: str) -> str:
    name = str(parameter_name or "")
    if any(token in name for token in ("vision_tower", "vision_model", "visual", "vision_encoder")):
        return "vision"
    if any(token in name for token in ("multimodal_projector", "mm_projector", "visual_projection", "projector")):
        return "projector"
    if any(token in name for token in ("language_model", "lm_head", "transformer", "model.layers", "text_model")):
        return "language"
    return "other"


def build_trainability_report(model: Any, *, frozen_name_sample_limit: int = 16) -> dict[str, Any]:
    by_group: dict[str, dict[str, int]] = {}
    total_parameters = 0
    trainable_parameters = 0
    frozen_parameter_names: list[str] = []

    for name, parameter in list(getattr(model, "named_parameters", lambda: [])()):
        count = _parameter_count(parameter)
        requires_grad = bool(getattr(parameter, "requires_grad", False))
        group_name = _group_name(name)
        group_stats = by_group.setdefault(group_name, {"total_parameters": 0, "trainable_parameters": 0, "frozen_parameters": 0})
        group_stats["total_parameters"] += count
        total_parameters += count
        if requires_grad:
            group_stats["trainable_parameters"] += count
            trainable_parameters += count
        else:
            group_stats["frozen_parameters"] += count
            if len(frozen_parameter_names) < int(frozen_name_sample_limit):
                frozen_parameter_names.append(str(name))

    frozen_parameters = total_parameters - trainable_parameters
    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": frozen_parameters,
        "trainable_fraction": (float(trainable_parameters) / float(total_parameters)) if total_parameters else 0.0,
        "fully_trainable": frozen_parameters == 0,
        "frozen_parameter_names_sample": frozen_parameter_names,
        "by_group": by_group,
    }


def assert_full_model_trainable(model: Any) -> dict[str, Any]:
    report = build_trainability_report(model)
    if report["frozen_parameters"] > 0:
        sample = ", ".join(report["frozen_parameter_names_sample"]) or "(unknown)"
        raise ValueError(
            "Full-model training requires every parameter to be trainable, but "
            f"{report['frozen_parameters']} parameters are frozen. Sample names: {sample}"
        )
    return report
