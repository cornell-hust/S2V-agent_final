from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

from saver_v3.cli.common import load_yaml_mapping


@dataclass
class VllmPredictionConfig:
    deprecated_payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "VllmPredictionConfig":
        return cls(deprecated_payload=dict(mapping or {}))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "VllmPredictionConfig":
        return cls.from_mapping(load_yaml_mapping(path))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def run_vllm_prediction_job(config: VllmPredictionConfig) -> Dict[str, Any]:
    raise RuntimeError(
        "run_vllm_prediction_job is deprecated because it used the retired compact-trace message predictor path. "
        "Use saver_v3.inference.policy_rollout.run_policy_rollout_job instead. "
        f"Received deprecated payload keys: {sorted((config.deprecated_payload or {}).keys())}"
    )
