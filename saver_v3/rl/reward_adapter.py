from __future__ import annotations

from typing import Any, Dict, Optional

from saver_v3.core.reward import score_rollout_trace


def score_rollout_trace_v2(
    rollout: Dict[str, Any],
    *,
    reward_version: str = "timesearch_v3",
    weights: Optional[Dict[str, float]] = None,
    reward_config: Optional[Dict[str, Any]] = None,
    llm_judge: Optional[Any] = None,
) -> Dict[str, Any]:
    return score_rollout_trace(
        dict(rollout or {}),
        reward_version=reward_version,
        weights=weights,
        reward_config=reward_config,
        llm_judge=llm_judge,
    )


def summarize_reward_trace(rollout: Dict[str, Any], *, reward_version: str = "timesearch_v3") -> Dict[str, Any]:
    summary = score_rollout_trace_v2(rollout, reward_version=reward_version)
    return {
        "total_reward": float(summary.get("total_reward") or 0.0),
        "components": dict(summary.get("components") or {}),
        "reward_version": str(reward_version),
    }
