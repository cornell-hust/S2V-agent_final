from .reward_adapter import score_rollout_trace_v2, summarize_reward_trace

__all__ = ["RLJobConfig", "run_rl_job", "score_rollout_trace_v2", "summarize_reward_trace"]


def __getattr__(name: str):
    if name in {"RLJobConfig", "run_rl_job"}:
        from .runtime import RLJobConfig, run_rl_job

        return {"RLJobConfig": RLJobConfig, "run_rl_job": run_rl_job}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
