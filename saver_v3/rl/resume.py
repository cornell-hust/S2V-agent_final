from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class RLTrainingResumeState:
    checkpoint_path: str
    epoch: float
    global_step: int
    resume_iteration: int
    completed_iteration: int
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def load_trainer_resume_state(
    checkpoint_path: str | Path,
    *,
    source: str = "",
) -> RLTrainingResumeState:
    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve()
    trainer_state_path = resolved_checkpoint / "trainer_state.json"
    if not trainer_state_path.exists():
        raise FileNotFoundError(
            f"RL resume checkpoint is missing trainer_state.json: checkpoint={resolved_checkpoint}"
        )
    payload = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    epoch_value = max(0.0, _safe_float(payload.get("epoch"), 0.0))
    global_step = max(0, _safe_int(payload.get("global_step"), 0))
    resume_iteration = max(0, int(round(epoch_value)))
    completed_iteration = max(-1, int(resume_iteration) - 1)
    return RLTrainingResumeState(
        checkpoint_path=str(resolved_checkpoint),
        epoch=float(epoch_value),
        global_step=int(global_step),
        resume_iteration=int(resume_iteration),
        completed_iteration=int(completed_iteration),
        source=str(source or ""),
    )


def _iter_summary_checkpoint_candidates(rl_dir: Path) -> Iterable[tuple[str, Path]]:
    for summary_path in sorted(rl_dir.glob("iter_*/summary.json")):
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        for key in (
            "trainer_resume_checkpoint",
            "standard_trainer_checkpoint",
            "epoch_resume_checkpoint",
            "loadable_authority_checkpoint",
            "latest_checkpoint",
        ):
            raw_value = str(payload.get(key) or "").strip()
            if raw_value:
                yield (f"{summary_path.name}:{key}", Path(raw_value).expanduser().resolve())


def _iter_root_checkpoint_candidates(rl_dir: Path) -> Iterable[tuple[str, Path]]:
    for checkpoint_dir in sorted(rl_dir.glob("checkpoint-*")):
        if checkpoint_dir.is_dir():
            yield ("root_checkpoint", checkpoint_dir.expanduser().resolve())


def resolve_rl_training_resume_state(rl_dir: str | Path) -> Optional[RLTrainingResumeState]:
    resolved_rl_dir = Path(rl_dir).expanduser().resolve()
    if not resolved_rl_dir.exists():
        return None
    best_state: Optional[RLTrainingResumeState] = None
    seen: set[str] = set()
    for source, checkpoint_dir in list(_iter_summary_checkpoint_candidates(resolved_rl_dir)) + list(
        _iter_root_checkpoint_candidates(resolved_rl_dir)
    ):
        checkpoint_key = str(checkpoint_dir)
        if checkpoint_key in seen:
            continue
        seen.add(checkpoint_key)
        try:
            candidate = load_trainer_resume_state(checkpoint_dir, source=source)
        except Exception:
            continue
        if best_state is None:
            best_state = candidate
            continue
        if candidate.resume_iteration > best_state.resume_iteration:
            best_state = candidate
            continue
        if candidate.resume_iteration == best_state.resume_iteration and candidate.global_step > best_state.global_step:
            best_state = candidate
    return best_state
