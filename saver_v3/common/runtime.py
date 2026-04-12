from __future__ import annotations

import fcntl
import os
import re
import shutil
import sys
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, TypeVar


T = TypeVar("T")

_RUNTIME_LOG_LINE_PATTERN = re.compile(
    r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[(?:main|rank \d+/\d+)\](?: .*)?$"
)
DEFAULT_PROCESS_GROUP_TIMEOUT_SEC = 3600.0
_DYNAMIC_TASK_FILENAME_RE = re.compile(r"^task\.(\d{8})\.json$")


@dataclass(frozen=True)
class DistributedRuntime:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


@dataclass(frozen=True)
class ShardSpec:
    num_shards: int = 1
    shard_index: int = 0

    @property
    def is_sharded(self) -> bool:
        return self.num_shards > 1


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def distributed_runtime_from_env(env: Optional[Mapping[str, str]] = None) -> DistributedRuntime:
    env = env or os.environ
    world_size = max(1, _safe_int(env.get("WORLD_SIZE", 1), 1))
    rank = _safe_int(env.get("RANK", 0), 0)
    local_rank = _safe_int(env.get("LOCAL_RANK", rank), rank)
    if rank < 0:
        rank = 0
    if rank >= world_size:
        rank = 0
    if local_rank < 0:
        local_rank = 0
    return DistributedRuntime(rank=rank, world_size=world_size, local_rank=local_rank)


def resolve_shard_spec(
    *,
    num_shards: int = 0,
    shard_index: int = -1,
    runtime: Optional[DistributedRuntime] = None,
) -> ShardSpec:
    runtime = runtime or distributed_runtime_from_env()
    resolved_num_shards = int(num_shards)
    resolved_shard_index = int(shard_index)

    if resolved_num_shards < 0:
        raise ValueError("num_shards must be non-negative.")
    if resolved_num_shards == 0 and resolved_shard_index >= 0:
        raise ValueError("shard_index requires num_shards to be provided.")

    if resolved_num_shards == 0:
        if runtime.is_distributed:
            resolved_num_shards = int(runtime.world_size)
            resolved_shard_index = int(runtime.rank)
        else:
            resolved_num_shards = 1
            resolved_shard_index = 0
    elif resolved_shard_index < 0:
        resolved_shard_index = int(runtime.rank) if runtime.is_distributed else 0

    if resolved_num_shards < 1:
        raise ValueError("num_shards must be at least 1 after resolution.")
    if not 0 <= resolved_shard_index < resolved_num_shards:
        raise ValueError(
            f"Resolved shard_index={resolved_shard_index} is outside [0, {resolved_num_shards - 1}]."
        )
    return ShardSpec(num_shards=resolved_num_shards, shard_index=resolved_shard_index)


def shard_sequence(values: Sequence[T], *, num_shards: int, shard_index: int) -> list[T]:
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1.")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index={shard_index} must satisfy 0 <= shard_index < {num_shards}.")
    return [value for index, value in enumerate(values) if index % num_shards == shard_index]


def _dynamic_task_ready_path(queue_dir: str | Path) -> Path:
    return Path(queue_dir) / "_READY"


def _dynamic_task_pending_dir(queue_dir: str | Path) -> Path:
    return Path(queue_dir) / "pending"


def _dynamic_task_claimed_dir(queue_dir: str | Path) -> Path:
    return Path(queue_dir) / "claimed"


def _dynamic_task_completion_counter_path(queue_dir: str | Path) -> Path:
    return Path(queue_dir) / "_COMPLETED_COUNT"


def _dynamic_task_filename(task_index: int) -> str:
    return f"task.{int(task_index):08d}.json"


def _dynamic_task_index_from_path(path: str | Path) -> Optional[int]:
    match = _DYNAMIC_TASK_FILENAME_RE.fullmatch(Path(path).name)
    if match is None:
        return None
    return int(match.group(1))


def initialize_dynamic_task_queue(
    queue_dir: str | Path,
    *,
    num_tasks: int,
    runtime: Optional[DistributedRuntime] = None,
    timeout_sec: float = DEFAULT_PROCESS_GROUP_TIMEOUT_SEC,
    poll_interval_sec: float = 0.1,
) -> Path:
    runtime = runtime or distributed_runtime_from_env()
    queue_path = Path(queue_dir)
    ready_path = _dynamic_task_ready_path(queue_path)
    pending_dir = _dynamic_task_pending_dir(queue_path)
    claimed_dir = _dynamic_task_claimed_dir(queue_path)
    completion_counter_path = _dynamic_task_completion_counter_path(queue_path)

    if runtime.is_main_process:
        if queue_path.exists():
            shutil.rmtree(queue_path)
        pending_dir.mkdir(parents=True, exist_ok=True)
        claimed_dir.mkdir(parents=True, exist_ok=True)
        for task_index in range(max(0, int(num_tasks))):
            (pending_dir / _dynamic_task_filename(task_index)).write_text("", encoding="utf-8")
        completion_counter_path.write_text("0\n", encoding="utf-8")
        ready_path.write_text(
            f"{max(0, int(num_tasks))}\n",
            encoding="utf-8",
        )
        return queue_path

    deadline = None if float(timeout_sec) <= 0.0 else time.time() + max(1.0, float(timeout_sec))
    while not ready_path.exists():
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError(f"Timed out while waiting for dynamic task queue initialization at {queue_path}")
        time.sleep(max(0.01, float(poll_interval_sec)))
    return queue_path


def claim_next_dynamic_task_index(
    queue_dir: str | Path,
    *,
    runtime: Optional[DistributedRuntime] = None,
) -> Optional[int]:
    runtime = runtime or distributed_runtime_from_env()
    pending_dir = _dynamic_task_pending_dir(queue_dir)
    claimed_dir = _dynamic_task_claimed_dir(queue_dir)
    if not pending_dir.exists():
        return None
    claimed_dir.mkdir(parents=True, exist_ok=True)

    for pending_path in sorted(pending_dir.glob("task.*.json")):
        task_index = _dynamic_task_index_from_path(pending_path)
        if task_index is None:
            continue
        claimed_path = claimed_dir / f"{pending_path.stem}.rank{int(runtime.rank):02d}.json"
        try:
            pending_path.rename(claimed_path)
        except FileNotFoundError:
            continue
        except OSError:
            continue
        return task_index
    return None


def record_dynamic_task_completion(
    queue_dir: str | Path,
    *,
    runtime: Optional[DistributedRuntime] = None,
) -> int:
    runtime = runtime or distributed_runtime_from_env()
    counter_path = _dynamic_task_completion_counter_path(queue_dir)
    counter_path.parent.mkdir(parents=True, exist_ok=True)
    with counter_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            current_value = _safe_int(handle.read().strip(), 0)
            completed = current_value + 1
            handle.seek(0)
            handle.truncate()
            handle.write(f"{completed}\n")
            handle.flush()
            os.fsync(handle.fileno())
            return completed
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def sharded_output_path(output_path: str | Path, *, num_shards: int, shard_index: int) -> Path:
    path = Path(output_path)
    if num_shards <= 1:
        return path
    shard_tag = f"shard{int(shard_index):02d}-of-{int(num_shards):02d}"
    if path.suffix:
        return path.with_name(f"{path.stem}.{shard_tag}{path.suffix}")
    return path / shard_tag


def resolve_inference_device_map(device_map: Any, *, runtime: Optional[DistributedRuntime] = None) -> Any:
    runtime = runtime or distributed_runtime_from_env()
    if runtime.is_distributed and device_map == "auto":
        try:
            import torch
        except Exception:
            return {"": int(runtime.local_rank)}
        if not torch.cuda.is_available():
            return {"": int(runtime.local_rank)}
        try:
            visible_cuda_devices = int(torch.cuda.device_count())
        except Exception:
            return {"": int(runtime.local_rank)}
        if visible_cuda_devices <= 0:
            return {"": int(runtime.local_rank)}
        local_rank = int(runtime.local_rank)
        if 0 <= local_rank < visible_cuda_devices:
            return {"": local_rank}
        return {"": 0}
    return device_map


def runtime_prefix(runtime: Optional[DistributedRuntime] = None) -> str:
    runtime = runtime or distributed_runtime_from_env()
    if runtime.is_distributed:
        return f"[rank {runtime.rank}/{runtime.world_size}]"
    return "[main]"


def log_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write_runtime_line(message: str) -> None:
    try:
        from tqdm.auto import tqdm
    except Exception:
        print(str(message), flush=True)
        return
    tqdm_write = getattr(tqdm, "write", None)
    if callable(tqdm_write):
        try:
            tqdm_write(str(message))
            return
        except Exception:
            pass
    print(str(message), flush=True)


def runtime_log(
    message: str,
    *,
    runtime: Optional[DistributedRuntime] = None,
    main_process_only: bool = False,
) -> None:
    runtime = runtime or distributed_runtime_from_env()
    if main_process_only and not runtime.is_main_process:
        return
    if _RUNTIME_LOG_LINE_PATTERN.match(str(message)):
        _write_runtime_line(str(message))
        return
    _write_runtime_line(f"[{log_timestamp()}] {runtime_prefix(runtime)} {message}")


def should_log_progress(completed: int, total: int, every: int) -> bool:
    if total <= 0:
        return False
    if completed <= 0:
        return False
    if completed == 1 or completed == total:
        return True
    if every <= 0:
        return False
    return completed % every == 0


class _NullProgressBar:
    def update(self, n: int = 1) -> None:
        del n

    def set_postfix_str(self, s: str = "", refresh: bool = True) -> None:
        del s, refresh

    def close(self) -> None:
        return None


def _supports_live_progress_output() -> bool:
    for stream_name in ("stderr", "stdout"):
        stream = getattr(sys, stream_name, None)
        isatty = getattr(stream, "isatty", None)
        if not callable(isatty):
            continue
        try:
            if bool(isatty()):
                return True
        except Exception:
            continue
    return False


def create_progress_bar(
    *,
    total: Optional[int],
    desc: str,
    runtime: Optional[DistributedRuntime] = None,
    unit: str = "item",
    leave: bool = True,
    position: Optional[int] = None,
):
    runtime = runtime or distributed_runtime_from_env()
    resolved_total = None if total is None or int(total) <= 0 else int(total)
    resolved_position = position
    if resolved_position is None and runtime.is_distributed:
        resolved_position = max(0, int(runtime.local_rank))
    if not _supports_live_progress_output():
        return _NullProgressBar()
    try:
        from tqdm.auto import tqdm
    except Exception:
        return _NullProgressBar()
    return tqdm(
        total=resolved_total,
        desc=str(desc),
        unit=str(unit),
        leave=bool(leave),
        dynamic_ncols=True,
        mininterval=0.1,
        position=resolved_position,
    )


def init_torch_distributed(runtime: Optional[DistributedRuntime] = None) -> bool:
    runtime = runtime or distributed_runtime_from_env()
    if not runtime.is_distributed:
        return False
    try:
        import torch
    except Exception:
        return False
    if not torch.distributed.is_available():
        return False
    if torch.distributed.is_initialized():
        return True
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(int(runtime.local_rank))
    torch.distributed.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(seconds=DEFAULT_PROCESS_GROUP_TIMEOUT_SEC),
    )
    return True


def distributed_barrier(runtime: Optional[DistributedRuntime] = None) -> None:
    runtime = runtime or distributed_runtime_from_env()
    if not runtime.is_distributed:
        return
    try:
        import torch
    except Exception:
        return
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.cuda.is_available():
            torch.distributed.barrier(device_ids=[int(runtime.local_rank)])
        else:
            torch.distributed.barrier()
