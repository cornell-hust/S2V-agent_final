"""Common v3 utilities shared by training and inference entrypoints."""

from saver_v3.common.checkpoint_manifest import (
    CHECKPOINT_MANIFEST_FILENAME,
    CheckpointManifest,
    load_checkpoint_manifest,
    resolve_latest_checkpoint_manifest,
    write_checkpoint_manifest,
)
from saver_v3.common.fa3_guard import (
    AttentionBackendDecision,
    AttentionBackendUnavailableError,
    ensure_fa3_training_ready,
    resolve_attention_backend,
)
from saver_v3.common.runtime_env import (
    DistributedRuntime,
    RuntimeEnv,
    distributed_runtime_from_env,
    resolve_runtime_env,
    runtime_log,
)

__all__ = [
    "AttentionBackendDecision",
    "AttentionBackendUnavailableError",
    "CHECKPOINT_MANIFEST_FILENAME",
    "CheckpointManifest",
    "DistributedRuntime",
    "RuntimeEnv",
    "distributed_runtime_from_env",
    "ensure_fa3_training_ready",
    "load_checkpoint_manifest",
    "resolve_attention_backend",
    "resolve_latest_checkpoint_manifest",
    "resolve_runtime_env",
    "runtime_log",
    "write_checkpoint_manifest",
]
