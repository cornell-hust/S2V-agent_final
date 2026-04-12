from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

from batch_run_saver_rollout import main as batch_run_saver_rollout_main
from saver_v3.cli.common import load_yaml_mapping


def _mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return value or {}


@dataclass
class PolicyRolloutConfig:
    base_model: str
    data_path: str
    output_path: str
    data_root: str = ""
    include_splits: str = ""
    indices: str = ""
    start_index: int = 0
    count: int = 0
    max_turns: int = 14
    rollout_batch_size: int = 4
    progress_every: int = 10
    proposal_model_path: str = ""
    proposal_torch_dtype: str = "auto"
    proposal_device: str = ""
    torch_dtype: str = "auto"
    device_map: str = "auto"
    attn_implementation: str = ""
    vllm_mode: str = "colocate"
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.9
    vllm_guided_decoding_regex: str = ""
    vllm_server_host: str = "127.0.0.1"
    vllm_server_port: int = 8000
    vllm_server_timeout: float = 240.0
    max_new_tokens: int = 512
    max_total_images: int = 28
    max_seq_length: int = 8192
    keep_recent_text_messages: int = 20
    keep_recent_tool_image_messages: int = 0
    max_image_side: int = 640
    max_image_pixels: int = 0
    num_preview_frames: int = 8
    preview_sampling_fps: float | None = None
    log_dir: str = ""

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "PolicyRolloutConfig":
        server = dict(_mapping(mapping.get("server")))
        client = dict(_mapping(mapping.get("client")))
        io_cfg = dict(_mapping(mapping.get("io")))
        rollout_cfg = dict(_mapping(mapping.get("rollout")))
        data_path = str(io_cfg.get("data_path") or "").strip()
        if not data_path:
            raise ValueError(
                "run_policy_rollout_vllm now requires io.data_path pointing to raw SAVER JSONL. "
                "The legacy io.input_manifest compact-trace predictor path has been retired."
            )
        return cls(
            base_model=str(mapping.get("base_model") or "").strip(),
            data_path=data_path,
            output_path=str(io_cfg.get("output_path") or "").strip(),
            data_root=str(io_cfg.get("data_root") or "").strip(),
            include_splits=str(io_cfg.get("include_splits") or "").strip(),
            indices=str(io_cfg.get("indices") or "").strip(),
            start_index=int(io_cfg.get("start_index", 0) or 0),
            count=int(io_cfg.get("count", io_cfg.get("max_records", 0)) or 0),
            max_turns=int(rollout_cfg.get("max_turns", 14) or 14),
            rollout_batch_size=int(rollout_cfg.get("rollout_batch_size", 4) or 4),
            progress_every=int(rollout_cfg.get("progress_every", 10) or 10),
            proposal_model_path=str(rollout_cfg.get("proposal_model_path") or "").strip(),
            proposal_torch_dtype=str(rollout_cfg.get("proposal_torch_dtype") or "auto"),
            proposal_device=str(rollout_cfg.get("proposal_device") or "").strip(),
            torch_dtype=str(mapping.get("torch_dtype") or server.get("dtype") or "auto"),
            device_map=str(mapping.get("device_map") or "auto"),
            attn_implementation=str(mapping.get("attn_implementation") or "").strip(),
            vllm_mode=str(server.get("mode") or "colocate"),
            vllm_tensor_parallel_size=int(server.get("tensor_parallel_size", 1) or 1),
            vllm_gpu_memory_utilization=float(server.get("gpu_memory_utilization", 0.9) or 0.9),
            vllm_guided_decoding_regex=str(client.get("guided_decoding_regex") or "").strip(),
            vllm_server_host=str(server.get("host") or "127.0.0.1"),
            vllm_server_port=int(server.get("port", 8000) or 8000),
            vllm_server_timeout=float(server.get("timeout_sec", 240.0) or 240.0),
            max_new_tokens=int(client.get("max_tokens", 512) or 512),
            max_total_images=int(client.get("max_total_images", 28) or 28),
            max_seq_length=int(client.get("max_seq_length", 8192) or 8192),
            keep_recent_text_messages=int(client.get("keep_recent_text_messages", 20) or 20),
            keep_recent_tool_image_messages=int(client.get("keep_recent_tool_image_messages", 0) or 0),
            max_image_side=int(client.get("max_image_side", 640) or 640),
            max_image_pixels=int(client.get("max_image_pixels", 0) or 0),
            num_preview_frames=int(client.get("num_preview_frames", 8) or 8),
            preview_sampling_fps=(
                None if client.get("preview_sampling_fps") is None else float(client.get("preview_sampling_fps") or 0.0)
            ),
            log_dir=str(io_cfg.get("log_dir") or "").strip(),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "PolicyRolloutConfig":
        return cls.from_mapping(load_yaml_mapping(path))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _append_flag(argv: list[str], flag: str, value: Any) -> None:
    text = str(value or "").strip()
    if text:
        argv.extend([flag, text])


def run_policy_rollout_job(config: PolicyRolloutConfig) -> Dict[str, Any]:
    argv: list[str] = [
        "--data", config.data_path,
        "--output", config.output_path,
        "--policy-backend", "qwen",
        "--model-path", config.base_model,
        "--torch-dtype", config.torch_dtype,
        "--device-map", config.device_map,
        "--use-vllm", "true",
        "--vllm-mode", config.vllm_mode,
        "--vllm-tensor-parallel-size", str(config.vllm_tensor_parallel_size),
        "--vllm-gpu-memory-utilization", str(config.vllm_gpu_memory_utilization),
        "--vllm-server-host", config.vllm_server_host,
        "--vllm-server-port", str(config.vllm_server_port),
        "--vllm-server-timeout", str(config.vllm_server_timeout),
        "--max-new-tokens", str(config.max_new_tokens),
        "--max-total-images", str(config.max_total_images),
        "--max-seq-length", str(config.max_seq_length),
        "--keep-recent-text-messages", str(config.keep_recent_text_messages),
        "--keep-recent-tool-image-messages", str(config.keep_recent_tool_image_messages),
        "--max-image-side", str(config.max_image_side),
        "--max-image-pixels", str(config.max_image_pixels),
        "--num-preview-frames", str(config.num_preview_frames),
        "--max-turns", str(config.max_turns),
        "--rollout-batch-size", str(config.rollout_batch_size),
        "--progress-every", str(config.progress_every),
    ]
    _append_flag(argv, "--data-root", config.data_root)
    _append_flag(argv, "--include-splits", config.include_splits)
    _append_flag(argv, "--indices", config.indices)
    if not config.indices:
        argv.extend(["--start-index", str(config.start_index), "--count", str(config.count)])
    _append_flag(argv, "--proposal-model-path", config.proposal_model_path)
    _append_flag(argv, "--proposal-torch-dtype", config.proposal_torch_dtype)
    _append_flag(argv, "--proposal-device", config.proposal_device)
    _append_flag(argv, "--attn-implementation", config.attn_implementation)
    _append_flag(argv, "--vllm-guided-decoding-regex", config.vllm_guided_decoding_regex)
    if config.preview_sampling_fps is not None:
        argv.extend(["--preview-sampling-fps", str(config.preview_sampling_fps)])
    _append_flag(argv, "--log-dir", config.log_dir)
    return dict(batch_run_saver_rollout_main(argv) or {})
