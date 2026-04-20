from __future__ import annotations

import copy
import gc
import json
import os
import inspect
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import torch

from saver_v3.data.config import DEFAULT_POLICY_MAX_NEW_TOKENS
from saver_v3.model.qwen_policy import (
    QwenGenerationPolicy,
    _STRUCTURED_STOP_STRINGS,
    _trim_to_first_structured_block,
    load_generation_processor_for_checkpoint,
)
from saver_v3.common.runtime import (
    distributed_runtime_from_env,
    init_torch_distributed,
    resolve_inference_device_map,
    runtime_log,
)
from saver_v3.rl.trl_compat import patch_vllm_guided_decoding_params
from saver_v3.model.vllm_transport import encode_transport_payload


def _build_rl_completion_mask(
    completion_ids: torch.Tensor,
    *,
    eos_token_id: int,
) -> torch.Tensor:
    if completion_ids.ndim != 2:
        raise ValueError("completion_ids must be rank-2 for RL token trace construction.")
    is_eos = completion_ids == int(eos_token_id)
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=completion_ids.device)
    eos_rows = is_eos.any(dim=1)
    eos_idx[eos_rows] = is_eos.int().argmax(dim=1)[eos_rows]
    sequence_indices = torch.arange(is_eos.size(1), device=completion_ids.device).expand(is_eos.size(0), -1)
    return (sequence_indices <= eos_idx.unsqueeze(1)).to(dtype=torch.long)


def _tokenize_trimmed_completion_text(
    processor: Any,
    text: str,
) -> torch.Tensor:
    tokenizer = getattr(processor, "tokenizer", processor)
    token_ids = list(tokenizer.encode(str(text or ""), add_special_tokens=False) or [])
    return torch.tensor([token_ids], dtype=torch.long)


def _build_limit_mm_per_prompt(args: Any) -> Dict[str, int]:
    limits: Dict[str, int] = {
        "video": 8,
    }
    max_total_images = int(getattr(args, "max_total_images", 0) or 0)
    if max_total_images > 0:
        limits["image"] = int(max_total_images)
    return limits


def _resolve_rollout_episode_batch_size(args: Any) -> int:
    explicit_batch_size = max(
        int(getattr(args, "rollout_batch_size", 0) or 0),
        int(getattr(args, "eval_rollout_batch_size", 0) or 0),
        int(getattr(args, "rollout_episode_batch_size", 0) or 0),
    )
    if explicit_batch_size > 0:
        return explicit_batch_size
    return max(
        1,
        int(getattr(args, "per_device_train_batch_size", 1) or 1)
        * max(1, int(getattr(args, "rl_steps_per_generation", 1) or 1))
        * max(1, int(getattr(args, "num_generations", 1) or 1)),
    )


def _resolve_vllm_max_model_len(args: Any) -> int:
    return max(
        4096,
        int(getattr(args, "max_seq_length", 0) or 0)
        + int(
            getattr(
                args,
                "policy_max_new_tokens",
                getattr(args, "max_new_tokens", 0),
            )
            or 0
        ),
    )


def _resolve_vllm_prompt_token_budget(args: Any, *, max_tokens: int) -> int:
    max_model_len = _resolve_vllm_max_model_len(args)
    prompt_budget = max_model_len - int(max_tokens)
    configured_prompt_budget = int(getattr(args, "max_seq_length", 0) or 0)
    if configured_prompt_budget > 0:
        prompt_budget = min(prompt_budget, configured_prompt_budget)
    return max(0, int(prompt_budget))


def build_vllm_runtime_settings(args: Any) -> Dict[str, Any]:
    return {
        "use_vllm": bool(getattr(args, "use_vllm", True)),
        "vllm_mode": str(getattr(args, "vllm_mode", "colocate") or "colocate").strip().lower(),
        "vllm_tensor_parallel_size": max(1, int(getattr(args, "vllm_tensor_parallel_size", 1) or 1)),
        "vllm_gpu_memory_utilization": float(getattr(args, "vllm_gpu_memory_utilization", 0.35) or 0.35),
        "vllm_guided_decoding_regex": str(getattr(args, "vllm_guided_decoding_regex", "") or ""),
        "vllm_server_host": str(getattr(args, "vllm_server_host", "127.0.0.1") or "127.0.0.1"),
        "vllm_server_port": int(getattr(args, "vllm_server_port", 8000) or 8000),
        "vllm_server_timeout": float(getattr(args, "vllm_server_timeout", 240.0) or 240.0),
        "vllm_server_auto_launch": bool(getattr(args, "vllm_server_auto_launch", False)),
        "vllm_server_per_rank": bool(getattr(args, "vllm_server_per_rank", False)),
        "vllm_server_max_lora_rank": max(1, int(getattr(args, "vllm_server_max_lora_rank", 64) or 64)),
    }


def _resolve_vllm_base_model_path(model_path: str | Path) -> str:
    resolved_model_path = Path(model_path)
    adapter_config_path = resolved_model_path / "adapter_config.json"
    if not adapter_config_path.exists():
        return str(resolved_model_path)
    try:
        payload = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    except Exception:
        return str(resolved_model_path)
    base_model_path = str(payload.get("base_model_name_or_path") or "").strip()
    return base_model_path or str(resolved_model_path)


def build_remote_vllm_lora_request(checkpoint_path: str | Path) -> Optional[Dict[str, Any]]:
    resolved_checkpoint_path = Path(checkpoint_path)
    adapter_config_path = resolved_checkpoint_path / "adapter_config.json"
    if not adapter_config_path.exists():
        return None
    return {
        "lora_name": f"adapter-{resolved_checkpoint_path.name}",
        "lora_int_id": 1,
        "lora_path": str(resolved_checkpoint_path),
        "base_model_name": _resolve_vllm_base_model_path(resolved_checkpoint_path),
    }


def _materialize_vllm_lora_request(lora_request: Optional[Dict[str, Any]]) -> Any:
    if not lora_request:
        return None
    try:
        from vllm.lora.request import LoRARequest

        return LoRARequest(
            lora_name=str(lora_request.get("lora_name") or ""),
            lora_int_id=int(lora_request.get("lora_int_id") or 0),
            lora_path=str(lora_request.get("lora_path") or ""),
            base_model_name=str(lora_request.get("base_model_name") or ""),
        )
    except Exception:
        return dict(lora_request)


def _maybe_reset_vllm_prefix_cache(llm: Any) -> None:
    if llm is None:
        return
    reset_fn = getattr(llm, "reset_prefix_cache", None)
    if callable(reset_fn):
        try:
            reset_fn()
        except Exception:
            pass


def _args_with_overrides(args: Any, **overrides: Any) -> Any:
    payload = dict(vars(args)) if hasattr(args, "__dict__") else {}
    payload.update({key: value for key, value in overrides.items() if value is not None})
    return SimpleNamespace(**payload)


def _load_processor_or_placeholder(
    *,
    model_path: str | Path,
    processor_loader: Optional[Callable[[str | Path], Any]] = None,
) -> Any:
    resolved_processor_loader = processor_loader or load_generation_processor_for_checkpoint
    try:
        return resolved_processor_loader(model_path)
    except Exception:
        return SimpleNamespace(tokenizer=None)


def _runtime_is_distributed(runtime: Any) -> bool:
    explicit_flag = getattr(runtime, "is_distributed", None)
    if explicit_flag is not None:
        try:
            return bool(explicit_flag)
        except Exception:
            pass
    try:
        return int(getattr(runtime, "world_size", 1) or 1) > 1
    except Exception:
        return False


def _should_route_distributed_colocate_vllm_through_server(*, args: Any, runtime: Any) -> bool:
    settings = build_vllm_runtime_settings(args)
    return (
        bool(settings["use_vllm"])
        and str(settings["vllm_mode"]) == "colocate"
        and _runtime_is_distributed(runtime)
        and int(settings["vllm_tensor_parallel_size"]) <= 1
    )


def _resolve_auto_vllm_server_port(*, args: Any, runtime: Any) -> int:
    configured_port = int(getattr(args, "vllm_server_port", 8000) or 8000)
    if configured_port != 8000 or not _runtime_is_distributed(runtime):
        return configured_port
    try:
        base_port = int(str(os.environ.get("MASTER_PORT", "29500") or "29500"))
    except Exception:
        base_port = 29500
    return min(65535, max(1024, base_port + 200))


def _resolve_rank_local_vllm_server_port(*, args: Any, runtime: Any) -> int:
    base_port = _resolve_auto_vllm_server_port(args=args, runtime=runtime)
    local_rank = max(0, int(getattr(runtime, "local_rank", 0) or 0))
    return min(65535, max(1024, base_port + local_rank))


def _resolve_sft_vllm_runtime_args(*, args: Any, runtime: Any) -> Any:
    if not _should_route_distributed_colocate_vllm_through_server(args=args, runtime=runtime):
        return args
    return _args_with_overrides(
        args,
        vllm_mode="server",
        vllm_server_auto_launch=True,
        vllm_server_per_rank=True,
        vllm_server_port=_resolve_rank_local_vllm_server_port(args=args, runtime=runtime),
    )


def _is_vllm_server_healthy(host: str, port: int, *, timeout_sec: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"http://{host}:{int(port)}/health/", timeout=float(timeout_sec)) as response:
            return int(getattr(response, "status", 200) or 200) == 200
    except Exception:
        return False


def _wait_for_vllm_server_ready(
    host: str,
    port: int,
    *,
    timeout_sec: float,
    process: Optional[subprocess.Popen[Any]] = None,
    log_path: Optional[Path] = None,
) -> None:
    deadline = time.time() + max(1.0, float(timeout_sec))
    while time.time() < deadline:
        if _is_vllm_server_healthy(host, port, timeout_sec=min(2.0, float(timeout_sec))):
            return
        if process is not None and process.poll() is not None:
            detail = ""
            if log_path is not None:
                detail = f" See {log_path}."
            raise RuntimeError(
                f"vLLM server process exited before becoming healthy at http://{host}:{int(port)}/health/.{detail}"
            )
        time.sleep(1.0)
    raise TimeoutError(f"Timed out while waiting for vLLM server at http://{host}:{int(port)}/health/")


def _build_local_rank_vllm_server_env(runtime: Any) -> Dict[str, str]:
    env = os.environ.copy()
    for key in (
        "RANK",
        "WORLD_SIZE",
        "LOCAL_RANK",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "GROUP_RANK",
        "ROLE_RANK",
        "ROLE_WORLD_SIZE",
        "VLLM_DP_RANK",
        "VLLM_DP_RANK_LOCAL",
        "VLLM_DP_SIZE",
        "VLLM_DP_MASTER_PORT",
    ):
        env.pop(key, None)
    env["CUDA_VISIBLE_DEVICES"] = str(max(0, int(getattr(runtime, "local_rank", 0) or 0)))
    env["VLLM_HOST_IP"] = "127.0.0.1"
    env["VLLM_LOOPBACK_IP"] = "127.0.0.1"
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    return env


def _launch_vllm_server_subprocess(
    *,
    args: Any,
    model_path: str | Path,
    host: str,
    port: int,
    tensor_parallel_size: int,
    data_parallel_size: int,
    env: Optional[Dict[str, str]] = None,
) -> tuple[subprocess.Popen[Any], Any, Path]:
    code_dir = Path(__file__).resolve().parent.parent
    log_dir = Path(str(getattr(args, "log_dir", code_dir / "logs") or (code_dir / "logs")))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"auto_vllm_server_{int(port)}.log"
    log_handle = log_path.open("a", encoding="utf-8")
    cmd = [
        sys.executable,
        "-m",
        "saver_v3.model.vllm_server",
        "--model",
        str(model_path),
        "--host",
        str(host),
        "--port",
        str(int(port)),
        "--tensor-parallel-size",
        str(int(tensor_parallel_size)),
        "--data-parallel-size",
        str(int(data_parallel_size)),
        "--gpu-memory-utilization",
        str(float(getattr(args, "vllm_gpu_memory_utilization", 0.35) or 0.35)),
        "--dtype",
        str(getattr(args, "torch_dtype", "auto") or "auto"),
        "--max-model-len",
        str(int(_resolve_vllm_max_model_len(args))),
        "--limit-mm-per-prompt-video",
        str(int(_build_limit_mm_per_prompt(args).get("video", 8))),
        "--max-lora-rank",
        str(max(1, int(getattr(args, "vllm_server_max_lora_rank", 64) or 64))),
        "--log-level",
        "info",
        "--enable-lora",
    ]
    max_total_images = int(getattr(args, "max_total_images", 0) or 0)
    if max_total_images > 0:
        cmd.extend(["--limit-mm-per-prompt-image", str(max_total_images)])
    process = subprocess.Popen(
        cmd,
        cwd=str(code_dir),
        env=env or os.environ.copy(),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log_handle, log_path


def _should_isolate_single_rank_external_launcher(runtime: Any, *, tp_size: int) -> bool:
    return int(tp_size) <= 1 and _runtime_is_distributed(runtime)


def _resolve_single_rank_external_launcher_env(runtime: Any) -> Dict[str, str]:
    local_rank = max(0, int(getattr(runtime, "local_rank", 0) or 0))
    try:
        base_port = int(str(os.environ.get("MASTER_PORT", "29500") or "29500"))
    except Exception:
        base_port = 29500
    isolated_port = min(65535, max(1024, base_port + 100 + local_rank))
    return {
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "1",
        "GROUP_RANK": "0",
        "ROLE_RANK": "0",
        "ROLE_WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(isolated_port),
        "CUDA_VISIBLE_DEVICES": str(local_rank),
        "VLLM_HOST_IP": "127.0.0.1",
        "VLLM_LOOPBACK_IP": "127.0.0.1",
    }


@contextmanager
def _temporary_single_rank_external_launcher_env(runtime: Any, *, tp_size: int):
    if not _should_isolate_single_rank_external_launcher(runtime, tp_size=tp_size):
        yield
        return
    overrides = _resolve_single_rank_external_launcher_env(runtime)
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(int(getattr(runtime, "local_rank", 0) or 0))
    except Exception:
        pass
    try:
        os.environ.update(overrides)
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _should_fallback_eval_vllm_colocate(*, args: Any, runtime: Any) -> bool:
    settings = build_vllm_runtime_settings(args)
    return _runtime_is_distributed(runtime) and str(settings["vllm_mode"]) == "colocate"


def _is_peft_model(model: Any) -> bool:
    return bool(
        hasattr(model, "peft_config")
        and hasattr(model, "named_parameters")
        and hasattr(model, "merge_adapter")
        and hasattr(model, "unmerge_adapter")
    )


def _iter_named_weights_for_vllm(model: Any):
    merged_adapter = False
    peft_prefix = str(getattr(model, "prefix", "") or "")
    if _is_peft_model(model):
        model.merge_adapter()
        merged_adapter = True
    try:
        for name, param in model.named_parameters():
            resolved_name = str(name)
            if merged_adapter:
                resolved_name = resolved_name.removeprefix("base_model.model.").replace(".base_layer", "")
                if peft_prefix and peft_prefix in resolved_name:
                    continue
                if "original_module" in resolved_name:
                    continue
                resolved_name = resolved_name.replace("modules_to_save.default.", "")
            yield resolved_name, param.data
    finally:
        if merged_adapter:
            model.unmerge_adapter()


def _materialize_named_weights_for_collective_rpc(
    weights: List[tuple[str, Any]],
) -> List[tuple[str, Any]]:
    materialized: List[tuple[str, Any]] = []
    for name, weight in weights:
        resolved_weight = weight
        if hasattr(resolved_weight, "detach"):
            resolved_weight = resolved_weight.detach()
        if hasattr(resolved_weight, "device") and hasattr(resolved_weight, "cpu"):
            if str(getattr(getattr(resolved_weight, "device", None), "type", "")) != "cpu":
                resolved_weight = resolved_weight.cpu()
        if hasattr(resolved_weight, "contiguous"):
            resolved_weight = resolved_weight.contiguous()
        materialized.append((str(name), resolved_weight))
    return materialized


def _request_input_signature(request_input: Dict[str, Any]) -> str:
    try:
        return str(encode_transport_payload(request_input))
    except Exception:
        return json.dumps(request_input, ensure_ascii=False, sort_keys=True, default=str)


def _group_duplicate_request_inputs(
    request_inputs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    group_index_by_signature: Dict[str, int] = {}
    for request_index, request_input in enumerate(list(request_inputs or [])):
        signature = _request_input_signature(request_input)
        existing_group_index = group_index_by_signature.get(signature)
        if existing_group_index is None:
            group_index_by_signature[signature] = len(groups)
            groups.append(
                {
                    "signature": signature,
                    "request_input": request_input,
                    "request_indices": [int(request_index)],
                }
            )
            continue
        groups[existing_group_index]["request_indices"].append(int(request_index))
    return groups


def _flatten_prompt_major_completion_ids(outputs: List[Any]) -> List[List[int]]:
    completion_ids: List[List[int]] = []
    for output in list(outputs or []):
        for candidate in list(getattr(output, "outputs", []) or []):
            completion_ids.append(list(candidate.token_ids))
    return completion_ids


def _expand_grouped_completion_ids(
    *,
    request_groups: List[Dict[str, Any]],
    grouped_completion_ids: List[List[List[int]]],
    total_request_count: int,
) -> List[List[int]]:
    expanded_completion_ids: List[Optional[List[int]]] = [None] * int(total_request_count)
    if len(grouped_completion_ids) != len(request_groups):
        raise ValueError(
            "Grouped completion ids must align with grouped request inputs: "
            f"{len(grouped_completion_ids)} vs {len(request_groups)}"
        )
    for request_group, completions in zip(request_groups, grouped_completion_ids):
        request_indices = list(request_group.get("request_indices") or [])
        if len(completions) != len(request_indices):
            raise ValueError(
                "Prompt-group completion count does not match duplicated request count: "
                f"{len(completions)} vs {len(request_indices)}"
            )
        for request_index, completion_ids in zip(request_indices, completions):
            expanded_completion_ids[int(request_index)] = list(completion_ids)
    missing = [idx for idx, value in enumerate(expanded_completion_ids) if value is None]
    if missing:
        raise ValueError(f"Missing expanded completion ids for request indices: {missing}")
    return [list(value or []) for value in expanded_completion_ids]


def _vllm_worker_reload_supports_weights_iterator() -> bool:
    try:
        from vllm.v1.worker.gpu_worker import GPUWorker
    except Exception:
        return False
    try:
        signature = inspect.signature(GPUWorker.reload_weights)
    except Exception:
        return False
    return "weights_iterator" in signature.parameters


def _build_vllm_guided_sampling_kwargs(guided_decoding_regex: str) -> Dict[str, Any]:
    regex = str(guided_decoding_regex or "").strip()
    if not regex:
        return {}

    patch_vllm_guided_decoding_params()
    from vllm import SamplingParams

    try:
        sampling_signature = inspect.signature(SamplingParams)
    except Exception:
        sampling_signature = None
    parameter_names = (
        set(sampling_signature.parameters.keys())
        if sampling_signature is not None
        else set()
    )

    if "guided_decoding" in parameter_names:
        from vllm.sampling_params import GuidedDecodingParams

        return {
            "guided_decoding": GuidedDecodingParams(regex=regex),
        }

    if "structured_outputs" in parameter_names:
        from vllm.sampling_params import StructuredOutputsParams

        return {
            "structured_outputs": StructuredOutputsParams(regex=regex),
        }

    if "guided_decoding_regex" in parameter_names:
        return {"guided_decoding_regex": regex}

    if "regex" in parameter_names:
        return {"regex": regex}

    raise TypeError(
        "Installed vLLM SamplingParams does not expose a supported guided decoding field."
    )


def _build_vllm_structured_stop_kwargs(
    *,
    parameter_names: Optional[set[str]] = None,
) -> Dict[str, Any]:
    if parameter_names is None:
        try:
            from vllm import SamplingParams

            parameter_names = set(inspect.signature(SamplingParams).parameters.keys())
        except Exception:
            return {}
    else:
        parameter_names = set(parameter_names)

    sampling_kwargs: Dict[str, Any] = {}
    if "stop" in parameter_names:
        sampling_kwargs["stop"] = list(_STRUCTURED_STOP_STRINGS)
    if "include_stop_str_in_output" in parameter_names:
        sampling_kwargs["include_stop_str_in_output"] = True
    return sampling_kwargs


class _SAVERVLLMClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        server_port: int = 8000,
        connection_timeout: float = 0.0,
        group_port: Optional[int] = None,
    ) -> None:
        try:
            from trl.extras.vllm_client import VLLMClient
        except Exception as exc:
            raise ImportError("vLLM server mode requires TRL's VLLMClient to be importable.") from exc
        resolved_group_port = int(group_port or min(65535, int(server_port) + 1000))
        self._client = VLLMClient(
            host=host,
            server_port=server_port,
            group_port=resolved_group_port,
            connection_timeout=connection_timeout,
        )
        self.base_url = self._client.base_url
        self.session = self._client.session
        self.host = self._client.host
        self.group_port = resolved_group_port
        self.rank: Optional[int] = None
        self.pynccl_comm: Any = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def init_communicator(self, *, device_index: int = 0) -> None:
        import requests
        from trl.extras.vllm_client import PyNcclCommunicator, StatelessProcessGroup

        response = requests.get(f"{self.base_url}/get_world_size/")
        if response.status_code != 200:
            raise RuntimeError(f"Request failed: {response.status_code}, {response.text}")
        vllm_world_size = int(response.json()["world_size"])
        world_size = vllm_world_size + 1
        self.rank = vllm_world_size
        response = self.session.post(
            f"{self.base_url}/init_communicator/",
            json={"host": "0.0.0.0", "port": int(self.group_port), "world_size": int(world_size)},
        )
        if response.status_code != 200:
            raise RuntimeError(f"Request failed: {response.status_code}, {response.text}")
        time.sleep(0.1)
        pg = StatelessProcessGroup.create(
            host=self.host,
            port=int(self.group_port),
            rank=int(self.rank),
            world_size=int(world_size),
        )
        self.pynccl_comm = PyNcclCommunicator(pg, device=int(device_index))

    def update_named_param(self, name: str, weights: torch.Tensor) -> None:
        if self.pynccl_comm is None or self.rank is None:
            raise RuntimeError("Communicator not initialized. Call init_communicator first.")
        payload = {"name": name, "dtype": str(weights.dtype), "shape": tuple(weights.shape)}
        response = self.session.post(f"{self.base_url}/update_named_param/", json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Request failed: {response.status_code}, {response.text}")
        self.pynccl_comm.broadcast(weights, src=int(self.rank))
        self.pynccl_comm.group.barrier()

    def close_communicator(self) -> None:
        import requests

        try:
            response = self.session.post(f"{self.base_url}/close_communicator/")
        except requests.ConnectionError:
            response = None
        if response is not None and response.status_code != 200:
            raise RuntimeError(f"Request failed: {response.status_code}, {response.text}")
        self.pynccl_comm = None
        self.rank = None

    def generate_multimodal(
        self,
        *,
        llm_inputs: List[Dict[str, Any]],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
        stop: Optional[List[str]] = None,
        include_stop_str_in_output: bool = False,
        lora_request: Optional[Dict[str, Any]] = None,
    ) -> List[List[int]]:
        payload: Dict[str, Any] = {
            "llm_inputs_b64": encode_transport_payload(llm_inputs),
            "n": int(n),
            "repetition_penalty": float(repetition_penalty),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "min_p": float(min_p),
            "max_tokens": int(max_tokens),
            "guided_decoding_regex": guided_decoding_regex,
            "stop": list(stop or []),
            "include_stop_str_in_output": bool(include_stop_str_in_output),
        }
        if lora_request:
            payload.update(
                {
                    "lora_name": str(lora_request.get("lora_name") or ""),
                    "lora_int_id": int(lora_request.get("lora_int_id") or 0),
                    "lora_path": str(lora_request.get("lora_path") or ""),
                    "base_model_name": str(lora_request.get("base_model_name") or ""),
                }
            )
        response = self.session.post(
            f"{self.base_url}/generate_multimodal/",
            json=payload,
        )
        if response.status_code == 200:
            return response.json()["completion_ids"]
        raise RuntimeError(f"vLLM multimodal request failed: {response.status_code} {response.text}")


class _VllmColocateRuntime:
    def __init__(
        self,
        *,
        args: Any,
        runtime: Any,
        model_path: str | Path,
    ) -> None:
        self.args = args
        self.runtime = runtime or distributed_runtime_from_env()
        self.settings = build_vllm_runtime_settings(args)
        self.enabled = bool(self.settings["use_vllm"])
        self.base_model_path = _resolve_vllm_base_model_path(model_path)
        self.llm: Any = None
        self.mode = "colocate"
        self._last_loaded_step: Optional[int] = None
        self.tp_group: Any = None
        self.tp_group_ranks: List[int] = [int(getattr(self.runtime, "rank", 0) or 0)]
        self.local_rank_in_tp_group: int = 0

        if not self.enabled:
            return
        if self.settings["vllm_mode"] != "colocate":
            raise ValueError(f"Unsupported vLLM mode: {self.settings['vllm_mode']}")
        self._initialize_tensor_parallel_group()
        self.llm = self._build_llm()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _initialize_tensor_parallel_group(self) -> None:
        tp_size = int(self.settings["vllm_tensor_parallel_size"])
        if tp_size <= 1:
            return
        world_size = int(getattr(self.runtime, "world_size", 1) or 1)
        rank = int(getattr(self.runtime, "rank", 0) or 0)
        if world_size % tp_size != 0:
            raise ValueError(
                f"vllm_tensor_parallel_size ({tp_size}) must divide world size ({world_size}) evenly."
            )
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            raise RuntimeError(
                "vLLM tensor parallel colocation requires torch.distributed to be initialized before runtime setup."
            )
        groups = [
            list(range(group_start, group_start + tp_size))
            for group_start in range(0, world_size, tp_size)
        ]
        self.tp_group, _ = torch.distributed.new_subgroups_by_enumeration(groups)
        group_index = rank // tp_size
        self.tp_group_ranks = list(groups[group_index])
        self.local_rank_in_tp_group = int(self.tp_group_ranks.index(rank))

    def _build_llm(self) -> Any:
        try:
            patch_vllm_guided_decoding_params()
            from vllm import LLM
        except Exception as exc:
            raise ImportError(
                "vLLM-backed rollout route requires `vllm` to be importable in the current environment."
            ) from exc

        max_num_seqs = max(1, int(getattr(self.args, "vllm_max_num_seqs", 0) or 0))
        if max_num_seqs <= 0:
            max_num_seqs = _resolve_rollout_episode_batch_size(self.args)
        tp_size = max(1, int(self.settings["vllm_tensor_parallel_size"]))
        max_model_len = _resolve_vllm_max_model_len(self.args)
        llm_kwargs: Dict[str, Any] = {
            "model": self.base_model_path,
            "tensor_parallel_size": tp_size,
            "gpu_memory_utilization": float(self.settings["vllm_gpu_memory_utilization"]),
            "max_num_seqs": int(max_num_seqs * tp_size),
            "max_model_len": int(max_model_len),
            "seed": int(getattr(self.args, "seed", 0) or 0)
            + (int(getattr(self.runtime, "rank", 0) or 0) // tp_size),
            "limit_mm_per_prompt": _build_limit_mm_per_prompt(self.args),
        }
        if bool(getattr(self.runtime, "is_distributed", False)) or tp_size > 1:
            llm_kwargs["distributed_executor_backend"] = "external_launcher"
        torch_dtype = str(getattr(self.args, "torch_dtype", "auto") or "auto").strip()
        if torch_dtype and torch_dtype != "auto":
            llm_kwargs["dtype"] = torch_dtype
        disable_v1_multiprocessing = tp_size <= 1
        previous_v1_multiprocessing = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
        if disable_v1_multiprocessing:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        try:
            runtime_log(
                "using colocated vLLM runtime"
                + f": local_rank={int(getattr(self.runtime, 'local_rank', 0) or 0)} tp_size={tp_size}",
                runtime=self.runtime,
                main_process_only=True,
            )
            with _temporary_single_rank_external_launcher_env(self.runtime, tp_size=tp_size):
                return LLM(**llm_kwargs)
        finally:
            if disable_v1_multiprocessing:
                if previous_v1_multiprocessing is None:
                    os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
                else:
                    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = previous_v1_multiprocessing

    def _driver_model(self) -> Any:
        if self.llm is None:
            raise RuntimeError("vLLM runtime is not initialized.")
        llm_engine = getattr(self.llm, "llm_engine", None)
        direct_model_executor = getattr(llm_engine, "model_executor", None)
        if direct_model_executor is not None:
            driver_worker = getattr(direct_model_executor, "driver_worker", None)
            model_runner = getattr(driver_worker, "model_runner", None)
            model = getattr(model_runner, "model", None)
            if model is not None:
                return model
        engine_core_client = getattr(llm_engine, "engine_core", None)
        inproc_engine_core = getattr(engine_core_client, "engine_core", None)
        indirect_model_executor = getattr(inproc_engine_core, "model_executor", None)
        if indirect_model_executor is not None:
            driver_worker = getattr(indirect_model_executor, "driver_worker", None)
            model_runner = getattr(driver_worker, "model_runner", None)
            model = getattr(model_runner, "model", None)
            if model is not None:
                return model
        raise AttributeError("Unable to resolve a local vLLM driver model for direct weight loading.")

    def _reload_weights(self, source_model: Any) -> None:
        if self.llm is None:
            raise RuntimeError("vLLM runtime is not initialized.")
        weights = list(_iter_named_weights_for_vllm(source_model))
        try:
            llm_model = self._driver_model()
        except AttributeError:
            llm_model = None
        if llm_model is not None:
            llm_model.load_weights(iter(weights))
            return
        collective_rpc = getattr(self.llm, "collective_rpc", None)
        if callable(collective_rpc):
            if not _vllm_worker_reload_supports_weights_iterator():
                raise RuntimeError(
                    "This vLLM runtime does not support in-memory weight reload via collective_rpc. "
                    "Use single-GPU in-process colocate runtime or upgrade vLLM."
                )
            collective_rpc(
                "reload_weights",
                kwargs={
                    "weights_iterator": _materialize_named_weights_for_collective_rpc(weights),
                    "is_checkpoint_format": True,
                },
            )
            return
        raise AttributeError("Unable to resolve a vLLM weight reload path for the current runtime.")

    def ensure_weights_synced(self, source_model: Any, *, global_step: int) -> None:
        if not self.enabled or self.llm is None:
            return
        step_value = int(global_step)
        if self._last_loaded_step == step_value:
            return
        self._reload_weights(source_model)
        self.reset_prefix_cache()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self._last_loaded_step = step_value

    def build_sampling_params(
        self,
        *,
        max_tokens: int,
        do_sample: bool,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        repetition_penalty: Optional[float],
        guided_decoding_regex: str = "",
        num_completions: int = 1,
        seed: Optional[int] = None,
    ) -> Any:
        patch_vllm_guided_decoding_params()
        from vllm import SamplingParams

        sampling_kwargs: Dict[str, Any] = {
            "n": max(1, int(num_completions)),
            "repetition_penalty": float(repetition_penalty) if repetition_penalty is not None else 1.0,
            "temperature": float(temperature) if do_sample and temperature is not None else (1.0 if do_sample else 0.0),
            "top_p": float(top_p) if do_sample and top_p is not None else 1.0,
            "top_k": int(top_k) if do_sample and top_k is not None else -1,
            "min_p": 0.0,
            "max_tokens": int(max_tokens),
        }
        if seed is not None:
            sampling_kwargs["seed"] = int(seed)
        sampling_kwargs.update(_build_vllm_structured_stop_kwargs())
        sampling_kwargs.update(_build_vllm_guided_sampling_kwargs(guided_decoding_regex))
        return SamplingParams(**sampling_kwargs)

    def reset_prefix_cache(self) -> None:
        _maybe_reset_vllm_prefix_cache(self.llm)

    def close(self) -> None:
        llm = self.llm
        self.llm = None
        self._last_loaded_step = None
        self.tp_group = None
        self.tp_group_ranks = [int(getattr(self.runtime, "rank", 0) or 0)]
        self.local_rank_in_tp_group = 0
        if llm is not None:
            shutdown_fn = getattr(llm, "shutdown", None)
            if callable(shutdown_fn):
                try:
                    shutdown_fn()
                except Exception:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _VllmExternalLauncherRuntime:
    def __init__(
        self,
        *,
        args: Any,
        runtime: Any,
        model_path: str | Path,
    ) -> None:
        self.args = args
        self.runtime = runtime or distributed_runtime_from_env()
        self.settings = dict(build_vllm_runtime_settings(args))
        self.settings["vllm_mode"] = "external_launcher"
        self.enabled = bool(self.settings["use_vllm"])
        self.base_model_path = str(model_path)
        self.llm: Any = None
        self.mode = "external_launcher"

        if not self.enabled:
            return
        tp_size = int(self.settings["vllm_tensor_parallel_size"])
        if tp_size != 1:
            raise ValueError(
                "Direct external-launcher SFT rollout eval currently requires vllm_tensor_parallel_size=1."
            )
        if _runtime_is_distributed(self.runtime):
            init_torch_distributed(self.runtime)
            if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
                raise RuntimeError(
                    "Direct external-launcher SFT rollout eval requires torch.distributed to be initialized under torchrun."
                )
        self.llm = self._build_llm()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _build_llm(self) -> Any:
        try:
            patch_vllm_guided_decoding_params()
            from vllm import LLM
        except Exception as exc:
            raise ImportError(
                "Direct external-launcher SFT rollout eval requires `vllm` to be importable in the current environment."
            ) from exc

        max_num_seqs = max(1, int(getattr(self.args, "vllm_max_num_seqs", 0) or 0))
        if max_num_seqs <= 0:
            max_num_seqs = _resolve_rollout_episode_batch_size(self.args)
        max_model_len = _resolve_vllm_max_model_len(self.args)
        llm_kwargs: Dict[str, Any] = {
            "model": self.base_model_path,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": float(self.settings["vllm_gpu_memory_utilization"]),
            "max_num_seqs": int(max_num_seqs),
            "max_model_len": int(max_model_len),
            "seed": int(getattr(self.args, "seed", 0) or 0) + int(getattr(self.runtime, "rank", 0) or 0),
            "limit_mm_per_prompt": _build_limit_mm_per_prompt(self.args),
            "distributed_executor_backend": "external_launcher",
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
            "enable_lora": False,
        }
        torch_dtype = str(getattr(self.args, "torch_dtype", "auto") or "auto").strip()
        if torch_dtype and torch_dtype != "auto":
            llm_kwargs["dtype"] = torch_dtype

        previous_v1_multiprocessing = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
        previous_host_ip = os.environ.get("VLLM_HOST_IP")
        previous_loopback_ip = os.environ.get("VLLM_LOOPBACK_IP")
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
        os.environ.setdefault("VLLM_LOOPBACK_IP", "127.0.0.1")
        try:
            runtime_log(
                (
                    "using per-rank external-launcher vLLM runtime"
                    f": local_rank={int(getattr(self.runtime, 'local_rank', 0) or 0)} "
                    f"world_size={int(getattr(self.runtime, 'world_size', 1) or 1)}"
                ),
                runtime=self.runtime,
                main_process_only=True,
            )
            return LLM(**llm_kwargs)
        finally:
            if previous_v1_multiprocessing is None:
                os.environ.pop("VLLM_ENABLE_V1_MULTIPROCESSING", None)
            else:
                os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = previous_v1_multiprocessing
            if previous_host_ip is None:
                os.environ.pop("VLLM_HOST_IP", None)
            else:
                os.environ["VLLM_HOST_IP"] = previous_host_ip
            if previous_loopback_ip is None:
                os.environ.pop("VLLM_LOOPBACK_IP", None)
            else:
                os.environ["VLLM_LOOPBACK_IP"] = previous_loopback_ip

    def ensure_weights_synced(self, source_model: Any, *, global_step: int) -> None:
        del source_model, global_step

    def build_sampling_params(
        self,
        *,
        max_tokens: int,
        do_sample: bool,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        repetition_penalty: Optional[float],
        guided_decoding_regex: str = "",
        num_completions: int = 1,
        seed: Optional[int] = None,
    ) -> Any:
        patch_vllm_guided_decoding_params()
        from vllm import SamplingParams

        sampling_kwargs: Dict[str, Any] = {
            "n": max(1, int(num_completions)),
            "repetition_penalty": float(repetition_penalty) if repetition_penalty is not None else 1.0,
            "temperature": float(temperature) if do_sample and temperature is not None else (1.0 if do_sample else 0.0),
            "top_p": float(top_p) if do_sample and top_p is not None else 1.0,
            "top_k": int(top_k) if do_sample and top_k is not None else -1,
            "min_p": 0.0,
            "max_tokens": int(max_tokens),
        }
        if seed is not None:
            sampling_kwargs["seed"] = int(seed)
        sampling_kwargs.update(_build_vllm_structured_stop_kwargs())
        sampling_kwargs.update(_build_vllm_guided_sampling_kwargs(guided_decoding_regex))
        return SamplingParams(**sampling_kwargs)

    def generate_completion_ids(
        self,
        llm_inputs: List[Dict[str, Any]],
        *,
        sampling_params: Any,
        lora_request: Optional[Dict[str, Any]] = None,
    ) -> List[List[int]]:
        if not self.enabled or self.llm is None:
            return []
        generation_kwargs: Dict[str, Any] = {
            "sampling_params": sampling_params,
            "use_tqdm": False,
        }
        resolved_lora_request = _materialize_vllm_lora_request(lora_request)
        if resolved_lora_request is not None:
            generation_kwargs["lora_request"] = resolved_lora_request
        outputs = self.llm.generate(llm_inputs, **generation_kwargs)
        return _flatten_prompt_major_completion_ids(list(outputs or []))

    def reset_prefix_cache(self) -> None:
        _maybe_reset_vllm_prefix_cache(self.llm)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def close(self) -> None:
        llm = self.llm
        self.llm = None
        if llm is not None:
            shutdown_fn = getattr(llm, "shutdown", None)
            if callable(shutdown_fn):
                try:
                    shutdown_fn()
                except Exception:
                    pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _VllmServerRuntime:
    def __init__(
        self,
        *,
        args: Any,
        runtime: Any,
        model_path: str | Path,
    ) -> None:
        self.args = args
        self.runtime = runtime or distributed_runtime_from_env()
        self.settings = build_vllm_runtime_settings(args)
        self.enabled = bool(self.settings["use_vllm"])
        self.base_model_path = _resolve_vllm_base_model_path(model_path)
        self.mode = "server"
        self.client: Any = None
        self._last_loaded_step: Optional[int] = None
        self._managed_server_process: Optional[subprocess.Popen[Any]] = None
        self._managed_server_log_handle: Any = None
        self._managed_server_log_path: Optional[Path] = None
        self._per_rank_local_server = bool(self.settings.get("vllm_server_per_rank", False))
        self._client_communicator_initialized = False
        if not self.enabled:
            return
        if self.settings["vllm_mode"] != "server":
            raise ValueError(f"Unsupported vLLM mode: {self.settings['vllm_mode']}")
        should_launch_local_server = bool(self.settings.get("vllm_server_auto_launch", False)) and (
            self._per_rank_local_server or bool(getattr(self.runtime, "is_main_process", True))
        )
        if should_launch_local_server:
            self._maybe_launch_managed_server()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        should_create_client = self._per_rank_local_server or bool(getattr(self.runtime, "is_main_process", True))
        if should_create_client:
            self.client = _SAVERVLLMClient(
                host=str(self.settings["vllm_server_host"]),
                server_port=int(self.settings["vllm_server_port"]),
                group_port=min(65535, int(self.settings["vllm_server_port"]) + 1000),
                connection_timeout=float(self.settings["vllm_server_timeout"]),
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _ensure_client_communicator(self) -> None:
        if self.client is None or self._client_communicator_initialized:
            return
        self.client.init_communicator(device_index=int(getattr(self.runtime, "local_rank", 0) or 0))
        self._client_communicator_initialized = True

    def _maybe_launch_managed_server(self) -> None:
        host = str(self.settings["vllm_server_host"])
        port = int(self.settings["vllm_server_port"])
        timeout_sec = float(self.settings["vllm_server_timeout"])
        if _is_vllm_server_healthy(host, port, timeout_sec=min(2.0, timeout_sec)):
            runtime_log(
                f"reusing existing local vLLM server at http://{host}:{port}",
                runtime=self.runtime,
                main_process_only=True,
            )
            return
        tp_size = max(1, int(self.settings["vllm_tensor_parallel_size"]))
        world_size = max(1, int(getattr(self.runtime, "world_size", 1) or 1))
        if world_size % tp_size != 0:
            raise ValueError(
                f"vllm_tensor_parallel_size ({tp_size}) must divide world size ({world_size}) evenly for server launch."
            )
        data_parallel_size = 1 if self._per_rank_local_server else max(1, world_size // tp_size)
        runtime_log(
            (
                "auto-launching embedded vLLM server for distributed SFT rollout eval: "
                f"host={host} port={port} tensor_parallel_size={tp_size} data_parallel_size={data_parallel_size}"
            ),
            runtime=self.runtime,
            main_process_only=True,
        )
        process, log_handle, log_path = _launch_vllm_server_subprocess(
            args=self.args,
            model_path=self.base_model_path,
            host=host,
            port=port,
            tensor_parallel_size=tp_size,
            data_parallel_size=data_parallel_size,
            env=_build_local_rank_vllm_server_env(self.runtime) if self._per_rank_local_server else None,
        )
        self._managed_server_process = process
        self._managed_server_log_handle = log_handle
        self._managed_server_log_path = log_path
        try:
            _wait_for_vllm_server_ready(
                host,
                port,
                timeout_sec=timeout_sec,
                process=process,
                log_path=log_path,
            )
        except Exception:
            self._stop_managed_server()
            raise

    def _stop_managed_server(self) -> None:
        process = self._managed_server_process
        self._managed_server_process = None
        if process is not None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                try:
                    process.kill()
                    process.wait(timeout=5)
                except Exception:
                    pass
        log_handle = self._managed_server_log_handle
        self._managed_server_log_handle = None
        self._managed_server_log_path = None
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass

    def ensure_weights_synced(self, source_model: Any, *, global_step: int) -> None:
        if not self.enabled:
            return
        step_value = int(global_step)
        if self._last_loaded_step == step_value:
            return
        if self.client is not None:
            self._ensure_client_communicator()
            for name, weights in _iter_named_weights_for_vllm(source_model):
                self.client.update_named_param(name, weights)
            self.reset_prefix_cache()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        self._last_loaded_step = step_value

    def build_sampling_params(
        self,
        *,
        max_tokens: int,
        do_sample: bool,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        repetition_penalty: Optional[float],
        guided_decoding_regex: str = "",
        num_completions: int = 1,
        seed: Optional[int] = None,
    ) -> Any:
        payload = {
            "n": max(1, int(num_completions)),
            "repetition_penalty": float(repetition_penalty) if repetition_penalty is not None else 1.0,
            "temperature": float(temperature) if do_sample and temperature is not None else (1.0 if do_sample else 0.0),
            "top_p": float(top_p) if do_sample and top_p is not None else 1.0,
            "top_k": int(top_k) if do_sample and top_k is not None else -1,
            "min_p": 0.0,
            "max_tokens": int(max_tokens),
            "guided_decoding_regex": str(guided_decoding_regex or "") or None,
            **_build_vllm_structured_stop_kwargs(),
        }
        if seed is not None:
            payload["seed"] = int(seed)
        return payload

    def generate_completion_ids(
        self,
        llm_inputs: List[Dict[str, Any]],
        *,
        sampling_params: Any,
        lora_request: Optional[Dict[str, Any]] = None,
    ) -> List[List[int]]:
        if not self.enabled:
            return []
        if self._per_rank_local_server and self.client is not None:
            return self.client.generate_multimodal(
                llm_inputs=llm_inputs,
                lora_request=lora_request,
                **dict(sampling_params or {}),
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = int(torch.distributed.get_world_size())
            rank = int(torch.distributed.get_rank())
            gathered_inputs = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered_inputs, llm_inputs)
            input_sizes = [len(batch or []) for batch in gathered_inputs]
            if self.client is not None:
                flat_inputs = [entry for batch in gathered_inputs for entry in (batch or [])]
                completion_ids = self.client.generate_multimodal(
                    llm_inputs=flat_inputs,
                    lora_request=lora_request,
                    **dict(sampling_params or {}),
                )
            else:
                completion_ids = None
            payload = [completion_ids]
            torch.distributed.broadcast_object_list(payload, src=0)
            completion_ids = list(payload[0] or [])
            start_index = sum(input_sizes[:rank])
            end_index = start_index + int(input_sizes[rank])
            return completion_ids[start_index:end_index]
        if self.client is None:
            raise RuntimeError("vLLM server runtime requires a main-process client for generation.")
        return self.client.generate_multimodal(
            llm_inputs=llm_inputs,
            lora_request=lora_request,
            **dict(sampling_params or {}),
        )

    def reset_prefix_cache(self) -> None:
        if self.client is not None:
            self.client.reset_prefix_cache()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def close(self) -> None:
        client = self.client
        self.client = None
        self._last_loaded_step = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                pass
        if client is not None:
            try:
                client.close_communicator()
            except Exception:
                pass
        if self._per_rank_local_server or bool(getattr(self.runtime, "is_main_process", True)):
            self._stop_managed_server()


def _should_use_static_external_launcher_runtime(
    *,
    args: Any,
    runtime: Any,
    prefer_direct_local_rank_runtime: bool,
) -> bool:
    if not bool(prefer_direct_local_rank_runtime):
        return False
    settings = build_vllm_runtime_settings(args)
    return (
        str(settings["vllm_mode"]) == "colocate"
        and int(settings["vllm_tensor_parallel_size"]) <= 1
        and _runtime_is_distributed(runtime)
    )


def create_vllm_runtime(
    *,
    args: Any,
    runtime: Any,
    model_path: str | Path,
    prefer_direct_local_rank_runtime: bool = False,
) -> Any:
    effective_args = args
    settings = build_vllm_runtime_settings(effective_args)
    if not bool(settings["use_vllm"]):
        return None
    if _should_use_static_external_launcher_runtime(
        args=effective_args,
        runtime=runtime,
        prefer_direct_local_rank_runtime=prefer_direct_local_rank_runtime,
    ):
        return _VllmExternalLauncherRuntime(args=effective_args, runtime=runtime, model_path=model_path)
    if settings["vllm_mode"] == "server":
        return _VllmServerRuntime(args=effective_args, runtime=runtime, model_path=model_path)
    if settings["vllm_mode"] == "colocate":
        return _VllmColocateRuntime(args=effective_args, runtime=runtime, model_path=model_path)
    raise ValueError(f"Unsupported vLLM mode: {settings['vllm_mode']}")


def create_sft_external_vllm_runtime(
    *,
    args: Any,
    runtime: Any,
    model_path: str | Path,
) -> Any:
    settings = build_vllm_runtime_settings(args)
    if not bool(settings["use_vllm"]):
        return None
    return _VllmExternalLauncherRuntime(args=args, runtime=runtime, model_path=model_path)


class VllmQwenGenerationPolicy(QwenGenerationPolicy):
    def __init__(
        self,
        *,
        runtime: Any,
        source_model: Any = None,
        step_resolver: Callable[[], int],
        guided_decoding_regex: str = "",
        remote_lora_request: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=source_model, **kwargs)
        self.vllm_runtime = runtime
        self.source_model = source_model
        self.step_resolver = step_resolver
        self.guided_decoding_regex = str(guided_decoding_regex or "")
        self.remote_lora_request = dict(remote_lora_request or {}) or None
        self.capture_rl_token_traces = False
        self._last_rl_token_traces: Optional[List[Dict[str, Any]]] = None

    def pop_last_rl_token_traces(self) -> Optional[List[Dict[str, Any]]]:
        traces = self._last_rl_token_traces
        self._last_rl_token_traces = None
        return traces

    def _resolve_prompt_token_budget(self) -> int:
        candidate_budgets: List[int] = []
        if int(self.max_seq_length) > 0:
            candidate_budgets.append(int(self.max_seq_length))
        runtime_args = getattr(self.vllm_runtime, "args", None)
        if runtime_args is not None:
            runtime_budget = _resolve_vllm_prompt_token_budget(runtime_args, max_tokens=self.max_new_tokens)
            if runtime_budget > 0:
                candidate_budgets.append(int(runtime_budget))
        if not candidate_budgets:
            return 0
        return min(candidate_budgets)

    def _build_vllm_prompt_payload(
        self,
        prepared_messages: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], str]:
        prompt_budget = self._resolve_prompt_token_budget()
        if prompt_budget <= 0:
            prompt_text = self.processor.apply_chat_template(
                prepared_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prepared_messages, prompt_text

        fitted_messages, exact_fit = self._fit_prepared_messages_to_max_length(
            prepared_messages,
            max_length=prompt_budget,
        )
        if exact_fit:
            prompt_text = self.processor.apply_chat_template(
                fitted_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = self._build_prompt_text_with_truncation(
                fitted_messages,
                max_length=prompt_budget,
                truncation_side="left",
            )
        return fitted_messages, prompt_text

    def generate_from_messages(self, messages: List[Dict[str, Any]]) -> str:
        outputs = self.generate_from_messages_batch([messages])
        return outputs[0] if outputs else ""

    def generate_from_messages_batch(
        self,
        messages_batch: List[List[Dict[str, Any]]],
    ) -> List[str]:
        if not messages_batch:
            return []
        self._last_rl_token_traces = None
        if not self.vllm_runtime.enabled:
            if self.source_model is None:
                raise RuntimeError("vLLM generation policy requires a source HF model when runtime is disabled.")
            return super().generate_from_messages_batch(messages_batch)

        current_step = int(self.step_resolver() or 0)
        runtime_args = getattr(self.vllm_runtime, "args", None)
        base_seed = int(getattr(runtime_args, "seed", 0) or 0)
        runtime_rank = int(getattr(getattr(self.vllm_runtime, "runtime", None), "rank", 0) or 0)
        if self.source_model is not None:
            self.vllm_runtime.ensure_weights_synced(self.source_model, global_step=current_step)
        if not self.use_generation_cache:
            reset_fn = getattr(self.vllm_runtime, "reset_prefix_cache", None)
            if callable(reset_fn):
                _maybe_reset_vllm_prefix_cache(self.vllm_runtime)
            else:
                _maybe_reset_vllm_prefix_cache(getattr(self.vllm_runtime, "llm", None))

        prompt_payload_batch = [
            self._build_vllm_prompt_payload(self.prepare_messages(messages))
            for messages in messages_batch
        ]
        prepared_messages_batch = [payload[0] for payload in prompt_payload_batch]
        prompt_text_batch = [payload[1] for payload in prompt_payload_batch]

        llm_inputs: List[Dict[str, Any]] = []
        prompt_input_tensors: List[Dict[str, torch.Tensor]] = []
        for prepared_messages, prompt_text in zip(prepared_messages_batch, prompt_text_batch):
            image_inputs, video_inputs = self._extract_vision_inputs(prepared_messages)
            llm_input: Dict[str, Any] = {"prompt": prompt_text}
            multimodal_data: Dict[str, Any] = {}
            if image_inputs:
                multimodal_data["image"] = image_inputs
            if video_inputs:
                multimodal_data["video"] = video_inputs
            if multimodal_data:
                llm_input["multi_modal_data"] = multimodal_data
            llm_inputs.append(llm_input)
            if self.capture_rl_token_traces:
                processor_kwargs: Dict[str, Any] = {
                    "text": prompt_text,
                    "padding": False,
                    "return_tensors": "pt",
                }
                if image_inputs:
                    processor_kwargs["images"] = image_inputs
                if video_inputs:
                    processor_kwargs["videos"] = video_inputs
                processor_inputs = self.processor(**processor_kwargs)
                prompt_input_tensors.append(
                    {
                        key: value.detach().cpu()
                        for key, value in processor_inputs.items()
                        if isinstance(value, torch.Tensor)
                    }
                )

        def _generate_completion_ids_for_inputs(
            request_inputs: List[Dict[str, Any]],
            *,
            seed: Optional[int],
            num_completions: int = 1,
        ) -> List[List[int]]:
            sampling_params = self.vllm_runtime.build_sampling_params(
                max_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                guided_decoding_regex=self.guided_decoding_regex,
                num_completions=int(num_completions),
                seed=seed,
            )
            if hasattr(self.vllm_runtime, "generate_completion_ids"):
                return self.vllm_runtime.generate_completion_ids(
                    request_inputs,
                    sampling_params=sampling_params,
                    lora_request=self.remote_lora_request,
                )
            orig_size = len(request_inputs)
            runtime_settings = getattr(self.vllm_runtime, "settings", {}) or {}
            tp_size = max(1, int(runtime_settings.get("vllm_tensor_parallel_size", 1) or 1))
            llm_inputs_local = list(request_inputs)
            if tp_size > 1 and getattr(self.vllm_runtime, "tp_group", None) is not None:
                gathered_llm_inputs = [None for _ in range(tp_size)]
                torch.distributed.all_gather_object(
                    gathered_llm_inputs,
                    llm_inputs_local,
                    group=self.vllm_runtime.tp_group,
                )
                llm_inputs_local = [entry for batch in gathered_llm_inputs for entry in batch]
            generation_kwargs: Dict[str, Any] = {
                "sampling_params": sampling_params,
                "use_tqdm": False,
            }
            if self.remote_lora_request:
                generation_kwargs["lora_request"] = dict(self.remote_lora_request)
            outputs = self.vllm_runtime.llm.generate(llm_inputs_local, **generation_kwargs)
            completion_ids = _flatten_prompt_major_completion_ids(list(outputs or []))
            if tp_size > 1 and getattr(self.vllm_runtime, "tp_group", None) is not None:
                local_rank_in_tp_group = int(getattr(self.vllm_runtime, "local_rank_in_tp_group", 0) or 0)
                span = int(orig_size) * max(1, int(num_completions))
                tp_slice = slice(local_rank_in_tp_group * span, (local_rank_in_tp_group + 1) * span)
                completion_ids = completion_ids[tp_slice]
            return completion_ids

        if self.do_sample and str(getattr(self.vllm_runtime, "mode", "") or "").strip().lower() == "colocate":
            request_groups = _group_duplicate_request_inputs(llm_inputs)
            unique_prompt_count = len(request_groups)
            if unique_prompt_count < len(llm_inputs):
                runtime_log(
                    "rl grouped sampling debug: "
                    f"batch_size={len(llm_inputs)} "
                    f"unique_prompts={unique_prompt_count} "
                    f"deduped_requests={len(llm_inputs) - unique_prompt_count} "
                    f"max_group_size={max(len(group['request_indices']) for group in request_groups)}",
                    runtime=distributed_runtime_from_env(),
                    main_process_only=True,
                )
                if hasattr(self.vllm_runtime, "generate_completion_ids"):
                    grouped_completion_ids = []
                    for request_group in request_groups:
                        grouped_completion_ids.append(
                            _generate_completion_ids_for_inputs(
                                [dict(request_group["request_input"])],
                                seed=None,
                                num_completions=len(request_group["request_indices"]),
                            )
                        )
                else:
                    unique_request_inputs = [dict(request_group["request_input"]) for request_group in request_groups]
                    sampling_params_batch = [
                        self.vllm_runtime.build_sampling_params(
                            max_tokens=self.max_new_tokens,
                            do_sample=self.do_sample,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=self.top_k,
                            repetition_penalty=self.repetition_penalty,
                            guided_decoding_regex=self.guided_decoding_regex,
                            num_completions=len(request_group["request_indices"]),
                            seed=None,
                        )
                        for request_group in request_groups
                    ]
                    unique_request_count = len(unique_request_inputs)
                    runtime_settings = getattr(self.vllm_runtime, "settings", {}) or {}
                    tp_size = max(1, int(runtime_settings.get("vllm_tensor_parallel_size", 1) or 1))
                    unique_request_inputs_local = list(unique_request_inputs)
                    if tp_size > 1 and getattr(self.vllm_runtime, "tp_group", None) is not None:
                        gathered_llm_inputs = [None for _ in range(tp_size)]
                        torch.distributed.all_gather_object(
                            gathered_llm_inputs,
                            unique_request_inputs_local,
                            group=self.vllm_runtime.tp_group,
                        )
                        unique_request_inputs_local = [entry for batch in gathered_llm_inputs for entry in batch]
                        sampling_params_batch = list(sampling_params_batch) * int(tp_size)
                    generation_kwargs: Dict[str, Any] = {
                        "sampling_params": sampling_params_batch,
                        "use_tqdm": False,
                    }
                    if self.remote_lora_request:
                        generation_kwargs["lora_request"] = dict(self.remote_lora_request)
                    outputs = self.vllm_runtime.llm.generate(unique_request_inputs_local, **generation_kwargs)
                    grouped_completion_ids = [
                        [list(candidate.token_ids) for candidate in list(getattr(output, "outputs", []) or [])]
                        for output in list(outputs or [])
                    ]
                    if tp_size > 1 and getattr(self.vllm_runtime, "tp_group", None) is not None:
                        local_rank_in_tp_group = int(getattr(self.vllm_runtime, "local_rank_in_tp_group", 0) or 0)
                        tp_slice = slice(
                            local_rank_in_tp_group * unique_request_count,
                            (local_rank_in_tp_group + 1) * unique_request_count,
                        )
                        grouped_completion_ids = grouped_completion_ids[tp_slice]
                completion_ids_batch = _expand_grouped_completion_ids(
                    request_groups=request_groups,
                    grouped_completion_ids=grouped_completion_ids,
                    total_request_count=len(llm_inputs),
                )
            else:
                completion_ids_batch = _generate_completion_ids_for_inputs(
                    llm_inputs,
                    seed=None,
                    num_completions=1,
                )
        elif self.do_sample:
            completion_ids_batch = _generate_completion_ids_for_inputs(llm_inputs, seed=None)
        else:
            completion_ids_batch = _generate_completion_ids_for_inputs(llm_inputs, seed=None)
        raw_output_texts = self.processor.batch_decode(
            completion_ids_batch,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        output_texts = [_trim_to_first_structured_block(output_text) for output_text in raw_output_texts]
        if self.capture_rl_token_traces:
            eos_token_id = int(
                getattr(getattr(self.processor, "tokenizer", self.processor), "eos_token_id", 0)
                or getattr(getattr(self.processor, "tokenizer", self.processor), "pad_token_id", 0)
                or 0
            )
            token_traces: List[Dict[str, Any]] = []
            for prepared_messages, prompt_text, output_text, prompt_inputs in zip(
                prepared_messages_batch,
                prompt_text_batch,
                output_texts,
                prompt_input_tensors,
            ):
                completion_tensor = _tokenize_trimmed_completion_text(self.processor, output_text)
                completion_mask = _build_rl_completion_mask(
                    completion_tensor,
                    eos_token_id=eos_token_id,
                )
                trace_spec: Dict[str, Any] = {
                    "prompt_trace": {
                        "prompt_ids": prompt_inputs["input_ids"].detach().cpu(),
                        "prompt_mask": prompt_inputs["attention_mask"].detach().cpu(),
                        "multimodal_inputs": {},
                    },
                    "completion_ids": completion_tensor.detach().cpu(),
                    "completion_mask": completion_mask.detach().cpu(),
                }
                for key, value in prompt_inputs.items():
                    if key in {"input_ids", "attention_mask"}:
                        continue
                    if isinstance(value, torch.Tensor):
                        trace_spec["prompt_trace"]["multimodal_inputs"][key] = value.detach().cpu()
                token_traces.append(trace_spec)
            self._last_rl_token_traces = token_traces
        return output_texts


def _maybe_load_adapter_source_model(
    *,
    model_path: str | Path,
    args: Any,
    runtime: Any,
    processor_loader: Optional[Callable[[str | Path], Any]] = None,
    hf_policy_class: Optional[type[QwenGenerationPolicy]] = None,
    max_new_tokens: int,
    max_total_images: int,
    max_seq_length: int,
    keep_recent_tool_image_messages: int,
    keep_recent_text_messages: int,
    max_image_side: int,
    max_image_pixels: int,
    do_sample: bool,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    repetition_penalty: Optional[float],
) -> tuple[Any, Any]:
    resolved_processor_loader = processor_loader or load_generation_processor_for_checkpoint
    resolved_hf_policy_class = hf_policy_class or QwenGenerationPolicy
    processor = resolved_processor_loader(model_path)
    if not (Path(model_path) / "adapter_config.json").exists():
        return None, processor
    source_policy = resolved_hf_policy_class.from_pretrained(
        model_path,
        torch_dtype=str(getattr(args, "torch_dtype", "auto") or "auto"),
        device_map=resolve_inference_device_map(getattr(args, "device_map", "auto"), runtime=runtime),
        attn_implementation=str(getattr(args, "attn_implementation", "") or "").strip() or None,
        max_new_tokens=int(max_new_tokens),
        max_total_images=int(max_total_images),
        max_seq_length=int(max_seq_length),
        keep_recent_tool_image_messages=int(keep_recent_tool_image_messages),
        keep_recent_text_messages=int(keep_recent_text_messages),
        max_image_side=int(max_image_side),
        max_image_pixels=int(max_image_pixels),
        do_sample=bool(do_sample),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        use_generation_cache=False,
    )
    return getattr(source_policy, "model", None), getattr(source_policy, "processor", processor)


def build_vllm_policy_from_model_path(
    *,
    args: Any,
    runtime: Any,
    model_path: str | Path,
    max_new_tokens: int,
    max_total_images: int,
    max_seq_length: int,
    keep_recent_tool_image_messages: int,
    keep_recent_text_messages: int,
    max_image_side: int,
    max_image_pixels: int,
    do_sample: bool,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    repetition_penalty: Optional[float],
    use_generation_cache: bool = True,
    prefer_direct_local_rank_runtime: bool = False,
    step_resolver: Optional[Callable[[], int]] = None,
    policy_class: Optional[type[VllmQwenGenerationPolicy]] = None,
    processor_loader: Optional[Callable[[str | Path], Any]] = None,
    hf_policy_class: Optional[type[QwenGenerationPolicy]] = None,
) -> VllmQwenGenerationPolicy:
    runtime_settings = build_vllm_runtime_settings(args)
    use_static_external_launcher_runtime = _should_use_static_external_launcher_runtime(
        args=args,
        runtime=runtime,
        prefer_direct_local_rank_runtime=prefer_direct_local_rank_runtime,
    )
    remote_lora_request = None
    if str(runtime_settings.get("vllm_mode", "")) == "server" or use_static_external_launcher_runtime:
        remote_lora_request = build_remote_vllm_lora_request(model_path)
    if remote_lora_request is not None:
        source_model = None
        processor = _load_processor_or_placeholder(model_path=model_path, processor_loader=processor_loader)
    else:
        source_model, processor = _maybe_load_adapter_source_model(
            model_path=model_path,
            args=args,
            runtime=runtime,
            processor_loader=processor_loader,
            hf_policy_class=hf_policy_class,
            max_new_tokens=max_new_tokens,
            max_total_images=max_total_images,
            max_seq_length=max_seq_length,
            keep_recent_tool_image_messages=keep_recent_tool_image_messages,
            keep_recent_text_messages=keep_recent_text_messages,
            max_image_side=max_image_side,
            max_image_pixels=max_image_pixels,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
    runtime_model_path = (
        _resolve_vllm_base_model_path(model_path)
        if use_static_external_launcher_runtime and remote_lora_request is not None
        else model_path
    )
    runtime_impl = create_vllm_runtime(
        args=args,
        runtime=runtime,
        model_path=runtime_model_path,
        prefer_direct_local_rank_runtime=prefer_direct_local_rank_runtime,
    )
    if runtime_impl is None:
        raise RuntimeError("Requested a vLLM policy, but use_vllm=false disabled runtime construction.")
    resolved_policy_class = policy_class or VllmQwenGenerationPolicy
    return resolved_policy_class(
        runtime=runtime_impl,
        source_model=source_model,
        step_resolver=step_resolver or (lambda: 0),
        guided_decoding_regex=str(getattr(args, "vllm_guided_decoding_regex", "") or ""),
        remote_lora_request=remote_lora_request,
        processor=processor,
        max_new_tokens=int(max_new_tokens),
        max_total_images=int(max_total_images),
        max_seq_length=int(max_seq_length),
        keep_recent_tool_image_messages=int(keep_recent_tool_image_messages),
        keep_recent_text_messages=int(keep_recent_text_messages),
        max_image_side=int(max_image_side),
        max_image_pixels=int(max_image_pixels),
        do_sample=bool(do_sample),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        use_generation_cache=bool(use_generation_cache),
    )


def build_inline_vllm_policy_factory(
    *,
    args: Any,
    vllm_runtime: Any,
    policy_class: Optional[type[VllmQwenGenerationPolicy]] = None,
) -> Callable[..., QwenGenerationPolicy]:
    resolved_policy_class = policy_class or VllmQwenGenerationPolicy

    def _factory(
        *,
        eval_model: Any,
        processor: Any,
        rollout_eval_config: Any,
        state: Any,
        runtime: Any,
    ) -> QwenGenerationPolicy:
        del runtime
        return _build_inline_vllm_policy(
            args=args,
            vllm_runtime=vllm_runtime,
            resolved_policy_class=resolved_policy_class,
            eval_model=eval_model,
            processor=processor,
            rollout_eval_config=rollout_eval_config,
            state=state,
        )

    return _factory


def _build_inline_vllm_policy(
    *,
    args: Any,
    vllm_runtime: Any,
    resolved_policy_class: type[VllmQwenGenerationPolicy],
    eval_model: Any,
    processor: Any,
    rollout_eval_config: Any,
    state: Any,
) -> QwenGenerationPolicy:
    return resolved_policy_class(
        runtime=vllm_runtime,
        source_model=eval_model,
        step_resolver=lambda: int(getattr(state, "global_step", 0) or 0),
        guided_decoding_regex=str(getattr(args, "vllm_guided_decoding_regex", "") or ""),
        processor=processor,
        max_new_tokens=int(
            getattr(
                rollout_eval_config,
                "policy_max_new_tokens",
                getattr(
                    args,
                    "policy_max_new_tokens",
                    getattr(args, "max_new_tokens", DEFAULT_POLICY_MAX_NEW_TOKENS),
                ),
            )
            or 256
        ),
        max_total_images=int(getattr(rollout_eval_config, "max_total_images", getattr(args, "max_total_images", 0)) or 0),
        max_seq_length=int(getattr(rollout_eval_config, "max_seq_length", getattr(args, "max_seq_length", 0)) or 0),
        keep_recent_tool_image_messages=int(
            getattr(
                rollout_eval_config,
                "keep_recent_tool_image_messages",
                getattr(args, "keep_recent_tool_image_messages", 0),
            )
            or 0
        ),
        keep_recent_text_messages=int(
            getattr(
                rollout_eval_config,
                "keep_recent_text_messages",
                getattr(args, "keep_recent_text_messages", 0),
            )
            or 0
        ),
        max_image_side=int(getattr(rollout_eval_config, "max_image_side", getattr(args, "max_image_side", 0)) or 0),
        max_image_pixels=int(getattr(rollout_eval_config, "max_image_pixels", getattr(args, "max_image_pixels", 0)) or 0),
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        repetition_penalty=None,
        use_generation_cache=bool(getattr(rollout_eval_config, "use_generation_cache", True)),
    )


def build_lazy_inline_vllm_policy_factory(
    *,
    args: Any,
    runtime_builder: Optional[Callable[..., Any]] = None,
    policy_class: Optional[type[VllmQwenGenerationPolicy]] = None,
) -> Callable[..., Any]:
    resolved_policy_class = policy_class or VllmQwenGenerationPolicy
    resolved_runtime_builder = runtime_builder or create_vllm_runtime

    def _factory(
        *,
        eval_model: Any,
        processor: Any,
        rollout_eval_config: Any,
        state: Any,
        runtime: Any,
    ) -> tuple[QwenGenerationPolicy, Callable[[], None]]:
        effective_args = _resolve_sft_vllm_runtime_args(args=args, runtime=runtime)
        if effective_args is not args:
            runtime_log(
                "SFT inline rollout eval rerouting distributed colocate vLLM through managed server runtime for multi-GPU compatibility.",
                runtime=runtime,
                main_process_only=True,
            )
        runtime_log(
            "SFT inline rollout eval booting lazy vLLM runtime at epoch end.",
            runtime=runtime,
            main_process_only=True,
        )
        vllm_runtime = resolved_runtime_builder(
            args=effective_args,
            runtime=runtime,
            model_path=str(getattr(args, "model_path", "") or ""),
        )
        if vllm_runtime is None:
            raise RuntimeError("Lazy inline vLLM policy factory expected a runtime, but vLLM is disabled.")
        runtime_log(
            (
                "SFT inline rollout eval attached lazy vLLM runtime: "
                f"base_model_path={getattr(vllm_runtime, 'base_model_path', 'unknown')} "
                f"mode={str(getattr(vllm_runtime, 'mode', getattr(effective_args, 'vllm_mode', 'colocate')))}"
            ),
            runtime=runtime,
            main_process_only=True,
        )
        policy = _build_inline_vllm_policy(
            args=args,
            vllm_runtime=vllm_runtime,
            resolved_policy_class=resolved_policy_class,
            eval_model=eval_model,
            processor=processor,
            rollout_eval_config=rollout_eval_config,
            state=state,
        )

        def _cleanup() -> None:
            close_fn = getattr(vllm_runtime, "close", None)
            if callable(close_fn):
                close_fn()

        return policy, _cleanup

    return _factory


def build_sft_external_recovery_vllm_policy_factory(
    *,
    args: Any,
    runtime_builder: Optional[Callable[..., Any]] = None,
    policy_class: Optional[type[VllmQwenGenerationPolicy]] = None,
    processor_loader: Optional[Callable[[str | Path], Any]] = None,
) -> Callable[..., Any]:
    resolved_runtime_builder = runtime_builder or create_sft_external_vllm_runtime
    resolved_policy_class = policy_class or VllmQwenGenerationPolicy

    def _factory(
        *,
        checkpoint_path: Path,
        model_path: str | Path,
        torch_dtype: str,
        attn_implementation: Optional[str],
        rollout_eval_config: Any,
        runtime: Any,
    ) -> Any:
        del model_path
        effective_args = _args_with_overrides(
            args,
            torch_dtype=str(torch_dtype or getattr(args, "torch_dtype", "auto") or "auto"),
            attn_implementation=attn_implementation
            if attn_implementation is not None
            else getattr(args, "attn_implementation", None),
        )
        remote_lora_request = build_remote_vllm_lora_request(checkpoint_path)
        processor = _load_processor_or_placeholder(
            model_path=checkpoint_path,
            processor_loader=processor_loader,
        )
        runtime_model_path = (
            _resolve_vllm_base_model_path(checkpoint_path)
            if remote_lora_request is not None
            else str(checkpoint_path)
        )
        runtime_log(
            (
                "SFT external rollout-eval will use direct torchrun external-launcher vLLM runtime: "
                f"checkpoint={checkpoint_path} runtime_model={runtime_model_path} "
                f"world_size={int(getattr(runtime, 'world_size', 1) or 1)}"
            ),
            runtime=runtime,
            main_process_only=True,
        )
        vllm_runtime = resolved_runtime_builder(
            args=effective_args,
            runtime=runtime,
            model_path=runtime_model_path,
        )
        if vllm_runtime is None:
            raise RuntimeError("External SFT rollout-eval vLLM policy factory requires use_vllm=true.")
        policy = resolved_policy_class(
            runtime=vllm_runtime,
            source_model=None,
            step_resolver=lambda: 0,
            guided_decoding_regex=str(getattr(effective_args, "vllm_guided_decoding_regex", "") or ""),
            remote_lora_request=remote_lora_request,
            processor=processor,
            max_new_tokens=int(
                getattr(
                    rollout_eval_config,
                    "policy_max_new_tokens",
                    getattr(
                        effective_args,
                        "policy_max_new_tokens",
                        getattr(effective_args, "max_new_tokens", DEFAULT_POLICY_MAX_NEW_TOKENS),
                    ),
                )
                or 256
            ),
            max_total_images=int(
                getattr(rollout_eval_config, "max_total_images", getattr(effective_args, "max_total_images", 0)) or 0
            ),
            max_seq_length=int(
                getattr(rollout_eval_config, "max_seq_length", getattr(effective_args, "max_seq_length", 0)) or 0
            ),
            keep_recent_tool_image_messages=int(
                getattr(
                    rollout_eval_config,
                    "keep_recent_tool_image_messages",
                    getattr(effective_args, "keep_recent_tool_image_messages", 0),
                )
                or 0
            ),
            keep_recent_text_messages=int(
                getattr(
                    rollout_eval_config,
                    "keep_recent_text_messages",
                    getattr(effective_args, "keep_recent_text_messages", 0),
                )
                or 0
            ),
            max_image_side=int(
                getattr(rollout_eval_config, "max_image_side", getattr(effective_args, "max_image_side", 0)) or 0
            ),
            max_image_pixels=int(
                getattr(rollout_eval_config, "max_image_pixels", getattr(effective_args, "max_image_pixels", 0)) or 0
            ),
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            use_generation_cache=bool(getattr(rollout_eval_config, "use_generation_cache", True)),
        )

        def _cleanup() -> None:
            try:
                vllm_runtime.close()
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return policy, _cleanup

    return _factory


def build_recovery_vllm_policy_factory(
    *,
    args: Any,
    runtime_builder: Optional[Callable[..., Any]] = None,
    policy_class: Optional[type[VllmQwenGenerationPolicy]] = None,
    processor_loader: Optional[Callable[[str | Path], Any]] = None,
    hf_policy_class: Optional[type[QwenGenerationPolicy]] = None,
) -> Callable[..., Any]:
    resolved_runtime_builder = runtime_builder or create_vllm_runtime
    resolved_policy_class = policy_class or VllmQwenGenerationPolicy

    def _factory(
        *,
        checkpoint_path: Path,
        model_path: str | Path,
        torch_dtype: str,
        attn_implementation: Optional[str],
        rollout_eval_config: Any,
        runtime: Any,
    ) -> Any:
        del model_path
        effective_args = _args_with_overrides(
            args,
            torch_dtype=str(torch_dtype or getattr(args, "torch_dtype", "auto") or "auto"),
            attn_implementation=attn_implementation
            if attn_implementation is not None
            else getattr(args, "attn_implementation", None),
        )
        use_static_external_launcher_runtime = (
            resolved_runtime_builder is create_vllm_runtime
            and _should_use_static_external_launcher_runtime(
                args=effective_args,
                runtime=runtime,
                prefer_direct_local_rank_runtime=True,
            )
        )
        source_model = None
        remote_lora_request = None
        if (
            str(getattr(effective_args, "vllm_mode", "colocate") or "colocate").strip().lower() == "server"
            or use_static_external_launcher_runtime
        ):
            processor = _load_processor_or_placeholder(
                model_path=checkpoint_path,
                processor_loader=processor_loader,
            )
            remote_lora_request = build_remote_vllm_lora_request(checkpoint_path)
        else:
            source_model, processor = _maybe_load_adapter_source_model(
                model_path=checkpoint_path,
                args=effective_args,
                runtime=runtime,
                processor_loader=processor_loader,
                hf_policy_class=hf_policy_class,
                max_new_tokens=int(
                    getattr(
                        rollout_eval_config,
                        "policy_max_new_tokens",
                        getattr(
                            effective_args,
                            "policy_max_new_tokens",
                            getattr(effective_args, "max_new_tokens", DEFAULT_POLICY_MAX_NEW_TOKENS),
                        ),
                    )
                    or 256
                ),
                max_total_images=int(
                    getattr(rollout_eval_config, "max_total_images", getattr(effective_args, "max_total_images", 0)) or 0
                ),
                max_seq_length=int(
                    getattr(rollout_eval_config, "max_seq_length", getattr(effective_args, "max_seq_length", 0)) or 0
                ),
                keep_recent_tool_image_messages=int(
                    getattr(
                        rollout_eval_config,
                        "keep_recent_tool_image_messages",
                        getattr(effective_args, "keep_recent_tool_image_messages", 0),
                    )
                    or 0
                ),
                keep_recent_text_messages=int(
                    getattr(
                        rollout_eval_config,
                        "keep_recent_text_messages",
                        getattr(effective_args, "keep_recent_text_messages", 0),
                    )
                    or 0
                ),
                max_image_side=int(
                    getattr(rollout_eval_config, "max_image_side", getattr(effective_args, "max_image_side", 0)) or 0
                ),
                max_image_pixels=int(
                    getattr(rollout_eval_config, "max_image_pixels", getattr(effective_args, "max_image_pixels", 0)) or 0
                ),
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=None,
            )
        runtime_model_path = (
            _resolve_vllm_base_model_path(checkpoint_path)
            if use_static_external_launcher_runtime and remote_lora_request is not None
            else checkpoint_path
        )
        vllm_runtime = resolved_runtime_builder(
            args=effective_args,
            runtime=runtime,
            model_path=runtime_model_path,
            prefer_direct_local_rank_runtime=True,
        )
        if vllm_runtime is None:
            raise RuntimeError("vLLM recovery policy factory requires use_vllm=true and an initialized runtime.")
        policy = resolved_policy_class(
            runtime=vllm_runtime,
            source_model=source_model,
            step_resolver=lambda: 0,
            guided_decoding_regex=str(getattr(effective_args, "vllm_guided_decoding_regex", "") or ""),
            remote_lora_request=remote_lora_request,
            processor=processor,
            max_new_tokens=int(
                getattr(
                    rollout_eval_config,
                    "policy_max_new_tokens",
                    getattr(
                        effective_args,
                        "policy_max_new_tokens",
                        getattr(effective_args, "max_new_tokens", DEFAULT_POLICY_MAX_NEW_TOKENS),
                    ),
                )
                or 256
            ),
            max_total_images=int(
                getattr(rollout_eval_config, "max_total_images", getattr(effective_args, "max_total_images", 0)) or 0
            ),
            max_seq_length=int(
                getattr(rollout_eval_config, "max_seq_length", getattr(effective_args, "max_seq_length", 0)) or 0
            ),
            keep_recent_tool_image_messages=int(
                getattr(
                    rollout_eval_config,
                    "keep_recent_tool_image_messages",
                    getattr(effective_args, "keep_recent_tool_image_messages", 0),
                )
                or 0
            ),
            keep_recent_text_messages=int(
                getattr(
                    rollout_eval_config,
                    "keep_recent_text_messages",
                    getattr(effective_args, "keep_recent_text_messages", 0),
                )
                or 0
            ),
            max_image_side=int(
                getattr(rollout_eval_config, "max_image_side", getattr(effective_args, "max_image_side", 0)) or 0
            ),
            max_image_pixels=int(
                getattr(rollout_eval_config, "max_image_pixels", getattr(effective_args, "max_image_pixels", 0)) or 0
            ),
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            use_generation_cache=bool(getattr(rollout_eval_config, "use_generation_cache", True)),
        )

        def _cleanup() -> None:
            try:
                vllm_runtime.close()
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return policy, _cleanup

    return _factory
