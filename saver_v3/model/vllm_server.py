from __future__ import annotations

import argparse
import inspect
import logging
import os
import socket
from contextlib import asynccontextmanager
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Any, Optional

import torch

from saver_v3.rl.trl_compat import patch_vllm_guided_decoding_params
from saver_v3.model.vllm_transport import decode_transport_payload


logger = logging.getLogger(__name__)
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _build_guided_decoding_sampling_kwargs(
    guided_decoding_regex: str,
    *,
    parameter_names: Optional[set[str]] = None,
) -> dict[str, Any]:
    regex = str(guided_decoding_regex or "").strip()
    if not regex:
        return {}

    patch_vllm_guided_decoding_params()
    if parameter_names is None:
        try:
            from vllm import SamplingParams

            parameter_names = set(inspect.signature(SamplingParams).parameters.keys())
        except Exception:
            return {}
    else:
        parameter_names = set(parameter_names)

    if "guided_decoding" in parameter_names:
        from vllm.sampling_params import GuidedDecodingParams

        return {"guided_decoding": GuidedDecodingParams(regex=regex)}

    if "structured_outputs" in parameter_names:
        from vllm.sampling_params import StructuredOutputsParams

        return {"structured_outputs": StructuredOutputsParams(regex=regex)}

    if "guided_decoding_regex" in parameter_names:
        return {"guided_decoding_regex": regex}

    if "regex" in parameter_names:
        return {"regex": regex}

    raise TypeError(
        "Installed vLLM SamplingParams does not expose a supported guided decoding field."
    )


def _get_open_port() -> int:
    try:
        from vllm.utils import get_open_port as _vllm_get_open_port

        return int(_vllm_get_open_port())
    except Exception:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            return int(sock.getsockname()[1])


def _parse_optional_bool(value: str | None) -> Optional[bool]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean flag, got {value!r}.")


def _chunk_list(items: list[Any], num_chunks: int) -> list[list[Any]]:
    if num_chunks <= 0:
        return [list(items)]
    base, remainder = divmod(len(items), num_chunks)
    return [
        items[index * base + min(index, remainder) : (index + 1) * base + min(index + 1, remainder)]
        for index in range(num_chunks)
    ]


class WeightSyncWorkerExtension:
    pynccl_comm = None
    client_rank = None

    def init_communicator(self, host: str, port: int, world_size: int) -> None:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.parallel_state import get_world_group
        from vllm.distributed.utils import StatelessProcessGroup

        if self.pynccl_comm is not None:
            raise RuntimeError("Weight update group already initialized. Call close_communicator first.")
        rank = get_world_group().rank
        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size)
        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)
        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype: torch.dtype, shape: tuple[int, ...]) -> None:
        if self.pynccl_comm is None:
            raise RuntimeError("Communicator not initialized. Call init_communicator first.")
        weight = torch.empty(shape, dtype=dtype, device=self.device)
        self.pynccl_comm.broadcast(weight, src=self.client_rank)
        self.pynccl_comm.group.barrier()
        self.model_runner.model.load_weights(weights=[(name, weight)])

    def close_communicator(self) -> None:
        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None
            self.client_rank = None


def _build_single_rank_external_launcher_env(master_port: int) -> dict[str, str]:
    return {
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "1",
        "GROUP_RANK": "0",
        "ROLE_RANK": "0",
        "ROLE_WORLD_SIZE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(int(master_port)),
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    }


def _llm_worker(args: argparse.Namespace, data_parallel_rank: int, master_port: int, connection: Connection) -> None:
    from vllm import LLM

    os.environ["VLLM_DP_RANK"] = str(data_parallel_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(data_parallel_rank)
    os.environ["VLLM_DP_SIZE"] = str(args.data_parallel_size)
    os.environ["VLLM_DP_MASTER_PORT"] = str(master_port)

    use_single_rank_external_launcher = (
        int(args.data_parallel_size) == 1 and int(args.tensor_parallel_size) == 1
    )
    if use_single_rank_external_launcher:
        os.environ.update(_build_single_rank_external_launcher_env(master_port))
        os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
        os.environ.setdefault("VLLM_LOOPBACK_IP", "127.0.0.1")

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "dtype": str(args.dtype),
        "max_model_len": int(args.max_model_len) if args.max_model_len else None,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enforce_eager": args.enforce_eager,
        "kv_cache_dtype": str(args.kv_cache_dtype),
        "worker_extension_cls": "saver_v3.model.vllm_server.WeightSyncWorkerExtension",
        "enable_lora": bool(args.enable_lora),
        "max_lora_rank": int(args.max_lora_rank),
    }
    if use_single_rank_external_launcher:
        llm_kwargs["distributed_executor_backend"] = "external_launcher"
    if args.revision:
        llm_kwargs["revision"] = str(args.revision)
    limit_mm_per_prompt: dict[str, int] = {"video": max(1, int(args.limit_mm_per_prompt_video))}
    if int(args.limit_mm_per_prompt_image) > 0:
        limit_mm_per_prompt["image"] = int(args.limit_mm_per_prompt_image)
    llm_kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt

    llm = None
    try:
        llm = LLM(**{key: value for key, value in llm_kwargs.items() if value is not None})
        connection.send({"status": "ready"})

        while True:
            try:
                command = connection.recv()
            except KeyboardInterrupt:
                if llm is not None:
                    llm.collective_rpc(method="close_communicator")
                break
            if command["type"] in {"call", "fire_and_forget"}:
                method = getattr(llm, command["method"])
                result = method(*command.get("args", ()), **command.get("kwargs", {}))
                if command["type"] == "call":
                    connection.send(result)
            elif command["type"] == "shutdown":
                break
    except Exception as exc:
        try:
            connection.send({"status": "error", "message": f"{type(exc).__name__}: {exc}"})
        except Exception:
            pass
        raise
    finally:
        if llm is not None:
            try:
                llm.collective_rpc(method="close_communicator")
            except Exception:
                pass
            shutdown_fn = getattr(llm, "shutdown", None)
            if callable(shutdown_fn):
                try:
                    shutdown_fn()
                except Exception:
                    pass
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception:
                pass


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SAVER's multimodal vLLM server.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--enable-prefix-caching", type=_parse_optional_bool, default=True)
    parser.add_argument("--enforce-eager", type=_parse_optional_bool, default=None)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--enable-lora", action="store_true")
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--limit-mm-per-prompt-image", type=int, default=0)
    parser.add_argument("--limit-mm-per-prompt-video", type=int, default=8)
    parser.add_argument("--log-level", default="info")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest

    args = _build_arg_parser().parse_args(argv)
    master_port = _get_open_port()
    connections: list[Connection] = []
    processes: list[Process] = []
    for data_parallel_rank in range(int(args.data_parallel_size)):
        parent_connection, child_connection = Pipe()
        process = Process(
            target=_llm_worker,
            args=(args, int(data_parallel_rank), int(master_port), child_connection),
        )
        process.start()
        connections.append(parent_connection)
        processes.append(process)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ready_connections = set()
        while len(ready_connections) < int(args.data_parallel_size):
            for connection in connections:
                if connection in ready_connections:
                    continue
                msg = connection.recv()
                if isinstance(msg, dict) and msg.get("status") == "ready":
                    ready_connections.add(connection)
                    continue
                if isinstance(msg, dict) and msg.get("status") == "error":
                    raise RuntimeError(f"vLLM worker failed during startup: {msg.get('message')}")
        yield
        for connection in connections:
            try:
                connection.send({"type": "shutdown"})
            except Exception:
                pass
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join()

    app = FastAPI(lifespan=lifespan)

    class GenerateMultimodalRequest(BaseModel):
        llm_inputs_b64: str
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: Optional[str] = None
        stop: Optional[list[str]] = None
        include_stop_str_in_output: bool = False
        lora_name: str = ""
        lora_int_id: int = 0
        lora_path: str = ""
        base_model_name: str = ""

    class GenerateResponse(BaseModel):
        completion_ids: list[list[int]]

    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    @app.get("/health/")
    async def health():
        return {"status": "ok"}

    @app.get("/get_world_size/")
    async def get_world_size():
        return {"world_size": int(args.tensor_parallel_size) * int(args.data_parallel_size)}

    @app.post("/generate_multimodal/", response_model=GenerateResponse)
    async def generate_multimodal(request: GenerateMultimodalRequest):
        llm_inputs = list(decode_transport_payload(request.llm_inputs_b64) or [])
        lora_request = None
        if request.lora_path:
            lora_request = LoRARequest(
                lora_name=str(request.lora_name or "adapter"),
                lora_int_id=max(1, int(request.lora_int_id or 1)),
                lora_path=str(request.lora_path),
                base_model_name=str(request.base_model_name or "") or None,
            )
        sampling_kwargs: dict[str, Any] = {
            "n": int(request.n),
            "repetition_penalty": float(request.repetition_penalty),
            "temperature": float(request.temperature),
            "top_p": float(request.top_p),
            "top_k": int(request.top_k),
            "min_p": float(request.min_p),
            "max_tokens": int(request.max_tokens),
        }
        try:
            sampling_parameter_names = set(inspect.signature(SamplingParams).parameters.keys())
        except Exception:
            sampling_parameter_names = set()
        sampling_kwargs.update(
            _build_guided_decoding_sampling_kwargs(
                str(request.guided_decoding_regex or ""),
                parameter_names=sampling_parameter_names,
            )
        )
        if request.stop and "stop" in sampling_parameter_names:
            sampling_kwargs["stop"] = list(request.stop)
        if "include_stop_str_in_output" in sampling_parameter_names:
            sampling_kwargs["include_stop_str_in_output"] = bool(request.include_stop_str_in_output)
        sampling_params = SamplingParams(
            **sampling_kwargs,
        )
        chunked_inputs = _chunk_list(llm_inputs, int(args.data_parallel_size))
        for connection, prompts in zip(connections, chunked_inputs):
            if not prompts:
                prompts = [{"prompt": "<placeholder>"}]
            connection.send(
                {
                    "type": "call",
                    "method": "generate",
                    "kwargs": {
                        "prompts": prompts,
                        "sampling_params": sampling_params,
                        "use_tqdm": False,
                        "lora_request": lora_request,
                    },
                }
            )
        all_outputs = [connection.recv() for connection in connections]
        all_outputs = [output for output, prompts in zip(all_outputs, chunked_inputs) if prompts]
        completion_ids = [
            list(output.token_ids)
            for outputs in all_outputs
            for request_output in outputs
            for output in request_output.outputs
        ]
        return {"completion_ids": completion_ids}

    @app.post("/init_communicator/")
    async def init_communicator(request: InitCommunicatorRequest):
        world_size = int(args.tensor_parallel_size) * int(args.data_parallel_size) + 1
        kwargs = {"method": "init_communicator", "args": (request.host, int(request.port), int(world_size))}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
        return {"message": "ok"}

    @app.post("/update_named_param/")
    async def update_named_param(request: UpdateWeightsRequest):
        dtype = torch.__getattribute__(request.dtype.split(".")[-1])
        kwargs = {"method": "update_named_param", "args": (request.name, dtype, tuple(request.shape))}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
        return {"message": "ok"}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():
        for connection in connections:
            connection.send({"type": "call", "method": "reset_prefix_cache"})
        return {"message": "ok", "success": all(connection.recv() for connection in connections)}

    @app.post("/close_communicator/")
    async def close_communicator():
        kwargs = {"method": "close_communicator"}
        for connection in connections:
            connection.send({"type": "fire_and_forget", "method": "collective_rpc", "kwargs": kwargs})
        return {"message": "ok"}

    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level=str(args.log_level))


if __name__ == "__main__":
    main()
