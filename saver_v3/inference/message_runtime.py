from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from saver_v3.common.runtime import distributed_runtime_from_env
from saver_v3.inference.vllm_qwen3vl import LocalRankVllmConfig, build_local_rank_vllm_engine
from saver_v3.model import build_vllm_inputs, load_qwen3vl_processor
from saver_v3.model.qwen_policy import QwenGenerationPolicy


@dataclass
class MessageSamplingConfig:
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1


VllmSamplingConfig = MessageSamplingConfig
TorchSamplingConfig = MessageSamplingConfig


class VllmQwen3MessageRuntime:
    def __init__(
        self,
        *,
        model_path: str,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        limit_mm_per_prompt: Dict[str, int] | None = None,
    ):
        self.model_path = model_path
        self.processor = load_qwen3vl_processor(model_path)
        self.engine = build_local_rank_vllm_engine(
            LocalRankVllmConfig(
                model_path=model_path,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                limit_mm_per_prompt=limit_mm_per_prompt,
            )
        )

    def generate(self, messages_batch: Sequence[Sequence[Dict[str, Any]]], sampling: MessageSamplingConfig | None = None) -> List[str]:
        try:
            from vllm import SamplingParams
        except Exception as exc:
            raise ImportError("vLLM runtime requires the vllm package.") from exc
        sampling_cfg = sampling or MessageSamplingConfig()
        llm_inputs = build_vllm_inputs(self.processor, messages_batch)
        sampling_params = SamplingParams(
            n=1,
            repetition_penalty=1.0,
            max_tokens=int(sampling_cfg.max_tokens),
            temperature=float(sampling_cfg.temperature),
            top_p=float(sampling_cfg.top_p),
            top_k=int(sampling_cfg.top_k),
            seed=42,
        )
        outputs = self.engine.generate(llm_inputs, sampling_params=sampling_params, use_tqdm=False)
        completions: List[str] = []
        for request_outputs in outputs:
            if not request_outputs.outputs:
                completions.append("")
                continue
            completions.append(str(request_outputs.outputs[0].text or ""))
        return completions

    def generate_from_messages(self, messages, *, sampling=None):
        results = self.generate([messages], sampling=sampling)
        return results[0] if results else ""

    def generate_from_messages_batch(self, messages_batch, *, sampling=None):
        return self.generate(messages_batch, sampling=sampling)


class TorchQwen3MessageRuntime:
    def __init__(
        self,
        *,
        model_path: str,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_3",
        max_new_tokens: int = 512,
        max_total_images: int = 0,
        max_seq_length: int = 0,
        max_image_side: int = 0,
        max_image_pixels: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        use_generation_cache: bool = True,
    ):
        runtime = distributed_runtime_from_env()
        try:
            import torch
        except Exception as exc:
            raise ImportError("TorchQwen3MessageRuntime requires torch to be installed.") from exc
        if torch.cuda.is_available():
            torch.cuda.set_device(int(runtime.local_rank))
            device_map: Any = {"": int(torch.cuda.current_device())}
        else:
            device_map = None
        self.policy = QwenGenerationPolicy.from_pretrained(
            model_path=model_path,
            torch_dtype=str(torch_dtype or "auto"),
            device_map=device_map,
            attn_implementation=str(attn_implementation or "").strip() or None,
            max_new_tokens=int(max_new_tokens),
            max_total_images=int(max_total_images),
            max_seq_length=int(max_seq_length),
            max_image_side=int(max_image_side),
            max_image_pixels=int(max_image_pixels),
            do_sample=bool(float(temperature) > 0.0),
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            use_generation_cache=bool(use_generation_cache),
        )

    def generate(self, messages_batch: Sequence[Sequence[Dict[str, Any]]], sampling: MessageSamplingConfig | None = None) -> List[str]:
        sampling_cfg = sampling or MessageSamplingConfig()
        if (
            int(sampling_cfg.max_tokens) != int(self.policy.max_new_tokens)
            or float(sampling_cfg.temperature) != float(self.policy.temperature or 0.0)
            or float(sampling_cfg.top_p) != float(self.policy.top_p or 1.0)
            or int(sampling_cfg.top_k) != int(self.policy.top_k or -1)
        ):
            self.policy.max_new_tokens = int(sampling_cfg.max_tokens)
            self.policy.temperature = float(sampling_cfg.temperature)
            self.policy.top_p = float(sampling_cfg.top_p)
            self.policy.top_k = int(sampling_cfg.top_k)
            self.policy.do_sample = bool(float(sampling_cfg.temperature) > 0.0)
        return list(self.policy.generate_from_messages_batch([list(messages) for messages in messages_batch]))

    def generate_from_messages(self, messages, *, sampling=None):
        results = self.generate([messages], sampling=sampling)
        return results[0] if results else ""

    def generate_from_messages_batch(self, messages_batch, *, sampling=None):
        return self.generate(messages_batch, sampling=sampling)
