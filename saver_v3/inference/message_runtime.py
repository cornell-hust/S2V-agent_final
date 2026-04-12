from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from saver_v3.inference.vllm_qwen3vl import LocalRankVllmConfig, build_local_rank_vllm_engine
from saver_v3.model import build_vllm_inputs, load_qwen3vl_processor


@dataclass
class VllmSamplingConfig:
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1


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

    def generate(self, messages_batch: Sequence[Sequence[Dict[str, Any]]], sampling: VllmSamplingConfig | None = None) -> List[str]:
        try:
            from vllm import SamplingParams
        except Exception as exc:
            raise ImportError("vLLM runtime requires the vllm package.") from exc
        sampling_cfg = sampling or VllmSamplingConfig()
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
        """Single-example interface required by counterfactual verification."""
        results = self.generate([messages], sampling=sampling)
        return results[0] if results else ""

    def generate_from_messages_batch(self, messages_batch, *, sampling=None):
        """Batch interface required by counterfactual verification."""
        return self.generate(messages_batch, sampling=sampling)
