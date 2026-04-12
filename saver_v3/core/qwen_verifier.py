from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from saver_v3.model.model_loading import build_hf_model_init_kwargs, ensure_flash_attention_supported_dtype
from saver_v3.model.qwen_policy import (
    DEFAULT_MODEL_PATH,
    _build_generation_kwargs,
    _configure_qwen_processor,
    _to_pil_image,
)


DEFAULT_VERIFIER_MODEL_PATH = os.environ.get("SAVER_QWEN_VERIFIER_MODEL_PATH", DEFAULT_MODEL_PATH)


def _clamp_score(value: Any) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, score))


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _normalize_view_score(score_dict: Dict[str, Any]) -> Dict[str, float]:
    normalized = {
        "exist_support": _clamp_score(score_dict.get("exist_support")),
        "category_support": _clamp_score(score_dict.get("category_support")),
        "temporal_support": _clamp_score(score_dict.get("temporal_support")),
        "precursor_support": _clamp_score(score_dict.get("precursor_support")),
        "finalize_support": _clamp_score(score_dict.get("finalize_support")),
        "counterfactual_support": _clamp_score(score_dict.get("counterfactual_support")),
    }
    overall = score_dict.get("overall_support")
    if overall is None:
        overall = (
            normalized["exist_support"]
            + normalized["category_support"]
            + normalized["temporal_support"]
            + normalized["precursor_support"]
            + normalized["finalize_support"]
            + normalized["counterfactual_support"]
        ) / 6.0
    normalized["overall_support"] = _clamp_score(overall)
    return {key: round(value, 6) for key, value in normalized.items()}


def _build_output_schema_example() -> str:
    template = {
        "full": {
            "exist_support": 0.0,
            "category_support": 0.0,
            "temporal_support": 0.0,
            "precursor_support": 0.0,
            "finalize_support": 0.0,
            "counterfactual_support": 0.0,
            "overall_support": 0.0,
        },
        "keep": {
            "exist_support": 0.0,
            "category_support": 0.0,
            "temporal_support": 0.0,
            "precursor_support": 0.0,
            "finalize_support": 0.0,
            "counterfactual_support": 0.0,
            "overall_support": 0.0,
        },
        "drop": {
            "exist_support": 0.0,
            "category_support": 0.0,
            "temporal_support": 0.0,
            "precursor_support": 0.0,
            "finalize_support": 0.0,
            "counterfactual_support": 0.0,
            "overall_support": 0.0,
        },
    }
    return json.dumps(template, ensure_ascii=False, indent=2)


class QwenSelfVerifier:
    """Structured Qwen-based scorer for SAVER counterfactual evidence views."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        max_images_per_view: int = 6,
    ):
        self.model = model
        self.processor = processor
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.max_images_per_view = int(max_images_per_view)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path = DEFAULT_VERIFIER_MODEL_PATH,
        *,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        attn_implementation: Optional[str] = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        max_images_per_view: int = 6,
    ) -> "QwenSelfVerifier":
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except Exception as exc:
            raise ImportError(
                "QwenSelfVerifier requires a recent transformers build with Qwen3-VL support. "
                "Install it with `pip install git+https://github.com/huggingface/transformers accelerate`."
            ) from exc

        model_init_kwargs = build_hf_model_init_kwargs(
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device_map=device_map,
            prefer_inference_dtype=True,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(str(model_path), **model_init_kwargs)
        ensure_flash_attention_supported_dtype(
            model,
            attn_implementation=attn_implementation,
        )
        model.eval()
        processor = _configure_qwen_processor(AutoProcessor.from_pretrained(str(model_path)))
        return cls(
            model=model,
            processor=processor,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_images_per_view=max_images_per_view,
        )

    def score_views(
        self,
        *,
        views: Dict[str, Dict[str, Any]],
        claim: Dict[str, Any],
        verification_mode: str,
        question: str,
    ) -> Dict[str, Dict[str, float]]:
        messages = self._build_messages(views=views, claim=claim, verification_mode=verification_mode, question=question)
        output_text = self._generate(messages)
        parsed = _extract_json_object(output_text)
        view_scores: Dict[str, Dict[str, float]] = {}
        for view_name in ("full", "keep", "drop"):
            view_scores[view_name] = _normalize_view_score(parsed.get(view_name) or {})
        return view_scores

    def _build_messages(
        self,
        *,
        views: Dict[str, Dict[str, Any]],
        claim: Dict[str, Any],
        verification_mode: str,
        question: str,
    ) -> List[Dict[str, Any]]:
        system_prompt = (
            "You are a SAVER counterfactual evidence verifier. "
            "Given a structured anomaly claim and four evidence views, score how well each view supports the claim. "
            "Return valid JSON only. Begin with { and end with }. "
            "Use double quotes for all keys and strings. "
            "Do not wrap the JSON in markdown fences. "
            "Do not include explanations, comments, or any extra text outside the JSON object."
        )

        schema_text = (
            "Return a JSON object with exactly these keys: full, keep, drop. "
            "Each key maps to an object with numeric values in [0,1] for "
            "exist_support, category_support, temporal_support, precursor_support, finalize_support, "
            "counterfactual_support, overall_support.\n"
            "Follow this exact JSON skeleton:\n"
            f"{_build_output_schema_example()}"
        )
        user_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Task: {question or 'Verify anomaly claim support.'}\n"
                    f"Verification mode: {verification_mode}\n"
                    f"Claim JSON: {json.dumps(claim, ensure_ascii=False)}\n"
                    f"{schema_text}\n"
                    "Use the images and timestamps from each view to estimate support. "
                    "The keep view should test sufficiency, the drop view should test necessity, "
                    "and finalize_support should reflect whether the selected evidence is sufficient to finalize.\n"
                ),
            }
        ]

        for view_name in ("full", "keep", "drop"):
            payload = views.get(view_name) or {}
            summary_text = payload.get("summary_text") or f"View {view_name} has no summary."
            user_content.append({"type": "text", "text": f"[{view_name}] {summary_text}"})
            for image in (payload.get("images") or [])[: self.max_images_per_view]:
                user_content.append({"type": "image", "image": _to_pil_image(image)})
            if payload.get("timestamps"):
                timestamp_text = ", ".join(f"{float(value):.3f}s" for value in payload["timestamps"])
                user_content.append({"type": "text", "text": f"[{view_name}] timestamps: {timestamp_text}"})

        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user_content},
        ]

    def _generate(self, messages: List[Dict[str, Any]]) -> str:
        inputs = self._build_inputs(messages)
        inputs = self._move_to_model_device(inputs)
        generation_kwargs = self._generation_kwargs()
        output_ids = self.model.generate(**inputs, **generation_kwargs)
        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, output_ids)]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text[0]

    def _build_inputs(self, messages: List[Dict[str, Any]]) -> Any:
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            prompt_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            images = []
            for message in messages:
                for item in message.get("content", []):
                    if item.get("type") == "image" and "image" in item:
                        images.append(item["image"])
            processor_kwargs: Dict[str, Any] = {
                "text": prompt_text,
                "padding": True,
                "return_tensors": "pt",
            }
            if images:
                processor_kwargs["images"] = images
            return self.processor(**processor_kwargs)

    def _move_to_model_device(self, inputs: Any) -> Any:
        if not hasattr(inputs, "to"):
            return inputs
        device = getattr(self.model, "device", None)
        if device is None:
            return inputs
        return inputs.to(device)

    def _generation_kwargs(self) -> Dict[str, Any]:
        return _build_generation_kwargs(
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
        )
