from __future__ import annotations
import warnings

from copy import deepcopy
from typing import Any, Dict, List, Sequence


def _process_vision_info(messages: Sequence[Dict[str, Any]], *, return_video_kwargs: bool = True):
    try:
        from third_party_ports.timesearch_r.time_r1.utils.qwen_vl_utils import process_vision_info
    except Exception as exc:
        raise ImportError(
            "Qwen3-VL multimodal batching requires TimeSearch-R qwen_vl_utils dependencies "
            "(notably torchvision/qwen-vl-utils) in the active runtime environment."
        ) from exc
    return process_vision_info(messages, return_video_kwargs=return_video_kwargs)


def _apply_chat_template(processor: Any, messages: Sequence[Dict[str, Any]], *, add_generation_prompt: bool) -> str:
    return processor.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=add_generation_prompt)


def build_hf_generation_batch(processor: Any, messages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    prompt_text = _apply_chat_template(processor, messages, add_generation_prompt=True)
    copied_messages = deepcopy(list(messages))
    image_inputs, video_inputs, video_kwargs = _process_vision_info(copied_messages, return_video_kwargs=True)
    batch = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        fps=(video_kwargs or {}).get("fps"),
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return batch


def build_sft_training_batch(processor: Any, examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    warnings.warn("build_sft_training_batch uses incorrect prefix-based label masking. Use saver_v3.sft.training instead.", DeprecationWarning)
    full_messages = [deepcopy(dict(example).get("messages") or []) for example in examples]
    prefix_messages = [messages[:-1] for messages in full_messages]
    full_text = [_apply_chat_template(processor, messages, add_generation_prompt=False) for messages in full_messages]
    prefix_text = [_apply_chat_template(processor, messages, add_generation_prompt=False) for messages in prefix_messages]
    image_inputs, video_inputs, video_kwargs = _process_vision_info(full_messages, return_video_kwargs=True)
    full_batch = processor(
        text=deepcopy(full_text),
        images=image_inputs,
        videos=video_inputs,
        fps=(video_kwargs or {}).get("fps"),
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    prefix_batch = processor(
        text=deepcopy(prefix_text),
        images=image_inputs,
        videos=video_inputs,
        fps=(video_kwargs or {}).get("fps"),
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    labels = full_batch["input_ids"].clone()
    attention_mask = full_batch.get("attention_mask")
    if attention_mask is not None:
        labels = labels.masked_fill(attention_mask == 0, -100)
    for index in range(labels.shape[0]):
        prefix_length = int(prefix_batch["attention_mask"][index].sum().item())
        labels[index, :prefix_length] = -100
    full_batch["labels"] = labels
    return full_batch


def build_vllm_inputs(processor: Any, messages_batch: Sequence[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    prompts = processor.apply_chat_template(list(messages_batch), tokenize=False, add_generation_prompt=True)
    if isinstance(prompts, str):
        prompts = [prompts]
    llm_inputs: List[Dict[str, Any]] = []
    for prompt, messages in zip(prompts, messages_batch):
        image_inputs, video_inputs, video_kwargs = _process_vision_info(deepcopy(list(messages)), return_video_kwargs=True)
        mm_data: Dict[str, Any] = {}
        mm_kw: Dict[str, Any] = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
            for key, value in (video_kwargs or {}).items():
                if isinstance(value, list):
                    if len(value) == 1:
                        mm_kw[key] = value[0]
                    elif value:
                        mm_kw[key] = value  # preserve full list for multi-video
                else:
                    mm_kw[key] = value
        llm_inputs.append({
            "prompt": prompt,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": mm_kw,
        })
    return llm_inputs
