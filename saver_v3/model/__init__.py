"""Qwen3-VL training-facing model helpers."""

from saver_v3.model.qwen3vl import (
    DEFAULT_QWEN3_VL_8B_INSTRUCT_MODEL,
    build_qwen3vl_inputs,
    configure_qwen3vl_processor,
    extract_vision_inputs,
    load_qwen3vl_model,
    prepare_qwen3vl_for_full_training,
)
from saver_v3.model.qwen3vl_loader import load_qwen3vl_full_model
from saver_v3.model.qwen3vl_processor import load_qwen3vl_processor
from saver_v3.model.qwen3vl_vision_io import (
    build_hf_generation_batch,
    build_sft_training_batch,
    build_vllm_inputs,
)
from saver_v3.model.trainability import assert_full_model_trainable, build_trainability_report

__all__ = [
    "DEFAULT_QWEN3_VL_8B_INSTRUCT_MODEL",
    "assert_full_model_trainable",
    "build_hf_generation_batch",
    "build_qwen3vl_inputs",
    "build_sft_training_batch",
    "build_trainability_report",
    "build_vllm_inputs",
    "configure_qwen3vl_processor",
    "extract_vision_inputs",
    "load_qwen3vl_full_model",
    "load_qwen3vl_model",
    "load_qwen3vl_processor",
    "prepare_qwen3vl_for_full_training",
]
