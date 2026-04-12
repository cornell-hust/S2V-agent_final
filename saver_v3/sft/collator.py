from __future__ import annotations

from typing import Any, Dict, Sequence

from saver_v3.model.qwen3vl_vision_io import build_sft_training_batch
import warnings
warnings.warn(
    "saver_v3.sft.collator.EpisodeSFTCollator uses prefix-based label masking "
    "which is INCORRECT for multi-turn agent dialogues. Use saver_v3.sft.training.BatchEpisodeSFTCollator instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)




class EpisodeSFTCollator:
    def __init__(self, processor: Any):
        self.processor = processor

    def __call__(self, examples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return build_sft_training_batch(self.processor, examples)
