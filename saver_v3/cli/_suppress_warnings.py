"""Suppress noisy third-party warnings that pollute training and eval logs.

Call suppress_third_party_warnings() at the top of CLI entry points,
before any heavy imports.  Coverage:

  - vLLM LoRA visual-module, random-seed, chunked-prefill, kernel-config warnings
  - Transformers tokenizer PAD/BOS/EOS, slow-image-processor advisories
  - PyTorch distributed barrier device warning
"""
from __future__ import annotations

import os
import warnings


def suppress_third_party_warnings() -> None:
    # vLLM: LoRA visual module (2800+ lines), random seed, chunked prefill, kernel configs
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

    # Transformers: tokenizer PAD/BOS/EOS mismatch, slow image processor advisory
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    # PyTorch distributed: barrier() device context warning
    warnings.filterwarnings("ignore", message=r".*barrier\(\).*using the device under current context.*")
