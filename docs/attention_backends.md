# Attention Backends

This top-level layer is intentionally strict:

1. Use `flash_attention_3`.
2. Do not fall back to FA2.
3. Do not fall back to SDPA.

## Preconditions

- Hopper-class GPUs are expected on every rank.
- CUDA 12.3+ is required for the intended FA3 path.
- A FA3-capable Python install must be present before launch.
- `transformers>=4.57.0` is the minimum baseline for Qwen3-VL.

## Operational Rule

- Keep `SAVER_V3_ATTN_BACKEND=fa3`.
- Keep `SAVER_V3_REQUIRE_FA3=1`.
- Run `python scripts/check_attention_backend.py` before DS8 training or vLLM inference.

If the checker reports `ready: false`, do not start the run. This layer fails
closed by design.

## Current Enforcement

- `configs/model/attention_fa3_only.yaml` records the policy.
- `scripts/check_attention_backend.py` reports whether the host satisfies the
  FA3-only contract.
- `saver_v3.common.fa3_guard.ensure_fa3_training_ready()` enforces the same
  contract inside the Python runtime.
