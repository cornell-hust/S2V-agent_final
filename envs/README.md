# Environment Templates

Two environment templates are provided:

- `train-qwen3-vl-8b-full.yml` for full-model SFT or RL training on 8 GPUs
- `infer-vllm.yml` for vLLM rollout evaluation or policy inference

The templates are intentionally conservative and should be adjusted to the
actual cluster image, CUDA version, and wheel availability.

FlashAttention-3 is not hard-pinned in the Conda files because Hopper builds
are cluster-specific. Create the environment first, install or verify the FA3
path that matches the local CUDA image, and then run
`python scripts/check_attention_backend.py`.

`runtime.env.example` captures runtime variables that should not be baked into
tracked config files.
