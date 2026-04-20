# Repository Layout

`idea2_v3` separates method logic, runtime orchestration, and imported reference
ports so the SAVER-specific parts stay inspectable.

## Design Rules

- `configs/` stores reusable YAML or JSON templates for raw-data preparation, SFT, rollout inference/eval, RL, and model policy.
- `envs/` stores environment definitions and runtime env examples.
- `scripts/` stores `torchrun` and DeepSpeed launch wrappers for raw-data preparation, training, and local-rank vLLM inference.
- `docs/` explains how the pieces fit together and what is currently implemented.
- `saver_v3/` contains the owned package code for v3 data preparation, runtime wrappers, and training/inference entrypoints.
- `third_party_ports/` contains selectively copied reference code from `idea2_v2` and `TimeSearch-R` that v3 depends on.
- `tests/` contains repository-level smoke and unit coverage.

## Config Layering

The expected config layering is:

1. `configs/model/qwen3_vl_8b_full.yaml`
2. `configs/model/attention_fa3_only.yaml`
3. `configs/deepspeed/zero3_full_model.json`
4. `configs/prepare_sft/*.yaml` for raw -> prepared conversion
5. Task-specific config under `configs/sft/`, `configs/inference/`, `configs/rollout_eval/`, or `configs/rl/`

The shell wrappers in `scripts/` reference those files directly so the CLI
surface stays small and explicit.

## Runtime Split

- SFT uses the DeepSpeed launcher and delegates to `saver_v3.sft.training.run_standard_sft`.
- Policy inference uses raw SAVER data with one vLLM engine per local rank and full SAVER tool rollout.
- Rollout eval uses the same raw-data rollout path plus `summarize_saver_metrics` and semantic metrics.
- RL uses a dedicated TRL + vLLM GRPO route while reusing SAVER method, reward, and rollout semantics from the v2 stack.
