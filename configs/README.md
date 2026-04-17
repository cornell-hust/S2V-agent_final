# Config Templates

This directory stores the owned v3 runtime contract.

- `model/` contains Qwen3-VL-8B defaults and the FA3-only policy file.
- `deepspeed/` contains the shared ZeRO-3 template used by the DS8 wrappers.
- `prepare_sft/` contains the one-shot raw-SAVER -> `compact_trace_v2` preparation template.
- `sft/` contains the full-model SFT template for `train_sft_ds`; it now consumes `data.prepared_data_path`.
- `inference/` contains the full-rollout policy inference template for `run_policy_rollout_vllm`.
- `rollout_eval/` contains the raw-SAVER rollout-eval template for `run_sft_rollout_eval_vllm`.
- `rl/` contains the full-model active RL template for `train_rl_ds`; it now requires materialized runtime item caches and the pure-pack episode GRPO route.

These files are implementation-facing, not placeholder stubs.
