# vLLM Inference

The v3 inference path follows the `TimeSearch-R` local-rank pattern instead of a
single managed HTTP server.

## Launch Pattern

Both official wrappers use raw SAVER JSONL and launch under `torchrun`:

```bash
torchrun --nproc_per_node=8 --module saver_v3.cli.run_policy_rollout_vllm ...
torchrun --nproc_per_node=8 --module saver_v3.cli.run_sft_rollout_eval_vllm ...
```

Each rank:

- pins `CUDA_VISIBLE_DEVICES` to its `LOCAL_RANK`
- builds a local vLLM engine with `tensor_parallel_size=1`
- uses `distributed_executor_backend="external_launcher"`
- writes one shard output file or shard-local rollout-eval artifacts

This matches the multi-GPU inference structure used in `TimeSearch-R`.

## Shared Config

- Policy rollout config: `configs/inference/vllm_qwen3_vl_8b_rollout.yaml`
- Rollout-eval config: `configs/rollout_eval/vllm_qwen3_vl_8b.yaml`
- Model defaults: `configs/model/qwen3_vl_8b_full.yaml`

## Wrapper Commands

```bash
bash scripts/run_policy_rollout_vllm.sh
bash scripts/run_sft_rollout_eval_vllm.sh
```

## Scope

- Policy rollout uses raw canonical SAVER records and executes full SAVER tool rollouts.
- Rollout eval reuses the same raw-data rollout path, then scores with `summarize_saver_metrics` and semantic metrics.
- The old compact-trace message predictor path is deprecated and intentionally exits with a migration error.
