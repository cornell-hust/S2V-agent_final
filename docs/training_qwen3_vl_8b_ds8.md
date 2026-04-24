# Qwen3-VL-8B Full-Model DeepSpeed Training

This layer targets single-node 8 GPU full-model training for
`/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/qwen3-vl-8b-Instruct`.

## Inputs

- Model defaults: `configs/model/qwen3_vl_8b_full.yaml`
- Attention policy: `configs/model/attention_fa3_only.yaml`
- DeepSpeed config: `configs/deepspeed/zero3_full_model.json`
- Raw -> prepared manifest config: `configs/prepare_sft/qwen3_vl_8b_prepare.yaml`
- SFT task config: `configs/sft/qwen3_vl_8b_full_train.yaml`
- RL task config: `configs/rl/qwen3_vl_8b_grpo_train.yaml`

## Implemented Entry Points

- `saver_v3.cli.prepare_sft_manifest`
- `saver_v3.cli.train_sft_ds`
- `saver_v3.cli.train_rl_ds`

`train_sft_ds` now delegates to `saver_v3.sft.training.run_standard_sft`, so the official v3 SFT path uses `compact_trace_v5` episode-format data and the v3-owned SFT training/runtime stack instead of the broken step-format dataset.

`train_rl_ds` launches the trajectory-level TRL + colocated-vLLM GRPO route. The active RL contract now requires materialized runtime item caches and only accepts message-only rollout supervision with `messages + assistant_supervision + advantage`. During generation, each scored rollout is immediately materialized into a final `episode_spec`; training then consumes only `episode_specs` and prepared batches, without an intermediate feature-layer contract. Episode-level completion-only tensors (`prompt_ids`, `prompt_mask`, `completion_ids`, `completion_mask`, `advantage`, `old_policy_token_log_probs`, multimodal inputs) are derived online from that unified schema. Replay-buffer flags, legacy empty-batch flags, and trace-only active RL payloads are removed and fail fast.

The active verification contract is `next_tool` only. Legacy `recommended_action` payloads and retired wrappers such as `training.py`, `rollout.py`, `saver_v3/sft_training.py`, and `saver_v3/common/training.py` are no longer valid entrypoints.

## Wrapper Commands

```bash
bash scripts/prepare_sft_manifest.sh
bash scripts/train_sft_qwen3_vl_8b_ds8.sh
bash scripts/train_rl_qwen3_vl_8b_ds8.sh
```

## Underlying Launch Shape

```bash
deepspeed \
  --num_nodes 1 \
  --num_gpus 8 \
  --node_rank 0 \
  --master_addr 127.0.0.1 \
  --master_port 29500 \
  --module saver_v3.cli.train_sft_ds \
  --config configs/sft/qwen3_vl_8b_full_train.yaml \
  --model-config configs/model/qwen3_vl_8b_full.yaml \
  --attention-config configs/model/attention_fa3_only.yaml \
  --deepspeed-config configs/deepspeed/zero3_full_model.json
```

## Template Decisions

- `bf16` is the default dtype.
- ZeRO-3 is the default optimizer strategy for the SFT full-model path in this layer; the active RL path uses `configs/deepspeed/zero2_rl.json`.
- `NPROC_PER_NODE` defaults to 8.
- Full-model training means language, vision, and projector parameters stay trainable.
- `flash_attention_3` is hard-required.
- Official SFT consumes `data.prepared_data_path`; legacy `data.train_manifest` is no longer part of the v3 SFT contract.
