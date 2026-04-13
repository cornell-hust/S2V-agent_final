# idea2_v3

`idea2_v3` is the owned Qwen3-VL-8B full-model refactor for SAVER. It keeps
SAVER method semantics aligned with `idea2_v2`, but switches the training and
inference orchestration to a cleaner v3 layout that borrows the local-rank
vLLM and multi-GPU launcher patterns from `TimeSearch-R`.

Current implementation status:

- SFT: delegated to `saver_agent.training.run_standard_sft`, so the official v3
  path now trains on `compact_trace_v2` episode-format prepared data instead of
  the broken lightweight step dataset.
- Prepared-data conversion: raw canonical SAVER JSONL can be converted into
  official `compact_trace_v2` manifests through `saver_v3.cli.prepare_sft_manifest`.
- Policy inference: official v3 inference is now full rollout on raw SAVER data
  through `batch_run_saver_rollout.py`, wrapped by
  `saver_v3.cli.run_policy_rollout_vllm`.
- SFT rollout eval: official evaluation is now delegated to
  `saver_agent.evaluation.run_rollout_evaluation`, which keeps cache,
  proposal-model, scoring, and metric semantics aligned with v2.
- RL: implemented through the TRL + vLLM GRPO route with SAVER-aligned rewards
  and corrected reward-config / judge-default forwarding.

Locked design decisions:

- Policy model: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/qwen3-vl-8b-Instruct`
- Training mode: full-model SFT and full-model RL only
- Training launcher: single-node 8 GPU DeepSpeed
- Attention policy: `flash_attention_3` only, with no FA2 or SDPA fallback
- Inference launcher: `torchrun --nproc_per_node=8` with one local vLLM engine per rank
- Data semantics: `compact_trace_v2` for prepared SFT, raw canonical SAVER JSONL for rollout inference/eval

## Repository Layout

```text
idea2_v3/
|- README.md
|- pyproject.toml
|- configs/
|- docs/
|- envs/
|- scripts/
|- saver_v3/
|- saver_agent/
|- data_utils/
|- tests/
`- third_party_ports/
```

## Quick Start

Prepare official SFT data:

```bash
python -m saver_v3.cli.prepare_sft_manifest \
  --config configs/prepare_sft/qwen3_vl_8b_prepare.yaml
```

Run SFT under DeepSpeed:

```bash
bash scripts/train_sft_qwen3_vl_8b_ds8.sh --run
```

Run policy rollout inference under vLLM:

```bash
bash scripts/run_policy_rollout_vllm.sh --run
```

Run rollout evaluation under vLLM:

```bash
bash scripts/run_sft_rollout_eval_vllm.sh --run
```

## Primary Docs

- Layout and ownership boundaries: `docs/repo_layout.md`
- 8 GPU DeepSpeed training layer: `docs/training_qwen3_vl_8b_ds8.md`
- Local-rank multi-GPU vLLM inference: `docs/inference_vllm.md`
- FA3-only attention contract: `docs/attention_backends.md`

## Attention Backend Policy

This repository fails closed on attention backend selection:

- `flash_attention_3` is the only accepted backend.
- Hopper-class GPUs are required on every visible rank.
- `flash_attn_interface` or `flash_attn` must be importable in the runtime env.
- No FA2 fallback is implemented.
- No SDPA fallback is implemented.

Run `python scripts/check_attention_backend.py` before training or inference.
If it reports `ready: false`, do not start the job.
