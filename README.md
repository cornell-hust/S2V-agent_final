# S2V-Agent: Search-to-Verify for Video Anomaly Understanding

Agentic event-chain search and evidence-faithful reinforcement learning for Video Anomaly Understanding (VAU). NeurIPS 2026 submission.

**Paper**: `docs/paper_drafts/search_to_verify_neurips_v10.md`

## Architecture

```
saver_v3/
├── core/       — reward, tools, rollout, verification, event_chain, schema
├── model/      — Qwen3-VL-8B loader, vLLM generation, model utilities
├── data/       — dataset, training_data, compact_trace, prepared_schema
├── sft/        — SFT training (teacher-judge rewritten trajectories)
├── rl/         — GRPO RL with FECV reward (timesearch_v3)
├── metrics/    — evaluation, semantic_metrics, offline_scoring
├── inference/  — vLLM rollout, message_runtime, predictor
├── teacher/    — teacher judge (Qwen3-VL-32B)
├── common/     — experiment_logging, runtime, message_budget
├── cli/        — CLI entrypoints (train_sft_ds, train_rl_ds, eval)
└── rollout/    — rollout compatibility layer
```

## Prerequisites

### Hardware
- 8× H200 GPU (110 CPU cores, 1.1TB RAM)
- Flash Attention 3 required (Hopper-class GPUs)

### Software
```bash
# Create conda environment
conda env create -f envs/train-qwen3-vl-8b-full.yml
conda activate idea2_v3

# Or install manually
pip install -e ".[train]"

# Verify FA3
python scripts/check_attention_backend.py
```

### Models
```bash
# Base policy model
MODEL_PATH=/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/qwen3-vl-8b-Instruct

# Teacher judge model
TEACHER_PATH=/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/Qwen3-VL-32B-Instruct

# Proposal model (SigLIP, optional for feature-guided search)
PROPOSAL_PATH=/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/siglip
```

## Quick Start — Full Pipeline

`run_full_pipeline.sh` is the official v3 training entrypoint. It validates runtime caches, auto-resolves or rebuilds the base/teacher prepared SFT files, then runs **Stage 1c materialized cache resolution** to build/validate the offline SFT/runtime caches consumed by SFT, per-epoch rollout eval, and RL. It checks metadata against the active config plus source provenance, runs SFT with per-epoch rollout eval, then launches RL and final RL eval.

```bash
# Base compact-trace SFT branch
EXP_NAME=exp8 bash scripts/run_full_pipeline.sh

# Teacher-judge official branch: auto reuse or auto regenerate
EXP_NAME=exp8 \
TEACHER_JUDGE_MODEL_PATH=/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/Qwen3-VL-32B-Instruct \
PROPOSAL_MODEL_PATH=/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/siglip \
bash scripts/run_full_pipeline.sh
```

If `TEACHER_JUDGE_MODEL_PATH` is set, the pipeline switches SFT to `sft_train.teacher_rollout_primary.compact_trace_v2.jsonl`. It now validates both base prepared and teacher prepared files against the active `preview/prompt/rollout_trace` config and their recorded `source_runtime` / `source_prepared` provenance. Any stale or mismatched file is rebuilt before training.

## Step-by-Step Guide

### Step 0: Environment Setup

```bash
# Source environment (loads proxy, conda, paths)
source ~/.bashrc

# Set environment variables
export SAVER_V3_DATA_ROOT=/mnt/shared-storage-user/mineru2-shared/zengweijun
export WANDB_PROJECT=idea2_v3_qwen3_vl_8b
```

### Step 1: Data Preprocessing

The official v3 file roles are:

- `msad_saver_runtime_train.jsonl`: RL train input and SFT compact-trace source
- `msad_saver_runtime_test.jsonl`: rollout-eval and RL-eval input
- `sft_train.compact_trace_v2.jsonl`: base prepared SFT file
- `sft_train.teacher_rollout_primary.compact_trace_v2.jsonl`: optional teacher-judge prepared SFT file
- `sft_train.compact_trace_v2.materialized_messages_v1.jsonl`: offline materialized SFT messages cache used by the official SFT path
- `msad_saver_runtime_train.materialized_items_v1.jsonl`: offline materialized runtime-items cache used by RL train
- `msad_saver_runtime_test.materialized_items_v1.jsonl`: offline materialized runtime-items cache used by SFT rollout eval and RL eval
- `*.frame_cache` / `*.feature_cache`: required caches for SFT, RL, and rollout evaluation whenever `seek_evidence` is present

#### 1a. Convert canonical annotations to runtime train/test JSONL

```bash
python convert_to_saver_agent.py \
  --input data_utils/msad_saver_with_qwen.jsonl \
  --adapter msad_saver_qwen \
  --mode agent_train \
  --include-splits train \
  --output data_utils/msad_saver_runtime_train.jsonl

python convert_to_saver_agent.py \
  --input data_utils/msad_saver_with_qwen.jsonl \
  --adapter msad_saver_qwen \
  --mode agent_train \
  --include-splits test \
  --output data_utils/msad_saver_runtime_test.jsonl
```

Each runtime row carries the agent-facing supervision used by rollout, reward, and compact-trace preparation, including `structured_target`, `evidence.evidence_moments`, `structured_target.event_chain_target`, `proposal_supervision`, and `stage_supervision`.

#### 1b. Prepare base SFT data (`compact_trace_v2`)

```bash
# Edit configs/prepare_sft/qwen3_vl_8b_prepare.yaml first:
#   saver_config_source: configs/sft/qwen3_vl_8b_full_train.yaml
#   io.input_data_path: /path/to/msad_saver_runtime_train.jsonl
#   io.output_path: /path/to/sft_train.compact_trace_v2.jsonl
#   io.data_root: /path/to/data_root

bash scripts/prepare_sft_manifest.sh --run

# Or directly:
python -m saver_v3.cli.prepare_sft_manifest \
  --config configs/prepare_sft/qwen3_vl_8b_prepare.yaml \
  --override "saver_config_source=configs/sft/qwen3_vl_8b_full_train.yaml"
```

This writes both `sft_train.compact_trace_v2.jsonl` and `sft_train.compact_trace_v2.jsonl.meta.json`. The metadata now records the exact `preview/prompt/rollout_trace` snapshot plus `source_runtime` provenance so stale prepared files fail fast instead of being silently reused.

#### 1c. Materialized offline caches (new official preprocessing stage)

This is now part of the preprocessing pipeline, not an optional runtime optimization. The official pipeline builds these automatically in **Stage 1c**, but you can also materialize them manually.

```bash
python prepare_materialized_cache.py \
  --mode sft \
  --input data_utils/sft_train.compact_trace_v2.jsonl \
  --output data_utils/sft_train.compact_trace_v2.materialized_messages_v1.jsonl \
  --include-splits train \
  --config configs/sft/qwen3_vl_8b_full_train.yaml \
  --model-config configs/model/qwen3_vl_8b_full.yaml \
  --proposal-model-path /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/siglip \
  --proposal-torch-dtype auto

python prepare_materialized_cache.py \
  --mode runtime \
  --input data_utils/msad_saver_runtime_train.jsonl \
  --output data_utils/msad_saver_runtime_train.materialized_items_v1.jsonl \
  --include-splits train \
  --config configs/sft/qwen3_vl_8b_full_train.yaml \
  --data-root /mnt/shared-storage-user/mineru2-shared/zengweijun

python prepare_materialized_cache.py \
  --mode runtime \
  --input data_utils/msad_saver_runtime_test.jsonl \
  --output data_utils/msad_saver_runtime_test.materialized_items_v1.jsonl \
  --include-splits test \
  --config configs/sft/qwen3_vl_8b_full_train.yaml \
  --data-root /mnt/shared-storage-user/mineru2-shared/zengweijun
```

These three files all get their own `.meta.json` sidecars. The metadata is now strict: if the source JSONL, split filter, or config snapshot drifts, consumers fail fast and the official pipeline rebuilds the cache instead of silently reusing stale data.

#### 1c. Optional teacher-judge prepared SFT branch

You can materialize the teacher branch manually, but the official `run_full_pipeline.sh` path now handles it automatically.

```bash
python annotate_teacher_judge_sft.py \
  --input data_utils/sft_train.compact_trace_v2.jsonl \
  --output data_utils/sft_train.teacher_rollout_primary.compact_trace_v2.jsonl \
  --model-path /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/Qwen3-VL-32B-Instruct \
  --proposal-model-path /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/siglip \
  --proposal-torch-dtype auto \
  --include-splits train \
  --input-mode auto \
  --torch-dtype bf16 \
  --device-map auto \
  --attn-implementation flash_attention_3 \
  --max-new-tokens 384 \
  --max-images 8 \
  --batch-size 1
```

```bash
torchrun --standalone --nproc_per_node=8 \
  annotate_teacher_judge_sft.py \
  --input data_utils/sft_train.compact_trace_v2.jsonl \
  --output data_utils/sft_train.teacher_rollout_primary.compact_trace_v2.jsonl \
  --model-path /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/Qwen3-VL-32B-Instruct \
  --proposal-model-path /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/MLLMs/siglip \
  --proposal-torch-dtype auto \
  --include-splits train \
  --input-mode auto \
  --torch-dtype bf16 \
  --device-map auto \
  --attn-implementation flash_attention_3 \
  --max-new-tokens 384 \
  --max-images 8 \
  --topk-frames-per-view 4 \
  --frame-cache-max-cached-videos 64 \
  --batch-size 8
```

#### 1d. Build required caches

Current v3 SFT, RL, and rollout-eval all require video-level `.frame_cache` and `.feature_cache`. Build `frame_cache` before `feature_cache`.

```bash
python build_frame_cache.py \
  --data data_utils/msad_saver_runtime_train.jsonl \
  --data-root /abs/path/to/data_root \
  --include-splits train

python build_frame_cache.py \
  --data data_utils/msad_saver_runtime_test.jsonl \
  --data-root /abs/path/to/data_root \
  --include-splits test

python build_feature_cache.py \
  --data data_utils/msad_saver_runtime_train.jsonl \
  --data-root /abs/path/to/data_root \
  --include-splits train \
  --model-path /abs/path/to/proposal_model

python build_feature_cache.py \
  --data data_utils/msad_saver_runtime_test.jsonl \
  --data-root /abs/path/to/data_root \
  --include-splits test \
  --model-path /abs/path/to/proposal_model
```

#### 1e. Verify required files exist

```bash
ls data_utils/msad_saver_runtime_train.jsonl
ls data_utils/msad_saver_runtime_test.jsonl
ls data_utils/sft_train.compact_trace_v2.jsonl
ls data_utils/sft_train.compact_trace_v2.jsonl.meta.json
```

### Step 2: SFT Training

```bash
# Edit config: configs/sft/qwen3_vl_8b_full_train.yaml
#   data.prepared_data_path: /path/to/sft_train.compact_trace_v2.jsonl
#   optimization.epochs: 5
#   output_dir: artifacts/<EXP_NAME>/sft

# Dry run (print command only):
bash scripts/train_sft_qwen3_vl_8b_ds8.sh

# Execute:
bash scripts/train_sft_qwen3_vl_8b_ds8.sh --run
```

Training behavior:

- The official YAML keeps `optimization.use_sample_weights: true` because VAU supervision mixes heterogeneous anomaly categories, event-chain completeness, and tool-heavy examples; weighting helps prevent easy high-frequency patterns from dominating the loss. You can still disable it by setting `optimization.use_sample_weights: false`.
- v3 SFT now strictly validates `sft_train*.meta.json` against the active `preview`, `prompt`, and `rollout_trace` config, and also checks recorded `source_runtime` / `source_prepared` provenance in hybrid mode. If preprocessing inputs drift after preparation, training aborts instead of silently consuming stale prepared data.
- `run_full_pipeline.sh` automatically switches between the base prepared file and the teacher-prepared file. Manual direct SFT training uses whichever prepared JSONL you point `data.prepared_data_path` at.

### Step 3: SFT Evaluation

```bash
# Edit config: configs/rollout_eval/vllm_qwen3_vl_8b.yaml
#   base_model: /path/to/sft_checkpoint
#   io.data_path: /path/to/msad_saver_runtime_test.jsonl
#   io.data_root: /path/to/data_root
#   proposal.model_path: /path/to/proposal_model
#   io.output_dir: artifacts/<EXP_NAME>/eval/sft

# Dry run:
bash scripts/run_sft_rollout_eval_vllm.sh

# Execute:
bash scripts/run_sft_rollout_eval_vllm.sh --run
```

**Output:** `metrics.json` + `semantic_metrics.json` in the output directory.

### Step 3b: Direct Fixed-Observation Qwen3-VL-8B Baseline

```bash
# Edit config: configs/fixed_baseline_eval/vllm_qwen3_vl_8b_fixed.yaml
#   base_model: /path/to/raw_qwen3_vl_8b_instruct
#   io.data_path: /path/to/msad_saver_runtime_test.jsonl
#   io.data_root: /path/to/data_root
#   io.output_dir: artifacts/<EXP_NAME>/eval/fixed_baseline

# Dry run:
bash scripts/run_fixed_baseline_eval_vllm.sh

# Execute:
bash scripts/run_fixed_baseline_eval_vllm.sh --run
```

This path is **direct one-shot inference**, not agentic rollout. It uses fixed preview frames, strict JSON output, post-hoc scorer adaptation only for metric computation, and direct HF generation under `torchrun` (one full model replica per GPU, KV cache enabled, Flash Attention 3, `batch_size=1`).

**Output:** merged raw predictions, normalized scorer-ready predictions, `metrics.json`, `semantic_metrics.json`, and `table_metrics.json`.

### Step 4: RL Training (GRPO with FECV Reward)

```bash
# Edit config: configs/rl/qwen3_vl_8b_grpo_train.yaml
#   policy_init_from: /path/to/best_sft_checkpoint
#   data.train_manifest: /path/to/msad_saver_runtime_train.jsonl
#   data.eval_manifest: /path/to/msad_saver_runtime_test.jsonl
#   data.materialized_train_items_path: /path/to/msad_saver_runtime_train.materialized_items_v1.jsonl
#   data.materialized_eval_items_path: /path/to/msad_saver_runtime_test.materialized_items_v1.jsonl
#   data.require_materialized_runtime_cache: true
#   data.data_root: /path/to/data_root
#   proposal.model_path: /path/to/proposal_model
#   rewards.reward_version: timesearch_v3

# Dry run:
bash scripts/train_rl_qwen3_vl_8b_ds8.sh

# Execute:
bash scripts/train_rl_qwen3_vl_8b_ds8.sh --run
```

The official active RL path is now pure-pack only: `train_rl_ds` materializes or loads runtime-item caches, converts rollouts into episode tensor packs, and trains on `episode_inputs` with explicit `advantages` and `old_policy_token_log_probs`. Legacy replay-buffer flags, legacy empty-batch flags, and server/client vLLM flags are removed and fail fast.

**Reward (timesearch_v3):**
```
R(τ) = 1.0 × R_acc + 0.35 × R_fecv + 0.05 × R_protocol

R_acc (7 items, 3 families):
  decision:  category              (binary match)
  grounding: temporal_iou          (interval IoU)
  semantic:  trigger_evidence      (LLM judge)
             summary               (LLM judge)
             event_chain × 3       (LLM judge)

R_fecv = 0.5*support + 0.2*trigger_necessity
       + 0.2*negative_resistance + 0.1*parsimony

R_protocol: {-1.0, +0.75, +1.0} (verify-before-finalize)
```

### Step 5: RL Evaluation

Same as Step 3 but pointing to the RL checkpoint:

```bash
bash scripts/run_sft_rollout_eval_vllm.sh --run \
  --override "base_model=/path/to/rl_checkpoint" \
  --override "io.output_dir=artifacts/<EXP_NAME>/eval/rl"
```

## Config Files

| Config | Purpose | Key Fields |
|--------|---------|-----------|
| `configs/sft/qwen3_vl_8b_full_train.yaml` | SFT training | data.prepared_data_path, optimization.epochs |
| `configs/rl/qwen3_vl_8b_grpo_train.yaml` | RL training | policy_init_from, data.materialized_*_items_path, rewards.reward_version |
| `configs/rollout_eval/vllm_qwen3_vl_8b.yaml` | Evaluation | base_model, io.data_path, io.output_dir |
| `configs/prepare_sft/qwen3_vl_8b_prepare.yaml` | Data prep | saver_config_source, io.input_data_path, io.output_path |
| `configs/model/qwen3_vl_8b_full.yaml` | Model config | model_path, torch_dtype |
| `configs/deepspeed/zero3_full_model.json` | DeepSpeed | ZeRO-3 settings |

## 6 Primary Evaluation Metrics

| Metric | Type | What It Measures |
|--------|------|-----------------|
| **Existence Acc.** | Standard | Binary anomaly detection |
| **Temporal mIoU** | Standard | Temporal localization quality |
| **QA Accuracy** | Standard | Per-field semantic understanding |
| **Event-Chain F1** | Novel | Stage-level chain recovery |
| **Evidence F1@3** | Novel | Moment-level evidence retrieval |
| **FECV Sufficiency** | Novel | Counterfactual evidence faithfulness |

Direct fixed-observation baselines should report the first 5 metrics. `FECV Sufficiency` stays `—` in baseline main-table rows because those methods do not execute the verification protocol.

## S2V-Bench Dataset

2,960 videos (MSAD 720 + ECVA 2,240) with structured event-chain annotations:
- 114 anomaly categories
- Precursor → Trigger → Confirmation stage labels
- Evidence moment IDs per stage
- Train/Test: 1,980 / 980

## Key Design Decisions

- **Model**: Qwen3-VL-8B (full-param training), Qwen3-VL-32B (teacher judge)
- **Attention**: Flash Attention 3 only (Hopper GPUs)
- **Training**: DeepSpeed ZeRO-3, 8× H200
- **Reward**: timesearch_v3 (metric-aligned, 7-item R_acc)
- **Verification**: 6-branch counterfactual protocol (full, minimal, drop×3, hard_neg)
- **Visual budget**: K=8 frames per tool call, T_max=14 turns
- **Actions**: scan_timeline, seek_evidence, verify_hypothesis, finalize_case

## Troubleshooting

```bash
# Check FA3 availability
python scripts/check_attention_backend.py

# Check GPU memory (for vLLM colocate mode)
nvidia-smi

# Verify imports
python -c "from saver_v3.core.reward import TIMESARCH_V3_COMPONENT_WEIGHTS; print('OK')"
python -c "from saver_v3.core.rollout import SaverRolloutRunner; print('OK')"
```
