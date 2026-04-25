# AgenticVAU — SEEK Project Guide

> **Inherits from:** [`/Users/cornell/Desktop/ssh_worker/AGENTS.md`](../AGENTS.md)
> All server config (SSH, paths, conda, git workflow, task routing) is defined in the parent document. This file covers project-specific flow, evaluation, and experiment plans.

## Project Overview

| Field | Value |
|-------|-------|
| Paper | SEEK-VAU: Agentic Event-Chain Search and Evidence-Faithful Learning for Video Anomaly Understanding |
| Model short name | SEEK |
| Venue | NeurIPS 2026 |
| Remote path | `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3` |
| Paper drafts | `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3/docs/paper_drafts` |
| Conda env | `qwen3-vl` |
| Policy model | Qwen3-VL-8B (`$DATA_ROOT/Wmh/MLLMs/qwen3-vl-8b-Instruct`) |
| Teacher model | Qwen3-VL-32B (`$DATA_ROOT/Wmh/MLLMs/Qwen3-VL-32B-Instruct`) |
| Hardware | 8×H200 GPU server |
| DATA_ROOT | `/mnt/shared-storage-user/mineru2-shared/zengweijun` |
| Artifacts | `artifacts/<EXP_NAME>/` under project root |
| WandB project | `idea2_v3_qwen3_vl_8b` |

### Key Directories

```
saver_v3/          — Core library (core/, model/, data/, sft/, rl/, metrics/, inference/, cli/)
configs/           — All YAML/JSON configs (sft, rl, rollout_eval, model, deepspeed, prepare_sft)
scripts/           — Bash entrypoints (run_full_pipeline.sh, train_*.sh, run_*.sh)
data_utils/        — Training/test data (.jsonl files)
artifacts/         — Experiment outputs (checkpoints, metrics, reports)
docs/              — Execution documents and task specs
```

### Project File Map

Observed from the remote repo on 2026-04-16. Use this section as the fast path-lookup map when locating code.

- `saver_v3/core/` — method semantics and rollout primitives: reward, environment, rollout, counterfactual verification, event chain logic, prompts, tool registry, semantic answer parsing, self verification, proposal utilities.
- `saver_v3/model/` — Qwen/VLM model loading and generation backends: `qwen_policy.py`, `vllm_generation.py`, `vllm_server.py`, model/runtime helpers.
- `saver_v3/data/` — dataset and preprocessing internals: `dataset.py`, `runtime_items.py`, `materialized_cache.py`, `prepared_schema.py`, `compact_trace.py`, `training_data.py`.
- `saver_v3/sft/` — SFT runtime and training implementation.
- `saver_v3/rl/` — RL runtime and trainers: `timesearch_aligned_grpo_trainer.py`, `trl_grpo_trainer.py`, `grpo_trainer_env.py`, `runtime.py`, `cli_shared.py`.
- `saver_v3/inference/` — rollout inference/eval entrypoints, policy rollout, fixed-baseline eval.
- `saver_v3/metrics/` — evaluation, semantic metrics, offline scoring.
- `saver_v3/teacher/` — teacher judge and teacher-annotation helpers.
- `saver_v3/cli/` — CLI entrypoints such as prepared-data generation.
- `saver_v3/rollout/` — rollout compatibility layer.
- `saver_v3/common/` — shared runtime/logging/budget utilities.
- `saver_v3/tests/` and top-level `tests/` — package-local and repo-level tests.
- `third_party_ports/timesearch_r/` — imported reference code/ports from TimeSearch-R kept inside this repo.

### Config And Script Map

- `configs/model/` — model and attention backend configs.
- `configs/deepspeed/` — DeepSpeed configs for SFT and RL.
- `configs/prepare_sft/qwen3_vl_8b_prepare.yaml` — prepared SFT manifest generation.
- `configs/sft/` — SFT training configs.
- `configs/rl/qwen3_vl_8b_grpo_train.yaml` — main RL config.
- `configs/rollout_eval/` — SFT/RL rollout-eval configs, including `vllm_qwen3_vl_8b.yaml` and `vllm_qwen3_vl_8b_rl_lowmem.yaml`.
- `configs/inference/` — policy inference rollout configs.
- `scripts/run_full_pipeline.sh` — official end-to-end pipeline wrapper.
- `scripts/prepare_sft_manifest.sh` — prepared SFT manifest wrapper.
- `scripts/train_sft_qwen3_vl_8b_ds8.sh` — SFT launcher.
- `scripts/train_rl_qwen3_vl_8b_ds8.sh` — RL launcher.
- `scripts/run_sft_rollout_eval_vllm.sh` — rollout eval launcher.
- `scripts/run_policy_rollout_vllm.sh` and `scripts/run_policy_inference_vllm.sh` — rollout/inference wrappers.
- `scripts/run_fixed_baseline_eval_vllm.sh` — fixed baseline evaluation.

### RL Runtime Note

- As of 2026-04-24, the default RL DeepSpeed config should be `configs/deepspeed/zero3_full_model.json`.
- As of the current RL config, `gradient_accumulation_steps` for the active 8-GPU setup is `4`.
  This preserves the GRPO whole-iteration effective batch: `8 GPUs * per_device_batch_size=1 * GA=4 = num_generations=4 * rollout_count=8 = 32`.
- As of the current RL config, the default throughput-oriented 8-GPU RL shape is `rollout_count=8`, `num_generations=4`, and `dataloader_num_workers=4` with `dataloader_prefetch_factor=2`.
  This keeps one prompt-group per GPU per iteration shard, preserves depth-related budgets, and avoids partial-rank tail imbalance.
- Optional RL memory-pressure fallback: `configs/deepspeed/zero3_offload_rl.json`.
  Use it only when `zero3_full_model.json` still exceeds memory; expect it to be slower than plain ZeRO-3 because parameters and optimizer state are offloaded to CPU.
- As of 2026-04-17, the default RL `keep_recent_tool_image_messages` should be `3`.
  The protected canonical initial scan is retained separately; this budget then prioritizes the latest `scan_timeline` image tool message and uses the remaining slots for the most recent other image-bearing tool messages.
- As of the active v5 contract, training and inference do not use `severity`, `counterfactual_type`, or any FECV / counterfactual-faithfulness semantics.
  Old artifacts should be treated as legacy and rejected by active entrypoints rather than silently reused.

### Data Prep File Map

Primary raw / canonical inputs under `data_utils/`:

- `msad_saver_with_qwen.jsonl` — primary MSAD canonical annotation source.
- `ecva_saver_with_qwen235B.jsonl` — ECVA canonical annotation source.
- `data_utils/splits.py` and `data_utils/video_paths.py` — split and video-path helpers.

Current official preprocessing outputs under `data_utils/`:

- `msad_saver_runtime_train.jsonl` — runtime train input for RL and compact-trace preparation.
- `msad_saver_runtime_test.jsonl` — runtime test input for rollout eval / RL eval.
- `sft_train.compact_trace_v5.jsonl` — base prepared SFT file.
- `sft_train.teacher_rollout_primary.compact_trace_v5.jsonl` — teacher-judge prepared SFT file.
- `sft_train.compact_trace_v5.materialized_messages_v5.jsonl` — offline base `materialized_sft_messages_v5` cache.
- `sft_train.compact_trace_v5.teacher_materialized_messages_v5.jsonl` — offline teacher `materialized_sft_messages_v5` cache used by the default SFT path.
- `msad_saver_runtime_train.materialized_items_v5.jsonl` — offline `materialized_runtime_items_v5` cache for RL train.
- `msad_saver_runtime_test.materialized_items_v5.jsonl` — offline `materialized_runtime_items_v5` cache for rollout eval / RL eval.
- `*.meta.json` and `*.summary.json` sidecars next to the prepared/materialized files — provenance, validation, and summary metadata.
- Active verify payloads must use `next_tool`; legacy `recommended_action` / `verifier_recommended_action` are rejected.
- Active data / cache / eval / RL artifacts must share the same v5 `protocol_signature`: `explicit_first_scan`, main-rollout `finalize_case_only`, verifier `next_tool_only`, `max_turns=10`, and `policy_max_new_tokens=1024`.

### Data Prep Command Ownership

- Runtime-train/test conversion: `convert_to_saver_agent.py`
- SFT prepared manifest generation: `scripts/prepare_sft_manifest.sh` and `python -m saver_v3.cli.prepare_sft_manifest`
- Offline materialized cache generation: `prepare_materialized_cache.py`
- Teacher-judge prepared branch: `annotate_teacher_judge_sft.py`
- Video cache building: `build_frame_cache.py`
- Feature cache building: `build_feature_cache.py`

### Cache And Artifact Locations

- `*.frame_cache` and `*.feature_cache` are required runtime inputs for the current SFT/RL/rollout-eval path.
- These cache files are generated next to the resolved source videos under `DATA_ROOT`, not inside the repo tree.
- Repo-side summaries and metadata stay in `data_utils/`; the heavyweight frame/feature cache payloads live with the videos.

### Experiment Output Map

- `artifacts/<EXP_NAME>/sft/` — SFT checkpoints.
- `artifacts/<EXP_NAME>/rl/` — RL checkpoints.
- `artifacts/<EXP_NAME>/eval/` — rollout eval outputs for SFT/RL.
- `artifacts/<EXP_NAME>/logs/` — pipeline/runtime logs.
- Existing observed runs include `artifacts/exp3/`, `artifacts/exp4/`, `artifacts/liger_smoke_fast/`, and fixed-baseline directories.
- `wandb/` — offline WandB run directories.
- `docs/tasks/` — execution/task docs.
- `docs/paper_drafts/` — paper drafts.
- `refine-logs/EXPERIMENT_PLAN.md` — experiment planning scratch artifact.
- For code analysis or experiment-result analysis, fetch the latest experiment logs from the server-side artifacts tree at `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3/artifacts`.
- The local Mac workspace does not contain authoritative experiment logs for this project; do not treat local files as the log source of truth.
- Artifact protection rule: without explicit user approval, do not delete, overwrite, move, or use bulk sync with deletion semantics against any path under `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3/artifacts`.

### Documentation Sync Rule

- When the server-side code, file layout, training entrypoints, preprocessing flow, or artifact paths change, update the corresponding project documentation in the same round of work.
- Keep this document aligned with the actual implementation on the server at `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3`.
- Do not leave project docs stale after code changes; otherwise later debugging and code navigation will drift from reality and create avoidable misunderstanding.

### TimeSearch-R Reference

- `TimeSearch-R` is the code foundation for developing this paper.
- Reference repository path: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/TimeSearch-R`
- When modifying this project to follow the `TimeSearch-R` direction or style, first reuse its ideas, design patterns, and code where applicable before introducing project-specific alternatives.

### Paper Draft Rule

- Paper drafts live under `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3/docs/paper_drafts`.
- When asked to read the paper, always read the numerically latest draft in that directory, regardless of filename prefix.
- Current latest draft observed on 2026-04-15: `search_to_verify_neurips_v10.md` (legacy filename; current paper title is SEEK-VAU).

### Server Access

```bash
# Step 1: SSH to GPU server (Server 2)
ssh -CAXY ws-410ca32fa4aae17e-worker-6qqv4.zengweijun+root.ailab-mineru2.pod@h.pjlab.org.cn

# Step 2: Attach to an existing GPU tmux session if needed
tmux ls
tmux a -t <session-name>

# Step 3: Activate environment
source ~/.bashrc
conda activate qwen3-vl
cd /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3
export SAVER_V3_DATA_ROOT=/mnt/shared-storage-user/mineru2-shared/zengweijun
export WANDB_PROJECT=idea2_v3_qwen3_vl_8b
```

- Agent-started experiment rule: when an agent launches experiments autonomously, always do so on the GPU server via `ssh -CAXY ws-410ca32fa4aae17e-worker-6qqv4.zengweijun+root.ailab-mineru2.pod@h.pjlab.org.cn`.
- Do not start experiments on non-GPU servers or local machines when acting autonomously.
- Agent-started experiment rule: when an agent launches experiments autonomously, always use the tmux session named `agent`.
- If `agent` already exists, attach with `tmux a -t agent`.
- If `agent` does not exist, create it with `tmux new -s agent` and run the experiment there.
- Do not create ad hoc tmux session names for self-started experiments unless the user explicitly requests a different session.

### Remote Code Reading Rule

- When reading or modifying server-side code over SSH, read the target implementation as completely as practical before drawing conclusions.
- Do not inspect only a few lines around a symbol and then infer behavior for the whole path; follow the full control flow across the relevant functions, configs, and callsites first.
- If the issue spans multiple stages, read the entire affected chain end to end before proposing a diagnosis or fix.
- When analyzing code behavior or experiment results, use the latest server-side logs under `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3/artifacts`; the local Mac copy has no experiment logs and is not sufficient for runtime analysis.
- For code reading, code review, debugging, and profiling in this project, default to **multiple Codex read-only investigation workers** before answering or proposing changes.
- Preferred split for `idea2_v3`:
  - **Server 1 worker:** read code, configs, docs, offline artifacts, and historical logs from `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3`.
  - **Server 2 worker:** inspect live training/eval state via `tmux`, `ps`, `nvidia-smi`, `nvidia-smi dmon`, `pidstat`, and currently growing logs under `artifacts/<EXP_NAME>/logs/`.
- If one server cannot read shared files stably, switch to the other server and continue until the relevant code path and logs are read in full.
- Do **not** answer code-reading or code-review questions from partial snippets just because one SSH connection is unstable; use shared filesystem failover and keep reading.

### Shared Filesystem Rule

- Server 1 and Server 2 share the same filesystem under `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh`.
- If SSH to one server fails or is unstable, it is valid to switch to the other server and continue working on the same project files.
- Persistent reconnect rule: if Server 1 cannot be reached or disconnects, immediately try Server 2; if Server 2 cannot be reached or disconnects, immediately try Server 1.
- Keep alternating and retrying between the two servers until the task is complete. Do not give up on server access after a single failure or a short burst of disconnections.
- Frequent disconnects are expected in this project. Reconnect repeatedly, switch targets often when needed, and continue failover attempts instead of abandoning the task.
- Any code/document/data path under `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3` should be treated as shared across both servers unless proven otherwise.
- Stable shared-filesystem access takes priority over attachment to a specific machine; prefer whichever server currently gives complete reads of code and logs.

### Local Mirror Sync Rule

- High-priority rule: the local Mac source mirror at `/Users/cornell/Desktop/ssh_worker/AgenticVAU/code` is the primary code-reading location for this project and the default code-editing location for planning and normal edits.
- Before modifying project code, first read the corresponding file from the local mirror under `/Users/cornell/Desktop/ssh_worker/AgenticVAU/code`.
- Default edit path: modify the local mirror first, then immediately upload the modified file to the matching server-side project path under `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3`.
- Stable-connection exception: if the SSH/session is stable and direct server-side editing is more practical, it is valid to edit the server copy first, but immediately after that change you must download the modified file back into the matching local mirror path.
- When uploading to or downloading from the server project, obey that project's `.gitignore` rules. Do not sync files that are intentionally ignored on the server side.
- Never use `rsync --delete` or any equivalent bulk-delete sync against the server project root or its `artifacts/` subtree unless the user explicitly approves that deletion scope.
- Keep the local mirror and the server copy aligned at all times. Do not leave local-only edits or server-only edits pending across turns.
- If a server-side file changes first for any reason, sync that change back into the local mirror before making further edits so the two trees remain identical.
- When reporting work, assume `/Users/cornell/Desktop/ssh_worker/AgenticVAU/code` is the primary inspection and planning surface, while `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3` is the execution-side copy that must stay synchronized with local.

### Quick Start: Running Experiments

**无数据预处理变更时**（大多数情况）：直接使用 `run_full_pipeline.sh`，脚本会自动校验缓存、执行 SFT 3 epoch 训练+逐 epoch eval+RL+RL eval。

```bash
EXP_NAME=exp_xxx bash scripts/run_full_pipeline.sh
```

`EXP_NAME` 是实验名称，输出保存至 `artifacts/<EXP_NAME>/`。

**涉及数据预处理修改时**：先执行下文 `Stage 0: Data Preparation` 的命令，再启动同一条 pipeline。

> 完整参数和可选覆盖项见 README。

---

## 1. Experiment Flow

### 1.1 Full Pipeline Overview

```
Data Prep ──→ SFT (3 epochs, per-epoch eval) ──→ RL (GRPO+FECV) ──→ RL Eval
                                                                      │
                                                          ┌───────────┴───────────┐
                                                     Ablations              Sensitivity
                                                    (Table 2,3)            (reward weights)
```

### 1.2 Stage Details

#### Stage 0: Data Preparation

**Purpose:** Convert raw annotations → SFT training format (`compact_trace_v5`).

```bash
# Convert raw MSAD/ECVA → canonical runtime format
python convert_to_saver_agent.py \
  --input data_utils/msad_saver_with_qwen.jsonl \
  --adapter msad_saver_qwen \
  --mode oracle_sft \
  --output data_utils/msad_saver_sft_train.jsonl

# Prepare SFT manifest (compact_trace_v5)
bash scripts/prepare_sft_manifest.sh --run
```

**Verify:** `data_utils/sft_train.compact_trace_v5.jsonl` and `data_utils/msad_saver_runtime_test.jsonl` exist and are non-empty.

#### Stage 1: SFT Training

**Purpose:** Full-parameter fine-tuning on teacher-rewritten trajectories.

```bash
# Config: configs/sft/qwen3_vl_8b_full_train.yaml
# Key fields to check/modify:
#   data.prepared_data_path → sft_train.compact_trace_v5.jsonl
#   optimization.epochs → 3
#   output_dir → artifacts/<EXP_NAME>/sft

bash scripts/train_sft_qwen3_vl_8b_ds8.sh --run
```

- DeepSpeed ZeRO-3, 8×H200
- Assistant-turn-only loss masking
- Per-epoch checkpoint saved
- **After each epoch:** run SFT eval (Stage 2) to pick best checkpoint

#### Stage 2: SFT Evaluation

**Purpose:** Evaluate SFT checkpoint on test set via vLLM rollout.

```bash
# Config: configs/rollout_eval/vllm_qwen3_vl_8b.yaml
# Key fields:
#   base_model → path to SFT checkpoint
#   io.data_path → data_utils/msad_saver_runtime_test.jsonl
#   io.output_dir → artifacts/<EXP_NAME>/eval/sft_epoch_N

bash scripts/run_sft_rollout_eval_vllm.sh --run
```

**Output:** `metrics.json` + `semantic_metrics.json` in output directory.

#### Stage 3: RL Training (GRPO, stage-verified reward)

**Purpose:** Reinforce stage-complete, protocol-consistent behavior via the active `timesearch_v4` reward.

```bash
# Config: configs/rl/qwen3_vl_8b_grpo_train.yaml
# Key fields:
#   policy_init_from → best SFT checkpoint (from Stage 2)
#   data.train_manifest → data_utils/msad_saver_runtime_train.jsonl
#   rewards.reward_version → timesearch_v4

bash scripts/train_rl_qwen3_vl_8b_ds8.sh --run
```

#### Stage 4: RL Evaluation

Same as Stage 2 but pointing to RL checkpoint:

```bash
bash scripts/run_sft_rollout_eval_vllm.sh --run \
  --override "base_model=<RL_CHECKPOINT_PATH>" \
  --override "io.output_dir=artifacts/<EXP_NAME>/eval/rl"
```

### 1.3 Script Interaction Rules

| Scenario | Action |
|----------|--------|
| Standard pipeline (SFT, RL, Eval) | Use existing `scripts/*.sh --run` |
| New ablation variant | Create new YAML config in `configs/`, use existing script with `--override` |
| New experiment type not covered by scripts | Create new script in `scripts/`, follow existing naming convention |
| Config modification | Edit YAML directly, document changes in experiment report |

### 1.4 Checkpoint Management

- Save all epoch checkpoints during SFT (for best-epoch selection)
- After best epoch identified, **keep**: best SFT checkpoint, final RL checkpoint
- **Do not delete** any checkpoint without explicit evaluation comparison
- Checkpoint naming: `artifacts/<EXP_NAME>/<stage>/checkpoint-<step>/`

---

## 2. Evaluation Standards

### 2.1 Primary Metrics (6 metrics)

| # | Metric | Type | What It Measures | Target Threshold | Auto-Retune If Below |
|---|--------|------|-----------------|-----------------|---------------------|
| 1 | **Existence Acc.** | Standard | Binary anomaly detection | ≥ 80% | Adjust SFT data balance or RL reward |
| 2 | **Temporal mIoU** | Standard | Temporal localization quality | ≥ 0.35 | Check scan_timeline behavior |
| 3 | **QA Accuracy** | Standard | Per-field semantic understanding | ≥ 0.55 | Review teacher judge quality |
| 4 | **Event-Chain F1** | Novel | Stage-level chain recovery | ≥ 0.45 | Check event-chain completeness target |
| 5 | **Evidence F1@3** | Novel | Moment-level evidence retrieval | ≥ 0.35 | Review seek_evidence tool usage |
| 6 | **FECV Sufficiency** | Novel | Counterfactual evidence faithfulness | ≥ 0.45 | Increase w_fecv or review verification |

> **Note:** These are initial targets based on task difficulty estimates. After the first full pipeline run, update thresholds to reflect actual baseline performance. The key requirement is **RL > SFT on all 6 metrics** with meaningful improvement.

### 2.2 Secondary Metrics (for supplementary)

- Category Macro-F1
- Precursor mIoU
- ROUGE-L
- Evidence Precision / Recall
- Protocol Compliance (% of verify-before-finalize ordering)
- Verify-Finalize Followthrough
- Mean Inspected Clip Ratio
- Mean Turns

### 2.3 Evaluation Decision Logic

```
IF all 6 metrics ≥ threshold:
    → PASS. Proceed to next experiment phase.
IF any metric < threshold AND this is the first run:
    → UPDATE thresholds based on observed difficulty. Log reasoning.
IF any metric < threshold AND thresholds already calibrated:
    → RETUNE. Check the "Auto-Retune If Below" column for diagnosis direction.
    → Max 2 retune attempts per experiment before moving on and logging the gap.
IF RL metrics < SFT metrics on any primary metric:
    → CRITICAL. Investigate reward function, learning rate, KL coefficient.
    → This is a severe issue — log and pause for review.
```

### 2.4 Claim Verification Matrix

Each experiment should map to at least one paper claim:

| Claim | Tests | Key Metrics | Required Evidence |
|-------|-------|-------------|-------------------|
| **C1:** VAU as agentic event-chain search | Table 1 (vs baselines), Table 2 (event modeling) | Event-Chain F1, Evidence F1@3 | SEEK > all baselines on novel metrics |
| **C2:** Verification-as-action improves evidence quality | Table 3 (w/o verification ablation) | FECV Sufficiency, Protocol Compliance | Full model > w/o verify on FECV |
| **C3:** FECV-grounded RL improves evidence faithfulness | Table 3 (w/o FECV reward ablation) | FECV Sufficiency, Evidence F1@3 | Full model > w/o FECV on faithfulness metrics |

---

## 3. Current Experiment Plan

### Phase 1: Foundation (Data Prep + SFT)

| # | Experiment | Config | Script | Paper Target | Priority |
|---|-----------|--------|--------|-------------|----------|
| 1.1 | Data preparation | `configs/prepare_sft/qwen3_vl_8b_prepare.yaml` | `scripts/prepare_sft_manifest.sh --run` | Prerequisite | P0 |
| 1.2 | SFT training (3 epochs) | `configs/sft/qwen3_vl_8b_full_train.yaml` | `scripts/train_sft_qwen3_vl_8b_ds8.sh --run` | Prerequisite | P0 |
| 1.3 | SFT eval (per-epoch, pick best) | `configs/rollout_eval/vllm_qwen3_vl_8b.yaml` | `scripts/run_sft_rollout_eval_vllm.sh --run` | Partial Table 1 | P0 |

### Phase 2: Core RL

| # | Experiment | Config | Script | Paper Target | Priority |
|---|-----------|--------|--------|-------------|----------|
| 2.1 | RL training (GRPO+FECV, from best SFT) | `configs/rl/qwen3_vl_8b_grpo_train.yaml` | `scripts/train_rl_qwen3_vl_8b_ds8.sh --run` | Table 1 (ours) | P0 |
| 2.2 | RL eval on test set | `configs/rollout_eval/vllm_qwen3_vl_8b.yaml` | `scripts/run_sft_rollout_eval_vllm.sh --run` | Table 1 (ours) | P0 |

### Phase 3: Method Ablations (Table 3)

| # | Variant | Modification | Paper Target | Priority |
|---|---------|-------------|-------------|----------|
| 3.1 | w/o active search | Disable `scan_timeline` + `seek_evidence`, fixed observation | Table 3 | P1 |
| 3.2 | w/o event-chain completeness target | Remove stage coverage from reward | Table 3 | P1 |
| 3.3 | w/o policy-internal verification | Remove `verify_hypothesis` action | Table 3 | P1 |
| 3.4 | w/o FECV reward | Set w_fecv=0.0 | Table 3 | P1 |
| 3.5 | w/o optional local routing | Remove auxiliary local signals | Table 3 | P1 |
| 3.6 | Verify as postprocessing | Remove verify action, apply post-hoc | Table 3 | P1 |

**Implementation:** For each ablation, agent creates a new YAML config (e.g., `configs/rl/ablation_no_fecv.yaml`) modifying the relevant parameters from the base RL config. Runs RL training + eval for each.

### Phase 4: Event-Chain Ablation (Table 2)

| # | Variant | Modification | Paper Target | Priority |
|---|---------|-------------|-------------|----------|
| 4.1 | Trigger-only | S_y = {trg} only, ignore precursor/confirmation | Table 2 | P1 |
| 4.2 | Precursor + Trigger | S_y = {pre, trg}, ignore confirmation | Table 2 | P1 |
| 4.3 | Full chain (reuse Phase 2 results) | S_y = {pre, trg, conf} | Table 2 | — |

### Phase 5: Reward Weight Sensitivity

| # | Variant | w_fecv | Paper Target | Priority |
|---|---------|--------|-------------|----------|
| 5.1 | FECV weight = 0.0 | 0.0 | Sensitivity analysis | P2 |
| 5.2 | FECV weight = 0.15 | 0.15 | Sensitivity analysis | P2 |
| 5.3 | FECV weight = 0.35 (default, reuse Phase 2) | 0.35 | Sensitivity analysis | — |
| 5.4 | FECV weight = 0.50 | 0.50 | Sensitivity analysis | P2 |

### Phase 6: Qualitative & Analysis

| # | Task | Paper Target | Priority |
|---|------|-------------|----------|
| 6.1 | Select 3+ qualitative cases from RL eval outputs | Qualitative Studies | P2 |
| 6.2 | Self-consistency validation (self vs oracle sufficiency correlation) | Section 5.4 | P2 |
| 6.3 | Generate summary comparison tables across all experiments | All Tables | P2 |

### Phase Dependencies

```
Phase 1 (P0) ──→ Phase 2 (P0) ──→ Phase 3 (P1) ──→ Phase 6 (P2)
                                ──→ Phase 4 (P1)
                                ──→ Phase 5 (P2)
```

- P0 runs must complete sequentially (each depends on prior output)
- P1 ablations can run in parallel once Phase 2 is complete
- P2 tasks can run after their dependencies finish

### Experiment Naming Convention

```
artifacts/
├── exp_main/              — Phase 1+2: main pipeline
│   ├── sft/               — SFT checkpoints
│   ├── eval/
│   │   ├── sft_epoch_1/   — Per-epoch SFT eval
│   │   ├── sft_epoch_2/
│   │   ├── sft_epoch_3/
│   │   ├── sft_best/      — Best SFT eval
│   │   └── rl/            — RL eval
│   └── rl/                — RL checkpoints
├── abl_no_search/         — Phase 3.1
├── abl_no_eventchain/     — Phase 3.2
├── abl_no_verify/         — Phase 3.3
├── abl_no_fecv/           — Phase 3.4 (also reused as Phase 5.1)
├── abl_no_local/          — Phase 3.5
├── abl_verify_postproc/   — Phase 3.6
├── abl_trigger_only/      — Phase 4.1
├── abl_pre_trigger/       — Phase 4.2
├── sens_fecv_015/         — Phase 5.2
└── sens_fecv_050/         — Phase 5.4
```

---

## Result Output Format

### Per-Experiment Report

Each experiment produces `artifacts/<EXP_NAME>/report.md`:

```markdown
# Experiment Report: <EXP_NAME>

## Config
- Base config: <path>
- Overrides: <list of changes>
- Hypothesis: <what this experiment tests>
- Paper target: <Table N, row M>

## Training Summary
- Duration: <hours>
- Final loss: <value>
- Best checkpoint: <path>

## Metrics
| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Existence Acc. | X% | ≥80% | PASS/FAIL |
| Temporal mIoU | X | ≥0.35 | PASS/FAIL |
| QA Accuracy | X | ≥0.55 | PASS/FAIL |
| Event-Chain F1 | X | ≥0.45 | PASS/FAIL |
| Evidence F1@3 | X | ≥0.35 | PASS/FAIL |
| FECV Sufficiency | X | ≥0.45 | PASS/FAIL |

## Comparison
<vs baseline or vs full model, depending on experiment type>

## Conclusion
<1-3 sentences: what this experiment shows for the paper claim>
```

### Summary Table

After all experiments in a phase complete, generate `artifacts/summary_table_<phase>.md` with a unified comparison table matching the paper's table format.

### Git Commit Convention

```
exp(<phase>): <brief description>

Example:
exp(sft): complete 3-epoch SFT training, best at epoch 3
exp(ablation): w/o FECV reward — FECV Sufficiency drops 12 points
exp(sensitivity): reward weight sweep w_fecv={0.0,0.15,0.35,0.50}
```

Commit after: each training completion, each evaluation completion, each report generation.

---

## Cross-Review Workflow (Claude × Codex)

Every experiment step, code change, and design decision must pass cross-model adversarial review.

### Metrics-Driven Improvement Loop（指标驱动的对抗优化闭环）

实验完成后，由 Claude 分析指标并提出改进方案，再调用 Codex 进行对抗性审查，多轮迭代后确定最优方案，最后由多个 Codex 并行落实代码修改。

```
┌─────────────────────────────────────────────────────────────┐
│ Phase A: 对抗性方案设计（2-3 轮）                              │
│                                                             │
│  Claude 分析指标  ──→  Claude 提出改进方案                      │
│       ↑                     │                               │
│       │                     ↓                               │
│  Claude 综合判断  ←──  Codex 理性分析/挑战方案                   │
│  (接受/修改/否决)          (找漏洞、质疑假设、提出替代方案)         │
│                                                             │
│  重复 2-3 轮直到双方达成一致 → 输出最终方案                       │
└─────────────────────────────────────────────────────────────┘
                          │
                          ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase B: 多 Codex 并行实现                                    │
│                                                             │
│  最终方案拆分为独立子任务                                       │
│       │                                                     │
│       ├──→ Codex Agent 1: 修改 reward function               │
│       ├──→ Codex Agent 2: 修改 training config               │
│       └──→ Codex Agent 3: 修改 evaluation logic              │
│                                                             │
│  Claude 汇总审查所有修改 → 提交 → 启动新实验                     │
└─────────────────────────────────────────────────────────────┘
```

**对抗分析规则：**
- Claude 提出方案时必须附带具体指标证据和预期改进幅度
- Codex 审查时必须从反面论证：方案可能失败的原因、被忽略的副作用、更简单的替代方案
- 每轮对抗必须有明确结论：接受/修改/否决，不允许模糊收场
- 达成一致后方案写入 `docs/tasks/` 作为执行文档

### General Rules

- No code merged without Codex-pro implementation or review pass
- If Claude and Codex disagree, log both perspectives and escalate to user
- When launching multiple child Codex agents in parallel, default every subagent to `gpt-5.4` with `xhigh` reasoning effort.

### Default Skill Bundles

When user asks to review, read, inspect, or answer questions about code, default to exactly one of these three scenarios, using only 1-2 skills per turn unless the user explicitly asks for more:

- **Read / understand code logic:** `smart-explore` + `systematic-debugging`
- **Code review / code audit:** `caveman-review` + `systematic-debugging`
- **Question answering / performance analysis / optimization discussion:** `caveman` + `system-profile`

If an exact skill name is unavailable in the current harness, use the closest equivalent workflow and state the substitution once.

---

## Autonomy Policy

- Agent may autonomously plan and execute experiments.
- Agent may append new P3 experiments when current results are insufficient to support the paper claims.
