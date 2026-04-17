# AgenticVAU — S2V-Agent Project Guide

> **Inherits from:** [`/Users/cornell/Desktop/ssh_worker/AGENTS.md`](../AGENTS.md)
> All server config (SSH, paths, conda, git workflow, task routing) is defined in the parent document. This file covers project-specific flow, evaluation, and experiment plans.

## Project Overview

| Field | Value |
|-------|-------|
| Paper | Search-to-Verify: Agentic Event-Chain Search and Evidence-Faithful Learning for VAU |
| Venue | NeurIPS 2026 |
| Remote path | `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3` |
| Conda env | `qwen3-vl` |
| Policy model | Qwen3-VL-8B (`$DATA_ROOT/Wmh/MLLMs/qwen3-vl-8b-Instruct`) |
| Teacher model | Qwen3-VL-32B (`$DATA_ROOT/Wmh/MLLMs/Qwen3-VL-32B-Instruct`) |
| Hardware | 8×H200, 110 CPU cores, 1.1TB RAM |
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

### Server Access

```bash
# Step 1: SSH to GPU server (Server 2)
ssh -CAXY ws-410ca32fa4aae17e-worker-zvlln.zengweijun+root.ailab-mineru2.pod@h.pjlab.org.cn

# Step 2: Attach to GPU tmux session (may already be active)
tmux a -t 8H200

# Step 3: Activate environment
source ~/.bashrc
conda activate qwen3-vl
cd /mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3
export SAVER_V3_DATA_ROOT=/mnt/shared-storage-user/mineru2-shared/zengweijun
export WANDB_PROJECT=idea2_v3_qwen3_vl_8b
```

### Quick Start: Running Experiments

**无数据预处理变更时**（大多数情况）：直接使用 `run_full_pipeline.sh`，脚本会自动校验缓存、执行 SFT 多 epoch 训练+逐 epoch eval+RL+RL eval。

```bash
EXP_NAME=exp_xxx bash scripts/run_full_pipeline.sh
```

`EXP_NAME` 是实验名称，输出保存至 `artifacts/<EXP_NAME>/`。

**涉及数据预处理修改时**：先按 README 中的 Data Preparation 章节执行数据转换和 SFT manifest 准备，再启动 pipeline。

```bash
# 1. 数据转换（参考 README "Data Preparation" 章节）
python convert_to_saver_agent.py --input ... --adapter ... --mode oracle_sft --output ...

# 2. 准备 SFT manifest
bash scripts/prepare_sft_manifest.sh --run

# 3. 启动 pipeline
EXP_NAME=exp_xxx bash scripts/run_full_pipeline.sh
```

> 完整参数和可选覆盖项见 README。

---

## 1. Experiment Flow

### 1.1 Full Pipeline Overview

```
Data Prep ──→ SFT (5 epochs, per-epoch eval) ──→ RL (GRPO+FECV) ──→ RL Eval
                                                                      │
                                                          ┌───────────┴───────────┐
                                                     Ablations              Sensitivity
                                                    (Table 2,3)            (reward weights)
```

### 1.2 Stage Details

#### Stage 0: Data Preparation

**Purpose:** Convert raw annotations → SFT training format (compact_trace_v2).

```bash
# Convert raw MSAD/ECVA → canonical runtime format
python convert_to_saver_agent.py \
  --input data_utils/msad_saver_with_qwen.jsonl \
  --adapter msad_saver_qwen \
  --mode oracle_sft \
  --output data_utils/msad_saver_sft_train.jsonl

# Prepare SFT manifest (compact_trace_v2)
bash scripts/prepare_sft_manifest.sh --run
```

**Verify:** `data_utils/sft_train.compact_trace_v2.jsonl` and `data_utils/runtime_test.jsonl` exist and are non-empty.

#### Stage 1: SFT Training

**Purpose:** Full-parameter fine-tuning on teacher-rewritten trajectories.

```bash
# Config: configs/sft/qwen3_vl_8b_full_train.yaml
# Key fields to check/modify:
#   data.prepared_data_path → sft_train.compact_trace_v2.jsonl
#   optimization.epochs → 5
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
#   io.data_path → data_utils/runtime_test.jsonl
#   io.output_dir → artifacts/<EXP_NAME>/eval/sft_epoch_N

bash scripts/run_sft_rollout_eval_vllm.sh --run
```

**Output:** `metrics.json` + `semantic_metrics.json` in output directory.

#### Stage 3: RL Training (GRPO + FECV)

**Purpose:** Reinforce evidence-faithful behavior via FECV-grounded rewards.

```bash
# Config: configs/rl/qwen3_vl_8b_grpo_train.yaml
# Key fields:
#   policy_init_from → best SFT checkpoint (from Stage 2)
#   data.train_manifest → data_utils/runtime_train.jsonl (or runtime_test.jsonl for RL data)
#   rewards.reward_version → timesearch_v3

bash scripts/train_rl_qwen3_vl_8b_ds8.sh --run
```

**Reward (timesearch_v3):**
```
R(τ) = 1.0 × R_acc + 0.35 × R_fecv + 0.05 × R_protocol
```

- GRPO group size G=8, lr=5e-7, KL=0.01, T_max=14 turns
- DeepSpeed ZeRO-3, 8×H200

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
    → This is a severe issue — log and pause (see Exception Handling).
```

### 2.4 Claim Verification Matrix

Each experiment should map to at least one paper claim:

| Claim | Tests | Key Metrics | Required Evidence |
|-------|-------|-------------|-------------------|
| **C1:** VAU as agentic event-chain search | Table 1 (vs baselines), Table 2 (event modeling) | Event-Chain F1, Evidence F1@3 | S2V-Agent > all baselines on novel metrics |
| **C2:** Verification-as-action improves evidence quality | Table 3 (w/o verification ablation) | FECV Sufficiency, Protocol Compliance | Full model > w/o verify on FECV |
| **C3:** FECV-grounded RL improves evidence faithfulness | Table 3 (w/o FECV reward ablation) | FECV Sufficiency, Evidence F1@3 | Full model > w/o FECV on faithfulness metrics |

---

## 3. Current Experiment Plan

### Phase 1: Foundation (Data Prep + SFT)

| # | Experiment | Config | Script | Paper Target | Priority |
|---|-----------|--------|--------|-------------|----------|
| 1.1 | Data preparation | `configs/prepare_sft/qwen3_vl_8b_prepare.yaml` | `scripts/prepare_sft_manifest.sh --run` | Prerequisite | P0 |
| 1.2 | SFT training (5 epochs) | `configs/sft/qwen3_vl_8b_full_train.yaml` | `scripts/train_sft_qwen3_vl_8b_ds8.sh --run` | Prerequisite | P0 |
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

## Exception Handling

### Tier 1: Auto-Fix (agent handles autonomously)

| Issue | Action |
|-------|--------|
| OOM during training | Reduce per-device batch size by 50%, retry |
| Single GPU fault | Restart with remaining GPUs (adjust NPROC_PER_NODE) |
| Loss spike (not NaN) | Continue training, monitor for 100 steps |
| vLLM eval OOM | Reduce `max_model_len` or `gpu_memory_utilization` |
| Disk space warning | Clean `/data/tmp/`, old `__pycache__/` |
| Git push failure | Retry with exponential backoff (3 attempts) |
| WandB connection error | Continue training, logs saved locally |

### Tier 2: Pause & Log (requires user review)

| Issue | Action |
|-------|--------|
| Loss NaN persisting > 50 steps | Save state, log to `artifacts/<exp>/ISSUE.md`, pause |
| 3+ consecutive OOM after batch reduction | Log hardware state, pause |
| Data file missing or corrupted | Log error, pause |
| RL metrics regress below SFT on all metrics | Log comparison, pause |
| Checkpoint save fails (disk full) | Log disk state, pause |
| Unknown Python exception during training | Log full traceback, pause |

**Pause format:** Write `artifacts/<exp>/ISSUE.md` with:
```markdown
## Issue: <title>
- **Time:** <timestamp>
- **Experiment:** <exp_name>
- **Phase:** <phase>
- **Error:** <full error message / traceback>
- **Actions Taken:** <what auto-fix was attempted>
- **Recommended Fix:** <agent's diagnosis>
- **Status:** PAUSED — awaiting user review
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
exp(sft): complete 5-epoch SFT training, best at epoch 3
exp(ablation): w/o FECV reward — FECV Sufficiency drops 12 points
exp(sensitivity): reward weight sweep w_fecv={0.0,0.15,0.35,0.50}
```

Commit after: each training completion, each evaluation completion, each report generation.

---

## Cross-Review Workflow (Claude × Codex)

Every experiment step, code change, and design decision must pass cross-model adversarial review.

### Code Review Flow

```
Claude (design/plan) ──→ Codex-pro (implement/challenge) ──→ Claude (review/accept)
```

| Step | Actor | Action | Tool |
|------|-------|--------|------|
| 1. Design | Claude | Draft experiment plan, algorithm design, config changes | — |
| 2. Implement | Codex-pro | Write/modify code based on Claude's spec | Codex MCP (`xhigh` effort) |
| 3. Review | Claude | Verify implementation matches spec, check for bugs/hallucinations | Read + Grep |
| 4. Challenge | Codex-pro | Adversarial review of Claude's reasoning and experiment logic | Codex MCP (`xhigh` effort) |
| 5. Resolve | Claude | Integrate feedback, finalize | — |

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
- Codex calls use `xhigh` effort level for maximum reasoning depth
- If Claude and Codex disagree, log both perspectives and escalate to user
- Applies to: training scripts, config changes, reward function modifications, evaluation logic, data processing

---

## Code Execution Policy

All code modifications can be executed via **two paths**:

| Path | When to Use | How |
|------|-------------|-----|
| **Remote Codex-pro** | Code changes on server (training scripts, configs, data processing) | SSH to server → run Codex-pro in project dir |
| **Local Mac Codex** | Code review, design docs, lightweight edits | Run Codex locally via MCP |

**Remote execution:** Codex-pro runs on the server with full access to the project at `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v3`. It can directly edit, test, and commit code.

**Local execution:** Codex on Mac handles design, review, and coordination. Code changes are pushed via SSH or delegated to remote Codex-pro.

**Selection logic:**
- GPU-dependent code (training, eval) → remote Codex-pro on Server 2
- Dependency installs → remote Codex-pro on Server 1
- Design docs, experiment plans, paper edits → local Mac Codex
- Code review of remote changes → either (shared FS)

---

## Autonomy Policy

- **Full autonomy:** Agent plans and executes experiments without human approval
- **Self-planning:** After each phase, agent reviews results and decides:
  - Whether to proceed to next phase
  - Whether to adjust hyperparameters and re-run
  - Whether ablation results suggest additional experiments
- **Threshold update:** After Phase 1+2, agent recalibrates metric thresholds based on observed baseline performance
- **Experiment plan update:** Agent may add new experiments to this plan if results suggest additional investigations needed for paper claims. New experiments are appended to the relevant phase with priority P3.
