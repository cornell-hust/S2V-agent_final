# SFT Improvement Plan v1: Beat Fixed Baseline

**Date:** 2026-04-14
**Goal:** Exceed fixed_baseline on ALL 6 primary metrics, especially existence_accuracy and temporal_miou
**Current status:** 5/7 metrics already surpass baseline; existence_acc (-17.1pts) and temporal_miou (-5.5pts) lag

## Current Metrics Comparison (Epoch 3 vs Baseline)

| Metric | Baseline | Epoch 3 | Gap |
|--------|----------|---------|-----|
| existence_accuracy | 0.846 | 0.675 | -0.171 ❌ |
| temporal_miou | 0.392 | 0.337 | -0.055 ❌ |
| category_macro_f1 | 0.229 | 0.428 | +0.199 ✅ |
| temporal_r1_at_0_3 | 0.275 | 0.575 | +0.300 ✅ |
| event_chain_f1 | 0.289 | 0.553 | +0.264 ✅ |
| evidence_f1_at_3 | 0.168 | 0.321 | +0.153 ✅ |

## Root Cause Analysis

### 1. Existence Accuracy Gap (0.675 vs 0.846)
- Training data: 360 normal (75%), 120 anomaly (25%). Test: 50/50.
- Despite 3x more normal training examples, model over-predicts anomaly (63.8% of test predictions).
- Root cause: anomaly trajectories are longer (more tool calls, evidence seeking, verification) → SFT loss signal per-sample is much stronger for anomaly examples.
- 46/66 normal false-positives are labeled assault — category hallucination.
-  in config but no weights in data → feature is NON-FUNCTIONAL.

### 2. Temporal mIoU Gap (0.337 vs 0.392)
- Temporal grounding quality is close to baseline but slightly below.
- Model does well on temporal_r1_at_0.3 (0.575 vs 0.275) — coarse localization is excellent.
- But mIoU penalizes imprecise boundaries more heavily.
- Likely: model finds the right temporal region but interval boundaries are too loose.

## Proposed Changes (exp2)

### Change 1: Add token-length-normalized sample weights
**Rationale:** Anomaly trajectories have ~3-5x more tokens than normal trajectories. With uniform weighting, anomaly samples dominate the total loss. Normalizing by token count should balance the effective learning signal.

**Implementation:**
- Compute each sample's tokenized sequence length
- Set  (sqrt to soften the correction)
- Write weights into the JSONL data file
-  already in config → will auto-activate

### Change 2: Lower learning rate from 1e-5 to 3e-6
**Rationale:** With only 480 samples and 7.5 steps/epoch, the model overfits quickly at 1e-5. Vad-R1 uses 1e-6 for similar small-data SFT. We use 3e-6 as a middle ground.

### Change 3: Reduce gradient_accumulation_steps from 8 to 2
**Rationale:** Effective batch drops from 64 to 16. With 480 samples, this gives 30 steps/epoch instead of 7.5 — 4x more gradient updates per epoch for finer convergence.

### Change 4: Increase epochs from 3 to 8
**Rationale:** With lower LR and smaller effective batch, we need more epochs. 8 epochs × 30 steps = 240 total steps vs current 3 × 7.5 = 22.5 steps. Per-epoch eval will find the best checkpoint.

### Change 5: Relax max_grad_norm from 1.0 to 2.0
**Rationale:** More conservative clipping can slow convergence on small noisy datasets. Vad-R1 uses 5.0. We use 2.0 as moderate relaxation.

## Expected Impact
- Change 1 → directly addresses anomaly bias → existence_accuracy improvement
- Changes 2-4 → better convergence → all metrics improvement
- Change 5 → faster convergence, modest improvement

## Experiment Config


## Acceptance Criteria
- existence_accuracy ≥ 0.846 (match baseline)
- temporal_miou ≥ 0.392 (match baseline)
- All other metrics maintain or improve from epoch 3 levels
