# SFT Improvement Plan v2 (Post Adversarial Review)

**Date:** 2026-04-14
**Goal:** Exceed fixed_baseline on ALL primary metrics
**Status:** Waiting for exp1 epoch 4-5 results as diagnostic baseline

## Revised Changes (after architect adversarial review)

### Change 1: Class-conditional sample weights [MODIFIED]
- Normal samples: weight=1.0
- Anomaly samples: weight=0.6
- Rationale: Anomaly trajectories dominate gradient via longer sequences. Direct class weighting is simpler and more targeted than token-length normalization.
- Implementation: New weighted JSONL file, pipeline override to use it.

### Change 2: Cosine LR with peak 1e-5 [MODIFIED from flat 3e-6]
- Peak LR: 1e-5 (same as current), decay to 1e-6 over 8 epochs
- Warmup: 10% of total steps
- Rationale: Cosine schedule explores faster initially, fine-tunes later. Strictly better than flat low LR.

### Change 3: Keep effective batch at 64 [REVERSED from reducing to 16]
- gradient_accumulation_steps=8 (unchanged)
- Rationale: With 480 samples, batch=16 gives only ~4 anomaly samples per batch. Too noisy.

### Change 4: 8 epochs with per-epoch eval [ACCEPTED]
- Per-epoch existence_accuracy tracking is critical diagnostic
- Per-epoch eval finds best checkpoint

### Change 5: Keep max_grad_norm=1.0 [REVERSED from relaxing to 2.0]
- Rationale: No evidence of clipping issues. Standard safe value.

### Change 6 (NEW): Hard-normal trajectory augmentation [HIGH PRIORITY]
- Generate 60-120 trajectories on normal videos with full search behavior
- Agent searches, finds benign content, verifies, concludes normal
- Breaks the shortcut: finding content != anomaly
- Requires separate data generation pass before exp2

## Experiment Config (exp2)
EXP_NAME=exp2
Data: sft_train.compact_trace_v2.weighted.jsonl (with sample weights)
optimization.epochs=8
optimization.warmup_ratio=0.10
LR schedule: cosine 1e-5 to 1e-6 (handled by lr_scheduler_type=cosine + epochs)
gradient_accumulation_steps=8 (unchanged)
max_grad_norm=1.0 (unchanged)
use_sample_weights=true

## Decision Gates
1. After exp1 epoch 5 completes: analyze existence_accuracy trend
2. If existence_accuracy is declining epoch-over-epoch: confirms behavioral bias, proceed with hard-normal augmentation
3. If existence_accuracy plateaus or improves: sample weights may be sufficient, skip hard-normal for exp2
4. After exp2 epoch 4: compare vs baseline, decide if hard-normal (exp3) needed

## Acceptance Criteria
- existence_accuracy >= 0.846
- temporal_miou >= 0.392
- All other metrics maintain or improve from exp1-epoch3 levels
