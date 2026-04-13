# Experiment Plan: S2V-Agent SFT Phase — Beat Fixed Baseline

**Problem**: S2V-Agent SFT model underperforms fixed-observation baseline on existence_accuracy (0.675 vs 0.846) and temporal_miou (0.337 vs 0.392), while outperforming on 5 other metrics.
**Method Thesis**: Agentic event-chain search with evidence-faithful verification outperforms fixed-observation VAU systems.
**Date**: 2026-04-14

## Claim Map
| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|----------------------------|---------------|
| C1: Agentic search > fixed observation | Core paper contribution | SFT model beats baseline on ALL 6 primary metrics | B1 |
| C2: Sample weighting fixes anomaly bias | Enables C1 by fixing existence_accuracy | exp2 existence_acc >= 0.846 | B2 |
| C3: Hard-normal augmentation fixes behavioral shortcut | Root cause fix for over-prediction | exp3 existence_acc >= 0.85 + other metrics maintained | B3 |

## Experiment Blocks

### Block 1 (exp1): Baseline SFT — DONE
- Status: Completed epochs 1-3 (epoch 4+ killed by tmux crash)
- Best: epoch 3 — 5/7 metrics beat baseline, existence_acc=0.675, temporal_miou=0.337

### Block 2 (exp2): Sample-Weighted SFT — RUNNING
- Claim tested: C2
- Changes: sample_weight (normal=1.0, anomaly=0.6), 8 epochs, warmup=0.10
- Success criterion: existence_acc >= 0.846, temporal_miou >= 0.392, other metrics maintained
- Failure interpretation: behavioral asymmetry (not just gradient imbalance) is the root cause -> proceed to exp3

### Block 3 (exp3): Hard-Normal Augmentation — PLANNED
- Claim tested: C3
- Changes: +120 hard-normal trajectories with scene-specific queries, 600 total samples
- Success criterion: ALL metrics beat baseline
- Failure interpretation: 480-600 samples fundamentally insufficient, need data scaling or RL

## Run Order and Milestones
| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|-----------|------|------|---------------|------|------|
| M1 | Diagnose exp1 | exp1 ep1-3 | 5/7 beat baseline | 3h | Low |
| M2 | Fix existence_acc | exp2 8 epochs | exist_acc >= 0.846? | 5h | Medium: may not be sufficient |
| M3 | Fix behavioral bias | exp3 8 epochs | ALL metrics beat baseline? | 5h | Low: addresses root cause |
| M4 | Proceed to RL | Best SFT checkpoint | ALL metrics > baseline | 0h | Gate |

## Compute Budget
- Per experiment: ~5 GPU-hours on 8xH200 (training ~15min + eval ~4h)
- Total SFT phase: ~15 GPU-hours (3 experiments)
- Biggest bottleneck: per-epoch eval (240 videos x 8 epochs = ~30 min/eval)
