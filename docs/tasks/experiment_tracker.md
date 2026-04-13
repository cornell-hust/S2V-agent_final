# SFT Improvement Experiment Tracker

## Baseline Reference
- Fixed baseline: artifacts/fixed_baseline/msad_test/metrics.json
- Key targets: existence_acc>=0.846, temporal_miou>=0.392

## Experiments

| Run | Status | Changes | exist_acc | cat_f1 | temp_miou | r1@0.3 | chain_f1 | evid_f1 | Notes |
|-----|--------|---------|-----------|--------|-----------|--------|----------|---------|-------|
| Baseline | DONE | Fixed-observation Qwen3-VL-8B | 0.846 | 0.229 | 0.392 | 0.275 | 0.289 | 0.168 | Reference |
| exp1-ep2 | DONE | Base SFT, 5ep, lr=1e-5, batch=64 | 0.596 | 0.188 | 0.078 | 0.625 | 0.560 | 0.258 | Early epoch |
| exp1-ep3 | DONE | Same | 0.675 | 0.428 | 0.337 | 0.575 | 0.553 | 0.321 | Best exp1, 5/7 beat baseline |
| exp1-ep4+ | KILLED | Same | - | - | - | - | - | - | tmux session died |
| exp2 | RUNNING | +sample_weight, 8ep, warmup=0.10 | ? | ? | ? | ? | ? | ? | Training epoch 1/8 |
| exp3 | PLANNED | +hard-normal augment (600 samples) | ? | ? | ? | ? | ? | ? | Waiting for exp2 results |

## Decision Gates
- exp2 epoch 4: If existence_acc >= 0.80 → exp2 may be sufficient
- exp2 epoch 8: If existence_acc < 0.80 → proceed to exp3 with hard-normals
- exp3 epoch 4: If all metrics beat baseline → SFT phase complete, proceed to RL
