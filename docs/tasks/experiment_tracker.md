# SFT Improvement Experiment Tracker

## Baseline Reference
- Fixed baseline: artifacts/fixed_baseline/msad_test/metrics.json

## Results Summary

| Run | exist_acc | cat_f1 | temp_miou | r1@0.3 | r1@0.5 | chain_f1 | evid_f1 | Beat BL |
|-----|-----------|--------|-----------|--------|--------|----------|---------|---------|
| Baseline | **0.846** | 0.229 | **0.392** | 0.275 | 0.133 | 0.289 | 0.168 | — |
| exp1-ep3 | 0.588 | 0.320 | 0.258 | 0.458 | 0.175 | **0.550** | 0.258 | 5/7 |
| exp2-ep1 | 0.758 | 0.311 | 0.228 | 0.475 | 0.175 | 0.000 | 0.219 | 4/7 |
| exp3-ep1 | 0.746 | **0.353** | 0.257 | **0.492** | **0.183** | 0.000 | 0.225 | 4/7 |

## Key Findings
1. **existence_acc**: exp2 showed weights help (+17pts), exp3 mild weights maintain gain
2. **temporal_miou**: MAX_KEY_FRAMES 8->16 had NO effect (0.257 vs 0.258). Bottleneck is model decision, not input resolution
3. **event_chain_f1**: 0.0 at epoch 1 for exp2 and exp3. Need epoch 2-3 to see if it recovers (exp1 only had epoch 2-3 data)
4. **Deadlock bug found+fixed**: verify_hypothesis loop causing 36% null answers in exp2. Fix applied to exp3 eval

## Remaining: exp3 epochs 2-3 running, then exp4 (architectural changes)
