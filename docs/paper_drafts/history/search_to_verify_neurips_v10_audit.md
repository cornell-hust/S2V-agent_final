# Search-to-Verify v10 Audit Notes

## Paper Delta Map

This audit records the specific paper updates required by the active RL collapse-fix refactor.

### Stale v9/v10-pre-fix content that was removed

- The old single evidence-faithfulness formula
  - `R_fecv = 0.6 support + 0.2 minimal + 0.2 specificity`
- The main-text claim that the implementation uses a uniform six-branch counterfactual verifier
- Stale RL settings
  - `w_prot = 0.1`
  - `T_max = 14`
  - `G = 8`
  - `8 x H200`
  - `DeepSpeed ZeRO-3`

### Implemented replacement story

- Claim 3 remains the same top-level scientific claim.
- Its current implementation is now explicitly described through three reward branches:
  - `easy_normal`
  - `suspicious_normal`
  - anomaly `online_core`
- The trainer-side stabilization story is now explicit:
  - zero-variance groups use EMA fallback for non-trivial partitions
  - `easy_normal` remains intentionally zeroed
  - collapse reduction is treated as training-stability evidence, not final benchmark evidence

## Result-to-Claim Gate

### Supported claims

- The current implementation **materially reduces reward collapse** in active RL training.
- The current implementation **substantially improves trainable signal throughput** by reducing all-zero-advantage and all-filtered groups.
- Claim 3 can now be described as a **multi-granular, branch-specific FECV implementation** rather than a single monolithic reward term.

### Unsupported claims

- The collapse problem is fully solved.
- Final benchmark metrics improve because of this refactor.
- All constant-bucket pathologies are removed.

## Experiment Audit

### Evidence used

- Pre-fix log: `pipeline_20260420_144800.log`
- Post-fix log: `pipeline_20260420_161552.log`
- Current reward implementation:
  - `saver_v3/core/reward.py`
- Current trainer fallback implementation:
  - `saver_v3/rl/grpo_trainer_env.py`

### Key audit numbers

| Diagnostic | Pre-fix | Post-fix |
| --- | --- | --- |
| Total rollout groups | 27 | 76 |
| `zero_advantage_count=4` | 20 | 0 |
| `filtered_below_min_weight > 0` | 20 | 2 |
| `filtered_below_min_weight = 4` | 20 | 1 |
| Old anomaly bucket `0.290625` | 6 | 0 |
| Old normal bucket family `1.248625 / 1.292375 / 1.357475 / 1.381625` | 16 | 0 |
| Residual constant groups | not highlighted | `1.290844 x 7`, `0.254138 x 3` |

### Audit conclusion

- The paper may claim **substantial collapse reduction**.
- The paper may claim **higher effective trainable signal**.
- The paper must **not** claim final downstream performance gains from these logs.
- The paper must **not** claim full collapse elimination because residual constant groups remain.
