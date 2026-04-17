# 2026-04-17 RL Compute-Loss Acceleration

## Summary

- Active RL now prefetches `reference_token_log_probs` after `_materialize_episode_inputs()` and before `compute_loss()`.
- `compute_loss()` now prefers cached `reference_token_log_probs` and falls back to inline reference KL forward only when the cache is missing.
- `_iter_loss_microbatches()` no longer uses a rigid fixed-size split. It now uses a balanced splitter to avoid degenerate tails such as `3+1` and `3+3+2`.

## Scope

- Updated active RL trainer only:
  - `saver_v3/rl/timesearch_aligned_grpo_trainer.py`
- Updated RL runtime tests only:
  - `tests/test_rl_runtime.py`

## Behavior Notes

- No reward, advantage, rollout, FECV, or communication semantics were changed.
- No user-facing config was added.
- `compute_loss_microbatch_size` remains the preferred policy microbatch target, but the implementation may use `target + 1` to eliminate bad tail splits.
- `reference` prefetch uses an internal target of `max(4, compute_loss_microbatch_size)` and stores CPU-side per-token logprobs only.
- Inactive / zero-weight episode batches do not run reference forward; they receive zero-valued `reference_token_log_probs` aligned with `completion_ids`.

## Validation

- `python -m py_compile saver_v3/rl/timesearch_aligned_grpo_trainer.py tests/test_rl_runtime.py`
- `PYTHONPATH=. pytest tests/test_rl_runtime.py -q -k "reference_log_probs or balanced_batch_sizes or iter_loss_microbatches or prepare_inputs_keeps_episode_batches_on_cpu_until_loss_microbatch or compute_loss_logs_progress_per_batch"`
  - `10 passed`
- `PYTHONPATH=. pytest tests/test_rl_runtime.py -q -k "compute_liger_loss_logs_liger_and_reference_forward or compute_loss_does_not_disable_liger_before_batch_compute or prepare_inputs_does_not_recompute_old_logprobs_for_active_batches or materialize_episode_inputs_merges_full_signature_bucket_before_loss_microbatching"`
  - `4 passed`
