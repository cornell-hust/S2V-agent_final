# idea2_v2 Port Provenance

Source tree: `/mnt/shared-storage-user/mineru2-shared/zengweijun/Wmh/ideas/idea2_v2/code`

Source commit: `f7e6f0cfea3118015df1f818a9b5da08732255d5`

Copied modules live under `third_party_ports/v2/saver_agent/` and preserve v2 behavior for category canonicalization, event-chain scoring, metrics, teacher-judge normalization/package logic, rollout parsing, and TimeSearch-style rollout adapters.

Adjustments made for v3:

- Rewrote internal `saver_agent.*` imports to local relative imports so the port is self-contained.
- Inlined the small `normalize_query_text` helper in `event_chain.py` instead of carrying the full v2 proposal runtime.
- Removed top-level dependencies on v2 model-loading/Qwen runtime helpers from `teacher_judge.py`; equivalent small helpers are local and optional model execution still requires the external `transformers` stack at call time.
- Made `environment.py` lazy-import the tool registry only inside `SaverVideoInteraction.execute_predictions`, allowing parser/rollout utilities to import before the v3 tool runtime exists.
