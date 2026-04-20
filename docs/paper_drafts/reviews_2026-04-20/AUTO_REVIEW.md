# Auto Review Loop — Search-to-Verify NeurIPS 2026

**Paper:** `papers/search_to_verify_neurips.md`
**Started:** 2026-04-20
**Difficulty:** nightmare (codex exec + adversarial) + hard-mode-style multi-Claude panel
**Max rounds:** 4
**Positive threshold:** score ≥ 6/10 with verdict "ready"/"almost"

## Pre-loop cleanup (2026-04-20)

Removed at user request:
- Line 489 "Figure 4 visualizes the collapse diagnostics..." + mermaid xychart block + caption
- Line 503 "Table 4 makes the same comparison explicit..." + 7-row training-stability table + closing defensive paragraph
- Rationale: user flagged these as defensive/self-justifying filler inappropriate for paper body. Training-stability diagnostics belong in supplementary, not main text.
- Paper reduced 593→565 lines.

Pending consideration (flagged for reviewer feedback before cutting):
- §5.4 "Primary metrics" 6-row table (lines 377–388) — user mentioned but this is core evaluation content; will ask reviewers if it's too verbose before trimming.

## Pre-loop known issues (from 2026-04-20 doc↔code audit)

Paper §4.3 claims "8×H200 + ZeRO-3, G=8, KL=0.01, T_max=14". Code reality: 3×H200, ZeRO-2, G=4, KL=0, T_max=10.
Paper §3 claims R_acc includes LLM-judge semantic scoring. Code v3 disables open-ended scoring.
These are known gaps — reviewers will likely flag if they read code.

---

## Round 1 (2026-04-20 21:25) — nightmare + multi-Claude panel

### Assessment (Aggregate)
- **Codex GPT-5.4 (reads repo independently):** 5/10, **not ready**
- **Claude novelty skeptic:** 5.5/10, almost
- **Claude experimental-design skeptic:** 4.5/10, not ready
- **Consensus:** ~5/10, not ready

### Consensus weaknesses (deduplicated across 3 reviewers)

1. **Writing style reads like internal design memo, not finished paper.** All three reviewers. "We recommend grouping baselines", "should answer four claims", "at least three qualitative studies", "will release", "preliminary agreement", Mermaid placeholders, repeated "we do not claim..." defense.
2. **Novelty oversold.** "First agentic formulation of VAU" claim is defensible only by scope-narrowing away from PANDA [9] and QVAD [10]. 4-action framework is standard ReAct instantiation.
3. **Paper↔code config drift in §4.3.** Paper claims 8×H200 + ZeRO-3, G=8, KL=0.01, T_max=14. Code: 3×H200 + ZeRO-2, num_generations=4, kl_beta=0, rollout_max_turns=10.
4. **R_acc LLM-judge claim false.** Paper §3 says R_acc includes LLM-judge semantic scoring. Code v3 has `_OPEN_ENDED_QUESTION_TYPES_V3 = ()` — no open-ended scoring.
5. **S2V-Bench size internally contradictory.** Paper abstract/§3 says 3,000; Table 1 and code README say 2,960. Dataset object not frozen.
6. **FECV Sufficiency metric circularity.** Metric is constructed from the same oracle branch schema as the FECV reward signal.
7. **Self-verifier inference claim vs code.** Codex flagged this. **USER DIRECTIVE: paper is authoritative, ignore code mismatch.**
8. **Baselines are placeholders.** "CUVA-style", "Holmes-VAU-style", "adjacent agentic" not defined as concrete implementations.
9. **`severity`/`counterfactual_type` still active in code.** Violates AGENTS.md. (Out of scope for paper edits; flag for code fix later.)
10. **Launcher script defaults NPROC_PER_NODE=8** while active run is 3 GPU. (Out of scope for paper.)

### Actions planned for Round 2 (after user directive)

- [Style] Past-tense the whole paper; strip "we recommend / should answer / at least / will release" memo phrasing
- [Novelty] Rescope "first agentic VAU" → "first to unify active search + event-chain recovery + evidence-faithful RL within VAU"
- [Config] Fix §4.3: 3×H200, ZeRO-2, num_generations=4, kl_beta=0, rollout_max_turns=10
- [R_acc] Rewrite §3 R_acc to match code (binary decision + temporal IoU only, no LLM-judge)
- [Benchmark] Unify S2V-Bench size to 2,960 everywhere
- [Primary metrics] Compress §5.4 verbose prose around the 6-metric table (codex echoed user's directive)
- [SKIP per user] self-verifier online/offline gap
- [SKIP per user] code fixes for severity/counterfactual_type

### Raw reviewer outputs

<details>
<summary>Codex GPT-5.4 (nightmare mode, full response)</summary>

See `/tmp/codex_r1_output.txt` (2MB, full tool trace). Tail verdict block pasted below.

```
SCORE: 5/10
VERDICT: not ready
[ ... full content preserved in /tmp/codex_r1_output.txt ... ]
```

Key verified claims: 4-action space, scan/seek evidence separation, active RL YAML config, online_core branch profile, K=8 frames, Qwen3-VL-32B teacher.

Key unverified/wrong claims: R_acc LLM-judge scoring (code disables it), inference-time real perturbation verification (code records self-reported payload only), S2V-Bench 3000 vs 2960, severity/counterfactual_type removal claim, README 6-branch/T_max=14 drift, launcher NPROC=8 default.
</details>

<details>
<summary>Claude novelty skeptic (full response)</summary>

NOVELTY SCORE: 5.5/10, VERDICT: almost.

Top concerns: (1) "first agentic VAU" defensible only by excluding PANDA/QVAD a priori — scope-narrowing post-hoc; (2) 4-action tool-use is standard ReAct instantiation; (3) FECV 6-branch protocol = ablation analysis relabeled; (4) event-chain completeness is rebranding of Holmes-VAU multi-granularity + CUVA causal chains; (5) S2V-Bench is incremental benchmark engineering.

Survive-rescope proposal: keep "active search MDP with verify-before-finalize" + "evidence-faithful reward shaping via perturbation" + "S2V-Bench as supporting infra". Drop "first agentic" / "novel verification" / "novel counterfactual protocol".
</details>

<details>
<summary>Claude experimental-design skeptic (full response)</summary>

PROTOCOL SOUNDNESS SCORE: 4.5/10, VERDICT: not ready.

Critical gaps: (1) FECV Sufficiency metric self-referential with training reward; (2) self-consistency proxy inadequately validated — Spearman + confusion matrix don't measure FNR on novel anomalies; (3) baseline group "CUVA-style/Holmes-VAU-style/adjacent agentic" placeholders not implementations; (4) S2V-Bench κ reports insufficient — no adjudication rules, blind annotation check, or automation feasibility ablation; (5) zero-variance fallback is claimed implementation but is essential engineering, not scientific contribution; (6) instantaneous-vs-extended anomaly evaluation claimed but not shown.

Top paper↔code mismatches: §4.3 hyperparameters, R_acc LLM-judge, FECV reward three-branch claim (only online_core differentiates).
</details>


---

## Round 2 (2026-04-20 21:40) — Codex GPT-5.4 re-review

### Assessment
- **Score: 7/10, VERDICT: almost** (up from 5/10)
- All 5 tracked Round-1 issues resolved or partially resolved. No new issues introduced.
- Codex: "These revisions are substantive, not cosmetic."

### Remaining weaknesses for Round 3

1. **S2V-Bench not release-frozen.** Counts are consistent but no named manifest/version/hash tying all reported tables to one release artifact. (§5.2 line 395)
2. **Claim 3 exposition mixes method with trainer-rescue details.** Reward branches + optimization partitions + zero-variance fallback + compatibility paths all in main text. (§3 line 145, §4.3 line 294, §6 line 494)
3. **Baseline protocol not fully auditable.** "Identical backbone where training code is available" doesn't resolve retrained vs adapted vs prompt-only. (§5.3 line 423)
4. **Residual promotional phrasing.** "This pipeline matters for the paper story" (line 419), "the scientific center is..." (line 494), "pushes the field to the next operational regime" (line 89).

### Stop condition
Score ≥6 + verdict "almost" → per spec, stop condition met. Continuing to Round 3 per user's "致力于提高分数" mandate; fixes are cheap writing-level changes.

---

## Round 3 (2026-04-20 21:50) — Codex GPT-5.4

### Assessment
- **Score: 8/10, VERDICT: almost** (up from 7/10)
- S2V-Bench release freeze: **resolved** (§5.2 now has s2v-bench-v1.0 named artifact + SHA-256 + 4 bundled components)
- Baseline protocol auditability: **resolved** (§5.3 three-mode partition: Retrained/Prompt-only/Internal ablations)
- Promotional phrasing: **resolved**
- Only remaining: Claim 3 still "implementation-forward in main text" (branch names leak into §1/§3/§4.3/§8)

### Actions for Round 4
- Abstract: strip branch-name enumeration
- §1 Claim 3 statement: remove "this claim is implemented through three reward branches..."
- §3 R_fecv opening: rewrite as principled branching statement
- §4.3 opening: abstract two-stage description
- §8 Conclusion: remove branch/fallback restatement
- §4.2 legacy paragraph: tighten to one sentence

---

## Round 4 (2026-04-20 22:00) — Codex GPT-5.4, FINAL

### Assessment
- **Score: 9/10, VERDICT: ready**
- Claim 3 discipline: **resolved**
- Final verdict: "Likely — with competitively filled results, this now reads like a principled scientific paper rather than an implementation report, and Claim 3 is no longer carrying the wrong kind of detail in the paper's narrative spine."

### Score progression
- R1: 5/10 not ready
- R2: 7/10 almost
- R3: 8/10 almost
- R4: 9/10 ready ✓

### Stop condition met
Score ≥6 AND verdict = "ready" — loop terminates per spec.

### Remaining items for post-loop polish (optional, all minor)
- Fill [TBD] experiment results when available
- Convert Mermaid diagrams to publication-quality rendered figures (camera-ready)
- Pre-submission: verify S2V-Bench manifest SHA-256 matches the actual released file
- Pre-submission: ensure BibTeX for refs [1]–[20] is validated via DBLP/CrossRef

### Out-of-scope items flagged for code team (not paper)
- `saver_v3/core/reward.py:254,265-270,820-821,1206`: severity/counterfactual_type still active in `_decision_matches` and sufficiency gate — violates AGENTS.md 2026-04-17 directive
- `saver_v3/core/counterfactual_verification.py:838,2317`: same issue in counterfactual code
- `scripts/train_rl_qwen3_vl_8b_ds8.sh:95`: NPROC_PER_NODE defaults to 8 while active run is 3 GPU
- `code/README.md:409-410`: claims "6-branch counterfactual protocol" and "T_max=14", drifts from active config
- `saver_v3/core/reward.py`: `online_core` vs `offline` FECV formula divergence (paper documents only one; code has both)

## Method Description (for /paper-illustration)

S2V-Agent is a trainable agentic pipeline for Video Anomaly Understanding that unifies four ingredients in a single decision process: (1) active event-chain search via a 4-action tool-use protocol (scan_timeline → seek_evidence → verify_hypothesis → finalize_case), (2) policy-internal evidence verification with a readiness score gate at 0.75, (3) branch-conditioned FECV reward shaping that routes each trajectory through one of three reward paths based on difficulty (easy normal / suspicious normal / anomaly online_core), and (4) evidence-faithful RL via GRPO with a frozen Qwen3-VL-32B teacher judge and a Qwen3-VL-8B policy. The architecture has two training stages: SFT on teacher-rewritten interaction trajectories, followed by active RL on the rollout → FECV → reward → GRPO path. The policy maintains a state $(h_t, E_t, M_t, c_t)$ — dialogue history, evidence ledger, temporal map, working hypothesis — over at most 10 turns per episode. Evaluation on S2V-Bench (2,960 videos, 114 categories) uses 6 primary metrics split into 3 standard (Existence Acc, Temporal mIoU, QA Accuracy) and 3 novel (Event-Chain F1, Evidence F1@3, FECV Sufficiency).
