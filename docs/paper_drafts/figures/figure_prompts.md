# Figure Prompts for `search_to_verify_neurips_v1.md`

This file stores figure-generation prompts for a later polished paper pass. The Markdown drafts remain the source of truth, and these prompts should stay synchronized with both the English and Chinese paper drafts.

## 1. Graphical Abstract: Fixed Observation vs Search-to-Verify

**Target file:** `graphical_abstract.png`

**Suggested aspect ratio:** `1600x900`

**Prompt:**

Create a NeurIPS-style graphical abstract for a paper titled "From Search to Verify: Agentic Event-Chain Search and Counterfactual Evidence Verification for Video Anomaly Understanding." Use a clean academic visual style with a white background, restrained blue-green-orange accents, crisp vector geometry, and no photorealism. Split the composition into two contrasted halves. On the left, show conventional VAU as a fixed-observation pipeline: sampled frames or clips, one-shot reasoning, final label plus explanation. On the right, show the proposed Search-to-Verify Agent: full video timeline, active search with scan_timeline and seek_evidence, explicit recovery of precursor, trigger, and confirmation or aftermath, policy-internal verification with verify_hypothesis, and structured case finalization with finalize_case. Emphasize the conceptual gap as missing event-chain completeness and missing evidence necessity. Keep text minimal and figure-ready for a conference paper graphical abstract.

## 2. Method Overview: Agentic Event-Chain Search

**Target file:** `method_overview.png`

**Suggested aspect ratio:** `1600x1000`

**Prompt:**

Create a publication-quality method overview diagram for Search-to-Verify Agent, an agentic video anomaly understanding framework. Depict a video and a task query entering a multimodal policy that can choose among four actions: scan_timeline, seek_evidence, verify_hypothesis, and finalize_case. Show scan_timeline as broad temporal coverage, seek_evidence as targeted evidence acquisition, and an evidence ledger containing window ids, evidence ids, and stage hints. Highlight the target of reasoning as event-chain completeness over precursor, trigger, and confirmation or aftermath, rather than isolated event frames. Then show verify_hypothesis producing one of four decisions: sufficient, insufficient, misaligned, or redundant, together with the next action recommendation: continue_search, revise_claim, refine_evidence, or finalize. Finalization should output a structured anomaly report with category, interval, evidence ids, and semantic answer. Style should be modern, minimal, vectorized, and suitable for CVPR or NeurIPS.

## 3. Training Figure: Teacher-Rewritten SFT plus FECV-Grounded RL

**Target file:** `fecv_training_flow.png`

**Suggested aspect ratio:** `1600x1000`

**Prompt:**

Create an academic training-flow diagram for the Search-to-Verify Agent training pipeline. Start from runtime episodes, then oracle skeletons, then a teacher judge rewrite stage that produces teacher_rollout_primary step supervision for SFT. After SFT, show policy rollouts entering an FECV scoring module. The FECV block should summarize evidence-faithfulness diagnostics such as decision sufficiency, minimal-subset sufficiency, negative specificity, and stage-sensitive evidence drop effects. Then show reward aggregation centered on accuracy reward, FECV evidence faithfulness reward, and protocol finalize reward, followed by a GRPO update. Add a side branch for optional local credit routing signals including search_local, evidence_local, query_local, stage_local, and teacher_local, but clearly render them as secondary to the primary FECV reward path. Use clean arrows, consistent color semantics, and a minimalist NeurIPS methods-figure style.

## 4. Qualitative Case Panel: Event-Chain Recovery and Verification

**Target file:** `qualitative_case_template.png`

**Suggested aspect ratio:** `1800x1200`

**Prompt:**

Design a reusable qualitative analysis panel for an agentic video anomaly understanding paper. Include a horizontal video timeline with thumbnail placeholders, clearly marked precursor, trigger, and confirmation or aftermath regions, and a side evidence ledger that lists selected window ids and evidence ids. Add a verification panel showing the current hypothesis, the selected evidence subset, and a verification decision of sufficient, insufficient, misaligned, or redundant. Include a compact counterfactual view where one evidence item is dropped and the verification result changes. Finish with a structured final answer card containing anomaly category, interval, evidence ids, and semantic answer. The layout should look like a top-tier conference figure template, not a marketing graphic.
