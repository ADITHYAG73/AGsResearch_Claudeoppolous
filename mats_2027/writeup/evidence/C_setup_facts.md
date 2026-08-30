# Facts for section C (Setup) — every line traces to experiments.md or the shipped artifacts

## The NLA
- Official Anthropic release: `kitft/nla-gemma3-12b-L32-av` (AV) + `kitft/nla-gemma3-12b-L32-ar` (AR), Apache-2.0
- Base model: `google/gemma-3-12b-it`; activations taken from **layer 32 of 48**, residual stream, d = 3840
- AV: same architecture as the base; the activation is injected as a single token embedding
  into a fixed prompt (injection_scale 80000 from the sidecar); sampled at T = 1
- AR: base model truncated to 33 blocks + a learned linear head on the final token
- MSE is **direction-only**: both vectors L2-normalised to mse_scale = √3840 = 61.97 before
  comparison, so MSE = 2(1 − cos). Confirmed three ways: code defaults, README, sidecar values.
  Magnitude of the activation is invisible to the whole pipeline.

## Corpus and sampling
- 6 pilot passages: 5 cricket Wikipedia paragraphs + 1 French Revolution passage — the
  **maintainers' own example** (`doc_id = THEIR_EXAMPLE`), kept as the in-distribution reference.
  **It is INCLUDED in every number** (397 of the 2065 claims). Cricket-only the levels read
  THEME 70.0 / ENTITY 39.3 / DETAIL 34.6 (n=707/333/628) vs 69.1 / 43.8 / 36.0 for all six —
  the ordering is unchanged either way. [CORRECTED 30 Aug: this file previously said "excluded".]
- **Last 10 contiguous token positions** of each passage → 60 activations. The constraint is on
  the POSITION INDEX, not on passage length: every sampled index ≥ the official
  `_MIN_POSITION = 50` (nla/datagen/stage0_extract.py:35). Realised ranges: Eden Gardens 86–95,
  Test cricket 106–115, Dravid 138–147, Tendulkar 170–179, Laxman 245–254, THEIR_EXAMPLE 79–88
  — minimum sampled index is 79.
- **K = 4 resamples** per activation at T = 1 → 240 explanations
- Why cricket: a domain the author can adjudicate claim-vs-text in under 30 s; and
  in-distribution — AV→AR round-trip cos 0.996 on cricket vs 0.997 on their own example (SMOKE-01)
- Second corpus (SAVARKAR-01): 7 random ~600-char passages from a 2019 biography, length-matched;
  70 activations, 280 explanations; Dravid passage re-run as a regression canary (cos = 1.000000)

## Pipeline, in order
1. extract — base model forward pass, `hidden_states[33]` at each sampled position
2. AV decode — sglang server, `<explanation>` tags parsed, 0 CJK / 0 no-tag across all runs
3. Stage 3 decompose — `claude-haiku-4-5-20251001`, structured output, one explanation → atomic
   claims with level ∈ {THEME, ENTITY, DETAIL}; prompt FROZEN 21 Aug (PILOT-03); forward-looking
   claims excluded. 2065 claims from 240 explanations (DECOMP-01)
4. Stage 4 judge — same model, claim vs the exact prefix the model had read; S/C/N collected,
   **binary (supported / not) used for all analysis** — CONTRADICTED is structurally almost
   unreachable for "the text mentions X" meta-claims (1.9% of claims); JUDGE-02
5. Ablation — REWRITE one claim out of its explanation (the paper's method, recovered from its
   HTML; SOURCE-01), not deletion. 2305 variants. Guard: a rewrite may never be longer than the
   original (0.8% invented content otherwise). 2 of 2065 not ablatable, kept and flagged.
6. AR score — official `NLACritic.score()`, batch 1 (bit-exact; batched bf16 added 2.5e-4 noise
   to a Δ whose effects are ~1e-3, so batching was built and then not used). 2305 in 135 s.
7. Δ = mse(ablated) − mse(intact), same explanation, same activation, same resample.
   Δ > 0: removing the claim hurt reconstruction (load-bearing). Δ < 0: removal helped.
8. Relatedness (REL-01) — binary RELATED / UNRELATED on the 995 false claims, same model
9. Semantic matcher (MATCH-02) — two-pass Haiku, groups claims that assert the same thing across
   resamples; 115 groups with ≥3 resamples (vs 22 by exact-string match)

## Judge validation
- 150 stratified claims (50/level, balanced across the 10 position offsets) graded BLIND by the
  author + 30 retests = 180 items in 33.6 min. The sample was drawn by `harness/sample_stratified.py`,
  a SEEDED script (seed 20260822) — not chosen by a model. Harness exposes only
  claim_id / claim / prefix. Self-consistency 29/30 = 96.7%. Agreement with Haiku 133/150 = 88.7%.
- The paper reports NO validation for its confabulation judge.

## What was frozen when (so it can be checked)
- 21 Aug: Stage 3 prompt; grading conventions (vague ≠ false; grade against the prefix; alias → S)
- 22 Aug: quote rule (S only if the string is present); H2 kill condition
- 25 Aug: H1 detector rule (dBIC>10) pre-registered, then REVISED after the real data showed the
  control null had been under-skewed → H1 verdict downgraded to exploratory
- 26 Aug: PATCH-01 predictions (both the author's and Claude's)
- 28 Aug: SAVARKAR-01 predictions (both); p449 prediction sharpened to a falsifiable form

## The paper's own pipeline (SOURCE-01) — for the "we reproduced their method" claim
- Their confabulation analysis is four Haiku calls per explanation: decompose, verify, vibe
  (relatedness), match (recurrence). Outputs are in the paper's HTML; **prompts were never
  published**. Ours are reconstructions matching their output format.
- Their ablation is a rewrite, not a deletion (visible in their `rewritten_text`).
- Their matcher visibly erred on the one published example (claim 6, `appears_in [2,3,6]`,
  excludes the very explanation the claim came from).

## Infra (for the record, not the prose)
- RunPod A40 48 GB, SECURE, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`,
  torch 2.9.1+cu128 · transformers 5.3.0 · torchvision 0.24.1 · sglang 0.5.10.post1
- Total GPU spend across the six pod sessions = **$3.99** (1.23 + 1.08 + 0.40 + 0.34 + 0.49 + 0.45).
  [CORRECTED 30 Aug from "≈$3.5" — that figure was never summed.]
- API spend measured ONCE: $4.00 of Haiku on 28 Aug (SAVARKAR-01, read off the console). No stage
  records token usage, so every other Haiku figure in the project is an estimate, not a measurement.
