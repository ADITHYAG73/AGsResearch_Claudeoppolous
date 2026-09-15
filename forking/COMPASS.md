# Forking Paths — research compass

Drafted 2026-09-07 by Claude from a full read of both papers and both repos. **AG owns
every question here.** Claude's inferences are marked (inf). Everything else is sourced to
the paper, the code, or a Hugging Face page checked today.

Companion: `ROADMAP.md` (sized projects, free pre-GPU work). This file is the *why* and the
*what to ask*; ROADMAP is the *how*.

---

## 0. North star (Neel's three tiers: goal / sub-goal / objective)

**Goal (months):** one public write-up, with a figure and a number, that changes what a
reader should believe about forking tokens in current open models. Neel's novelty test:
"Should I assign different probabilities to propositions I care about after observing the
results?" Rigorous replications and clean negatives qualify. Methods expected to work do not.

**Sub-goal (weeks):** decided by AG from §3 below.

**Objective (days):** reproduce the 2026 noise law from shipped data on CPU. Nothing is
believed until that passes.

---

## 1. What is established (do not re-prove)

| Claim | Source | Strength |
|---|---|---|
| Forks exist: single probable-token swaps change the final answer | 2024, GPT-3.5, 7 tasks | Shown at S=30, noise unmodelled |
| Replicate noise in o_t is exactly multinomial; TVD falls as 1/√S (slope −0.49) | 2026 §3.3, App F | Strong, verified to 2% of the null |
| The true o_t curve is mostly flat with sparse sharp forks | 2026 Fig 1, App B | Shown on tinyMMLU, two 8B models |
| Smoothing recovers the S=200 curve from ~1% of samples | 2026 §3.1 | 5–22× at low S; worse than raw at big forks |
| Reasoning models fork at sentence granularity too | 2026 App C | One model, one task |
| Final-token confidence overstates certainty (40% → ~100% on one sequence) | 2024 §4.1 | One vivid example, not a rate |

## 2. What is open (the gaps, with where each is stated)

| # | Gap | Who names it | Status |
|---|---|---|---|
| G1 | **Open-ended outcomes.** Both papers only ever use categorical outcomes. | 2024 §2.2, §5; code has `StoryCloze_Open` drafted, answer fn commented out; 2026 App A | Designed, never run |
| G2 | **Do the 2024 odd-token forks survive high S?** 2026 shows S=30 jaggedness is largely dice. | Implied by 2026 §3.3; never stated | Unexamined |
| G3 | **Per-token swap effects (o_{t,w}) were dropped in 2026.** Only the pooled o_t is modelled. A smooth pooled curve is compatible with rare sharp swaps. | 2024 §2.4 survival analysis; absent from 2026 | Regression, unexamined |
| G4 | **Current models.** 2024 = GPT-3.5; 2026 = two 8B 2024-era models. | 2026 App A "larger models remains to be tested" | Untested |
| G5 | **Predict forks from activations, no sampling.** | 2024 §5 "it might even be possible to avoid sampling altogether if hidden activations can be used to predict forking" | Untouched |
| G6 | **Are forks causal control points?** Does intervening at a fork move the outcome more than intervening elsewhere? | 2024 §5 (steering by patching at forking tokens) | Untouched |
| G7 | **Adaptive sampling** — choose positions to sample instead of smoothing post hoc. | 2026 §4; repo has `resample_adaptive`, status unknown | Suggested |
| G8 | **Fix credible-interval coverage** (48–64% at nominal 90%). | 2026 App A | Reported, not fixed |
| G9 | **Sub-5% alternatives.** Both pipelines truncate at p ≥ 0.05, so a fork through a rare token is invisible by design. | Both methods sections; consequence is (inf) | Structural |
| G10 | **Where in a sequence do forks sit, and does it depend on task?** 2024 Fig 6 shows task-dependent positions. | 2024 §4 | One model, descriptive |

## 3. Pointed questions, ranked by information rate (Steinhardt's rule: cheap tests of
##    uncertain things first)

Each has: the question as a falsifiable sentence, what a yes and a no would each mean, the
cheapest test, and the model. **Models verified on Hugging Face 2026-09-07:**
Qwen3.8-27B (dense, Aug 2026, Apache 2.0, thinking on by default), Qwen3.6-27B (dense),
Gemma-4-31B and Gemma-4-12B (Jul 2026; the 26B-A4B is MoE, avoid). Smaller current dense
options need checking on the model pages before use. Neel's doc names Qwen 3.5/3.6 dense.

### Q1 — Do the 2024 odd-token forks survive S=200? (G2 + G4)
*Sentence:* On a current dense open model, fewer than half of the positions flagged as
forks at S=30 remain forks at S=200 under the same threshold.
*Yes:* a chunk of the 2024 headline was sampling noise; report the survival rate by token
class (punctuation / function / content / answer token).
*No:* odd-token forks are real and robust; 2024 stands on a stronger footing than its own
method gave it.
*Test:* 2026 sampler, S=200 every token, ~10 questions each from GSM8k, HotpotQA,
LastLetter (2024's task mix, prompts from `ebigelow/forking-paths`). Subsample S=30 from the
same draws (nested, as 2026 did) and count forks at both S. Noise law as the gate.
*Model:* Gemma-4-12B or a current Qwen dense ≤12B for cost; 27B for the headline if budget.
*Cost (inf, order of magnitude):* 30 questions × 300 positions × ~3 branches × 200 draws ×
~300 tokens ≈ 1.6B tokens. **Too much for RunPod credit at 27B.** At 4-token spacing: 400M.
This is the number to pin down before choosing. 2026 App D says stride 4 keeps forks; stride 16 misses them.

### Q2 — Does open-ended generation fork or drift? (G1)
*Sentence:* On story continuation, the dispersion of continuations as a function of
position shows change points detectable above the 1/√S noise floor, as MMLU does.
*Yes:* forks are a property of generation, not of multiple-choice scoring.
*No:* open-ended text drifts continuously; "forking token" is an artefact of discrete outcomes.
*Test:* outcome = embedding of the continuation (2024's own stated plan). Two scalars per
position: mean pairwise distance among the S continuations, and distance to the base path.
Replicate-split noise check on the scalar. **Control:** an LLM judge bins ~200 continuations
into a few outcome classes, and the categorical curve is compared to the embedding curve.
Without that control, "fork in embedding space" may be a style fork. (inf)
*Model:* same as Q1. *Cost:* lower than Q1 — 20 prompts, stride 4, S=100.

### Q3 — Is the pooled curve hiding sharp per-token swaps? (G3)
*Sentence:* At positions where o_t is flat, there exist alternatives w with p ≥ 0.05 whose
o_{t,w} differs from the base token's by TVD > 0.3.
*Yes:* 2026's "noise, not sensitivity" overstates; the swap-level claim of 2024 survives
even where the pooled curve is smooth.
*No:* flat o_t really means the model is indifferent to which probable token it picks.
*Test:* **zero GPU.** The 2026 stores record per-branch outcome counts (README: "the kept
next-token branches ... and the recorded outcome category of each resampled rollout").
Compute o_{t,w} for every branch in `data/s200`, and the survival statistic from 2024, on
their data. This is the cheapest real question in the list and it is a new result on
published data.

### Q4 — Can a probe on the residual stream predict a fork before sampling? (G5)
*Sentence:* A linear probe on layer-ℓ activations at position t predicts TVD(o_t, o_{t+1})
> ε with AUC > 0.7 on held-out questions.
*Yes:* fork detection in one forward pass; every downstream use becomes cheap.
*No:* forks are not linearly readable from the residual stream at that layer, or not before
the token is emitted.
*Test:* needs labelled forks from Q1 or Q2 plus cached activations from the same run.
**Design rule: cache activations during any sampling run**, so Q4 costs nothing extra later.
*Depends on:* Q1 or Q2.

### Q5 — Are forks causal control points? (G6)
*Sentence:* Steering or patching at a detected fork changes the final-answer distribution
more than the same intervention at a matched non-fork position.
*Depends on:* Q1/Q2 for fork labels, Q4 for a direction to steer along. Third project, not first.

### Q6 — Where do forks sit, and does position depend on task? (G10)
Free once Q1 data exists: histogram of fork positions by task, compare to 2024 Fig 6.
Descriptive; a side figure, not a project.

Not pursuing: G7 adaptive sampling (methods work, not interp; and their code may already
have it), G8 interval coverage (statistics fix, low interest for the field we care about),
G9 sub-5% branches (cost explodes; note as limitation).

## 4. Recommended order (Claude's, AG to overrule)

1. **Q3 first.** Zero GPU, days not weeks, a new result on their data, and it settles whether
   the pooled-curve framing of 2026 is hiding what 2024 cared about. Also forces us to
   master the store format before generating any.
2. **Q1 or Q2 second**, with activations cached. Choose by cost after pinning the token
   budget. Q2 is more original; Q1 is more defensible and reuses more code.
3. **Q4** on the cached activations. This is the interpretability paper.
4. Q5 only if Q4 is positive.

## 5. Neel's rules that bind here (from `research+writing_advice_45k.md`)
- "Replicating and extending an existing paper can be a good starting point, especially if
  you don't have an existing mentor." — this project is exactly that.
- Novelty = expanded knowledge. Rigorous replications and negatives count. Methods expected
  to work do not — so Q1 is only interesting if the answer is uncertain. It is.
- Be explicit about what is and is not novel, in the introduction and related work.
- Write goals in three tiers and check weekly. Prioritise and execute in separate modes.
- Write the plan and estimate time, then expect to deviate.

## 6. The question AG raised that neither paper answers
"I detect a fork. In what way is it useful?" Both papers list uses; neither tests one. A fork
is useful only if predictable without sampling (Q4) or exploitable as a control point (Q5).
Q1–Q3 are preconditions, and the write-up must say so.
