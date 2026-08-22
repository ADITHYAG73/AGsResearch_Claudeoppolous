
- **2026-08-18 (sources saved + first hand-check)** — Saved primary sources to
  `mats_2027/sources/`: `nla_paper_2026-05-07.txt` (full text, all sections, provenance
  header), `nla_paper_2026-05-07.raw.html` (raw primary artifact — interactive widget data
  lives here, not in the text dump), `nla_claim_analysis_fennec_s244.json`.
  - **The paper embeds one worked per-claim example** (widget "NLA Claim Analysis — fennec
    s244 (Common Pile)", grader `claude-haiku-4-5-20251001`): 11 claims with fields
    `level` (THEME/ENTITY/DETAIL), `subtype`, `verdict` (SUPPORTED/NOT_IN_TEXT),
    `relatedness` (DIRECT/ADJACENT), `appears_in`, `position_count`, `delta_mse_pct`.
    All three signals plus ground-truth labels, hand-checkable. **One context only, not
    the full dataset.**
  - **Hand-check on those 11 claims (n=11 — illustration, not a result):** recurrence works
    (true mean position_count 5.5 vs false 3.0); **the ablation signal is INVERTED** (true
    Δmse% mean −0.33 vs false −0.11); specificity shows nothing at this n (3/4, 2/3, 3/4).
    Concrete instance of the paper's "noisy on individual transcripts".
  - Two **true** claims had strongly negative Δmse (−2.6, −1.37) — i.e. removing a true
    claim *improved* reconstruction. Bears directly on Neel's "removed to improve
    reconstruction" framing.
  - **UNVERIFIED:** the sign convention of `delta_mse_pct` (I infer positive = removal
    raised MSE = claim load-bearing; consistent with the paper's aggregate finding, but
    not documented in what I extracted). **Confirm before building on it.**
  - **Compute datapoint (Appendix), settles §8a:** open-model NLA RL ran 3,000 steps on
    two 8xH100 nodes — ~1.5 days to 70% FVE, ~1 week to 75% FVE, ~$1,500–$5,000 compute.
    Training our own NLA is definitively out of envelope. Inference only.

- **2026-08-19 — the confound is real, not hypothetical (read the raw pipeline).**
  Extracted the full per-claim pipeline from the paper HTML (`fennec s244`). The AV emits
  **prose**, not claims — bolded-heading paragraphs. Haiku 4.5 then does **four** separate
  jobs: `decompose` (carve into claims AND assign level THEME/ENTITY/DETAIL), `verify`
  (SUPPORTED/NOT_IN_TEXT), `vibe` (relatedness DIRECT/ADJACENT), `match` (`appears_in` —
  which of the 10 token positions the claim recurs at).
  - **Therefore: 2 of our 3 signals share a source with the label.** specificity = Haiku
    (`decompose`); recurrence = Haiku (`match`); label = Haiku (`verify`). **Only Δmse is
    independent** (comes from the AR) — and it's the signal that looked weakest in the
    n=11 hand-check.
  - **Concrete mechanism, checkable:** claim_9 is labelled DETAIL/quote *and* NOT_IN_TEXT.
    Both hinge on the same act — search for an exact string. A claim may be called a
    "detail" *because* it is literally checkable, and literal checks fail most often. So
    "detail claims are more often false" may be partly an artifact of what Haiku calls a
    detail. Likewise `position_count` may measure claim **vagueness** (vague claims match
    across positions; exact quotes don't), which is the same axis as `level`.
  - **Failure mode to avoid:** a combined predictor that scores well by imitating Haiku
    rather than detecting confabulation. Would look like a strong positive result.
  - **Capping vs confounding are different:** label noise lowers the ceiling and adds
    ambiguity (makes good work look worse); correlated errors inflate agreement (makes bad
    work look better). Both are broken by the same fix — **independent labels not produced
    by Haiku** (stronger model + my own hand-labels).

- **2026-08-19 — domain choice is load-bearing, plus a possible judge inconsistency.**
  - **Verified from the widget's `prefix_text`:** the input passage contains "According to
    Annals of Joseon Dynasty, Nam Gon now set out to slander Jo...". Haiku judged
    claim_11 ('Text references "Annals of the Joseon Dynasty" (조선왕조실록)') **SUPPORTED**
    but claim_5 ('Text references "Joseon Wangjo Sillok"') **NOT_IN_TEXT** — the same work,
    romanized vs translated name. Consistent under a strict *string* reading, inconsistent
    under a *referential* reading; claim_11's parenthetical Korean name is also absent from
    the passage yet did not block SUPPORTED. **Rubric prompt not yet located — do not call
    this an error until checked.** Noticing it at all requires knowing the two names denote
    the same work → direct evidence that grader domain knowledge matters.
  - **Decision rationale:** picking a text domain I can adjudicate is not comfort, it is
    required by the design — the contribution depends on labels that are NOT Haiku's.
  - **Criterion to use:** not "a topic I understand" but "a domain where I can settle
    claim-vs-text in <30s, unambiguously". Most THEME/format claims need no expertise;
    ENTITY/title claims do.
  - **Design choice: hold genre fixed (Wikipedia-style prose, as in their Common Pile data),
    vary only the topic** to something I know cold. Keeps register/claim-mix comparable
    while making me a competent grader. Small general-text control slice if time allows.
  - **Constraints to respect:** (a) domain change breaks comparability with their 64/28/24 —
    never claim replication; (b) pick a topic well-represented in pretraining, else I am
    studying NLA-on-OOD-text instead; (c) **grade BLIND** — labels produced before seeing
    Δmse / level / position_count, randomized order, enforced by the harness, since I hold
    the hypothesis.

- **2026-08-19 (CORRECTION + rubric findings).**
  - **RETRACTED: the "possible judge inconsistency" logged above is wrong.** The widget's
    own AV explanation states: *"The phrase 'Annals of the Jos' **has appeared as** 'Joseon
    Wangjo Sillok'..."* — a declarative claim about the text, not a prediction or gloss.
    So Haiku's extraction is faithful and **NOT_IN_TEXT is correct** (the passage contains
    "Annals of Joseon Dynasty", never the romanization). claim_11 is correctly SUPPORTED.
    The authors cite this very case in their prose as their example of a false-but-related
    claim. **The judge was right; my criticism was not.** Domain-knowledge argument stands.
  - **The decompose/verify rubric is NOT published** — only its outputs appear in the page
    data. We cannot exactly reproduce their labelling. State as a limitation in Q6.
  - **A sibling rubric IS published** (the "factual groundedness" grader) and is a usable
    template. It defines a claim as a concrete testable assertion about what the text says,
    and **excludes**: predictions about what comes next ("expected NLA behavior"), vague
    tone/style/register/genre observations, meta-commentary/hedging, and **reasonable
    inferences not explicitly stated**. Rule worth stealing: **vague ≠ false**.
    → Note: the excluded "inference" category is the one closest to unverifiable claims
    about model cognition. Their protocol steps around the hardest case.
  - **Three verdict values, collapsed to two.** Page display code maps `SUPPORTED`→TRUE and
    **both `CONTRADICTED` and `NOT_IN_TEXT`**→FALSE. These differ: *contradicted* = text
    says otherwise; *not in text* = merely absent, which may still be true of the
    **activation**. Keeping them separate is a cheap analysis the paper did not do.
  - **Δmse sign convention CONFIRMED** (supersedes the earlier UNVERIFIED flag): their
    colour function grades Δmse ≥ 0 green, negative red → **positive = removal raised MSE =
    claim was load-bearing**.

- **2026-08-19 (conceptual core — why the AR angle is the right one).**
  - **NOT_IN_TEXT is a verdict about the passage, NOT about the activation.** Two
    indistinguishable-on-the-page cases hide under it:
    (a) **M genuinely inferred it** and encoded it; the AV read it out faithfully →
        not in text, but TRUE of the activation. This is the tool working (cf. their
        user-model, language-switching and eval-awareness results).
    (b) **The AV invented it** ("excessive expressivity") → not in text, FALSE of the
        activation. Genuine confabulation.
    **Text-based grading cannot separate (a) from (b).** That is why the published rubric
    excludes "reasonable inferences" — not because they are harmless, but because they are
    ungradable by that method.
  - **This is the strongest motivation for the project:** the AR judges a claim against the
    **activation**, not the text — in principle the only instrument that can separate (a)
    from (b). → Use this as the "why is this interesting" answer on the form.
  - **relatedness:** label space is DIRECT / ADJACENT / UNRELATED; display collapses
    DIRECT+ADJACENT → "Related", UNRELATED → "Unrelated" (the chart's False;related vs
    False;unrelated). Applied **only to false claims**. **Rubric NOT published** — so their
    second ablation finding (relevant-false hurts more than unrelated-false) rests on an
    unpublished rubric. Flag if used.
  - **Mechanics, stated precisely (for my own recall):** h_ℓ is a *residual-stream
    activation at layer ℓ*, not an embedding. It is L2-normalized, scaled by α, and
    **replaces the special token's embedding** (not the reverse) at the AV's layer-0
    embedding slot — a layer-42-into-layer-0 transplant, which is why it is out of
    distribution, why α matters, and why warm-start SFT is required. Tokenizer untouched.
    Sampling at T=1. Confabulation experiments take the activation at the **last token**;
    recurrence analysis uses the **last 10** positions. "Literal text" = the `prefix_text`
    passage preceding that token.

- **2026-08-19 (the cut, and a possible third label source).**
  - **"In the text" = in the PREFIX at capture time**, not the finished document. Forced by
    causality: the model only sees the prefix. Their warm-start prompt says so explicitly.
  - **Verified on the Planning-in-Poetry case** by reconstructing the token list from the
    page: prefix = `A rhyming couplet:⏎ He saw a carrot and had to grab it`; continuation =
    `⏎ His hunger was like a starving rabbit`. **"rabbit" is absent from the prefix** and
    appears only in the continuation. So at capture time it was a *plan in the activation*,
    not a fact about the text — the clean example of the 2x2's top-right cell.
  - **IDEA (mine to evaluate — could be a component or a better project):** for
    forward-looking claims, **the continuation is ground truth**. Free, mechanical, and —
    crucially — **not produced by Haiku**, so it breaks the shared-cause confound. It would
    let us validate the AR's reconstruction-delta on claims that live in the
    *beyond-the-text* region the paper never measured, bridging the gradable/ungradable
    halves. The paper crossed that bridge exactly once, by hand, via steering on one poem.
  - Caveats: (a) a plan is not an outcome — T=1 sampling, and their steering worked only
    ~50% of the time, so the continuation is evidence not proof; (b) covers only
    forward-looking claims, not user-model / genre / belief claims.

- **2026-08-19 (inference plan — pulled `nla_meta.yaml` + `run_config.yaml` from HF).**
  - **Three weight sets, run SEQUENTIALLY** (they do not co-fit; they don't need to):
    (1) `Qwen/Qwen3.6-27B` target M → hook out h_42 (~54GB bf16, ~36GB if truncated to 42
    layers); (2) `av_base/` + `av_rl_adapters/iter_000300` → explanations (~54GB);
    (3) `ar_reconstructor/` 43 layers + value head → reconstruct + MSE/FVE (~36GB).
    Dump artifacts to disk between stages → an 80GB card suffices.
  - **Scoping win: ablation is AR-only.** Every claim-deletion / paraphrase / repeat needs
    only stage 3 (the ~36GB model). Stages 1–2 run once.
  - **Config facts:** layer_index 42, d_model 5120, corpus `finefineweb_100k` (web text,
    NOT Common Pile), positions_per_doc 10, temperature 1.0, max_new_tokens 256.
    **Celeste's AV prompt asks for 2–3 snippets** (paper's Opus version: 4–5) → fewer
    claims per explanation, so more activations needed for the same claim count.
    **AR prompt:** `Summary of the following text: <text>{explanation}</text> <summary>`.
    They used **vLLM** for AV rollouts; transformers+peft for the AR.
  - **My left/right marker memory was RIGHT and is sourced after all** — not in the paper,
    but in the config: `injection_token_id: 158983`, `injection_left_neighbor_id: 29`,
    `injection_right_neighbor_id: 510`, `injection_char: ㈜`.
  - **TRAP 1: never quantize.** The measurement *is* activation reconstruction fidelity;
    4-bit shifts h_42 off-distribution and corrupts FVE in a way that looks like a result.
  - **TRAP 2 (open, must check EasyNLA source):** paper says activations are L2-normalized
    and α-scaled, and the AV is sensitive to α — but Celeste's extraction config says
    **`norm: none`**. Possibly raw-at-storage / normalized-at-injection. **Verify before
    trusting any FVE number.**
  - **Domain — correcting my earlier advice:** "hold genre fixed for comparability" was
    overweighted; we cannot compare to their 64/28/24 anyway (different NLA/base, FVE not
    comparable). Real criteria: in-distribution (FineFineWeb = web text → cricket
    commentary qualifies), gradable by me (commentary: yes), rich claim mix (commentary is
    dense in ENTITY + DETAIL/value claims — exactly the 28%/24% categories). **Cricket
    commentary is a defensible primary choice.** Caveat: Cricinfo commentary is
    copyrighted — check dataset licences before publishing excerpts.
  - **Day-zero check (free of the clock, it is env setup):** round-trip 5 cricket passages
    end to end and confirm FVE lands near the card's ~75.6% at step 300. If it comes out
    at ~20%, the α/normalization path is wrong.
