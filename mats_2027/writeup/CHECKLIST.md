# Write-up tracker — MATS Winter 2027, Neel Nanda stream

**Deadline: Fri 4 Sept 2026, 11:59pm PT. Extension available to 11 Sept.**
Started 29 Aug. Every number cited below must trace to `../experiments.md`.

Rules for this file:
- AG writes every sentence of the write-up. Claude reads for ACCURACY only, never restyles.
- Section E is AG's alone. Claude does not draft, edit, or suggest wording for it.
- Tick a box only when the section is written AND its numbers have been checked against
  `experiments.md`. Two ticks, two people.

Status key: `[ ]` not started · `[~]` drafting · `[x] AG` written · `[x] AG+C` numbers verified

---

## Writing order (not reading order)

Write B first — it is the thing you know best and it warms you up. Write A LAST.

| step | section | status | notes |
|---|---|---|---|
| 1 | B. Question and why | [ ] | ~150 words |
| 2 | C. Setup | [ ] | ~300 words, facts liftable from experiments.md |
| 3 | D. Results | [ ] | ~800 words, six subsections |
| 4 | E. What I verified myself | [ ] | ~150 words. **AG only.** |
| 5 | F. Limitations | [ ] | 5 bullets, ~100 words |
| 5b | F2. Reflections | [ ] | ~150 words: implications + what you'd do next (hooks go here) |
| 6 | G. Figures | [ ] | 3–4. Claude produces, AG verifies every number |
| 7 | A. Executive summary | [ ] | ≤600 words. LAST. |
| 8 | Form answers (Q1–Q8) | [ ] | separate from the doc; Q8 bans citing this project |
| 9 | Google Doc, anyone-with-link, exec summary first 1–3 pages | [ ] | |

---

## A. Executive summary — ≤600 words, read FIRST by Neel, written LAST by you
- [ ] One-sentence question (open with the question, not the result — the accepted write-up did)
- [ ] 4–5 bullet takeaways, negatives INCLUDED, each with a number
- [ ] One sentence: what you did vs what Claude did
- [ ] Graphs referenced (he expects them)
- [ ] Word count ≤600 — count it

## B. Question and why — ~150 words — WRITE FIRST
- [ ] The paper's claim: the AR "is only a weak per-claim verifier" (their words)
- [ ] Your question: is the signal ABSENT, or BURIED in noise?
- [ ] Why it matters: text-based grading cannot separate "the AV invented it" from "the model
      inferred it and the AV read it out"; the AR judges against the ACTIVATION, so it is in
      principle the only instrument that can. (sources/CLAUDE.md, 2026-08-19, "conceptual core")
- [ ] Neel's own framing: "claims that can be removed to IMPROVE reconstruction" — only briefly explored

## C. Setup — ~300 words
- [ ] NLA: official `kitft/nla-gemma3-12b-L32` (AV + AR), base Gemma-3-12B-IT, layer 32 of 48, d=3840
- [ ] Corpus: cricket Wikipedia, 6 pilot passages, last 10 token positions, K=4 resamples at T=1
      → 60 activations, 240 explanations
- [ ] Why cricket: a domain YOU can adjudicate claims in under 30 s; in-distribution (cos 0.996 vs
      their 0.997 on SMOKE-01)
- [ ] Why last-10 positions: the paper's recurrence analysis uses them; all ≥ MIN_POSITION 50
- [ ] Pipeline, one line each: extract → AV decode → Stage 3 decompose (Haiku 4.5, frozen) →
      Stage 4 judge (S/C/N, binary for analysis) → rewrite-out ablation → AR score → Δ
- [ ] Δ definition: mse(ablated) − mse(intact), same explanation, same activation; MSE is
      direction-only (= 2(1−cos)), confirmed from the shipped sidecars
- [ ] What was frozen when: Stage 3 prompt (PILOT-03, 21 Aug); grading conventions (21–22 Aug);
      H1 rule (25 Aug, then revised — see D3)
- [ ] The paper's pipeline was NOT published; recovered from its HTML (SOURCE-01) — decompose /
      verify / vibe / match are four Haiku calls; ablation is a REWRITE not a deletion

## D. Results — ~800 words, six subsections, every number from experiments.md

### D1. Specificity replicates — and probably understates
- [ ] THEME 69.1 / ENTITY 43.8 / DETAIL 36.0 % supported (n=887/427/751) vs paper 64/28/24
- [ ] Different NLA, base model, corpus, positions, judge prompt — ordering holds
- [ ] JUDGE-02: you graded 150 blind + 30 retest; self-consistency 96.7%; agreement 88.7% binary
- [ ] The two error modes sit at OPPOSITE ends (Haiku strict on THEME, you generous on DETAIL
      quotes) → both COMPRESS the gradient → true gap likely LARGER
- [ ] The paper never validated its judge (contrast: 97% on 186 for eval-awareness)

### D2. AUC 0.535 — what "weak" means
- [ ] 0.535 [0.510, 0.559], 1068 true vs 995 false, bootstrap over claims
- [ ] Above chance; barely. The paper gives no number.
- [ ] K-averaged: 0.615 [0.488, 0.736] on 91 matched groups — suggestive, CI includes chance

### D3. H1 (mislabelled mixture) — dead, with power
- [ ] Statement: related-false = two populations under one label → bimodal Δ
- [ ] Related-false cell = 975 of 995 false claims (98% related — REL-01)
- [ ] Dip test p = 0.992; power 76–100% across H1's own predicted 26–42% mixture range
- [ ] HONESTY: the pre-registered ΔBIC rule FIRED (+843) and was a skew artefact — a single
      skewed hump matched to the data gives ΔBIC>10 in 200/200 draws. Rule revised after seeing
      data → verdict is EXPLORATORY. Say this; it is a strength, not a weakness.
- [ ] Bounded: a mixture <20% would have been missed

### D4. H2 (noise-limited) — mechanism verified, payoff underpowered
- [ ] Noise is STOCHASTIC: spread(K)² = signal² + noise²/K fitted on K=1,4 predicts K=2,3 out of
      sample to 3%; two independent noise estimates agree (0.00189 vs 0.00212)
- [ ] Signal floor at 55% of the K=1 spread — the thing we WANT to exist
- [ ] noise/signal 1.51× at K=1 → 0.76× at K=4
- [ ] Bottleneck is RECURRENCE: only 115 of 2065 claims recur in ≥3 of 4 resamples
- [ ] HONESTY: the original kill condition (slope −0.5) was mis-specified — it ignored the floor

### D5. Removal improves reconstruction 30% of the time
- [ ] 30.0% of 2063 single-claim ablations have Δ < 0 (NOISE-01 saw 18.8% at n=399, but
      method AND positions changed — not attributable)
- [ ] Neel's exact phrasing; the paper never reports the rate
- [ ] By level: THEME +0.00032, ENTITY +0.00105, DETAIL +0.00145 — DETAIL 4.5× THEME, the
      OPPOSITE of what the specificity result predicts → H3 (redundancy), consistent-with not tested

### D6. Confabulation is IMPORT, not misbinding — two domains
- [ ] PATCH-01: planting Gavaskar/Umrigar/Thangavelu → 0/40 each; deleting Bradman → 25% vs 22.5%
      (no change). Prefix token NOT necessary, planted name NOT sufficient — at these positions
- [ ] Why: layer-32 activation 250 chars downstream barely encodes the name — centred cosine
      0.997 vs 0.42 for ONE token step, 0.01 for a different passage
- [ ] SAVARKAR-01: 63% false vs 50% on cricket (CIs disjoint); >90% of false person-claims name
      someone ABSENT from the passage in BOTH domains (93.6% / 98.3%)
- [ ] Reading: parametric knowledge is the source of the CORRECT specifics; a thinner activation
      gets the same specificity budget filled with famous-but-wrong names. Interpretation, not test.
- [ ] Both pre-registered predictions (yours and Claude's) were WRONG — say so

## E. What I verified myself — ~150 words — **AG WRITES THIS ALONE**
Prompts for yourself, not text:
- [ ] The 150+30 blind grading, and what it found (96.7% self-consistency)
- [ ] The conventions you ruled on (S/C/N, quote rule, relatedness, matcher)
- [ ] The position hypothesis you killed by eye before any data existed
- [ ] The Bradman catch — needed cricket knowledge
- [ ] The missing relatedness labels you found by checking your mental model against the data
- [ ] The three times a control caught Claude: ΔBIC skew artefact; "seeded by a prefix token";
      median-based noise ratio
- [ ] What you did NOT do: no hooks, no internal probing — the NLA was a black box throughout

## F. Limitations — 5 bullets, ~100 words
- [ ] Black-box evaluation only; no white-box work
- [ ] Judge validated on cricket only (88.7%); unknown on Savarkar
- [ ] Semantic matcher never spot-checked by a human
- [ ] H2 underpowered (recurrence, not noise); H1 verdict exploratory
- [ ] Savarkar carried to verdicts only; no Δ on a second domain
- [ ] Related-vs-unrelated contrast not reproducible here (no unrelated cell)

## G. Figures — Claude produces, AG verifies every number against experiments.md
- [ ] G1. Specificity: ours vs paper, by level, with CIs (and your labels vs Haiku's)
- [ ] G2. Δ distribution for related-false claims — the H1 null, with the skewed-null overlay
- [ ] G3. Dip-test power curve across mixture fraction — why the null means something
- [ ] G4. spread(K) with fitted curve and the two out-of-sample points — H2's mechanism
- [ ] G5. Savarkar vs cricket false rate by level, CIs
- [ ] (optional) G6. Import-vs-misbinding split, both domains

## Form answers — separate from the doc
- [ ] Q8 (evidence you can do research) — CANNOT cite this project. Use the two live blog posts.
- [ ] Time accounting: ~20h task + 2h summary/form. Reading admissions doc, GPU setup, waiting,
      form-filling do NOT count. Be honest about the clock — it was never formally started.
- [ ] LLM-use disclosure: agentic use is table stakes; the variance is in question choice and
      verification. Point at section E.

---

## What the two accepted write-ups do (structure only — read on 28/29 Aug)

|                        | write-up 1 (reasoning direction) | write-up 2 (backtracking) |
|---|---|---|
| total length           | ~2,100 words                     | ~4,000 words + 8 figures  |
| exec summary           | ~300 w, opens with the QUESTION  | ~550 w, opens with the PROBLEM, RQ1–3 framing |
| takeaways              | 5 bullets                        | 3 numbered, each with a metric |
| negative results       | "Initial Approach Failed" as a heading | one whole experiment framed as unsuccessful |
| limitations            | 5-point list, ~100 w             | dedicated subsection: scale, validation gaps, dataset, compute |
| figures                | 2 graphs + example outputs       | 8, captioned, referenced from prose |
| what author did vs LLM | not stated                       | one line (Claude generated phrase examples) |
| reflections section    | no                               | yes — interprets implications, admits time/resource limits |

**What this changes for us:**
- Length is NOT the signal. Both got in; one is half the other. Target ~2,500–3,000 words +
  4–6 figures. Do not pad.
- Both open the summary with the QUESTION/PROBLEM. Ours does too.
- Both have a FRAMED negative result. We have four. Give H1 its own heading, as they did.
- Write-up 2 has a **Reflections** section — implications + "what I'd do with more time". Add it:
  short, honest, and it is where the hooks/white-box gap gets stated as the next step, not hidden.
- Write-up 2 numbers its research questions (RQ1–3) and its results (R1–R4) and maps one to the
  other. Do that — it makes the exec summary trivial to write from.
- Neither has a real "what I verified myself" section. **Ours will, and it is a differentiator**
  — Neel's admissions doc weights it above everything else.
- Captions carry information in write-up 2. Every figure of ours gets a caption that states the
  finding, not just the axes.

---

## Log
- 2026-08-29 — checklist created. Nothing written yet. 6 days to deadline.
- 2026-08-29 — second accepted write-up read (backtracking, ~4k words, 8 figs). Added F2 Reflections; RQ→R numbering; caption rule.
