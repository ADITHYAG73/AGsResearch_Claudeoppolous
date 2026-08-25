# Hypothesis tracker — MATS 2027 / NLA project

Method: **state H, name its rivals, kill the cheapest rival first, record the verdict.**
A hypothesis with no named rivals is not yet a hypothesis.
Status values: `open` · `testing` · `supported` · `killed` · `parked`

---

## H1 — related-false claims are mislabelled, not confabulated

**Statement.** The AR's "removing related-false claims hurts reconstruction more than
unrelated ones" effect exists because a substantial fraction of related-false claims are
**not confabulations** — they are faithful readouts of the activation that the text-based
judge marks false because the string is absent from the prefix. (2x2 top-right cell.)

**Why it is plausible.** The paper's own stated principle: *"claims that don't reflect the
activation should, in theory, contribute little to reconstruction."* Run backwards:
related-false claims DO contribute → they DO reflect the activation.

**Predicts (CORRECTED 2026-08-19 — the earlier version was already falsified).**
The old wording said related-false claims behave like TRUE claims under activation-grounded
measures. That is inconsistent with the paper's own number: true claims sit at 0.25–0.37 pp,
related-false at **0.14**. If they behaved like true claims they would sit at ~0.30.

The mixture framing fixes this and makes the prediction quantitative:

- Related-false is **two populations under one label**. Some behave like TRUE (activation-
  true, text-false — the "Bhaskar" claims); the rest behave like unrelated-false
  (genuine confabulations — the "Anand" claims).
- Solving `0.14 = p·(true value) + (1−p)·0.06` for the paper's true range gives
  **p ≈ 26–42%**: roughly a quarter to two-fifths of related-false claims should sit in a
  HIGH-Δ mode, the rest near 0.06.
- **Signature: the per-claim Δ distribution for related-false claims is BIMODAL (or clearly
  over-dispersed) — not one hump at 0.14.**
- **This is what discriminates H1 from every rival.** R1 (lexical overlap), R2 (coherence)
  and R3 (length) all predict a *single* elevated cluster. Only "two populations, one label"
  predicts two modes.

**KILL CONDITION.** After per-claim variance is reduced (see precondition), if the
related-false Δ distribution is unimodal — one hump, no high-Δ mass beyond what the rivals
explain — H1 is dead. A clean negative result: the category is homogeneous and the AR's
partial signal has a boring cause.

**PRECONDITION (why H1 depends on H2).** A mixture is invisible through noise: two modes
only separate if the gap between them exceeds the within-mode spread. Per-claim spread is
currently ~5× the effect. **So H2 (variance reduction) is not a parallel project — it is
the enabling step for H1.** Order: stabilise each claim's Δ by repeated ablation under
varied surface conditions (paraphrases, AV resamples, shuffled bullets), *then* plot.
Note the two averagings are opposite in effect: averaging ACROSS claims destroys the shape
(what the paper did); averaging WITHIN a claim, across replays, sharpens it.

**Status:** `KILLED` — tested 2026-08-25 (H1-01), NOT SUPPORTED, exploratory.

> Related-false Delta (n=975, K=1) is **unimodal**: Hartigan dip p = **0.992**. The dip test
> has a 0% false-positive rate and **76-100% power** across H1's own predicted mixture range
> of 26-42%, measured at the observed noise. It found nothing.
>
> The frozen dBIC rule DID fire (+843.4) but that was a **skew artefact**: a single skewed
> hump matched to the real data produces dBIC>10 in **200/200** draws, median +846.5 - the
> real value sits BELOW the null's median. The internal control is that TRUE claims, which
> H1 says nothing about, score dBIC +2467, higher still.
>
> Downgraded to exploratory because the rule had to be revised after the real distribution
> was seen - Claude's SIM-02 skewed null used skew ~1.0 against a reality of 2.63-5.63.
>
> **Bounded:** a mixture below ~20% would have been missed (power 29% at p=0.20). H1 is dead
> in its STATED form, not in every conceivable form. And a dead H1 does NOT show the AR
> treats these claims alike - only that their measured Delta does not separate. See
> [[H3]], which would produce exactly this null whether or not a mixture exists.

**TODO (mine, before the form):** rewrite the Statement above in my own prose. The current
wording is Claude's. Neel reads the form answers first and flags LLM-sounding prose as a
significant negative signal — and the question has to read as mine because it is.

### Rivals (kill in this order — cheapest and most boring first)

| # | rival | test | cost | verdict |
|---|-------|------|------|---------|
| R3 | **Length / position artifact.** Related-false claims are simply longer, or sit elsewhere in the bullet list. | Compare claim length + list position by category. Pure data analysis. | trivial, no GPU | — |
| R2 | **Coherence, not content.** Deleting a topically-related bullet damages prose coherence; an unrelated bullet is a non-sequitur whose removal may help flow. | **Delete vs replace** with neutral filler of matched length. Effect vanishes ⇒ R2. | AR-only | — |
| R1 | **Lexical overlap.** The claim carries words/concepts genuinely in the activation ("Joseon", "Korean historical records"); deleting it deletes those regardless of the assertion. Claim as *vehicle*, not as truth. | Rewrite a related-false claim keeping vocabulary, flipping the assertion. Effect survives ⇒ R1; collapses ⇒ H1. | AR-only | — |

**Self-check.** I have stated I *want* the related/unrelated result to be wrong, and H1
conveniently makes it "not wrong, just mislabelled". Comfortable landing ⇒ maximum
suspicion. **Run R3 first.**

---

## H2 — the per-claim signal is noise-limited, not absent

**Statement.** The AR's failure as a per-claim verifier is a **signal-to-noise** problem,
not an absence of signal. Averaging Δ over K paraphrases / resamples raises per-claim
AUC substantially.

**Sizing (order-of-magnitude, NOT a measurement).** From the 11 embedded fennec claims,
converting Δmse% → FVE points via `ΔFVE ≈ Δmse% × (1−FVE)` with FVE assumed 0.78:
per-claim spread ≈ 4.7× the ~0.17pp effect; **d ≈ 0.71 → per-claim AUC ≈ 0.69** (chance 0.50).

**Predicts.** If noise is independent, d grows as √K: AUC 0.84 at K=4, 0.93 at K=9.
Justified by the paper's steganography result (AR responds to meaning, not surface form).

**KILL CONDITION.** If variance does **not** fall as ~1/√K, the noise is systematic rather
than stochastic, averaging cannot rescue it, and the approach is dead. That is itself a
fast, reportable negative result.

**Status:** `open` — d must be measured properly on my own data first.

---

## Failure modes — written BEFORE building (2026-08-21)

A negative result is NOT a failure: H1 unimodal ⇒ clean "category is homogeneous" finding;
H2 variance not ~1/√K ⇒ clean "noise is systematic" finding. Both produce a write-up.
**The real worst case is producing no answer at all.** Four ways, most likely first:

| # | failure | why it happens | defence (build it in NOW) |
|---|---------|----------------|---------------------------|
| 1 | **Inconclusive, found late** — the worst case | Δ effect ~0.17pp vs per-claim spread ~0.8pp; with ~150 passages the related-false histogram may be a blur: neither one hump nor two | **H2 FIRST.** Measure paired Δ variance early. If K=4 doesn't bring spread under the effect size, know it by hour 6, not 18 — and scope down to the variance finding alone |
| 2 | **Related-false cell too thin** | H1 lives in ONE cell. Paper: 29–67% of claims by level, but theme-heavy explanations (64% true) could starve it | **Count the cell from Stage 3 labels BEFORE Session B.** Extend the corpus if thin |
| 3 | **Labels are bad** | If the two human graders agree at ~70%, every downstream number is label-noise-limited, not AR-limited | The overlap set exists to measure this. A low agreement number is *itself* reportable — it is the validation the paper skipped |
| 4 | **Memorisation confound** — the one to watch | Gemma-3 has plausibly memorised famous Wikipedia subjects, so "activation-true, text-false" claims may reflect **pretraining, not reading**. Wouldn't kill the result — would change what it *means* | **Obscure-subject passages in the corpus** (minor grounds, domestic competitions, obscure tours). **Analyse famous vs obscure separately** and check whether the effect differs |

What is NOT on this list: infrastructure failure. That was the worst case a week ago;
SMOKE-01 retired it.

**Pilot-set rule:** the 6 SMOKE-01 passages have been READ (by me and by the AV's output).
They are the pilot set — allowed for intuition, **never in the analysis.**

---

## Standing methodological commitments

- **Grade BLIND.** My labels are produced before seeing Δmse / level / position_count,
  in randomised order, enforced by the harness. I hold the hypothesis; I am a biased grader.
- **Independent labels.** 2 of the 3 signals (specificity, recurrence) share a source with
  the label (all Haiku). Any headline that rests on Haiku-vs-Haiku agreement is suspect.
- **Every load-bearing number gets a check I wrote myself.** If Claude computes both the
  number and the check, the check is worthless.
- **No silent scope cuts.** If I drop a condition, it gets written down here with the reason.

---

## H3 (parked) - redundancy, not falsity, drives Delta toward zero

**Raised by AG, 2026-08-22**, from reading one worked ablation. Parked deliberately in order
to finish the ablation run first. Do NOT act on this before the AR scoring is done.

**Observation.** In the Laxman final-token explanation (pos 254, k=0), ablating c3
("The text mentions a Test match between India and West Indies") leaves the words
"West Indies" in the text TWICE - inside c4's quoted sentence, and again as c6
("The text mentions West Indies"). So the AR still reconstructs "West Indies" fine.

**Statement.** Single-claim Delta measures the MARGINAL contribution of one claim given every
other claim stays put - not the contribution of the CONCEPT. Any claim whose content is
duplicated elsewhere in its own explanation scores Delta ~ 0 whether it is true or false.

**Why it matters.** A source of Delta ~ 0 that is uncorrelated with truth. It is a candidate
mechanism for the paper's own finding that the AR is only a "weak per-claim verifier" - and
it is a component of the noise H2 tries to average away. If redundancy is a large share of
that noise, K-averaging will NOT fix it: it is systematic per claim, not stochastic across
resamples. **That intersects directly with H2's kill condition (variance must fall as
~1/sqrt(K)).**

**Test (data comes free with the AR run - no extra GPU).** Per claim, compute a redundancy
score: how much of its content appears elsewhere in the same explanation (needs the LLM
matcher, which SOURCE-01 already puts on the build list). Then check whether |Delta| is
systematically smaller for high-redundancy claims, and whether the true-vs-false Delta gap
widens once low-redundancy claims are isolated.

**Prior art:** none found. The paper runs single-claim deletion and never discusses
redundancy between claims inside one explanation.

**FIRST OBSERVATION CONSISTENT WITH H3 (SCORE-01, 2026-08-23) - NOT a test.**
Delta by claim level, n=2063:
```
THEME   n=886   mean Delta +0.00032 +/- 0.00007
ENTITY  n=426   mean Delta +0.00105 +/- 0.00025
DETAIL  n=751   mean Delta +0.00145 +/- 0.00026
```
The paper says THEME claims are true 64% of the time vs DETAIL 24%, and that true claims
matter MORE to reconstruction - so THEME should carry the HIGHEST Delta. It carries the
LOWEST, by 4.5x, with CIs nowhere near overlapping. H3 explains this directly: THEME content
("the text is about cricket") is restated throughout an explanation, so ablating one instance
removes almost nothing; a DETAIL claim (a specific invented quote) occurs once.
**This is consistent with H3 and does not test it.** Rivals not excluded: DETAIL claims may
simply carry more tokens; or specific claims may genuinely constrain the activation more.
The test still requires a per-claim redundancy score from the semantic matcher, plus a length
control.

**Status:** `parked` - revisit AFTER Stage 5 relatedness and the semantic matcher exist.

---

## H1 DETECTOR RULE - DECIDED AND PRE-REGISTERED 2026-08-25 (AG)

**THE RULE, frozen:**
> H1 verdict = **dBIC(2 vs 1) > 10** on per-claim Delta averaged over K=4, computed on the
> RELATED-FALSE claims only. Hartigan's dip statistic is reported alongside as a DESCRIPTIVE
> number, **not** as a gate.

**What it replaces.** `pipeline/analysis.py:120` froze the rule as `dBIC>10 AND dip_p<0.05`.
SIM-02 (2026-08-22) showed that conjunction scores **0/40** at the plausible noise level: the
dip test requires a visible VALLEY in the density, which is a strictly stronger claim than
"two populations under one label". Under the frozen rule H1 could not return a positive
result no matter what is true.

**Evidence for the replacement** (SIM-02, 40 seeds, n=2065, K=4, planted worlds):
```
                       detects planted   false alarm      false alarm
                       mixture           on Gaussian null on SKEWED null
  dBIC>10 AND dip       0/40                 0/40             0/40
  dBIC>10 alone        40/40                 0/40             0/40
```
The skewed null exists because dBIC's real failure mode is fitting two Gaussians to a single
LOPSIDED hump - a plain Gaussian null cannot expose that, and real Delta is right-skewed
(NOISE-01: 81% positive; SCORE-01: 70% positive). At the operating point it was never fooled.

**Also established by SIM-02, and it constrains K:** more averaging can make dBIC WORSE. At
lower noise the skew stops being masked and dBIC reads it as bimodality - 37/40 false alarms
on the skewed null at K=8, ratio 1.0. K=4 is both what the data has and inside the safe band.

**WHY THIS IS LEGITIMATE AND NOT GOALPOST-MOVING.** Changing a decision rule after seeing
results is how people talk themselves into findings. Three things make this defensible, and
all three must be stated in the write-up:
  1. the change was decided on PLANTED data, never on real Delta;
  2. it was decided BEFORE the real related-false Delta distribution was inspected - verifiably,
     since relatedness labels did not exist until REL-01 on 2026-08-25 and the related-false
     Delta distribution has not been computed as of this decision;
  3. it carries explicit false-alarm control against a realistic (skewed) null, not just a
     convenient one.

**If the constraint is ever broken** - i.e. if real related-false Delta has been looked at
before a rule change - say so in the write-up and downgrade the H1 verdict from confirmatory
to exploratory. That is not optional.

---

