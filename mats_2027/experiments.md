# Experiment log — MATS 2027 / NLA project

One entry per experiment. Fill **before** running (P1–P3) and **after** (R1–R4).
Nothing here is optional: this file IS the raw material for form questions
Q "conclusions", Q "strongest evidence against", Q "limitations", Q "LLM use".

Rule: **if it isn't in this file, it didn't happen.** Sessions evaporate; this doesn't.

---

## Template (copy for each experiment)

### EXP-NN — <one-line title>            `date` · `clock: Xh Ym`

**P1 · Question.** What am I trying to find out, in one sentence?
**P2 · Prediction.** What do I expect, and what would the rival explanations predict?
  State numbers where possible — a prediction with no number is hard to be wrong about.
**P3 · Kill condition.** What result makes me abandon or revise this?

**Setup.** Model + checkpoint + layer · data + n · what exactly was ablated/varied ·
  metric and its definition · seeds · anything held fixed.

**R1 · Result.** The numbers. Plot path if any.
**R2 · What I verified MYSELF.** Which numbers I recomputed independently, and how.
  (If Claude produced both the number and the check, the check is worthless — say so.)
**R3 · What could still be wrong.** Alternative explanations not yet excluded.
  Confounds. Anything I chose not to control and why.
**R4 · Decision.** Kept / revised / killed. What happens next because of this.

**LLM use.** What Claude did here, what I checked, what I did NOT check, and how
  surprised I would be to find a major error in each part.

---

### SCOUT-01 — is cricket usable as the topic filter?   `2026-08-19` · `clock: not started (scouting)`

**P1 · Question.** The Gemma-3-12B NLA was trained on `openbmb/Ultra-FineWeb` (split `en`,
docs 0–99,999, `positions_per_doc: 10`, `chunk_size: 256`, seed 42, stage-2 explanations
from `claude-sonnet-4-6`). Is there enough **cricket** text in that corpus to build a
held-out topic corpus from it?

**P2 · Prediction.** Cricket is common on the open web; expected a usable density.

**Setup.** Streamed `data/ultrafineweb_en/ultrafineweb-en-part-0500-of-2048.parquet`
(held out — training used docs 0–99,999, which live in shard 0001). Scanned 4,000 docs.
Counted docs containing cricket-specific terms.

**R1 · Result.** **Zero.** batsman/batsmen 0, wicket 0, innings 0, bowler 0.
"cricket" in *any* sense: 3/4000 (0.07%) — likely the insect or the metaphor.
football 15, soccer 10, basketball 7, sport(s) 82 (2.05%).
Median doc length 2186 chars. Quality score: min 0.501, median 0.781, max 1.000.
Sample docs: exam accessibility planning · a bilingual corpus-alignment tool · art
conservation methodology · stock-trading statistics · Indian Railways RDSO standards.

**R2 · What I verified myself.** Reproduced across 3 further held-out shards (0501, 0900,
1500; n=3000 each). cricket 0.00% in ALL four shards. biology 2.60–3.23% (mean 2.95),
education 2.23–2.73% (mean 2.49), history 0.71, engineering 0.57, programming 0.35,
ML 0.14. Densities are stable.
**CAVEAT — this is reproducibility, NOT independent verification.** Claude wrote both the
original count and the check, so a bug in the term lists or filter logic reproduces
faithfully across shards. **Still open: my own hand-tally** of 40 docs
(`data_scouting/sample_random_40.md`) — a keyword filter is structurally blind to documents
*about* a topic that don't use its vocabulary.

**R3 · What could still be wrong.** Only one shard sampled. Term list may be too narrow
(though batsman/wicket/innings/bowler returning exactly zero is hard to explain by
vocabulary choice). Shard 0500 might be unrepresentative.

**R4 · Decision (updated after READING the data, not just the density table).**
**Cricket killed as the topic.** Ultra-FineWeb is an
educational/technical corpus; sports is filtered out. At 0.07% for any sense of the word,
~300k docs would yield a couple of hundred candidates, most not the sport.
→ Re-scout for a topic that is BOTH dense in this corpus AND gradable by me in <30s.


**ADDENDUM 2026-08-19 — reading the sample reversed the recommendation.**
Sample openings (shard 0700): Anglo-Saxon boar helmets · copper casting alloys · mice
trained for sugar rewards (Calakos/Yin) · paternity law · CXCL16/oxLDL in diabetic
nephropathy · Windows 10 basics · coral reefs · SEGA CD 32X stub · router config ·
circuit diagrams.
- **Most of the corpus is gradable by any literate adult** — the topic filter existed to
  guarantee I could grade it, and that guarantee is largely free.
- **The biology hits are the HARDEST to grade, not the easiest** (dense biomedical jargon,
  alias-rich). The "biology because I'm comfortable with jargon" reasoning looks weak
  against the actual documents.
- **→ Consider NO topic filter.** Random held-out Ultra-FineWeb gives zero scanning
  overhead (vs ~30x for a 3% filter), the *exact* training distribution with no topic
  shift at all, and a diverse claim mix.
- **Protocol rule this surfaced:** one sampled doc states the aquatic biome is 75% of the
  planet — false about the world, irrelevant to us. Graders answer *"is this claim
  supported by this passage?"*, NEVER *"is this claim true?"* Must be explicit in the
  grading instructions or a second grader will get it wrong exactly where it matters.

**Cost of finding this out: ~5 minutes, zero GPU.** Scouting before committing works.

**LLM use.** Claude wrote the streaming/filter script and ran it; I have not yet
independently re-run the count. The headline (four cricket terms at exactly zero) is a
simple keyword count — low risk of subtle error, but unverified by me as of writing.

---

### SMOKE-01 — first pod session: verify the pipeline, test cricket viability   `2026-08-20` · `clock: NOT started (setup)`

**Pod.** A100-SXM4-80GB (SM 80, Ampere), 150 GB container disk, no network volume,
COMMUNITY $1.39/hr. torch 2.9.1+cu128 · transformers 5.3.0 · sglang 0.5.10.post1.
53 min, **~$1.23**. Chose Ampere over the RTX PRO 6000 Blackwell used on 2026-06-07
specifically to avoid the flashinfer/fa3 issues my own skill file records — worked, no
CUDA problems at all.

**P1 · Question.** Does the pipeline reproduce on a fresh machine, and is cricket text
in-distribution for this NLA?

**TEST A — tokenizer drift.** Marker at position 93; left 236813, marker 246566, right 954
— **all MATCH the sidecar**, exactly 1 occurrence, on transformers 5.3.0. (Same result
locally on 4.49, so the ids are stable across both.) ✅

**TEST B — embedding norms vs injection_scale. REFUTES A CLAIM CLAUDE MADE.**
raw table row norm median **0.9508** (p75 0.9582) · × embed_scale 61.97 = **58.92**
· injection_scale **80000** · **ratio = 1357.7x**.
→ Claude assumed the injected vector was "meant to be comparable to an ordinary token".
  **Wrong — it is ~1358x larger.**
→ Worse, this undermines Claude's *mechanism* story. The attention demo (62x norm captures
  80% of attention) **had no layer norm in it**. Gemma pre-norms every position
  independently at each block input, which strips per-position scale before Q/K/V. So raw
  norm cannot dominate attention that way. **The value of injection_scale is right; the
  explanation of why was not.** Needs proper investigation if it becomes load-bearing.

**TEST C — activation norms (layer 32, last position).**
THEIR_EXAMPLE 78890 · Laxman 72732 · Test cricket 69062 · Eden Gardens 68607 ·
Tendulkar 68602 · Dravid 64516.
→ **Validates the p75 heuristic**: real norms are 64.5k–78.9k and injection_scale=80000
  sits right at the top of that range, exactly as "75th percentile of activation norms"
  predicts.

**R1 · ROUND TRIP — the headline.**
| passage | T | pos | mse | cos |
|---|---|---|---|---|
| THEIR_EXAMPLE | 89 | 88 | 0.006 | **0.997** |
| Rahul Dravid | 148 | 147 | 0.006 | 0.997 |
| Test cricket | 116 | 115 | 0.005 | 0.997 |
| Eden Gardens | 96 | 95 | 0.009 | 0.996 |
| Sachin Tendulkar | 180 | 179 | 0.009 | 0.995 |
| VVS Laxman | 255 | 254 | 0.011 | 0.994 |

- **0.997 on their example = my recorded 2026-06-07 result exactly.** Pipeline verified on
  a different machine, different GPU arch, fresh install.
- **cricket cos mean 0.996 (0.994–0.997) vs their 0.997 → CRICKET IS IN-DISTRIBUTION.**
  The objection that killed cricket was about Ultra-FineWeb's *contents*, not about cricket
  being alien to the model. Now settled with a measurement, not an argument.
- Confirmed from running code: AR = **33 layers**, d_model 3840, mse_scale 61.97.
  `model.norm.weight | MISSING` appeared exactly as my skill file predicted (critic
  replaces final LayerNorm with Identity).

**R1b · FIRST CONFABULATIONS — verified against the passages myself.**
- **Laxman passage** (Laxman x3, Dravid x0): AV said *"a cricket match involving Dravid"*,
  *"biographical note about Dravid's performance"*. **Dravid appears ZERO times.**
- **Dravid passage** (Dravid x1, Tendulkar x0): AV said *"factual prose about Sachin
  Tendulkar"* — then in the NEXT sentence correctly *"summary statement about Dravid"*.
  **True and false claims side by side in one explanation.**
- Invented quotes: *"He is considered one of the greatest batsmen in cricket history"*,
  *"He served as a middle-order batsman throughout his career"* — neither in the passage
  (which is about Wisden awards, the Bradman Oration, balls-faced records).
→ The paper's **64/28/24 pattern reproducing in the first six activations**: theme claims
  right, entity claims wrong-but-right-kind-of-thing, quote claims invented.
→ **Vindicates the cricket argument:** I spotted the player swap instantly. A grader
  without cricket knowledge, seeing "a biography of Dravid" against a Laxman passage —
  two Indian batsmen of the same era — could easily wave it through.

**R1c · K=8 RESAMPLING (same activation, T=1).**
Laxman: cos sd 0.00031, mse mean 0.0101 sd 0.00062, **sd/mean 6.1%**, len 582–678.
Their example: cos sd 0.00023, mse mean 0.0064 sd 0.00047, **sd/mean 7.3%**, len 665–755.
7 of 8 Laxman explanations open with the identical phrase. **The AV is very stable at T=1**,
and the narrow length range means the short-vs-long-explanation ablation confound is mild.

**R2 · What I verified myself.** Checked the AV's entity claims against the actual passage
text by counting name occurrences — that is how the Dravid/Tendulkar/Laxman swaps were
established, not by taking Claude's word.

**R3 · What could still be wrong / does NOT hold.**
- **The K=8 test measures the WRONG QUANTITY for H2.** H2 needs the variance of **Δ** — the
  *paired* difference between full and claim-ablated reconstruction, both from the SAME
  explanation, where most resampling noise cancels inside the pair. I measured the spread
  of raw mse across *different* explanations. Rough arithmetic: the paper's true-vs-false
  gap ≈ 0.8% of mse ≈ 0.00008 absolute, vs measured sd 0.00062 — about **8x larger**. If Δ
  noise were really that big, K≈64 would be needed. If pairing cancels most of it, far
  fewer. **Which is true is the actual first experiment, and it belongs in Session B.**
- n=1 passage for the non-cricket group. Laxman consistently 0.995 vs French Revolution
  0.997 across all 8 resamples each — real but tiny, and length (255 vs 89 tokens) is
  confounded with content.
- Wikipedia is almost certainly in Gemma-3's pretraining, so famous-entity passages let the
  model produce world-true claims absent from the prefix. A live instance of H1's ambiguity.

**R4 · Decision.** Pipeline verified. **Cricket adopted as viable** (measured, not assumed).
Next: corpus + blind-labelling harness offline (free), then Session B for ablations.

**Bonus correction.** chars/token is **~4.0** for entity-dense cricket text, not the 4.9
Claude derived from plain prose — proper nouns and numbers tokenize into more pieces. So
passages clear the 50-token bar at ~200 chars, and the median Ultra-FineWeb doc is ~546
tokens, not 450.

**LLM use.** Claude wrote every script and drove the pod end to end (control plane via
RunPod MCP + SSH). I supplied the HF token myself; it was piped to the pod without its
value entering Claude's context. I have NOT independently re-run the cos numbers — but I
did verify the confabulations by hand against the passages, which is the load-bearing
claim. Test B's refutation of Claude's own assumption came from a measurement Claude
proposed against itself.

**Artifacts.** `mats_2027/runs/2026-08-20_smoke/` — 208 KB total (roundtrip_results.json,
variance_K8.json, acts.npy, passages.json, + logs). Predicted 0.51% of the 64.9 GB
downloaded; actual ratio held.

---

### PILOT-03 — Stage 3 decomposition prompt, tuned and FROZEN on the 6 pilot explanations   `2026-08-21` · `clock: ~1h`

**P1 · Question.** Can an LLM cut AV explanations into atomic, gradable claims faithfully — without
adding, dropping, or distorting — in the paper's output format?

**Setup.** `claude-haiku-4-5` (same family the paper used), **structured outputs**
(`output_config.format` json_schema; `level` and `subtype` are schema enums so stray categories
are impossible). Input: the 6 SMOKE-01 explanations (pilot set — never in the analysis). The
paper's decompose prompt was **never published** (checked paper, appendix, official repo,
EasyNLA); this is a reconstruction matching their output format. ~4 sync iterations, cents.

**R1 · Iterations.** v1: 58 claims — bare names as "claims" ("Dravid"), triple-counting, and
**one decomposer confabulation** (AV wrote "the India vs" and was cut off; Haiku completed it
to "India vs Pakistan"). v2 fixed all three → 53. v3 (structured outputs) → 50, content
unchanged. **v4 (forward-looking claims excluded) → 46. FROZEN.**

**R1b · The position finding (user's question).** "The text sets up a concluding statement" —
when does the AV say this? **Because we extract at the LAST token**, whose activation encodes
the model's plan for what comes next (the paper's Planning-in-Poetry phenomenon). So every
explanation has two kinds of claim: **backward-looking** (gradable against the prefix) and
**forward-looking** (the prefix *cannot* contain what follows it — grading it is a category
error). Rule adopted: forward-looking claims are dropped, not graded. This reverses Claude's
initial lean to keep them. **It also locates the user's prefix/continuation idea**: forward-
looking claims have free ground truth in the *rest of the paragraph*, a third label source
that does not come from Haiku.

**R2 · Verified by me.** Read all 46 claims against their source explanations (REVIEW file).
Claude's probes: all 12 quote-claims verbatim-present in the AV text (no invented quotes);
no near-duplicates >0.5 Jaccard; 1 borderline forward-looking claim remains (1/46, left).

**R3 · What could still be wrong.** (a) Residual soft double-counting (the Tendulkar swap
appears once as THEME, once as ENTITY) — dedupe downstream, report the rate. (b) The prompt
is a reconstruction, not the paper's; state in the write-up. (c) Omissions are the one error
class probes cannot catch — only a human reading for "is every assertion represented" can.

**R4 · Decision.** Prompt frozen. Batch path (50% off, `params_for()` shared with sync) to be
implemented and run AFTER Session A. Claims/explanation ≈ **8–9, not 2–3** → ~52k claims for
5,760 explanations → LLM judge + human validation on a sample is mandatory, not optional.

**LLM use.** Claude wrote the prompt and the probes; Haiku executed decomposition. The user
caught (1) that claims require explanations to exist — a sequencing gap in Claude's plan, and
(2) the position/forward-looking question that changed the grading rule. The decomposer-
confabulation ("India vs Pakistan") was found by Claude reading raw output against the AV text.

---

### SIM-01 — analysis dry-run on planted worlds   `2026-08-21` · `clock: ~1h`

**P1 · Question.** Before any real Δ exists: can the analysis code tell a planted mixture
(World A, H1 true: related-false = 35% at 0.31 + 65% at 0.06) from a planted single hump
(World B, H1 false: one hump at 0.14)? Noise sd per resample = 0.8, the ~5x-effect estimate
from the fennec example. n=400 claims/category.

**R1 · H2 code PASSES.** sd of claim-mean Δ: 0.82 → 0.60 → 0.43 → 0.32 for K=1,2,4,8; log-log
slope **−0.46** vs −0.50 expected. The 1/√K measurement works.

**R1 · H1 code CANNOT SEPARATE THE WORLDS — and it is not a code bug.**
| K | sd after averaging | verdict A | verdict B |
|---|---|---|---|
| 8 | 0.302 | one | one |
| 16 | 0.229 | one | one |
| 32 | 0.182 | one | one |
| 64 | 0.154 | one | one |
The planted modes are 0.25 apart; separation needs gap ≳ 2× averaged sd. At K=64 (8× the AR
cost) sd is still 0.154. **K in the hundreds would be needed. Under the estimated noise, the
bimodality test is infeasible at any affordable K.** The information is not there to recover.

**R2 · Verified by me.** Histograms printed and inspected: World A is visibly one broad bump.
`diptest` (standard library) replaced a hand-rolled O(n²) dip that timed out.

**R3 · What could still be wrong — THE KEY ONE.** The 0.8 noise sd is a back-of-envelope from
ONE transcript. It is the most important number in the project and **has never been measured
directly**. SMOKE-01's K=8 test measured raw-mse spread, the WRONG quantity: paired Δ (full vs
one-claim-ablated, same explanation) could be far smaller because the pair shares an
explanation. If paired sd is ~0.2, bimodality becomes viable at modest K.

**R4 · Decision — this is FAILURE MODE #1 ("inconclusive, found late") caught EARLY, for free.**
Options, user's call:
1. **Measure paired Δ noise FIRST** on the 6 pilot activations (AR-only mini-session, 24 GB
   card, cheap). Decides everything downstream. ← recommended
2. Change the H1 test from *shape* (bimodality) to *correlation* (does Δ track the `L`
   lookup-flag / alias status among related-false claims?). Correlation survives noise that
   kills mode-finding.
3. Narrow the claim class to where the mixture should be starkest (ENTITY/person).
**Session A is NOT blocked by this** — explanations are needed either way — but the analysis
design is, and the choice should be made before Session B.

**LLM use.** Claude wrote the simulation and the analysis. The result — that the test as
designed cannot work — came from Claude's own code disproving Claude's own plan. That is the
simulation working as intended.

---

### NOISE-01 — measure the paired-Δ noise directly (in progress)   `2026-08-21` · `clock: running`

**P1 · Question.** What is the sd of Δ for the SAME claim across K resamples of the same
activation? This is the number SIM-01 showed the whole H1 test hinges on, and it has never
been measured — the paper decoded each activation once; SMOKE-01's K=8 measured raw mse
spread (wrong quantity); the 0.8 in the simulation was a ratio extrapolated from 11 claims.

**P2 · Prediction.** None committed. Two live possibilities: paired sd ≈ 0.8 (bimodality test
infeasible; switch H1 to a correlation test) or paired sd ≪ 0.8 because full and ablated
reconstructions share an explanation and the noise cancels (original plan viable).

**P3 · Kill condition.** Not a hypothesis test — a dial reading. It decides the design.

**R1 · RESULTS.** A40 session, 35 min, **$0.23**. 48 explanations (0 CJK) → 401 claims (0 parse
problems) → 447 variants (48 intact + 399 single-claim ablations) → AR-scored.

Intact mse per activation, mean ± sd across K=8: 0.0057–0.0101, sd 0.0003–0.0008 (matches
SMOKE-01's K=8 on the two it covered — the pipeline is consistent day to day).

**THE NUMBER:**
```
median paired-Δ sd (same claim across resamples, n=6 claims, 3–5 resamples each) : 0.00115
spread of claim MEANS (between-claim 'effect' scale)                             : 0.00082
noise / effect, measured on this data                                            : 1.41×
```
**SIM-01 assumed 4–5×. Measured: ~1.4×.** That is the difference between "bimodality
infeasible at any K" and "feasible at modest K" — at 1.4×, K=4 averaging brings noise to
~0.7× the effect. **The design does not need to change — yet.** (See R3.)

**R1b · UNEXPECTED — "removal improves reconstruction" is common:**
```
Δ = mse(ablated) − mse(intact), all 399 ablations:  mean +0.00115  sd 0.00187
  Δ > 0 (removal HURT reconstruction) : 81.2%
  Δ < 0 (removal IMPROVED it)         : 18.8%
```
Nearly 1 in 5 claims, removed, make the AR reconstruct the activation BETTER. That is Neel's
exact phrasing ("claims that can be removed to improve reconstruction") — which the paper
never reported (they reported only "false claims hurt less"). First pass, n=399.

**R1c · The 2×2 gets real numbers in its cells (n=3 each — illustration, not result):**
```
[Laxman]  "the text mentions dravid"      FALSE (Dravid absent)   mean Δ −0.00003  ≈ 0
[Dravid]  "the text mentions dravid"      TRUE                    mean Δ +0.00006  ≈ 0
[Dravid]  "the text is about tendulkar"   FALSE                   mean Δ +0.00098  LOAD-BEARING
```
The same sentence, true in one passage and confabulated in the other, gets IDENTICAL weight
from the AR (≈0). The confabulated Tendulkar claim carries MORE weight than the true Dravid
claim — a Bhaskar candidate: text-false, but the activation plausibly encodes
"Tendulkar-adjacent Indian batsman" and the AR uses it. (Tendulkar sd 0.00136 > its mean.)

Δ by level: THEME +0.00087 (n=191) · ENTITY +0.00087 (n=70) · DETAIL +0.00168 (n=138).
DETAIL claims carry ~2× the weight. Consistent with the paper's finding that specific
claims matter more to reconstruction when true — but unlabelled here, so not a test.

**R2 · Verified by me.** Scripts are in `mats_2027/pipeline/` (`build_ablations.py`,
`score_ablations.py`, `noise_analysis.py`) for line-by-line re-verification — TODO before
the write-up. The sign of Δ was derived from the definition (mse is an error; Δ<0 ⇒
removal helped) and matches the paper's chart under the opposite sign convention.

**R3 · What could still be wrong.**
- **n=6 recurring claims with 3–5 resamples each.** sd from 3 points is rough. First
  reading, not a measurement to build on. The 1.4× could move with more data.
- Recurrence was matched by EXACT normalised text, so "mentions Dravid" ≠ "discusses
  Dravid". A fuzzier match would find more recurring claims. Deliberately strict for the
  first pass.
- All last-position (logged up front). Last-10-positions follow-up is the fix.
- Ablation = delete the whole carrier sentence. Rival R2 (prose damage) is NOT controlled
  here. Delete-vs-replace is still owed.
- Claims are UNLABELLED. "FALSE/TRUE" above is my hand-check of two known cases, not the
  harness.

**R4 · Decision.** Proceed with the original design: H2 (K-averaging) then H1 (bimodality),
because the measured noise ratio makes it plausible. **But** widen the recurrence match and
run the last-10-positions extraction to firm up the 1.4× before committing Session A money.
Record the 18.8% "removal improves" rate as a standalone finding to track.

**LLM use.** Claude drove the pod (control plane + SSH), wrote all scripts, and ran the
analysis. I have NOT yet re-verified the scripts line by line — flagged as TODO. The
interpretation of the Dravid/Tendulkar cells is mine and Claude's jointly; the labels on
those three claims are my hand-check from SMOKE-01.


**Setup.** Pod: A40 48 GB, SECURE, CA-MTL-1, $0.44/hr. torch 2.9.1+cu128 · transformers 5.3.0
· sglang 0.5.10.post1 · compute (8,6). **No base model** — the 6 SMOKE-01 activations are
reused from `acts.npy` bit-for-bit (5 cricket Wikipedia + the French Revolution reference),
all at the **last position** (pos = T−1). AV at T=1.0, **K=8** per activation → 48
explanations. Then (laptop) Stage 3 decompose → claims; `build_ablations.py` removes the
carrier sentence of each claim → variants. Then (pod, AR only) score every variant → paired
Δ = mse(ablated) − mse(full) within the same explanation; sd across the k in which the same
claim recurs.

**LIMITATION LOGGED UP FRONT (user's call to log it).** All 6 activations are LAST-position,
so every explanation is heavy with forward-looking content. The measured noise is therefore
for last-position activations specifically; the main experiment uses the last 10 positions.
The number *should* transfer — that is an assumption, not a result.
**→ FOLLOW-UP (user's idea): extract the last 10 positions for all 6 pilot passages and save
them to .npy, so this measurement can be repeated across positions. After NOISE-01, not during.**

---

### SOURCE-01 — the paper's confabulation pipeline, recovered from its own HTML   `2026-08-22`

Not an experiment — a source finding, from `sources/nla_paper_2026-05-07.raw.html` (the
fennec s244 widget) and `sources/nla_claim_analysis_fennec_s244.json`. The paper's prose
never describes this; the shipped grader OUTPUTS do.

**Their pipeline is four LLM calls per explanation**, grader stamped
`claude-haiku-4-5-20251001`:
| field | content |
|---|---|
| `decompose_response` | `<claim_N level="THEME" subtype="genre">...</claim_N>` + `<total_claims>11</total_claims>` |
| `verify_response` | `<verdict_N>SUPPORTED</verdict_N>` / `NOT_IN_TEXT` |
| `vibe_response` | `<relatedness_N>DIRECT</relatedness_N>` / `ADJACENT` — emitted ONLY for the NOT_IN_TEXT claims |
| `match_response` | `<appears_in_N>1,2,4,6,7,8</appears_in_N>` |

**1. Recurrence matching is a MODEL CALL, not string matching.** `match_response` is the
recurrence signal. Our `build_ablations.py::norm()` — `re.sub(r"[^a-z0-9 ]","",t.lower())` —
is the crude version, which is why only 6 of 401 claims cleared the >=3-appearances bar in
NOISE-01. **The 1.4x is computed on that small, phrasing-biased subset.**

**2. "Final-token claim list" is asymmetric.** They decompose the FINAL-position explanation
only; the other nine are searched as raw prose, never decomposed. `position_count` = how many
of the 10 the matcher found the claim in. *We are richer here:* POS-01 has K=4 at all 10
positions (240 explanations), so we can measure recurrence across positions AND across
resamples at the same position. They could not.

**3. Their ablation is a REWRITE, not a deletion.** `rewritten_text` is the full explanation
with one claim surgically excised and the prose reflowed. Compare the same explanation
ablated at claims 1 / 2 / 3 — one fact removed each time, everything else intact.
**We delete the whole carrier sentence**, which takes neighbouring claims as collateral and
damages prose — i.e. Rival R2 (coherence) is uncontrolled in our method and controlled by
construction in theirs. **Consequence: our Δ and their Δmse are not the same measurement.**

**4. The prompts are NOT published.** The HTML has `decompose_response`, `verify_response`,
`vibe_response`, `match_response` and NO corresponding `*_prompt` keys. Only four unrelated
widgets ship prompts (SAE agreement, factual groundedness, writing quality, eval awareness).
The "factual groundedness" prompt is the closest published relative of our Stage 3 prompt.

**5. A discrepancy in their own worked example.** Claim 6 = "Text mentions a Korean historical
figure named Jungjong"; matcher returned `appears_in: [2,3,6]`. Literal "Jungjong" occurs in
explanations 1,2,6,7,8. Explanation **1** asserts it ("matching earlier 'Joseon king Jungjong'
usage") and is the very explanation the claim was extracted from, yet is excluded; explanation
**3** contains no Jungjong at all (it is about Jo Gwang-jo and the 1519 purge) yet is included.
**CAVEAT: their matching prompt is unpublished** — under a stricter definition of "appears in"
(bullet's main point, not a passing mention) the exclusion could be correct. Flagged as a
discrepancy to check, NOT a proven error. n=1 claim, one example.
→ Same gap as the Haiku claim judge: matcher reliability is never reported, and recurrence is
one of the three signals we plan to combine.

**6. Their data contains a removal-improves case:** fennec claim 3, `delta_mse_pct: -1.5`.
So NOISE-01's 18.8% does not contradict them — it is present in their published example and
absent from their prose.

**Actions parked for the NEXT experiment (not DECOMP-01):**
- `norm()` -> an LLM matcher for recurrence.
- delete-carrier-sentence -> rewrite-claim-out ablation (kills R2 for free, and makes Δ
  comparable to their Δmse).

---

### DECOMP-01 — Stage 3 claims for all 240 POS-01 explanations   `2026-08-22` · `clock: running`

**P1 · Question.** Does the frozen Stage 3 decomposition prompt (PILOT-03) still work on
explanations from NON-final token positions? It was frozen on last-position explanations
only, and mid-position explanations are structurally different (they describe a mid-sentence
continuation rather than a completed passage).

**P2 · Prediction (recorded before running).** Claims/explanation stays flat across offsets.
Risk: mid-position explanations are vaguer, so DETAIL claims could thin out and claim counts
drop at earlier offsets.

**P3 · Kill / flag condition.** A collapse in parse rate or in claims/explanation at
non-final offsets ⇒ the prompt does not transfer, and that must be fixed before any AR spend.

**R1 · RESULTS. PASS.** `runs/2026-08-22_pos10/claims.parquet` — **2065 claims from 240/240
explanations**, model pinned and API-resolved to `claude-haiku-4-5-20251001`.
Claims/explanation: min 4 · p25 7 · median 8 · p75 10 · max 16.

```
offset n_expl  claims mean/expl    sd   THEME/ENTITY/DETAIL %
    -9     24     199      8.29  2.35   47.7 / 19.6 / 32.7
    -8     24     208      8.67  2.08   45.2 / 24.0 / 30.8
    -7     24     202      8.42  2.30   35.6 / 23.8 / 40.6
    -6     24     201      8.38  2.10   39.3 / 18.9 / 41.8
    -5     24     206      8.58  1.35   46.1 / 18.9 / 35.0
    -4     24     217      9.04  2.60   41.5 / 17.5 / 41.0
    -3     24     205      8.54  2.62   41.5 / 20.5 / 38.0
    -2     24     220      9.17  2.01   37.7 / 25.9 / 36.4
    -1     24     200      8.33  1.52   43.0 / 18.5 / 38.5
    +0     24     207      8.62  2.22   52.2 / 18.8 / 29.0
```
Flat. No degradation away from the final token. The only visible tilt is at offset 0
(52.2% THEME / 29.0% DETAIL vs ~41/38 elsewhere) — n=24 per offset, so NOT a result, but
worth re-checking once verdicts exist.

**R1b · A real failure mode found: Haiku degenerates on ~0.8% of calls.**
Two of 240 explanations (0.83%) produced runaway output — the model emitted dozens of
vacuous near-duplicate paraphrases ("The text presents information.") instead of stopping:
- `VVS Laxman pos=250 k=3` ran past `claim_idx: 104` and hit `max_tokens`, killing the whole
  run before anything was written.
- `Rahul Dravid pos=145 k=1` emitted **95 claims while its own `total_claims` field said 4**,
  and finished under the cap — so it did NOT truncate and would have entered the dataset
  silently if only the stop_reason were checked.

Re-sampling the SAME input returned clean output both times (14 and 10 claims). So it is
sampling noise, not a property of the explanation, and a **retry** is the right fix — not a
bigger `max_tokens`, which would have let the 95-claim case through at full length.
**The reliable detector is the model's own declared `total_claims` disagreeing with the
number of claim objects it emitted.** Both are now retry conditions in `stage3_decompose.py`.

**R2 · What I changed in the pipeline.** `stage3_decompose.py`: model pinned to the dated
snapshot (was the floating alias `claude-haiku-4-5`) and the API-resolved id is now recorded
per row; `max_tokens` 1200 -> 4000; serial loop -> `ThreadPoolExecutor` with `--workers`
(`ex.map` preserves input order, so output is deterministic); per-item failures no longer
abort the run (they are collected to `*_failures.json`); retry on truncation OR declared-count
mismatch. Backup of the pre-edit file: `stage3_decompose.py.bak`.

**R3 · Integrity checks run on the merged table.** 240/240 explanations present · 0
declared-vs-actual mismatches · 0 duplicate `claim_idx` within an explanation · 2065/2065
unique `claim_id` · single resolved model across all rows.

**R4 · What could still be wrong.**
- The two degenerate calls were re-run and merged, so those two explanations were decomposed
  at a different moment from the other 238. Same model, same prompt, same params.
- Claims/explanation being flat does not prove claim QUALITY is flat across offsets. Only the
  verdict stage will show that.
- **NOW CHECKED (user challenged the concern; it does not materialise).** The one judgement
  call in the prompt is "EXCLUDE forward-looking claims". Mid-position explanations are
  dominated by "what comes next" language, so that rule does far more work at offsets -9..-1
  than it was tuned on, and flat claim COUNTS are consistent with the rule leaking. Measured
  directly with a keyword detector over all 2065 claims:
  ```
  offset  -9   -8   -7   -6   -5   -4   -3   -2   -1   +0     overall
  rate   2.0% 1.4% 2.5% 2.5% 0.5% 1.4% 1.0% 1.4% 0.5% 2.4%     1.5%
  ```
  No trend with position (highest at -9/-7/-6 AND +0, lowest at -5/-1); a breakdown would show
  a monotone climb toward the early offsets. **Upper bound**, because the regex over-flags:
  most hits are "The text signals a justification for the red ball..." — a claim about content
  PRESENT in the text that merely uses the word "signals", not a forward-looking claim.
- 2065 claims is ~2k, which is the n SIM-01 needs to be re-run at.

**Cost.** ~252 Haiku calls (240 run + 8 stratified smoke + 2 retries + 2 debug). Tokens not
logged per call; from observed usage (~800 in / ~550 out) this is order 0.2M in / 0.14M out.
Well under $1 — not separately metered, so recorded as an estimate, not a measurement.
Wall clock 93 s at 8 workers. No GPU, no pod.

**Decision: batch API not used.** 50% discount on a <$1 job against unbounded latency, on an
experiment with a live pass/fail that might need re-running. Sync + 8 workers instead.

**LLM use.** Claude made the pipeline edits, ran the job, and ran the integrity checks. The
degeneration diagnosis came from re-sampling the failing input twice and comparing — evidence,
not inference. I have NOT re-verified the integrity script line by line (same standing TODO as
`noise_analysis.py`).

**Next (blocked on nothing):** Stage 4 text verdicts on these 2065 claims.

---

### JUDGE-01 - S/C/N verdicts on all 2065 claims   `2026-08-22` · `clock: running`

**PROVISIONAL - the judge is UNVALIDATED.** AG grades a 150 + 30-retest stratified sample
tomorrow; every number here carries an unmeasured error bar until then. Deliberate ordering:
the AR run does NOT depend on labels (Delta is computed from claims, never verdicts), so the
GPU spend is label-independent and re-labelling costs a numpy re-run, not a pod.

**P1 · Question.** Label every claim against the exact prefix the model had read.

**Scale: S / C / N**, identical to `harness/grade.html`, so judge and human are directly
comparable. The paper's shipped verdicts are binary (SUPPORTED / NOT_IN_TEXT); C+N collapse
to their "false" bucket. The split is kept because **H1 needs it** - a claim merely ABSENT
can still be encoded in the activation; a claim the text CONTRADICTS is far less likely to be
a faithful readout. Collapsing is one-way, so collecting three costs nothing.

**Conventions** are AG's, frozen in `grade.html` on 2026-08-21 and extended 2026-08-22 after
the warm-up, copied verbatim into the judge prompt so both graders measure one scale:
grade against the passage as written · vague is not false · predictions about what comes next
- grade only the factual part, else N · alias/translation -> S if the referent is unambiguous
· absent AND undisputed -> N, never C · **QUOTE claims -> S only if the quoted string is
actually present (ellipsis and typos allowed); topical resemblance is NOT enough.**

The quote rule was added TODAY, after AG graded `The text contains the sentence "This series
was remarkable...He was the man of the series"` as SUPPORTED against a passage reading
"This series is regarded as one of the greatest series of the century." Rationale AG accepted:
otherwise "the text contains sentence X" is unfalsifiable, and **invented quotes are one of
the paper's own headline confabulation types** - grading them leniently would erase the exact
phenomenon being measured. `quote` is 437/2065 = **21% of the dataset** (433 of them DETAIL),
so this single convention moves the DETAIL rate more than any other choice.

**Scale check before spending:** judge vs AG on the 12 warm-up claims = **10/12**. Both
disagreements informative: one was the quote claim above (the frozen rule overriding AG's own
earlier keystroke - working as intended), one a vague THEME claim on the "vague is not false"
line. n=12 - a scale check that judge and human grade the same thing, NOT an accuracy measure.

**R1 · RESULTS.** `runs/2026-08-22_pos10/verdicts.parquet` - 2065 verdicts, 0 failures, 273 s,
~$1. SUPPORTED 1070 · NOT_IN_TEXT 956 · **CONTRADICTED 39 (1.9%)**.

**THE SPECIFICITY ORDERING REPLICATES:**
| level | ours (Gemma-3-12B NLA, cricket Wikipedia, 10 positions) | paper (Opus 4.6 NLA, Common Pile, final position) |
|---|---|---|
| THEME | **69.1%** supported (n=887) | 64% |
| ENTITY | **43.8%** (n=427) | 28% |
| DETAIL | **36.0%** (n=751) | 24% |

THEME > ENTITY > DETAIL holds across a different NLA, base model, corpus, token positions and
an independently written judge prompt. Our rates are higher at every level; candidate causes
NOT yet separated - different NLA, cricket Wikipedia being far more formulaic than Common
Pile, our conventions keeping vague THEME claims, our judge prompt.

**CONTRADICTED is rare (1.9%)**, confirming AG's warm-up instinct (0 of 12). Those 39 are the
strongest confabulation cases in the set.

**R1b · POSITION HYPOTHESIS: DEAD on this data.** (AG's idea, floated 2026-08-22 04:00,
deliberately never promoted to a numbered hypothesis.) Prediction was MORE confabulation at
later positions.
```
offset   -9    -8    -7    -6    -5    -4    -3    -2    -1    +0
supp   45.7  48.6  56.4  51.2  52.4  47.5  55.6  54.1  58.0  48.8   (all +/- ~6.8)
early (-9..-5) 50.9% +/- 3.1 · late (-1..0) 53.3% +/- 4.8 · diff +2.4pp, pooled SE ~2.9pp
```
Not significant, and the SIGN IS OPPOSITE to the prediction. At n=2065 this is not an
underpowered null. AG called this himself from eyeballing the Laxman explanations before any
data existed ("i found confabulations even in earlier positions").
**Bounds:** judge unvalidated; 6 Wikipedia passages; last 10 tokens only. Dead for THIS
dataset, not universally.

**R1c · One cell flagged, NOT a finding.** DETAIL drops to 26.3% at offsets -1/0 vs 40.7% in
the middle (~3 SE). It is **1 of 9 cells, chosen after seeing the table**, on unvalidated
labels. Needs the agreement check and a pre-registered test before it is repeated aloud.

**R2 · What this unblocks.** The FALSE-RATE half of the H1 pooling confound is cleared:
position does not move whether a claim is true, so pooling claims across positions cannot
manufacture two humps by position. The Delta half still needs the AR run.

**R3 · What could still be wrong.** Judge unvalidated (tomorrow). Judge and Stage 3 decomposer
are the SAME model - a shared failure mode would be invisible; AG's second-judge idea
(runningdoc_AG.md, 2026-08-21) addresses this and is not yet run. Higher-than-paper rates are
unexplained.

**LLM use.** Claude wrote the judge prompt from AG's frozen conventions, ran it, and produced
the tables. AG set every convention and provided the only independent labels (12 so far).

---

### ABLATE-01 - rewrite-out ablation variants for all 2065 claims   `2026-08-22` · `clock: running`

**P1 · Question.** Build the ablated explanations the AR needs, using the paper's REWRITE
operation rather than our previous delete-the-carrier-sentence method (see SOURCE-01).

**P2 · Prediction.** Rewrites are shorter than the original by a few tens of characters;
structure preserved; no whole bullets lost.

**P3 · Failure conditions declared up front.** (a) truncation - a cut-off rewrite becomes a
SHORTER explanation and the AR scores it as a huge Delta, i.e. a fabricated result, not a
crash; (b) unchanged output - Delta == 0 for no reason.

**R1 · RESULTS.** `runs/2026-08-22_pos10/ablations.parquet` - **2305 variants (240 full +
2065 ablated)**, model `claude-haiku-4-5-20251001`, 0 API failures, 436 s at 8 workers,
~$1-2. Char delta vs original: p5 -138 · p25 -70 · **median -43** · p75 -28 · p95 -10.

Worked example (Laxman pos 254 k=0), removing "The text mentions a Test match between India
and West Indies":
```
-  ...factual cricket history format, with a summary of a Test match between India and West Indies.
+  ...factual cricket history format, with a summary of a Test series.
```
Eight words changed; the other two paragraphs byte-identical. **Deletion would have removed
the whole first paragraph and destroyed two other claims** (`encyclopedic/biographical
article structure`, `factual cricket history format`).

**R1b · A THIRD failure mode appeared that I had NOT declared: invented content.**
17 of 2065 rewrites (0.8%) came back LONGER than the original. Inspecting the worst (+174
chars): the model did not excise the claim, it wrote replacement text the AV never produced -
`"the greatest victories in Indian cricket"`, `"India's most memorable chapters in Indian
Test cricket"`. The AR would then have scored a text containing assertions that were never
in the explanation: **fabricated input producing a real-looking Delta.** Same corruption
class as truncation, arriving through the opposite door - and it would have been invisible,
because the output is fluent and plausible.
Guard added: **a rewrite that removes a claim can never be longer than the original**, so
`len(new) >= len(original)` is now a retry condition (it also catches the unchanged case).
25 corrupted variants re-run: **23 fixed**, 2 stubborn.

**R1c · The 2 that resist are a real category, not noise (0.1%).**
- `Sachin Tendulkar:171:k3:c11` - "The text discusses Sachin Tendulkar's records": the whole
  explanation is about Tendulkar's records; there is no minimal excision.
- `VVS Laxman:247:k2:c5` - "The text mentions a Test series": "Test series" occurs only inside
  a quoted sentence that is a separate claim.
These are claims for which "remove exactly this one claim" is not a well-defined operation.
Marked `valid=False` in the parquet, **not dropped** (no silent scope cuts). 2303 valid.

**R2 · Known limitation, logged before any Delta exists (H3, raised by AG).**
Single-claim ablation measures the MARGINAL contribution of a claim given the rest of the
explanation stays put. It cuts both ways, and both were observed today:
- **removes less than intended** - ablating "mentions a Test match between India and West
  Indies" leaves "West Indies" twice elsewhere, so the AR still reconstructs it -> Delta ~ 0
  for reasons unrelated to truth;
- **removes more than intended** - ablating "mentions Eden Garden" took out four mentions
  including one inside a separate quote claim -> Delta inflated.
The paper's own published example has the same behaviour (their claim-1 ablation also dropped
claim-2's content). Measured later with the matcher; not fixed here.

**R3 · NOT VERIFIED.** That exactly the target claim disappears and the others survive is
UNCHECKED. Re-decomposing and string-matching does NOT work: Haiku re-phrases the same claim
differently on a second pass ("The text is in an encyclopedic..." -> "The text is an
encyclopedic..."), so an exact-match check reported 5 claims lost and 5 gained when nothing
had changed. **This is `norm()` failing again** - the same defect that left NOISE-01 with 6
recurring claims out of 401. Verification needs the semantic matcher (SOURCE-01 build list),
which then serves three jobs: verifying ablations, recurrence across positions, and matching
claims across resamples for the paired-Delta noise.

**R4 · Also unbanked:** the rewrite is stochastic - the same (explanation, claim) rewritten
twice gave slightly different text. So the ablation contributes its own noise to Delta. The
paper had the same exposure and never reported it.

**Next:** SIM-01 re-run at the measured 1.4x noise with n~2065 (free, no GPU) - the power
gate on H1 - then batch `score_ablations.py` offline, then one AR-only pod session.

---

## Running index

| # | title | date | verdict |
|---|-------|------|---------|
| SCOUT-01 | is cricket usable in Ultra-FineWeb? | 2026-08-19 | ❌ 0.00% — killed as a *filter* |
| SMOKE-01 | pipeline verification + cricket viability | 2026-08-20 | ✅ cos 0.997; cricket 0.996 → adopted |
| PILOT-03 | Stage 3 decomposition prompt | 2026-08-21 | ✅ frozen at v4; forward-looking claims excluded |
| SIM-01 | analysis dry-run, planted worlds | 2026-08-21 | ⚠️ H2 code OK; **H1 bimodality infeasible at est. noise** — measure paired Δ first |
| NOISE-01 | paired-Δ noise, 6 pilot activations, K=8 | 2026-08-21 | ✅ **noise/effect ≈ 1.4×, not 4–5×**; 18.8% of ablations IMPROVE reconstruction |
| POS-01 | last-10-positions extraction, 6 passages, K=4 | 2026-08-22 | ✅ 60 activations + 240 explanations collected |
| SOURCE-01 | the paper's confabulation pipeline, from its HTML | 2026-08-22 | 📄 recurrence = **LLM matcher**; ablation = **rewrite**, not delete; prompts unpublished |
| DECOMP-01 | Stage 3 claims for all 240 explanations | 2026-08-22 | ✅ **2065 claims, prompt transfers across all 10 offsets**; Haiku degenerates on 0.8% of calls |
| JUDGE-01 | Stage 4 S/C/N verdicts on all 2065 claims | 2026-08-22 | ✅ **specificity replicates** THEME 69.1 / ENTITY 43.8 / DETAIL 36.0; **position hypothesis dead** (+2.4pp, SE 2.9) |
| ABLATE-01 | rewrite-out ablation variants | 2026-08-22 | ✅ 2305 variants; **0.8% invented content** (guarded); 0.1% not ablatable |

---

## Verification log (independent checks I ran myself)

Every load-bearing number gets a line here. Cheap to write, decisive in the write-up.

| date | claim being checked | how I checked it | outcome |
|------|--------------------|------------------|---------|
| 2026-08-19 | `mse_scale` in official sidecars = √d_model ⇒ direction-only MSE | computed √3584 = 59.8665 and √5376 = 73.3212 by hand against the shipped sidecar values | ✅ matches exactly |

---
### TODO (before Session A) — batch the AV client and the AR scorer
Recorded 2026-08-22 during POS-01. Both pod phases run **one sequence at a time**:
- AV: `stage2_generate.py` calls `NLAClient.generate()` per activation → ~7.6 s/explanation
  on an A40 (50 in 380 s). sglang serves concurrent requests natively; the client is the
  bottleneck, not the server. Session A = 5,760 explanations ⇒ ~12 h serial vs ~1.5–2 h
  with 8 in flight. **Check `nla_inference` client for a batch/async path first; if absent,
  fire concurrent HTTP requests (thread pool) against `/generate` — official docs:
  sglang native API.** Must keep `extract_explanation=False` + tag parsing as-is.
- AR: `score_ablations.py` calls `NLACritic.score()` per variant (batch 1; 0% GPU util
  seen in NOISE-01). Pad-and-batch the text inputs; the critic is a plain forward pass.
  ~2,000 variants today, ~50k in Session A.
Verification: batched outputs must reproduce the serial run on a 20-item subset
(AV: same distribution of lengths/no-tag; AR: identical mse to float tolerance).

---
## POS-01 — last-10-position extraction + K=4 explanations (2026-08-22, 04:00–05:15 IST)

**P1 · Question.** Does the AV's readout — and the text-false rate of its claims — depend on
*position within the passage*? All prior data (SMOKE-01, NOISE-01) is last-position only.
User's hypothesis (04:00, not yet written as H3): text-false rate rises toward the final token.
Also: is pooling claims across positions legal for H1's bimodality test (confound control).

**P2 · Prediction (recorded before any analysis).** If the last-position activation carries
predictive content, then (a) text-false rate rises toward offset 0; (b) among text-false
claims, AR weight (Δ) concentrates at offset 0. If flat → position is not a confound; pool.

**P3 · Kill condition.** Flat text-false rate across offsets −9…0 (within sampling noise of
~10 positions × 6 docs × K=4).

**Setup.** Pod A40 48 GB SECURE, CA-MTL-1, $0.44/hr, 100 GB disk, `runpod/pytorch:1.0.2-cu1281-torch280`
→ pinned torch 2.9.1+cu128 · transformers 5.3.0 · sglang 0.5.10.post1 (`pipeline/pod/pod_setup.sh`).
Base `google/gemma-3-12b-it` bf16, layer 32 (`hidden_states[33]`), **last 10 contiguous positions**
per passage, all ≥ MIN_POSITION 50. 6 pilot passages (5 cricket + French Revolution reference).
`stage2_generate.py --extract-only` then `--skip-extract --k 4` (base and AV server do not fit
together on 48 GB). AV at T=1.0, max_new_tokens 256, K=4 per activation.

**R1 · Artifacts.** `runs/2026-08-22_pos10/`: `activations.parquet` (60 rows, official Stage 0
schema + pos/title/norm), `explanations.parquet` (240 rows), all logs. **0 no-tag, 0 CJK**,
mean 656 chars, 7.56 s/explanation serial.

**R2 · Regression check vs SMOKE-01 `acts.npy` (same 6 last positions, A100 then, A40 now):**
cos 0.999992–0.999997, max|coord diff| 14–1020 on ‖h‖ ≈ 60–85k. **Close, not bit-identical.**
Inference: bf16 kernel differences across GPU architectures. ~3 orders of magnitude below AV
resampling noise (cos ≈ 0.995 between K resamples) → no downstream effect. Logged honestly.

**R3 · Activation norms vary within a passage** (Test cricket 61k–87k, Dravid 54k–74k over
10 adjacent tokens). Injection normalizes to 80000 and MSE is direction-only ⇒ the AV/AR never
see this. Open question parked: does ‖h‖ carry information (e.g. sentence-final tokens)? Free
to check from the parquet.

**R4 · Bugs hit.** (1) `.bashrc` returns early when non-interactive → `HF_TOKEN` never exported
→ 401 on gated Gemma. Fix: scripts grep the token out of `.bashrc` explicitly. (2) zsh
word-splitting of `$S` with flags (again) → inline the ssh command.

**Cost.** $0.40 (balance $8.75 → $8.35). Pod terminated 05:13 IST, 0 running.

**LLM use.** Claude wrote the `--extract-only/--skip-extract` patch and pod scripts, drove the
pod. No analysis run yet. Stage 3 decomposition (Haiku) not yet run on these explanations.

**Next (in order).** (1) Read Laxman K=4 at offsets −9/−5/−1/0 side by side — does "mentions
Dravid" appear before the final full stop? (2) Stage 3 on 240 explanations. (3) Text verdicts:
Haiku on all + user blind-grades a position-stratified sample. (4) **Position check before
pooling** — Δ/false-rate vs offset. (5) Re-run SIM-01 at the measured 1.4× noise with this n
before attempting H1's bimodality test. (6) AR scoring of ~2k ablations on a fresh pod (~$0.30).
