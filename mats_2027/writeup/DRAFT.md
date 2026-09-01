# Is the NLA activation reconstructor a weak per-claim verifier because the signal is absent, or because it is buried in noise?

*Adithya Giridharan · MATS 12.0 application task · September 2026*

---

## A. Executive summary

The NLA paper reports that the activation reconstructor is "only a weak per-claim verifier" of
the descriptions its verbalizer produces. I wanted to know which kind of weak: is there no
per-claim signal in reconstruction error at all, or is there one that is buried in noise? The
difference matters, because a buried signal is a sampling problem someone can pay to fix, and an
absent one is a dead end.

Setup: the official Gemma-3-12B NLA, layer 32, on 6 passages sampled at their last 10 token
positions with 4 resamples each — 240 explanations, 2,065 claims, each rewritten out of its
explanation and re-scored against the same activation. Δ = mse(claim removed) − mse(intact).
A second corpus, 7 pages of a 2019 biography, tested whether any of it transfers.

**What I found**

- **The signal exists and it is weak, now with a number: per-claim AUC 0.535 [0.510, 0.559].**
  The paper never quantifies "weak". Averaging over resamples moves it to 0.615, but that
  interval [0.488, 0.736] still includes chance.
- **The noise is random, not systematic — so averaging is the right lever.** Fitting
  spread(K)² = signal² + noise²/K on K=1 and K=4 predicts K=2 and K=3 to within 0.8%, and two
  independent noise estimates agree. There is a signal floor at 56% of the K=1 spread. **The
  bottleneck is not noise but recurrence**: only 110 claim-groups appear in ≥3 of 4 resamples.
  More resamples per activation would settle this in a day of GPU time.
- **My main hypothesis is dead, and I can say how dead.** I predicted that "false" claims are two
  populations — faithful readouts the text judge mislabels, plus real confabulations — which would
  make their Δ bimodal. It is one hump (dip test p = 0.992), and planted mixtures at the sizes I
  predicted are detected 86–100% of the time. A mixture below a fifth would still have been missed.
- **Removing a claim improves reconstruction 30% of the time.** The paper never reports this rate.
  Thematic claims carry ~2.7× less reconstruction weight than specific ones — but only after
  controlling for claims that quote the passage's final token; uncontrolled the gap looks 4.5×.
- **Specificity replicates on a different NLA, corpus and judge**: THEME 69.1% / ENTITY 43.8% /
  DETAIL 36.0% supported, against the paper's 64 / 28 / 24.
- **Confabulation is import, not misreading.** In both corpora, over 90% of false claims naming a
  person name someone absent from the passage entirely. 98% of false claims stay on-topic. And the
  less familiar corpus produced **more** confabulation, not less (63% false vs 50%) — refuting
  predictions I and my agent both wrote down in advance.

**What I checked myself.** I hand-graded 150 claims blind plus 30 retests (96.7% self-consistent,
88.7% agreement with the LLM judge) — the paper reports no validation of its own confabulation
judge. Several load-bearing claims failed when I checked them: a pre-registered detector rule that
turned out to fire on skew alone, a kill condition no dataset with real signal could have passed,
and a widely-repeated detail in my own notes that did not exist in the data.

**What this is not.** Everything here is black-box. The one causal intervention I attempted never
reached the representation — a control I ran afterwards showed the text edit moved the activation
0.7% as far as a single token step does — so that question is untested rather than answered.
Activation patching with hooks is the first thing I would do next.

---

## B. The question, and why it matters

The paper says "AR is only a weak per-claim verifier".

i wanted to investigate if its because of the singal being absent  or buried...

the signal being the delta(mse). in particular i hypothesise that observed delta of a claim is a sum of underlying delta and noise. by underlying delta i mean, the delta of the claim that we cud calculate in an ideal scenario with 100 percent accracy. 

now either one of the 2 cases is possible.

1. that the distribution in delta of the underlying for true and false claims is same. -> means they are indistinushable.
2. distribution may be different but the accompanying noise is overwhelming, may be due to the surrounding prose variation across resamples.

sources of confabulation :

1. residual activation vector genuinely encodes it and the AV faithfully read it.
2. AV being a LM in itself made it up.

case 1

case 2

<same as case 1>

so. in essence a text judge is primarily blind to residual activation and AV model thought process. the text judge in our case (Haiku 4.5) only receives the passage uptill the position (the prefix) and the claim..and its given 3 choices -> Supported,  or Not in TExt. 

The residual stream activation which served as the bedrock for generation of explanation using AV is available to  only AR for reconstruction in the whole system, so its in. prinicple the only instruemnt that cud tell case 1 from case 2 .

---

## C. Setup

Model details

I have used the official anthropic nla kitft/nla-gemma3-12b-L32-av` (AV) + `kitft/nla-gemma3-12b-L32-ar` (AR), Apache-2.0 . The base model that i used for this experiment was "google/gemma-3-12b-it" . ACtivations taken from layer 32 of 48 , dimension d of the vector is 3840.

The avativation verbaliser model (AV) is of same architecture as the base . The activation is injected as a single token embedding into a fixed prompt with injection_scale 80000 (available from the sidecar). I perform decoding at sampling temperature of  T = 1 (verify if my decoding terminology usage is right)

ACtivation Reconstructor model (AR) is the base model truncated to 33 blocks + a learned linear head on the final token

MSE is **direction-only**: both vectors L2-normalised to mse_scale = √3840 = 61.97 before comparison, so MSE = 2(1 − cos).

Corpus and sampling

I chose 6 passages for my experiment . 5 cricket Wikipedia paragraphs and 1 on the French Revolution — the latter is the NLA maintainers' own example passage (doc_id THEIR_EXAMPLE), which I kept as an in-distribution reference point rather than replacing it. It is included in every number I report (397 of the 2065 claims); cricket-only the levels read THEME 70.0 / ENTITY 39.3 / DETAIL 34.6 against 69.1 / 43.8 / 36.0 for all six, so the ordering is the same either way

Every position I sampled sits at token index 50 or later — that is the official pipeline's `_MIN_POSITION = 50` (`nla/datagen/stage0_extract.py:35`), a constraint on the position, not on the length of the passage. My lowest sampled index is 79. I had sampled on last 10 contiguous poistions of each passage with K=4. so in total i had 6 x 10 x 4 = 240 explanations.

K is the resampling per activation at T = 1.

I chose cricket because its a familiar topic for me one than i can grade quickly . 

I also took samples from a 2019 biography of great indian freedpom fighter Shri . Veer Savarkar (by Dr. Vikram Sampath) since i believed wikipedia is in almost every pretrained model's knowledge. although this is a dated biography in llm standards, i wondered if it would be FAR LESS represented in the model's training distribution than wikipedia cricket passages are, which would let me see what the AV does when the activation is thinner — and hence i chose it out of instinct and also my lvoe for the book.  

Steps i did 

In order to measure my agreement with the k=haiku judge i was using throughout the above processes, i validated it on one particular task.. i mean , i measured the agreement in labeling between me and haiku4.5 . for that process, i took 150 stratified claims — 50 per level, spread evenly across the 10 position offsets — drawn by a seeded script (`harness/sample_stratified.py`, seed 20260822) rather than chosen by a model, and I prepared an interfact (simple HTML page) that exposed me to the prefix (passage uptil the position) and the claim and i had 3 options in front of me (S/C/N) . I also undertook 30 retests to measure my own agreement rate and consistency . My self consistency rounded at 96.7 % . <he may ask or think why u did not agree with u 100 percent, do u think its better to show what and where and how much i erred so we can show it here>. my agreement with haiku was 88.7 % . here are a few samples were the two of su disagreed <may be do u think we wshud add them here a few may be>

to the best of my knowlwedge , the paper does not report any valiation for its confabulation detector judge models.

I reconstructed the SHAPE of their pipeline (decompose/verify/vibe/match) from the grader outputs shipped inside the paper's own HTML, and wrote my own prompts to match that output format — their prompts are not published. I checked this two ways: the HTML carries `decompose_response`, `verify_response`, `vibe_response` and `match_response` with no corresponding `*_prompt` keys (only four unrelated widgets ship prompts), and the official repo has no confabulation-analysis code in it at all.

The infrastructure that i used for these experiments :

Pod : RunPod A40 48 GB, SECURE, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`

Dependencies : torch 2.9.1+cu128 · transformers 5.3.0 · torchvision 0.24.1 · sglang 0.5.10.post1

Total GPU spend for the project = $3.99 across six pod sessions ($1.23 + $1.08 + $0.40 + $0.34 + $0.49 + $0.45). API spend was measured only once — $4.00 of Haiku on 28 Aug, read off the console; no stage records token usage, so every other API figure I have is an estimate.

**Figure G0.** The measurement pipeline. Colour marks where each stage ran: blue on a rented A40, orange as Haiku 4.5 API calls, gray on the laptop. Steps 3-5 reproduce the paper's confabulation analysis, whose prompts were never published and whose shape was recovered from the grader outputs shipped in the paper's HTML; the ablation is a rewrite, not a deletion. The judge in step 4 is the only stage validated against a human (1070 supported / 995 false overall; 150 claims graded blind, 88.7% agreement). Every count shown is read from the run artifacts, not transcribed.

![G0_pipeline](mats_2027/writeup/figures/G0_pipeline.png)

---

## D1. Specificity replicates

The paper reports that higher-level claims are more often true than specific ones: 64% of
THEME claims, 28% of ENTITY, 24% of DETAIL. I measured the same three rates on a different
NLA (the official Gemma-3-12B one), a different corpus, ten token
positions instead of the final one, and a judge prompt I wrote myself: **THEME 69.1%,
ENTITY 43.8%, DETAIL 36.0%** (n = 887 / 427 / 751). Every rate is higher than theirs, but
the ordering is identical, and the ordering is my claim.

I then checked the instrument. I graded 150 stratified claims blind, with 30 re-presented
under fresh ids, seeing only the claim and the prefix the model had actually read. I agreed
with myself on 29 of 30 and with Haiku on 133 of 150 (88.7%). The paper reports no validation
of its confabulation judge. Although the same paper validates a different grader at 97% on 186 items. The standard exists there but was not applied here.

The disagreements are the interesting part. Of the 17 claims where our
binary verdicts differ, 16 are cases where I said supported and Haiku said the text does not
contain it; exactly one runs the other way.  I observed little more thoroughly, Haiku was right about quoted strings and I was too generous. I was right about vague-but-correct THEME claims and Haiku was too strict.Those two mistakes sit at opposite ends of the specificity axis, so i believe they flatten the gradient.
**The real THEME-to-DETAIL gap is probably wider than either of us measured** (my labels give
46 points, Haiku's 40).

**Figure G1.** Claims about the passage's theme are supported far more often than claims about specific details, on a different NLA, base model, corpus and judge from the paper. AG's blind labels (orange) give a steeper gradient than the Haiku judge (blue): the two graders err at opposite ends, so the true gap is likely larger than either measured.

![G1_specificity](mats_2027/writeup/figures/G1_specificity.png)

---

## D2. What Δ does and does not tell you

"AR is only a weak per claim verifier"

main equation -> delta(mse) = mse(abalated explanation) - mse(original explanation)

if delta(mse) > 0 => removed claim worsened the recosntruction on ablated explanation -> claim is load bearing for reconstruction

if delta(mse) < 0 => removed claim helped in reconstruction -> claim is not load bearing for reconstruction

I set out to verify this "claim" (pun unintended) . in fact , as much as this might be a simple sounding statement, it took me a while to register it.The original question that I set out to finad an answer along with my favourite knowledge partner in crime (claude) was this -> does removing the false claims help in reconstruction better?

(Note this question was directly refered and taken from Neel Nanda's MATS 12.0 admissions doc, "Recommended Research Problems" tab, under *Improved Interpretability Methods* → natural language autoencoders, where he says he is particularly interested in using the activation reconstructor to measure the quality of a description — "which claims can be removed and improve reconstruction accuracy" — to help reduce hallucinations.)

and in the process this is what i found.

|                  | Δ > 0 (load-bearing) | Δ < 0 (removal helped) | total |
|------------------|---------------------:|-----------------------:|------:|
| **TRUE claims**  | 780 (73%)            | 288 (27%)              | 1068  |
| **FALSE claims** | 665 (67%)            | 330 (33%)              | 995   |

n = 2063 valid ablations, cricket, K = 1, verdicts binary (SUPPORTED vs not).
A detector that simply says "Δ > 0 means the claim is true" is right **53.8%** of the time,
against 51.8% for always guessing "true".

about 27% of true claims do contribute to a negative delta but they are not necessariyl false claims .

| Δ | level / subtype | claim |
|---:|---|---|
| −0.00083 | THEME / format | The text is a sports/cricket article format. |
| −0.00056 | THEME / format | The text is structured as a biographical article listing cricket statistics |
| −0.00052 | THEME / format | The text is structured as a sports article format |

All three are true, all three are vague, and all three say something the rest of the same
explanation already says. Removing one costs the reconstruction nothing, and slightly helps.

about 67% of false claims do contribute to a positive delta but they are not necessarily true claims either.

| Δ | level / subtype | claim |
|---:|---|---|
| +0.00927 | DETAIL / quote | The text contains the final token "Garden Gardens" |
| +0.00910 | DETAIL / date  | The text contains the phrase 'By summer 1789' |
| +0.00407 | DETAIL / quote | The text contains the phrase 'During the series against South Africa, the home team, against a touring Indian team, against the series ahead'. |

These are specific, wrong, and load-bearing. Note what they have in common: each one is pointing
at the right place in the passage and getting the content wrong. Across all false claims, the ones
that name the final token of the prefix carry a mean Δ of +0.00141 against +0.00038 for the rest —
nearly 4×, and 25.3% of the load-bearing false claims name it against 11.5% of the ones whose
removal helped.

so thinking about this , as mentioned in the oriignial paper also , the expalantions are made by AV which is a language model. the only ever signal for reconstruction is avaialble only for AR and not AV. AV although is injected with a desired position activation in order for us to get a model read out or explaantion at that position, its alanguage model

and that is the whole problem with reading Δ as a truth signal. Δ measures whether the AR needed
those words to rebuild the activation, not whether they were true. A vague true claim that the
explanation states three times is not needed even once; a confidently wrong claim that pins down
where in the passage the model was reading is needed badly. The two questions come apart, and the
53.8% is what that looks like as a number.

---

## D3. H1 — one population, not two

My hypothesis H1 : Claims that are marked false are efectively a combination of two different categories. 

Category 1 : AV had a faithful readout of the activation but the text judge marked it false because of absence of the words in passage

Category : Genuine confabulations

<examples??>

So based on H1, I predicted two bumps in the histogram of delta of the 975 (false) claims. if the test returned 1 bump then H1 is effectively ruled out.wrote

Dip test found out that there is no valley between two bumps ; it returned p = 0.992 on the 975 claims, ruling out H1.

So a null result means either the test is blind or data genuinely is not representative of the hypothesis. To test the effectiveness of the test, I had planted fake data that genuinely had two groups at H1's predicted mixture size and noise level. 

The dip test caught them 26 % at 20 % mixture, 86% of the time at 26% micture , 100 % at 35 % and above. so below 20 percent the test goes blind.

So the test would have found what H1 described, although a smaller mixture  (less than a fifth size) would have slipped past.

The rule i had pre-registered was not actually dip test. It was delta bic > 10 - "do two bellcurves fit better than one?"

And on my data (claims delta) the two bell curves did fit (+843.4) but my delta distibution was lop sided (skew of 2.63) and a lopsided single hill was fitted better by two bells.

To check that, I generated data that was ONE group by construction, lopsided by exactly the same 2.63, and ran the rule on it 200 times. It reported "two populations" in 200 of 200 runs, with a median score of +846.5 — higher than the +843.4 my real data scored. My evidence for two populations was weaker than what a single population typically produces.

Because the rule had to be revised post me seeing the data, H1's verdict is exploratory and not confirmatory. When i was brainstorming with my thinking and experimenting partner (Claude) we did not account for a lopsy enough distribution.

killing H1 does not mean AR treats category 1(activation-true, text-flase) and category 2(genuine confabulations) claims alike. it merely shows that their measured delta does not split into two groups. This measurement can't tell them apart at this size.

**Figure G2.** If related-false claims were two populations — faithful readouts the text judge mislabels, plus genuine confabulations — their Δ would be bimodal. It is a single right-skewed hump (dip p = 0.992). The ΔBIC rule that was pre-registered DID fire (+843), but a single skewed hump matched to this data fires it in 200/200 draws, so it is disqualified; the dip test, which has the power shown in G3, is the detector that counts. True claims (gray) are shown for scale: the same shape, shifted right.

![G2_h1_null](mats_2027/writeup/figures/G2_h1_null.png)

**Figure G3.** Planted mixtures at the geometry H1 implies (n=975, observed noise, K=1, 400 draws per point, seed 3). Across H1's own predicted 26–42% range the dip test detects them 86–100% of the time. At 20% it drops to 26% — a mixture below that would have been missed. H1 is dead in its stated form, not in every form.

![G3_dip_power](mats_2027/writeup/figures/G3_dip_power.png)

---

## D4. H2 — the noise is real and random

My hypothesis H2 : signal is there but its buried in noise 

to express it in the form of an equation,

observed_delta = underlying_delta + noise

if the noise is random, averaging over multiple resamples should neutralise it, otherwise it shall have no effect (in case of systematic noise)

my kill condition : if the spread doesn't shrink upon averaging the noise is systematic  and averaging can't rescue it.

methodology:

1. firstly i took the claims that appeared in all 4 resamples. 
2. for each claim, average its delta over K = 1, 2, 3 and 4 of its resamples. Then measure the SPREAD: the standard deviation, across the 31 claims, of those per-claim averages. One number per K. It says how far apart the claims sit from each other — not how much any single claim wobbles.
3. the spread shrank : 0.00225 → 0.00181 → 0.00164 → 0.00157 (K = 1, 2, 3, 4). These are overall numbers, one per K, across all 31 claims.
4. fit the model spread(K)² = signal² + noise²/K on only K=1 and K=4, then made it predict K=2 and K=3 — values it had never been shown. Both predictions came within 0.8% of what was observed.
5. the noise estimated from that fit is 0.00185; the noise estimated directly from within-claim variation is 0.00212 — two independent routes agreeing to 13%.

6. I concluded that the noise is random and not systematic.
The spread shrunk toward a floor at 0.00127 — 56% of where it started — and that floor is the genuine between-claim signal.

7. the kill conidition required the spread to fall as 1/√K, which on a log-log plot means a slope of −0.50.

8. I observed the slope to be  −0.262, which on the face of it reads as a partial fail.

9. But the condition itself was mis-specified,  A slope of −0.5 only happens if the spread can fall all the way to zero — i.e. if there were nothing but noise. But because there is a real signal floor, the spread flattens onto it, and the slope is necessarily shallower than −0.5. 

10. so the right test wasn't the slope, but whether the variance model predicts data it wasn't fitted on — which it does, to 0.8%.

Conclusion:

The overall objective or kind of an expected outcome or more desirable outcome if i may was to observe a raise in AUC. The unaveraged , per-claim AUC was 0.535[0.510, 0.559] barely above chance.

 The K-averaged on the matched groups: 0.615 [0.488, 0.736]. Moves the right way, but the interval includes chance, so it isn't established.

  The reason it's underpowered isn't noise — it's recurrence. only  110 claim-groups span more than or equal 3 of the 4 resamples. A decisive test needs roughly ten times more data , which means more resamples per activation - GPU session that i did not run.

  H2 is partially supported. the premise was right (i.e) the noise was real and random. Averaging did shrink it predicatably. the payoff was underpowered . one that i could not establish at the sample size.

**Figure G4.** Spread of per-claim mean Δ as more resamples are averaged (31 claims present in all four). A two-parameter model fitted on K=1 and K=4 alone predicts K=2 and K=3 to within 3% (orange diamonds). The noise averages away as H2 assumes; it converges on a real between-claim floor at 55% of the K=1 spread. noise/signal: 1.51× at K=1, 0.76× at K=4.

![G4_h2_spread](mats_2027/writeup/figures/G4_h2_spread.png)

---

## D5. Removal improves reconstruction 30% of the time

Your suggestion (link to neel's doc the specific part of the doc where he makes the statement we need to put it here in the final draft) was to look for claims that can be removed to *improve* reconstruction. Although the paper says false claims hurt reconstruction *less* than true ones ,it never reports how often removal actually helps. 

On my data it helps often:
**30.0% of 2063 single-claim ablations have Δ < 0** (mean Δ = +0.00088, sd 0.00264). Nearly a
third of the claims in these explanations are worse or **not useful** as per the AR.

One of my earlier pilots had this at 18.8%, but that run used a different ablation method.
In that run I had  the carrier sentence deleted rather than rewriting the claim out and also it was at  different set of token positions, so the two numbers measure different things and I am not treating the gap as a result.

The breakdown by claim level is the part I least expected (Table D5.1). Mean Δ rises from
**+0.00032 for THEME to +0.00105 for ENTITY to +0.00145 for DETAIL** — specific claims carry
4.5× the reconstruction weight of thematic ones, with confidence intervals nowhere near
overlapping. That headline does not survive a control, though. The AR rebuilds the activation at
the FINAL TOKEN of the passage, so a claim that quotes that token is load-bearing for a reason
that has nothing to do with specificity. Flagging every claim containing the passage's last
content word (368 of 2063, 17.8%) and re-running without them, the gradient falls to
**THEME +0.00031, ENTITY +0.00080, DETAIL +0.00084 — 2.7×, and ENTITY and DETAIL become
indistinguishable** (±0.00020 and ±0.00021). So the THEME-to-specific step is real; the
ENTITY-to-DETAIL step is an artefact of claims that name the final token. If truth were what drove Δ this should run the other way, since THEME claims are
supported 69% of the time and DETAIL claims only 36%. My reading is redundancy: a theme is
restated throughout an explanation, so removing one statement of it costs the reconstruction
almost nothing, while a specific detail appears once and its removal is felt.  **That is
consistent with the data, not tested by it** — two rivals survive, that DETAIL claims are simply
longer, and that specific claims genuinely constrain the activation more than vague ones do.

---

## D6. The AV imports names it knows

The AV imports famous names it knows

while reading the explanations for Rahul Dravid's Wikipedia biography, I noticed the AV kept writing about Don Bradman. Bradman is in that passage exactly once, and not as its subject: "In December 2011, he was the first non-Australian cricketer to deliver the Bradman Oration in Canberra." The records in the passage — most balls faced in Tests, longest time spent batting — are Dravid's. The AV attached them to Bradman instead, 21 claims across four token positions, including the flat assertion "The text is about Don Bradman" and, twice, describing him as Indian — "The text discusses Indian batsman Don Bradman" and "The text is about an Indian cricketer named Don Bradman".

But Bradman isn't the interesting part here. Across the claims on that passage the AV names Dravid 27 times, Bradman 21, Tendulkar 20, Lara 6, Richards 3,Pietersen 2, Gavaskar 2, Hobbs 1. Tendulkar does not appear in the passage at all. Neither do Lara, Richards, Pietersen, Gavaskar or Hobbs. Bradman only looked special because he is the one imported name that happened to also be in the text — so he alone fit a story about misreading
what was there. The rest are simply famous batsmen the model knows, written onto a passage that never mentions them.

I then designed an experiment to test whether the confabulation actually follows the name in the text. If the AV was picking up "Bradman" from the passage, then removing that name should remove Bradman from the explanations, and putting a different famous batsman there should pull the confabulation onto him instead. I edited one proper noun in that sentence and left everything else untouched, giving seven conditions: the original; Gavaskar (famous, holds records); Umrigar (real, far less famous); Thangavelu (a name the model cannot know); a version with no proper noun at all; the sentence deleted; and a rewritten-coherent version. Forty explanations per condition, sampled at the same ten token positions.

Both of us(me and opus5) had written predictions down beforehand. I expected Bradman to disappear when the sentence was deleted, and expected the Gavaskar condition to reverse the direction — Gavaskar's records attached to Dravid. Claude predicted the same direction with a weaker effect.

Neither happened. Deleting the sentence entirely left Bradman in 25% of the explanations, against 22.5% in the original. Planting Gavaskar, Umrigar or Thangavelu produced zero uses of each name in forty explanations. The name in the text was neither necessary nor sufficient.

Before reading anything into that, I checked whether the edit had done anything at all to the
thing the AV actually sees. It had not. When comparing the layer-32 activations between conditions, i observed the edited and unedited versions sit at a mean-centred cosine of 0.997 to 0.9995 of each other.

For scale, on the same axis: moving one token position along the same passage gives 0.422 on
average, and a different passage sits at roughly zero. Per condition, aligned by offset from the
end of the passage (the edits change token counts, so absolute positions are not comparable):

| what was changed in the text | mean centred cosine vs the original | worst of the 10 positions |
|---|---:|---:|
| Gavaskar substituted for Bradman (one word) | 0.9995 | 0.9989 |
| Umrigar substituted (one word) | 0.9989 | 0.9965 |
| Thangavelu substituted (one word) | 0.9993 | 0.9986 |
| proper noun removed, length kept | 0.9970 | 0.9796 |
| **whole sentence deleted (104 characters, 26 tokens)** | **0.9968** | 0.9875 |
| sentence rewritten coherently | 0.9979 | 0.9916 |
| *one token step along the same passage* | *0.422* | — |
| *a different passage entirely* | *−0.04* | — |

Deleting the sentence outright moved the activation **0.7% as far as a single token step does**,
and 3.5% even at the worst of the ten positions. (Cosines are mean-centred on the 60 activations
from the main run, deliberately not on the patch data itself; the raw residual stream is so
anisotropic that everything sits above 0.96 against everything else.)

So the honest reading is not "planting a name does nothing". It is that **at these token
positions the intervention never reached the representation**, and the question I set out to ask is untested rather than answered. The design could not have worked: I deliberately placed the edit far enough upstream that it would not change the sampled tokens, which is exactly why it had no effect on them. A real causal test needs activation patching with hooks, or sampling positions adjacent to the edit.

What the control does establish is worth keeping. At layer 32, at these positions, the residual stream barely encodes context from 250 characters back. So the AV cannot be reading "Bradman" out of the activation — i believe the name has to be coming from the model's own knowledge.

If the names are coming from the model's own knowledge rather than from the passage, then the
amount of confabulation i thought, should depend on how well the model knows the material. Cricket Wikipedia is about as battered a text as it gets. So I ran the same pipeline on something what i believed ,it may have almost certainly seen far less of: seven random pages from a 2019 biography of V. D. Savarkar, matched to the cricket passages for length, with the Dravid passage re-run in the same batch as a regression check (it came back at cosine 1.000000).

Both of us(me and opus5) predicted fewer confabulations there — my reasoning being that with less to recall, the AV would have less to invent. We were both wrong, and in the same direction.

| level | Savarkar | cricket | difference |
|---|---:|---:|---:|
| THEME | 49.2% supported (n=1330) | 70.0% (n=707) | −20.8 pp |
| ENTITY | 21.7% (n=695) | 39.3% (n=333) | −17.6 pp |
| DETAIL | 28.1% (n=705) | 34.6% (n=628) | −6.5 pp |
| **ALL** | **36.7%** (n=2730) | **50.5%** (n=1668) | **−13.8 pp** |

63% of the claims on the biography are false, against 50% on cricket, and the overall confidence intervals did not overlap. The unfamiliar text produced *more* confabulation, not less.

The two corpora agree on the mechanism, though. Taking every false claim that names a person, and asking whether that person appears in the passage at all:

| corpus | false person-claims | person is in the passage | person is absent (imported) |
|---|---:|---:|---:|
| Savarkar | 233 | 15 (6.4%) | 218 (93.6%) |
| cricket | 60 | 1 (1.7%) | 59 (98.3%) |

In both domains, **more than nine in ten false person-claims name somebody who is not in the text at all.** On the biography the imported names are Gandhi, Bhagat Singh, Tilak; on cricket they are Tendulkar, Dravid, Bradman. Re-binding a name that is genuinely present — the Bradman story I started from — is the rare case, not the common one.

One more measurement belongs here. The paper says twice, without a number, that false claims tend
to be related to the context rather than fabricated wholesale. I labelled all 995 false claims for
relatedness and **975 of them — 98% — are related to the passage.** Given a cricket activation the
AV confabulates cricket; given a passage about 1900s Maharashtra it confabulates Indian
nationalist history. Only 20 claims were off-topic, and several of those look mislabelled. That
number also costs me the paper's related-versus-unrelated comparison: with an unrelated cell that
small, I cannot reproduce it on this corpus.

Putting the three together — the names are imported, the imports stay in-domain, and there are
more of them on unfamiliar text — the reading I find most plausible is that **the model's own
knowledge is the source of the specifics the AV gets right, not the source of its errors.** The AV
appears to write to a fixed level of specificity whatever it is given. Where the activation is
well-grounded, in text the model has seen many times, the specifics it reaches for are more often
the correct ones. Where the activation is thinner, it fills the same specificity budget with the
nearest famous things it knows — Gandhi, Bhagat Singh, Tilak — and those are wrong.

I want to be clear that this is an interpretation of three results, not a test of anything. It is
consistent with all three and I cannot separate it from alternatives with the data I have. The
experiment that would test it is a third corpus at a third level of familiarity, with the
prediction stated in advance: confabulation should track familiarity monotonically, and the
specificity of the claims should not change across corpora even as their accuracy does.

**Figure G5.** Same judge, same prompt, length-matched passages. Both pre-registered predictions expected fewer confabulations on a 2019 biography than on cricket Wikipedia; the opposite happened (63% vs 50% false, CIs disjoint). In both domains >90% of false person-claims name someone absent from the passage — the dominant failure is importing a plausible entity, not misbinding a present one.

![G5_savarkar](mats_2027/writeup/figures/G5_savarkar.png)

---

## E. What I verified myself

*[E. What I verified myself — not yet written]*

---

## F. Limitations

Limitations

1. NLA as a black box. I used no hooks, no internal probing. Throughout this study, I treated the NLA as a black box. I tried to modulate the input to observe a causal effect on the explanations, but my control showed the edit never reached the representation — deleting a sentence ~250 characters upstream moved the layer-32 activation by less than 1% of what moving a single token position does (centred cosine 0.997 vs 0.42). So the intervention was not causal evidence, and I still don't know why the AV does what it does.

2. Judge valiadtion on cricket data (88.7%). The same was not done for the "Savarkar corpus" partly because I have not read the book completely and also due to time crunch.

3. The semantic matcher was never checked by a human. It is an LLM that decides which claims, across the four resamples of the same activation, are "the same claim". Those groups are the entire basis of the K-averaging result in D4 — 110 groups spanning at least 3 of the 4 resamples — and nobody has ever read a group and confirmed it is one claim rather than two.

4. The two hypothesis results are weaker than they look. H2's payoff could only be measured on the 110 claim-groups that span at least 3 of the 4 resamples, so the bottleneck is recurrence, not noise, and the K-averaged AUC's confidence interval still includes chance. H1's verdict is exploratory rather than confirmatory, because the detector rule I pre-registered turned out to be broken and had to be revised after I had already seen the real distribution.

5. I did not fully finish the Savarkar experiment to report the final delta. I took it as far as explanations, claim decomposition and verdicts — enough to compare confabulation rates across the two domains — but never ran the ablation or the AR scoring on it, so there is no Δ on a second domain.

6. The paper's related-vs-unrelated comparison cannot be reproduced on my data. 975 of the 995 false claims are RELATED to the passage, leaving about 20 unrelated ones, several of which look mislabelled. There is effectively no unrelated cell to compare against.

7. The final-token control is a heuristic, not a labelled category. It flags a claim when the last content word of the prefix appears in it, which cannot separate "names the final token" from "happens to reuse that word", and cannot fire at all on the 22 of 60 prefixes that end in a stopword. It rules that rival down, not out.

---

## F2. Reflections

The practical implication is narrow but usable: **you cannot filter an NLA description
claim-by-claim with the reconstructor as it stands.** An AUC of 0.535 is not a tool. But the
reason it fails is now specific rather than mysterious. The noise is stochastic, so averaging
does help; the limit is that only 110 claim-groups recurred often enough to average over. That is
a sampling problem with a known price — more resamples per activation — not a dead end. Someone
with a GPU budget could settle it in a day.

The confabulation results point somewhere different from where I started. If most false claims
import a name the passage never mentions, then checking descriptions against the *context* is the
wrong shape of defence, because the AV never saw the context. Checking named entities against
what the activation can actually support looks more promising, and cheaper.

With more time, in order: activation patching with hooks, which is the causal test my text-level
intervention could not perform; K=12 resamples to decide H2 properly; and a third corpus at a
third level of familiarity to test whether confabulation really does track how well the model
knows the material.

The honest gap is that none of this touched the model's internals. Everything here treats the NLA
as a black box, and the one intervention I attempted never reached the representation. That is
the first thing I would fix.
