# Is the NLA activation reconstructor a weak per-claim verifier because the signal is absent, or because it is buried in noise?

*Adithya Giridharan · MATS 12.0 application task · September 2026*

---

## A. Executive summary

The NLA paper says the activation reconstructor is "only a weak per-claim verifier" of the descriptions its verbaliser writes. That sentence is where this project started. I wanted to know which kind of weak: is there no per-claim signal in the reconstruction error, or is there one buried under noise? If buried, it is a sampling problem someone with GPU budget can fix. If absent, the direction is finished.

What I did: the official Gemma-3-12B NLA at layer 32. Six passages, last 10 token positions, four resamples each — **240 explanations, 2,065 claims**. Every claim is then rewritten out of its explanation and scored again against the same activation:

    Δ = mse(claim rewritten out) − mse(intact)

Then a second corpus, seven pages of a 2019 biography, to see whether any of it travels to text the model has seen less of.

**What I found**

- **The signal is there, it is weak, and now it has a number: per-claim AUC 0.535 [0.510, 0.559].** The paper never says how weak. Averaging pushes it to **0.615**, but that interval [0.488, 0.736] contains chance, so I am not claiming it.
- **The noise is random, not systematic**, which is what makes averaging the right lever. I fitted spread(K)² = signal² + noise²/K on K=1 and K=4 only, then asked it to predict K=2 and K=3; it got both **within 0.8%**. There is a floor at **56%** of the K=1 spread, which is the real signal. What stops me is not noise but recurrence: **only 110 claim-groups** recur in 3 or more resamples.
- **My main hypothesis is dead, and I can say how dead.** I predicted the false claims were two populations under one label, so their Δ would show two bumps. It is one hump, **dip test p = 0.992**. I planted fake mixtures at the sizes I predicted and the test caught them **86–100%** of the time, so it was not blind. Below a fifth it would have missed them.
- **Removing a claim improves reconstruction 30% of the time**, and the paper never reports that rate. Thematic claims carry about **2.7×** less weight than specific ones, but only after controlling for claims that quote the passage's final token. Uncontrolled it looks 4.5× and I would have reported the wrong number.
- **Specificity replicates**: **THEME 69.1% / ENTITY 43.8% / DETAIL 36.0%** supported against their 64 / 28 / 24 (their three numbers come from their figure, not their text).
- **Confabulation is import, not misreading.** In both corpora **over 90%** of false claims naming a person name somebody absent from the passage. And the less familiar corpus produced **more** confabulation, not less — **63% false against 50%** — the opposite of what Opus 5 / Fable 5 and I wrote down beforehand.

**What I checked myself.** I hand-graded **150 claims blind plus 30 retests**: **96.7% self-consistent**, **88.7% agreement** with the judge. The paper reports no validation of its own judge. And three things I was about to report did not survive checking: a detector rule I pre-registered fired on skew alone, a kill condition no dataset with real signal could pass, and a number I had repeated for days — a batting average of 99.94 — that does not exist anywhere in my data.

**What this is not.** All of this is black box; I never put a hook into the model. The one intervention I tried never reached the representation — my text edit moved the activation **0.7%** as far as one token position does — so that question is untested, not answered. What I would run next is the paper's own unrun suggestion: best-of-N explanations scored against the reconstructor.

---

## B. The question, and why it matters

The paper says the AR "is only a weak per-claim verifier".

I wanted to investigate whether that is because the signal is absent, or because it is buried in noise.

The signal here is Δ(mse). In particular I hypothesise that the observed Δ of a claim is the sum of an underlying Δ and noise:

Δ(observed) = Δ(underlying) + ε

By underlying Δ I mean the Δ of the claim that we could calculate in an ideal scenario, with 100 percent accuracy. ε is everything else: the run-to-run variation that comes from the prose around the claim changing between resamples.

Now either one of two cases is possible.

1. The distribution of underlying Δ is the same for true and false claims, which means they are indistinguishable and no amount of measurement will separate them.
2. The distributions differ, but the accompanying noise is overwhelming — possibly because of the surrounding prose variation across resamples.

There are also two sources of confabulation, and telling them apart is what makes this hard:

1. The residual activation vector genuinely encodes the thing, and the AV faithfully read it out.
2. The AV, being a language model in its own right, made it up.

In essence, a text judge is blind to both the residual activation and the AV's process. The judge in our case (Haiku 4.5) receives only the passage up to the sampled position — the prefix — and the claim, and returns supported, contradicted, or not in text. It can see whether the words are in the passage. It cannot see what the activation contained.

The residual stream activation that was the basis for the AV's explanation is available, in the whole system, only to the AR. So the AR is in principle the only instrument that could tell case 1 from case 2.

---

## C. Setup

**Model details.** I used the official Anthropic NLA release, `kitft/nla-gemma3-12b-L32-av` (AV) + `kitft/nla-gemma3-12b-L32-ar` (AR), Apache-2.0. The base model I used for this experiment was `google/gemma-3-12b-it`. Activations taken from layer 32 of 48, dimension d of the vector is 3840.

The activation verbaliser model (AV) is of same architecture as the base. The activation is injected as a single token embedding into a fixed prompt with injection_scale 80000 (available from the sidecar), sampled at temperature T = 1. The activation reconstructor (AR) is the base model truncated to 33 blocks plus a learned linear head on the final token.

MSE is **direction-only**: both vectors L2-normalised to mse_scale = √3840 = 61.97 before comparison, so MSE = 2(1 − cos). Magnitude is invisible to the whole pipeline.

**Corpus and sampling.** I chose 6 passages: 5 cricket Wikipedia paragraphs and 1 on the French Revolution — the latter is the NLA maintainers' own example passage, which I kept as an in-distribution reference point. It is included in every number I report (397 of the 2065 claims); cricket-only the levels read THEME 70.0 / ENTITY 39.3 / DETAIL 34.6 against 69.1 / 43.8 / 36.0 for all six, so the ordering is the same either way.

I sampled the last 10 contiguous positions of each passage with K = 4 resamples, so 6 × 10 × 4 = 240 explanations. Every position sits at token index 50 or later — the official pipeline's `_MIN_POSITION = 50` (`nla/datagen/stage0_extract.py:35`), a constraint on the position, not on the length of the passage; my lowest sampled index is 79. Cricket is in distribution for this NLA, measured rather than assumed: the AV→AR round trip returns cosine 0.996 on my passages against 0.997 on the maintainers' own example.

I chose cricket because it is a familiar topic for me, one I can grade quickly.

I also took samples from a 2019 biography of the indian freedom fighter V. D. Savarkar (by Vikram Sampath). Wikipedia is in almost every pretrained model's knowledge; this book, I reasoned, would be far less represented in the training distribution, which would let me see what the AV does when the activation is thinner.

Figure G0 shows the pipeline end to end. In short: extract the residual activation, verbalise it with the AV, decompose the explanation into atomic claims, judge each claim against the exact prefix the model had read, rewrite one claim out at a time, and re-score every variant with the AR. **Δ = mse(claim rewritten out) − mse(intact)**, on the same explanation and the same activation. Δ > 0 means removing the claim hurt reconstruction; Δ < 0 means removal helped. The ablation is a rewrite with the prose reflowed rather than a deletion, which is the paper's own method and is what keeps prose damage out of Δ.

The judge is `claude-haiku-4-5-20251001`. It sees only the prefix and the claim, and returns supported / contradicted / not-in-text; all analysis collapses that to binary supported-or-not, as the paper does. I validated it against my own blind grading; the numbers are in D1.

I reconstructed the SHAPE of their pipeline (decompose / verify / vibe / match) from the grader outputs shipped inside the paper's own HTML, and wrote my own prompts to match that output format — their prompts are not published. I checked this two ways: the HTML carries `decompose_response`, `verify_response`, `vibe_response` and `match_response` with no corresponding `*_prompt` keys, and the official repo has no confabulation-analysis code in it at all.

Infrastructure: a RunPod A40 48 GB, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, torch 2.9.1+cu128 · transformers 5.3.0 · sglang 0.5.10.post1. Total GPU spend for the project was **$3.99** across six pod sessions. API spend was measured only once — $4.00 of Haiku on 28 Aug — because no stage records token usage, so every other API figure I have is an estimate.

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
with myself on 29 of 30 and with Haiku on 133 of 150 (88.7%). The paper reports no validation of its confabulation judge, although the same paper validates a different grader at 97% on 186 items. The standard exists there; it was not applied here.

The disagreements are the interesting part. Of the 17 claims where our
binary verdicts differ, 16 are cases where I said supported and Haiku said the text does not
contain it; exactly one runs the other way.  Looking at them more closely: Haiku was right about quoted strings and I was too generous; I was right about vague-but-correct THEME claims and Haiku was too strict. Those two mistakes sit at opposite ends of the specificity axis, so I believe they flatten the gradient.
**The real THEME-to-DETAIL gap is probably wider than either of us measured** (my labels give
46 points, Haiku's 40).

Two examples, one in each direction. Where Haiku was right: I marked supported the claim *"The text contains the phrase 'considered by many as one of the'"*, where the prefix actually ends `...This series is regarded as one of the` — a near-miss paraphrase, and my single retest flip was a claim of exactly this type, where my second answer agreed with Haiku. Where I was right: *"The text mentions the Border-Gavaskar Trophy"*, on a prefix ending `...home series against Australia in`. Neither word appears, but the 2001 India–Australia home series is the Border–Gavaskar Trophy. A text judge structurally cannot make that call.

**Figure G1.** Claims about the passage's theme are supported far more often than claims about specific details, on a different NLA, base model, corpus and judge from the paper. My blind labels (orange) give a steeper gradient than the Haiku judge (blue): the two graders err at opposite ends, so the true gap is likely larger than either measured.

![G1_specificity](mats_2027/writeup/figures/G1_specificity.png)

---

## D2. What Δ does and does not tell you

The paper calls the AR "only a weak per-claim verifier". The main quantity is

Δ(mse) = mse(ablated explanation) − mse(original explanation)

If Δ is positive, removing the claim worsened the reconstruction, so the claim is load-bearing. If Δ is negative, removing the claim helped, so the claim is not load-bearing.

I set out to verify that "claim" — pun unintended. Simple as the statement sounds, it took me a while to register what it actually means. The original question I set out to answer, along with my favourite knowledge partner in crime (Opus 5 / Fable 5), was this: does removing false claims improve reconstruction?

(The question is taken from Neel Nanda's MATS 12.0 admissions doc — [Recommended Research Problems tab](https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/edit?tab=t.knytn7x826kv), section "Improved Interpretability Methods", the natural language autoencoders bullet — where he says he is "particularly interested in using the activation reconstructor to measure the quality of a description, e.g. figuring out which claims can be removed and improve reconstruction accuracy to help reduce hallucinations".)

And in the process, this is what I found.

| verdict | Δ positive (load-bearing) | Δ negative (removal helped) | total |
| --- | --- | --- | --- |
| TRUE claims | 780 (73%) | 288 (27%) | 1068 |
| FALSE claims | 665 (67%) | 330 (33%) | 995 |

That is 2063 ablations: every one of the 2065 claims except two, which could not be rewritten out of their explanation without changing something else. Verdicts are binary, supported against not supported.

A detector that simply says "positive Δ means the claim is true" is right **53.8%** of the time. That is 780 true claims with a positive Δ plus 330 false claims with a negative Δ, over 2063. Always guessing "true" gets **51.8%**, so the whole of Δ buys me about two points.

About 27% of true claims have a negative Δ, and they are not thereby false claims.

| Δ | level / subtype | claim |
| --- | --- | --- |
| −0.00083 | THEME / format | The text is a sports/cricket article format. |
| −0.00056 | THEME / format | The text is structured as a biographical article listing cricket statistics |

Both are true, both are vague, and both say something the rest of the same explanation already says. Removing one costs the reconstruction nothing, and slightly helps it.

About 67% of false claims have a positive Δ, and they are not thereby true claims either.

| Δ | level / subtype | claim |
| --- | --- | --- |
| +0.00927 | DETAIL / quote | The text contains the final token "Garden Gardens" |
| +0.00910 | DETAIL / date | The text contains the phrase 'By summer 1789' |

These are specific, they are wrong, and the AR still needs them. What I notice about both is that each one is pointing at the right place in the passage and then getting the content wrong. Across all false claims, the ones that name the final token of the prefix carry a mean Δ of +0.00141 against +0.00038 for the rest — nearly four times as much — and 25.3% of the load-bearing false claims name it, against 11.5% of the ones whose removal helped.

The explanation is written by the AV, which is a language model in its own right. The reconstruction signal is available only to the AR, never to the AV. The AV is handed an activation at a chosen position so that we can get a readout at that position, but it is still a language model doing what language models do.

And that is the whole problem with reading Δ as a truth signal. Δ measures whether the AR needed those words to rebuild the activation, not whether they were true. A vague true claim that the explanation already states three times is not needed even once; a confidently wrong claim that pins down where in the passage the model was reading is needed badly. The two questions come apart, and 53.8% is what that looks like as a number.

---

## D3. H1 — one population, not two

My hypothesis H1: claims that are marked false are effectively a combination of two different categories.

**Category 1.** The AV had a faithful readout of the activation, but the text judge marked it false because the words are absent from the passage. The kind of claim I had in mind is the one on the French Revolution passage — "The text contains the phrase 'By summer 1789'" — where the phrase is not in the prefix but the period almost certainly is in the activation.

**Category 2.** Genuine confabulations, such as "The text is about Don Bradman" on a passage about Rahul Dravid.

Based on H1, I predicted two bumps in the histogram of Δ for the 975 related-false claims. If the test returned one bump, H1 is effectively ruled out. That kill condition was written down on 19 August, before any of this data existed.

The dip test found no valley between two bumps: it returned p = 0.992 on the 975 claims, ruling out H1.

A null result, though, means either the test is blind or the data genuinely does not carry the hypothesised structure. To check that the test was not blind, I planted fake data that genuinely had two groups, at H1's own predicted mixture size and at the noise level I had measured.

The dip test caught them 26% of the time at a 20% mixture, 86% at a 26% mixture, and 100% at 35% and above. So at or below a fifth the test goes blind.

The test would therefore have found what H1 described. A smaller mixture, below about a fifth, would have slipped past.

The rule I had pre-registered was not actually the dip test. It was ΔBIC greater than 10 — "do two bell curves fit better than one?"

On my data the two bell curves did fit, at +843.4. But my Δ distribution is lopsided (skew 2.63), and a lopsided single hill is fitted better by two bells than by one, whether or not there are two populations underneath.

To check that, I generated data that was one group by construction, lopsided by exactly the same 2.63, and ran the rule on it 200 times. It reported "two populations" in 200 of 200 runs, with a median score of +846.5 — higher than the +843.4 my real data scored. My evidence for two populations was weaker than what a single population typically produces.

Because the rule had to be revised after I had seen the data, H1's verdict is exploratory and not confirmatory. When I was brainstorming with my thinking and experimenting partner (Opus 5 / Fable 5), we did not account for a distribution this lopsided.

Killing H1 does not mean the AR treats Category 1 (activation-true, text-false) and Category 2 (genuine confabulation) claims alike. It shows only that their measured Δ does not split into two groups. This measurement cannot tell them apart at this sample size.

**Figure G2.** If related-false claims were two populations — faithful readouts the text judge mislabels, plus genuine confabulations — their Δ would be bimodal. It is a single right-skewed hump (dip p = 0.992). The ΔBIC rule that was pre-registered DID fire (+843), but a single skewed hump matched to this data fires it in 200/200 draws, so it is disqualified; the dip test, which has the power shown in G3, is the detector that counts. True claims (gray) are shown for scale: the same shape, shifted right.

![G2_h1_null](mats_2027/writeup/figures/G2_h1_null.png)

**Figure G3.** Planted mixtures at the geometry H1 implies (n=975, observed noise, K=1, 400 draws per point, seed 3). Across H1's own predicted 26–42% range the dip test detects them 86–100% of the time. At 20% it drops to 26% — a mixture below that would have been missed. H1 is dead in its stated form, not in every form.

![G3_dip_power](mats_2027/writeup/figures/G3_dip_power.png)

---

## D4. H2 — the noise is real and random

My hypothesis H2: the signal is there, but it is buried in noise. Expressed as an equation,

Δ(observed) = Δ(underlying) + ε

If ε is random, averaging over multiple resamples should neutralise it. If it is systematic, averaging has no effect at all.

My kill condition, written 19 August before the data: if the spread does not shrink when I average, the noise is systematic and averaging cannot rescue it.

**Method**

1. I took the claims that appeared in all four resamples — 31 of them.
2. For each claim, I averaged its Δ over K = 1, 2, 3 and 4 of its resamples. Then I measured the spread: the standard deviation, across those 31 claims, of the per-claim averages. One number per K. It says how far apart the claims sit from each other, not how much any single claim wobbles.
3. The spread shrank: 0.00225, 0.00181, 0.00164, 0.00157 at K = 1, 2, 3, 4.
4. I fitted the model spread(K)² = signal² + noise²/K on K = 1 and K = 4 only, then made it predict K = 2 and K = 3 — values it had never been shown. Both predictions came within 0.8% of what was observed.
5. The noise estimated from that fit is **0.00185**. I also estimated the noise a completely different way, straight from how much a single claim moves between resamples, and got **0.00212**. Two different routes, 13% apart, which is close enough that I believe the model.
6. I concluded that the noise is random and not systematic. The spread shrank toward a floor at 0.00127 — 56% of where it started — and that floor is the genuine between-claim signal, which is the thing I wanted to exist.

**The kill condition was mis-specified, and I want to be explicit about that.**

7. The condition required the spread to fall as one over the square root of K. On a log-log plot that is a straight line whose slope is the exponent, so 1/√K means a slope of **−0.50**.
8. I observed a slope of −0.262, which on the face of it reads as a partial fail.
9. But a slope of −0.5 only happens if the spread can fall all the way to zero, which is to say if there were nothing but noise. Because there is a real signal floor, the spread flattens onto it and the slope is necessarily shallower than −0.5. The condition was drafted by Opus 5 / Fable 5 and I adopted it; neither of us noticed at the time that no dataset containing real signal could ever pass it.
10. So the right test was not the slope. It is whether the variance model predicts data it was not fitted on, which it does, to 0.8%.

Two earlier noise ratios from my lab notebook, 1.41× and 0.12× (they are not quoted anywhere in this write-up), used the median within-claim spread. That is the wrong statistic for splitting variance, because the distribution of within-claim spreads is badly skewed. They are superseded by the numbers above.

**Conclusion.** The outcome I was hoping for was a rise in AUC. The unaveraged per-claim AUC was 0.535, with a 95% interval of 0.510 to 0.559 — barely above chance. K-averaged over the matched groups it is 0.615, interval 0.488 to 0.736. It moves the right way, but the interval includes chance, so it is not established.

The reason it is underpowered is not noise. It is recurrence: only 110 claim-groups span at least 3 of the 4 resamples, so there are only 110 claims whose Δ can be averaged at all. A decisive test needs roughly ten times more, which means more resamples per activation — a GPU session I did not run.

H2 is therefore partially supported. The premise was right, the noise is real and random, and averaging shrinks it predictably. The payoff is underpowered, and I could not establish it at this sample size.

**Figure G4.** Spread of per-claim mean Δ as more resamples are averaged (31 claims present in all four). A two-parameter model fitted on K=1 and K=4 alone predicts K=2 and K=3 to within 0.8% (orange diamonds). The noise averages away as H2 assumes; it converges on a real between-claim floor at 56% of the K=1 spread. noise/signal: 1.46× at K=1, 0.73× at K=4 (values from pipeline/noise_fit.py).

![G4_h2_spread](mats_2027/writeup/figures/G4_h2_spread.png)

---

## D5. Removal improves reconstruction 30% of the time

Neel Nanda's suggestion in the MATS 12.0 admissions doc — [Recommended Research Problems tab](https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/edit?tab=t.knytn7x826kv), section "Improved Interpretability Methods" — was to look for claims that can be removed to *improve* reconstruction. Although the paper says false claims hurt reconstruction *less* than true ones, it never reports how often removal actually helps. 

On my data it helps often:
**30.0% of 2063 single-claim ablations have Δ < 0** (mean Δ = +0.00088, sd 0.00264). By the AR's own measure, then, nearly a third of the claims in these explanations are carrying no weight — though at a mean Δ this small, some of that will be scatter around zero rather than genuine harm.

One of my earlier pilots had this at 18.8%, but that run used a different ablation method.
In that run I had  the carrier sentence deleted rather than rewriting the claim out and also it was at a different set of token positions, so the two numbers measure different things and I am not treating the gap as a result.

The breakdown by claim level is the part I least expected. Mean Δ rises from
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
almost nothing, while a specific detail appears once and its removal is felt.  The paper's own future-work section reports the same thing from the other side: NLA explanations "often repeat the same content on multiple bullet points", and the authors propose reconstructing each bullet independently with a similarity penalty to fix it. So the repetition my reading depends on is something they observed too.

**That is consistent with the data, not tested by it** — two rivals survive, that DETAIL claims are simply
longer, and that specific claims genuinely constrain the activation more than vague ones do.

---

## D6. The AV imports names it knows

While reading the explanations for Rahul Dravid's Wikipedia biography, I noticed the AV kept
writing about Don Bradman. Bradman is in that passage exactly once, and not as its subject: "In
December 2011, he was the first non-Australian cricketer to deliver the Bradman Oration in
Canberra." The records in the passage — most balls faced in Tests, longest time spent batting —
are Dravid's. The AV attached them to Bradman instead, in 21 claims across four token positions,
including the flat assertion "The text is about Don Bradman" and, twice, describing him as Indian.

But Bradman isn't the interesting part here. Across the claims on that passage the AV names
Dravid 27 times, Bradman 21, **Tendulkar 20**, Lara 6, Richards 3, Pietersen 2, Gavaskar 2,
Hobbs 1 — and only Bradman appears in the passage at all. He looked special because he is the one
imported name that happened also to be in the text, so he alone fit a story about misreading what
was there. The rest are simply famous batsmen the model knows, written onto a passage that never
mentions them.

To test whether the confabulation follows the name, I edited that one proper noun and left
everything else untouched: seven conditions — the original, Gavaskar (famous, holds records),
Umrigar (real, far less famous), Thangavelu (a name the model cannot know), a version with no
proper noun, the sentence deleted, and a coherent rewrite — at forty explanations each. Both of us
had written predictions down first; I expected Bradman to vanish when the sentence was deleted and
the Gavaskar condition to reverse the direction. Neither happened. **Deleting the sentence left
Bradman in 25% of explanations against 22.5% in the original, and planting Gavaskar, Umrigar or
Thangavelu produced zero uses of each.**

Before reading anything into that, I checked whether the edit had changed what the AV actually
sees. It had not.

| what was changed in the text | mean centred cosine vs original | worst of 10 positions |
|---|---:|---:|
| one proper noun substituted (3 conditions) | 0.9989–0.9995 | 0.9965 |
| proper noun removed, length kept | 0.9970 | 0.9796 |
| **whole sentence deleted (104 chars, 26 tokens, 218–263 characters upstream of the sampled positions)** | **0.9968** | 0.9875 |
| *one token step along the same passage* | *0.422* | — |
| *a different passage entirely* | *−0.04* | — |

Deleting the sentence moved the activation **0.7% as far as a single token step does**. So the
honest reading is not "planting a name does nothing" — it is that **the intervention never reached
the representation**, and the question is untested rather than answered. The design could not have
worked: I placed the edit far enough upstream that it would not disturb the sampled tokens, which
is exactly why it did not reach them. What the failed control does establish is that at layer 32,
at these positions, the residual stream barely encodes context from roughly 250 characters back — so the
AV cannot be reading "Bradman" out of the activation at all.

If the names come from the model's own knowledge, the amount of confabulation should depend on how
well it knows the material. So I ran the same pipeline on seven random pages of a 2019 biography
of V. D. Savarkar, length-matched, with the Dravid passage re-run as a regression check (cosine
1.000000). Both of us predicted fewer confabulations. **We were both wrong: 63% of claims on the
biography are false against 50% on cricket** (36.7% supported, n=2730, vs 50.5%, n=1668;
intervals disjoint; the gap holds at every claim level).

The two corpora agree on the mechanism. Of the false claims that name a person, **93.6% on
Savarkar and 98.3% on cricket name someone absent from the passage entirely** — Gandhi, Bhagat
Singh and Tilak on the biography; Tendulkar, Dravid and Bradman on cricket. Re-binding a name that
is genuinely present, the Bradman story I started from, is the rare case.

I also labelled all 995 false claims for relatedness. **975 of them, 98%, are related to the passage.** Given a cricket activation the AV confabulates cricket. Their own figure gives the same split for their corpus and it comes out around **80% related** (83% for theme claims, 70% for entity, 91% for detail), so mine is noticeably higher. I think that is the corpus: theirs is mixed pretraining text and there is somewhere else to wander off to, mine is one narrow domain and there is not. The cost is that with only 20 unrelated claims, several of which look mislabelled to me, I cannot run their related-versus-unrelated Δ contrast at all.

Putting those together — names imported, imports staying in-domain, more of them on unfamiliar
text — the reading I find most plausible is that **the model's own knowledge is the source of the
specifics the AV gets right, not the source of its errors**: it writes to a fixed level of
specificity whatever it is given, and where the activation is thin it fills that budget with the
nearest famous things it knows. This is an interpretation of three results, not a test. The
experiment that would test it is a third corpus at a third level of familiarity, predicting in
advance that confabulation tracks familiarity while the specificity of the claims does not change.

**Figure G5.** Same judge, same prompt, length-matched passages. Both pre-registered predictions expected fewer confabulations on a 2019 biography than on cricket Wikipedia; the opposite happened (63% vs 50% false, CIs disjoint). In both domains >90% of false person-claims name someone absent from the passage — the dominant failure is importing a plausible entity, not misbinding a present one.

![G5_savarkar](mats_2027/writeup/figures/G5_savarkar.png)

---

## E. What I verified myself

The things that were my contributions:

1. I graded 150 claims blind plus 30 retests, 180 items in 33.6 minutes, seeing only the claim and the prefix. 96.7% self-consistent, 88.7% agreement with the judge. That is the human validation for my dataset.
2. I froze the grading conventions the adjudication was then judged against, and the 27 disagreements my blind grading produced turned out to split into two error modes at opposite ends of the specificity scale — I am too generous on quoted strings, the judge is too strict on vague-but-correct descriptions. (The adjudication itself was done by Opus 5 / Fable 5, applying my conventions; it is the same model family as the judge, so it is not a neutral referee.)
3. I spotted the Bradman substitution. It needed cricket knowledge, which is my forte, and it is the reason I chose cricket as the corpus.
4. I had a hypothesis that confabulation would increase towards the later token positions of a passage. I read the explanations by eye before any of the data existed and concluded it was not there; the measurement later agreed, at +2.4 percentage points early versus late against a pooled standard error of 2.9, with the sign opposite to my prediction.
5. I found that the relatedness labels did not exist at all, by checking my own mental model of the pipeline against the data. That blocked H1 until it was fixed.
6. I asked where a number in my own notes came from — the 99.94 batting average Opus 5 / Fable 5 said the model had blended in — and it was nowhere in the data. It had been carried over from a different run and had propagated into three files. The averages the model actually invented were 95.99 and 51.37, attributed to two batsmen who appear nowhere in the passage.

What I did not do, and would like to explore: no hooks, no internal probing. This was a black box throughout.

---

## F. Limitations

1. **NLA as a black box.** No hooks, no internal probing. I tried to modulate the input to observe a causal effect, but my own control showed the edit never reached the representation — deleting a sentence 218 to 263 characters upstream moved the layer-32 activation less than 1% as far as moving a single token position does. That intervention is not causal evidence, and I still don't know why the AV does what it does.

2. **The judge was validated on cricket only** (88.7% agreement with my blind grading). I did not repeat that for the Savarkar corpus, partly because I have not read the book fully and partly for time — so some of the 13.8-point gap between corpora could be the judge being harsher on unfamiliar Indian names rather than the AV confabulating more.

3. **The semantic matcher was never checked by a human.** An LLM decides which claims across the four resamples count as "the same claim", and those 110 groups are the entire basis of the K-averaging result in D4. I never sat down and read a group to confirm it really is one claim and not two.

4. **Both hypothesis results are weaker than they look.** H2's payoff could only be measured on the 110 groups spanning at least 3 of 4 resamples, so the bottleneck is recurrence rather than noise, and the K-averaged AUC's interval still includes chance. H1's verdict is exploratory rather than confirmatory, because the detector rule I pre-registered turned out to be broken and had to be revised after I had seen the real distribution.

5. **Savarkar stopped at verdicts.** I took it as far as explanations, decomposition and judging — enough to compare confabulation rates across domains — but never ran the ablation or the AR scoring, so there is no Δ on a second domain.

6. **No unrelated cell.** 975 of the 995 false claims are related to the passage, leaving about 20 unrelated ones, several of which look mislabelled. The paper's related-versus-unrelated comparison cannot be reproduced here.

7. **The final-token control is a heuristic**, not a labelled category: it flags a claim when the passage's last content word appears in it, so it cannot separate "quotes the final token" from "happens to reuse that word", and it never fires on the 22 of 60 passages ending in a stopword. It rules that rival down, not out.

---

## F2. Reflections

The practical implication is narrow but usable: **you cannot filter an NLA description
claim-by-claim with the reconstructor as it stands.** An AUC of 0.535 is not a tool. But the
reason it fails is now specific rather than mysterious. The noise is stochastic, so averaging
does help; the limit is that only 110 claim-groups recurred often enough to average over. That is a sampling problem with a price on it — more resamples per activation — and not a dead end. With GPU budget I think this gets settled in a day.

The confabulation results point somewhere different from where I started. If most false claims
import a name the passage never mentions, then checking descriptions against the *context* is the
wrong shape of defence, because the AV never saw the context. Checking the named entities against what the activation can actually support looks more promising to me, and cheaper.

With more time, the first thing I would run is one the paper names and did not do. Under
inference-time methods it notes that the pipeline "mostly uses AV outputs and discards the AR",
and that a simple extension is taking a best-of-N explanation scored against AR reconstruction. I
have the resamples and the scoring already. That asks a whole-explanation question rather than a
per-claim one, which sidesteps the recurrence bottleneck entirely — no claim matching is needed to
ask whether the best-reconstructing of four explanations contains fewer false claims than a
randomly chosen one.

After that, in order: whether the AV's confabulated name depends causally on the activation at
that position, which needs patching rather than a text edit; K=12 resamples to decide H2; and a
third corpus at a third level of familiarity to test whether confabulation tracks how well the
model knows the material.

The honest gap is that none of this touched the model's internals. Everything here treats the NLA
as a black box, and the one intervention I attempted never reached the representation. That is
the first thing I would fix.

---

## Appendix. One activation, end to end

Everything in this write-up comes out of the pipeline in Figure G0. Rather than describe it again, here is one activation carried all the way through, so that the tables in D2 and D5 can be read properly.

**The passage.** Eden Gardens, Wikipedia. The activation is taken at token position 87, and the prefix — which is all the judge ever sees — ends mid-quote:

> The stadium was established in 1864. The origins of its name are uncertain. According to some, the stadium is named after the Eden Gardens park where it is located, itself named after the Eden sisters, Emily and Fanny, of Lord Auckland, the Governor-General of India (1836–1842). Initially named the 'Auckland Circus Gardens', the park was renamed to the **'Eden Gardens**

**The explanation.** This is what the AV wrote from that single activation vector. It never sees the passage.

> Structured factual/historical article format, with a descriptive entry about a British colonial residence in Kolkata — factual, informative tone.
>
> The sentence "The garden was named 'Elphinstone Gardens', which was later renamed to 'Elphinstone Gardens" sets up a factual conclusion about the name's origin or etymology.
>
> Final token "Gardens" closes a quoted phrase ("was renamed to the 'Elphinstone Gardens"), requiring immediate completion — likely "in the 1800s" or "name by..." or "after the name of..." or "in the estate in 1885," referencing the original naming source.

Look at what it got right and what it got wrong. It knows it is at the end of a quoted park name that is being renamed. It knows the register, the century, the colonial Kolkata setting. And then it fills the name in as **Elphinstone Gardens**, which is not in the passage at all. Right slot, wrong content. This is the failure mode D6 is about, in one line.

**The claims, the verdicts, and Δ.** Stage 3 breaks the explanation into atomic claims, the judge marks each one against the prefix above, and then each claim is rewritten out and the explanation re-scored.

| Δ | level | verdict | claim |
|---:|---|---|---|
| **+0.00849** | ENTITY | false | The text mentions 'Elphinstone Gardens' as a name. |
| +0.00220 | DETAIL | false | The text contains a quoted phrase "was renamed to the 'Elphinstone Gardens". |
| +0.00108 | DETAIL | false | The text contains the sentence "The garden was named 'Elphinstone Gardens'…" |
| +0.00006 | THEME | **true** | The text has a factual, informative tone. |
| −0.00010 | ENTITY | false | The text describes a British colonial residence in Kolkata. |
| −0.00011 | THEME | **true** | The text is in a structured factual/historical article format. |

This one activation shows the whole problem in miniature. **The most load-bearing claim in the explanation is false** — take "Elphinstone Gardens" out and the reconstruction gets substantially worse, because that invented name is carrying the AR's information about where in the sentence the model was. Meanwhile both true claims sit at effectively zero, and one of them is slightly *better* off removed. If you tried to use the sign of Δ as a truth signal here, you would get four out of six wrong.
