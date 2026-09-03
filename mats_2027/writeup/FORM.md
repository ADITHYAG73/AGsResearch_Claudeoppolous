# MATS 12.0 application form — questions, and where the material is
Form: https://airtable.com/appnMboxg76F1QIDc/pagqu7wWWrUCZkNVI/form
Closes Fri Sept 4, 11:59pm PT (extensions to Sept 11).

**RULE FOR THIS FILE: AG writes every answer.** Neel states plainly that answers reading like
LLM output are "a significant negative signal — I see hundreds of them, and they blur together",
and he reads these FIRST as a preliminary filter. Claude supplies numbers and checks accuracy only.

---

## Admin
- Full name · Email · Resume (upload) · LinkedIn
- **Will you definitely join the research phase full-time (Jan 19 – Apr 10, Berkeley)?** Yes / No
- **Google Doc link:** https://docs.google.com/document/d/1H8bNinfS9vIK_NFSSVyHKUttA20XIlQoPZatC4rhdhU/edit
- ☑ First 1–3 pages are an executive summary — TRUE (A is 602 words, page 1)
- ☑ Anyone with the link can view — TRUE (set 2 Sep, Viewer)
- (Optional) other outputs: the repo — github.com/ADITHYAG73/AGsResearch_Claudeoppolous

---

## Q1. What question did you try to answer?
Source: section B. One line: the paper calls the AR "only a weak per-claim verifier" — is the
per-claim signal ABSENT, or present but BURIED IN NOISE? Δ(observed) = Δ(underlying) + ε.

## Q2. Why is this question interesting / why did you choose it?
Source: B + Neel's own doc (Recommended Research Problems → Improved Interpretability Methods).
The two answers have different consequences: buried = a sampling problem with a price on it;
absent = the direction is finished. And a text judge cannot separate "the AV invented it" from
"the model inferred it and the AV read it out" — the AR is the only instrument that sees the
activation.

## Q3. What conclusions have you reached?
Source: A's six bullets. AUC **0.535 [0.510, 0.559]**; noise is stochastic (out-of-sample to
**0.8%**), floor at **56%**, bottleneck is recurrence (**110 groups**); H1 dead at dip
**p=0.992** with **86–100%** power; **30%** of removals improve reconstruction; specificity
replicates **69.1/43.8/36.0** vs their 64/28/24; confabulation is IMPORT (**>90%** of false
person-claims name someone absent, both corpora) and the unfamiliar corpus gave **more**
confabulation (**63% vs 50%**).

## Q4. Technical setup — what you quantify, how you define and measure it, models/data/prompts/metrics
Source: section C + figure G0.
Official `kitft/nla-gemma3-12b-L32-av` / `-ar`, base `google/gemma-3-12b-it`, layer 32 of 48,
d=3840. Δ = mse(claim rewritten out) − mse(intact); MSE is direction-only (both L2-normalised
to √3840 = 61.97, so MSE = 2(1−cos)). 6 passages × last 10 positions × K=4 = 240 explanations,
2,065 claims, 2,063 valid ablations. Judge `claude-haiku-4-5-20251001`, binary supported/not.
Second corpus: 7 pages of a 2019 biography. Ablation is a REWRITE, not a deletion (their method).

## Q5. STRONGEST EVIDENCE **AGAINST** your hypotheses  ← he is asking you to argue against yourself
This is the question the whole project is well-placed for. Raw material:
- **H1 is dead.** Dip p = 0.992, and the test had 86–100% power at the mixture sizes H1 predicted.
- **The rule I pre-registered was broken.** ΔBIC fired (+843.4), but a single skewed hump matched
  to my data fires it in **200 of 200** draws with median **+846.5** — my "evidence" scored BELOW
  the null's median. H1's verdict is therefore exploratory, not confirmatory.
- **H2's payoff is not established.** K-averaged AUC 0.615, interval **[0.488, 0.736]** — contains
  chance. And H2's kill condition was itself mis-specified (no dataset with real signal could pass).
- **My headline gradient shrank under a control.** DETAIL/THEME 4.5× → **2.7×** once claims quoting
  the passage's final token are removed, and ENTITY vs DETAIL becomes indistinguishable.
- **The one causal experiment failed its own control.** The text edit moved the activation **0.7%**
  as far as a single token step — so PATCH-01 is untested, not refuted.
- **Both pre-registered SAVARKAR predictions were wrong**, in the same direction.

## Q6. Biggest limitations, and could you have addressed them?
Source: section F (7 bullets). "Could you have addressed them" — answer honestly per bullet:
matcher spot-check (yes, cheap, I didn't); Savarkar Δ (yes, one pod session); judge on Savarkar
(no — I lack the domain knowledge to grade it); hooks (no, out of scope at 20h).

## Q7. How did you use LLMs? Which ones? How did you ensure it wasn't slop?  ← THE ONE THAT MATTERS
Neel asks for detail: which parts you checked, which you didn't, how you prioritised, and how
surprised you'd be by a major error in each part. Your actual record:
- **Which:** Opus 5 / Fable 5 as the agent (code, pods, analysis); `claude-haiku-4-5-20251001`
  as decomposer / judge / rewriter / matcher inside the pipeline — same model family as the agent,
  which is itself a limitation worth naming.
- **What you checked yourself:** 150 claims blind + 30 retests (96.7% self-consistent, 88.7%
  agreement); read explanations before any numbers existed; killed the position hypothesis by eye.
- **Things that FAILED when checked — this is the answer to "how do you know it isn't slop":**
  the ΔBIC rule that fired on skew alone; a kill condition no dataset with real signal could pass;
  the **99.94** batting average that had propagated into three files and does not exist in the data;
  **115 vs 110** (member claims vs distinct resamples); a quoted passage taken from the wrong token
  position; and, on the final read, a claim about the paper that the paper's own figure contradicts.
- **What you did NOT check:** the matcher's groups; most of the pipeline code line by line.
  Say so, and say how surprised you'd be.

## Q8. 1–3 pieces of evidence you could do good research — ~100 words, **CANNOT cite this project**
The two live blog posts (free norm; flash attention) — written section by section by you after a
tutoring phase, with Claude proof-checking and running the experiments/visualisations.
Kaggle competition record. Karini AI production GenAI work if it fits.

## Q9. Prior experience with mechanistic interpretability
Honest: the NLA fork and Gemma-3 inference work from June (round-trip cos 0.997 verified), the
Gemma per-layer-embedding debugging, self-taught from Neel's own materials. No formal background.

## Q10. Why Neel's stream specifically?
Yours alone.

## Q11. Likelihood you'd join the training phase (Sept 28 – Oct 30) if accepted?

## Q12. (Optional) Anything else about the project not covered above?
Candidate: the time accounting. Be straight — the clock was never formally started; ~20h of
on-task work spread over 13 days; reading the admissions doc, GPU setup and waiting excluded per
his rules. Total GPU spend $3.99; API spend measured once at $4.00.
