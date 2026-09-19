# Blog 01 — AG's dictation (raw, in his words) + corrections from the journal

## Dictation 1 · 2026-09-19 (verbatim, lightly de-duplicated)

In this experiment we took a Qwen 3.5 9B parameter model, found out which problems it is
torn apart on from the tinyMMLU dataset. Among the 100 problems, 17 problems it got wrong.
In those 17, some amount of problems the choice was torn between. We recorded the zeroth
position behaviour — 10 times we sampled — and got the greedy answer from P0. Other than
that we used a stride of 4: every 4th token, if it had more than one candidate clearing the
5% probability, we mark that position as a potential for forking. We took row 41 as an
example because it was torn apart by Qwen 3.5 9B. Using stride 4, every 4th token, we
calculated the 5% threshold clearance and got some number of positions (250 odd? to check).
On those positions: the greedy decode path, and then putting the alternate token and
continuing the generation. Like this we did, and the final answer flipped in X percent of
times. We found supporting evidence in a recent model, as opposed to what was found in a
Llama-grade model three or four years back.

## Corrections (Claude, from `forking/experiments.md`)

| AG said | Journal says | Where |
|---|---|---|
| 17 problems it got wrong | **14 wrong** greedily (86/100 correct). **17 are torn** (top answer ≤80% of 10 samples). 7 are both. Separately, 7 are confidently wrong (≥90% on the wrong letter). | SCREEN-01 |
| 10 samples at P0 to get the greedy answer | 1 greedy answer **plus** 10 free samples at temperature 1, per item. The 10 give the split (o_0); the greedy gives the accuracy. | SCREEN-01 |
| ~250 positions on row 41 | Row 41's answer is 462 tokens → ~116 positions sampled at stride 4 → **55 had ≥2 candidates above 5% → 159 branches**. Row 80 (1,111 tokens): 278 sampled, 81 forking, 206 branches. | QWEN-03, QWEN-02 |
| answer flipped in X% of times | **6 of 104 alternatives** significant at p<0.01 (chance ≈1); **all 6 confirmed at S=200** (p ≤ .001). Modal answer flips at 2 of them at S=200 (t=52, t=76); t=164 goes from A 88% to a 53/47 coin flip. Largest TVD 0.35. | QWEN-03 |
| how many continuations per branch | **S=50** for the full run, then **S=200** at the 5 hit positions (17 branches). | QWEN-03 |
| Llama model "three or four years back" | Llama-3-8B-Instruct, **April 2024** — about 2.5 years old. The Llama result is on the Forking Fast paper's own released data (100 questions), not our run: 7.2% of 13,149 swaps flip the answer beyond chance, 83 strong ones under a flat pooled curve; row 80's ")(" vs ")[" fork, TVD 0.96. | SWAP-01 |
| (not mentioned) | The **null**: Qwen on row 80, 125 alternatives at 81 positions, max TVD 0.040, below the S=50 noise floor. No fork where the model is already sure (o_0 = 10/10 B). | QWEN-02 |
| (not mentioned) | Thinking OFF throughout; budgets 1,500 tokens because Qwen's answers run ~1,100; the paper's `option X` regex was disabled (it read running commentary as verdicts). | QWEN-02 |

Procedure as recalled is correct: greedy base path → every 4th position → keep tokens ≥5%
→ force each → sample continuations → tally final letters → compare tallies (TVD +
permutation test) → rerun the hits at S=200.

## Title (AG, 2026-09-19)
AG's words: "a forking experiment with Qwen 3.5 9B". Agreed form: **A forking-tokens
experiment with Qwen3.5-9B**. Plain by design (LessWrong norms; finding goes in sentence 1).

## Sentence 1 (AG to dictate next)
AG, verbatim: "I replicated a forking experiment on one of the recent open models, Qwen 3.5
9 billion. In the experiment, I found out that in a question that the model is confident
about, no token swap changes the final outcome. But on a question it was torn about, six
alternate tokens did change the final outcome for that particular question. And the 10 free
samples at the start help the experiment."

Claude's note: every clause maps to a journal number (QWEN-02: 0/125; QWEN-03: 6/104
confirmed at S=200; SCREEN-01: o_0 10/10 vs 5/5). Wording is AG's; not edited.
One clause to sharpen when writing: "help the experiment" → what the samples did is
*predict which question would fork*. AG's call.

## Section 1 — What the papers claimed and what was open (AG, verbatim)
"The 2024 forking paper by Bigelow et al. found that single tokens flip the answer on a
GPT-3.5 model. The follow-up paper in 2026, which was released recently, showed that much of
the jaggedness at low sample counts was noise, on a Llama-3-8B parameter model. I was curious
to replicate the study on one of the fairly recent models, and that was my motivation to do
this. Also, neither of the papers asked which questions fork. And I had to ask this because I
was primarily bottlenecked by compute."

Claude's fill-ins / checks for the prose pass:
- 2024: Bigelow, Holtzman, Tanaka, Ullman, arXiv 2412.07961, "Forking Paths in Neural Text
  Generation", GPT-3.5-turbo-instruct, S=30, 7 tasks. Claim verified in paper §4.
- 2026: Bigelow et al. (Goodfire), arXiv 2608.19611, "Forking Fast", Aug 2026, Llama-3-8B-
  Instruct + R1-Distill-Llama-8B, S=200/1000; noise is multinomial, slope −0.49 (§3.3). "Much
  of the jaggedness at low S is noise" — their abstract says "noise is largely an artifact of
  sampling". Verified.
- "neither asked which questions fork" — verified by full read (both run all 100 tinyMMLU /
  all items; item selection never discussed). State as absence, not as their error.
- Consider one clause: the 2026 paper also computed the position-0 split for every item
  (their o_0) — it is the quantity the screen uses; they just never used it to select.

## Section 2 — What I did (AG, verbatim)
"I chose a fairly recent model for the experiment, Qwen3.5-9B. The reason was simple: I was
curious to know if the forking still replicated in newer models. My thought process was that
the newer models, open or closed, should have fairly aced the benchmark datasets.

At first I took a random question from the tinyMMLU dataset. It turned out to be row 80.
When I tested this question with Qwen3.5-9B it gave the right answer, unlike the Llama model
which the 2026 paper reported, and it turned out that the Qwen model did not fork for that
question. Although I had predicted that newer models should not fork and probably answer
correctly, I was still disappointed with the negative result — not particularly
disappointed, but I was interested to see the branching for real, and the negative result
couldn't help me with what I was looking for.

So then I came up with a fairly simple experiment. Since I was bottlenecked by compute, I
had to choose questions where the model was uncertain or wrong. Even if it is wrong, if it
is always wrong it is not going to help; only if it is dicey, split between two or three
options, could it prove fruitful for the forking experiment. It turns out that in tinyMMLU
the model got X questions wrong and Y questions it was torn on — it gave different answers
different times. We sampled A times and in those A it was split between different final
outcomes.

Once I screened potential candidate questions, I narrowed down to one question. I recorded
its greedy answer once and sampled it 10 times at the zeroth position, and it split across
two final verdicts, option A and option C, five times each. With that in mind, I used a
stride length of 4 and swept across the generation: at every fourth token I intervened and
forced alternate tokens at those positions. By alternate tokens I mean tokens with more than
a 5% chance in the final softmax. For that question I found X positions, totalling Y
branches, and I had two sampling settings inspired by the paper, 50 and 200; the results
which persisted at 50 resamples did transfer to 200."

Claude's fill-ins (journal):
- row 80: NOT random — chosen because Llama's shipped data forks there (")(" vs ")[", TVD
  0.96). AG's "random" should become "the question where the 2026 paper's Llama data shows
  its clearest fork". Llama answered C (wrong); correct is B. Qwen: greedy B, o_0 10/10 B.
- SCREEN-01: 100 questions; greedy accuracy **86/100 → 14 wrong**; **17 torn** (top answer
  ≤ 8 of 10 samples); 7 both. A = **10** samples + 1 greedy.
- Row 41 (moral_scenarios): o_0 = 5 A / 5 C at S0=10 (screen); 30 A / 20 C at S0=50 (run).
- Stride 4 over a 462-token answer → **55 positions** with ≥2 candidates → **159 branches**;
  **104 alternatives** (the rest are the written tokens, one per position).
- S=50 full run; S=200 at the 5 hit positions (17 branches).
- "more than 5%": the rule is p ≥ 0.05, inherited from both papers.
- Thinking OFF; budgets 1,500 tokens (Qwen answers ~1,100); paper's regex "option X" pattern
  disabled — worth one sentence in Section 4 or a footnote.
- "should have aced the benchmark" — the screen says 86/100, so state it as the prior that
  turned out partly wrong.

## Section 3 — What I found (AG, verbatim, 2026-09-19)
"What I found in the course of my experiment: initially I experimented on row 80, because
that was the one which forked the most according to the original authors in their Llama
model. I ran it on my Qwen model. Not only did Qwen get the answer right, it never forked.
It turned out to be a negative result. But I wanted to see fork in action. So, as I said,
the screening experiment got me 17 probable questions, and from those I chose row 41. I
could clearly see the forking at S = 50. I scaled it out to S = 200 and it still persisted.
The details of the experiment: for the particular question, I found X positions which
qualified as forking candidates and Y branches in total across all those positions. In
essence, for that row, I found six alternate tokens changed the final outcome, sitting
across five different positions."

Claude's fill-ins (journal):
- X = **55** positions (of ~116 sampled at stride 4 in a 462-token answer); Y = **159**
  branches, of which **104** are alternatives (the other 55 are the written tokens).
- Row 80 on Qwen: **125** alternatives at **81** positions, largest effect TVD **0.040**,
  below the S=50 noise floor (~0.06-0.08). Answers B 10,276 of 10,300.
- Row 41: 6 of 104 significant at p<0.01 (chance ~1). Positions 52, 76, 136, 164 (x2), 184.
  All six hold at S=200 (p <= 0.001). Largest: t=164 "It" -> "However", A 88% -> 53/47,
  TVD 0.35. Verdict flips that hold at S=200: t=52 "Offering" -> "The" (C 72% -> A 55%) and
  t=76 "is" -> "can" (C 55% -> A 62%).

Two wording checks for the prose pass (AG's call on both):
1. "forked the most according to the original authors" - the authors never rank questions
   or name row 80. That ranking is from OUR analysis of their released data (SWAP-01).
   Accurate form: "the question with the clearest fork in the authors' released Llama data,
   by my own analysis of it".
2. "changed the final outcome" - precise meaning is "changed the distribution of final
   answers beyond sampling noise". The majority answer itself flips at 2 of the 5 positions
   at S=200; at t=164 it goes from 88% to a coin flip. Say which you mean, since a reader
   will take "changed the outcome" as "flipped the answer" all six times.

## Section 4 — What this does not show (AG, verbatim, 2026-09-19)
"There are a few caveats in the experiment that I did, a few limitations that I see.
Firstly, I have run only two questions in full. Definitely there isn't much to look at;
the evidence needs more data. It is not that the evidence is not already there — the
paper's original authors did do that — but for the model I chose to replicate on, and
considering I have experimented forking on only one row, it needs more data and more rows.
Moreover, the stride length I adopted was four, so forks that may exist between the sampled
positions — the experiment is blind to those. Also to be noted is the 5% cutoff: tokens
rarer than that threshold never get a chance in the forking experiment. And of course I have
tested only one model, with thinking turned off, so that deserves more models, and thinking
on versus off could be an interesting scenario."

CUT by AG, not content: a sentence that models with thinking on "invariably tend to land on
the correct answer more often than not". (Claude: right to cut — unsupported. Our only
thinking-on data point is the row 80 probe, which ran 3,000 tokens without finishing.)

EXCLUDED by AG, deliberately: the largest-effect point (TVD 0.35 vs the 0.5 bar), because he
does not yet understand it and will not write what he cannot explain. Not content.

Still to dictate: one or two lines on what comes next.

### Section 4, added limitation — effect size (AG, verbatim, after TVD was explained)
"We set out a bar of 50% for TVD, but Qwen's maximum fork ended up at 35%, about one third
of the time, and far below the paper's reported bracket fork on a Llama-3-8B parameter model."

Claude's checks for the prose pass:
- Numbers right: bar 0.5 (set in the QWEN-02 hypothesis sentence before the run); max 0.35
  at t=164 "It" -> "However", S=200.
- "about one third of the time" -> TVD is a share of ANSWERS that moved, not a share of
  time. Accurate: "about a third of the answers moved".
- "the paper's reported bracket fork" -> the paper does NOT report it. The ")(" vs ")["
  fork at row 80 t=228 (TVD 0.96) is from OUR analysis of their released data (SWAP-01).
  Same issue as the Section 3 note. Accurate: "the bracket fork I found in the authors'
  released Llama-3-8B data".

## What comes next (AG, verbatim, 2026-09-19)
"As a future work, I would like to address some of the limitations that I've noticed myself,
such as addressing the stride gap, expanding the question set, and of course the number of
other models that I can test for. In addition to that, I would want to answer one more
question regarding the usability of residual stream activations to predict the fork. So these
are few items that I wish to explore and I'm currently exploring and will subsequently share
in the next versions."

Claude's checks:
- "I'm currently exploring" -> nothing on Q4 has been run; it is parked in COMPASS.md with a
  status block. Draft uses "wish to explore" only. AG's call if any of it has started since.
- Added one sourced sentence: the 2024 paper's Discussion says "it might even be possible to
  avoid sampling altogether if hidden activations can be used to predict forking" (read in
  full 2026-09-07). Credits the idea where it came from. AG to keep or cut.
