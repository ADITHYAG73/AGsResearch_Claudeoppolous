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
