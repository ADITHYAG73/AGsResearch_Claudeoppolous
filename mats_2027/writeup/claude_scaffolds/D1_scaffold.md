# D1 SCAFFOLD — Claude's prose, NOT AG's. Rewrite before it goes in the Doc.
# Purpose: show one possible shape and the order the numbers can carry. ~210 words.

## D1. Specificity replicates, and probably understates the effect

The paper reports that higher-level claims are more often true than specific ones: 64% of
THEME claims, 28% of ENTITY, 24% of DETAIL. I measured the same three rates on a different
NLA (the official Gemma-3-12B one, not their Opus NLA), a different corpus, ten token
positions instead of the final one, and a judge prompt I wrote myself: **THEME 69.1%,
ENTITY 43.8%, DETAIL 36.0%** (n = 887 / 427 / 751). Every rate is higher than theirs, but
the ordering is identical, and the ordering is the claim.

I then checked the instrument. I graded 150 stratified claims blind, with 30 re-presented
under fresh ids, seeing only the claim and the prefix the model had actually read. I agreed
with myself on 29 of 30 and with Haiku on 133 of 150 (88.7%). The paper reports no validation
of its confabulation judge, though the same paper validates a different grader at 97% on 186
items — the standard exists there and was not applied here.

The disagreements are the interesting part, and they are lopsided. Of the 17 claims where our
binary verdicts differ, 16 are cases where I said supported and Haiku said the text does not
contain it; exactly one runs the other way. Adjudicating them, Haiku is right about quoted strings and
I am too generous; I am right about vague-but-correct THEME claims and Haiku is too strict.
Those two mistakes sit at opposite ends of the specificity axis, so both flatten the gradient.
**The real THEME-to-DETAIL gap is probably wider than either of us measured** (my labels give
46 points, Haiku's 40).
