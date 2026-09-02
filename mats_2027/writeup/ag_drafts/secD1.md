
## D1. Specificity replicates, and probably understates the effect

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

Two examples, one in each direction. Where Haiku was right: I marked supported the claim *"The text contains the phrase 'considered by many as one of the'"*, where the prefix actually ends `...This series is regarded as one of the` — a near-miss paraphrase, and my single retest flip was a claim of exactly this type, where my second answer agreed with Haiku. Where I was right: *"The text mentions the Border-Gavaskar Trophy"*, on a prefix ending `...home series against Australia in`. Neither word appears, but the 2001 India–Australia home series is the Border–Gavaskar Trophy. A text judge structurally cannot make that call.
