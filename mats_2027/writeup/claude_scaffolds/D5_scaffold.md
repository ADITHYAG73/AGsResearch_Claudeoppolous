# D5 SCAFFOLD — Claude's prose, NOT AG's. Rewrite in your voice before it goes in the Doc.
# Every number recomputed from runs/2026-08-23_ar/deltas.parquet on 30 Aug.

## D5. Removing a claim improves reconstruction 30% of the time

Neel's suggestion was to look for claims that can be removed to *improve* reconstruction. The
paper's finding is weaker than that — it says false claims hurt reconstruction *less* than true
ones — and it never reports how often removal actually helps. On my data it helps often:
**30.0% of 2063 single-claim ablations have Δ < 0** (mean Δ = +0.00088, sd 0.00264). Nearly a
third of the claims in these explanations are, by the AR's own measure, dead weight or worse.

One earlier pilot of mine put this at 18.8%, but that run used a different ablation method
(deleting the carrier sentence rather than rewriting the claim out) and a different set of token
positions, so the two numbers measure different things and I am not treating the gap as a result.

The breakdown by claim level is the part I did not expect (Table D5.1). Mean Δ rises from
**+0.00032 for THEME to +0.00105 for ENTITY to +0.00145 for DETAIL** — specific claims carry
4.5× the reconstruction weight of thematic ones, with confidence intervals nowhere near
overlapping. If truth were what drove Δ this should run the other way, since THEME claims are
supported 69% of the time and DETAIL claims only 36%. My reading is redundancy: a theme is
restated throughout an explanation, so removing one statement of it costs the reconstruction
almost nothing, while a specific detail appears once and its removal is felt. **That is
consistent with the data, not tested by it** — two rivals survive, that DETAIL claims are simply
longer, and that specific claims genuinely constrain the activation more than vague ones do.
