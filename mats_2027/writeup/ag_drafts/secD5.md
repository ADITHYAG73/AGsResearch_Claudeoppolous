## D5. Removing a claim improves reconstruction 30% of the time

Neel Nanda's suggestion in the MATS 12.0 admissions doc, under *Improved Interpretability Methods*, was to look for claims that can be removed to *improve* reconstruction. Although the paper says false claims hurt reconstruction *less* than true ones ,it never reports how often removal actually helps. 

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
almost nothing, while a specific detail appears once and its removal is felt. (shud we show an example here as well??) The paper's own future-work section reports the same thing from the other side: NLA explanations "often repeat the same content on multiple bullet points", and the authors propose reconstructing each bullet independently with a similarity penalty to fix it. So the repetition my reading depends on is something they observed too. **That is
consistent with the data, not tested by it** — two rivals survive, that DETAIL claims are simply
longer, and that specific claims genuinely constrain the activation more than vague ones do.
