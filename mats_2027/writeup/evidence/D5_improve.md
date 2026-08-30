# Evidence for D5 — 30% of removals IMPROVE reconstruction
Source: SCORE-01 (runs/2026-08-23_ar/deltas.parquet), n = 2063 valid ablations

- Δ > 0 (removal hurt): **70.0%**
- Δ < 0 (removal improved): **30.0%**
- mean Δ +0.00088, sd 0.00264, range [-0.0114, +0.0417]
- NOISE-01 saw 18.8% at n=399 — but ablation method (delete→rewrite) AND positions (1→10) both
  changed, so the difference is not attributable. State both numbers, do not compare them.
- Neel's phrasing "claims that can be removed to improve reconstruction" — the paper never reports
  the rate. Their one published example (fennec claim 3, Δmse% = −1.5) contains a case.

## Table D5.1 — Δ by claim level
| level | n | mean Δ | 95% CI half-width | Δ<0 |
|---|---:|---:|---:|---:|
| THEME | 886 | +0.00032 | ±0.00007 | 35.8% |
| ENTITY | 426 | +0.00105 | ±0.00025 | 28.6% |
| DETAIL | 751 | +0.00145 | ±0.00026 | 23.8% |

DETAIL carries **4.5× THEME's** Δ, CIs nowhere near overlapping. This is the OPPOSITE of what the
specificity result predicts if truth drove Δ (THEME is true 69% of the time, DETAIL 36%). Consistent
with H3 (redundancy: THEME content is restated throughout an explanation, so ablating one instance
removes little; a specific claim occurs once) — **consistent with, NOT a test of**. Rivals not
excluded: DETAIL claims carry more tokens; specific claims genuinely constrain the activation more.


## Table D5.2 — CONTROL (new, 30 Aug): does naming the FINAL TOKEN drive the gradient?
Script: `pipeline/final_token_control.py` (reproducible; heuristic stated in the docstring)

The AR reconstructs the activation **at the final token of the prefix**. A claim that names that
token is load-bearing for a reason unrelated to specificity. Flag = the prefix's last content word
(>=3 chars, not a stopword) occurs on a word boundary in the claim. **368 of 2063 claims (17.8%).**

| group | THEME | ENTITY | DETAIL | DETAIL/THEME |
|---|---:|---:|---:|---:|
| all claims | +0.00032 (886) | +0.00105 (426) | +0.00145 (751) | **4.5x** |
| flagged (names final token) | +0.00066 (23) | +0.00311 (45) | +0.00237 (300) | 3.6x |
| **unflagged** | +0.00031 (863) | +0.00080 (381) | +0.00084 (451) | **2.7x** |

**What this does to D5.** The gradient SURVIVES but is much weaker than 4.5x once the control is
applied, and among unflagged claims **ENTITY and DETAIL become indistinguishable**
(+0.00080 +-0.00020 vs +0.00084 +-0.00021 - CIs overlap almost completely). So the defensible
claim is: *thematic claims carry far less reconstruction weight than specific ones (2.7x), and the
apparent ENTITY-to-DETAIL step is an artefact of claims that quote the final token.*

**Limits of the control.** It is a word-match heuristic, not a labelled category: it cannot
separate "names the final token" from "happens to reuse that word", and 22 of the 60 prefixes end
in a stopword or a <3-char token where the flag can never fire. It is a rival ruled *down*, not
out. Quote the 4.5x only alongside the 2.7x.
