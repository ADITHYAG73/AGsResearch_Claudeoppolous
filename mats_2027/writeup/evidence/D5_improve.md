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
