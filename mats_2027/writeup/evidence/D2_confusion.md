# Evidence for D2 — confusion matrix and samples
Source: runs/2026-08-23_ar/deltas.parquet (SCORE-01), 2063 valid ablations, cricket, K=1.
Verdicts binary (SUPPORTED vs not), per JUDGE-02.

## Table D2.1 — sign of Δ vs judge verdict

|              | Δ > 0 (load-bearing) | Δ < 0 (removal helped) | total |
|--------------|---------------------:|-----------------------:|------:|
| TRUE claims  | 780 (73%) | 288 (27%) | 1068 |
| FALSE claims | 665 (67%) | 330 (33%) | 995 |

- x = **27%** of TRUE claims have Δ < 0
- y = **67%** of FALSE claims have Δ > 0
- accuracy of "Δ>0 ⇒ true": **53.8%**; of "always true": 51.8%
- mean Δ: true +0.00115, false +0.00059

## Samples — TRUE claims where removal HELPED (Δ<0)
- Δ=-0.00056  `THEME/format`  The text is structured as a biographical article listing cricket statistics
- Δ=-0.00083  `THEME/format`  The text is a sports/cricket article format.
- Δ=-0.00052  `THEME/format`  The text is structured as a sports article format

(all format claims — true, vague, and restated elsewhere in the same explanation)

## Samples — FALSE claims that are LOAD-BEARING (Δ>0)
- Δ=+0.00927  `DETAIL/quote`  The text contains the final token "Garden Gardens"
- Δ=+0.00910  `DETAIL/date`  The text contains the phrase 'By summer 1789'
- Δ=+0.00407  `DETAIL/quote`  The text contains the phrase 'During the series against South Africa, the home team, against a touring Indian team, against the series ahead'.

(all specific, wrong, and the only claim in their explanation carrying those words)
