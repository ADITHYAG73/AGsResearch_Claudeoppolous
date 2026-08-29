# Evidence for D1 — specificity replicates
Source: JUDGE-01 (verdicts.parquet, all 6 passages incl. reference) + JUDGE-02 (harness/verdicts_AG_MAIN.jsonl)

## Table D1.1 — supported rate by level

| level | ours (Gemma-3-12B NLA, cricket Wiki, 10 positions) | n | paper (Opus 4.6 NLA, Common Pile, final token) |
|---|---:|---:|---:|
| THEME | **69.1%** [66, 72] | 887 | 64% |
| ENTITY | **43.8%** [39, 49] | 427 | 28% |
| DETAIL | **36.0%** [33, 39] | 751 | 24% |

Ordering THEME > ENTITY > DETAIL holds on a different NLA, base model, corpus, token positions,
and an independently written judge prompt. Our rates are higher at every level (candidates, not
separated: different NLA; cricket Wikipedia is far more formulaic than Common Pile; our conventions
keep vague THEME claims). CONTRADICTED: 39/2065 = 1.9%.

## Table D1.2 — the same 150 claims, AG's labels vs Haiku's (JUDGE-02)

| level | agreement (binary) | AG says supported | Haiku says supported |
|---|---:|---:|---:|
| THEME  | 82.0% [69, 90] | 82.0% | 64.0% |
| ENTITY | 96.0% [87, 99] | 36.0% | 36.0% |
| DETAIL | 88.0% [76, 94] | 36.0% | 24.0% |
| ALL    | 88.7% [83, 93] | — | — |

- AG self-consistency on 30 retests: 29/30 = **96.7%** (the paper's own hand-validation standard
  for a DIFFERENT grader was 97% by two authors on 186 items; its confabulation judge was never validated)
- 10 of 27 disagreements were pure CONTRADICTED-vs-NOT_IN_TEXT — the three-way scale was a design
  error; binary is used everywhere
- Two error modes at OPPOSITE ends: Haiku under-credits THEME/format claims ("vague is not false"
  ignored); AG over-credits DETAIL/quote claims (the quote rule applied inconsistently — his one
  retest flip was a quote). **Both compress the gradient → the true THEME-to-DETAIL gap is larger
  than either measured** (Haiku 40pp, AG 46pp).
- Haiku's 64/36/24 vs the paper's 64/28/24: our judge closely reproduces theirs — same model, same bias
