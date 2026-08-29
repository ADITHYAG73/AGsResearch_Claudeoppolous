# Evidence for D6 (second half) — SAVARKAR-01, domain transfer
Source: runs/2026-08-28_savarkar/verdicts.parquet vs runs/2026-08-22_pos10/verdicts.parquet, same judge, same prompt

## Table D6.1 — supported rate, biography vs cricket

| level | Savarkar | n | cricket | n | diff |
|---|---:|---:|---:|---:|---:|
| THEME | 49.2% [46,52] | 1330 | 70.0% [67,73] | 707 | -20.8pp |
| ENTITY | 21.7% [19,25] | 695 | 39.3% [34,45] | 333 | -17.6pp |
| DETAIL | 28.1% [25,32] | 705 | 34.6% [31,38] | 628 | -6.5pp |
| ALL | 36.7% [35,39] | 2730 | 50.5% [48,53] | 1668 | -13.8pp |

ALL-level CIs are disjoint. **63% false on the biography vs 50% on cricket.** Both pre-registered
predictions (AG: fewer; Claude: fewer and less specific) were wrong in the same direction.

## Table D6.2 — for FALSE person-claims, is the named person in the passage at all?

| corpus | false person-claims | person IS in passage (wrong predicate) | person NOT in passage (imported) | most-imported names |
|---|---:|---:|---:|---|
| Savarkar | 233 | 15 (6.4%) | 218 (93.6%) | Gandhi, Agarkar, Birsa Munda, Savarkar, Indian |
| cricket | 60 | 1 (1.7%) | 59 (98.3%) | Sachin Tendulkar, Rahul Dravid, Don Bradman, Indian, Brian Lara |

**In both domains >90% of false person-claims name someone absent from the text.** Re-binding a
present name to a wrong predicate — the Bradman story — is the RARE case. The dominant failure is
IMPORT from parametric knowledge.

## The reading that fits PATCH-01 + REL-01 + SAVARKAR-01 together (interpretation, not a test)
Parametric knowledge is the source of the CORRECT specifics, not of the errors. Where the activation
is well-grounded (cricket Wikipedia, seen many times in pretraining) the specifics the AV writes are
more often right. Where it is thinner (a 2019 biography of 1890s provincial Maharashtra) the AV fills
the same specificity budget with the nearest famous entities it knows — Gandhi, Bhagat Singh, Tilak —
which are wrong. The test would be a third corpus at a third level of familiarity.

## AG's pre-registered p449 prediction — right entity, wrong mechanism
Predicted: re-binding of Babarao's/Mohani's imprisonment to Gandhi. Observed: Gandhi in 13 false
claims, all INVENTED facts ("the Gandhi–Nehru pact", "Gandhi called for a revolt"), no re-binding.
Fails by its own stated condition. His Afghanistan-window guess held ("Mohammad Nadir Khan",
"Hindu Taliban conspiracy" — none in the text).

## Position: flat on a second corpus (false rate by offset −9..0: 58.6–66.7%, no trend)

## Caveat
Judge validated on cricket only (88.7%). If it is harsher on unfamiliar Indian names, part of the
−13.8pp is judge, not AV. AG has no domain knowledge here — the point of the design, but it means a
blind sample would be a weaker standard than cricket's.
