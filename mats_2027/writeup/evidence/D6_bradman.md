# Evidence for D6 — the Bradman case (JUDGE-02 R6, PATCH-01)
Source: runs/2026-08-22_pos10/claims.parquet + verdicts.parquet, Rahul Dravid passage.

- "Bradman" occurs in the prefix ONCE, as: `irst non-Australian cricketer to deliver the Bradman Oration in Canber`
- The passage then says of **Dravid**: `He holds the records for the most balls faced in Test cricket and the longest time spent battin`
- Claims mentioning Bradman: **21**, across positions [142, 143, 144, 146], verdicts {'NOT_IN_TEXT': 17, 'SUPPORTED': 3, 'CONTRADICTED': 1}

## The claims
- pos 142 k=1 [NOT_] The text discusses Indian batsman Don Bradman
- pos 143 k=0 [SUPP] The text mentions Test batsman Don Bradman
- pos 143 k=1 [NOT_] The text mentions Sir Donald Bradman.
- pos 143 k=1 [NOT_] The text describes Bradman as the highest scoring batsman in Test cricket.
- pos 143 k=1 [NOT_] The text contains the phrase 'Sir Donald Bradman, the highest scoring batsman in Test cricket, the longest time spent'.
- pos 143 k=3 [NOT_] The text includes a biographical snippet about Don Bradman
- pos 143 k=3 [NOT_] The text lists cricket statistics for Don Bradman
- pos 144 k=0 [NOT_] The text contains the sentence "Don Bradman is the highest Test batting average...longest time spent batting".
- pos 144 k=0 [SUPP] The text mentions Don Bradman.
- pos 144 k=1 [NOT_] The text contains the sentence 'Don Bradman was the longest time batting'
- pos 144 k=2 [NOT_] The text is about Don Bradman
- pos 144 k=3 [SUPP] The text mentions Don Bradman
- pos 144 k=3 [NOT_] The text contains the sentence "Don Bradman...holds the record for the longest time spent batting"
- pos 146 k=0 [NOT_] The text is about Don Bradman
- pos 146 k=1 [NOT_] The text discusses Don Bradman.
- pos 146 k=1 [NOT_] The text contains factual descriptions of Bradman's record.
- pos 146 k=1 [NOT_] The text establishes a list of statistical superlatives for Bradman.
- pos 146 k=1 [NOT_] The text mentions most runs scored in Tests as one of Bradman's records.
- pos 146 k=1 [NOT_] The text mentions highest strike-rate in Tests as one of Bradman's records.
- pos 146 k=2 [CONT] The text is about an Indian cricketer named Don Bradman.
- pos 146 k=2 [NOT_] The text mentions Bradman's batting statistics in a footnote.

## What PATCH-01 then showed (causal test, 7 conditions, 40 explanations each)
- delete the Bradman sentence entirely → Bradman still in 25% of explanations (vs 22.5% original)
- plant Gavaskar / Umrigar / Thangavelu instead → 0/40 uses of each
- so at these positions the prefix token is NOT necessary and a planted name is NOT sufficient
- the activation 250 chars downstream barely encodes the name: centred cosine 0.997 vs 0.42 for one token step


## CORRECTION 2026-08-31 (AG asked where the 99.94 was attached — it wasn't)
- **99.94 does not occur anywhere in the Dravid data.** All 343 claims searched. The invented
  averages are **95.99 attributed to Brian Lara** (pos140 k2) and **51.37 attributed to Kevin
  Pietersen** (pos147 k0). 99.94 occurs twice in the whole project, both in the PATCH-01 run.
  The earlier note claiming the AV blended in Bradman's real average was a cross-run conflation.
- **Bradman is not the special case.** Name mentions in claims on the Dravid passage:

  | Dravid | Bradman | Tendulkar | Lara | Richards | Pietersen | Gavaskar | Hobbs |
  |---:|---:|---:|---:|---:|---:|---:|---:|
  | 27 | 21 | 20 | 6 | 3 | 2 | 2 | 1 |

  **Tendulkar is absent from the passage and imported 20 times.** Bradman only looked special
  because he is the one imported name that IS in the prefix.
- **Better framing for D6:** the Dravid case and SAVARKAR-01 are the SAME finding — import from
  parametric knowledge (>90% of false person-claims name someone absent, in both domains).
  Re-binding a present name is the rare variant. This also explains PATCH-01: deleting "Bradman"
  changed nothing because the prefix name was never the mechanism.
- Other names the AV asserts on this passage, all absent: "W.J. Hobbs", "Vivian Richards",
  "Kevin Pietersen", and it twice describes Bradman as Indian ("Indian batsman Don Bradman",
  "an Indian cricketer named Don Bradman").
