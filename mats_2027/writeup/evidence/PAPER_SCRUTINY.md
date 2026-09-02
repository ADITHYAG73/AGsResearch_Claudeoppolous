# Paper scrutiny — every claim we make ABOUT the paper, checked against the source
Checked 2026-09-02 against https://transformer-circuits.pub/2026/nla/index.html
(prose from the saved full text; the figure opened directly in the browser)

| # | our claim | verdict | evidence |
|---|---|---|---|
| 1 | "only a weak per-claim verifier" is their phrase | ✅ EXACT | "These trends hold in aggregate but are noisy on individual transcripts, so the AR is only a weak per-claim verifier." |
| 2 | they never quantify "weak" | ✅ | no number anywhere near that sentence |
| 3 | THEME 64 / ENTITY 28 / DETAIL 24 | ✅ but **from their FIGURE, not prose** — the numbers are printed inside `png/img_18fcfc16e92031e0.png`; they appear 0 times in the text. Say "from their figure". |
| 4 | the paper asserts false-claims-stay-related **without a number** | ❌ **WRONG — FIX THIS.** The same figure gives the split. Of FALSE claims, the related share is **Theme 29/(29+6)=83%, Entity 50/(50+21)=70%, Detail 67/(67+7)=91%**, ≈80% overall. Our 98% should be compared to ~80%, not offered as filling a gap. |
| 5 | no validation reported for their confabulation judge | ✅ | "The Haiku 4.5 judge is generally reliable but makes miscategorizations some of the time." No agreement number, no n. |
| 6 | but they DID validate a different grader | ✅ | "Two authors hand-graded 186 NLA explanations and found 97% agreement with the grader." |
| 7 | removing true claims hurts more than false | ✅ | "removing true claims from AV explanations hurts reconstruction more than removing false claims" |
| 8 | they never report the RATE at which removal improves | ✅ | no such rate anywhere in the text |
| 9 | best-of-N is their own unrun next step | ✅ | "Our current interpretability pipeline mostly uses AV outputs and discards the AR. A simple extension is taking a best-of-N NLA explanation against AR reconstruction." |
| 10 | they report NLAs repeat content across bullets | ✅ | "they often repeat the same content on multiple bullet points" |
| 11 | their prompts are not published | ✅ | HTML ships `*_response` keys with no `*_prompt`; the official repo has no confabulation-analysis code (0 hits for "decompose"/"confabulat") |

## Their related/unrelated split, read off the figure (new — we can now compare)
| level | true | false, related | false, unrelated | related share of false |
|---|---:|---:|---:|---:|
| Theme | 64% | 29% | 6% | 83% |
| Entity | 28% | 50% | 21% | 70% |
| Detail | 24% | 67% | 7% | 91% |

Ours: **975/995 = 98% related, 2% unrelated.** Theirs is ≈80% related. Plausible cause: their corpus is
pretraining-like text of mixed topics; ours is a single narrow domain, so there is less for the AV to
wander into. That is a comparison, not a gap.
