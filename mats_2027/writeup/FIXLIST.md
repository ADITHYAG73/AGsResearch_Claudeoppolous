# Compile-pass fix list — Claude does these at assembly, AG does the ones marked [AG]

Rule: Claude fills numbers, corrections, examples, tables, figures. Claude does NOT write
connective prose between AG's paragraphs — anything needing AG's voice is flagged, not written.

## Global
- [ ] Strip every scaffold header (`# D_ SCAFFOLD — Claude's prose...`) — currently in secD5
- [ ] Strip Claude provenance NOTE blocks (bottom of secD2, bottom of secC's FACTS list)
- [ ] Strip all remaining `<< >>` and `< >` author notes
- [ ] [AG] typo pass — Claude has deliberately NOT corrected AG's typos anywhere
- [ ] Renumber figures once the final set is fixed; check every in-text figure reference resolves

## B
- [ ] 3 leftover placeholders from 29 Aug (identified in the 29 Aug accuracy pass)

## C — 978 words, target ~300
- [ ] Trim: steps 1–7 now live in figure G0, so the prose should carry only what a diagram can't
      (which NLA and why it is the official one, why cricket, why Savarkar, judge validation)
- [ ] [AG] re-voice the Savarkar rationale sentence — Claude only flipped its polarity, marked <<AG>>

## D1
- [ ] Insert Table D1.3 (the two verified disagreement examples)
- [ ] [AG] confirm the cricket fact: 2001 India–Australia home series = Border–Gavaskar Trophy
- [ ] Decide: add per-level agreement (THEME 82% / ENTITY 96% / DETAIL 88%)? It is the direct
      answer to "does judge error vary by level", which was the reason for the check

## D2 — done, needs only the global strips

## D3 — WRITTEN 31 Aug, 409 words (target ~150-200, trim at compile). All numbers verified.
- [ ] 975 = RELATED-false, not all false (995). Draft still says "975 (false) claims" — add the
      98% figure from REL-01. STILL OUTSTANDING.
- [ ] Say the kill condition was written in advance (19 Aug) — pre-registration, not hindsight.
      STILL OUTSTANDING.
- [ ] Precision: at a 20% mixture power is 26%, so "at or below a fifth would likely be missed"
      is the honest phrasing; the draft says "below 20 percent the test goes blind"
- [ ] Stray word "wrote" pasted at the end of the beat-2 sentence
- [ ] Typo "micture"; "Category" missing its 2
- [ ] Category 1 example: 'By summer 1789' (Δ=+0.00910) — label as MOTIVATION, not evidence,
      because H1 died and no verified Category 1 instance exists
- [ ] Category 2 example: "The text is about Don Bradman" on the Dravid passage
- [ ] Figures G2 (histogram) and G3 (power curve) referenced from the prose

## D4 — not started

## D5
- [ ] **4.5× must not stand alone** — SCORE-01b: falls to 2.7× with the final-token control, and
      ENTITY vs DETAIL becomes indistinguishable. Rewrite that clause.
- [ ] [AG] decide whether to keep "dead weight or worse" (Claude's phrase, interpretive)
- [ ] [AG] check "the part I least expected" is actually true of you
- [ ] Optional: add the Δ<0 fractions (THEME 35.8% / ENTITY 28.6% / DETAIL 23.8%) — they run the
      opposite way to the means and strengthen the redundancy reading
- [ ] Optional: the "184" example, framed as illustrating the CONTROL, not redundancy

## D6 — not started

## E — [AG] ALONE. Claude does not touch this section.

## F — 428 words, long for a limitations list
- [ ] Trim to ~150–200
- [ ] [AG] keep-or-cut decision on bullet 7 (final-token control). Rule agreed: if you cannot
      explain it out loud at compile time, it comes out — Neel disqualifies results the author
      does not understand

## F2 — not started

## A — written last

## Figures
- [ ] G0 pipeline, G1 specificity, G2 H1 null, G3 dip power, G4 spread(K), G5 Savarkar
- [ ] T1 (binning) and T2 (two bells) were built for teaching — decide whether either earns a
      place in the write-up or an appendix. T2's left panel is a strong argument for D3.
- [ ] [AG] verify every number on every figure against experiments.md
