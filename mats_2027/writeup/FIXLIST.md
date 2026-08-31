# Compile-pass fix list — Claude does these at assembly, AG does the ones marked [AG]

Rule: Claude fills numbers, corrections, examples, tables, figures. Claude does NOT write
connective prose between AG's paragraphs — anything needing AG's voice is flagged, not written.

## Global
- [x] Scaffold headers stripped (secD5 done 31 Aug; none remain)
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

## D4 — WRITTEN 31 Aug, 521 words. All numbers verified against pipeline/noise_fit.py.
- [ ] Item 9 is still passive — name who mis-specified the kill condition. Honest form:
      the agent drafted it, AG adopted it, neither noticed no dataset with signal could pass it,
      the agent flagged it when the result came in. AG did NOT catch this one.
- [ ] Add the median-vs-RMS correction: 1.41x (NOISE-01) and 0.12x (NOISE-02) used the median
      within-claim sd, the wrong statistic for a variance decomposition. Superseded.
- [ ] "The spread shrunk toward a floor..." is Claude's sentence copied verbatim — revoice.
- [ ] Trim (target ~150-200 for a D subsection)

## D5
- [x] 4.5× no longer stands alone — SCORE-01b control inserted after it (31 Aug, Claude).
      [AG] read it: it is 4 sentences of Claude's prose inside your paragraph, condense to your voice.
- [ ] [AG] decide whether to keep "dead weight or worse" (Claude's phrase, interpretive)
- [ ] [AG] check "the part I least expected" is actually true of you
- [ ] Optional: add the Δ<0 fractions (THEME 35.8% / ENTITY 28.6% / DETAIL 23.8%) — they run the
      opposite way to the means and strengthen the redundancy reading
- [ ] Optional: the "184" example, framed as illustrating the CONTROL, not redundancy

## D6 — IN PROGRESS 31 Aug (beats 1-3 written; beat 4 Savarkar, beat 5 synthesis remain)
- [ ] [AG] confirm beat 2's account of YOUR predictions: DELETE -> Bradman disappears;
      FAMOUS -> reversal, Gavaskar's records onto Dravid; REAL -> some effect; INVENTED -> none
- [ ] [AG] "I designed an experiment" — you proposed the 7-condition structure and asked to read
      the passages first; the conditions were built jointly. Keep the verb or soften it.
- [ ] Beats 1-3 are Claude's prose lightly edited — revoice pass needed
- [ ] The 99.94 correction must NOT reappear anywhere (was in experiments.md + CLAUDE.md)

## E — [AG] ALONE. Claude does not touch this section.
- [ ] [AG] BEFORE writing E: the verification log in experiments.md has exactly ONE entry
      (the sqrt(d_model) check, 19 Aug). Go through what you actually checked and add the real
      entries, then write E from the log rather than from memory.
- [ ] What the record supports as AG's: the 150+30 blind grading; the Bradman observation;
      killing the position hypothesis by eye; finding the relatedness labels did not exist;
      reading the Laxman explanations before any numbers; insisting on advance kill conditions.
- [ ] What the record shows CLAUDE caught (do not claim these): the dBIC skew artefact; the H2
      kill-condition flaw; median-vs-RMS; the 115/110 conflation; the pink-ball quote from the
      wrong position; the final-token confound.

## F — 428 words, long for a limitations list
- [ ] Trim to ~150–200
- [ ] [AG] keep-or-cut decision on bullet 7 (final-token control). Rule agreed: if you cannot
      explain it out loud at compile time, it comes out — Neel disqualifies results the author
      does not understand

## F2 — not started

## A — written last

## Optional deliverable (not submitted)
- [ ] One-page DEFENCE NOTES: variance split, dip test, AUC, power — three lines each: what it
      asks, why it was chosen, what would have falsified it. For interview prep only.

## Figures
- [ ] G0 pipeline, G1 specificity, G2 H1 null, G3 dip power, G4 spread(K), G5 Savarkar
- [ ] T1 (binning) and T2 (two bells) were built for teaching — decide whether either earns a
      place in the write-up or an appendix. T2's left panel is a strong argument for D3.
- [ ] [AG] verify every number on every figure against experiments.md

## Length budget (checked 31 Aug)
Drafts total ~4,100 words against a ~2,500-3,000 target. C (978) and F (432) are the two furthest
over. Trim at compile, not during drafting.
