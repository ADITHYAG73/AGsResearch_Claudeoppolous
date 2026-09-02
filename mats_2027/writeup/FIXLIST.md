# Compile-pass fix list — Claude does these at assembly, AG does the ones marked [AG]

Rule: Claude fills numbers, corrections, examples, tables, figures. Claude does NOT write
connective prose between AG's paragraphs — anything needing AG's voice is flagged, not written.

## Global
- [x] Scaffold headers stripped (secD5 done 31 Aug; none remain)
- [x] Provenance blocks + author notes now stripped automatically by writeup/compile.py
- [x] `<< >>` / `< >` notes stripped by compile.py (source files keep them; DRAFT.md does not)
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

## A — DRAFTED 1 Sep by Claude, 582 words (limit 600). [AG] must revoice.
- [ ] [AG] read it first — it is the only thing Neel is guaranteed to read
- [ ] [AG] the "What I checked myself" paragraph names three failures; confirm you are happy
      naming them, and that the third (the 99.94) is described accurately

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

## Compile (1 Sep)
- writeup/compile.py assembles ag_drafts/ -> writeup/DRAFT.md, deterministic and re-runnable.
- Current total **6,009 words** against a 2,500-3,000 target. Over by ~2x.
  Worst offenders: D6 1,460 · C 777 · D2 686 · D4 518 · F 432.
- Figures placed: G0 in C, G1 in D1, G2+G3 in D3, G4 in D4, G5 in D6.
- STILL MISSING: section E (AG), and the trim pass.

## 2 Sep — editorial pass + Google Doc published WITH figures
Doc: https://docs.google.com/document/d/1JejK0cEeWmXyvSlCxHypom1rxiluO-ZH1R9wt5qUY-g/edit
- [x] Copy-edit of B, C, D1, D2, D3, D4, D5, E (spelling, capitalisation, chat abbreviations)
- [x] CONTENT CONFLICT resolved: A said activation patching was the next step, F2 said best-of-N.
      A now matches F2 (best-of-N first, patching as the first white-box step).
- [x] B said the judge had "3 choices" and listed two — now names supported / contradicted / not-in-text
- [x] E item 4 said "ending positions of sentences" — the hypothesis was about token positions in a
      PASSAGE, and the result is now stated (+2.4pp, pooled SE 2.9, sign opposite to prediction)
- [x] D2 now states the 2065-vs-2063 gap (two claims could not be rewritten out)
- [x] Equations: every underscore identifier rewritten for the Doc (markdown import escapes them as
      `injection\_scale`). writeup/export_for_docs.py does this and asserts zero underscores.
- [x] Spacing: export unwraps hard-wrapped paragraphs, list items and bold-led paragraphs
- [x] Tables: inline markdown stripped from cells (Docs renders header bold itself)
- [x] All six figures pasted at full PNG quality via the clipboard; 0 markers remain ("0 of 0")
- [x] FIGURE-INTERNAL ERRORS CAUGHT WHILE INSERTING:
      G1's legend said "AG blind labels" (third person in a first-person doc) -> "my blind labels"
      G4's legend had HARDCODED stale numbers (-3.2%, -0.1%, "55%") while its caption said 0.8%/56%.
      Now computed from full-precision noise_fit values: -0.8%, -0.8%, 56%.
- [ ] [AG] final read: typos in YOUR voice were corrected this pass; check none changed your meaning
- [ ] [AG] share settings: anyone-with-link
- [ ] Two earlier Google Docs exist from failed publishes; ask Claude to trash them


## 2 Sep, final pass — all 52 Doc comments addressed
FINAL DOC: https://docs.google.com/document/d/1H8bNinfS9vIK_NFSSVyHKUttA20XIlQoPZatC4rhdhU/edit
- [x] VOICE: A rewritten in AG's register (598->602 words, under the 600 cap). D2, D4, F, F2 audit-speak
      removed ("two independent routes agreeing to 13%", "Note what they have in common",
      "Someone with a GPU budget could settle it in a day").
- [x] PAPER SCRUTINY: all 11 claims about the paper verified in the browser against the live article.
      Two findings: (a) 64/28/24 is real but printed in their FIGURE, not their text — now stated;
      (b) our claim that they never quantify the related/unrelated split was WRONG — their figure
      gives it (83/70/91% of false claims related, ~80% overall). D6 rewritten to compare, not claim a gap.
- [x] Opus 5 named wherever the agent is referred to (A, D2, D3, D4, E).
- [x] EQUATIONS on their own lines (A, B, C, D2, D4); no underscores anywhere (export asserts zero).
- [x] AG's factual questions answered in the text: 53.8% shown as 780+330 over 2063; the −0.50 slope
      explained as the exponent on a log-log plot; "250 characters" corrected to the measured 218–263.
- [x] REPETITION: judge-validation claim cut from C (kept in D1); Δ definition no longer stated 3×.
- [x] APPENDIX added: one activation end to end — prefix, explanation, claims, verdicts, Δ.
      Eden Gardens pos 87: the AV invents 'Elphinstone Gardens' and that false claim is the most
      load-bearing in the explanation (+0.00849). Answers "what was the prefix / explanation / claim".
- [x] Six figures re-inserted at full PNG quality; 0 markers remain.
- [x] Superseded Docs trashed; this is the only one.
- [ ] [AG] final read + set sharing to anyone-with-link.
