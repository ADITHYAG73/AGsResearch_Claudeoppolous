# Still owed — kept OUT of the paper draft (AG, 2026-10-18: the article carries findings only)

Moved here from the draft's status box, Limitations and Section 7 on 2026-10-18.

## Checks before any number leaves the draft
- [ ] AG hand-checks ~20 random questions per condition against the raw transcripts, with code he writes himself.
      Until then every number is provisional (every number comes from scripts written by the assistant that ran the runs).
- [ ] AG hand-reads ~20 FT-01 "answered without calling the sandbox, wrong" cases (tool_on, s100) — the tool-skipping
      mechanism currently rests on the scorer's own labels.
- [ ] Decide the scoring rule (D3): strict six figures, or relative error with precision reported separately.
- [ ] Re-check every citation against the paper itself; add primary sources where Section 2 has open items.
- [ ] Voice pass: prose is Claude's. Title is a working title.

## Analyses the data already supports (no GPU)
- [ ] Practical stopping rule for thinking runs (stop when the same six-figure value appears twice), vs the oracle ceiling.
- [ ] Angles above 360°: are wrong answers exact values of a different angle, as in Qwen3.8-27B?
- [ ] Read the 61 transcendental runs that exhaust the 9B's budget with the sandbox, and the multiple-of-15° thinking runs that never finish.

## Fine-tuning follow-ups (GPU, AG's call)
- [ ] Untargeted second skill (MMLU-style) on base / s050 / s100 — so "maths changed" can be told apart from "the model changed".
- [ ] s025 eval, to place the onset.
- [ ] A mixed thinking/non-thinking or tool-containing SFT, to test the "thinking-mode-only data" reading.
- [ ] More than one seed / dataset / model.
