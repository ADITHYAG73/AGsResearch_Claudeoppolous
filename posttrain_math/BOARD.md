# Project board — maths ability and agentic ability, before and after fine-tuning

**Objective (AG, approved 2026-10-06):** measure how well a model evaluates mathematical
expressions (trigonometric, transcendental, composite and nested), both on its own and with a
general-purpose compute sandbox, and then measure how fine-tuning changes each of those two
abilities on the same questions.

**The four cells**

|  | no tool (model alone) | with compute sandbox (agentic) |
|---|---|---|
| **base model** | A | B |
| **fine-tuned model** | C | D |

**Done means:** all four cells filled on one model and one fine-tune, same frozen question set,
with confidence intervals and a failure breakdown for the sandbox cells, written up.
Everything beyond that (more models, dose curve, relearning) is a stretch goal, not the finish line.

**Money:** RunPod balance $111.18 on 2026-10-05. Every billed run goes in the cost ledger in
`mathematics/experiments.md` (opening balance, closing balance, estimate vs actual).

---

## Decisions that are AG's (blocking where marked)

- [x] **D1. Which fine-tune?** *(decided 2026-10-18: (b) own fine-tune on `nvidia/Nemotron-SFT-Instruction-Following-Chat-v4` — unrelated domain; Agentic-v4 queued as the second run. Maths datasets rejected: predictable direction, close to Cheng et al., contamination risk.)*
  - (a) Free first look: Hugging Face's public full fine-tunes of `Qwen/Qwen3.5-2B`
    (agentic RL on data tasks; best checkpoint only; maths-adjacent domain).
  - (b) Our own fine-tune on a domain AG names, with checkpoints saved along the way.
  - Both is allowed: (a) first because it costs no training.
- [ ] **D2. Model.** Whiteboard says Qwen3.5 9B / 4B. Stage 3's screen decides which has a
  baseline that is neither near 0% nor near 100%. If (a) is used, 2B is also needed.
- [ ] **D3. Sign off the question set** (stage 1 spec below) before it is frozen.
- [ ] **D4. Sign off the sandbox spec** (stage 2) — one fixed harness, written down exactly.

## Stage 0 — before spending
- [x] One-line objective agreed
- [x] Prior work on forgetting checked → `mathematics/LITERATURE_forgetting.md`
- [x] **Prior work on THIS design** (Codex, 2026-10-06): **the four-cell layout is already published** —
  Cheng et al. 2026, arXiv 2605.06326 (Qwen3-4B, tool-use training, competition maths). Open: a controlled
  function benchmark, a tool-failure breakdown, fine-tuning unrelated to tools. Claude's reading of that
  paper's table numbers DISAGREES with Codex's — read Tables 1, 4–7 properly before quoting either.
- [ ] Read Hugging Face multi-harness RL article in full (source files already located)

## Stage 1 — the question set (no GPU)
Proposed spec v0, for AG to edit:
- [x] **Trig**: sin, cos, tan — integer degrees (the existing 1,080), plus fractional, >360°, negative
- [x] **Transcendental**: e^x, ln x, log10 x, sqrt x, x^y, inverse trig
- [x] **Composite** (two different functions): sin(cos x), e^(sin x), ln(1 + x²) …
- [x] **Nested** (same function repeated): sin(sin x), sin(sin(sin x)), depth 1–3
- [x] **Mixed expressions**: e.g. sin(30°) + e^0.5 × ln 3
- [x] Every question tagged: family, depth, "value likely tabulated" yes/no, needs range reduction yes/no
- [x] Ground truth from a high-precision library, stored to more digits than we score
- [ ] **Scoring rule fixed in advance.** Open point: 5 decimal places penalises large values
  (tan 89° needs 7 significant figures) — decide decimals vs significant figures now, not after.
- [x] Degrees vs radians stated explicitly in every prompt
- [x] Generator is seeded; frozen file gets a hash; dry-run on the laptop with Qwen3.5-0.8B

## Stage 2 — the compute sandbox (no GPU)
- [x] One tool: run Python, return the output. General-purpose, not a trig helper.
- [x] Tool description saved verbatim with every run (`*.meta.json`); system text is Qwen's own template
- [x] Turn limit (6) and token limit (8,192/turn) fixed in advance; hitting a limit is recorded, not scored as wrong maths
- [x] Every run logs: did it call the tool, what code it wrote, what came back, what it answered
- [x] Failure labels: no call · wrong code (e.g. degrees passed as radians) · right result, wrong final answer · ran out of turns
- [x] Dry-run locally end to end

## Stage 3 — screen and baseline (GPU) → cells A and B
- [x] Rule-7 preflight written, opening balance recorded, watchdog running
- [ ] Screen 2B / 4B / 9B, no tool, thinking off, on the frozen set → pick the model (D2) — 4B and 9B done in full; **2B not run** (needed only if D1 = the public Qwen3.5-2B fine-tunes)
- [x] Cell A: full baseline, no tool — 4B and 9B, thinking off AND on, all 3,362 (BASE-01, 2026-10-06)
- [x] Cell B: full baseline, with sandbox — 4B and 9B, thinking off AND on, all 3,362; turn-limited questions re-asked at 12 turns into separate files
- [ ] AG checks the load-bearing numbers himself from the CSV

## Stage 4 — fine-tuned model → cells C and D
- [x] D1 decided (2026-10-18)
- [ ] If (a): download the public fine-tunes, run cells C and D
- [x] If (b): choose data, define "unrelated" precisely, train with checkpoints, run C and D at each — done for s050/s100 (FT-01); s025 saved, not evaluated
- [ ] A second, unrelated skill tracked as a control (so "maths fell" can be told apart from "the model broke") — **still open after FT-01**: IFEval is the manipulation check, not this control; candidate = MMLU-style knowledge set on base/s050/s100

## Stage 5 — results and paper
- [ ] Four-cell table with confidence intervals; breakdown by family, depth, tabulated or not
- [ ] Sandbox failure breakdown, base vs fine-tuned
- [ ] One-sentence claim, in AG's words
- [ ] Draft; every claim about other papers checked against the paper itself

## Stretch (only after "done")
- [ ] Dose curve: maths measured at many checkpoints
- [ ] Relearning test with a never-learned control
- [ ] Same family at 27B

## Log
- 2026-10-06 — Board created. Objective and four cells agreed.
- 2026-10-06 — Stage 1 and 2 built and dry-run. Question set frozen: 3,362 questions, sha256 c01aa943….
  Smoke passed on Qwen3.5-9B (56 q × 4 conditions). Full baseline launched on two A40 pods: 9B and 4B.
  **Open for AG:** (D3) scoring — strict 6 significant figures vs relative error with precision reported
  separately; (D1) which fine-tune. S3 backup: `s3://posttrain-math-459653581741-ap-south-1`.
- 2026-10-06 (22:50 IST) — **Baseline data collection complete: cells A and B, 4B and 9B, thinking off and
  on, 3,362 questions each.** Results and cost ledger in `experiments.md` → BASE-01. Both pods deleted,
  zero pods. Numbers are from Claude's scorer and are unchecked by AG. Still open: D1, D3.
- 2026-10-16 — Paper draft (baseline half) written and re-read three times:
  https://claude.ai/code/artifact/16e2050b-0bbe-4dde-820c-b4bf2ea87606 . Scorer bug fixed (is_zero), one false
  claim withdrawn, new results journaled as PAPER-01. Still open: AG's hand check, D3 scoring, D1 fine-tune.
- 2026-10-18 — D1 decided: own fine-tune on nvidia/Nemotron-SFT-Instruction-Following-Chat-v4 (Claude + Codex independent dataset search; maths sets considered and rejected). Next: licence, maths share in the data, contamination check.
- 2026-10-18 — FT-01 run: fine-tune done (5,000 rows, H200), checkpoints s025/s050/s100 in S3; evals of base(A100)/s050/s100 running. Open: untargeted-skill control.
- 2026-10-18 — FT-01 DONE. Cells C and D measured for s050 and s100 (results in experiments.md → "FT-01 RESULTS").
  Headline: sandbox + thinking on 98.3% → 81.4% (z −22.7) because the fine-tuned model stops calling the tool
  (no-call 0.2% → 38.5%); sandbox + thinking off 86.2 → 93.4; alone off 26.3 → 24.3; alone on 55.3 → 59.2 (termination).
  IFEval: the trained skill did not improve like-for-like. All training rows were thinking-mode. Cost ≈ $18.04; zero pods;
  IAM uploader user deleted. Open (AG's call): untargeted control, s025 eval, mixed-mode SFT, hand-read of no-call answers.
- 2026-10-18 — Paper draft updated with FT-01: new Section 5 (setup, Table 4, Table 5, Figures 6–9), abstract, intro, contributions, Discussion (now 6), Limitations (now 7), Appendix B cost. Per AG, owed items removed from the draft and kept in `OWED.md`.
