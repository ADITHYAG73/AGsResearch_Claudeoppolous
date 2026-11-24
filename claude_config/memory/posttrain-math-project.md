---
name: posttrain-math-project
description: "AG's paper project — maths expression evaluation alone vs with a Python sandbox, before/after fine-tuning (4 cells). FT-01 done 2026-10-18; AG plans a 'slightly different' follow-up experiment (not yet specified)."
metadata:
  node_type: memory
  type: project
  originSessionId: 5d62c3d5-0568-4062-a81c-214d5eefcc27
  modified: 2026-10-29T17:23:09.782Z
---

AG's approved one-line objective (2026-10-06): measure how well a model evaluates mathematical expressions (trigonometric, transcendental, composite, nested), both on its own and with a general-purpose compute sandbox, then measure how fine-tuning changes each ability on the same questions. Four cells: base vs fine-tuned × no-tool vs sandbox. He intends to publish it as a paper.

**Why:** grew out of [[mathematics-curiosity-thread]] (TRIG-01..04) plus his stated interest: "fine tuning and post training and how it affects the model's capabilities." The sandbox is general-purpose compute, not a trig helper — he corrected Claude on this.

**State as of 2026-10-29 (all committed + pushed, commit f1b87e7):**
- BASE-01 (2026-10-06): Qwen3.5-4B and 9B, 3,362 frozen questions × 4 conditions, A40.
- FT-01 (2026-10-18): Qwen3.5-4B full fine-tune on 5,000 rows of nvidia/Nemotron-SFT-Instruction-Following-Chat-v4 (every trained turn had non-empty thinking → thinking-mode-only training), 313 steps on an H200, checkpoints s025/s050/s100 in S3. Base re-run on A100 to match eval hardware. Headline: sandbox+thinking 98.3% → 81.4% because the fine-tuned model answers WITHOUT calling the tool (0.2% → 38.5%); sandbox+thinking-off rose 86.2 → 93.4 (calls more); alone-off 26.3 → 24.3; alone-on 55.3 → 59.2 (mostly finishing in budget). IFEval did not improve like-for-like. s050 ≈ s100. Cost ≈ $18.
- AG said 2026-10-29 he will come back to do "an experiment on this, but slightly different" — he has NOT said what changes. Ask him; do not guess the variant. Candidate follow-ups already written down (his call): untargeted control skill, s025 eval, mixed thinking/non-thinking or tool-containing SFT.

**Where things live:**
- Journal `posttrain_math/experiments.md` (BASE-00/01, DATA-01, FT-01 design, rule-7 preflight, run log, cost ledger, FT-01 RESULTS); board `posttrain_math/BOARD.md`; open checks `posttrain_math/OWED.md` (kept OUT of the paper draft at AG's request — the draft carries findings only).
- Paper draft (Claude Docs): https://claude.ai/code/artifact/16e2050b-0bbe-4dde-820c-b4bf2ea87606 — Section 5 = fine-tuning, Figures 6–9.
- Reusable pipeline: `posttrain_math/ft/` (prep_data.py, train.py with hub-dtype bit-exact save, train_pipeline.sh, eval_setup.sh, eval_model.sh, ifeval.py, monitor.sh, s3_sync.py); analysis `posttrain_math/analysis/` (ft01_numbers.py, ft01_figures.py, call_rates.py, run_ft01_analysis.sh); results `posttrain_math/runs/2026-10-18_ft01/`.
- S3 (private, ap-south-1): `s3://posttrain-math-459653581741-ap-south-1/posttrain_math/` — ft01/{data,checkpoints/{s025,s050,s100},results,train_logs,analysis}, runs/2026-10-06_base (corrected is_zero-fixed scores uploaded 2026-10-18).

**Gotchas learned in FT-01 (don't relearn):** A100-80GB OOMs for the 4B full FT (moved to H200); on H200, fla 0.5.2 + Triton gives wrong grads in chunk_bwd_dqkwg → install tilelang (verified vs reference); save checkpoints in hub dtypes (48 fp32 tensors) and copy mtp.* back; HF dataset range-reads hit 429 → honour Retry-After; vLLM results depend on GPU type → compare on the same GPU; the pod S3 uploader was a temporary IAM user, deleted after (make a new one per run); `gh` push needs `gh auth switch --user ADITHYAG73` (default account IamAGP 403s), switch back after.

**How to apply:** Work from BOARD.md; decisions are AG's. Prior work in `mathematics/LITERATURE_forgetting.md` — plain "fine-tune and maths drops" is published (TRACE 2310.06762, Dong 2310.05492), so a single before/after score is not the contribution. Follow [[gpu-run-preflight]], [[gpu-idle-watchdog]], [[one-experiment-at-a-time]], [[originator-sources-only]].
