# Post-training maths project — experiment journal

## COST LEDGER
| run | date | GPU | opened | closed | actual | estimated |
|---|---|---|---|---|---|---|
| BASE-00 smoke | 2026-10-06 | A40 | $104.04 | | | ~$0.25 |

## BASE-00 — smoke test of the baseline pipeline (2026-10-06)
**What:** Qwen3.5-9B, 56 questions spread across all seven families, all four conditions
(model alone and with the compute sandbox, thinking off and on). Purpose: prove the pipeline on
the real engine and MEASURE tokens per question and tokens per second, so the full-run estimate
is anchored to this model on these questions and not to an earlier project.

**Built and tested before the pod existed (laptop, Qwen3.5-0.8B):**
- `gen_questions.py` → `data/questions_v0.jsonl`, 3,362 questions, sha256 c01aa943…; every truth
  computed twice by code sharing no arithmetic (mpmath at 40 digits; Python `math` doubles), 0
  disagreements; 10 hand-known values spot-checked.
- `run_eval.py` — prompts rendered by the model's own chat template; loop runs model-written code.
- `score.py` — 6 significant figures; truncation kept apart from wrong answers.
- Dry run: all four conditions ran end to end on 5 questions.
- **Found by rendering the template rather than assuming:** Qwen3.5 emits tool calls as
  `<tool_call><function=run_python><parameter=code>…`, not JSON. A JSON parser would have recorded
  "no tool call" on every question. Also: Qwen3.5's thinking mode adds NO extra system message
  (Qwen3.8's did), so that confound from TRIG is absent here.

**Decisions taken as defaults so the run could start — AG to overrule:**
- Scoring: **6 significant figures**, not 5 decimal places (fair across magnitudes; e^10 ≈ 22026).
- Token budget: **8,192 per model turn**, up to 6 turns (AG: "generous, max the tokens out").
- Thinking: **both**, as separate conditions, with and without the sandbox.
- Greedy decoding. The user prompt is identical in all four conditions; only tool availability
  and thinking differ.

### Rule-7 preflight (from reading `run_eval.py`)
1. **Progress line per unit of work?** Yes: one per chunk, with count, questions/s, tokens/s, ETA
   and how the chunk's questions ended.
2. **Written incrementally?** Yes: one JSON line per question, flushed per chunk; a restart skips
   ids already on disk, per condition.
3. **Memory growth?** vLLM reserves a fixed 90% of the card at start; Qwen3.5-9B in bf16 is
   ~19 GB of a 48 GB A40, the rest is KV cache. Host side holds one chunk of transcripts at a time.
4. **Killed at 80%?** Every finished chunk survives; rerun the same command to continue.
**Sandbox safety:** model-written code runs as `nobody` with a 10 s timeout and cannot write to
`/root/out`; setup asserts both before any model is loaded.
**Cost line:** setup ~12 min + 9B download and load ~5 min + smoke ~10 min ≈ 30 min on an A40 at
$0.35/hr plus disk ≈ **$0.25**. Opening balance **$104.04**.

### BASE-00 result — smoke PASSED (Qwen3.5-9B, 56 questions, A40, vLLM 0.29.0)
Setup 7 min incl. both sandbox assertions (runs as uid 65534; cannot write to `/root/out`).
Engine init 215 s. Artifacts: `runs/2026-10-06_smoke/`.

| condition | correct (6 s.f.) | within 1e-3 | no answer | tokens/question | wall |
|---|---|---|---|---|---|
| notool_off | 33.9% | 41.1% | 0 | 8.6 (median 9) | 1 s |
| tool_off | 91.1% | 98.2% | 0 | 76 | 17 s |
| notool_on | 39.3% | 51.8% | **27/56 truncated** | **5,928** (median 7,639) | 6.8 min, 817 tok/s |
| tool_on | 94.6% | 96.4% | 2 truncated | 793 (median 329) | 6.1 min |

n=56: directional only, not a result.

**Read from the raw answers, not the table:**
- **Scoring rule conflates two failures.** In `tool_off`, 4 of 5 non-correct are
  `right_value_wrong_answer`: the sandbox returned e.g. `0.017455064928217585` and the model wrote
  `0.017455` — six DECIMAL PLACES where six SIGNIFICANT FIGURES were asked, one digit short. All
  four have |value| < 0.1. That is a precision-instruction error, not a maths error. Proposed:
  score maths by relative error and report "gave the requested precision" separately (the TRIG
  study separated value from format for the same reason). AG's call; needs no re-run.
- `sqrt(-2)` with the tool: the model used `cmath` and answered `1.41421j` instead of "undefined".
  Scored wrong, correctly: the prompt says undefined over the reals.
- **Thinking alone often does not terminate.** 27/56 hit 8,192 tokens, mostly composite (10/16)
  and nested (8/12). Read one (`tan(1°)`): it reaches 0.0174551 early, then re-derives by Taylor
  series and loops on a 1.1e-10 discrepancy against a recalled value. Over-verification, not
  inability — same family of behaviour as TRIG-04's `dec2`. Budget kept at 8,192.
- Model alone, thinking off, shows the TRIG pattern on a different model: near-misses on
  fractional angles, blunders after reduction (`sin(424°)` → −0.743, truth +0.899).

## BASE-01 — full baseline, Qwen3.5-9B and Qwen3.5-4B (2026-10-06)
All 3,362 questions × 4 conditions per model. Order: notool_off, tool_off, tool_on, then
notool_on last with the 668-question core subset first, so the expensive condition has complete
coverage of every family early.

### Rule-7 preflight
1. Progress: one line per 256-question chunk (count, q/s, tok/s, ETA, how the chunk ended).
2. Incremental: JSONL flushed per chunk, per condition; rerunning the same command resumes.
3. Memory: fixed at start by vLLM (90% of the card; measured 18.1 GB weights + 19.6 GB KV cache on
   the A40). Host RAM holds one chunk of transcripts.
4. Kill at 80%: all finished chunks survive; at most one chunk (≈10 min of thinking-on) is lost.
**Cost line, anchored to the smoke measurement on these questions and this model:**
tokens per model ≈ 3,362 × (8.6 + 76 + 793 + 5,928) ≈ **22.9 M**, 87% of it thinking-alone.
At the smoke's 817 tok/s that is 7.8 h; throughput should rise with 256 concurrent streams but
that is NOT measured, so the range is **3–8 h per model, $1.5–4 at $0.49/hr**. Two pods in
parallel (9B, 4B): **$3–8 total**. First chunk of each condition replaces this estimate with a
measured one.

### BASE-01 results (run finished 2026-10-06 22:47 IST; scored with `score.py`, strict 6 s.f.)
**Every number below is from Claude's scorer and has NOT been checked by AG.** Raw files:
`runs/2026-10-06_base/` (10 files per model, checksums matched the pod before each pod was
deleted; copied to S3). Model commits: 9B `c2022362…`, 4B `851bf6e8…`.

| model | condition | correct (6 s.f.) | within 1e-3 | no answer | correct of those that answered | mean tokens | total tokens |
|---|---|---|---|---|---|---|---|
| 4B | alone, thinking off | 26.0% | 36.7% | 1 | 26.0% | 12 | 0.04 M |
| 4B | alone, thinking on | 55.1% | 68.1% | **994** | 78.2% | 4,820 | 16.20 M |
| 4B | sandbox, thinking off | 86.3% | 91.3% | 2 | 86.4% | 77 | 0.26 M |
| 4B | sandbox, thinking on | 94.4% | 95.4% | 153 | 98.9% | 1,007 | 3.38 M |
| 9B | alone, thinking off | 38.0% | 50.4% | 0 | 38.0% | 9 | 0.03 M |
| 9B | alone, thinking on | 51.1% | 59.4% | **1,354** | 85.5% | 5,732 | 19.27 M |
| 9B | sandbox, thinking off | 92.9% | 99.6% | 0 | 92.9% | 83 | 0.28 M |
| 9B | sandbox, thinking on | 94.4% | 96.3% | 123 | 98.0% | 777 | 2.61 M |

"No answer" = hit the 8,192-token budget, the 6-turn limit, or a malformed call. n = 3,362 each.

Correct (%) by family:

| family | n | 4B alone off | 4B alone on | 4B sbx off | 4B sbx on | 9B alone off | 9B alone on | 9B sbx off | 9B sbx on |
|---|---|---|---|---|---|---|---|---|---|
| trig_int | 1078 | 38.9 | 79.2 | 90.1 | 93.9 | 53.3 | 74.4 | 91.6 | 96.2 |
| trig_var | 1074 | 26.8 | 52.8 | 92.3 | 94.7 | 35.1 | 48.3 | 93.3 | 94.9 |
| transcendental | 507 | 24.5 | 52.3 | 60.0 | 94.7 | 52.5 | 45.2 | 93.3 | 87.0 |
| composite | 275 | 9.8 | 23.3 | 89.8 | 94.5 | 13.1 | 24.0 | 93.5 | 98.9 |
| nested | 172 | 1.2 | 16.9 | 82.6 | 96.5 | 7.0 | 14.5 | 95.3 | 97.7 |
| mixed | 240 | 0.0 | 25.4 | 96.2 | 92.9 | 0.4 | 27.9 | 95.8 | 91.7 |
| undefined | 16 | 93.8 | 75.0 | 100.0 | 93.8 | 68.8 | 56.2 | 68.8 | 100.0 |

Where sandbox questions went (count of 3,362):

| | 4B think off | 4B think on | 9B think off | 9B think on |
|---|---|---|---|---|
| correct, used the sandbox | 2,771 | 3,167 | 3,082 | 3,167 |
| correct, never called it | 131 | 6 | 42 | 6 |
| never called it, wrong | **300** | 0 | 8 | 3 |
| sandbox printed the right value, model wrote it wrongly | 158 | 29 | 143 | 37 |
| sandbox printed a wrong value (wrong code) | 0 | 7 | 86 | 26 |
| turn limit (6) | 0 | **143** | 0 | 17 |
| out of tokens | 2 | 10 | 0 | **106** |
| mean sandbox calls per question | 0.87 | 2.21 | 0.99 | 1.30 |

Re-ask of turn-limited questions at 12 turns (separate files, `*_turns12.jsonl`; the 6-turn
files are untouched): 4B 127/143 correct → merged **98.2%**; 9B 15/17 → merged **94.8%**.

**What the tables say (measured), and what they do not:**
- **A sandbox does not give 100%.** Best cell is 98.2% (4B, thinking on, 12 turns). The residue is
  not maths: output-precision slips, wrong code, running out of turns or tokens.
- **The 9B is NOT better than the 4B at thinking alone on the headline number (51.1 vs 55.1).**
  That is a budget effect: the 9B failed to finish 1,354 questions against the 4B's 994. Among
  questions it did finish it is more accurate (85.5 vs 78.2) — but those are different,
  self-selected subsets, so that comparison is NOT clean either. Unresolved; do not quote either
  number alone.
- **With thinking off the 4B often does not call the sandbox at all** (300 wrong without a call,
  mostly `transcendental`: 60.0%); the 9B almost always calls (8). Thinking on removes it.
- **Thinking + sandbox costs the 9B on `transcendental` (93.3 → 87.0).** 62 of 507 gave no answer.
  Not yet read — the traces are on disk.
- Both models reach exactly 3,173 correct in sandbox+thinking. Checked: coincidence, not a file
  mix-up — 3,017 shared, 156 unique to each.
- `undefined` has n=16; its percentages mean little.
- Strict 6 s.f. still conflates maths with output precision (the D3 decision, open). "Within 1e-3"
  is the nearest thing here to a maths-only number: 9B sandbox/off is 99.6% on it.

### Cost ledger — BASE-00 + BASE-01
| | |
|---|---|
| Opening balance (before smoke) | $104.04 |
| Before the full run | $103.84 |
| 19:05 IST, mid-run | $99.58 |
| After both pods deleted, 22:52 IST (provisional — RunPod posts the last minutes late) | **$97.25** |
| Spent, smoke + full baseline, two models | **$6.79** (provisional) |
| Estimate written before launch | $3–8 for the full run |

4B pod `7i0bkqnch457ja` deleted ~19:10 IST; 9B pod `yflbh95nnlsp72` deleted 22:50 IST.
`list-pods` returned an empty list at 22:51 IST. Tokens generated: 19.9 M (4B) + 22.2 M (9B),
84% of it in the thinking-alone condition (42.08 M total, recomputed 2026-10-16 from the raw files; the 87% first written here was the preflight estimate's share, not the measured one), 2,348 of whose 6,724 runs never produced an answer.

## PAPER-01 — first draft and two re-reads (2026-10-16, no GPU)
Draft: https://claude.ai/code/artifact/16e2050b-0bbe-4dde-820c-b4bf2ea87606 (Claude Docs). Every number
comes from `analysis/paper_numbers.py` → `analysis/paper_numbers.json`.
**Found by re-reading against the transcripts (all fixed):**
- Claude's "9B writes wrong code, e.g. degrees as radians" was FALSE. 77 of the 80 `tool_gave_wrong_value`
  runs have correct code that prints `f"{x:.6f}"` (6 decimal places); all 77 within 1e-3. Other 3 = undefined Qs.
- **Scorer bug:** `score.py` ignored the generator's `is_zero` flag; cos(90°), sin(180°), tan(180°), cos(270°),
  sin(360°), tan(360°) carry mpmath residues ~1e-43 and were unscoreable. Fixed; all files rescored; changes
  ≤ 0.2 pts (e.g. sandbox+thinking 94.4 → 94.6 both models). Old scores: `runs/2026-10-06_base/scored_before_is_zero_fix/`.
- "Right value already in a truncated trace" matcher read the "1." of a numbered list as 1.0 (sin(90°)).
  Now needs ≥5 significant digits and skips simple values: 65.2% (4B, 598/917), 73.2% (9B, 932/1274).
**New results (from data already on disk):**
- Thinking turns a right answer into a failure 182 (4B) / 402 (9B) times; 167 / 394 of those are truncation.
- Budget curve (exact under greedy): thinking alone ~0 correct below 2k tokens, still rising at 8,192;
  sandbox+thinking 9B 70.9% by 512 tokens. Oracle stop-at-first-correct-value ceiling: 72.9% (4B), 78.8% (9B).
- Matched-angle control: on the SAME first-quadrant d, thinking off — d 73.4/89.9%, −d 74.9/86.5%,
  d+360° 3.0/7.8% (4B/9B). Thinking on: d+360° 80.0/69.3%. 9B thinking on plain d falls 89.9 → 66.3%.
- Tabulation gradient replicates thinking off: mult-15 67.1/82.9 > other mult-5 44.4/72.2 > other 35–52%.
- Both-finished subset (1,633 Qs, thinking alone): 9B 88.7% vs 4B 85.4% — the 4B>9B headline is budget.

## DATA-01 — inspecting the fine-tune dataset (2026-10-18, no GPU, $0)

**D1 decided by AG:** own fine-tune on `nvidia/Nemotron-SFT-Instruction-Following-Chat-v4` (NVIDIA, created
2026-10-09, HF last-modified 2026-10-15, 3,850,582 conversations, 80.7 GB JSONL). Chosen as the *unrelated*
domain. Claude and Codex searched independently; Codex's top 3 were all NVIDIA maths sets (Math-v4, Math-v2,
Math-v3); Claude's were Math-v4 / Agentic-v4 / IFC-v4. Maths rejected after AG asked why: predictable direction,
adjacent to Cheng et al. 2605.06326, contamination risk. Agentic-v4 queued as the second run.

**Licence:** OpenMDW-1.1 (Linux Foundation, official text read from github.com/OpenMDW/OpenMDW 1.1/). Permissive:
"deal in the Model Materials without restriction"; keep the licence + notices on redistribution; patent/copyright
suit termination clause; no obligations on outputs. Not yet OSI-approved (OSI review thread, Aug 2026). Card adds
upstream prompt sources (WildChat-1M ODC-By, LMArena).

**Sample:** 600 rows, 200 per split, by HTTP range reads at 40 seeded random offsets per file
(`data_inspect/sample_ifc4.py`, seed 20261018; rows in `data_inspect/ifc4_sample600.jsonl`). Read by Claude by eye
(first user message of 80 chat + 30 length-control + 30 multi-turn rows; full text of several).
- Every row has `reasoning_content` on assistant turns (600/600). Zero tool calls/tool roles (0/600).
- `train_turns_mask` marks only the last assistant turn for training in the rows inspected.
- Heavily multilingual in `chat` (Chinese, Japanese, German, Spanish, French, Ukrainian, Kurdish) and code-heavy.
- **Maths is present, by eye:** chat ~13/80 (number series, 9.9 vs 9.11 ×2, 1.11 vs 1.9, polar-rose petals,
  continued-fraction asymptotics, probability, riddles); length-control ~4/30 incl. "Multiply 327492 and 238742"
  answered by mental long multiplication in the reasoning trace (correct, checked: 78,186,095,064); multi-turn IF
  ~1/30 (gallery commission arithmetic). Counts are by-eye judgement, not a classifier.
- **No trig/transcendental expression evaluation seen** in the 140 rows read. Contamination check against the
  3,362 questions not yet run.
- Implication (inference): `multi_turn_instruction_following` (69% of the dataset) is the cleanest "unrelated"
  split; `chat` carries the most maths. Filtering decision is AG's.

## FT-01 — fine-tune Qwen3.5-4B on instruction-following chat, then re-measure all four cells (2026-10-18)

**Question.** Does fine-tuning on an unrelated skill (multi-turn instruction following, no maths
targeted, no tool calls) change (a) tool-free maths and (b) maths with the sandbox, and does the
sandbox shield the model from whatever the fine-tune does to its own maths?

**Design, and why each piece is there (all choices Claude's; AG delegated them 2026-10-18).**
- Data: `nvidia/Nemotron-SFT-Instruction-Following-Chat-v4`, split `multi_turn_instruction_following`
  only (the split with almost no maths in DATA-01). Rows drawn uniformly at random by range reads,
  seed 20261018; each cut after its last `train_turns_mask` assistant turn; only that turn trained.
  Reasoning traces kept, rendered by Qwen3.5's own template as `<think>…</think>` (checked: the
  generation prompt is an exact token prefix of the full row). Rows > 8,192 tokens dropped.
- Contamination check vs the 3,362 questions (expression strings and ≥5-digit answers) before training.
- Training: full fine-tune of the language model (vision tower frozen), fp32 master weights + AdamW,
  bf16 autocast, gradient checkpointing, batch 1 × accumulation 16, peak LR 1e-5, 5% warmup then
  cosine to 10%, betas (0.9, 0.95), wd 0, clip 1.0, one epoch. Full fine-tune rather than LoRA
  because LoRA is reported to protect off-target skills (Biderman 2405.09673, abstract only) and the
  question is what an ordinary fine-tune does.
- Size rule, fixed now, before seeing throughput: K = min(5000, floor(90 min × measured tok/s ÷
  mean row length)), from a probe on the pod. One epoch over K rows.
- Dose: checkpoints at 25%, 50%, 100%. 50% and 100% are evaluated; 25% is stored for later.
- **Same-hardware baseline.** BASE-01 ran on A40s; these runs are on A100s. Greedy decoding is
  deterministic per machine, but different GPUs round bf16 differently and that alone can flip
  answers. So the untouched base model is re-run on the identical A100 setup, and every
  base-vs-fine-tuned comparison uses that re-run. Base on A40 vs base on A100 also measures how
  much answers move from hardware alone — the noise floor any fine-tuning effect has to beat.
- Same software as BASE-01: vllm 0.29.0, transformers pinned to 5.18.0, same `run_eval.py`,
  `score.py`, questions sha256 c01aa943…, greedy, seed 0, 8,192 tokens/turn, 6 turns, re-ask at 12.
- Save path verified before training: the untrained model is saved through the same routine and
  every one of the 738 tensors must match the hub checkpoint bit for bit (mtp.* tensors, which
  transformers drops on load, are copied back unchanged).
- Did the fine-tune take? IFEval (google/IFEval, Google's official scoring code), thinking off and
  on, for base and each checkpoint. If IFEval does not move, the maths result is uninterpretable.
- Known limits, stated now: one training run, one seed, one model size; the data was generated
  partly by Qwen3.5-397B (same family), so it may be gentler than foreign data.

**Rule-7 preflight (read from the code: ft/prep_data.py, ft/train.py, ft/eval_model.sh, run_eval.py).**
1. Progress lines? prep: every 500 rows fetched; train: every 10 micro-steps (loss, tok/s, peak
   memory, ETA); eval: one line per 256-question chunk (run_eval.py, unchanged).
2. Incremental writes? loss.jsonl per optimizer step; checkpoints at 25/50/100% each uploaded to
   S3 as it lands; eval results one line per question, synced to S3 every 5 minutes.
3. Memory? Training: fixed after the first longest row — fp32 weights + grads (~17 GB each) +
   AdamW (~34 GB) + activations at ≤8,192 tokens with checkpointing; the probe measures it before
   the run (A100 80 GB; if it does not fit, the fallback is recorded before use). Eval: vLLM fixes
   90% of the card at start.
4. Killed at 80%? Training: the 25% and 50% checkpoints are already in S3. Eval: everything but
   the last ≤5 minutes is in S3.
**Cost line.** A100 SXM 80 GB, $1.79/hr (capacity read 17:20 IST, Medium stock). Training pod
≈2.5 h incl. setup ≈ $4.5. Three eval pods (base re-run, s050, s100), each ≈2–3 h on A100 vs the
4.4 h measured on A40 ≈ $4–5.5 each. **Total ≈ $17–21; treat as a floor** (past estimates ran
1.3–1.5× over). Opening balance **$71.26** (17:17 IST; a separate project's H200 pod was also
billing $5.29/hr on the account).
Watchdog: `mats_2027/pipeline/pod_watchdog.sh` on every pod.

### FT-01 run log (times UTC; IST = UTC+5:30)
- 11:56 pods created: `vqvoinv2yx1fq8` ft01-train-4b (A100 SXM, US-MD-1, interp image, transformers
  5.17.0) and `5wrepjap49a1bv` ft01-eval-base-a100 (A100 SXM, EUR-IS-1, CUDA 13.0, driver 580.159.04).
  $1.79/hr each. Watchdogs running on both. S3 access via IAM user `ft01-pod-uploader` (inline policy:
  this bucket, prefix `posttrain_math/*` only); delete the user after the run.
- Base-eval setup reproduced BASE-01's software exactly: torch 2.13.0+cu130, transformers 5.18.0, vllm 0.29.0.
- **Problem 1 — `boto3` not installed on the training pod.** ssh sessions do not inherit the image's
  Docker ENV (`PIP_BREAK_SYSTEM_PACKAGES`), so pip refused. Would have crashed at the first checkpoint
  upload. Fixed with `--break-system-packages`; S3 write tested from the pod.
- **Problem 2 — Hugging Face rate limit.** Unauthenticated: 3,000 requests per 5-minute fixed window
  (response header `ratelimit-policy: "fixed window";"resolvers";q=3000;w=300`). The first fetcher gave
  up after 4 quick retries and would have silently dropped rows. Rewritten to honour `Retry-After`;
  second attempt: 7,000/7,000 rows, 96 waits, 0 other errors.
- **Problem 3 (mine) — `pkill -f` over ssh killed its own shell** (the pattern matched the ssh command
  line — the exact trap recorded 2026-09-22). Left the old data prep running beside the new one, both
  writing one file. Caught by listing processes; old one killed by PID before either wrote output.
- Data (12:16): 7,000 drawn, 6,984 unique, 108 dropped > 8,192 tokens, 5,000 kept; 14.8 M tokens,
  6.6 M in trained turns; median row 2,637 tokens, p95 6,235; 3.4 user turns on average.
  2,075/5,000 rows match a crude maths keyword regex (`\d+ op \d+`, sin, compute, …) — a loose upper
  bound, not a count of maths conversations.
- **Contamination: 6 expression hits + 6 answer hits, all textbook constants** — cos 30°, sin 45°,
  cos 45°, tan 60°, √2, √25; e (2.71828, twice), ln 2, √13, √33, and 23.4499 (our √14.1·√39; row not yet
  read). 12 of 3,362 questions (0.36%). Every result will be reported with and without these 12 ids:
  q00091 q00135 q00136 q00182 q02415 q02438 q02158 q02226 q02426 q02446 q03070 q03211.
- **Problem 4 — the save test caught a real bug.** The hub checkpoint stores 48 tensors in fp32
  (24 `linear_attn.norm.weight`, 24 `linear_attn.A_log`); the first save routine cast all tensors to
  bf16 (max change 0.0039). Every fine-tuned checkpoint would have differed from the base for a reason
  unrelated to fine-tuning. Fixed: each tensor saved in the hub's own dtype. Re-test must be bit-exact.
- Kernel log line is NOT evidence either way: the decorator-wrapped functions report the transformers
  module, so it cannot show whether fla's kernel is used. The probe's tokens/s is the real check.
- 12:24 save test PASSED on the A100 after the dtype fix (all 738 tensors bit-exact).
- **Problem 5 — out of memory on the A100.** Probe: 44.1 GiB weights+grads+activations + 31.3 GiB AdamW
  = 75.4 of 79.2 GiB; the run died at row ~12 (before the first optimizer step) on a 2.27 GiB logits
  allocation with 9 GiB fragmented. The preflight called this "tight"; it does not fit. Recipe NOT changed
  to squeeze it (no 8-bit optimizer, no frozen embeddings — those would change what is being tested).
  Moved training to an H200 141 GB (`qftwj3iayhv5k0`, EUR-IS-4, $5.29/hr, driver 570.211.01), same image,
  same data (sha256 fead3fcc… identical to the A100 copy), plus `expandable_segments:True` (allocator only).
- **Problem 6 (mine) — the probe measured speed wrongly.** It timed the first rows, which include one-off
  kernel autotuning: 267 tok/s. The real run logged 3,087 tok/s over its first 10 rows. The pre-written
  rule K = min(5000, 90 min × tok/s ÷ 2,968) gives 485 from the bad number and 5,000 from the real one.
  The rule is unchanged; the probe now times rows 8–23 after warm-up. Expected K = 5,000 (all rows),
  i.e. one epoch over 14.8 M tokens (6.6 M trained).
- The A100 training pod `vqvoinv2yx1fq8` is reused as the s050 eval pod (vLLM venv installed while it
  waits for the checkpoint's `_COMPLETE` marker in S3). Its driver is 580.126.16 vs 580.159.04 on the
  base-eval pod; same GPU model, same CUDA 13.0 toolkit in the venv. Noted, not expected to matter.
- Kernel check, from the transformers 5.17.0 source (integrations/hub_kernels.py): on fallback it logs
  "`chunk_gated_delta_rule` is falling back to its reference PyTorch implementation". Count of that line
  in the A100 logs: 0 → fla kernels were in use.
- **Problem 7 — fla refuses Hopper + Triton 3.4.** H200 probe stopped with fla's own guard: "Triton >= 3.4.0
  and < 3.7.1 on Hopper GPUs produces incorrect results for gated chunk_bwd_dqkwg (see #640)". The image
  pins torch 2.8 / Triton 3.4.0. The A100 (Ampere) is not affected. Fix = the maintainers' stated
  option: `pip install tilelang` (0.1.15; dry run confirmed torch/Triton untouched). fla's backend then
  logs `common.chunk_bwd_dqkwg -> tilelang`. **Checked, not assumed:** fla output and gradients vs the
  transformers reference PyTorch `torch_chunk_gated_delta_rule` on Qwen3.5-4B shapes (T=1024, 32 heads,
  bf16): relative error out 0.0044, dq 0.0045, dk 0.0048, dv 0.0042, dg 0.0038, dbeta 0.0042 — bf16
  rounding level. Script `ft/gdr_check.py` (on the pod, copy in scratch). Relaunched ~12:42 UTC.

### FT-01 cost ledger (prices from the RunPod create responses; balance from the billing API)
| pod | id | GPU | $/hr | created (UTC) | ended | role |
|---|---|---|---|---|---|---|
| ft01-eval-base-a100 | 5wrepjap49a1bv | A100 SXM 80GB | 1.79 | 11:56 | 14:24 (404 confirmed) | base re-run on A100 — 2.45 h ≈ $4.38 |
| ft01-train-4b → s050 eval | vqvoinv2yx1fq8 | A100 SXM 80GB | 1.79 | 11:56 | 14:46 (404 confirmed) | training (OOM) → s050 eval — 2.79 h ≈ $4.99 |
| ft01-train-4b-h200 | qftwj3iayhv5k0 | H200 141GB | 5.29 | 12:27 | 13:23 (terminated, 404 confirmed) | training — 0.93 h ≈ $4.94 |
| ft01-eval-s100-a100 | 19j4pwzd7l6wkq | A100 SXM 80GB | 1.79 | 13:04 | 15:13 (terminated; list-pods empty) | s100 eval + analysis (US-KS-2, driver 580.159.04 = base pod) — 2.09 h ≈ $3.73 |

| time (UTC) | balance | spend rate | note |
|---|---|---|---|
| 11:47 | $71.26 | $5.32/hr | opening; all of it the other session's H200 (`bp83ipehsc3oha`, gemma4devs) |
| 12:38 | $64.23 | $8.94/hr | = my three pods (1.79 + 1.79 + 5.29 = 8.87) + disk; the other H200 is no longer billing |
| 15:15 | $49.33 | $0.00/hr | zero pods on the account. PROVISIONAL — RunPod posts the last minutes late; re-read before quoting |
| later | $49.23 | $0.00/hr | re-read: list-pods empty; −$0.10 = late-posted minutes. Two global volumes still exist (storage only, not FT-01) |

Spent 11:47→12:38: $7.03 on the account. Mine by rate × time ≈ $3.5 (two A100s × 42 min + H200 × 11 min);
the rest is the other session's H200 before it stopped. Estimate before launch: $17–21 (floor); it will
rise by ≈ $4–5 because training moved to an H200 at $5.29/hr.
- 12:50 s025 checkpoint in S3 (17 files, 9.34 GB, _COMPLETE last); stored, not evaluated (plan). 13:03 s050 in S3; s050 eval pod found the marker and loaded it 13:04.
- 12:40–13:20 training on the H200: 5,000 rows, 313 optimizer steps, ~6,000–8,000 tok/s, peak 88.7 GiB; loss(last10) 0.99 at the start → 0.70–0.94 later (per-row noise; loss.jsonl in S3). Checkpoints s025/s050/s100 in S3 (17 files, 9.34 GB each). All logs + data verified in S3 before the H200 was terminated.
- First scored cell (model alone, thinking off), PRELIMINARY: base A100 26.3% (A40 base 26.1%) vs s050 24.4%; trig_int 39.3→34.7, trig_var 26.9→20.5, transcendental 24.9→31.0, composite 9.8→14.2, mixed 0.0→4.6 (within 1e-3: 0.4→22.5); s050 has 24 runs hitting 8,192 tokens with thinking OFF (base 0).
- **Design gap, stated 2026-10-18 (AG asked whether IFEval is the control):** the CONTROL is the untouched base
  re-run on A100; IFEval is a manipulation check (did the fine-tune change the trained skill). There is NO
  untargeted second skill yet, so FT-01 can show "maths changed while the trained skill changed" but not
  "maths changed more than other untrained skills". Candidate fix (later chunk, AG's call): a knowledge
  benchmark (MMLU-style) on base / s050 / s100, same pods and harness style.

#### FT-01 run log — base re-run finished (2026-10-18 14:21 UTC)
- `ALL DONE Qwen3.5-4B-rerun` at 14:21:37. 23 result files on the pod, 23 in S3 at
  `posttrain_math/ft01/results/Qwen3.5-4B-rerun/`, sizes identical (checked with boto3 on the pod), plus the eval log.
- Pod `5wrepjap49a1bv` terminated 14:24 UTC, get-pod returns 404. Uptime 8,812 s × $1.79/hr ≈ **$4.38**. Watchdog `wd_base` stopped.
- **IFEval base, prompt-level strict: thinking off 82.4%, thinking on 37.5%.** The thinking-on number is mostly a budget
  measurement, not instruction following: **329 of 541 responses never closed `</think>` inside 8,192 tokens**
  (mean 6,229 tokens). An unclosed think leaves no scored response, so those prompts fail. Read thinking-on IFEval
  only together with its unclosed count, and compare models on the closed subset as well as overall.

#### FT-01 run log — s050 eval finished (2026-10-18 14:41 UTC)
- `ALL DONE Qwen3.5-4B-ifc4-s050` at 14:41:47. 24 files on the pod = 24 in S3, sizes identical. 35 turn-limit questions re-asked at 12 turns.
- Pod `vqvoinv2yx1fq8` (training A100 reused for the s050 eval) terminated 14:46 UTC, 404 confirmed.
  Uptime 10,036 s × $1.79/hr ≈ **$4.99** (this includes the failed A100 training attempt). Watchdog `wd_train` stopped.
- **IFEval, prompt-level strict, base → s050:** thinking off **82.4% → 72.8%** (down 9.6 pts); thinking on
  **37.5% → 62.3%**, with unclosed thinking falling 329 → 151 of 541.
  Thinking-off mean response length went **424 → 762 tokens**, and **34 of 541 hit the 8,192 cap** (base: TBD from the raw file).
- **Checked on the pod, not assumed:** the thinking-off failures are genuine, not a parsing artefact. Only 3 of 541
  contain `</think>`; failures are commas when commas were banned, wrong word or capital counts, and
  responses that run to the cap. This matches the 24 thinking-off maths runs that hit 8,192 tokens.
- **So "did the fine-tune take" does not have a one-word answer.** The thinking-on gain is real but is partly
  "finishes its thinking more often". Thinking off got WORSE at the very skill that was trained.
  Hypothesis, NOT tested: the training rows were rendered in the thinking format (`reasoning_content` → `<think>…</think>`),
  so the model was trained mainly in thinking-on mode. Check the share of training rows with non-empty reasoning in
  `ft01/data/train.jsonl` before the analysis claims anything about it.

**FT-01 cost, closing (provisional):** my four pods by uptime × rate = 4.38 + 4.99 + 4.94 + 3.73 = **$18.04** + container disk.
Account 11:47 → 15:15: $71.26 → $49.33 = $21.93, of which ≈ $3.5 is the other session's H200 before 12:38.
Pre-launch estimate was $17–21 floor, +$4–5 for the H200 move → **inside the estimate.**
IAM user `ft01-pod-uploader` deleted 15:16 UTC (access key, inline policy, user; GetUser → NoSuchEntity).
Local copy of its key (`scratchpad/s3env`) deleted.

#### FT-01 RESULTS (2026-10-18). Qwen3.5-4B, full fine-tune on 5,000 rows of Nemotron IFC-v4, 313 steps.
All numbers are strict 6 s.f., the frozen 3,362 questions, greedy, same harness as BASE-01, all on A100 SXM.
Paired = same question before/after (McNemar). Hardware noise = same base model, A40 vs A100.
Computed on the last eval pod by `analysis/run_ft01_analysis.sh`; outputs in S3 `posttrain_math/ft01/analysis/`
and locally `runs/2026-10-18_ft01/` (556 KB: numbers json/txt, 6 figures, call rates, train-think check).

| cell | base A100 | s050 | s100 | s100 vs base (+gained/−lost, z) | hardware noise (z) |
|---|---|---|---|---|---|
| alone, thinking off | 26.3 | 24.4 | 24.3 | +144/−211, **z −3.56** | +17/−9, 1.57 |
| alone, thinking on | 55.3 | 60.6 | 59.2 | +374/−242, **z +5.32** | +265/−259, 0.26 |
| sandbox, thinking off | 86.2 | 94.5 | 93.4 | +293/−50, **z +13.1** | +25/−29, −0.54 |
| sandbox, thinking on | **98.3** | **80.8** | **81.4** | +29/−597, **z −22.7** | +36/−30, 0.74 |

- Excluding the 12 contamination-hit ids changes no conclusion (largest change: s100 alone-off z −3.56 → −3.67).
- **The biggest effect is the sandbox with thinking on, and it is a BEHAVIOUR change, not a maths change.**
  Share of answers given WITHOUT calling the sandbox (`analysis/call_rates.py`):
  base 0.2% → s050 **55.1%** → s100 **38.5%** (thinking on); base 13.2% → s050 3.3% → s100 5.8% (thinking off).
  Of the 571 questions s100 lost in that cell, 435 are "answered from memory without calling, wrong", 90 truncated.
  With thinking off the opposite happened: the fine-tuned model calls MORE and gains (no_call_wrong 303 → 61).
- **Training data, checked (not assumed):** all 5,000 trained rows have NON-EMPTY thinking (`train_think_check.txt`).
  So the model was trained only in thinking mode, on a chat dataset with no tools. Reading (interpretation, not tested):
  thinking-on behaviour shifted toward "think, then answer directly", which in a tool setting means "don't call".
- **Shield statistic (per question, change alone − change with sandbox), s100:** thinking off −9.2 pts [−10.7, −7.8];
  thinking on **+20.8 pts [18.9, 22.7]**. The sandbox did NOT shield maths from the fine-tune with thinking on; it was
  where the damage concentrated. With thinking off, the sandbox cell improved more than the alone cell.
- **Alone, thinking on:** the +4 pts is mostly "finishes inside 8,192 tokens more often" (no answer 1000 → 220),
  not better arithmetic. Note the hardware noise in this cell is huge in flips (+265/−259) but balanced (z 0.26).
- **Alone, thinking off:** −2 pts, z −3.6. Small and real. Families move in opposite directions (F6): trig −5.6/−5.9,
  composite +8.7, mixed +6.7. 8 s100 runs (24 at s050) hit the 8,192 cap with thinking OFF; base 0.
- **IFEval (manipulation check) — the trained skill did NOT improve in quality:** prompt-strict, base → s100:
  thinking off 82.4 → 75.0 (+26/−66, z −4.2); thinking on 37.5 → 62.1 (+167/−34, z +9.4), BUT on the 190 prompts where
  both models closed `</think>`, 95.3 → 90.5 (z −2.3). The thinking-on gain is entirely "stops thinking in time"
  (unclosed 329 → 158). So the fine-tune took — it changed behaviour strongly and shortened thinking — but it did not make
  the model follow instructions better on any like-for-like comparison.
- **Loss barely moved** (10-step mean ≈0.85 → ≈0.79 over 313 steps, F4). The data was already close to the model's
  own distribution. Most of what changed is length/termination and tool-calling behaviour.
- **s050 ≈ s100 everywhere** (F2): the change happens in the first half of training and then plateaus.
  s025 was saved and not evaluated.
- Figures: F1 four cells, F2 dose, F3 IFEval (unclosed counts printed), F4 loss, F5 flips vs hardware noise,
  F6 family heatmap (x-labels overlapped; code fixed, regenerate on the next pod).

**What FT-01 can and cannot claim.** CAN: an off-domain SFT (instruction-following chat, thinking-mode only) moves
maths in all four cells by more than hardware noise, and moves the sandbox cell most, through tool-call behaviour.
CANNOT: that maths changed more than untrained skills (no untargeted control yet — design gap above); that any of this
generalises beyond one 4B model, one dataset, one seed; why the call rate fell (interpretation only).

**Open, AG's call:** (1) an untargeted control (MMLU-style) on the three models; (2) s025 eval to place the onset;
(3) a mixed thinking/non-thinking or tool-containing SFT to test the "thinking-mode-only data" reading — that is the
experiment that would turn the interpretation into a result; (4) read ~20 of the no-call wrong answers by hand.
- **Correction (2026-10-18, found while writing the paper section):** the A40 baseline scores in S3 were the
  PRE-`is_zero`-fix versions (26.0% vs the corrected 26.09%), so the first hardware-noise numbers compared
  old-scored A40 files with new-scored A100 files. Corrected files uploaded to S3 (all 10 baseline scored files,
  HTTP 200), noise recomputed from S3: notool_off +15/−9 (z 1.22), notool_on +265/−259 (0.26),
  tool_off +24/−29 (−0.69), tool_on +30/−30 (0.00). No conclusion changes. F5 redrawn; ft01_numbers.json patched.
