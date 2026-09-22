# Forking — experiment log

Format per entry: ID · date · question · method · result · caveats · artifacts. One entry
per experiment; results are provisional until AG has done an independent check.

---

## NOISE-LAW-01 · 2026-09-07 · zero GPU

**Question.** Does the 2026 paper's central claim hold when recomputed from their shipped
data with independently written code: replicate TVD of o_t falls as S^-1/2, matching an
i.i.d. multinomial null?

**Why first.** Every later step (smoothing, fork detection, our own sampling budget) assumes
this noise law. If it failed here, nothing downstream could be trusted. It also verifies the
store format and their loader before we generate any data of our own.

**Data.** `data/s1000/llama/row039_everytoken.json.gz` — tinyMMLU row 39, Llama-3-8B-
Instruct, 344 positions, 535 branches, 1000 draws per branch. Loader reproduces the
recorded `o_t_full` exactly (`matches_recorded == True`).

**Method.** `forking/pipeline/noise_law.py`. For S in {5,…,500}: split the 1000 draws per
branch into 1000/S disjoint contiguous blocks; estimate o_t per block with the paper's
tok_p-weighted per-branch histogram; TVD between every pair of blocks at every position;
mean over positions and pairs. Null: at each position draw multinomial(S) counts per branch
from that branch's full-pool histogram, build two independent estimates, TVD; 20 reps per
position. Slope by least squares on log-log.

**Result.**
| S | measured | null | ratio |
|---|---|---|---|
| 5 | 0.3146 | 0.3167 | 0.993 |
| 20 | 0.1660 | 0.1653 | 1.004 |
| 50 | 0.1057 | 0.1058 | 0.999 |
| 100 | 0.0751 | 0.0753 | 0.997 |
| 200 | 0.0537 | 0.0529 | 1.016 |
| 500 | 0.0360 | 0.0341 | 1.056 |

Slope through S≤200: **−0.483** (paper −0.4903; theory −0.5). All-S slope −0.480. Ratio to
null 0.99–1.02 through S=250 (paper 0.98–1.01). **Reproduced.**

**Caveats.** One question, one model. S=500 has only 2 blocks → 344 pairs, noisiest point,
ratio 1.056; not evidence of excess noise. Paper's slope was fitted on two questions; row 12's
S=1000 every-token store is not shipped (their cp-2 commit says it was unrecoverable). The
null anchors on the full-pool histogram as truth, which itself carries 1/√1000 noise — a
small downward bias on the null at large S, consistent with the ratio drifting above 1.

**Independent check owed (AG).** Pick two positions by hand, compute o_t from draws 0–199
and 200–399 with a spreadsheet or five lines of Python you write, and match the script's
per-position TVD. Not done yet.

**Artifacts.** `forking/runs/2026-09-07_noise/noise_law.json`, `noise_law.png`.

---

## SWAP-01 · 2026-09-13 · zero GPU · compass Q3

**Question.** At positions where the pooled curve o_t is flat, do individual next-token
branches still produce different final-answer distributions, beyond sampling noise? The 2026
paper models only the pooled o_t; the 2024 paper's per-token claim is about o_{t,w}.

**Data.** All 200 shipped S=200 stores (100 tinyMMLU rows × Llama-3-8B every-token track and
R1-Distill-Llama-8B sentence track). Per branch: 200 recorded outcomes.

**Method.** `forking/pipeline/swap_effects.py`. For every position with ≥2 kept branches and
every alternative w: effect = TVD(o_{t,base}, o_{t,w}) on 200 draws each; permutation test
(pool 400 outcomes, shuffle, split, 500 reps) for a p-value; local pooled-curve change =
TVD(o_t, o_{t+1 grid}) from `o_t_full`; alternative token decoded and classed. Answer-letter
tokens (A–D as base or alt) excluded from the headline — swapping the letter itself is
trivially a fork (280 such branches on Llama).

**Result — Llama track (clean; 0% Other).**
- 13,149 non-letter alternative branches over 7,489 positions, 100 questions.
- **7.2% of swaps are significant at p<0.01** (chance 1%). 315 have effect ≥0.2; 55 have ≥0.5.
- 84.3% of branch positions sit where the pooled curve is flat (local change <0.05). Sig rate
  there is 4.6%, vs 21.0% where the pooled curve moves. So swaps concentrate at visible forks —
  but **83 significant swaps with effect ≥0.2 sit at flat positions, in 41 of 100 questions.**
  These are per-token forks the pooled curve cannot show. Examples: row 80 t=228 ")(" vs ")["
  effect 0.96 with pooled change 0.04; row 24 "drug" vs "beta" 0.745; row 46 "The" vs "Ah" 0.455.
- Token class: digits 12.1%, mixed 9.6%, punct/symbol 8.5%, whitespace 6.8%, words 6.9%.
  Odd-token forks are real at S=200 but only modestly more common than word forks.

**Result — DeepSeek track: DATA-QUALITY PROBLEM, do not use as-is.**
- **46.9% of all recorded outcomes are "Other"** (Llama: 0%). Per-store diagnostics say
  `answered_rate ≈ 0.12`: most 1536-token continuations never reach a parseable answer.
  Pooled Other share falls from 0.53 (early) to 0.41 (late); 38 of 100 rows end with
  Other >0.5 at the last position.
- The top DeepSeek "swap" (row 93 t=843, "." vs ")", effect 0.995) is an artefact: the base
  branch is Other 199/200 because the continuation after "The correct answer is D." never
  restates the letter, while the ")" branch parses as D 200/200. The extractor only sees the
  continuation, not the prefix. The 2026 paper's Fig 10 caption acknowledges "the fork occurs
  when continuations stop resolving to an answer" — so they knew, and the pooled analysis
  carries it. Raw text exists per row (`text_sidecar` → `rowNNN.jsonl.zst`) but was not
  shipped in the release.
- Consequence: any DeepSeek number in the paper that treats Other as an outcome category is
  partly measuring truncation, not reasoning. **Report as a limitation of their release.**

**Caveats.** p-values are per-branch, no multiple-comparison correction (7.2% vs 1% chance
is the honest comparison). Effect sizes on 200 draws are biased upward by noise (null mean
≈0.027); the ≥0.2 cut is well above it. "Flat" uses a single grid step; a fork one step later
would still count as flat here. No independent check by AG yet.

**Independent check owed (AG).** Row 80 t=228: pull the two branches' 200 outcomes, count
A–D for each, compute TVD by hand. Should be ≈0.96.

**Artifacts.** `forking/runs/2026-09-13_swap/`: `swap_effects.parquet/.csv` (all),
`swap_effects_nonletter.csv`, `fig1_effect_vs_null.png`, `fig2_swap_vs_flatness.png`,
`fig3_class_rates.png`.

---

## QWEN-01 · 2026-09-14 · in progress

**Hypothesis (AG's question, 2026-09-14):** "Will a modern open model still exhibit this
behaviour?" Operationalised: on row 80, Qwen3.5-9B (thinking off) has at least one position
where two next tokens with p≥0.05 lead to final-answer distributions differing by TVD ≥0.5 on
200 draws each. Llama-3-8B's value at token 228 was 0.96 (")(" vs ")[", C↔A).
*If yes:* per-token forks are a property of autoregressive generation, not of a 2024 model.
*If no:* either the newer model is robust on this item, or its forks sit below the 5% cutoff;
one question cannot separate those — say so.

**Infra.**
| | |
|---|---|
| Pod | `tijztrbxq5pc9t` "forking-qwen01", RunPod SECURE, EU-SE-1 |
| GPU | NVIDIA A40 48 GB, $0.49/hr, created 2026-09-14 16:11 UTC |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (torch 2.8.0+cu128 preinstalled) |
| Disk | 80 GB container disk, no network volume |
| SSH | direct 194.68.245.209:22022, key `~/.ssh/id_ed25519_runpod_new`; helper `pipeline/pod/pssh.sh` |
| Stack | transformers 5.17.0 + flash-linear-attention + causal-conv1d (Qwen3.5 hybrid linear attention) |
| Model | Qwen/Qwen3.5-9B, bf16, thinking OFF via `enable_thinking=False`; eos <|im_end|>,<|endoftext|> |
| Sampling | S=200, N0=200, T=1.0, p_thresh 0.05, top_k 10, base ≤400 tok, continuation ≤400 tok, `--only-forks` |
| Watchdog | `mats_2027/pipeline/pod_watchdog.sh`, 6-min idle alarm, log `runs/2026-09-14_qwen01/watchdog.log` |
| Balance before | $55.40 |
| Budget | expected $1–2; stop and inspect if generation passes ~3 h |

**Divergences from the paper's Llama run, all deliberate:** newer model; thinking off (the
paper's Llama had no thinking mode); `--only-forks` skips positions with a single candidate
(so no full o_t curve — this run answers the per-token question only); continuation text saved.

**Log of the session (UTC).**
- 16:11 pod created. 16:12 SSH up, files copied, watchdog self-test OK.
- 16:13 `pod_setup.sh` launched detached → `/root/setup.log`.


**Question.** Does a current dense open model (Qwen3.5-9B, thinking off) show a per-token
fork on tinyMMLU row 80, where Llama-3-8B flips C↔A on ")(" vs ")[" at token 228?
**Plan.** Local smoke on Qwen3.5-0.8B (laptop, then DELETE the local checkpoint — AG's
instruction), then one pod session for the 9B at S=200 every position, then terminate.
Adapter: `forking/pipeline/run_qwen.py` (wraps the sampler unchanged; saves continuation text).
- **Smoke (laptop, 2026-09-14):** Qwen3.5-0.8B on MPS, float32, S=3, 60-token base, 40-token
  continuations. 566 s. Base text has no `<think>` tag (thinking off works); eos = <|endoftext|>,
  <|im_end|>; 115 branches at 28 of 60 positions; 345 continuations saved as text; regex
  coverage 1.4% (expected — 40-token continuations rarely reach an answer), logit fallback
  handled the rest with 5/5 agreement on the resolved ones. Qwen3.5 loads via
  AutoModelForCausalLM despite the vision tower. **Pod note:** Qwen3.5 is a hybrid
  linear-attention model; without `flash-linear-attention` and `causal-conv1d` transformers
  falls back to reference PyTorch kernels ("correct but much slower") — install both on the pod.
  Local checkpoint deleted after the run (AG's instruction); cache confirmed clean.
- 16:20 setup done: transformers 5.17.0 on torch 2.8.0+cu128, Qwen3.5-9B downloaded (~18 GB
  on GPU in bf16), 5-token generate OK ("Hi there! How can"). **Kernel install failed** in one
  pip command: `causal-conv1d` has no wheel matching this stack and the source build died,
  taking `flash-linear-attention` with it. Fixed fla separately (0.5.2, imports). For
  causal-conv1d, per its official `setup.py` (Dao-AILab/causal-conv1d): it first tries a
  prebuilt wheel named by cuda/torch/abi/python, else compiles; needs torch importable at build
  time and CUDA_HOME. Pod has nvcc at /usr/local/cuda-12.8 but CUDA_HOME unset. Retrying with
  `CUDA_HOME=/usr/local/cuda` and `--no-build-isolation`. Without it the depthwise conv falls
  back to PyTorch — correct, slower; the bigger kernels (fla) are now in place.
- 16:24 `causal-conv1d` 1.7.0 built and imports (CUDA_HOME + --no-build-isolation was the fix;
  ninja pulled in). Both fast kernels present. Balance $55.35 before launch.
- 16:25 run launched via `run_qwen01.sh` → `/root/out/run.log`.
- 16:26 first launch crashed at import: the adapter located the sampler package relative to
  its own path (laptop layout); on the pod the file sits in /root. Fixed to also look in
  /root/forking-fast or `FORKING_FAST_ROOT`. Claude's error, ~2 min lost.
- 16:28 relaunched, pid 3798. Model loaded, thinking off, eos [<|endoftext|>, <|im_end|>].
  **Base path decoded; 280 branches at 108 positions after `--only-forks`** (Llama's shipped
  row 80 had branches at far fewer positions — Qwen is less certain token-to-token on this item,
  or its top-10 has more ≥5% ties; check after). GPU 99%, 40 GB. 280 × 200 continuations ahead.
- 16:38 **No interim logging — their run.py prints once per question; inside a question the
  280-branch loop is silent.** Attempted a live stack dump with py-spy: blocked
  (`ptrace_scope=1`, /proc/pid/mem denied in the container). Evidence the run is working:
  GPU 99%, 41 GB, process state R, 76 threads, wchar growing. Adapter patched LOCALLY to
  log one timestamped line per branch (prefix length, mean continuation length, seconds);
  applies from the next run. Not hot-swapped — the current run keeps its seed stream.
  Note the seed changes: per-branch calls use seed+i, so a rerun with the new adapter will
  not be draw-identical to this one. Neither is wrong; both are T=1 samples.
- 16:50 **Standing rule adopted (AG): pre-launch checklist for every billed run** — progress
  logs, incremental checkpoints, memory trajectory, kill safety — answered in the journal
  from READING the code, before the pod exists. Saved as memory `gpu-run-preflight`. For this
  run the answers would have been: no / no / grows (prefix-sorted batches, 40→43 GB of 46) /
  nothing survives. Risk accepted for QWEN-01 rather than restart; rerun plan if it OOMs:
  gen_batch 100 + per-branch logging + per-branch checkpoint appends.
- 18:13 stuck-or-slow check: CPU +60 s per 60 s, GPU 287 W, 45 GB — working, not stuck.
  Memory peaked 45.2 GB of 46 in the long-prefix tail; never OOMed.
- **18:27 KILLED at AG's decision after 2 h 07 min of generation, no output written.**
  Their code writes once at the end; nothing survived. AG: "kill ..no rerun..am just bit
  disappointed." Pod terminated immediately after; logs (setup, kernels, run) pulled to
  `runs/2026-09-14_qwen01/qwen01_logs.tgz`.

**Result.** None. QWEN-01 produced no data.

**Cost.** ~2 h 17 min of A40 at $0.49/hr ≈ **$1.12**. Balance to be confirmed below.

**What was learned (all recoverable value of the run):**
1. Qwen3.5-9B on this stack works end to end: loads via AutoModelForCausalLM, thinking off
   via `enable_thinking=False`, eos [<|endoftext|>, <|im_end|>], fla + causal-conv1d build.
2. On row 80, thinking off, Qwen's base path has **108 forking positions, 280 branches** —
   far more than Llama's shipped row 80. That number is real and cheap to regenerate.
3. Throughput with HF `generate` at batch 200 on an A40 is far below what 280×200×≤400
   tokens needs in a sitting: >2 h without finishing. First-principles floor ~26 ms/step
   (18 GB weights / 696 GB/s); observed effective rate evidently 2–4× worse.
4. Their code: silent inside a question, writes once at the end, sorts branches by prefix
   length so memory climbs late. All flagged now; none flagged before launch — Claude's miss.

**If ever rerun (parked, AG's call):** vLLM/SGLang with prefix caching instead of HF generate;
S=50 at stride 4 first, S=200 only at positions that look interesting; per-branch log +
per-branch checkpoint (adapter already patched for the log); gen_batch ≤100; cost line
(tokens × rate → hours → dollars) written before the pod exists. Estimated <20 min.
- 18:28 pod `tijztrbxq5pc9t` terminated (HTTP 204). `list-pods` → []. Watchdog and monitor
  stopped. **Balance $54.27** (was $55.40) → session cost **$1.13**. The "spend/hr $0.50"
  reading is RunPod's lagging meter; zero pods confirmed.

---

## QWEN-02 · 2026-09-15 · PRE-LAUNCH CHECKLIST (rule 7) — answered before any pod exists

**Question.** Same as QWEN-01: does Qwen3.5-9B (thinking off) show per-token forks on row 80
comparable to Llama's ")(" vs ")[" flip? Falsifiable sentence unchanged.

**Docs read today, all official:** vLLM SamplingParams (n, temperature, logprobs,
prompt_logprobs, seed), LLM.generate / LLM.chat + chat_template_kwargs (source
`vllm/entrypoints/llm.py`), TokensPrompt (`vllm/inputs/llm.py`), automatic prefix caching
(opt-in `enable_prefix_caching=True`; hybrid/Mamba models hit at block boundaries), GPU
install page (fresh venv; wheel bundles torch; CUDA 12.8 wheels via extra index), vLLM
releases (0.27.0 added Qwen3.5 dense; 0.29.0 current on PyPI with Qwen3.5 GDN fixes;
`model_executor/models/qwen3_5.py` + registry entries for both text-only and
ConditionalGeneration names), Qwen3.5-9B card (thinking off via `enable_thinking=False`;
its "vLLM from main" note predates 0.27 and is stale).

**Stack.** vLLM 0.29.0 in `/root/venv` (fresh, per docs), model Qwen/Qwen3.5-9B bf16,
`max_model_len 2048` (prompt ~250 + base ≤400 + cont ≤400 + fallback suffix, with margin),
prefix caching on, A40 48 GB $0.49/hr SECURE.

**The four answers, from reading `fpa_vllm.py` (ours):**
1. Progress: one timestamped line per chunk of 8 branches with tok/s and elapsed. YES.
2. Checkpoints: each branch's answers + continuation text appended to `row080_branches.jsonl`
   as it completes; `--resume` skips done branches. YES.
3. Memory: vLLM pre-allocates KV at startup (`gpu_memory_utilization 0.90`) and schedules
   within it; no growth pattern like HF generate. Peak = fixed. YES, bounded.
4. Kill at 80%: 80% of branches on disk, resumable. YES.

**Cost line (written before launch).**
- Smoke: 2 branches × S=5 → ≤4k tokens. Purpose: stack check + first real tok/s. ~2 min.
- Main, plan A: `--only-forks --stride 4 --S 50`. QWEN-01 saw 280 branches at 108 positions
  at stride 1; stride 4 → ~70 branches → 70 × 50 × ≤400 = **≤1.4M tokens**. At a vLLM rate of
  ~2–5k tok/s (to be measured in the smoke, not assumed): **5–12 min**.
- Main, plan B (only if A looks interesting): S=200 at the top positions, ~10 branches ×
  200 × 400 = 0.8M tokens, ~5 min.
- Budget cap for the session: **$2** (≈4 h A40). Setup ~15 min ≈ $0.12.

**Dry-run on laptop** (fake vLLM with the real Qwen tokenizer): see next entry.
- 17:07 IST dry-run of `fpa_vllm.py` on the laptop with a fake vLLM (real Qwen3.5 tokenizer,
  rigged generator): 7 branches at 4 positions → 3 chunks, progress lines with tok/s,
  `row080_branches.jsonl` 7 lines with counts and fallback counts, `row080_base.json` with
  topk; `--resume` skipped 7 and ran 2 more (9 lines). Enumeration, extraction, checkpoint
  and resume verified. Fake model deleted with the scratchpad; no real weights touched.

**Session log (UTC).**
- 11:38 pod `57553tl3yjm4qx` "forking-qwen02" created: A40 48 GB SECURE, CA-MTL-1, host
  CUDA 13.0 (driver; cu128 wheels are forward-compatible), image
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 80 GB disk. Balance before: $54.26.
- 11:43 direct SSH 63.141.33.115:22019. Files copied (fpa_vllm.py, row080.json, setup,
  launcher). `pod_setup_vllm.sh` launched → /root/setup.log. Watchdog started (8-min alarm).
- 11:58 setup FAILED at the generate test: vllm 0.29.0 pinned torch 2.13.0 (CUDA 13.0) from
  PyPI, while my `--extra-index-url .../cu128` resolved torchaudio 2.11.0+cu128; transformers'
  audio_utils raises on the CUDA mismatch at import. Fix: `pip install torchaudio==2.13.0`
  from PyPI (matches torch), and the setup script now installs vllm with NO extra index.
  Generate test relaunched. Lesson for the recipe: with vLLM, let its wheel decide torch.
- 12:06 torchaudio==2.13.0 does not exist; vllm 0.29.0 REQUIRES torchaudio, so uninstall is
  not an option. Fix that worked: `pip install --force-reinstall --no-deps torchaudio==2.11.0`
  from plain PyPI (the +cu128 build was the problem, not the version). `import vllm` OK.
  Two SSH round-trips were lost because my `pkill -f`/`pgrep -f` patterns matched the SSH
  session's own command line and killed it — do not kill by pattern over ssh; use pids.
- 12:08 generate test relaunched, chained into the 2-branch smoke (`smoke`: S=5, S0=5,
  only-forks, chunk 2) so no idle gap between them.
- 12:20 generate test FAILED at engine init: "An attempt has been made to start a new process
  before the current process has finished its bootstrapping phase" — vLLM v1 spawns the
  EngineCore with multiprocessing `spawn`; my gen_test.py and fpa_vllm.py ran `LLM(...)` at
  module top level with no `if __name__ == "__main__"` guard. The official
  `examples/basic/offline_inference/basic.py` (read this morning) has `def main()` +
  the guard; I read it and did not carry it over. Claude's miss. Fix: wrap both in main().
  Model never downloaded yet (hf cache 22 MB), so no wasted download.
- 11:53 generate test OK through vLLM ("Hi there! How can", 395 s incl. 19 GB download).
  vLLM startup: weights 18.1 GB, KV reserved to 40 GB (0.9), CUDA graphs 16 s.
- 11:57 **smoke (2 branches, S=5) passed end to end**: base path, branches, sampling, regex +
  logit fallback, checkpoint jsonl, progress lines. Files pulled to `runs/2026-09-15_qwen02/smoke/`.
  **FINDING that changes the design:** Qwen3.5-9B (thinking off) writes long LaTeX/markdown
  answers. Base path hit the 400-token cap with `finish=length`, no "answer is" reached.
  All 10 smoke continuations also hit 400 with no answer → 8/10 went to the logit fallback.
  The paper's Llama budgets (400/400) do not fit this model. This is the DeepSeek "Other"
  problem in a different costume, caught at S=5 instead of S=200.
  Throughput at 10 concurrent sequences: 275 tok/s (~27 tok/s per stream) — meaningless for
  the plan; vLLM's rate scales with concurrency, so smoke2 measures it at 200 sequences.
  Side note: at t=0 the forced logit read says B (the correct answer) 4/5 — Llama said C.
- 12:01 smoke2 launched: 4 branches × S=50 (200 sequences), base_max 800, cont_max 600.
  Purpose: (a) does the base path finish at 800? (b) real tok/s at 200 concurrent.
- 12:01 **smoke2 (4 branches × S=50 = 200 sequences): 1,555 tok/s** at 200 concurrent
  (vs 275 at 10). 0.12M tokens in 77 s. → the 1.4M-token main run ≈ 15 min. Throughput is
  no longer the constraint.
  **But the budget problem is worse than smoke1 showed:** base path hit 800 tokens, still
  `finish=length`, still no "answer is" (it is mid-way through evaluating options one by one);
  all 200 continuations hit 600. Fallback used on ~50%. And the other ~50% that the regex
  "resolved" are suspect: their pattern `\boption\s+\(?([ABCD])` matches Qwen's running
  commentary ("Option C uses Rxzy...") and takes the LAST match — that is not a final answer.
  Checking which patterns fired (next). Probe launched: greedy base path to 3000 tokens,
  thinking off AND on, to learn how long Qwen actually takes to answer this item.
- 12:12 smoke2 regex audit: **97/97 regex "answers" came from pattern 4 (`option X` mention)**
  — the last option Qwen happened to be discussing when truncated. Not verdicts. Extractor
  patched (ours, recorded): a regex answer counts only if an explicit "answer is / answer:"
  pattern matched; option-mentions go to the logit fallback; `regex_kind` saved per
  continuation. Dry-run passes.
- 12:14 **base-length probe:** thinking OFF finishes at **1,112 tokens**, `finish=stop`,
  ends "The answer is (B)" (correct). Thinking ON: 3,000 tokens and still inside <think>.
  → thinking off confirmed; budgets must be ~3× the paper's Llama setting.
- 12:20 **MAIN RUN launched** (AG chose stride 4): `--stride 4 --S 50 --S0 50 --only-forks
  --chunk 8 --base-max-tokens 1500 --cont-max-tokens 1500 --max-model-len 4096`, pid 12436,
  log /root/out/main.log, checkpoints /root/out/main/row080_branches.jsonl. Cost line: ~195
  branches, ~6.4M tokens, ~68 min, ~$0.56 at the measured 1,555 tok/s. Live monitor on the
  log; every [fpa] progress line reaches the session. Balance at launch $53.94.
- 13:43 **MAIN RUN DONE: 206 branches, 7.03M tokens, 78.6 min generation.** Bundle
  `runs/2026-09-15_qwen02/qwen02_all.tgz` (md5 1142d0f1…, verified both ends), extracted to
  `runs/2026-09-15_qwen02/out/main/` (`row080_base.json`, `row080_branches.jsonl` 206 lines).
- 13:45 pod `57553tl3yjm4qx` terminated (204), `list-pods` → []. Monitors + watchdog stopped.
  **Balance $53.23** (was $54.26) → session cost **$1.03** (2 h 07 min A40 incl. setup, two
  smokes, one probe, main run).

**RESULT — QWEN-02.**
| | |
|---|---|
| Continuations | 10,300 (206 branches × 50) |
| Finished with a stop | 95.6% |
| Explicit "The answer is (X)" | 94.6%; logit fallback 5.4%; "Other" 0 |
| Answers | B 10,276 · C 14 · A 9 · D 1 (B is correct) |
| Alternative-token branches | 125 at 81 positions |
| Largest swap effect (TVD base vs alt) | **0.040**; none ≥0.1 |
| S=50 noise floor for a 98%-B branch | 95th pct 0.060, 99th 0.080 |
| Base-branch B share by 200-token window | 0.993 · 0.997 · 1.000 · 1.000 · 1.000 · 1.000 |

**Verdict.** On row 80, Qwen3.5-9B (thinking off) shows **no per-token fork at stride 4 and
S=50**. Every one of 125 alternative tokens that cleared 5% leaves the final answer at B
≥96%; the largest swap effect (0.040) is below the sampling-noise floor. Llama-3-8B on the
same item flipped C↔A on a bracket (TVD 0.96). **Hypothesis sentence: NOT met.**

**What this does and does not say.** Does: on an item the newer model is confident and
correct on, forcing any probable alternative token does not move it. Does not: that Qwen
never forks — this item is easy for it (t=0 already 96% B). Three open readings: (a) forks
need an item the model is uncertain on; (b) forks below the 5% cutoff; (c) MMLU contamination
makes tinyMMLU the wrong probe for this model. (a) and (c) are testable for cents: greedy
pass over all 100 tinyMMLU items, thinking off, score them, pick the wrong/uncertain ones.

**Method notes (for the write-up).** vLLM 0.29.0 replaced HF generate: 1,500–1,700 tok/s
at 200 concurrent vs an unfinished 2 h at batch 200 yesterday. Budgets 1,500/1,500 (Qwen's
greedy answer is 1,111 tokens; the paper's 400 truncates it). Their regex's `option X`
pattern was disabled for this model (fires on running commentary — 97/97 in smoke2);
`regex_kind` saved per continuation. Everything else follows the paper: p≥0.05 branches
incl. greedy, T=1, top_p=1, all tokens, forced logit read on misses.

**Independent check owed (AG).** Pick one branch line from `row080_branches.jsonl`, count
the letters in its 50 `conts[*].text` endings by hand, compare to `counts`.

---

## SCREEN-01 · 2026-09-15 · PRE-LAUNCH CHECKLIST — proposed, not launched

**Question.** On which tinyMMLU items is Qwen3.5-9B (thinking off) uncertain or wrong? Per
item: greedy answer + o_0 from S0=10 free samples. Two outputs: accuracy on the 100 (AG's
contamination question) and a ranking by o_0 spread (where a fork can exist).
**Not a forking test.** Item selection, which both papers skipped.

**Script.** `forking/pipeline/screen_t0.py` (ours). Same extraction rule as fpa_vllm.py.
Dry-run on the fake vLLM: 3 rows, checkpoint 3 lines, progress line per row. Questions file
`forking/questions/tinymmlu_100.json` built from the 100 Llama stores (rows 0–99, letters
D28/C27/B26/A19), so row numbering matches the paper exactly.

**Four answers.** (1) progress: one line per question with greedy verdict, o_0, seconds —
YES. (2) checkpoint: `screen.jsonl` appended per question, `--resume` — YES. (3) memory:
vLLM fixed at startup, 11 sequences per item — bounded. (4) kill at 80%: 80 rows on disk — YES.

**Cost line.** 100 × 11 answers × ~1,100 tokens ≈ 1.2M tokens → ~13 min at ~1,500 tok/s.
Setup + download ~15 min (pod was terminated). **≈ 30 min, ≈ $0.25.** Cap $0.60.
Then, optionally, one full stride-4 run on the most split item ≈ $0.60 (separate go).
- 16:43 UTC **GO (AG).** Pod `0e90vqlox4inty` "forking-screen01", A40 SECURE, EU-SE-1,
  same image. Balance before: $53.23. Setup runs the fixed `pod_setup_vllm.sh` (no extra
  index; torchaudio guard), then `run_screen.sh main --S0 10`.
- 16:50 setup's generate test failed: `FileNotFoundError: '/root/<stdin>'` — the test was a
  `python - <<PY` heredoc; vLLM's spawned EngineCore re-imports the main module and stdin has
  no file. Same family as yesterday's `__main__` guard error, second costume. Stack itself is
  correct (torch 2.13.0, torchaudio 2.11.0, vllm 0.29.0). Setup script fixed to write a guarded
  file and run it. Guarded test relaunched, chained into `run_screen.sh main --S0 10`.
- 16:57 generate test OK (441 s incl. download). **SCREEN-01 launched**, pid 3824, S0=10,
  max_tokens 1500, log /root/out/main.log, checkpoint /root/out/main/screen.jsonl. Live
  monitor on per-question lines. Watchdog running.
- 17:01 first two rows in: row 0 D ok (906 tok, o_0 9D/1A, fallback 7/11), row 1 C ok
  (535 tok, o_0 10C). **But 79 s/question** — my script ran questions one at a time (11
  sequences in flight); vLLM needs hundreds. Cost-line estimate was 5× too optimistic for that
  reason, not the model. Patched to chunks of 10 questions (110 sequences per call, checkpoint
  per chunk), dry-run + resume verified on the laptop, serial run killed BY PID (rows 0–1 kept),
  relaunched with `--resume`, pid 5698, log main2.log. Also to inspect after: row 0's 7/11
  fallbacks — regex may be missing a Qwen phrasing.
- 17:02 relaunch FAILED: "Free memory 4.46/44.42 GiB" — killing the serial run's parent pid
  left vLLM's spawned EngineCore (pid 4229) alive holding 40 GB. Killed it via
  `nvidia-smi --query-compute-apps=pid` (by GPU handle, not by name pattern). Relaunched
  again with --resume → main3.log. **Lesson for the recipe: to stop a vLLM job, kill the
  parent AND the pids nvidia-smi lists, then confirm memory.used == 0 before relaunch.**
- 17:21 **SCREEN-01 DONE: 100 rows in 17.2 min** (batched). Bundle
  `runs/2026-09-15_screen01/screen01_all.tgz` (md5 ea02a325…, verified), extracted to
  `runs/2026-09-15_screen01/out/main/screen.jsonl` (100 lines; rows 0–1 from the serial run
  lack `sampled_finish`, harmless).
- 17:23 pod `0e90vqlox4inty` terminated (204), `list-pods` → []. **Balance $52.89** (was
  $53.23) → **session cost $0.34** (~40 min A40). Monitors + watchdog stopped.

**RESULT — SCREEN-01 (Qwen3.5-9B, thinking off, tinyMMLU test, S0=10, T=1, cap 1500).**
- **Greedy accuracy 86/100.** AG's contamination worry: not saturated — 14 wrong.
- **17 items have a top answer ≤80% of 10 samples** (torn at t=0); 7 of those are also wrong
  greedily. **7 more are confidently wrong** (≥90% on the wrong letter: rows 8, 9, 18, 53,
  62, 74, 77) — a different phenomenon from torn, worth its own look later.
- Most torn: **row 91** (WRONG; A4/B3/C3), **row 41** (ok; A5/C5, 0 truncations, 480-tok
  greedy — the cleanest candidate), 42/49/69 (50% but 6–9 of 10 truncated at 1500 → spread
  partly an artefact; need cap ≥2500 if used), 60 (WRONG; A5/B3/C2), 32/55/56 (60/40).
- Row 80 (today's full run): 10/10 B — consistent with "no fork" there. Row 39 (paper's
  S=1000 dev item): 10/10 B, correct — Llama was three-way split on it.
- Truncation: math-heavy items run past 1,500 at T=1 (rows 0, 42, 49, 69, 75, 97); for any
  full run on those, budget 2,500.

**Recommended next full run (AG's call):** row 41 first — 50/50 at t=0, clean finishes,
short answer (480 tok → cheap, ~$0.25 at stride 4/S=50), correct greedily so a fork would be
"right answer → wrong answer". Then row 91 (torn AND wrong, 3-way). Row 60 third.

---

## QWEN-03 · 2026-09-15 · row 41 — full stride-4 run on a TORN item

**Question.** SCREEN-01 found row 41 split 5 A / 5 C at t=0 (greedy A, correct, 480 tokens,
no truncation). Does forcing a probable alternative token at some position flip Qwen3.5-9B
between A and C? Same falsifiable sentence as QWEN-02 (TVD ≥0.5 on some branch pair), now
on an item where the answer distribution is not already collapsed.
**Settings.** Identical to QWEN-02: stride 4, S=50, S0=50, budgets 1500/1500, model_len
4096, only-forks, chunk 8. Same script (`fpa_vllm.py`), same extraction rule.
**Four answers.** Unchanged from QWEN-02 (progress per chunk, per-branch jsonl, bounded
memory, resumable).
**Cost line.** Base ~480 tok → ~120 stride-4 positions, maybe ~35 forking → ~90 branches ×
50 × ~300 tok ≈ 1.4M tokens ≈ 10–15 min; + setup/download ~12 min. **≈ 25 min, ≈ $0.22.**
**Opening balance: $52.89.**
- 17:36 UTC **GO (AG).** Pod `an7l3guifqyciz` "forking-qwen03-row41", A40 SECURE, EU-SE-1,
  host CUDA 12.8. Row 41 = moral_scenarios (two scenarios, Wrong/Not wrong; correct A).
  Chain: SSH → copy → watchdog → setup (fixed script, guarded generate test) → run.
- 17:41 setup FAILED on `an7l3guifqyciz`: **host driver 570.195 (CUDA 12.8)**; vllm 0.29.0's
  torch 2.13.0+cu130 → "The NVIDIA driver on your system is too old (found version 12080)".
  Today's two working pods were driver 580 / CUDA 13.0 — I did not pin it. Pod terminated
  (~5 min, ~$0.04). New pod created with `gpu.allowedCudaVersions=["13.0"]`; setup script
  header now records the host requirement.
- 17:41 pod `ob972k9bimdbcw` "forking-qwen03-row41b", A40 SECURE, CA-MTL-1, host CUDA 13.0
  (pinned). Chain relaunched. Opening balance for QWEN-03 remains $52.89; the failed pod's
  ~$0.04 is charged to this experiment.
- 17:48 generate test OK (405 s incl. download). **QWEN-03 run launched**, pid 3150,
  stride 4, S=50, budgets 1500, log /root/out/main.log, checkpoints /root/out/main/. Live
  monitor attached.
- 18:40 **QWEN-03 DONE: 159 branches at 55 positions, 3.59M tokens, 45.2 min.** Bundle
  `runs/2026-09-15_qwen03_row41/qwen03_main.tgz` (md5 4fbe907d…, verified), extracted to
  `out/main/` (base json, 159-line jsonl, log). Pod kept up pending S=200 decision.

**RESULT — QWEN-03 (row 41, moral_scenarios, correct A).**
| | |
|---|---|
| Continuations | 7,950; 91.4% stop; 98.6% explicit "answer is"; 1.4% fallback; 0 Other |
| Answers | A 6,655 · C 1,276 · B 15 · D 4 |
| o_0 (t=0, S0=50) | A 30 / C 20 |
| Alternatives tested | 104 at 55 positions (462-token answer, stride 4) |
| **p<0.01 swaps** | **6** (chance ≈1); p<0.05: 10 |
| **Verdict flips** (modal answer changes) | **3**: t=164, 76, 52 |
| Largest | t=164 " It"→" However": A42/C8 → A23/C27, TVD 0.38, p=0.0003 |
| Base-branch A share by 80-token window | 0.47 · 0.62 · 0.90 · 0.94 · 0.96 · 0.98 |

**Verdict.** Qwen3.5-9B DOES fork on an item it is torn on. Three positions flip the modal
answer between A and C at S=50; the strongest is a discourse token ("However" vs "It" at
t=164) — the 2024 paper's "unexpected forking tokens" class. The pooled curve shows the
answer being decided gradually between t≈80 and t≈200 (A share 0.47→0.90), with these
sharp per-token effects inside that window. **Hypothesis sentence (TVD ≥0.5): not met at
S=50 (max 0.38); the weaker per-token claim (beyond noise, verdict-flipping) IS met.**
Contrast with QWEN-02 (row 80): 0 of 125 vs 6 of 104. The screen's o_0 predicted which.

**Caveats.** S=50 per branch; the 6 p-values are per-swap, no multiple-comparison
correction (6 vs ~1 expected is the honest comparison). Stride 4 → true forks between
sampled positions unseen. One item. `swap_effects_row041.json` holds all 104.

**Proposed S=200 confirmation pass (same pod, model loaded):** positions 52, 76, 136, 164,
184 — every branch there (≈13), S=200 → ≈2,600 continuations × ~300 tok ≈ 0.8M tokens ≈
8 min ≈ $0.07. Separate output dir `out/s200`. Awaiting AG's go.
- 18:46 **GO (AG) for the S=200 pass**: `--positions 52,76,136,164,184 --S 200 --S0 200`,
  out /root/out/s200, log s200.log. Same base path (greedy, deterministic), so positions map 1:1.
- 18:48 first S=200 launch died at argparse: `--positions` existed only in the HF adapter,
  not in fpa_vllm.py. Added (filters enumerated branches to the listed positions), dry-run
  on fake vLLM OK, relaunched. ~3 min idle pod.
- 19:14 **S=200 pass DONE: 17 branches at 5 positions, 3,400 continuations, 2.22M tokens,
  23.1 min** (my 8-min line was low: 52 and 76 had more candidates than assumed, and 800
  sequences/chunk). Bundle `qwen03_s200.tgz` (md5 00c94195…, verified) → `out/s200/`.
- 19:15 pod `ob972k9bimdbcw` terminated (204), `list-pods` → []. **Balance $52.10** (opening
  $52.89) → **QWEN-03 cost $0.79** incl. the bad-host pod (~$0.04) and the flag miss (~$0.03).

**S=200 CONFIRMATION (row 41).** All six S=50 hits stay significant at S=200 (p ≤ 0.001);
effect sizes shrink toward the truth as noise halves; 2 of the 3 verdict flips survive.
| t | base → alt | S=50 TVD (p) | S=200 TVD (p) | S=200 tallies | flip @200 |
|---|---|---|---|---|---|
| 164 | It → However | 0.38 (.0003) | **0.35 (<.0001)** | A175/C25 → A105/C92 | no — 53/47, a coin flip |
| 52 | Offering → The | 0.28 (.007) | **0.27 (<.0001)** | C144/A56 → A109/C91 | **yes** |
| 76 | is → can | 0.30 (.004) | 0.17 (.001) | C110/A90 → A123/C76 | **yes** |
| 136 | at → " | 0.26 (.004) | 0.16 (<.0001) | A140/C59 → A172/C28 | no |
| 184 | this → under | 0.24 (.004) | 0.12 (.001) | A183/C17 → A158/C38 | no |
| 164 | It → Therefore | 0.16 (.006) | 0.09 (.0003) | A175/C25 → A194/C6 | no |
Also newly significant at S=200 only at p<0.05: 76 → in (0.12, p=.02), 76 → to (0.12, p=.03).
Non-hits stayed non-hits (136 → a, 76 → itself, 184 → it, 136 → rude).

**Reading.** t=164 is the strongest effect in the project on a current model: after "It"
the answer is A 88%; after "However" it is 53/47. That is not a verdict flip but a return
to full uncertainty from near-certainty, triggered by one discourse token at p=0.36.
t=52 is a clean verdict flip that holds at S=200: "Offering" (restating scenario 1 as an
action) → C 72%; "The" → A 55%. **Hypothesis sentence (TVD ≥0.5): still not met (max 0.35
at S=200). The per-token claim — probable alternative tokens move the final-answer
distribution far beyond sampling noise, and can flip the verdict — is met on this item.**

**Independent check owed (AG).** `out/s200/row041_branches.jsonl`, the two lines at t=164:
count A/C in the 200 `conts[*].text` endings by hand; expect 175/25 and 105/92.

---

## BLOG-01 — first post written, rejected by LessWrong, published on AG's own site
**2026-09-20 → 2026-09-22. Zero GPU spend.**

**Live:** https://adithyag73.github.io/first_principles/forking-tokens/

**What it says.** SWAP-01 + QWEN-02 + SCREEN-01 + QWEN-03 as one claim: on a question
Qwen3.5-9B is confident about, no alternative token moves the final answer; on a question it
is torn about, six do, and two flip the verdict. Two figures, both measured. Text is AG's
dictation; Claude fixed spelling, tense and numbers and did not rewrite the voice.

**Process rule that held.** AG dictated; Claude proofread. Every claim about the two papers
was checked against the live arXiv HTML before publishing (`blog01/PAPER_CHECK.md`, 6/6).
One of Claude's own facts was wrong and was caught by that check: row 80 is **one of** the
clearest forks in the Llama data, not the clearest — row 50 t=236 "false"→"true" is larger
(0.990 vs 0.960). Row 80's distinction is that it is the largest among positions where the
*pooled* curve is flat. Corrected in `FACTS_FOR_AG.md` and in the post.

**LessWrong: rejected, 2026-09-20.** The message is a template — confirmed by reading
lesswrong.com/moderation, where 8,108 rejected posts carry the same text. The one
substantive line in it is right and is Claude's error, not AG's: the post never says why
forking matters. Claude proposed the four-section structure and left the motivation section
out of it. **Still not addressed** — it needs AG's own words and he has not dictated them.

**Published instead on AG's own GitHub Pages site**, third article after flash-attention and
free-normalization. House layout, TOC and CSS copied from the existing two; figures served
from `forking-tokens/figures/`, not hotlinked. Commit `442e045`.

**Infra note for next time.** The machine's active `gh` account is `IamAGP` (the forking-fast
fork) and it gets 403 on `ADITHYAG73/first_principles`. Push with
`gh auth switch --user ADITHYAG73`, then switch back. Both are in the keyring.

**Editor lesson (LessWrong, but general).** Their editor pastes markdown literally — `**bold**`
survived as asterisks, wrapped lines became separate paragraphs, and list options were
dropped. Retyping as plain paragraphs and verifying character-for-character against
`forking/questions/tinymmlu_100.json` was the only reliable path. Images inserted twice on
both attempts; duplicates deleted manually.

**Still owed.** AG's hand check at t=164 (above), and the motivation paragraph.
