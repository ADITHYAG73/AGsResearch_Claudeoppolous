# LATENT-CUBE — journal

## Question (AG, 2026-10-29)
When Qwen3.5-4B is asked to *imagine* the second Taylor term of sin x and never writes it, is −x³/6
(or x³) present in its hidden state — and is it present for the same reason as in other x³ prompts?
AG's own caveat: a shared representation alone does not separate "thinking" from "memorising" (he
memorised the series too). The composition prompts (sin(x/2) → −x³/48, sin x·eˣ, sin²x) are the test of
combining memorised pieces into results that are rarely written down.

Prior work to read AFTER design (AG's order): Anthropic "On the Biology of a Large Language Model"
(Dallas→Texas→Austin latent two-hop); Yang et al. arXiv 2402.16837 (latent multi-hop, words).

## CAPTURE-01 design (decided with AG 2026-10-29)
- Model Qwen/Qwen3.5-4B @ 851bf6e (text-only `Qwen3_5ForCausalLM`), bf16, eager attention, **thinking off**
  (one forward pass, no scratchpad), **batch 1** (no padding — TRIG-01 showed the residual moves with padding).
- Prompts: `gen_prompts.py` → `prompts.jsonl`, 201 = 16 functions × terms 1–4 × templates
  {write, imagine, hold} + 9 free-form latent/control prompts. Ground truth exact (Fractions).
  "k-th term" = k-th non-zero Maclaurin term (AG's reading). 11 items per template have power 3.
- Readouts (AG: record all three): write = about to say it; imagine = end of instruction;
  hold = while answering "capital of France".
- Saved per prompt, ALL tokens (prompt + greedy reply ≤ 48 tokens, teacher-forced in one pass):
  R0 embedding, R1 each layer's output, R2 mixer output, R4 MLP neurons, R5 MLP output, full logits,
  attention weights of the 8 full-attention layers (fp16); DeltaNet memory S at 3 positions
  (instr_end, prompt_end, reply_end; fp32). Plus meta json (chat text, ids, tokens, reply, positions,
  sanity numbers). AG: over-collect; storage is S3.
- Sanity checks per prompt: (1) teacher-forced argmax == generated reply; (2) final norm(R1[31])·Eᵀ ==
  logits; (3) R1[l] = R1[l−1] + R2[l] + R5[l].
- Dry run on 2 prompts with `--debug` FIRST (AG's instruction), inspect chat template, positions,
  shapes, the cache structure for S, and the three checks; only then all 201.

## Rule-7 preflight (written before the pod exists), from reading capture.py
1. **Progress line per unit of work?** Yes: one timestamped line per prompt with reply, the three
   sanity numbers, file size and ETA; `MODEL READY` and `CAPTURE DONE` markers.
2. **Incremental results?** Yes: one .safetensors + .json per prompt, written as soon as the prompt
   finishes; the .json is written last, so its presence = prompt complete; reruns skip completed prompts.
   [CORRECTED 2026-11-24: superseded during the run. Final capture.py writes each variant's .safetensors then
   .json, and `<id>.done` after ALL variants of a prompt; resume keys on `.done` (capture.py lines 127, 149).]
   Results sync to S3 in the background every 5 min.
3. **Memory trajectory?** Flat. Weights ~9.3 GB; per prompt the largest live tensors are logits
   [~90 × 248,320] fp32 ≈ 90 MB and eager attention for T ≤ ~100 (tiny); `cap` is cleared each prompt.
   Generation KV/DeltaNet state is per prompt and freed. Peak well under 20 GB on an 80 GB card.
4. **Killed at 80%?** ~80% of prompts are on local disk and, up to the last 5-minute sync, in S3; a
   rerun resumes from the first missing .json.
- **Disk:** ~260 MB per prompt (S alone is 150 MB) × 201 ≈ 52 GB + model 9 GB → container disk 150 GB.
- **Cost line:** per prompt ≈ generate ≤48 tokens (~2 s) + 1 full pass + 3 prefix passes + write
  260 MB (~3 s) ≈ 6 s → 201 × 6 s ≈ 20 min; setup + download ~10 min; S3 upload 52 GB at ~100 MB/s
  ≈ 9 min (overlaps). ≈ 45 min × $1.79/hr (A100 SXM) ≈ **$1.35**; treat as a floor (TRIG estimates ran
  1.3–1.5× over).
- Opening balance 2026-10-29 19:17 UTC: **$127.92**, spend rate $1.80/hr = the gemma4devs pod
  (`gemma4-exp056-eval`, not mine).

## CAPTURE-01 run log (2026-10-29, UTC)
- 19:18 pod `ww8r6g9w3o2bps` (latentcube-capture01), A100 SXM 80GB, $1.79/hr, image interp-pod:torch28-tf5170
  (torch 2.8.0+cu128, transformers 5.17.0, fla 0.5.2, causal_conv1d). Watchdog `wd_lc` running.
- S3: IAM user `latentcube-pod-uploader` (prefix `latent_cube/*` only) created after AG approved; the first attempt
  was blocked by the auto-mode classifier, the second failed on a ',' in a tag value (same as FT-01). DELETE AFTER RUN.
- Dry run 1 (2 prompts): chat template correct (thinking off), positions right, cache path for the DeltaNet state is
  `cache.layers[i].recurrent_states[0]` (read from a debug print, not assumed). Checks: logits recompute diff 0;
  residual-sum diff ≤ 0.5% of |R1| (bf16 rounding); 1 argmax mismatch = near-tie 26.125 vs 26.0.
- **Finding: the model does not obey "imagine / do not write it".** It says the term (and still answers Paris).
  → AG decision: capture each imagine/hold prompt twice — `free` (model's own reply) and `forced` (a reply written
  for it that does NOT contain the term: "Okay." / "The capital of France is Paris.").
- Early behaviour note (1 prompt, not a result): "write" sin x 2nd term → model answered −x²/2 (wrong), while in the
  hold prompt it wrote −x³/6 (right).
- **Claude errors, journaled:** (1) told AG the model "never answered Paris" from a 60-character log preview — the
  full reply did answer Paris → global rule 10 "never truncate" added (~/.claude/CLAUDE.md + claude.ai Preferences).
  (2) a launch command AG rejected had already executed on the pod (old script version) → an accidental full run
  (201 prompts, 41.9 GB) ran 19:24–19:32 alongside; found from `ps`, stopped, its files moved to `v1_superseded/`
  (not deleted). (3) mis-estimated 1.9 GB/prompt by dividing both runs' disk use by one run's count; real 0.45 GB.
  (4) `pgrep -f s3_sync.py` over ssh matched its own command line and killed the ssh shell (known pitfall, repeated).
- 19:37 intended run started; resumed 19:40:38 after the cleanup. Rate ≈ 3.5 s/prompt. S3 upload ≈ 83 MB/s
  (26.8 GB landed 19:35:38→19:41:09, from the bucket listing).
- 19:53:37 **disk full** at prompt 197 (lat|square_area, forced variant): out/ had grown to 97 GB (free replies longer
  than estimated) + 42 GB superseded run + 9 GB model = 150 GB. Claude's watcher only read the LAST log line and
  missed the crash line above it; AG spotted 107% disk in the RunPod console. Nothing finished was lost.
- Recovery (20:05–20:08): stopped run + uploaders by PID; `verify_free.py` compared pod vs S3 by name + byte size
  (527 verified, 0 mismatches) and deleted ONLY verified .safetensors (56.5 GB freed); resumed → prompts 197–201 done
  20:08:27, capture exit 0. Superseded-run files already in S3 (246, sizes match) marked synced to avoid re-upload.
- Upload done 20:32:16; final verify every folder: 0 missing, 0 size mismatches. Prompt-by-prompt check against S3:
  201 prompts, 877 required files, 0 missing. **S3 total `latent_cube/capture01/` = 140.6 GB:**
  `out/` (new format: `<id>__free|forced.{safetensors,json}` + `<id>.done`; files WITHOUT the __free/__forced suffix
  are the superseded first run), `dryruns/dry{,2,3}/`, `meta/` (code, prompts, all logs incl. finish_final.log,
  pip_freeze, nvidia_smi, df).
- 20:34 pod terminated, get-pod 404, list-pods empty. Watchdog stopped. IAM user `latentcube-pod-uploader` deleted
  (GetUser → NoSuchEntity); local key copy deleted.
- **Cost:** pod 19:18:39 → ~20:34 ≈ 1 h 16 min × $1.79–1.81/hr ≈ **$2.29**. Balance $127.92 → $125.68 at 20:34
  (PROVISIONAL: the API still showed spend $1.81/hr seconds after terminate; RunPod posts late — re-read).
  Pre-launch estimate was $1.35 floor → ~1.7× over (disk-full recovery + upload slower than the first measurement).

## DASHBOARD-01 (Codex build, 2026-10-30)
- Opening AWS figure for the S3 reads: month-to-date net **$0.00** for 2026-10 at 23:20 IST (credits offset charges;
  Cost Explorer lags up to ~24 h). Budget: 15 GB of S3 reads, about $2 at most.
- Brief: `dashboard/BRIEF.md`. Credentials: 12-h federation token, GetObject/ListBucket on `latent_cube/*` only (tested).
- Done 2026-10-31 00:44 IST. 13.46 GB S3 read (limit 15). Site 54.6 MB, 438 files. Logit gate max diff 0.0649, RMSE 0.00566.
  Sampling, disclosed on the page: norm maps first 128 tokens + 3 readouts (24,811/33,551 positions; 65 captures affected);
  lens/MLP/attention all tokens only on the opening capture, 3 readouts elsewhere; cross-prompt neurons 64 of 9,216.
  Claude's independent check: 132 comparisons, 0 mismatches. Published: https://claude.ai/artifact/Vhkj2EQGXtyufVUcUbCKYe

## JLENS-01 — J-lens on the forced replies (opened 2026-11-24, cloud session)
**Question (AG's, unchanged):** at the tokens of a FORCED reply that never contains the term (e.g. "The capital of France
is Paris."), is the hidden Taylor term (sin x, 2nd term: −x³/6) readable with Anthropic's Jacobian lens, and not for the
control (cos x, 2nd term: −x²/2)?
**Why the J-lens:** the logit lens only reads the next token ("The"), so it cannot see a term held for later (AG's
objection, 2026-11-02). The J-lens transports a layer's state through J_ℓ = E[∂h_final/∂h_ℓ] averaged over current AND
later positions, then unembeds. Paper: transformer-circuits.pub/2026/workspace, §"The J-space is subject to directed
modulation" — same protocol as ours (instruction + teacher-forced unrelated carrier sentence, lens read inside it).
**Code:** Anthropic's official github.com/anthropics/jacobian-lens (Apache-2.0, commit 581d398, 2026-07-01) — NOT the
community mirror. We fit our own lens on Qwen/Qwen3.5-4B @ 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a.

**AG's prediction, recorded 2026-11-24 BEFORE any lens readout exists:** in the readouts at forced-reply tokens of the
sin x prompts he expects tokens like " cube" / "³", "sin", "x" and "6" to appear. He said he will keep an open mind and
not re-fit the prediction after seeing data.
**Hit definition: NOT YET DECIDED (AG's call).** Tokenisation checked 2026-11-24: "−x³/6" → "−" "x" "³" "/" "6"
(ids 32160, 87, 41776, 14, 21); single-token candidates " cube" 22997, " six" 4590, " three" 2250, " third" 4688,
" Taylor" 15957. cos control's term "−x²/2" should give "²"/"2" instead.
**Caveat AG accepted:** a forced reply is text we fed the model, not its own behaviour; the paper made the same choice.
**Plan:** (1) fit the lens (GPU); (2) positive control — the paper's 3²−2 directed-modulation prompts from the repo's
data/experiments/directed-modulation.json on our model; if "seven" never appears, stop; (3) apply the lens to CAPTURE-01
R1 at forced-reply tokens on CPU, sin x vs cos x. AG approved provisioning the GPU on 2026-11-24 (~16:30 IST), with
"back up data and results to S3 and delete them".

### JLENS-01 change of plan + rule-7 preflight (written 2026-11-24 ~11:20 UTC, BEFORE the pod exists)
- **No fitting needed.** Anthropic's own walkthrough (jacobian-lens/walkthrough.ipynb) uses Qwen/Qwen3.5-4B with the lens
  `neuronpedia/jacobian-lens`, revision `qwen-n1000`, file `qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt`
  (406 MB = 31 layers × 2560² fp16; branch commit 16a01f3, 2026-06-20; its config.yaml: fit 2026-06-13 on a B200 with
  Anthropic's fit_lens, wikitext-103, max_seq_len 128, dim_batch 64, bf16). The model id was unpinned, but Qwen3.5-4B's last
  commit is our 851bf6e (2026-03-02), so the lens was fit on our weights. The community mirror is not used.
- **Code:** `latent_cube/jlens01/run.py`, three parts — `sanity` (unembed check; the walkthrough's boot prompt J vs logit lens;
  R1 of the forced sin-x hold capture saved for a CPU comparison with CAPTURE-01), `dm` (positive control: the paper's
  directed-modulation prompts; 24 math + 22 category targets × 25 phrasings incl. no-instruction × 3 carriers = 3,450 trials;
  the user-turn wording "Write the following sentence. {instruction}\nSentence: {carrier}" is OURS, the repo gives only
  phrasings + carriers), `cube` (all 338 CAPTURE-01 captures from their exact token ids — `jlens01/captures.json`, pulled from
  the dashboard data, 0 length mismatches — J-lens and logit-lens ranks of 36 tracked single-token ids at every layer ×
  position, J-lens top-20, and the J-transported states fp16 for any later hit definition on CPU).
- **Dry run PASSED** locally (CPU, Qwen3.5-0.8B @ 2fc0636, identity "lens"): all 3 parts ran; unembed check diff 0; with an
  identity lens the J-lens and logit-lens lists are identical, as they must be.
1. **Progress lines:** sanity prints each check; dm prints every 50 trials with ETA; cube every 10 captures with ETA.
2. **Incremental results:** dm appends one jsonl line per trial (flushed); cube writes one .safetensors + .json per capture
   (.json last = complete); both resume and skip finished work.
3. **Memory:** flat. Weights ~9 GB bf16, J fp32 0.8 GB, per-layer unembed of ≤877 positions × 248,320 fp32 ≈ 0.9 GB, rank
   computed one tracked id at a time. Peak well under 20 GB on a 48 GB A40.
4. **Killed at 80%:** everything finished is on the pod disk; nothing reaches S3 until the upload step — so the upload runs
   after EACH part, not once at the end.
- **Backup:** S3 presigned PUT URLs from the AWS connector (tested from this container: HTTP 200), ≤5 GB per object; cube
  output (~5.3 GB transported states + ranks) is split into parts <4.5 GB. Prefix `latent_cube/jlens01/`.
- **Pod:** `latentcube-jlens01`, template 8miemycg5h (image ghcr.io/adithyag73/interp-pod:torch28-tf5170), NVIDIA A40 SECURE
  **$0.59/hr** (catalog read 2026-11-24), CUDA ≥12.8. Watchdog: no API key in the cloud container, so Claude polls the pod with
  the RunPod connector and a send_later check fires 90 min after creation to terminate if anything is left.
- **Cost line:** boot+install ~5 min, model 9 GB + lens 0.4 GB download ~3 min, sanity ~1 min, dm 3,450 trials × ~0.2 s ≈ 12 min,
  cube 33,551 tokens ≈ 10 min, uploads ~5 min → ~36 min; ×1.5 for the tail (past runs ran 1.3–1.7× over) ≈ 55 min ×
  $0.59 ≈ **$0.55**. **Opening balance $88.10** (RunPod get-balance, 2026-11-24 ~10:56 UTC).

### JLENS-01 run log + results (2026-11-24, UTC)
- 11:17 pod `uzbes1kx4m8g0g` (latentcube-jlens01), A40 SECURE $0.59/hr, EU-SE-1, image interp-pod:torch28-tf5170
  (torch 2.8.0+cu128, transformers 5.17.0, fla 0.5.2), jacobian-lens 581d398 installed `--no-deps`. Run from git branch
  `von_neumann/ecstatic-rubin-q4en3l` @ 0a14bc2. Driven entirely through the RunPod connector's exec-pod (no SSH).
- 11:19:52 sanity: unembed(final block) vs model logits **max|diff| = 0**. Walkthrough boot prompt at "boot": J-lens L16 top-5
  ['?', '？', "'?", **' Italy'**, '____']; logit lens L16 ['shaw','วย','amaz','REA','举世'] — the hidden hop appears only in the
  J-lens, as the walkthrough says. L24 both show "shape". Model's next token ' the' / ' called' / ' Euro'.
- 11:20–11:30 dm: 3,450 trials, 0 errors. 11:30–11:44 cube: 338 captures, 0 errors. Upload via presigned PUT, all HTTP 200:
  `s3://…/latent_cube/jlens01/{small.tar, full.tar.part00, full.tar.part01, SHA256SUMS}`. small.tar downloaded to the cloud
  container: sha256 e520b94d… matches the pod's SHA256SUMS. Analysis outputs: `jlens01/results/`.

**RESULT 1 — positive control, category targets: REPLICATES the paper's pattern on Qwen3.5-4B.** Hit = a target member at
J-lens top-1 at any (layer 0–30, carrier position): no instruction 0/66 (0%), mention ("control") 59/396 (15%), **focus
100/330 (30%)**, suppress (dismissal + don't-think) 80/858 (9%). Top-5: 3% / 36% / 67% / 22%. Same order as the paper:
baseline ≈ 0 < suppress < mention < focus, and suppression is above zero (the "white bear" effect).

**RESULT 2 — positive control, MATH targets: FAILS on this model.** Focus 3/360 top-1 (1%), top-5 2%; baseline 0/72;
mention 2/432; suppress 0/936. Inspected one focus trial ("Concentrate on 4 * 2…"): the J-lens at mid layers shows the
carrier's own words and punctuation ("carefully", "letter", "**"), nothing about 8. The paper reports math modulation on
Claude models and says it grows with model size; on a 4B model the silent arithmetic answer is not readable at top-1/top-5.
Caveat: the user-turn wording is ours, not the paper's (they did not release it).

**RESULT 3 — AG's question (forced replies, ranks over layers 8–27 and the forced-reply tokens): NO SIGNAL.**
- No tracked math token reaches top-20 in ANY of 137 forced captures for "³", "²", " cube", "6", "2", "3", " six", " two".
- No power-specificity: median best rank of "³" is 72 / 70 / 55 / 54 for hidden powers 0 / 1 / 2 / 3; of "²" 34 / 49 / 30 / 28.
  "²" outranks "³" even when the hidden term is a cube. These ranks look like generic "maths-prompt" tokens, not the term.
- sin x vs cos x, 2nd term (best J-lens rank of "³"): hold 94 vs 91, imagine 47 vs 51. "²": 78 vs 78, 24 vs 27. Indistinguishable.
- What DOES show at top-1: " Paris" in the hold captures (11 of 28 at power 3) — that is the reply itself; " third" top-1 in 2.
**AG's prediction** (" cube"/"³", "sin", "x", "6") is **not supported**: " cube" median rank ~4,000, "sin" ~16,000, never top-20.
**Interpretation (Claude's, for AG to judge):** because the math positive control fails on this model, the null on −x³/6 is
NOT evidence that the term is absent; the lens on Qwen3.5-4B does not surface silently computed arithmetic even for 4 * 2.
The categories work, so the pipeline works. The informative next step is a bigger model (Neuronpedia also fitted
Qwen3.6-27B) for the math control, before reading anything into the cube null.
- 11:55 pod deleted (HTTP 204); list-pods empty. S3 sizes match the pod byte for byte. Safety-deadline task cancelled.
- **Cost:** 11:17 → 11:55 ≈ 38 min × $0.59 ≈ **$0.37** (estimate was $0.55 incl. tail). Opening $88.10; closing balance to be
  read ≥5 min after terminate (RunPod posts late) — PROVISIONAL until then. S3 storage added ≈ 6.75 GB.
