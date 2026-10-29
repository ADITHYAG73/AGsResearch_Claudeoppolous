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
