# mathematics/experiments.md — journal

Thread opened 2026-09-17 (session "MATHEMATICS"). Question still AG's to write; the running
candidate is "does the Kantamneni–Tegmark helix (arXiv 2502.00873) survive in a 2026
RL-trained model". Nothing below tests that yet.

---

## MATH-SMOKE-01 — first look at Qwen3.8-27B on a GPU (planned 2026-09-19)

**Goal.** Infrastructure only: load the model, print the architecture, run ONE GSM8K
question with thinking on, cache the residual stream at five layers over the whole trace
with forward hooks, terminate. No hypothesis. Output = files AG reads on CPU.

**Model facts (verified from HF config.json, 2026-09-19).** `Qwen/Qwen3.8-27B`, Apache 2.0,
not gated, 18 safetensors shards (~56 GB bf16). `architectures=[Qwen3_5ForConditionalGeneration]`,
`model_type=qwen3_5` (Qwen3.8 dense reuses the Qwen3.5 architecture). Text config:
**64 layers, hidden 5120**, 24 heads / 4 KV heads, vocab 248,320, 262k context,
`layer_types` = 3 × linear_attention then 1 × full_attention, repeating (`full_attention_interval=4`)
→ 48 Gated DeltaNet layers + 16 full-attention layers (indices 3, 7, 11, …, 63).

**Loading path (official transformers doc, model_doc/qwen3_5).** "Use `Qwen3_5ForCausalLM` for
text-only generation"; `Qwen3_5ForConditionalGeneration` is the multimodal class. The DeltaNet
path "needs the optional `causal_conv1d` and `fla` packages for its fast kernels — without
them, the model silently falls back to slower and more memory hungry PyTorch ops." Both
installed in `pod_setup_q38.sh`; the smoke logs whether each imports.

**Stack.** Image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (host CUDA 12.8 — pin it;
the console warned the A100 may land on 13.0/13.2), transformers 5.17.0 (the version QWEN-01
ran on this image), flash-linear-attention, causal-conv1d. Same stack as QWEN-01.

**Storage.** RunPod **Global Volumes (beta, launched 2026-09-14)**: region-independent, mounts
at `/workspace-global`, $0.09/GB/mo + IOPS, "written infrequently and read across
deployments: model weights…", pods only at launch, object-storage backed (no POSIX guarantees,
not for venvs or frequent writes). Source: docs.runpod.io/pods/storage/types and
runpod.io/blog/global-volumes-beta. This removes the data-centre pin that made a network
volume a bad bet on 2026-09-19 stock. Plan: weights on the global volume, venv in the
container. `pod_setup_q38.sh` picks `/workspace-global` if present, else `/workspace`.
**Not in the RunPod skill (v1.2.0) or MCP references — 5 days old. AG found it in the console.**

### Rule-7 preflight (answered from reading the code, before any pod exists)
1. **Timestamped progress per unit of work?** Yes. `snapshot_download` prints per-shard
   progress; the smoke prints a verdict line after load, arch, trace, acts, done.
2. **Results written incrementally?** Yes. `arch.txt` → `trace.json` → `acts/L{n}.npy` →
   `done.json`, each written before the next stage starts.
3. **Memory trajectory / peak vs card?** Weights 56 GB bf16 + KV for ≤ ~2.5k tokens + one
   full-sequence forward for hooks (activations kept on CPU as float16; 5 layers × ~2.5k ×
   5120 × 2 B ≈ 130 MB). Peak well under 80 GB. `done.json` records `peak_vram_gb`.
4. **Killed at 80%?** Everything already written survives; `done.json` absent = incomplete.
   Weights survive on the global volume regardless.
- **Cost line.** A100 80 GB secure $1.59/h. Download ~56 GB at ~100–200 MB/s ≈ 5–10 min;
  pip ≈ 3 min; load ≈ 2 min; generate ≤ 2k tokens ≈ 2–3 min; hooks < 1 min; pull artifacts.
  **≈ 25–35 min ≈ $0.70–0.95.** Global volume ≈ 56 GB × $0.09 ≈ **$5/month** standing, AG's call.
- **Idle watchdog:** run alongside, as always.

### Dry run (laptop, 2026-09-19)
Same script, `--model Qwen/Qwen3.5-0.8B` (24 layers, hidden 1024, same architecture
class), MPS float32, layers 0/11/23, 300 new tokens. Env: kaggle-code-comps venv,
transformers 5.11.0, torch 2.12.0, kernels absent (slow fallback, logged as such).
**PASSED, exit 0, 278 s.** Loader `Qwen3_5ForCausalLM` worked first try; decoder stack found at
`model.layers` (n=24); `arch.txt` 17 KB with the per-layer type table (3:1 pattern confirmed
in the loaded model, not just the config); trace 75 prompt + 300 generated tokens, thinking
on, truncated at my 300 cap (expected); hooks on layers 0/11/23 gave [375, 1024] each,
rows == sequence length (asserted), mean ‖h‖ 1.1 → 2.0 → 10.5 rising with depth.
Aside, not a result: the 0.8B forgot the four muffin eggs (16−3=13, ×2=26); gold is 18.
Local checkpoint deleted after the run.

### 2026-09-20 — decisions and a storage design change (before any pod)
- **AG chose: pod + global volume** (over a custom serverless worker or the text-only vLLM
  worker). Checked first: serverless docs list container disk / network volume / S3 only, and
  the global-volume launch post says pods only — the two cannot be combined today.
- The Hub banner "Qwen3.8-27B is live on Runpod Serverless" = Hub repo
  `runpod-serverless-templates/qwen3.8-27b-q8_0`: **llama.cpp, Q8_0 GGUF, 48 GB, OpenAI API.**
  Quantized and no hooks — not usable for interp. Settled via the Hub catalog API.
- **The RunPod API cannot create or attach a global volume** (`create-network-volume` types
  are STANDARD / HIGH_PERFORMANCE only; `create-pod` mounts are persistent / network only).
  So AG creates the volume and deploys the pod in the console; Claude drives over SSH.
- **Design change (Claude's catch while reading the docs again):** the volume is object
  storage — "no atomic rename, no file locking, no permission bits". The HF cache uses
  symlinks and `.incomplete` → rename. So the HF cache never touches the volume. Weights are
  PLAIN FILES: download to container disk (`snapshot_download(local_dir=…)`), load from there,
  then `save_to_volume.sh` copies file by file with size checks and writes a `DONE` marker
  last. Next session `pod_setup_q38.sh` sees `DONE` and restores volume → local instead of
  downloading. Both directions print MB/s — nobody has published global-volume throughput,
  so this run measures it. Container disk must be ≥ 100 GB.
- Rule-7 deltas: (1) save/restore print a timestamped line per file; (4) a killed save leaves
  no `DONE` marker, so the next session re-downloads rather than loading a partial copy.

### MATH-SMOKE-01 — RUN LOG (2026-09-19 evening IST; pod clock UTC)
**Opening balance $52.08** (AG supplied it; Claude had not recorded it — standing rule missed).
Pod `bay9lu1p0lc5cm` "naval_tan_octopus", A100 80GB PCIe, SECURE, **EU-RO-1**, $1.59/h, image
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, container disk 100 GB, global volume
`like_aquamarine_bandicoot`. Deployed by AG in the console (the API cannot attach a global
volume); driven by Claude over SSH. Created 16:12:24 UTC.
- 16:13 first contact. Host driver 595.91.07 / CUDA **13.2** (console warned "only 12.8
  compatible"): torch 2.8.0+cu128 sees the GPU and a 4096² matmul runs — newer drivers run
  older CUDA builds; the warning is conservative. Global volume mounted at
  **`/workspace-global`, type `fuse.geesefs`** (FUSE over S3-style object storage, 1.0 P
  nominal) — confirms the plain-files design. API `get-pod` shows `mounts: {}` for it.
- 16:14 watchdog self-test OK (detached with nohup; a tool-backgrounded copy would die at the
  10-min tool timeout). Setup launched via standalone launcher, pid 774.
- 16:15 **KERNELS FALLBACK — Claude's error.** `pod_setup_q38.sh` was copied from the QWEN-01
  script WITHOUT the fix this journal's sibling (`forking/experiments.md`, QWEN-01 16:24)
  records: causal-conv1d needs `CUDA_HOME` + `--no-build-isolation`. And because fla shared
  the pip command, it failed too. Setup carried on to the download by design.
- 16:16 `fix_kernels.sh` run in parallel with the download: fla 0.5.2 OK; causal-conv1d 1.7.0
  installed in 10 s once CUDA_HOME was set (prebuilt wheel found). **TODO: fold into
  pod_setup_q38.sh** (done below).
- 16:19 **DOWNLOAD OK: 52 GB, 18/18 shards, 245 s ≈ 212 MB/s** from HF, unauthenticated.
- 16:19:59 smoke launched, pid 2881. **LOAD OK in 11 s** (files still in the host page cache —
  1.5 TB RAM host; do not expect 11 s from a cold disk). `Qwen3_5ForCausalLM` first try.
  26.90 B params, 64 layers at `model.layers`, both kernels OK.
- 16:20:45 **SMOKE DONE, 48.1 s total, peak VRAM 54.1 GB.** Trace: 117 prompt + 161 generated
  tokens in 37 s, finished, **answer 18 correct**. Thinking block is terse (one paragraph,
  "total used 7. Remainder 9. … $18") — GSM8K item 0 is far too easy to elicit a long chain.
  Activations: layers 0/15/31/47/63, **[278, 5120]** each, rows == sequence length (asserted).
  Mean ‖h‖: 14.4 → 60.8 → 80.3 → 113.1 → **401.2** at the last layer.
- 16:20:53 `save_to_volume.sh` started once the model was in VRAM. First five shards:
  3782 MB/17 s, 2902/12, 2425/8, 3804/16, 2002/6 → **~220–330 MB/s write** as seen by `cp`
  (FUSE may buffer; the size check reads the FUSE view, so this is an upper bound on true
  upload speed).
- Artifacts pulled to `mathematics/runs/2026-09-19_smoke/` (arch.txt 39 KB, trace.json,
  done.json, 5 × 2.8 MB npy, setup/smoke/kernel logs, watchdog log).

**What the timings say about AG's actual frustration.** Deploy → model in VRAM was ~8 min,
of which: pip ~1 min, kernels ~10 s when done right, download ~4 min, load 11 s. The weights
were never the slow part. What made earlier sessions painful was vLLM/sglang installs,
kernel compiles, server start, and debugging them live. The fix for THAT is a baked Docker
image + saved pod template + one launcher script — not a volume and not serverless.
- 16:24:31 **SAVE OK: 53,012 MB to the global volume in 216 s = 244 MB/s write**, 29 files,
  every size checked, `DONE` marker written last.
- 16:24:48 read-back probe, shard 1 (written first), `dd` to /dev/null: **141 MB/s**, and `cmp`
  against the local copy = **byte-identical**. Caveat: single stream, one shard, same pod that
  wrote it. The honest restore number comes next session — `pod_setup_q38.sh` sees `DONE`,
  restores volume → local and prints MB/s.
- 16:27:15 watchdog ALARM, GPU idle 6 min — correct: all work was done and Claude was waiting
  for AG's word to terminate (pod was created by AG in the console, so not Claude's to delete
  unasked). ~3 min of idle billing, ≈ $0.08. `pip freeze` (159 packages) captured first.
- 16:28:5x AG: "terminate". `delete-pod` → 204; `list-pods` → []; local watchdogs killed.
- **Closing balance $51.69. MATH-SMOKE-01 cost $0.39** (opening $52.08), ~16.5 min of pod.
  The balance API still showed "spend/hr $1.60" for a while after the pod list was empty —
  a lagging field, polled until it cleared.

**VERDICT.** Infrastructure goal met: architecture dump, one thinking-on math trace, residual
stream at five layers via forward hooks, all on disk, 8 min from deploy to results.
**Global volume: works, but on tonight's numbers it reads SLOWER than Hugging Face
(141 vs 212 MB/s) and costs ~$4.70/month for 52 GB.** Keep until one real restore on a
fresh pod is measured; delete if it loses again. Its real use is what HF cannot serve:
gated/private weights, fine-tunes, and cached activations reachable from any data center.

**Pinned versions for the future image (from the pod's pip freeze):** torch 2.8.0+cu128,
transformers 5.17.0, flash-linear-attention / fla-core 0.5.2, causal-conv1d 1.7.0,
accelerate 1.15.0, triton 3.4.0, safetensors 0.8.0, tokenizers 0.23.2, numpy 2.1.2.
Full list: `runs/2026-09-19_smoke/pip_freeze.txt`.

**NEXT (no GPU needed):** Dockerfile + GitHub Actions build (linux/amd64; AG's Mac is ARM) →
public image → RunPod template via API → one launcher that takes a model name. Then a harder
question than GSM8K item 0 — it produced one paragraph of thinking. The research question is
still AG's to write.

### 2026-09-19 (late) — two RunPod pod templates created (free; no pod)
A template saves SETTINGS only (image, disk, ports, env, start args, CUDA filter). Installs
vanish only if the IMAGE already holds the software.
| Template | id | Image | What it removes | What it does NOT remove |
|---|---|---|---|---|
| `ag-interp-torch28` | `31hq5fjfvy` | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (the tag MATH-SMOKE-01 ran on; newer 1.3.x tags exist, deliberately not adopted untested) | console clicking: 100 GB disk, SSH, HF env, CUDA filter 12.8/13.0/13.2 | the ~1 min pip + kernel step (`pod_setup_q38.sh`) — needs a custom image |
| `ag-vllm-0290-qwen38-27b` | `mf96kxlalr` | `vllm/vllm-openai:v0.29.0` (official, Docker Hub, 9.6 GB, 2026-09-09) | ALL installs — vLLM is in the image; args start the OpenAI server on :8000 with Qwen3.8-27B, `--reasoning-parser qwen3`, 16k context | the 52 GB weight download (~4 min) and vLLM's own model load; **no sshd in this image → no SSH, logs via console only**; CUDA filter 13.0/13.2 |
**UNTESTED: neither template has been deployed.** Unknowns for the vLLM one: image pull time
on a RunPod host, whether RunPod passes `args` to this image's entrypoint as expected (API
echoed them back as `cmd`), first-token time. vLLM serves text; it cannot give activations.

### 2026-09-19 (22:36–22:50 IST) — custom interp image BUILT; third template created. $0.
- Repo **github.com/ADITHYAG73/interp-pod-image** (public, new; local clone at
  `../interp-pod-image/`). Account chosen by Claude: ADITHYAG73 (research identity); the `gh`
  active account (IamAGP) was not switched — token passed per command. Secret scan before push: clean.
- GitHub Actions run 35457050763: **success, first try.** Image
  **`ghcr.io/adithyag73/interp-pod:torch28-tf5170`** (+ `:latest`), linux/amd64.
- Evidence from the build log (`verify.py --build`): torch 2.8.0+cu128, transformers 5.17.0,
  flash-linear-attention 0.5.2, causal-conv1d 1.7.0 all OK; nnsight 0.7.0 installed WITHOUT
  moving torch/transformers; classes present: Qwen3_5 / Qwen2 / Gemma3 / Gemma4 / Olmo3
  `ForCausalLM`. causal-conv1d wheel "built" in 6 s at 196 MB → inference: the prebuilt wheel
  was fetched, not compiled.
- **Anonymous manifest pull = HTTP 200** → package is public, RunPod needs no registry credential.
  (The visibility API itself returned 403: the gh token lacks `read:packages`. Not needed.)
- Template **`ag-interp-ghcr-torch28-tf5170`**, id **`8miemycg5h`** → that image, 100 GB disk,
  SSH + Jupyter, HF env, CUDA 12.8/13.0/13.2. The stock-image template `31hq5fjfvy` is KEPT as
  the fallback until the custom image has run on a GPU.
- **UNTESTED ON A GPU.** Open risks: (1) does the image still start sshd/Jupyter — it should,
  the base ENTRYPOINT/CMD were not touched, but unverified; (2) image pull time on a RunPod
  host (base layers are likely cached there, our layers are new); (3) kernels import under a
  real driver (`python3 /opt/interp/verify.py` on the pod answers this in seconds).
- Next billed step, when AG says go: deploy from `8miemycg5h`, run verify + `/opt/interp/smoke.py`,
  and time deploy → model in VRAM against last night's 8 min.

---

## IMAGE-TEST-01 — does the custom image work on a GPU? (2026-09-19, 22:52 IST)
**Goal.** Prove `ghcr.io/adithyag73/interp-pod:torch28-tf5170` via template `8miemycg5h`:
sshd comes up, `verify.py` passes under a real driver (kernels import), and deploy → model in
VRAM beats MATH-SMOKE-01's ~8 min (which included ~1 min pip and my kernel mistake).
Same model, same GSM8K-0 question, same layers, so the timings compare.
**Opening balance $51.63** (17:22 UTC). It was $51.69 at last close: −$0.06 with no pod running
→ the global volume's storage/IOPS charge or billing lag; watch it.
**Pod:** created by Claude through the API this time (no global volume → no console needed).
A100-SXM4-80GB SECURE, $1.59/h, stock Medium.
**Rule-7 preflight (from the code that will run — `run_imagetest.sh` + `/opt/interp/smoke.py`):**
1. Progress: timestamped stage lines (VERIFY / DOWNLOAD / SMOKE / ALL DONE); download prints
   per-file progress; smoke prints a verdict per stage.
2. Incremental: smoke writes arch.txt → trace.json → acts/*.npy → done.json, one at a time.
3. Memory: measured last night, peak 54.1 GB on an 80 GB card.
4. Killed at 80%: files stay on the pod's disk until terminate; nothing is lost that the
   next run cannot regenerate in minutes. Artifacts are pulled before terminate.
Cost line: ~10–12 min × $1.59/h ≈ **$0.27–0.32**. Watchdog: yes, 6-min idle alarm.

### IMAGE-TEST-01 — RUN LOG AND VERDICT (2026-09-19, pod clock UTC)
Pod `xh0uh1hraktl4p` "math-image-test-01", A100-SXM4-80GB SECURE, $1.59/h, host driver 570.172.08
(CUDA 12.8), created by Claude via API from template `8miemycg5h`.
| Stage | Clock | Note |
|---|---|---|
| create-pod | 17:23:16 | |
| runtime up, SSH reachable | ~17:24:19 | **image pull + start ≈ 63 s** |
| verify.py | 17:24:58 → 17:25:07 | **PASSED**: versions, 5 model classes, `fla` and `causal_conv1d` import under a real driver |
| download 52 GB | 17:25:07 → 17:25:57 | **50 s ≈ 1 GB/s** (last night EU-RO-1: 245 s) — data-centre dependent |
| smoke | 17:25:59 → 17:27:12 | load 25 s, generate 46.8 s, hooks 0.2 s, peak VRAM 54.1 GB |
| ALL DONE | 17:27:14 | **3 min 58 s from create-pod**, vs ~8 min last night. **Zero installs.** |
| terminate | ~17:28:30 | `delete-pod` 204, `list-pods` [] |
- sshd and the scripts in `/opt/interp/` were present → the base image's start command survived our layers.
- **Reproducibility check (free, and worth having):** tonight's 161 generated token ids == last
  night's, and all five activation arrays are **BIT-IDENTICAL** to last night's — different pod,
  different data centre, A100 SXM vs A100 PCIe, host CUDA 12.8 vs 13.2, stock image vs custom
  image. Greedy decoding + same GPU architecture + same software stack ⇒ same bits. (Contrast
  POS-01, 2026-08-22: A100 vs A40 was NOT bit-identical.)
- Watchdog: self-test OK; it counted 3 idle polls during the download, then "working again".
- Balance: opening $51.63 → $51.54 read right after terminate (spend/hr field still lagging at
  $1.60, as last night). Expected cost ≈ 5.3 min × $1.59/h ≈ $0.14. Re-read later for the true close.
**VERDICT: the custom image works. `8miemycg5h` is now the default interp template;
`31hq5fjfvy` (stock image + setup script) stays as the fallback.**

---

## TRACE-01 — first attribution graph on the actual question (2026-09-19 23:16 IST). $0, no GPU.
**What:** Neuronpedia Circuit Tracer via API (key `NEURONPEDIA_APIKEY` in `.env`), prompt
`calc: 36+59=` (the Biology paper's exact prompt), model `qwen3-4b`, defaults. 38 s.
Graph: https://www.neuronpedia.org/qwen3-4b/graph?slug=calc-36-59-09192316 — JSON saved in
`runs/2026-09-20_circuit_tracer/` (8 MB; 1548 nodes, 103,523 links; transcoders
`mwhanna/qwen3-4b-transcoders`, circuit-tracer 3f0965b6).
**Why Qwen3-4B:** not a choice — graph generation needs per-model transcoders. Source
(`apps/webapp/lib/utils/graph.ts`, read 2026-09-19): enabled = gemma-2-2b, gemma-3-4b-it,
qwen3-4b, qwen3-1.7b; prompt cap 64 tokens. None exist for Qwen3.8-27B.
**Findings (descriptive, n=1 prompt):**
1. **The model is NOT confident:** next token `?\n\n` 35.7%, **`9` 24.6%** (correct first
   digit; Qwen emits digits singly), ` ` 11.6%. The `calc:` format is Claude's, not Qwen's.
   A fair test needs a prompt format where Qwen puts most of its mass on the digit.
2. Direct inputs to the `9` logit: late layers 29–35, all at the `=` token. Top positive:
   L34 "9", L33 "say 9", L29 "nine" (Neuronpedia auto-labels) = OUTPUT features.
3. One hop back, inputs to L33 "say 9": **the largest are reconstruction-error nodes**
   (L32 +7.5, L30 +5.4, L31 +5.4, L24 +5.0) — i.e. the part the transcoders do NOT explain —
   then L29 "nine" (+6.3) and generic "mathematical expressions"/"numbers" features.
   **No "_6+_9"-style lookup feature and no magnitude feature appears in the top 14.**
   Whole-graph influence: error nodes 119 vs features 828.
**Reading (Claude's, not a result):** on this prompt the interpretable part of the graph shows
the END of the computation ("say 9") and the unexplained error nodes carry the step we care
about. That is exactly the limitation the Biology paper and Neuronpedia's landscape page
name. It does not show Qwen3-4B lacks lookup features — they may sit at the operand tokens
(`6`, `9`), several hops upstream, or inside the error. Not checked.
**Cheap next (AG's call):** (a) find a prompt format with >80% on the digit, regenerate;
(b) walk the graph from the operand tokens forward instead of from the logit back;
(c) same prompt on gemma-2-2b for a second model.

### TRACE-02 — the three follow-ups (2026-09-19 23:27–23:50 IST). $0, no GPU.
Tools written: `mathematics/pipeline/np_graph.py` (generate + save + print next-token probs, with
retries), `walk_graph.py` (features per prompt position), `last_pos.py` (most important features at
the final position). Labels cached in `runs/2026-09-20_circuit_tracer/labels_<model>.json`.
**1. Prompt format (Qwen3-4B), probability on the correct first digit `9`:**
| format | P(`9`) | note |
|---|---|---|
| `calc: 36+59=` (paper's) | 0.246 | top guess `?` 0.357 |
| `36+59=` | not in top 4 | top `?` 0.273; `3` 0.048 |
| `Question: What is 36+59?\nAnswer: ` | **0.812** | runner-up `3` 0.181 |
| few-shot `12+30=42\n…36+59=` | — | **HTTP 500 from Neuronpedia every time** (server side; cause unknown) |
`36 + 59 = ` also 500'd once. Use the Question/Answer format.
**3. Second model, same prompt, gemma-2-2b: P(`9`) = 0.875** (runner-up `1` 0.056).
**2. Walking the graphs.** By position: features at operand tokens are identity features
("3", "6", "plus sign", "addition") — the model reading the question. At the FINAL position,
most important features in BOTH models are late-layer output features: Qwen L29 "nine", L33
"say 9", L34 "9", L35 "single digit numbers"/"digits"; Gemma L24–25 features whose top boosted
tokens are `9 / nine / ninety`. Competitors visible: Qwen L34 "say 3"; Gemma L23 boosting `6/six`.
Feature share of influence: Qwen 76%, rest is reconstruction error.
**NOT found in either model's top ~26 final-position features: anything labelled like a
"6+9" lookup, a "sum ≈ 95" magnitude feature, or a carry.** Caveats: auto-labels are vague
("confidence intervals" for a feature that boosts `9`); only top-K by importance at one
position were labelled; one prompt; attention is not decomposed by this method at all.
**Reading (Claude's):** with this tool and auto-labels we can see the model DECIDING to say 9
and cannot yet see HOW it got there. To see the "how" needs either (a) many prompts with the
operands varied systematically, watching which features track a, b, a+b, ones digit — the
Biology paper's operand-plot method — or (b) the helix probing of Kantamneni–Tegmark on our
own cached activations.

---

## PARABOLA-01 — AG's first own prompt on Qwen3.8-27B (2026-09-19 23:52 IST / 18:21 UTC)
**Prompt (AG's, verbatim, saved in `runs/2026-09-20_parabola/prompt.txt`, typo "right words" and
double space kept):** *Imagine a parabola which is open right words. Now give me the equation of
the one that is opening leftwards with its  vertex two units from the origin*
**Noted before running (Claude, not a change to the prompt):** underdetermined twice — direction
of the 2-unit shift ((2,0), (−2,0), or off-axis) and the focal parameter a. General form
y² = −4a(x − h). So there is no gold answer; the observable is which assumptions the model makes
and whether it flags the ambiguity. AG's original curiosity (session start): does "parabola"
trigger y² = 4ax for a model the way it does for him.
**Opening balance $51.49.** Template `8miemycg5h` (custom image), A100 80GB SECURE $1.59/h, via API.
**Rule-7 preflight (code read: `run_prompt.sh` + patched `smoke_q38.py`):**
1. Progress: stage banners; download progress; **generation now streams tokens to the log**
   (TextStreamer added tonight — last night's generate was silent until done).
2. Incremental: arch.txt → trace.json → acts/*.npy → done.json.
3. Memory: 54 GB measured on GSM8K; longer sequence adds KV + a 4k-token forward for hooks — small.
4. Killed mid-generation: the streamed text survives in prompt.log; trace.json would not exist.
**Cost line:** HF generate ran at ~3.4–4.3 tok/s on this model (161 tok in 37–47 s). Cap 4096 new
tokens → worst case ~20 min generation + ~3 min setup ≈ 23 min ≈ **$0.61**; if it answers in
~800 tokens, ≈ 7 min ≈ **$0.19**. Watchdog on, 6-min idle alarm.

### PARABOLA-01 — RUN LOG AND RESULT (2026-09-19/20, pod clock UTC)
Pod `753axby4tk5jgi` "math-parabola-01", A100-SXM4-80GB SECURE, **US-KS-2**, $1.59/h, driver 580.159.04,
template `8miemycg5h`, created by Claude via API.
| Clock | Event |
|---|---|
| 18:21:53 | create-pod. SSH reachable ~30 s later. |
| 18:22–18:23 | **BUG 1 (Claude):** `set -- $HP` — zsh does not word-split an unquoted variable, so host and port went as ONE argument; scp/ssh/launch all failed. **The watchdog's self-test refused to start** on the bad address — that is how it was noticed. ~1.5 min idle. |
| 18:23:52 | relaunched with explicit values. `verify.py` PASSED, zero installs. |
| 18:24:00 → 18:26:56 | download 52 GB in **176 s** (US-KS-2; vs 50 s and 245 s elsewhere). |
| 18:26:56 → 18:26:58 | **BUG 2 (Claude):** `--answer "-4a"` — argparse read `-4a` as a flag, run died in 2 s. My completion grep matched "ALL DONE" and missed the lowercase `error:`. A 1-second local parse check would have caught it; skipped. Now done first. Fix `--answer=-4a`. |
| 18:28:06 | RUN2. Load 12 s. |
| 18:28:19 → 18:29:42 | **951 tokens in 82.6 s (11.5 tok/s** — faster than GSM8K's 3.4–4.3; longer sequence amortises). finished=True. |
| 18:29:42 | acts: layers 0/15/31/47/63, **[1035, 5120]**, mean ‖h‖ 14.4/56.2/81.2/116.5/395.6. Peak VRAM 54.6 GB. |
| ~18:31 | artifacts pulled, prompt-on-pod == prompt.txt asserted, `delete-pod` 204, `list-pods` []. |
**Balance $51.49 → $51.36 at 18:31:34 (spend/hr $0.00). Cost so far $0.13**; late postings possible.
~4 min of the ~9.5 min pod life was my two bugs (~$0.10).
Streaming note: TextStreamer flushes by line, so nothing appeared for the first ~25 s; fine.

**WHAT THE MODEL DID (descriptive; n=1, greedy):**
- Thinking 460 words / answer 85 words. Same two registers as GSM8K: telegraphic thinking
  ("Need parse.", "Need final answer. Ensure not overdo?"), fluent LaTeX answer.
- **It caught both ambiguities on its own, in the first paragraph of thinking:** "vertex two units
  from origin could be at (2,0) or (-2,0) or any point distance 2 from origin" and "scale not
  specified. Infinite parabolas." It also read "right words" as a typo for "rightwards".
- **It wrote the general constraint h² + k² = 4** in its thinking — the vertex lies on a circle of
  radius 2 — then dropped that from the final answer.
- Its template is **(y−k)² = 4p(x−h)**, p<0 opens left — the US textbook form with **p**, not AG's
  **y² = 4ax** with **a**. It did reconstruct "y² = 4x opens right with vertex origin" as the
  parabola AG was imagining, and treated the task as reflect + translate.
- Final answer: general form, then y² = −4(x−2) for vertex (2,0), p=1 (its preferred reading:
  "opening leftwards from x=2 toward origin"), then y² = −4(x+2) for (−2,0), then "infinitely many".
- It considered asking for clarification ("Need maybe ask clarification? But final should answer
  with assumptions.") and chose not to.
**On AG's session-opening question (does "parabola" trigger the equation for a model):** in TEXT,
yes — the standard form appears within the first ~60 words of thinking. Whether anything
shape-like exists in the activations is NOT addressed by this run; the [1035, 5120] arrays at five
layers are on disk for that, and nobody has looked at them.

---

## QUEUE — AG's two asks for the next session (recorded 2026-09-20, before he rested). Nothing run.
**A. PARABOLA-02 — "does the model visualise?" (AG: "the ideas you proposed for parabola are worth
trying").** Qwen3.8-27B is natively vision-language. Compare its internal state for the WORDS
"a parabola opening rightwards" against an IMAGE of one, with left-opening (words + image) as
controls. Falsifiable: if words-right sits closer to image-right than to image-left, that is a
defensible "picture-like" state; if words only resemble other equations, it is formula retrieval.
To settle BEFORE any pod (AG's calls): which layers (all 64 is ~cheap), which token position to
read, how many paraphrases/images per condition, what counts as "closer". Needs the multimodal
class `Qwen3_5ForConditionalGeneration` + processor — NOT yet loaded by us; test on the laptop
with Qwen3.5-0.8B first.
**B. EXP-01 — "can the models calculate e^x, and if yes, how?" (AG's question, his words).**
Not yet scoped. Things to pin down with him first: numeric e^x (e^2 ≈ 7.389) vs symbolic; which x
(integers, decimals, negatives, large); thinking on vs off — the three-mechanisms split applies:
(1) one forward pass = recall/interpolation, (2) chain of thought = series or log tables written
out, (3) tools = excluded by his objective. Cheapest first look: accuracy vs x with thinking off
and on, read the thinking text for the METHOD it uses. Literature not checked yet.

---

## NUMBERS-01 — does Qwen3.8-27B hold numbers in a structured way? (2026-09-20)
**Why this experiment, and why now.** AG's backward-learning principle (recovered today from the
2026-09-13 transcript, now memory `backward-learning`): give him a concrete runnable artifact, let
him ask, walk back only as far as each question needs. Claude had just written an 8-sitting forward
reading plan for Kantamneni–Tegmark — the third forward reading plan in two weeks, and wrong each
time. This replaces it.
**The adaptation, and why it is not a replication.** The paper (§4.1) feeds SINGLE-TOKEN integers
and Fourier-analyses the layer-0 residual stream. Qwen3.8 splits numbers into digits (confirmed in
the probe: "36" → ["3","6"]). So we read at the number's **last digit token**, the only position
that has seen both digits. §5.5 of the paper explicitly says digit-tokenising models "must use
additional algorithms to collate digit tokens" — nobody has looked. Numbers **10–99 only**, so every
prompt has identical token count and the number sits at a fixed index (no position confound).
Two conditions: `bare` = "36"; `arith` = "Output ONLY a number. 36+" (the paper's own prompt).
Cache **all 65 depths** (embedding + 64 layers) — last night's 5 layers were Claude's placeholder,
which AG called out.
**Built-in control the data answers by itself:** if that position holds only "6", then 36 ≈ 46 and
36 ≉ 31. If it holds "36", the reverse.
**Dry run, laptop, Qwen3.5-0.8B, CPU, free — PASSED and already informative:**
| depth | same-last-digit | same-first-digit | \|a−b\|≤2 | period-10 lift |
|---|---|---|---|---|
| layer 1 | +0.567 | +0.118 | +0.278 | +0.636 |
| layer 24 (last) | +0.171 | +0.363 | +0.498 | +0.244 |
→ early layers hold mostly the LAST DIGIT; by the last layer the representation has assembled the
whole number (first digit and near-neighbours dominate). Period-10 structure present throughout.
This is the 0.8B model; the 27B is the question.
**Rule-7 preflight (from the code):** (1) timestamped line per condition + tokenisation assert
before the model loads; (2) each condition's .npy written before the next starts; (3) memory =
54 GB weights + a batch of 90 nine-token prompts ≈ nothing, activations go to CPU as float16
(~60 MB); (4) killed mid-run → condition `bare` already on disk. Prompt lengths are asserted equal
before any forward pass.
**Cost line:** download ≤4 min + load ~25 s + two forward passes (seconds) ≈ 5–6 min × $1.59/h ≈ **$0.16**.

### NUMBERS-01 — RUN LOG (2026-09-20, pod clock UTC)
Pod `5ddoa5wmuiucy1` "math-numbers-01", A100-SXM4-80GB SECURE, **US-MD-1**, template `8miemycg5h`.
Opening balance **$51.24**.
| Clock | Event |
|---|---|
| 05:54:43 | create-pod; SSH reachable on the first poll |
| 05:55:24 | `verify.py` PASSED, zero installs |
| 05:55:34 → 05:58:32 | download 52 GB in **178 s** |
| 05:58:38 | tokenisation asserted: `36` → `["3","6"]`, `46` → `["4","6"]` — digits are separate, as expected |
| 05:58:48 | LOAD OK in **7 s** (host page cache) |
| 05:58:48 → 05:59:28 | `bare` [90,65,5120] in 39.7 s; `arith` in **0.5 s** (weights already resident) |
| 05:59:28 | DONE, peak VRAM **68.2 GB** (higher than generation's 54.6 — a 90-prompt batch with all-layer hooks) |
| ~06:01 | artifacts pulled (115 MB), `delete-pod` 204, `list-pods` [] |
No bugs this run. Total pod life ~7 min.
**RESULT (descriptive; the interpretation is AG's to reach).** `bare` condition, average cosine
similarity after mean-centring:
| depth | same last digit | same first digit | \|a−b\| ≤ 2 | period-10 lift |
|---|---|---|---|---|
| 0 (embedding) | **+1.000** | −0.111 | −0.009 | +1.111 |
| 1 | +0.732 | +0.040 | +0.117 | +0.815 |
| 4 | +0.280 | +0.191 | +0.299 | +0.330 |
| 16 | +0.139 | +0.276 | +0.373 | +0.135 |
| 32 | +0.171 | +0.401 | +0.504 | +0.154 |
| 48 | +0.225 | +0.429 | +0.522 | +0.232 |
| 64 | +0.127 | **+0.592** | **+0.644** | +0.081 |
The built-in control answers itself: at the embedding the reading is **exactly** the last digit
(36 and 46 identical, cos = 1.000 — the embedding of token "6" cannot depend on what preceded it),
and by the final block the ordering has inverted. The 0.8B dry run shows the same shape.
**Artifact for AG (interactive, hover any pair, slider over 17 depths, both conditions):**
https://claude.ai/artifact/H7du5JUmGXjGpmGbemmo1k — deliberately contains NO interpretation
(backward-learning discipline: he looks, he asks, Claude answers one level down and stops).
Analysis JSON `runs/2026-09-20_numbers/sim.json`; probe `pipeline/pod/numbers_probe.py`.
**Open, not yet done:** Fourier spectrum per depth (the paper's Fig 2); whether a helix FITS these
activations; whether the structure is causal (patching) rather than merely present; 3-digit numbers;
a single-token-number control model (GPT-J) if a null ever needs interpreting.

### PROBE-01 — is the FIRST digit readable at the SECOND digit's position? (2026-09-20, free, no GPU)
Run on the NUMBERS-01 activations already on disk. Script `pipeline/probe_first_digit.py`,
results `runs/2026-09-20_numbers/probe_first_digit.json`.
**v1 was DISCARDED — Claude's error.** Its `last digit` sanity control read 0.0% at every depth,
which is impossible (at depth 0 the vector IS the last digit's embedding). Cause: the split held
out last digits 8,9, so those classes never appeared in training and the probe could not emit
them. A control that cannot pass is not a control; the numbers beside it were not reportable.
**v2 design.** Probe = plain multinomial logistic regression (Claude's own, ~40 lines, no sklearn),
**NOT the paper's sinusoidal probe** — so these numbers are NOT comparable to their 99%.
- target: first digit, 9 classes, chance 11.1%; split holds out 2 LAST digits (test numbers unseen,
  all first-digit classes present in training); 8 random splits → mean ± sd.
- control A: predict the LAST digit, split holds out 2 FIRST digits → must be high at depth 0.
- control B: training labels shuffled → must sit at chance.
**RESULT (both controls pass: A = 100.0% at depth 0, B = 4–11% throughout).**
| depth | kind | first digit | last digit ctrl |
|---|---|---|---|
| 0 embedding | — | **11.1% ± 0.0** (exactly chance) | 100.0% |
| 1 | linear attn | **100.0% ± 0.0** | 100.0% |
| 2, 3 | linear attn | 99.3% ± 1.8 | 100.0% |
| 4 | first FULL attn | 96.5% ± 5.5 | 98.8% |
| 32 | full attn | 99.3% ± 1.8 | 93.8% |
| 64 | full attn | 97.9% ± 5.5 | **68.8%** |
**AG predicted "linear", before the run.** Confirmed in the narrow sense: the first digit is fully
readable at block 1, three blocks before any full-attention block exists.
**Why Claude will NOT let that stand as support for the hypothesis:** 100% at the *very first*
block is the signature of COPYING, not computing. The linear-attention mixer contains a width-4
`conv1d` that mechanically blends preceding positions; the first digit's embedding may simply be
sitting there near-verbatim. Presence ≠ assembly. Presence ≠ use (probes show presence only).
**Incidental:** the last-digit control decays 100% → 68.8% by block 64 — deep in the model the
identity of the token being sat on is lost while the first digit is retained. Untested explanation.
**Methodological lesson worth keeping:** the cosine-similarity square we read all afternoon said
36 vs 46 fell to 0.14, which *looked* like the first digit mattering more with depth. The probe
says the first digit was fully readable from block 1 the whole time. Cosine over the whole vector
is a far blunter instrument than a probe along one direction. Both results are correct.

---

## WHERE THIS STANDS (written 2026-09-20 for AG to pick up cold)
**Established, by us:** (a) at the embedding, the reading position holds ONLY the last digit —
all nine numbers ending in 6 are bit-identical, verified against the real Qwen3.8-27B tokenizer
(ids `36`→[18,21], `46`→[19,21]); (b) the first digit is readable from block 1 onward at ~100%,
controls passing; (c) an instrument + artifact AG can drive himself.
**NOT established:** that anything was *computed* rather than copied; that the structure is a
helix; that the model *uses* any of it; any comparison to the published 99% (different probe).
**Prior work found 2026-09-20, AFTER Claude had wrongly told AG this was open ground:**
- Štefánik et al., ACL 2026, arXiv **2510.26285** — §4.1 "Multi-token Numbers" is our exact
  question. Reads at the last token, recovers earlier pieces with a sinusoidal probe, **99% at
  offset −1**, drops past 3 tokens. Models: Llama 3 (1B/3B/8B), Llama 3.2, OLMo 2 (1B/7B/13B),
  Phi 4 15B. Excluded >30B. Code: `prompteus/numllama` (**MIT**) and `prompteus/param-sin`
  (**NO LICENCE** — do not build a publishable artifact on it).
- Wen et al., ICML 2026, arXiv **2606.03645** "The Shape of Addition" — carry geometry,
  Qwen3-4B / Qwen3-8B / Gemma-3-4B, 3-term 10-digit addition, final layer, UMAP.
  Code: `RL-MIND/Shape-of-Addition` (MIT).
**The gap, verified from config.json today:** every model in both papers is standard attention
(`Qwen3ForCausalLM`, no `layer_types`). The hybrid starts at Qwen3.5 (`layer_types` =
[full_attention, linear_attention], `full_attention_interval` 4). **Nobody has tested the hybrid.**
Honest sizing: this is a transfer test, not a discovery, and "yes it transfers" is a weak paper.
**THE ONE DECISION WAITING:** run Qwen3-8B (standard attention, 36 layers) against Qwen3.5-9B
(hybrid, 32 layers) — same family, near-same size, different mixer — with the SAME probe at every
block. If the standard model ALSO hits ~100% at its block 1, then linear attention has nothing to
do with it and both AG's and Claude's reading of block 1 was wrong. That is the test that can
embarrass the hypothesis, which is why it is the one worth running. ~1 pod session, <$0.50.
Use the paper's MIT sinusoidal probe if the result is meant to be comparable to theirs.

---

## PAPER READING — all three read in full (2026-09-21)
Claude had read only Kantamneni fully; AG asked for all three cover to cover plus a side-by-side.
Done. Texts in `mathematics/papers/*.txt`; repos cloned to `reference_repos/` (gitignored):
`LLM-addition` (no licence), `numllama` (MIT), `param-sin` (no licence), `Shape-of-Addition` (MIT).
**Artifact:** https://claude.ai/artifact/6ncw52cWocJoo9nPEvYwk3 — intent, procedure, each mechanism
as hand-runnable pseudocode, and every claim tagged SHOWN / CLAIMED / ADMITTED.

**P2 = Štefánik et al., ACL 2026 (2510.26285) — what it actually does.** Two aims: (a) quantify
that sinusoidal number representations are near-universal; (b) argue **general-purpose interp
tools give wrong answers on numbers**. Evidence: 8 models across 3 families agree *perfectly* on
their top-63 Fourier frequencies (control word-pieces do not); the `param-sin` probe (imposes the
sine shape, learns frequency/amplitude/phase per feature) reads numbers at >90% at most layers;
a probe trained on ONE layer transfers to others at ~100% in the middle of the net; multi-token
numbers recover the previous piece at **99%** from the last token, collapsing past 3 tokens;
**TunedLens says the opposite** and that is the tool's failure, not the model's; steering toward
the expected sine repairs up to 30% of multiplication and 42% of division errors. Key
methodological point AG should keep: probes trained on numbers in **natural sentences** generalise
to synthetic contexts, but probes trained on synthetic contexts do NOT generalise back — so
conclusions drawn only from synthetic arithmetic prompts may not hold in practice.
Admitted: 3 families, nothing >30B, English-only, probe blind spot at the first ordinal item.

**P3 = Wen et al., ICML 2026 (2606.03645) — what it actually does.** Explains *failure*, not
capability. Qwen3-4B, 10,000 sums of three 10-digit integers, greedy, **generation stopped at the
first wrong digit** so every analysed activation follows a correct history (a control worth
stealing). Claim: the carry is not binary but a continuous **Carry Potential**
Φ_p = Σ_j r_(p+j)/10^j, with true carry = floor(Φ). The model perceives Φ̂ = Φ + ε,
ε ~ N(0, σ²), σ≈0.05, and floors that — so errors spike when Φ lands near an integer and vanish
near half-integers. The predicted **bathtub** error curve fits at R²=0.80. Probes: model's own
digit 98.8%, true digit 94.9% (the gap IS the errors), raw sum 98.6%, carry 96.8%, Φ 92.1%.
Steering along the carry direction flips digits, and boundary states flip at α≈0.3 vs α≈0.5 for
stable ones. **Sharpest result is an ablation:** own raw-sum + TRUE carry → 96.0%; TRUE raw-sum +
own carry → 90.5%. **The carry is the bottleneck, not the column arithmetic.**
Caveat Claude flags: the pictures are UMAP, a non-linear projection — distances there are not
distances in the model; the steering result is what rescues the geometry claim.

**THE THREE IN SEQUENCE:** P1 asks what a number IS and where addition happens → P2 asks whether
that is general (yes, and your tool decides your answer) → P3 asks why multi-digit fails
(continuous carry, noisily rounded). **Every model in all three uses classic attention in every
block.** That is the dashed box in the artifact and it is where our question lives.
**Three gaps, honestly sized:** (1) nobody has isolated the step that actually adds — the real
prize, two papers three years apart stop at the same wall; (2) nobody has fitted a helix to a
digit-split model, though P1 says such models "must use additional algorithms"; (3) ours — does
any of it transfer to a hybrid linear-attention model, and does the mixer type change where the
number is assembled. (3) is a transfer test, not a discovery, and "yes it transfers" is thin.

---

## TRIG-01 — can it evaluate sin/cos/tan in degrees, to 5 decimals, with no tools?
**AG's design, his words:** sweep the whole cycle in degrees, answers to exactly five decimal
places, "let's use a calculator as well for the ground truth because mathematics is verifiable
rewards". He named the landmark angles he can do himself — 0/30/45/60/90, ASTC — and the thing he
cannot: "I can never do this in between things like 37 degree." That is the question. The model is
asked 37 as readily as 30.

His four choices, confirmed: 1° sweep (not 5°); thinking both off and on; keep the poles and see
what it says; greedy. Plus: cache activations, because "when we are running forward pass
inference, we are anyway paying the GPU cost."

**Script:** `mathematics/pipeline/pod/trig_sweep.py`. Ground truth from Python `math` (double
precision), poles returned as `None` for tan(90°) and tan(270°). Three things scored separately:
whether a number parses at all, the absolute error, and format compliance (exactly 5 dp — a
different kind of failure from a wrong value, and worth not conflating).

### Rule-7 preflight — answered from READING the code, before the pod exists
1. **Timestamped progress line per unit of work?** Yes — one line per batch with count, rate and
   ETA (`trig_sweep.py`, the `log(f"  {n_done}/{len(tasks)}...")` call at the end of the batch loop).
2. **Results to disk incrementally?** Yes — one JSON line per ask, `f.flush()` after every batch,
   append mode. A `--resume` is not needed as a flag: the run reads the existing jsonl at startup
   and skips `(fn, deg)` pairs already on disk, so a kill costs at most one batch.
3. **Does working memory grow, and what is the peak?** Two growths, both bounded and computed:
   - `acts_store` holds one array per batch, `[48, 65, 5120]` float16 = **32 MB/batch**; 23 batches
     = **0.7 GB host RAM**, written once at the end. The pod has 200+ GB. Fine.
   - VRAM: 27B bf16 ≈ 54 GB weights, plus KV for batch×(prompt+new). Condition `off`:
     48 × ~64 tokens, negligible. Condition `on`: 24 × ~2,100 tokens ≈ 1.5 GB. Peak ~57 GB against
     an **80 GB** A100. Headroom is real but not luxurious — hence batch 24 for `on`, not 48.
4. **If killed at 80%, what survives?** Every answer up to the last flush, and the run resumes from
   them. **The activations do NOT** — they are written once at the end. Accepted deliberately:
   they are a free by-product, condition `off` is ~8 minutes, and re-running it is cheaper than
   the complexity of streaming 65-layer tensors. Named here so it is a decision, not a surprise.
   Resumed runs write `acts_<tag>_<k>.npy` with a fresh k, so nothing already paid for is
   overwritten.

**Cost line.** A100 80 GB, $1.19/hr (PCIe) — 1,080 asks × 24 new tokens + 216 asks × ≤2,048 new
tokens ≈ 0.47 M generated tokens. Restore + load ~5 min · condition `off` ~8 min · condition `on`
~30 min (the long chains dominate) · ~5 min slack = **~50 min ≈ $1.00**, call it **$1.40** if
thinking runs long. Balance at start **$50.59**.

### Bug caught by the dry run — and it would have been silent
The first version hooked every decoder block and read `h[:, -1, :]`. That is correct for a single
forward pass and **wrong inside `generate()`**, which runs one forward pass per new token: the
hook kept firing and the stored vector ended up being the *last generated* token, not the last
*prompt* token. Caught by comparing the saved array against a separate prompt-only forward pass:
**max abs difference 3.08** — a different vector, not noise. Nothing in the run would have looked
wrong; the shapes, the log lines and the results table were all correct.
**Fix:** capture only the prefill pass (`seq_len > 1`), armed per batch, with an assertion that
exactly 65 tensors were caught. Re-verified the same way: **max abs difference 0.0010–0.0016**
across three spot-checked rows, which is float16 storage rounding on values whose norm reaches
~400. This is the check the run would not have done for itself.

### Laptop dry run (Qwen3.5-0.8B, CPU, step 90°, condition off) — harness PASSED
12/12 parsed, 7/12 exactly five decimal places, activations `(12, 25, 1024)`, resume path exercised.
The *answers* were nonsense — sin(90°) returned 0.00000, cos(180°) returned 0.99999 — but that is
a 0.8B model, and the point of the dry run is the harness, not the mathematics. It does establish
one thing worth stating in advance: **a model can emit a perfectly formatted five-decimal number
that is wrong by 2.0.** Format compliance is not competence, which is exactly why the two are
scored separately.

### Pod, and a cost correction
`e59ekvvxkjy16s`, A100 80 GB PCIe, US-MO-1, CUDA 13.0, driver 580.159.04. Image
`ghcr.io/adithyag73/interp-pod:torch28-tf5170` — torch 2.8.0+cu128, transformers 5.17.0,
**zero installs**, SSH reachable and GPU readable within ~10 min of create (the direct-SSH block
is `null` at create time and appears a minute or two later; use it, not the proxy, which needs a
PTY that a non-interactive `ssh` will not give).
**The catalog price was the wrong number to budget from.** `get-capacity` says $1.19/hr for A100
PCIe; the created pod reports **`cost: 1.59`/hr** — the 100 GB container disk is billed on top of
the GPU. My preflight cost line said ~$1.00 on the $1.19 figure; at the real rate 50 min is
**~$1.33**. Inside the $1.40 ceiling, but quote `cost` from the create response in future, not the
catalog.
**Watchdog flaw noted, not fixed mid-run:** `pod_watchdog.sh` hardcodes `$0.44/hr` in its alarm
line, so it understates this pod's burn by 3.6×. It only prints — it never kills — so behaviour is
unaffected; fix the constant to read the real rate afterwards.
**zsh trap hit twice in one session,** despite already being in the journal: `SSH="ssh -i k ... host"`
then `$SSH "cmd"` makes zsh treat the whole string as ONE command name. Helper scripts
(`pod.sh`/`put.sh`/`get.sh`) now wrap ssh and scp; never a variable.

### Two checks placed BEFORE the paid sweep, both testing assumptions rather than code
1. **`check_template.py` — is `enable_thinking` honoured?** If the flag were silently ignored the
   two conditions would be the same experiment and the whole contrast would be fiction. Asserts
   the two rendered prompts differ; the run stops if it fails.
2. **`check_padding.py` — is the last-prompt-token state invariant to LEFT PADDING?** This one is
   specific to the architecture and is the check I nearly skipped. Batches are left-padded so the
   read position is index −1, which is harmless under classic attention because pad tokens are
   masked out. But **3 of every 4 blocks here are Gated DeltaNet**, a linear-attention mixer
   carrying a *recurrent state written token by token* — it ignores pad tokens only if the
   implementation explicitly honours the mask. If it does not, the answer to sin(37°) would depend
   on which other prompts shared its batch, and the sweep would not be reproducible at all.
   Test: the same prompt alone vs. in a batch behind a much longer one (forcing heavy padding);
   compare the greedy next token and the residual at every layer.

### The template check passed — and found a CONFOUND in AG's condition B
`enable_thinking` is honoured (34 prompt tokens off, 74 on), so the two conditions are genuinely
different. But the difference is **not only the thinking block**. The rendered prompts:

- **off:** `<|im_start|>user\n…<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`
  — an *empty, pre-closed* think block, so the model is committed to not deliberating.
- **on:** prepends a **system message that was never asked for**:
  *"Reasoning effort is set to xhigh. Please think carefully through the task, validate key
  assumptions, consider plausible alternatives, and prioritize correctness, consistency, and
  clarity in the final answer."* — then opens `<think>`.

**So condition B = chain of thought PLUS an instruction to be careful.** Any B-vs-A difference
cannot be attributed to the scratchpad alone; the system prompt is a second live variable. This
is Qwen's chat template doing it, not our code — it would have been invisible had the check only
asserted "the strings differ" instead of printing them.
**Not silently fixed.** AG asked for "thinking on and off", and on/off *as the model ships it* is
the honest reading of that request — it is also the comparison a user would actually experience.
Recorded as a limitation. If isolating the scratchpad matters, condition B needs a re-run with
the system message stripped, which is a third condition and AG's call, not mine.

### Padding check: NOT INVARIANT — and this is a real finding, not a nuisance
    prompt alone 34 tokens, in batch 387  ->  353 pad tokens prepended
    greedy next token:  alone '0'   padded '0'        (same)
    residual max|diff| = 2.73438   mean 0.016434      (largest at layer 63)
    VERDICT: NOT INVARIANT
The **generated token was unchanged**, but the residual stream at the last prompt token is
**not** invariant to how much left padding the prompt received. On a hybrid model this is the
expected direction of failure — a Gated DeltaNet block carries a recurrent state written token
by token, so pad tokens can enter it in a way masked softmax attention would not allow. Claude's
inference, not a sourced claim: the divergence growing to a maximum at the deepest layer (63) is
consistent with a small early perturbation accumulating through the stack.
**Honest scoping of what this does and does not threaten.** The test was deliberately extreme:
353 pad tokens. In the actual sweep, batches are consecutive tasks whose prompts differ by at
most ~2 tokens ("sin(7 degrees)" vs "sin(357 degrees)"), so the real padding spread is ~2, not
353. So this does NOT show the sweep is contaminated — it shows the assumption cannot be taken
for granted and must be measured at the spread that actually occurred.
**Therefore a control was added and run: the same asks at `--batch 1`** (zero padding), compared
answer-for-answer against the batched run. That is the decisive check for the RESULTS. The
activations remain the weaker case: they are the quantity the check showed to be sensitive.

### CONDITION A RESULT — thinking off, 1080 asks in 51 s at 21 asks/s
Far faster than the ~8 min estimated. Activations `(1080, 65, 5120)` float16, all 65 depths.
**Parsed 1078/1078. Exactly five decimals 99.6%.** Format compliance is essentially perfect and,
as the dry run warned, says nothing about correctness.

**Exact to 5 decimal places: 68.9% overall** — sin 86.9%, cos 80.8%, **tan 38.8%**.
AG's memorised angles (deg mod 90 in {0,30,45,60}, n=46): **100%**. The in-between angles he said
he could never do: **67.5%**. So on a single forward pass with no scratchpad it produces
sin(37°) = 0.60182 correct to five decimals.

**The quadrant gradient is the clean structure in the data:**

| | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| sin | 98.9% | 96.7% | 76.7% | 75.6% |
| cos | 96.7% | 97.8% | 82.2% | 46.7% |
| tan | 92.2% | 39.3% | 6.7% | 16.9% |

**A hypothesis of Claude's that FAILED when tested, recorded because it nearly got reported.**
The eight largest errors are near-perfect sign flips — tan(269°) answered −57.29 against a true
+57.28996, tan(268°) answered −28.63625 against +28.63625 — which suggested "computes the
reference-angle magnitude correctly, applies the quadrant sign rule wrongly". Tested across all
1078: **sign-only errors are 9.6% of errors**, and magnitude-correct (71.9%) beats signed-correct
(68.9%) by only 3 points. The story holds in exactly two cells (Q4 cos: 19 sign-only errors;
Q3 tan: 12) and nowhere else. **The largest errors are sign flips; the typical error is not.**
Reading the tail and generalising from it was the mistake.

**tan degrades toward its poles**, as the magnitude grows: 12.5% exact within 5° of a pole,
32.5% at 5–45°, 46.1% at 45–90°. Five decimal places on tan(89°)=57.28996 demands seven
significant figures, against six for a sine — so part of "tan is worse" is the format demanding
more precision, not the function being harder. Not disentangled here.
**The poles themselves, asked directly:** tan(90°) → `Undefined` (correct); tan(270°) → `0.00000`
(wrong, and confidently formatted). It knows the concept at one pole and not the other.
**All 4 format failures are the same case:** tan(89/269/271/279) answered `-57.290` — three
decimals, and the same value emitted for three different angles.

### TRIG-01b queued — the result above cannot distinguish recall from computation
A 5-decimal trig table over integer degrees is ~1000 numbers and is thoroughly represented on the
web. **"97% on integer degrees" is equally consistent with looking the value up and with
computing it,** and no amount of integer-degree data separates them. So vary the input where a
table does not reach, holding the target value fixed where possible:
`int` (control) · `half` d+0.5 · `dec2` d+0.ab (untabulated) · `over360` d+360 · `neg` −d.
**`over360` and `neg` are the diagnostic ones**: the correct answer is the same real number as
the control, so any drop isolates the reduction/symmetry step rather than the function. A drop on
`dec2` alone is weaker — a fractional input may be harder to compute as well as harder to recall.
A `dec2` score near the control would be strong evidence for genuine computation.

### Preflight answer #3 was WRONG by 23 GB — recorded because the checklist exists to be scored
The preflight estimated condition B's peak at **~57 GB** ("48 × ~64 tokens negligible; 24 × ~2,100
tokens ≈ 1.5 GB KV"). Measured during the run: **80.4 GB of 81.9 GB, i.e. 98% of the card.** The
KV cache for batch 24 × 2048 tokens across 64 layers is far larger than the 1.5 GB guessed; the
hybrid's full-attention blocks grow their KV with sequence length while only the DeltaNet blocks
hold fixed state, and that split was not accounted for.
**It did not cost anything, and the reason is preflight answer #2, not #3:** results flush to
jsonl every batch and the run resumes from disk, so an OOM would have cost one batch of 24. The
safety net held where the estimate failed. Had the run written once at the end — the pattern rule
7 exists to catch — an OOM at 90% would have destroyed everything.
**Lesson for the next preflight:** compute KV explicitly as
`2 × layers × kv_heads × head_dim × seq × batch × dtype_bytes`, do not eyeball it, and on a hybrid
count only the full-attention layers' growth.

### CONDITION B — thinking on. The traces show the mechanism, and it is not one mechanism.
**~99% exact to 5 dp** against condition A's 68.9% (final n pending; the paired same-angle
comparison is the number to quote, not these two marginals, because B ran a 5° grid).

**What it actually does, read off the traces — this is the part that answers AG's question.**
Two different behaviours, in the same condition:

- **sin(35°) — no computation at all.** Full trace, 78 tokens: *"sin 35° = 0.573576436... rounded
  5 decimals = 0.57358."* It states the value to nine decimal places and rounds it. Nothing is
  derived. Claude's inference, not a sourced claim: this is what retrieval looks like from the
  outside, and it is indistinguishable in the text from having known it.
- **tan(95°) — symbolic manipulation over a recalled anchor.** It writes
  `tan(95°) = tan(90+5) = -cot(5°)`, then `= -1/tan(5°)`, recalls `tan 5° ≈ 0.087488663525924`,
  takes the reciprocal to get `-11.430052302761`, and checks the sixth decimal before rounding.
  It also starts a Taylor series `tan x = x + x³/3 + 2x⁵/15 + 17x⁷/315` and converts 5° to
  π/36 = 0.08726646... to 50 digits — i.e. it has a genuine numerical method available and
  reaches for it when the value is not directly to hand.

**So "how does it do trigonometry" has no single answer even within one model and one prompt
format.** It recalls where it can and manipulates where it cannot, and the two are mixed inside a
single response. That is a finding about the *scratchpad* mechanism, not the forward pass.

**A measurement artifact caught before it was scored as a maths error.** B's only wrong answer
was tan(95°) — the trace above, which had already reached the correct −11.430052, then **hit the
2048-token budget** mid-verification and emitted `3.04617419786e-09`. That is truncation.
`trig_analysis.py` now flags any generation with `n_gen_tokens >= max_new_tokens`, reports them
separately and **excludes them from accuracy**. This is the same failure the journal records from
QWEN-02 (2026-09-15), where the paper's 400-token budget truncated Qwen and the answer regex then
read running commentary as the verdict. Second sighting; now handled in code.
**Condition A checked for the same artifact and is clean:** max 10 generated tokens against a
budget of 24, **0/1080 truncated**. So the A-vs-B gap is not a budget artifact.

### CONDITION B FINAL, and the paired number that matters
**213 of 213 exact to 5 dp = 100.0%** (1 of 214 excluded as truncated: tan(95°), see above).
**Paired on the same 213 asks: thinking off 87.8% → thinking on 100.0%**, median 9 generated
tokens → 190. Quote the PAIRED pair, not the marginals: condition B ran a 5° grid, and 5° grids
are not a random sample of angles (see the tabulation result below) — A scores 87.8% on that grid
against 68.9% on the full 1° sweep.
**Confound restated:** B also carries the unrequested "Reasoning effort xhigh / think carefully"
system message, so this is chain-of-thought PLUS an instruction to be careful, not CoT alone.
**Poles under thinking went strange:** tan(90°) → a 1000-digit string of 9s; tan(270°) → `-1.6`.
Condition A answered `Undefined` at tan(90°). So thinking made the finite values perfect and the
undefined ones worse — the 9s look like an attempt to express "infinity" inside a format that
demands a number. Claude's reading, not a tested claim.

### THE TABULATION GRADIENT — strongest result of the session, and it cost nothing
Prompted by noticing A scored 87.8% on the 5° grid but 68.9% over all integers. Split condition
A's 1078 finite asks by how likely the angle is to appear in a printed table:

| angle class | n | exact to 5dp | 95% CI (Wilson) |
|---|---|---|---|
| multiple of 15 | 70 | **94.3%** | [86.2, 97.8] |
| multiple of 5, not 15 | 144 | **84.7%** | [78.0, 89.7] |
| even, not multiple of 5 | 432 | **67.6%** | [63.0, 71.8] |
| odd, not multiple of 5 | 432 | **60.9%** | [56.2, 65.4] |

**Multiples of 5: 87.9% vs 64.2% for the rest — a 23.6 point gap, two-proportion z = 6.68.**
Per function: sin +16.3, cos +17.0, **tan +37.0**.

**Why this is evidence and not just a correlation.** Under a COMPUTATION account there is no
mechanism that makes sin(35°) easier than sin(34°): the arithmetic is identical, both angles
tokenize as two digits, and neither is nearer a pole or a special value. Under a RETRIEVAL
account the mechanism is immediate — 35° appears in tables and in text far more often than 34°,
so the value is more available. The **monotone ordering** 15 > 5 > even > odd is the ordering of
tabulation frequency, and the effect is largest for tan, whose values are least memorable and most
often looked up. Together with the sin(35°) trace (states nine decimals, derives nothing), the
reading Claude offers is: **on integer degrees this is substantially recall, not calculation.**
**What would falsify it:** if `dec2` (untabulated, two decimal places) scores near the control in
TRIG-01b, recall cannot be the main story. That run is in flight.
**Honest limit:** "appears in tables" and "appears in text at all" are the same variable here and
this design does not separate them; the even-vs-odd gap (67.6 vs 60.9) has no tabulation story
Claude can defend and is unexplained.

### TRIG-01b RESULT — two dissociable mechanisms, and it CORRECTS the reading above
1,347 asks, 90 Q1 base angles x 5 input variants x 3 functions, thinking off.

| variant | n | exact 5dp | ≤1e-4 | ≤1e-3 | ≤0.1 | median \|err\| when wrong |
|---|---|---|---|---|---|---|
| `int` (control) | 270 | **95.9%** | 97.4% | 97.8% | 97.8% | 0.10 |
| `neg` (−d) | 267 | **98.5%** | 99.6% | 100.0% | 100.0% | 0.00002 |
| `half` (d+0.5) | 270 | 70.7% | 82.2% | 87.0% | 93.3% | 0.0003 |
| `dec2` (d+0.ab) | 270 | **3.0%** | 30.7% | **75.6%** | 91.5% | 0.0002 |
| `over360` (d+360) | 270 | **4.4%** | 4.4% | 4.4% | 7.8% | **1.38** |

**`dec2` and `over360` both collapse at 5 dp, and they are NOT the same failure.** `dec2` is
still 75.6% correct at 1e-3 — the value is right to about three decimals and misses the fifth
(median error 21 units of the last demanded digit). `over360` is 4.4% at EVERY tolerance out to
0.1 — the value itself is wrong.

**The dissociation, tested directly.** For each wrong answer, ask whether it is nonetheless an
exact 5 dp value of the same function at *some other* integer angle. A computation that goes
wrong lands on an arbitrary number; a lookup that goes wrong lands on a crisp entry for the wrong
key:

| variant | n wrong | wrong BUT an exact entry elsewhere |
|---|---|---|
| `over360` | 258 | **221 (85.7%)** |
| `dec2` | 262 | 13 (5.0%) |
| `half` | 79 | 4 (5.1%) |
| `neg` | 4 | 0 |

Examples: sin(361°) → **0.62932 = sin(39°) exactly**; cos(362°) → cos(−6°); sin(364°) → sin(84°);
tan(365°) and tan(366°) → 1.00000 = tan(45°). It is not computing and missing — it is fetching the
wrong entry.

**CONCLUSION, and it REVISES what Claude wrote an hour earlier.** After the tabulation gradient
the reading offered was "on integer degrees this is substantially recall, not calculation." That
was too simple, and `dec2` is the evidence that changed it: if the model could only look values
up, sin(37.87°) should produce garbage or a neighbouring table entry — instead it produces a value
good to three decimals that matches no table entry. **Both mechanisms are present and they are
separable:**
- **Retrieval** — exact to 5 dp, available for tabulated angles, and the source of the
  tabulation gradient. Its failure signature is a crisp value for the wrong key (85.7%).
- **Approximate computation** — available for any angle, worth roughly three decimals, never five.
  Its signature is a near-miss matching no table entry (95%).
- **Symmetry `sin(−d) = −sin(d)` is applied essentially perfectly (98.5%, 100% at 1e-3)**, so the
  model does hold and apply at least one exact identity.
- **Reduction mod 360 is absent** (4.4%), even though the correct answer is a value it retrieves
  95.9% of the time from the unshifted angle. It never gets as far as reducing.
**So the five-decimal performance on integer degrees is retrieval; the model's own calculator is
worth about three decimals.** The prompt's "exactly 5 decimal places" is what makes the two
separable at all — at a 1e-3 tolerance the distinction would have been invisible.

### PAD CONTROL — the answers survive batching; the activations are the weaker artifact
90 asks run BOTH at `--batch 1` (zero padding) and inside a batch of 48 (left-padded):

- **identical raw answer string: 87/90 = 96.7%**
- **exact-to-5dp: 77.8% at batch 1 vs 77.8% at batch 48** — aggregate accuracy identical
- the 3 that differ are **all `tan`, and all were already wrong in both conditions**:
  tan(96°) −16.38197 vs −14.30067 (truth −9.51436); tan(228°) −0.90040 vs −1.07868
  (truth +1.11061); tan(324°) −0.50953 vs −0.64941 (truth −0.72654)

**Reading:** left padding perturbs the computation enough to change ~3% of answers, and it does so
exactly where the model is already unstable — large-magnitude tan values it gets wrong anyway. No
case flipped from right to wrong. **The headline accuracy numbers are safe.**
**The activations are the artifact that stays caveated:** the padding check measured the last-token
residual at 2.73 max divergence under extreme (353-token) padding, and nothing here rehabilitates
that. Any probe on `acts_off_step1_0.npy` should be treated as carrying batch-composition noise;
the honest fix if a probe result ever matters is to re-extract at `--batch 1`, which for 1080 asks
costs about 90 seconds of A100 time.

### Run closed
Pod `e59ekvvxkjy16s` **terminated, zero pods confirmed**; watchdog stopped. A100 80 GB PCIe,
driver 580.159.04, torch 2.8.0+cu128, transformers 5.17.0. Artifacts in
`mathematics/runs/2026-09-22_trig/`: four results jsonl (A 1080 · B 216 · variants 1350 ·
batch-1 control 90), `acts_off_step1_0.npy` (1080, 65, 5120) float16 718 MB + index,
`pip_freeze.txt`, `gpu.txt`, `followups.log`, `watchdog.log`, error-vs-angle figure.
`.gitignore` now excludes `mathematics/runs/**/*.npy` — 718 MB is over GitHub's 100 MB file limit
and would have failed the push (verified with `git check-ignore`).

### PROBE-TRIG — is the answer in the residual stream BEFORE it speaks? Yes, from ~depth 52.
Free, CPU, on the cached `acts_off_step1_0.npy` (last PROMPT token, all 65 depths, 1078 usable).
Target: the tenths digit of the answer (first digit after the point), 5 random 75/25 splits, own
multinomial logistic regression (NOT a published probe — not comparable to paper numbers).

**Baseline correction, Claude's error in the script:** it printed "chance = 10.0%", which
understates the bar. The tenths digit is NOT uniform — digit 9 takes 222 of 1078 because |sin| and
|cos| pile up near 1.0. **The honest baseline is the majority-class rate, 20.6%** ("always say 9").

| depth | TRUE digit | SAID digit | shuffled |
|---|---|---|---|
| 0 | 20.8% | 21.9% | 21.3% |
| 8 | 28.8% | 31.4% | 11.3% |
| 24 | 24.9% | 26.4% | 12.2% |
| 44 | 35.3% | 38.4% | 10.1% |
| 48 | 40.1% | 43.4% | 9.3% |
| **52** | **55.3%** | **57.3%** | 12.5% |
| **56** | **62.6%** | **63.9%** | 11.9% |
| 60 | 64.4% | 63.5% | 11.1% |
| 64 | 61.5% | 61.4% | 10.5% |

- **At depth 0 the probe is at the majority-class rate (20.8/21.9 vs 20.6)** — i.e. no information.
  Correct: the embedding of the last prompt token ("places." etc.) cannot know the answer.
- **The answer becomes linearly readable in the last third, jumping 48 → 56 (40% → 63%).**
  At depth 56 that is **3.1× the majority baseline**. So the value is largely determined in the
  forward pass, before a single character is emitted — it is not being assembled digit by digit
  during generation.
- **TRUE and SAID track each other closely** (62.6 vs 63.9 at depth 56), which is expected rather
  than informative: the two digits agree on **80.3%** of rows, so the probe cannot separate "knew
  the right answer" from "was going to say this" on this data. The wrong-only subset is too small
  to train on, so the tempting claim — it knew better than it said — is NOT tested here.
- **Shuffled control sits at 9–12% at every depth past 0**, below the majority rate, because with
  destroyed labels the overfit weights swamp the bias term rather than falling back to "guess 9".
  It does its job (no signal) but is a weaker control than a majority-class predictor.

**CAVEAT that limits all of the above:** the pad check showed this exact quantity — the last-prompt
-token residual — is NOT invariant to batch padding (max 2.73 divergence under extreme padding).
These activations were captured inside batches of 48. The accuracy trend across depth is very
unlikely to be an artifact of that, but any number here should be re-measured at `--batch 1`
(~90 s of A100) before it is load-bearing in a write-up.
