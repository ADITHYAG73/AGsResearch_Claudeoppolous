# HANDOFF — read this first (written 2026-11-24 for a fresh Claude Code session, local or cloud)

AG (Adithya Giridharan) is moving this research to scheduled cloud Claude Code sessions so it moves a little every
day (30–45 min), even on days he has no bandwidth. This file is everything a new session needs that is NOT in the
code. The long history lives in the journals; this is the map.

## 0. Before anything: how to work with AG
- Read `claude_config/global_CLAUDE.md` (his global rules, copied from `~/.claude/CLAUDE.md` on his Mac) and
  `claude_config/memory/` (what earlier sessions learned about how he works). Follow them. The most important:
  - **No sycophancy, no hallucination.** Check with tools; say what you checked and what you inferred.
  - **Never truncate** (global rule 10): no previews, no low generation caps, no conclusions from partial views.
  - **Backward learning** (`memory/backward-learning.md`): start from one concrete result he can question down to
    atomic ideas. Never hand him a forward reading plan or a long paper to read front to back.
  - **Plain-text maths in chat** (x³/6, not LaTeX source) unless it renders.
  - **Let him build ideas**; be proactive otherwise (`memory/let-him-build-the-idea.md`).
  - **Rented GPU = rule 7** (preflight in the journal before any billed run, watchdog, terminate, confirm zero
    pods, opening/closing balance in the cost ledger). Over-collect activations to S3 (`memory/overcollect-on-gpu.md`).
- AG owns the research questions and decides what a result means. Write results into the experiment's
  `experiments.md`; commit and push only what he asked for, never secrets.

## 1. The objective (AG's words, unchanged since Sept)
Understand how language models do mathematics **with no tools** — what is "in its brain". Two projects grew
out of it.

## 2. Timeline of the mathematics thread
| Date | What | Where |
|---|---|---|
| 2026-09-17→20 | Thread opened. Infra: custom pod image `ghcr.io/adithyag73/interp-pod:torch28-tf5170`, RunPod templates. First looks at Qwen3.8-27B: GSM8K smoke, attribution graphs (Neuronpedia), AG's parabola prompt, number representations (NUMBERS-01, PROBE-01). | `mathematics/experiments.md` (MATH-SMOKE-01 … WHERE THIS STANDS) |
| 2026-09-21 | Read the arithmetic-mechanism papers (Kantamneni–Tegmark helix 2502.00873 etc.). | `mathematics/experiments.md` → PAPER READING |
| 2026-09-22→24 | **TRIG-01…04**: sin/cos/tan in degrees, no tools. Thinking off ≈ recall from a memorised table (sin 361° returns sin 39° exactly); thinking on fixes range reduction, sign and rounding, but the values still come from recall. HELIX-01. | `mathematics/experiments.md`, dashboard https://claude.ai/artifact/9gFKBqkC8Je2b55pukMdpb |
| 2026-10-06 | **posttrain_math** paper project: 3,362 frozen maths questions × {alone, Python sandbox} × {thinking off, on} on Qwen3.5-4B/9B (BASE-01). | `posttrain_math/BOARD.md`, `experiments.md` |
| 2026-10-16 | Paper draft (Claude Docs): https://claude.ai/code/artifact/16e2050b-0bbe-4dde-820c-b4bf2ea87606 | `posttrain_math/experiments.md` → PAPER-01 |
| 2026-10-18 | **FT-01**: fine-tuned Qwen3.5-4B on 5,000 instruction-following rows (thinking-mode only). Sandbox + thinking fell 98.3% → 81.4% because the model stopped calling the tool (no-call 0.2% → 38.5%). Added to the paper as Section 5. | `posttrain_math/experiments.md` → FT-01 RESULTS; `OWED.md` = checks still owed |
| 2026-10-29 | **latent_cube** (current): AG's interp question — asked to *imagine* the 2nd Taylor term of sin x (−x³/6) without writing it, is it inside the model? CAPTURE-01: 201 prompts, every layer's internals for Qwen3.5-4B, 140.6 GB in S3. | `latent_cube/experiments.md` |
| 2026-10-30/31 | Two pages: what the model wrote (https://claude.ai/artifact/X1zpW9k8qaEe6fxQWatueh) and the Explorer, built by Codex (https://claude.ai/artifact/Vhkj2EQGXtyufVUcUbCKYe). Claude's independent check: 132 comparisons, 0 mismatches. | `latent_cube/behaviour/`, `latent_cube/dashboard/` (STATUS.md, BRIEF.md) |
| 2026-11-01/02 | First result + its flaw (below). Found Anthropic's J-space paper as the right tool. | this file, §3 |

## 3. Where the current work stands (latent_cube)
**Behaviour (read in full by Claude, not scored automatically):** the model never obeys "imagine, do not write
it" on Taylor prompts — it writes the term (often then says "As requested, I will not write it out").
It often works out loud (writes the series, counts terms). Its answer to one question changes with wording:
sin x 2nd term → −x²/2 asked directly, 0 when told to imagine, −x³/6 in the Paris prompt. Some "wrong" answers
are a counting convention (constant term counted as the first term).

**First internal result (logit lens):** at the moment just before the reply, "³", "3", "^", "6" rank the same
for sin x and cos x (e.g. best rank of ³: sin 815, cos 789, out of 248,320). **AG's objection, accepted:** the logit
lens only asks about the very NEXT token ("The"), so it cannot see a term planned for later. So: no evidence
either way yet.

**Next step (agreed, not done):**
1. AG reads ONE section of Anthropic's "A global workspace in language models" (2026-07-06,
   http://transformer-circuits.pub/2026/workspace/index.html): **"The J-space is subject to directed modulation"**
   (copy a sentence while computing 3²−2 silently — his experiment, done right). Then, only if he wants,
   "Comparison to Related Techniques" and "The J-space mediates internal reasoning". Go one example at a time.
2. Claude verifies the pre-fitted **J-lens for Qwen3.5-4B**: a community mirror says it copies Neuronpedia's
   (MIT, n=1000 texts): https://huggingface.co/JamesBentley/jacobian-lens-mirror — NOT yet verified against
   Neuronpedia's own repo, nor against our model revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
3. Why the J-lens can see past the next token (from the paper's Methods, read 2026-11-02): J_ℓ is the average,
   over prompts and over positions t and every LATER position t′ ≥ t, of ∂h_final,t′ / ∂h_ℓ,t — how a nudge at layer ℓ
   now changes the model's final state at the current AND all later tokens. The logit lens has no J_ℓ, so it only reads
   the next-token prediction. If the lens holds: J-lens = softmax(W_U · norm(J_ℓ · h)) — one extra matrix on the residual stream we already have in
   S3 (R1). Rerun the sin-vs-cos comparison at the three readouts. No GPU needed for reading.
4. Later options AG listed: composition prompts (sin(x/2) → −x³/48) as the recall-vs-composing test; a "cube
   direction" from mean differences; patching on a GPU only if something shows up; a better prompt borrowed from
   the J-space paper (copy an unrelated sentence) so the model cannot blurt the answer.

## 4. Where things live
- **GitHub (this repo, public):** all code, journals, prompts, small results, the dashboard site.
- **S3 (private, needs AWS credentials — NOT in this repo):** `s3://posttrain-math-459653581741-ap-south-1/`
  - `latent_cube/capture01/out/<id>__free|forced.{safetensors,json}` — the 140.6 GB capture (format in
    `latent_cube/capture.py` docstring and `latent_cube/dashboard/BRIEF.md`). Files without __free/__forced are a
    superseded first run.
  - `posttrain_math/…` — baseline and FT-01 runs, checkpoints s025/s050/s100.
- **Compute (2026-11-24): Modal or RunPod from cloud sessions — both verified working (§6). Which to default to is AG's call.**
  - Why (AG, after reading https://www.runpod.io/articles/comparison/runpod-vs-modal, 2026-11-24): a scheduled
    cloud session should launch GPU work as one `modal run` of a Python function that ends by itself, not
    create/SSH/watch/delete a pod. The post itself says that for "a set of Python functions that need to run on
    assorted hardware on assorted schedules … the recommendation is to use [Modal]".
  - Prices (Modal's own pricing page, read 2026-11-24): A100 80GB $0.000694/s ≈ $2.50/hr, A100 40GB ≈ $2.10/hr,
    H100 ≈ $3.95/hr, H200 ≈ $4.54/hr, L40S ≈ $1.95/hr; Starter plan $0/month with **$30/month free compute**.
    Base rates are preemptible (guaranteed non-preemptible = 3x, region pinning 1.15–1.75x), so every job must
    **checkpoint and resume** (latent_cube/capture.py already resumes per prompt). Volumes $0.09/GiB/month.
    For comparison, RunPod A100 SXM cost us $1.79/hr (Oct 2026); H200 $5.29/hr.
  - Docs read: functions run HF transformers on a GPU via `@app.function(gpu=..., image=...)`; timeout 1 s–24 h
    (default 300 s); `modal shell` gives an interactive shell (not SSH). Token auth for a headless cloud VM
    (`modal token set` / env vars `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET`, per modal.com/docs/reference/modal.config).
  - **Rules for a scheduled session until AG sets a budget:** stay within the free $30/month; write rule-7
    preflight (incl. a cost line) into the journal BEFORE any `modal run`; smoke-test on the smallest GPU/input
    first; send results to S3 (or a Modal Volume) incrementally; log estimated vs actual cost in the experiment's
    cost ledger; anything above the free credit needs AG's explicit go (ask in the daily summary).
  - RunPod (image `ghcr.io/adithyag73/interp-pod:torch28-tf5170`) is ALSO usable from cloud sessions since
    2026-11-24 via the RunPod connector (see §6): `exec-pod` runs commands inside a pod without SSH.
- **Not in the repo on purpose:** `.env`, API keys, AWS/RunPod credentials, raw tensors (>100 MB, regenerable),
  `latent_cube/dashboard/site_pub/` (base64 copy for publishing, made by `make_publish.py`).

## 5. Open housekeeping
- `posttrain_math/OWED.md`: AG's hand-checks still owed for the paper.
- (Closed 2026-11-24: HF token replaced with a new one `cloud-claude`; global volume `like_aquamarine_bandicoot`
  deleted by AG — list-global-volumes returns empty.)

## 6. Cloud session setup (verified 2026-11-24, each by a real call from a cloud session)
| Service | How it is reached | Check that passed |
|---|---|---|
| GitHub | repo access + push to the session branch | pushed `von_neumann/ecstatic-rubin-q4en3l` |
| AWS S3 | claude.ai **AWS connector** (`run_script`, `get_presigned_url`); not the CLI | listed `latent_cube/capture01/out/tay__sin x__2__hold*` |
| RunPod | claude.ai **custom connector** `https://mcp.runpod.io/` (OAuth) — pods, volumes, balance, `exec-pod` | list-pods empty, balance $88.10 |
| Hugging Face | network secret on host `huggingface.co` (proxy injects it; no env var exists) | whoami → token `cloud-claude`; Qwen3.5-4B config 200 |
| Modal | env vars `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` in the Default environment; **must** `pip install 'modal[api-proxy-support]'` | token verified; `modal app list` |
- **Modal gotcha (cost an hour on day 1):** all outbound traffic goes through a local HTTP proxy. The Modal client
  supports proxies but needs the optional `python-socks` package; without it `modal` hangs silently (debug log:
  "A proxy is configured … but the 'python-socks' package is not installed"). Fix: install the extra above.
  Claude first misdiagnosed this as "gRPC is blocked by the proxy" from one README line — check with
  `MODAL_LOGLEVEL=DEBUG` before concluding anything.
- If the AWS connector says "sign in again", AG disconnects and reconnects it at claude.ai/customize/connectors.
- Claude in Chrome is not available to cloud sessions (it lives in AG's Chrome on his Mac).
- Standing instructions for these daily sessions: `claude_config/memory/autonomous-daily-assistant.md`.
