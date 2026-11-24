# Brief for Codex: a dashboard to get a FEEL for the LATENT-CUBE data

## Who it is for
AG, a researcher who has never handled this kind of data and is still learning what "activation", "neuron",
"residual stream" mean. He wants to wander through the data and build intuition. He learns backwards from concrete
examples he can poke at. Plain English on screen; define each term where it first appears. Maths in rendered form
(KaTeX from cdnjs is fine) — never raw LaTeX source and never flattened ASCII when showing formulae.

## The experiment (read latent_cube/experiments.md and latent_cube/gen_prompts.py and latent_cube/capture.py first)
AG's question: when Qwen3.5-4B is asked to *imagine* the second Taylor term of sin x (−x³/6) and does not write it,
is it present inside the model? Same question for other prompts that need a cube without saying it.
No analysis has been done yet. Do NOT try to answer the question; build the instrument for looking.

## The data (140.6 GB, private S3, READ ONLY)
Bucket posttrain-math-459653581741-ap-south-1, region ap-south-1, prefix latent_cube/capture01/
Credentials: temporary read-only ones are already in your environment (AWS_ACCESS_KEY_ID etc). Do not use any other
AWS credentials or read ~/.aws. Never write to S3.
- out/<id>__free.{json,safetensors}: model's OWN reply captured. All 201 prompts.
- out/<id>__forced.{json,safetensors}: a reply WE wrote for it (no term in it: "Okay." or "The capital of France is
  Paris."). Only imagine/hold prompts (137 prompts).
- 338 pairs total. <id> = the id in prompts.jsonl with | → __ (see the `fn` line in capture.py for the exact mapping).
- IGNORE out/ files with neither __free nor __forced in the name (superseded first run), and dryruns/.
- meta/ has code + logs.

JSON (a few KB each): id, set, group, func, kth, template, prompt, power, coef, answer (exact ground truth), variant,
chat_text, token_ids, tokens, prompt_len, captured_reply, written_reply, free_reply, free_reply_finish, positions
{instr_end, prompt_end, reply_end}, check_* numbers, saved (tensor shapes). Example for a forced hold prompt: T=52 tokens.

safetensors (200 MB to >1 GB each; read with HTTP range requests, header first — never download a whole file):
- R0 [T,2560] embedding output, F16
- R1 [32,T,2560] output of each decoder layer (residual stream, BEFORE the final norm), F16
- R2 [32,T,2560] token-mixer output, F16
- R4 [32,T,9216] MLP neuron values, F16
- R5 [32,T,2560] MLP output, F16
- logits [T,248320] F16
- attn [8,16,T,T] softmax attention of layers 3,7,...,31, F16
- S [3,24,32,128,128] DeltaNet memory at instr_end, prompt_end, reply_end, F32 (leave out of the dashboard)

Model anatomy, from config + official code: 32 layers, every 4th (3,7,...,31) full attention, the rest Gated DeltaNet.
Thinking was OFF: the prompt ends with an empty <think></think> block, so the prompt_end token is "ĊĊ". Everything
after prompt_len is reply.

## Logit lens (what AG wants explained on the page)
Take a layer's R1 vector, apply the model's final RMSNorm and output layer, as if the model stopped at that layer:
  h = x / sqrt(mean(x²) + 1e-6) × (1 + w),  logits = h · Eᵀ
w = model.language_model.norm.weight [2560]; E = model.language_model.embed_tokens.weight [248320,2560] (tied to the
output layer). Get both by range-reading the safetensors from Hugging Face Qwen/Qwen3.5-4B at revision
851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a (no token needed; the 2 shards are model.safetensors-0000{1,2}-of-00002).
MANDATORY CHECK: this formula applied to R1[31] must reproduce the saved logits; print the max difference and
stop if it is not ~0. Compute in float32 from the F16 values (bf16 was used in the model, so a tiny difference is
expected: report it, do not hide it).

## What the dashboard must let him do
1. Overview: what this data is, in plain words; counts by template/variant/function; size of everything.
2. Prompt browser: filter by function, term number, template, variant, power. For the selected one show side by side:
   the prompt, the exact ground-truth term, the model's FULL free reply, the captured reply. Every token listed.
3. Layer × token map for the selected prompt: heatmap with 32 rows and T columns of the residual-stream size
   (norm of R1), plus how much the mixer (R2) and the MLP (R5) add at each cell. Token strings on the axis.
4. Logit lens: for the selected prompt, layer × token grid; each cell shows the top token and its probability, hover
   for top 5. Also let him pick a token of interest (at least: ³ id 41776, "3" id 18, "6" id 21, "^" id 61,
   " cube" id 22997, "x" id 87, "2" id 17) and see its rank at every layer × token.
5. MLP neurons: for a selected layer × token, the top 20 most active of the 9,216 neurons with their values; for a
   selected neuron, how active it is across prompts at the same readout position.
6. Attention: for a selected attention layer and head, the T × T pattern with token labels.
7. Comparing two prompts: pick any two (default: sin x vs cos x, same term and template), and show cosine similarity
   of their residual stream at each layer at a chosen readout position (instr_end, prompt_end, or reply_end).
8. Across all prompts at one readout position and one layer: 2-D PCA scatter, coloured by power of x, function, or
   template. Hover shows the prompt.
Start the page on the prompt "tay|sin x|2|hold" (forced) so the first view is real.

## Rules, non-negotiable
- NO automatic judging of whether the model's answers are correct: no regex, no string matching, no scoring, no
  "correct/wrong" labels. Show ground truth next to the full reply; the reader judges.
- NEVER truncate: no shortened replies, no "..." previews, every token and every reply in full. If a cell is too small
  for its text, wrap it or show the rest on hover/click, never lose it. If you must subsample ANYTHING for size, state
  it on the page, say exactly what and how, and tell me.
- Never invent or simulate data. Every number comes from these files. If something is unavailable, say so on the page.
- Don't run the model. This Mac overheats: no heavy CPU loops; matrix products for the logit lens are fine if batched
  and run at low thread counts (OMP_NUM_THREADS=4). Do NOT store raw tensors on the Mac beyond a temp cache:
  scratch files go under /tmp and are deleted when you finish. Keep S3 reads under about 15 GB in total.
- Preprocess once in latent_cube/dashboard/build_data.py (rerunnable, resumes, prints a timestamped line per prompt,
  writes incrementally). Output goes to latent_cube/dashboard/site/. Round floats sensibly; store compactly.

## Deliverable
latent_cube/dashboard/site/index.html, with data files beside it, loaded by relative fetch. Static, no server code.
Total size of site/ at most 60 MB; each single file at most 15 MB, so split data per prompt and fetch it on selection.
It will be published on claude.ai, so these limits apply: external scripts ONLY from cdnjs.cloudflare.com with exact
pinned versions, stylesheets only from Google Fonts, no other network calls, no alert/confirm/prompt dialogs, no
downloads. Must work at phone width with no horizontal page scroll, in light and dark theme (define colours as CSS
variables; dark via prefers-color-scheme with :root:not([data-theme="light"]) and :root[data-theme="dark"]).
Put <title>Latent Cube Explorer</title> at the top; no doctype/html/head/body tags in index.html.

When finished, print: what you built, every check result (the logit-lens check number, any subsampling, S3 GB read,
site/ size), and anything you could not do. Say plainly what you did not verify.
