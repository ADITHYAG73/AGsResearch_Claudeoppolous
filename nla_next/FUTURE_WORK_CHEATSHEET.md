# NLA paper — Future Work, as a cheat sheet

Source: https://transformer-circuits.pub/2026/nla/ § Future Work, re-read from the live page
2026-09-08 (629 words; one paragraph + three "usefulness" bullets + three "other" bullets).
Text in quotes is theirs. Everything after "→" is my reading, not theirs.

Legend: **[TRAIN]** needs the AV/AR RL loop · **[INFER]** runs on a released NLA ·
**[HAVE]** runs on data already in this repo.

---

## 1. Toward general activation language models (ALMs)

- Framing: "models that translate between natural language and activation space." AV = reader
  (activation → text), AR = writer (text → activation).
- Ambition: "train each side as a general-purpose tool rather than for reconstruction alone."
- **Read side (AV)** — train jointly on many activation-to-text tasks, reconstruction being
  "one objective among several":
  - answering questions about activations
  - inferring user characteristics
  - predicting the outcome of patching experiments
- **Write side (AR)** — "produce steering vectors and probes from natural language
  descriptions", i.e. a general text-to-activation interface.
- End state they imagine: ALMs as "the primary way interpretability researchers interact with
  model internals" — what a token represents, which earlier tokens shaped it, how a steering
  vector would change behaviour, what a patching experiment would reveal — "without running
  the underlying experiment."
  → **[TRAIN]**, and at frontier scale. Not a 20-hour project. The *evaluation* of such
  claims (does the AR's steering vector actually steer?) is [INFER] — see §5.

## 2. Improving the usefulness of NLAs — three named difficulties

"reliability (distinguishing real claims from hallucinated ones), legibility (parsing the
output), and cost (running them at scale)."

### 2a. Reliability
- Idea: "penalizing statements that are factually inconsistent with the context during NLA RL."
- Their caution: no ground truth for what an activation holds, so they are "wary of generally
  constraining NLA explanations" — but claims *about the context* are verifiable, so
  "penalizing obvious hallucinations seems safe."
  → **[TRAIN]**. The inference-time cousin is exactly what we built: judge + Δ. A
  small-scale RL run with this penalty is the first training project that reuses our
  judge pipeline as the reward signal.

### 2b. Legibility
- Complaints: parentheticals with quotes mid-phrase; "they often repeat the same content on
  multiple bullet points."
- Fix 1: "Claude-graded style penalties during RL." → **[TRAIN]**
- Fix 2 (architectural): "have the AR reconstruct each AV bullet independently, and include a
  loss term for the cosine similarity between per-bullet reconstructions."
  → the *loss* is [TRAIN]; the *measurement* — reconstruct each bullet alone, pairwise
  cosine — is **[HAVE]**: one AR-only pod session on our 240 explanations. This is H3
  (redundancy) in their words, and would explain the 30% of removals that improve MSE.

### 2c. Cost
- Problem: "inference cost makes it infeasible to run NLAs over every token in a production
  RL run."
- Fix 1: "train smaller models as the AV and AR using the target model's activations"; use the
  small NLA as a first pass that flags tokens for the full one. → **[TRAIN]**, but at the
  *small* end — this is the cheapest genuine training project on the list.
- Fix 2: "encouraging early results running NLAs on mean-pooled activations over transcripts
  or datasets." No numbers given. → **[INFER]**: pool over a passage, run the AV, judge the
  claims. Cheap; evaluation design is the hard part.

## 3. Other future directions

### 3a. Extending NLAs beyond activations
- "apply NLAs to other model internals, for instance gradients or LoRA adapters, to
  understand what finetunes represent."
- Two routes named: "applying activation-trained NLAs to these directly, or train on them
  specifically using a similar procedure to NLA RL."
  → route 1 **[INFER]** — but only if the finetune's base has a released NLA
  (Qwen2.5-7B-Instruct, Gemma-3-12B-IT, Gemma-3-27B-IT, Llama-3.3-70B-Instruct; Celeste's
  Qwen3.6-27B). Route 2 **[TRAIN]**. Neither has a result in the paper — "in principle".
  AG's text2cypher RL finetune is the candidate; base model still to be confirmed.

### 3b. Characterizing what NLAs cannot verbalize
- "We have speculated that some activation content may be unverbalizable, but this is
  currently untested."
- Method they suggest: find "what information is available to mechanistic techniques like
  SAEs but not to NLAs" → "map the boundaries of NLAs".
  → **[HAVE] + a public SAE.** Our NLA is Gemma-3-12B-IT layer 32; Neel's doc pairs Gemma 3
  with Gemma Scope 2. Gating check (not yet done): does Gemma Scope 2 ship a residual SAE for
  the 12B model at or near layer 32? If yes: on cached activations, which strong SAE features
  are named by no claim? This is the only paper-sized question on the list.

### 3c. Inference-time methods
- "Our current interpretability pipeline mostly uses AV outputs and discards the AR."
- Idea 1: "taking a best-of-N NLA explanation against AR reconstruction." → **[HAVE]**: K=4
  resamples + scorer + verdicts already exist. Does the lowest-MSE resample have fewer false
  claims than a random one? CPU job.
- Idea 2: "ablating individual claims within an NLA explanation to measure how much each
  contributes to reconstruction." → **done** — that was the MATS application.

---

## If training is on the table (AG, 2026-09-08: "I'm open to spending money")

Frameworks that exist, all official or from the paper's own circle:
- `kitft/natural_language_autoencoders` — Anthropic's release; the paper's own training code
  (GRPO for the AV, MSE regression for the AR, KL to init). Apache-2.0. Already forked.
- `asherps/EasyNLA` — Celeste's framework (Neel's scholar); Karvonen-style injection, GRPO via
  LoRA adapters, FVE baseline script. MIT. Cloned read-only in `reference_repos/`.
- `ceselder/nanoNLA` — predecessor, marked superseded.
The cheapest training projects on this list, in order: **2c-small-NLA** (small AV/AR on a
big model's activations), then **2a** (reliability penalty, reusing our judge as reward),
then **3a route 2** (train on LoRA weights). Each needs a completion condition sized before
a pod is started — §8a of CLAUDE.md.
