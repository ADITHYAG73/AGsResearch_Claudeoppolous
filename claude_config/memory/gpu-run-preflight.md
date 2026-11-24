---
name: gpu-run-preflight
description: "Before ANY GPU run (sampling, finetuning, RL, agentic finetuning, anything billed by the hour) read the code and flag missing progress logging, missing checkpointing and memory growth BEFORE launch — not after."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 97b40c74-fd52-4128-aca4-98f0ea8d3d71
  modified: 2026-09-14T16:48:24.775Z
---

Before launching anything on a rented GPU, Claude must read the code path that will run and
answer four questions in writing, in the journal, before the pod is created:

1. **Progress:** does it print a timestamped line per unit of work (branch, step, batch)?
   If not, wrap or patch so it does.
2. **Checkpointing:** does it write results to disk incrementally? If it holds everything in
   memory and writes once at the end, a crash or kill loses all of it. Add per-unit appends.
3. **Memory trajectory:** does working memory grow through the run (sorted-by-length batches,
   accumulating caches)? Estimate the peak against the card before choosing batch size.
4. **Kill safety:** if this is killed at 80%, what survives?

**Why:** 2026-09-14, QWEN-01. Claude read the Forking Fast `run.py` before launch, saw it
prints once per question and writes once at the end, and did not flag either. The run then
went 40+ minutes with zero visibility on a billing A40, GPU memory climbing 40→43 GB of 46
because branches are sorted shortest-prefix-first, with no way to inspect (ptrace blocked in
the container) and nothing recoverable if it OOMed. AG: "these kind of things we shud have
already flagged... red flags shud be flagged during reading only... otherwise its not going
to help us." He extended it explicitly to every future run: finetuning, RL, agentic
finetuning, "anything and everything."

**How to apply:**
- The four questions are a pre-launch checklist entry in the experiment journal. No pod
  until they are answered.
- Wrap, do not edit, third-party research code — but wrapping includes logging and
  checkpointing, not only device and prompt handling.
- Treat research-lab code (including Goodfire, Anthropic) as a record of what they ran, not
  as engineered tooling. Expect silent loops, undefined helpers, end-only writes.
- Pair with [[gpu-idle-watchdog]] (detects a stuck run) and [[check-before-you-claim]]
  (read the code, not the assumption). This memory covers the case the watchdog cannot: a run
  that is busy, healthy-looking, and unrecoverable.
