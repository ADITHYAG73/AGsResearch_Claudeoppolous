---
name: check-before-you-claim
description: Adithya's standing rule - reach for a tool before answering from knowledge; get the real traceback and the previous run's log before proposing any cause.
metadata:
  type: feedback
---

Adithya has said repeatedly that Claude reasons well but **does not reach for tools when it
has them**, and instead answers from its own knowledge. He asks for: debug logs left in
place, long jobs run in the background, and the documentation / codebase / previous run logs
actually read before a cause is proposed.

**Why:** on 2026-08-23 a RunPod AR job crashed with a bare `std::bad_alloc`. Claude proposed
three causes from knowledge in a row (deprecated `torch_dtype`, missing `accelerate`, host
RAM exhaustion), patched and re-ran after each, and all three were wrong. The actual cause -
the pod image tag was different from the one the working runs used - was recorded in
`mats_2027/experiments.md` and would have been found by one grep. Enabling `faulthandler`
showed the crash was at *import* time, not during weight loading, which refuted all three
theories at once. Cost: ~$0.23 of GPU time, an hour, and the user's momentum on a day he
started excited.

**How to apply:**
- Before naming a cause, get the real traceback (`faulthandler`), and diff against the log of
  the last run that WORKED. Both are usually free and local.
- Grep the repo's own records before reconstructing a config from memory - especially pod
  images, versions, and flags.
- Never pass a value seen only as a truncated preview (a `head -c 40` of an SSH key is not
  the key).
- Run remote/long commands with `run_in_background`; blocking the conversation while a GPU
  bills is not acceptable.
- Do not "optimise" a recorded working recipe. See [[one-experiment-at-a-time]].
