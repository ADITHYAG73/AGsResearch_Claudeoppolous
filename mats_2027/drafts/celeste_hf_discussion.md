# DRAFT — Hugging Face discussion post (NOT SENT)
Target: https://huggingface.co/ceselder/qwen3.6-27b-nla-rl  → Community → New discussion
Rewrite in my own voice before sending. Drop Q3 if I can trace it myself.

---

**Title:** Licence terms, and a question about `mse_scale`

Hi — thanks for releasing this. The `nla_meta.yaml` sidecar and the
every-100-steps checkpoint layout made it straightforward to get oriented, and
the note recommending step 300 over the final checkpoint was useful.

Two questions, the first one blocking for me:

**1. Licence.** The model card lists the licence as "other". I'd like to run
inference-only experiments with the AV and AR (no training, no redistribution of
weights) and publish the results as a write-up with plots. Is that permitted,
and do Qwen3.6-27B's base terms flow through? Happy to include whatever
attribution you'd prefer.

**2. `mse_scale`.** `nla_meta.yaml` has `extraction.norm: none` and doesn't
include `injection_scale` or `mse_scale`. Reading EasyNLA's `nla/config.py`, an
absent `mse_scale` resolves to `sqrt(d_model)`, which would make the
reconstruction MSE direction-only — consistent with the official NLA repo's
`MSE = 2(1 − cos)`. Can you confirm that's what this checkpoint was trained
with, or whether it was overridden on the command line? I want to be sure I
describe the metric correctly.

**3. (only if I can't trace it myself — DELETE otherwise)** `nla/utils/hooks.py`
registers `karvonen_inject_in_residual` (norm-matched add at block 1), while
`inject_at_marked_positions` does embedding replacement. Which path produced
this checkpoint?

Thanks!

---

## Notes to self
- Purely about the artifact. Do NOT mention the MATS application, do NOT ask for
  project advice or feedback on direction. If asked what it's for, say so plainly.
- **Do not block on a reply.** If the licence question is unresolved by write-up
  time, publish using the official Apache-2.0 `kitft` checkpoints and keep
  Celeste's as an unpublished cross-check. Dependency removed.
- Try to answer Q3 from the training entrypoint + `utils/patch_vllm_lens.py` first.
  Asking something that's in their own config reads badly.
