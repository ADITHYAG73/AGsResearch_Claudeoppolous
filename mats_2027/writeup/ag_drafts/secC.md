**Model details.** I used the official Anthropic NLA release, `kitft/nla-gemma3-12b-L32-av` (AV) + `kitft/nla-gemma3-12b-L32-ar` (AR), Apache-2.0. The base model I used for this experiment was `google/gemma-3-12b-it`. Activations taken from layer 32 of 48, dimension d of the vector is 3840.

The activation verbaliser model (AV) is of same architecture as the base. The activation is injected as a single token embedding into a fixed prompt with injection_scale 80000 (available from the sidecar), sampled at temperature T = 1. The activation reconstructor (AR) is the base model truncated to 33 blocks plus a learned linear head on the final token.

MSE is **direction-only**: both vectors L2-normalised to mse_scale = √3840 = 61.97 before comparison, so MSE = 2(1 − cos). Magnitude is invisible to the whole pipeline.

**Corpus and sampling.** I chose 6 passages: 5 cricket Wikipedia paragraphs and 1 on the French Revolution — the latter is the NLA maintainers' own example passage, which I kept as an in-distribution reference point. It is included in every number I report (397 of the 2065 claims); cricket-only the levels read THEME 70.0 / ENTITY 39.3 / DETAIL 34.6 against 69.1 / 43.8 / 36.0 for all six, so the ordering is the same either way.

I sampled the last 10 contiguous positions of each passage with K = 4 resamples, so 6 × 10 × 4 = 240 explanations. Every position sits at token index 50 or later — the official pipeline's `_MIN_POSITION = 50` (`nla/datagen/stage0_extract.py:35`), a constraint on the position, not on the length of the passage; my lowest sampled index is 79. Cricket is in distribution for this NLA, measured rather than assumed: the AV→AR round trip returns cosine 0.996 on my passages against 0.997 on the maintainers' own example.

I chose cricket because it is a familiar topic for me, one I can grade quickly.

I also took samples from a 2019 biography of the indian freedom fighter V. D. Savarkar (by Vikram Sampath). Wikipedia is in almost every pretrained model's knowledge; this book, I reasoned, would be far less represented in the training distribution, which would let me see what the AV does when the activation is thinner.

Figure G0 shows the pipeline end to end. In short: extract the residual activation, verbalise it with the AV, decompose the explanation into atomic claims, judge each claim against the exact prefix the model had read, rewrite one claim out at a time, and re-score every variant with the AR. **Δ = mse(claim rewritten out) − mse(intact)**, on the same explanation and the same activation. Δ > 0 means removing the claim hurt reconstruction; Δ < 0 means removal helped. The ablation is a rewrite with the prose reflowed rather than a deletion, which is the paper's own method and is what keeps prose damage out of Δ.

The judge is `claude-haiku-4-5-20251001`. It sees only the prefix and the claim, and returns supported / contradicted / not-in-text; all analysis collapses that to binary supported-or-not, as the paper does. I validated it against my own blind grading — the numbers are in D1. To the best of my knowledge the paper reports no validation for its own confabulation judge.

I reconstructed the SHAPE of their pipeline (decompose / verify / vibe / match) from the grader outputs shipped inside the paper's own HTML, and wrote my own prompts to match that output format — their prompts are not published. I checked this two ways: the HTML carries `decompose_response`, `verify_response`, `vibe_response` and `match_response` with no corresponding `*_prompt` keys, and the official repo has no confabulation-analysis code in it at all.

Infrastructure: a RunPod A40 48 GB, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, torch 2.9.1+cu128 · transformers 5.3.0 · sglang 0.5.10.post1. Total GPU spend for the project was **$3.99** across six pod sessions. API spend was measured only once — $4.00 of Haiku on 28 Aug — because no stage records token usage, so every other API figure I have is an estimate.
