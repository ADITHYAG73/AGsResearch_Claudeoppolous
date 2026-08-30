Model details

I have used the official anthropic nla kitft/nla-gemma3-12b-L32-av` (AV) + `kitft/nla-gemma3-12b-L32-ar` (AR), Apache-2.0 . The base model that i used for this experiment was "google/gemma-3-12b-it" . ACtivations taken from layer 32 of 48 , dimension d of the vector is 3840.

The avativation verbaliser model (AV) is of same architecture as the base . The activation is injected as a single token embedding into a fixed prompt with injection_scale 80000 (available from the sidecar). I perform decoding at sampling temperature of  T = 1 (verify if my decoding terminology usage is right)

ACtivation Reconstructor model (AR) is the base model truncated to 33 blocks + a learned linear head on the final token

MSE is **direction-only**: both vectors L2-normalised to mse_scale = √3840 = 61.97 before comparison, so MSE = 2(1 − cos).


Corpus and sampling

I chose 6 passages for my experiment . 5 cricket Wikipedia paragraphs and 1 on the French Revolution — the latter is the NLA maintainers' own example passage (doc_id THEIR_EXAMPLE), which I kept as an in-distribution reference point rather than replacing it. It is included in every number I report (397 of the 2065 claims); cricket-only the levels read THEME 70.0 / ENTITY 39.3 / DETAIL 34.6 against 69.1 / 43.8 / 36.0 for all six, so the ordering is the same either way

Every position I sampled sits at token index 50 or later — that is the official pipeline's `_MIN_POSITION = 50` (`nla/datagen/stage0_extract.py:35`), a constraint on the position, not on the length of the passage. My lowest sampled index is 79. I had sampled on last 10 contiguous poistions of each passage with K=4. so in total i had 6 x 10 x 4 = 240 explanations.

K is the resampling per activation at T = 1.

I chose cricket because its a familiar topic for me one than i can grade quickly . 

I also took samples from a 2019 biography of great indian freedpom fighter Shri . Veer Savarkar (by Dr. Vikram Sampath) since i believed wikipedia is in almost every pretrained model's knowledge. although this is a dated biography in llm standards, i wondered if it would be FAR LESS represented in the model's training distribution than wikipedia cricket passages are, which would let me see what the AV does when the activation is thinner — and hence i chose it out of instinct and also my lvoe for the book.  <<AG: this is your reasoning, not a fact — re-voice this sentence in your own words. I only flipped the polarity, which was inverted.>>

Steps i did 

<what do u think i think we can put in a flow chart here.. i would request u to implement the flow chart with contents, i will edit as i see fit>

In order to measure my agreement with the k=haiku judge i was using throughout the above processes, i validated it on one particular task.. i mean , i measured the agreement in labeling between me and haiku4.5 . for that process, i took 150 stratified claims — 50 per level, spread evenly across the 10 position offsets — drawn by a seeded script (`harness/sample_stratified.py`, seed 20260822) rather than chosen by a model, and I prepared an interfact (simple HTML page) that exposed me to the prefix (passage uptil the position) and the claim and i had 3 options in front of me (S/C/N) . I also undertook 30 retests to measure my own agreement rate and consistency . My self consistency rounded at 96.7 % . <he may ask or think why u did not agree with u 100 percent, do u think its better to show what and where and how much i erred so we can show it here>. my agreement with haiku was 88.7 % . here are a few samples were the two of su disagreed <may be do u think we wshud add them here a few may be>

to the best of my knowlwedge , the paper does not report any valiation for its confabulation detector judge models.

I reconstructed the SHAPE of their pipeline (decompose/verify/vibe/match) from the grader outputs shipped inside the paper's own HTML, and wrote my own prompts to match that output format — their prompts are not published. I checked this two ways: the HTML carries `decompose_response`, `verify_response`, `vibe_response` and `match_response` with no corresponding `*_prompt` keys (only four unrelated widgets ship prompts), and the official repo has no confabulation-analysis code in it at all.

The infrastructure that i used for these experiments :

Pod : RunPod A40 48 GB, SECURE, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`

Dependencies : torch 2.9.1+cu128 · transformers 5.3.0 · torchvision 0.24.1 · sglang 0.5.10.post1

Total GPU spend for the project = $3.99 across six pod sessions ($1.23 + $1.08 + $0.40 + $0.34 + $0.49 + $0.45). API spend was measured only once — $4.00 of Haiku on 28 Aug, read off the console; no stage records token usage, so every other API figure I have is an estimate.




---
FACTS NOT YET IN THIS SECTION (from C_setup_facts.md — write them or point at figure G0):
- Delta's definition and sign convention: mse(ablated) - mse(intact), same explanation, same
  activation, same resample. Delta > 0 load-bearing, Delta < 0 removal helped.
- The ablation is a REWRITE with the prose reflowed, not a deletion — this is the paper's own
  method, and it is what stops the coherence rival from contaminating Delta.
- Cricket is in-distribution, measured: AV->AR round-trip cos 0.996 on cricket vs 0.997 on the
  maintainers' own example (SMOKE-01).
- Why the last 10 positions: it is what the paper's recurrence analysis uses.
- Savarkar regression canary: the Dravid passage re-run in that batch returned cos = 1.000000.
- S/C/N was collected but ALL analysis uses the binary supported / not-supported, as the paper does.
Figure G0 (writeup/figures/G0_pipeline.png) already carries the first two.
