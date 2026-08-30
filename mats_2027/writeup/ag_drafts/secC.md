Model details

I have used the official anthropic nla kitft/nla-gemma3-12b-L32-av` (AV) + `kitft/nla-gemma3-12b-L32-ar` (AR), Apache-2.0 . The base model that i used for this experiment was "google/gemma-3-12b-it" . ACtivations taken from layer 32 of 48 , dimension d of the vector is 3840.

The avativation verbaliser model (AV) is of same architecture as the base . The activation is injected as a single token embedding into a fixed prompt with injection_scale 80000 (available from the sidecar). I perform decoding at sampling temperature of  T = 1 (verify if my decoding terminology usage is right)

ACtivation Reconstructor model (AR) is the base model truncated to 33 blocks + a learned linear head on the final token

MSE is **direction-only**: both vectors L2-normalised to mse_scale = √3840 = 61.97 before comparison, so MSE = 2(1 − cos).


Corpus and sampling

I chose 6 passages for my experiment . 5 cricket Wikipedia paragraphs and 1 on French Revolution (just random)

All these passages had a minimum of 50 tokens or more. I had sampled on last 10 contiguous poistions of each passage with K=4. so in total i had 6 x 10 x 4 = 240 explanations.

K is the resampling per activation at T = 1.

I chose cricket because its a familiar topic for me one than i can grade quickly . 

I also took samples from a 2019 biography of great indian freedpom fighter Shri . Veer Savarkar (by Dr. Vikram Sampath) since i believed wikipedia is in almost every pretrained model's knowledge. although this is a dated biography in llm standards, i wondered if this cud be so much in training distrobution of the model as compared to wikipedia cricket pssages and hence i chose it out of instinct and also my lvoe for the book.

Steps i did 

<what do u think i think we can put in a flow chart here.. i would request u to implement the flow chart with contents, i will edit as i see fit>

In order to measure my agreement with the k=haiku judge i was using throughout the above processes, i validated it on one particular task.. i mean , i measured the agreement in labeling between me and haiku4.5 . for that process, i took 15- stratifed claims selected by opus5 and I prepared an interfact (simple HTML page) that exposed me to the prefix (passage uptil the position) and the claim and i had 3 options in front of me (S/C/N) . I also undertook 30 retests to measure my own agreement rate and consistency . My self consistency rounded at 96.7 % . <he may ask or think why u did not agree with u 100 percent, do u think its better to show what and where and how much i erred so we can show it here>. my agreement with haiku was 88.7 % . here are a few samples were the two of su disagreed <may be do u think we wshud add them here a few may be>

to the best of my knowlwedge , the paper does not report any valiation for its confabulation detector judge models.

I reconstructed the prompts(decompose/verify/vibe/match) based on the original article of Anthropic (pls verify this for accuracy once that the prompts are indeed absent in the original article and paper and repo or even in appendix or something)

The infrastructure that i used for these experiments :

Pod : RunPod A40 48 GB, SECURE, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`

Dependencies : torch 2.9.1+cu128 · transformers 5.3.0 · torchvision 0.24.1 · sglang 0.5.10

Total GPU spend for the project ≈ $3.5


