# Blog 01 — claims about the two papers, checked against the live arXiv HTML (2026-09-20)

Method: `curl https://arxiv.org/html/<id>`, tags stripped, grep for the supporting text.
Both pages fetched today; quotes below are verbatim from them.

| # | Claim in DRAFT.md | Verdict | Supporting text |
|---|---|---|---|
| 1 | 2024 paper found single tokens can flip the answer | VERIFIED | abstract: "We find many examples of forking tokens, including surprising ones such as punctuation marks, suggesting that LLMs are often just a single token away from saying something very different." |
| 2 | ...on GPT-3.5 | VERIFIED | "We evaluated OpenAI's GPT-3.5 completion model (gpt-3.5-turbo-instruct-0914; ~$2 per 1M tokens)." |
| 3 | 2026 paper: low-sample jaggedness was largely sampling noise | VERIFIED | abstract: "noise is largely an artifact of sampling rather than an LLM's sensitivity to each individual token or reasoning step"; "variation across reasoning rollouts is modeled well as multinomial sampling noise (§3.3)" |
| 4 | ...on Llama-3-8B, on tinyMMLU | VERIFIED | "Llama-3-8B-Instruct ... and the native reasoning model DeepSeek-R1-Distill-Llama-8B ... as they solve problems in tinyMMLU" |
| 5 | 2024 paper raises activations-predict-forks | VERIFIED | Discussion: "it might even be possible to avoid sampling altogether if hidden activations can be used to predict forking." (draft paraphrases; not in quote marks) |
| 6 | Neither paper asks which questions fork | VERIFIED as absence | 0 occurrences of screen / select·filter·choose questions / "which questions" in either text. |

Nuance found, not in the draft: the 2024 paper DID restrict its question set for cost —
"We randomly sampled question-answer pairs for all datasets ... ~$500 in total", using
tinyBenchmarks' 100-item subsets. So they were also compute-limited and also subsampled;
the difference is that they sampled at RANDOM, not by how uncertain the model was.
Claim 6 stands as written.

Not checkable here: the 2026 slope value (-0.4903) sits in stripped math markup; the draft
does not cite it.
