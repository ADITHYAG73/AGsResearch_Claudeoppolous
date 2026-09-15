# Forking Paths — roadmap (drafted 2026-09-06, Claude; AG to edit)

Sources read in full: arXiv 2412.07961 (Bigelow, Holtzman, Tanaka, Ullman, Dec 2024) and
arXiv 2608.19611 (Bigelow et al., Goodfire, Aug 2026). Code: `reference_repos/forking-fast`
(fork of ericb-goodfire/forking-fast). The 2024 authors' code is cited by that repo as
`github.com/ebigelow/forking-paths` — NOT yet opened.

## What the two iterations established

| | 2024 | 2026 |
|---|---|---|
| Question | Do single tokens fork the outcome? | How cheaply can the fork curve be measured? |
| Model | GPT-3.5-instruct (API), S=30 | Llama-3-8B-Instruct, R1-Distill-Llama-8B, S=200 (S=1000 on 3 q) |
| Outcome | categorical, 2nd LLM extracts the answer | categorical, regex (A/B/C/D/Other) |
| Tasks | 7 tasks, 4 domains, all forced categorical | tinyMMLU only |
| Result | forks exist; some at odd tokens (punctuation) | noise is exactly multinomial (slope −0.49); true curve flat + sparse forks; smoothing gives 5–22× at low S |
| Weakness | change-point on a collapsed 1-D line; S=30 noise never modelled | worse than raw at big forks (ε≥0.20); credible intervals fail coverage (48–64% at nominal 90%); dev/eval same task family |

## Open questions, in the authors' own words or by their own omission

1. **Open-ended outcomes.** 2024 §2.2 and Discussion: "more open-ended tasks could be analyzed
   using R as a semantic vector embedding" — never run. 2026 Limitations: "open-ended outcomes
   ... remains to be tested." Both papers name it, neither does it.
2. **Do the 2024 odd-token forks survive high S?** 2026 shows S=30 jaggedness is largely dice.
   Nobody re-examined the punctuation/space forks at S≥200. GPT-3.5-instruct is retired, so
   any re-run is on an open model.
3. **Larger / current models.** 2026 Limitations: two 8B models only. Neel's list: Qwen 3.5/3.6.
4. **Predict forks from activations without sampling.** 2024 Discussion: "it might even be
   possible to avoid sampling altogether if hidden activations can be used to predict forking."
   The interpretability hook. Untouched in 2026.
5. **Adaptive sampling.** 2026 Discussion: choose which positions to sample (optimal experiment
   design) instead of post-hoc smoothing. The repo has `resample_adaptive` — status unknown.
6. **Fix the credible intervals.** 2026 reports the failure and moves on.

## Candidate projects, sized to one rented GPU and a few weeks of evenings

A. **Open-ended forks (Q1).** Qwen 3.5 9B, 20 prompts (story continuation, short essays),
   S=100 every 4 tokens. Outcome = embedding of the continuation. Two statistics per position:
   spread among the S continuations, and drift from o_0. Check the 1/√S noise law on the
   scalar. Completion condition: one figure per statistic + a yes/no on "sparse forks vs
   continuous drift". Risk: embeddings measure surface not meaning — needs a judge-labelled
   subset as a control (JUDGE-02 machinery reusable).
B. **High-S replication of odd-token forks (Q2).** Qwen 3.5 9B on the 2024 task mix
   (GSM8k, HotpotQA, LastLetter), S=200 every token, 10 questions. Count forks at
   punctuation/space tokens at S=30 (subsampled) vs S=200. Completion condition: a table of
   fork counts by token class at both S, with the noise-law check. Cleanest negative-or-
   positive result. Most reusable code.
C. **Activation prediction of forks (Q4).** Needs A or B's data first. Second project.

Recommendation: **B first** (4–6 weeks, nearly all code exists), then A. C only after one of
them is written up.

## Free work before any GPU
- Reproduce §3.3 noise slope from `data/s1000` and `data/s200` on CPU (verification log entry 1).
- Open the dashboard zip; read 5 questions the way Figure 1 was read.
- Open `ebigelow/forking-paths`; record how outcomes and prompts were built per task.
- `--smoke` run of the sampler on a tiny Qwen locally (CPU or Colab) to find template/EOS breakage.
- Ask the maintainers about the missing LICENSE before publishing anything derived.

## Contact policy (see chat 2026-09-06)
Email only with a finished artifact attached. PR only for code that fixes or extends the repo.
