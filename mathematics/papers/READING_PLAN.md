# Reading plan — Kantamneni & Tegmark, "Language Models Use Trigonometry to Do Addition" (arXiv 2502.00873)

Read in full by Claude on 2026-09-20 from the arXiv HTML (main text + appendices A–F), saved as
`kantamneni_tegmark_2502.00873.txt`; PDF alongside. Method agreed with AG: **one idea per sitting,
worked by hand, his pace, define a word before using it.** AG knows linear algebra and trig; the gap
is interpretability vocabulary, not mathematics.

| # | Sitting | Paper section | Hand exercise |
|---|---|---|---|
| 1 | The claim in one equation: helix(a) = C·B(a) | §4.2, Fig 1, Fig 3 | compute the T=10 and T=100 entries of B(3) and B(63) |
| 2 | Why these four clocks (2, 5, 10, 100) and why a line too | §4.1, Fig 2, App. B | which clocks can tell 17 from 67? from 12? |
| 3 | How they FOUND it: Fourier transform + PCA on layer-0 outputs | §4.1, §4.3 | — (read a spectrum plot) |
| 4 | How they showed the model USES it: activation patching | §4.4, Fig 4, Table 1 | trace one clean/corrupted pair by hand |
| 5 | The Clock algorithm in GPT-J: who moves what, where | §5.1–5.3, Figs 5–7 | label the four stages on the layer axis |
| 6 | Neurons, and what is NOT known | §5.4, §5.5 | cos(a+b) identity by hand |
| 7 | Why a clock at all, and why errors are ±10 | App. E, App. F | — |
| 8 | What carries over to Qwen3.8 and what cannot | §5.5 | — |

## Things to keep honest about (from the paper itself)
- Models: GPT-J 6B, Pythia-6.9B, Llama3.1-8B. a, b in 0–99. **Every number is ONE token** in these
  models (GPT-J 0–361, Pythia 0–557, Llama 0–999). §5.5 says models that tokenise each digit
  separately (they name Gemma-2-9B) "must use additional algorithms to collate digit tokens".
  **Qwen3.8 writes one digit per token (seen in our own trace), so the paper does not port directly.**
- §5.5: "we do not know the exact mechanism" that turns helix(a), helix(b) into helix(a+b). The trig
  identity is a hypothesis; they could not isolate it. Same for Nanda et al. 2023.
- The helix fit underperforms a PCA baseline on 3 of 6 tasks (Table 1) → numbers carry more structure
  than the helix. It is weaker on Llama3.1-8B (gated MLPs) → "could be one method of an ensemble".
- Prompts that worked: GPT-J/Pythia `Output ONLY a number. {a}+{b}=`; Llama
  `The following is a correct addition problem.\n{a}+{b}=`. Plain `a+b=` did worse — same thing we
  saw on Qwen3-4B.
- Errors (GPT-J, 19.5% wrong): mostly off by −10 (45.7%) or +10 (27.9%); a chi-squared test says this
  is NOT a carrying failure; the read-out from the a+b helix is periodic with period 10.
