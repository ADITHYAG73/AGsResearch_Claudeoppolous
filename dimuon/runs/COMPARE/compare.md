# COMPARE-01: our measured Dimuon rule against what is public

Retrieved 2026-11-10. Our rule was frozen (MULTISTEP-01 + PREDICT-01) before any of this was read.
Scripts: `pipeline/cmp_horner.py`, `cmp_variantX.py`, `cmp_anpaure.py`, `cmp_damping.py`.
Outputs: `runs/COMPARE/{horner,variantX,anpaure_p6,damping}.{txt,json}`.
Third-party code was cloned into the session scratchpad only, not into this repo.

## 1. Sources and who they come from

**Official (Thinking Machines).** None of these gives the update rule.

| source | what it says about the mechanism |
|---|---|
| Tinker docs, Optimizers page, https://tinker-docs.thinkingmachines.ai/tinker/optimizers/ | "developed in-house at TML specifically for LoRA finetuning". The parameters are `learning_rate` (required, "typically 10–100×" AdamW's), `beta1` (default 0.9, "EMA coefficient for the momentum") and `grad_clip_norm` (default 0, "maximum global gradient norm"). In their SFT runs about 15× the AdamW learning rate worked best (about 60× in RL). Dimuon "benefitted from a higher LoRA rank" and has "slightly better learning-rate invariance across LoRA ranks". No formula. |
| SDK `tinker/types/optimizer.py` (tinker 0.39.0) | `DimuonOptimizerConfig(version=1)`. `DimuonParams` docstrings: "EMA coefficient for the momentum" and "Maximum global gradient norm". `create_lora_training_client` exposes no LoRA alpha. |
| Announcement (republished at https://www.neperos.com/article/to4z77b50089c76a; I found no thinkingmachines.ai/news page for Dimuon, and /news/dimuon/ returns 404) | Performance claims only. Cites no papers. |
| `thinking-machines-lab/tinker-cookbook` @ 8486b8d (2026-11-06) | `grep -ri dimuon` finds nothing. |
| Horace He (TML) on X, https://x.com/cHHillee/status/2110569587955908774 | Not fetchable (HTTP 402). A search snippet quotes it as "a variant of muon that works far better for lora fine-tuning". |
| "LoRA Without Regret", https://thinkingmachines.ai/blog/lora/ (Schulman et al., 2025-09-29) | Background, not Dimuon. It uses W' = W + (α/r)BA with α = 32. "The 1/r scaling factor makes the optimal learning rate approximately independent of rank". "The optimal LR changes by a factor of less than 2 between rank=4 and rank=512". |

**Third-party reverse engineering.** All three works used the same black-box method we did: Tinker plus an Adam-based gradient meter.

| source | origin |
|---|---|
| `anpaure/dimuon` @ d6570bb (2026-10-28), https://github.com/anpaure/dimuon. README and `dimuon.py` (104 lines). | Independent reverse engineering. Validated on 39 runs, 13 models, ranks 1–128, up to 20 steps. |
| `catid/dimuon` @ c9b4e19 (2026-10-29), https://github.com/catid/dimuon | Not independent. Its `dimuon.py` has the same docstring, coefficients and core functions as anpaure's, plus batching. The repository does not credit anpaure. It is a benchmark study (Qwen2.5-32B) and was used here only as a cross-reference. |
| `romeerp/dimuon-replay` @ d32fbee (2026-11-07), https://github.com/romeerp/dimuon-replay, with the write-up https://romeerp.github.io/reverse-engineering-dimuon/ (2026-10-28) and notes https://romeerp.github.io/reverse-engineering-dimuon/notes/ (Romeer Pillay) | Independent. Rank 1 (Qwen3-8B o_proj, SST-2), plus continuation tests at ranks 1–64. Precision-level fingerprinting of the orthogonaliser. |
| theblackcat102 (Zhi Rui Tam), "What does LoRA specialized Muon optimizer even mean?", https://zrt.wtf/do-lora-optimizers-matter/ (2026-11-06) | Not a reverse engineering: it uses anpaure's `dimuon.py` and ablates it. It says TML "didn't share any details". It calls the rank factor a "magic update size … fitted", and says the meaning of "Di" is unknown. Its findings: the feature-side conditioning carries the gain over Muon, the choice of statistic barely matters, and coupling the two factors changes the loss by 3e-5. |

**Papers.** These were checked for the specific points cited. The announcement cites none of them.

- Polar Express, Amsel et al., arXiv 2505.16932: https://arxiv.org/html/2505.16932
- PoLoRA, Ghosh, Parshakova and Gower, arXiv 2607.17620: https://arxiv.org/html/2607.17620. Not Dimuon, but it derives the same sandwich form.
- Shampoo, Gupta, Koren and Singer, arXiv 1802.09568, Algorithm 1: https://arxiv.org/pdf/1802.09568
- Old Optimizer, New Norm, Bernstein and Newhouse, arXiv 2409.20325 (abstract only): https://arxiv.org/abs/2409.20325
- Searches turned up nothing on "dimension-aware" or "dualized" Muon for LoRA that relates to Dimuon. I did not search further.

## 2. Point-by-point comparison

In the tables, A = anpaure, R = romeerp, T = theblackcat102 (reading anpaure), Off = official. "Agree" means the same equation; numerical details are noted separately.

### Momentum and state

| ingredient | public statement | ours (journal entry) | verdict |
|---|---|---|---|
| Momentum form | Off: "EMA coefficient for the momentum". A: "Ordinary EMA", `m_new = beta1*m_old + (1-beta1)*scale*g` (dimuon.py:88). R: M_t = βM_{t−1} + (1−β)G̃_t, computed as `old.mul(beta).add(g, alpha=1-beta)` (identified_machinery.py:106; R says it chose "multiply-then-scaled-add" among five expressions). | m32 = βS + (1−β)c·g, EMA, not Nesterov and not a sum (MULTISTEP-01 Q2) | Agree. R pins the floating-point operation order further than we did. |
| Momentum precision | A: computed in fp32, used unrounded, "rounding the stored copy once to BF16, after using its unrounded value". R: fp32 current value, bf16 recurrent state. | Same (MULTISTEP-01 Q2; PREDICT-01 P2) | Agree. |
| β of this call | A: β changes between steps fit. | β of the current call (MULTISTEP-01; PREDICT-01 P3) | Agree. |
| Bias correction | A: zero-initialised ordinary EMA; "first-gradient initialization … do[es] not" fit. R: "No … bias correction is invented". | Not identifiable, because every quantity is scale-invariant (MULTISTEP-01 caveats) | Both sides omit it; nobody shows it is absent. |

### Preconditioner

| ingredient | public statement | ours (journal entry) | verdict |
|---|---|---|---|
| Statistic and its timing | A: "Previous momentum's row/column energies", `rows = old.square().sum(1)`. R: e_i = Σ_j M²_{t−1,ij}, computed from decoded bf16 old momentum (`observed_metric_scales`). | Row and column energies of the previous stored bf16 momentum (MULTISTEP-01 Q1/Q2) | Agree, including the one-step lag. |
| Exponents | A: "a quarter-root on both sides", pre and post. R: D_r = (·)^(−1/4) on both sides, before and after Φ. | Fitted −0.262/−0.253/−0.259/−0.253, adopted as −1/4 on four sides (MULTISTEP-01 Q1) | Agree. |
| Row damping | A: `(rows/rows.max() + 1e-4)^-0.25`. R: `(er/er.max() + 1e-4)^-0.25`; the ridge was "inferred from one rank-one A update and rounded to 1e-4". | l + 1e-4·max(l), then l^(−1/4). This differs from A and R only by a per-tensor scalar, which the final normalisation removes, so the two are exactly equivalent. Sharply identified (MULTISTEP-01). | Agree. |
| Column damping | A and R: the same relative 1e-4 on the column side. R identified it at rank 1, where the column side is a scalar and so not testable. A says "independent sweeps … select the relative damping 1e-4" without saying which side. | Open in MULTISTEP-01: 0 to 1e-4 were indistinguishable at ranks 8 and 32. **Now tested (§3.4): at rank 32 it IS identified, with a sharp minimum at 1e-4·max.** | Agree. Our test closes our open item. |
| Zero-momentum branch | A: identity per side when the peak is 0 (dimuon.py:43). R: "Zero old momentum uses relative energies of one". | S = 0 gives Dl = Dr = 1 (MULTISTEP-01) | Agree. |
| A/B coupling | A: "A and B separately". T: coupling "none", and adding PoLoRA-style coupling changes the loss by 3e-5. PoLoRA itself couples through BᵀPB and AQAᵀ. | p* = 0 for six coupling families (FRESH-01 Q1b) | Agree with A, R and T. Dimuon is not PoLoRA. |

### Orthogonaliser (Polar Express)

| ingredient | public statement | ours (journal entry) | verdict |
|---|---|---|---|
| Iteration | Polar Express paper: odd quintics, 5-step coefficient table, safety `(a/1.01, b/1.01**3, c/1.01**5)` applied to all but the final constant polynomial. With steps = 5 that means all five used stages are divided. A: same five stages, all divided. R: same numbers (experiment.json). | Same coefficients and safety factor, 5 steps (FIRSTSTEP-01 item 3). Our coefficients were typed from memory and match the paper's code to every printed digit. | Agree. |
| Input normalisation | Paper code: `X/(‖X‖·1.01 + 1e-7)` in bf16. A: the same in fp32, then cast to fp16 (`guard 1e-7` only in fp16). R: fp16(x/(max(‖x‖, 1e-7)·1.01)), normalised in fp32. | X0 = fp16(G32/(1.01‖G32‖)); "eps, if any, does not show" (FIRSTSTEP-01 items 4, 9) | Agree. The floor form (+1e-7 vs clamp at 1e-7) is untested anywhere and only matters for tiny inputs. |
| Precision | A: "FP16 orthogonalization" (the paper uses bf16). R: fp16 stores. | fp16, not bf16 (FIRSTSTEP-01 item 1) | Agree. Both A and we disagree with the paper's bf16. |
| Evaluation form | Paper and A: expanded form, A = XXᵀ; B = bA + cA²; X = aX + BX (A via `baddbmm`). R: **Horner form, X ← X·(aI + R(bI + cR)) with R = XᵀX and a separate fp16 store of cR**. In R's test, dropping that store missed two ratios by 20% and 4.6%. The paper mentions Horner as an option. | Expanded form, fused (FIRSTSTEP-01 item 4). It was not bit-exact: 43% of entries identical at r = 1, and 1.5–2× extra deviation at r = 2..16, unexplained. | **Disagree. R is right (§3.1):** Horner reproduces D1 bit-for-bit at r = 1, 2. |
| Orientation | A: transposes to wide, Gram r×r. R: tall, Gram XᵀX (r×r). | Not identifiable (FRESH-01). Our code uses the wide orientation with an r×r Gram. | Agree on the r×r Gram. |
| GPU reduction | R: the residual CPU/GPU gap is from split-K with fp16 partial sums; R reproduced two GPU paths by changing the split count and partial precision. | Not modelled | Tested in §3.1 and §3.2. It is not what the server does in our D1 data, and it does not explain variant X. |

### Final scale, clipping and the rest

| ingredient | public statement | ours (journal entry) | verdict |
|---|---|---|---|
| Final scale | A: `lr*sqrt(r)*(r/32)**0.175 * d/‖d‖` with `rank = min(shape)`; "log-log slope 0.675"; worst discrepancy "1.2 parts per million", ranks 1–128. R: C(r) = √r(r/32)^0.175; "origin of 0.175 is still a mystery"; held-out rank 16 to 5.6e-8. A: "the 32 cannot be separated from LoRA alpha = 32". T: "magic update size … fitted". | The same, within ±1.3e-7, with implied exponent 0.1750000 at every r ≠ 32 (FIRSTSTEP-01 item 5; `FS_cache/norm.json`) | Agree. **Nobody explains it**, including the official sources. |
| A vs B | A, R: the same rule and norm for each factor. A: "every LoRA factor, including the output layer"; MoE experts are normalised separately; fused gate/up is one matrix. | The same, including unembed and MLP (PREDICT-01 P8). Not tested on MoE. | Agree. A's MoE and fused-parameter claims are untested by us. |
| Clipping | Off: "maximum global gradient norm". A: "one global norm, before momentum". R: "Global raw clipping precedes EMA"; coefficient applied only if norm > clip. | c = min(1, clip/N), with N over all LoRA tensors favoured, applied to g before the EMA (FRESH-01 Q2; PREDICT-01 P1) | Agree. A states "across all factors" without our ambiguity. |
| Weight decay | A: "None". Off: no parameter for it. | No W-proportional term (FRESH-01 Q3) | Agree. |
| Zero-gradient step with nonzero momentum | A: "An expert with zero current gradient but nonzero momentum still takes a full-norm step." | The same (MULTISTEP-01 Q2; PREDICT-01 P2) | Agree. |
| Step nondeterminism | A: "clients can compute different gradients from identical inputs". A selects a "loss-matched observer". R: multiple GPU numeric paths. | Two gradient versions and two step variants, X and Y (PREDICT-01 Finding 0) | Agree that it exists. Nobody characterises X (§3.2). |
| Lag, −1/4 and the norm | No reverse-engineering source gives theory. R: "inverse fourth root [of energy] is an inverse square root of the norm". | not explained | See §4. |

## 3. Offline tests on our data

### 3.1 Horner-form fp16 Polar Express (R) on D1, step 1, B = 0, all 152 modules

Run with `cmp_horner.py 1`. Each cell is relF(U_sim, U_data), median, followed by the share of the recovered fp16 matrix entries that are bit-identical, median.

| rank | ours (fused, expanded) | **Horner + fp16 cR store (R)** | Horner without the cR store | Horner, split-K 4 | Horner, split-K 16 |
|---|---|---|---|---|---|
| 1 | 5.70e-4, 0.43 | **6.3e-8, 1.00** | 3.96e-4, 0.15 | 1.6e-7, 1.00 | 9.2e-8, 1.00 |
| 2 | 6.34e-3, 0.02 | **9.1e-8, 1.00** | 4.48e-3, 0.03 | 1.20e-3, 0.18 | 8.1e-4, 0.29 |
| 4 | 6.28e-3, 0.03 | **1.76e-4, 0.91** | 3.48e-3, 0.06 | 3.99e-3, 0.06 | 2.86e-3, 0.08 |
| 8 | 5.68e-3, 0.04 | **1.83e-3, 0.13** | 3.23e-3, 0.07 | 3.93e-3, 0.06 | 3.60e-3, 0.06 |
| 16 | 6.58e-3, 0.04 | **2.79e-3, 0.09** | 3.63e-3, 0.07 | 4.96e-3, 0.05 | 4.44e-3, 0.06 |
| 32 | 1.87e-2, 0.01 | **8.07e-3, 0.04** | 8.53e-3, 0.03 | 1.41e-2, 0.02 | 1.31e-2, 0.02 |
| 64 | 2.46e-2, 0.01 | **1.03e-2, 0.03** | 1.08e-2, 0.03 | 1.67e-2, 0.02 | 1.53e-2, 0.02 |

- Fully bit-identical modules under Horner: 149/152 at r = 1, 91/152 at r = 2, 14/152 at r = 4. Max relF is 2.9e-4, 2.0e-2 and 1.0e-2 respectively.
- **This also resolves FIRSTSTEP-01 item 7**, the 1.5–2× extra deviation at r = 2..16. Each pair below is the deviation from the "X0 fp16, exact after" reference, for the data and then for the Horner simulation:
  - r = 2: 6.59e-3 vs 7.09e-3
  - r = 4: 6.53e-3 vs 6.55e-3
  - r = 8: 5.19e-3 vs 4.99e-3
  - r = 16: 5.62e-3 vs 5.66e-3
  - r = 32: 1.39e-2 vs 1.39e-2
  - r = 64: 1.80e-2 vs 1.80e-2

  Our expanded emulation gives 2.66e-3 to 3.58e-3 at r = 2..16. So the "unexplained" excess was the server's Horner rounding, which our emulation did not have.
- At r ≥ 8 Horner is 2–3× closer than our emulation but still not exact. The likely cause is GPU accumulation order in the length-r products. Our split-K emulation of the long Gram reduction makes things worse at every r ≥ 2, so the CPU-style full fp32 accumulation of R is the better model for this data.

### 3.2 Variant X (PREDICT-01 Finding 0)

The test step is P6 state 1 (3362e501…, rank 8, T1), step 2. X is the trunk, d7d226c6…; Y is the repeats, 92f25fda…. At this step lora_A has zero momentum, so dA is a pure orthogonaliser test. Every second tensor; relF median / p90 / max. Run with `cmp_variantX.py`.

| orthogonaliser | A vs X | A vs Y | B vs X | B vs Y |
|---|---|---|---|---|
| ours x0f16 (frozen rule) | 2.83e-2 / 5.95e-2 / 9.81e-2 | 4.90e-3 / 9.84e-3 / 1.69e-2 | 1.32e-2 | 4.63e-3 |
| ours fused fp16 | 2.85e-2 | 5.79e-3 | 1.39e-2 | 5.39e-3 |
| Horner (R) | 2.98e-2 | **3.82e-3** | 1.41e-2 | **1.40e-3** / 3.38e-3 / 5.76e-3 |
| Horner, split-K 2 / 8 / 32 | 2.92e-2 / 2.93e-2 / 2.91e-2 | 4.62e-3 / 4.58e-3 / 4.28e-3 | 1.40e-2 to 1.43e-2 | 4.48e-3 / 3.30e-3 / 3.31e-3 |
| X vs Y (measured) | 2.83e-2 | – | 1.41e-2 | – |

- **No public description reproduces X.** Every variant sits at the X–Y distance from X.
- Horner improves the fit to Y, especially for preconditioned B (4.6e-3 → 1.4e-3).
- A, by its own account, sees gradients that differ from identical inputs. X being a step taken on a different backward-pass gradient version would fit that, but it cannot be tested offline here: all three state-1 oracles are bit-identical.

### 3.3 anpaure's `dimuon.py` run unchanged on our P6 chain (rank 8, steps 1–6, every second tensor)

The optimizer was fed our oracle gradients, rolled its own bf16 momentum, and started each step from the measured weights. Run with `cmp_anpaure.py`; output in `anpaure_p6.txt`. Each cell is median relF for A / B.

| step | ours (frozen) | anpaure fp16 (default) | anpaure fp64 | our rule + Horner PE |
|---|---|---|---|---|
| 1 | – / 5.09e-3 | – / 5.62e-3 | – / 5.28e-3 | – / **2.03e-3** |
| 2 | 2.83e-2 / 1.32e-2 | 2.85e-2 / 1.39e-2 | 2.83e-2 / 1.32e-2 | 2.98e-2 / 1.41e-2 |
| 3 | 2.07e-2 / 1.13e-2 | 2.10e-2 / 1.15e-2 | 2.07e-2 / 1.15e-2 | 2.08e-2 / 1.18e-2 |
| 4 | 9.45e-3 / 1.01e-2 | 9.97e-3 / 1.02e-2 | 9.39e-3 / 1.01e-2 | 1.07e-2 / 1.01e-2 |
| 5 | 9.73e-3 / 9.86e-3 | 9.61e-3 / 1.02e-2 | 9.73e-3 / 9.85e-3 | 9.48e-3 / 9.90e-3 |
| 6 | 8.84e-3 / 9.98e-3 | 9.24e-3 / 1.02e-2 | 8.82e-3 / 9.97e-3 | 8.42e-3 / 1.02e-2 |

- anpaure's implementation is the same rule as ours. The difference between its fp64 run and ours is ≤ 2% of the residual, which is consistent with exact equivalence (relative vs absolute damping cancels).
- Beyond step 1 the ~1% residual in this chain is not set by the orthogonaliser: Horner does not move it.
- The same holds for the P6 fresh twins (fresh-law steps, our rule vs Horner, A / B):
  - T0: 5.0e-3 / 5.0e-3 → 2.4e-3 / 1.9e-3
  - T1: 1.7e-2 / 1.6e-2 → 1.6e-2 / 1.7e-2
  - T2: 2.3e-2 / 1.7e-2 → 2.3e-2 / 1.7e-2

  This is consistent with gradient-version mismatch on T1/T2, which PREDICT-01 already suspected.

### 3.4 Column damping at rank 32 (P2s/P4s solo chain, step 3 from aab50395, lr 1e-3, every second tensor)

Row damping fixed at 1e-4·max. Column damping eps_c·max, median relF A / B. Run with `cmp_damping.py`.

| eps_c | 0 | 3e-5 | 6e-5 | **1e-4** | 1.5e-4 | 3e-4 | 1e-3 | 1e-2 | 1e-1 |
|---|---|---|---|---|---|---|---|---|---|
| A | 9.36e-3 | 8.63e-3 | 7.71e-3 | **5.91e-3** | 8.10e-3 | 1.09e-2 | 1.27e-2 | 3.19e-2 | 1.16e-1 |
| B | 4.04e-3 | 3.93e-3 | 3.81e-3 | **3.53e-3** | 3.86e-3 | 4.26e-3 | 5.45e-3 | 2.87e-2 | 1.18e-1 |

- With 1e-4 relative to the mean instead of the max: A 9.12e-3, B 3.97e-3. Worse than relative to the max.
- Median column-energy spread (min/max) is 3.9e-2 for A and 3.9e-2 for B.
- So the column damping is identified, at exactly the public value. It is one state, one step and one rank.

### 3.5 The rank factor (r/32)^0.175

- Every public formula is this exact expression, and all of them are empirical. Ours matches it to ±1.3e-7 at ranks 1–64; A reports 1.2 ppm over ranks 1–128 and R reports 5.6e-8 on held-out rank 16.
- No source, official or third-party, derives it.
- Tinker exposes no LoRA alpha (α = 32 fixed; `create_lora_training_client` has no alpha argument). So (r/32)^0.175 and (α/r)^(−0.175) cannot be told apart, as A also says.
- My own inference, not sourced and not a test: the init row norm of A is the same at r = 4 and r = 8 (‖A‖_F/√r ≈ 0.58 in PILOT-01 and P6). With roughly orthogonal ΔB, the first-step effective change (α/r)·ΔB·A then scales as r^(−0.325) in Frobenius norm and r^(−0.825) in spectral norm. Neither is rank-invariant, so a simple "keep the effective update fixed" argument does not produce 0.175.
- The official docs say only that Dimuon has "slightly better learning-rate invariance across LoRA ranks". That is consistent with 0.175 being a tuned constant chosen so the optimal learning rate stays put across ranks. It is a guess, not an explanation.

## 4. What the theory says (for a student)

Sourced statements are marked [S]; my inferences are marked [I].

- **Muon as steepest descent.** [S] Bernstein and Newhouse: with EMAs switched off, Adam, Shampoo and Prodigy are each "equivalent to steepest descent under a particular norm". Muon's msign(G) = UVᵀ is the steepest-descent direction under the spectral norm. Polar Express is just a fast, accurate way to compute msign in low precision (Amsel et al.).
- **Weighted spectral norm.** [S] PoLoRA eq. (13)–(14): minimising ⟨G, ΔW⟩ subject to ‖P^{1/2} ΔW Q^{1/2}‖₂ ≤ τ gives ΔW = −τ P^{−1/2} msign(P^{−1/2} G Q^{−1/2}) Q^{−1/2}.
  - [I] Dimuon's direction Dl·PE(Dl·m·Dr)·Dr is exactly this form with P^{1/2} = diag(l)^{1/4} and Q^{1/2} = diag(c)^{1/4}, i.e. P = diag(l)^{1/2} and Q = diag(c)^{1/2}.
  - [I] So Dimuon is steepest descent, applied to the momentum, under a spectral norm in which row i is weighted by (row energy)^{1/4} and column j by (column energy)^{1/4}: ‖diag(l)^{1/4} Δ diag(c)^{1/4}‖₂.
  - [I] Rows and columns that have recently carried a lot of momentum are counted as "longer", so the step along them is shorter.
- **The −1/4 exponent.** [S] Shampoo's matrix update is W ← W − η L^{−1/4} G R^{−1/4}, with L = Σ GGᵀ and R = Σ GᵀG (Gupta et al., Alg. 1): "The 1/4 exponent arises from our analysis".
  - [I] diag(L)_i is a row energy and diag(R)_j a column energy. So Dimuon's Dl, Dr are a diagonal-Shampoo preconditioner with no accumulation (a one-step-old statistic instead of a running sum).
  - [I] Instead of applying L^{−1/4}GR^{−1/4} directly, Dimuon whitens with it, orthogonalises (Muon), and then un-whitens. That is why the same −1/4 appears twice.
  - [I] Combined, rows and columns see an overall exponent of −1/2 on their energy, i.e. −1 on their root-mean-square size.
- **Rank 1 is a good sanity check.** [I] At r = 1, PE of a vector is just its direction, so U ∝ Dl²·m, i.e. U_i ∝ m_i / sqrt(S_i² + 1e-4·max S²).
  - That is RMSprop/Adam with β₂ = 0, using the previous momentum as the second-moment estimate and an ε relative to the largest entry.
  - So Dimuon interpolates between Adam (entrywise normalisation, at r = 1) and Muon (spectral normalisation, when the energies are uniform).
  - [S] theblackcat102's ablation agrees on where the effect comes from: adding only the feature-side conditioners to Muon recovers Dimuon's loss, and the choice of statistic matters little.
- **Why the lag (previous momentum, not the current one)?** No source explains it.
  - [I] A statistic built from the very matrix being orthogonalised would partly cancel itself: whitening m by its own row energies and then orthogonalising throws away exactly the information the whitening adds. Using last step's momentum keeps the preconditioner a genuine "metric" that is fixed during the step, as the steepest-descent picture requires.
  - [I] It also means the statistic is already in memory (bf16 momentum), so no second-moment buffer is stored. This matches catid's measurement that Dimuon's optimizer state is half of Muon's and a quarter of AdamW's.
  - The MULTISTEP-01 table row "symmetric −1/4, stats from CURRENT g_t" fails badly (0.57–0.80), so the lag is measured, not assumed.
- **Damping.** [S] R: the ridge "keeps inactive coordinates from getting unbounded multipliers". [I] Relative-to-max damping caps any multiplier at (1e-4)^(−1/4) = 10 times the strongest row's, and makes the preconditioner scale-free.
- **Final scale.** [I] Normalising to a fixed Frobenius norm makes the step size independent of gradient scale. That is why `lr` enters only as a scalar, and why loss scale and β changes did not matter (FIRSTSTEP-01 items 5, 8, 9). √r is the Frobenius norm of an exactly orthogonal r-column matrix. The (r/32)^0.175 factor has no theory (§3.5).

## 5. Summary

- **Disagreements:** one, the orthogonaliser's arithmetic form. romeerp's Horner form with an fp16 cR store is right and our expanded form is wrong at the rounding level:
  - Bit-exact on D1 at r = 1 (149/152 modules) and r = 2 (91/152), versus our 43% and 2% of entries.
  - D1 residual at r = 1 / 2 / 4 / 8 / 64 falls from 5.7e-4 / 6.3e-3 / 6.3e-3 / 5.7e-3 / 2.5e-2 to 6e-8 / 9e-8 / 1.8e-4 / 1.8e-3 / 1.0e-2.
  - It also explains our "unexplained" extra deviation at r = 2..16.
- Everything else agrees at the level of equations: momentum, storage, preconditioner source, exponents, row damping, zero branch, final scale, clipping, no weight decay, no coupling.
- **Public sources explain what we could not:**
  - Horner rounding.
  - The column damping. Public sources assert 1e-4·max; we confirm it at rank 32 in §3.4.
  - With PoLoRA's LMO identity and Shampoo, a norm interpretation of the update. That identification is my inference; no Dimuon source makes it.
- **What we measured that no public source states:**
  - The step itself has two variants (X/Y, 3% apart in A) at a fixed gradient. A mentions gradient nondeterminism only.
  - The orthogonaliser output is an exact (fp32 scalar) × (fp16 matrix) at fresh steps, and continuing steps are off the grid.
  - β = 0 versus 0.9 differ by one X0 flip at step 1.
  - lr enters only as a scalar, and the r32 "lr anomaly" is the variant split.
  - The rule holds unchanged on gpt-oss-20b. A also covers gpt-oss and 12 other models.
  - The zero-gradient-step cosine tests of bf16 storage.
- **Still open everywhere:** the origin of (r/32)^0.175 and of the 32 vs α; variant X; bias correction; the exact norm-floor form; GPU accumulation order at r ≥ 4.
