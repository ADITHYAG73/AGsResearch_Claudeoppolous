# dimuon/ experiments journal

All runs: base model `Qwen/Qwen3.5-4B`, attention-only LoRA (`train_attn=True, train_mlp=False,
train_unembed=False`) unless stated, tinker SDK 0.39.0. Every client / step / export is logged in
`runs/calls.jsonl` by `pipeline/tk.py`.

---

## PILOT-01 — 2026-11-10 (coordinator; summarised from README "What is known")

- Question: can one Dimuon step be observed through ordinary SDK calls?
- Setup: `pipeline/pilot.py`; rank 4, seed 0; one `forward_backward(cross_entropy)` on one
  20-token sentence, `optim_step(DimuonParams(learning_rate=1e-3))`; export before/after with
  `save_weights_external` + `get_external_weights_urls`.
- Result (coordinator's measurement): export = PEFT `adapter_model.safetensors`, float32, 304
  tensors (152 lora_A (r, d_in), 152 lora_B (d_out, r)), `lora_alpha = 32`. Init `lora_B = 0`,
  `lora_A` random, Frobenius ≈ 1.16. After one Dimuon step every `lora_A` is exactly unchanged;
  every `lora_B` update has Frobenius ≈ 0.00139 regardless of d_out (1024…8192), with 4 singular
  values close but not equal (e.g. 0.000737, 0.000731, 0.000698, 0.000606).
- Per-step knobs: Dimuon `learning_rate, beta1, grad_clip_norm`; AdamW
  `learning_rate, beta1, beta2, eps, weight_decay, grad_clip_norm`.
- Files: `runs/pilot01/w0`, `runs/pilot01/w1`.

---

## Harness — 2026-11-10

`pipeline/tk.py`: `make_client`, `client_from_state`, `export` (dict of float64 arrays),
`batch` (3 frozen token-id texts in `TEXTS`, per-token loss weight), `dimuon()/adamw()` param
builders, `step` (forward_backward cross_entropy + optim_step), `upload` (experimental),
`COUNTS` + `runs/calls.jsonl`. `pipeline/jobs.py`: one-step job runner used by V1–V3.
All V/D runs below use `TEXTS[0]` (the pilot sentence, 19 target tokens), weight 1.0 unless
stated. `fb` metric reported by the server is `loss:sum` = 30.403992 at weight 1 (sum, not mean:
3.040400 at weight 0.1, 304.039917 at weight 10).

---

## V1 — init determinism — 2026-11-10

- Setup: rank 8, seed 0, three Dimuon clients + three AdamW clients; one Dimuon client seed 1.
  Scripts `pipeline/v12.py`, analysis `pipeline/an_v12.py`. Files `runs/V12/*.npz|json`.
- Result:
  - Same seed (0), same rank: all 152 `lora_A` **bit-identical** across all 6 clients (3 Dimuon,
    3 AdamW) — the init does not depend on the optimizer choice.
  - Seed 1 vs seed 0: different (max |Δ| 0.0395; max |corr| between corresponding tensors 0.023;
    mean per-tensor Frobenius 1.6333 vs 1.6328 at rank 8).
  - `lora_B` = 0 exactly at init in every client.
  - D1 cross-check: rank-8 seed-0 `A0` in D1 (a later client) is bit-identical to V12's.
- Caveat: identical across clients created on the same day from the same account; whether
  different backend hardware could change it is untested.

## V2 — step repeatability (noise floor) — 2026-11-10

- Setup: rank 8, seed 0, `TEXTS[0]`, weight 1. Dimuon lr=1e-3 (beta1 0.9, no clip) in 3
  separate clients; AdamW lr=1, eps=1, betas (0.9, 0.95), wd 0, no clip in 3 separate clients.
- Result: **noise floor is zero.** Dimuon: all 152 updated `lora_B` bit-identical across the 3
  clients (max |Δ| = 0, relative Frobenius = 0). AdamW: same, bit-identical across 3 clients.
  A fourth Dimuon repeat (D1 `r8_dim_lr1e-3.npz`, a later client) is also bit-identical.
- Update size for reference: Dimuon rank 8 lr 1e-3: per-tensor Frobenius 0.002219 for all 152
  tensors (min = median = max to 4 s.f.), max |entry| 5.317e-4. `lora_A` unchanged (max |ΔA| = 0)
  in every Dimuon and AdamW run.
- Caveat: "zero" means zero for this config and batch; the noise floor for other batches /
  larger inputs is not measured.

## V3 — AdamW step-1 gradient oracle — 2026-11-10

- Question: does `delta = -lr * g / (|g| + eps)` exactly describe Tinker AdamW step 1 (B0 = 0, wd 0,
  no clip), so that `g = -sign(delta) * eps * |delta| / (lr - |delta|)`?
- Setup: rank 8, seed 0, AdamW, settings (lr, eps): (1, 1) [V12], (1, 10), (1, 0.1), (0.5, 1),
  (1, 100), (1e-3, 1e-12); plus (1, 1) with loss weight 2; plus (1, 1) with beta1 0.5, beta2 0.99.
  Script `pipeline/v3.py`, analysis `pipeline/an_v3.py`. Files `runs/V3/*.npz`.
- Result (M1 = eps outside sqrt; reference = (lr 1, eps 100)):

  | setting | rel. Frobenius vs ref | elementwise rel. median / p99 / max | sign agreement |
  |---|---|---|---|
  | lr1 eps10 | 5.97e-8 | 3.6e-8 / 1.3e-7 / 2.3e-7 | 1.000000 |
  | lr1 eps0.1 | 2.17e-6 | 4.8e-8 / 3.4e-7 / 1.35e-5 | 1.000000 |
  | lr0.5 eps1 | 2.16e-7 | 3.9e-8 / 1.3e-7 / 1.3e-6 | 1.000000 |
  | lr1 eps1 | 2.16e-7 | 3.9e-8 / 1.3e-7 / 1.3e-6 | 1.000000 |
  | lr1 eps1, weight 2 (÷2) | 3.79e-7 | 3.7e-8 / 1.4e-7 / 1.9e-6 | 1.000000 |
  | lr1 eps1, betas (0.5, 0.99) | 1.55e-7 | 3.5e-8 / 1.3e-7 / 1.0e-6 | 1.000000 |

  The alternative M2 (`delta = -lr g / sqrt(g^2 + eps)`) fails: relative errors 2.2 to 30 between
  settings. So M1 holds to float32 inversion precision (~1e-7), and step 1 is bias-corrected
  (independent of beta1/beta2).
- **The gradient is bf16.** Recovered |G| values lie on an 8-significant-bit grid (median relative
  distance to the grid 3.3e-8, p99 1.1e-7 — same as inversion noise; a random value would be
  ~1e-3 away). After snapping to that grid, G from **all 7 settings is bit-identical**
  (0 differing elements out of 3,670,016), including weight-2 ÷ 2.
- Default-ish regime (lr 1e-3, eps 1e-12): sign(delta) = -sign(G) for 100% of elements; |delta| =
  1e-3 exactly (float32) for 76.4% of elements; the rest are within a few ulp except elements with
  |G| down to 4.3e-9 where eps matters; M1 predicts every element to max abs error 2.3e-10
  (~2 float32 ulps at 1e-3). So delta = -lr sign(g) up to eps/|g| and rounding.
- Gradient linearity: weight ×2 gives exactly 2G (bit-identical after snap). **But ×10 and ×0.1 do
  not** (measured in D1): G(w=10)/10 vs G(w=1) relative Frobenius 1.28e-2, G(w=0.1)/0.1 vs G 1.23e-2;
  only ~2.9% of elements bit-equal; elementwise relative error up to ~2.8e4 on near-zero elements.
  This is ~3× the bf16 half-ulp (3.9e-3), so it is consistent with bf16 arithmetic inside the
  backward pass, not just rounding of the final G (inference, not tested). Use the separately
  recorded G for scaled-loss runs, never 10·G.
- Gradient magnitudes at rank 8: |G| max 15.94, median 0.0115, min 4.3e-9, no exact zeros.
- Oracle verdict: trustworthy and exact (bit-level after bf16 snap) for step 1 with B0 = 0. Not
  tested: steps > 1, nonzero B0, gradients of lora_A (they are 0 at B0 = 0).

## V4 — weight upload — 2026-11-10

- Setup: `pipeline/v4.py`. Took `runs/pilot01/w0` (rank 4, seed 0), set every `lora_B` to
  N(0, 1e-3²) (numpy seed 123), wrote `runs/V4/up/adapter_model.safetensors` + the original
  `adapter_config.json`, then `RestClient.external_weights_upload_start(base_model, LoraConfig(rank=4,
  seed=0, attn only))`.
- Result: **failed at the first call**: `BadRequestError 400 — "Uploading external weights is not
  enabled for this organization."` One attempt, no retry; no file was PUT, no client created.
  `tk.upload` + `v4.py` are written and ready if the feature is enabled. Round-trip untested.

---

## D1 — first dataset — 2026-11-10

- Setup: `pipeline/d1.py`. seed 0, attention-only, `TEXTS[0]`, one client per file, B0 = 0.
  Oracle = AdamW lr 1, eps 100, betas (0.9, 0.95), wd 0, no clip; G recovered by M1 then
  snapped to bf16. Dimuon = `DimuonParams(learning_rate, beta1, grad_clip_norm=0)`.
- Files (`runs/D1/`, manifest `runs/D1/manifest.json`, per-file `<file>.json` with client
  model_id, params, loss, A0 sha256, stats):
  - ranks 1, 2, 4, 8, 16, 32, 64: `r{r}_oracle.npz` (`A0/`, `G/`, raw `dB/`) and
    `r{r}_dim_lr1e-3.npz` (`U/` = lora_B after one Dimuon step).
  - rank 8 only: `r8_dim_lr1e-4|1e-2|1e-1.npz`; `r8_dim_lr1e-3_w10|w0.1.npz` with matching
    `r8_oracle_w10|w0.1.npz`; `r8_dim_lr1e-3_b0|b0.5.npz` (beta1 0.0 and 0.5).
  - 703 MB total (npz gitignored).
- Sanity statistics (no fitting):
  - In every file: max |ΔA| = 0; A0 sha256 equal between oracle and Dimuon file of each rank;
    152 lora_B tensors; max relative bf16-snap residual 1.6e-7 (1.8e-7 at w10).
  - Dimuon lr 1e-3, per-tensor Frobenius of U (min = median = max to ≥7 s.f. in every file):
    r1 0.00054525, r2 0.00087055, r4 0.00138992, r8 0.00221914, r16 0.00354307,
    r32 0.00565685, r64 0.00903171.
  - G global Frobenius: r1 668.3, r2 601.1, r4 368.7, r8 284.2, r16 191.9, r32 138.5, r64 96.0.
    Exactly-zero G elements: 1 at r32, 1 at r64, 0 elsewhere.
  - Rank 8 lr sweep: U(lr)/(lr/1e-3) vs U(1e-3) relative Frobenius 5.3e-8 (1e-4), 4.1e-8 (1e-2),
    3.6e-8 (1e-1).
  - Rank 8 loss scale: U(w=10) vs U(w=1) relative Frobenius 2.35e-2; U(w=0.1) 2.42e-2; per-tensor
    Frobenius unchanged (0.00221914). G itself differs by ~1.2% (see V3).
  - Rank 8 beta1: U(b0) and U(b0.5) bit-identical to each other; vs U(b0.9) relative Frobenius
    1.50e-6 (not bit-identical).
  - Cross-checks: `r8_dim_lr1e-3` bit-identical to V12 `dim_s0_r0`; `r8_oracle` dB bit-identical
    to V3 `lr1_eps100`.

## Usage for V1–V4 + D1

37 training clients, 36 forward_backward + 36 optim_step, 73 exports, 1 rejected upload_start
(`runs/calls.jsonl`; the two D1 rank-1 smoke-run clients are included and were reused, not rerun).

---

## FIRSTSTEP-01 — what function maps G to U at step 1 — 2026-11-10 (analysis agent; no Tinker calls)

- Question: exact map from the step-1 lora_B gradient G (d_out x r) to the Dimuon update U, using
  D1 only (152 modules x r = 1..64, rank-8 lr / loss-scale / beta1 variants). Measurement noise is
  zero (V2), so every residual below is model error or finite-precision arithmetic.
- Code (`pipeline/`): `fs_load.py` (loader + cached float64 SVDs), `fs_spec.py` (spectral curve,
  schedules), `fs_sim.py` / `fs_ns16.py` (full-matrix Newton–Schulz emulation in f64/f32/fp16/bf16),
  `fs_grid.py` (recover the fp16 matrix behind U), `fs_fitcoef.py` (free 15-coefficient fit, held
  out), `fs_final.py` (main table), `fs_noise.py` (noise floor), `fs_norm.py`, `fs_beta.py`,
  `fs_precond.py`, `fs_figs.py`. Cache + tables in `runs/FS_cache/` (`final_eval.txt|json`,
  `noise_floor.json`, `norm.json`, `transfer_raw.npy`; SVD npz gitignored).
  Figures: `figures/fs_transfer_function.png`, `figures/fs_residual_by_rank.png`.
- **Disclosure:** the Polar-Express coefficients were typed from memory (published Muon-variant
  schedule, Amsel et al. 2025), not looked up, and no Dimuon reimplementation was consulted. The data
  selected them; a free refit did not do better on held-out modules (below).

### Findings

1. **U is a scaled fp16 matrix.** Every U tensor at every rank = (one float32 scalar per tensor) x
   (a matrix whose entries are exactly representable in IEEE fp16, 11 significant bits): 100% of
   entries on the 11-bit grid in every tested module of every rank; 10-bit grid fails (0–60%).
   So the orthogonaliser runs in **fp16** (not bf16), and the result is rescaled in float32.
2. **Spectral picture holds; normalisation is s_i/||G||_F.** In G's singular basis, P^T U Q is
   diagonal up to median off-diagonal/total 7e-4 (r=2) … 1.6e-2 (r=64), and U has 5e-4 … 3.7e-2 of
   its norm outside col-span(G) — both fully accounted for by fp16 rounding (item 4). All 19,304
   singular values collapse on one curve of x = s_i/||G||_F (`fs_transfer_function.png`): linear
   below x≈1e-3 (slope ≈1000), then equi-oscillating between ≈0.875 and ≈1.12 out to x=1, with
   ~120 wiggles. No row/column energy scaling and no dependence on A: at r=8 (pe5_x0h model, 26
   modules) relF 5.3e-3 at exponent 0, vs row^±0.01 2.8e-2, col^±0.01 1.5e-2, (AA^T)^±0.01 6.2e-3 /
   6.9e-3, (AA^T)^±0.05 1.2e-2.
3. **The polynomial is the Polar-Express 5-step schedule with safety factor 1.01**: per-step odd
   quintics a x + b x^3 + c x^5 with (a,b,c) = (8.28721201814563, −23.595886519098837,
   17.300387312530933), (4.107059111542203, −2.9478499167379106, 0.5448431082926601),
   (3.9486908534822946, −2.908902115962949, 0.5518191394370137), (3.3184196573706015,
   −2.488488024314874, 0.51004894012372), (2.300652019954817, −1.6689039845747493,
   0.4188073119525673), each divided as (a/1.01, b/1.01^3, c/1.01^5), and X0 = G/(1.01·||G||_F).
   Per-singular-value rms vs data (spectral, float64): r2 8.9e-3, r4 4.4e-3, r8 1.9e-3, r16 8.2e-4,
   r32 5.1e-4, r64 3.9e-4. Neighbours, full-matrix median relF on every 5th module, X0-fp16 model
   (r2 / r8 / r64): this rule 6.5e-3 / 4.9e-3 / 1.8e-2; last step exempt from the safety factor
   1.1e-2 / 1.3e-2 / 2.1e-2 (the closest rival, 1.2–2.7x worse); safety 1.005 6.2e-2 / 9.3e-2 /
   9.0e-2; 1.02 5.7e-2 / 1.0e-1 / 1.0e-1; safety on coefficients only 4.0e-2 / 6.3e-2 / 7.3e-2;
   on input only 4.6e-2 / 1.0e-1 / 9.2e-2; 6 steps 3.5e-2 / 7.7e-2 / 2.2e-1; 4 steps 2.3e-1 /
   3.6e-1 / 4.1e-1. Spectral-only fits: Muon quintic (3.4445, −4.7750, 2.0315) x3..x8 rms ≥1.4e-1
   per rank; cubic NS x5..x15 ≥4.4e-2 at r2; a single fixed quintic fitted freely, best pooled
   rms 1.7e-2 (N≥6) vs 1.4–1.7e-3 for this schedule in the same metric.
   A free 15-coefficient fit (`fs_fitcoef.py`, half the modules) moved the coefficients by up to 3%
   and improved train but not test (full-matrix held-out relF: r2 4.1e-3 PE vs 5.5e-3 fit, r4 5.5e-3
   vs 6.3e-3, r8 5.5e-3 vs 5.2e-3, r16 5.8e-3 vs 5.9e-3, r32 1.41e-2 vs 1.44e-2, r64 1.80e-2 vs
   1.82e-2). No evidence for different coefficients.
4. **Precision placement.** X0 = fp16( G_f32 / (||G||_f32·1.01) ) — normalised in float32, then
   cast. Evidence: the out-of-span part of U at r=32/64 (3% of ||U||, from fp16 rounding of X0
   amplified ~1000x by the polynomial on tiny singular directions) has **cosine 0.86–0.90** with the
   same quantity from this model, but cosine −0.2…−0.4 if G is cast to fp16 first and normalised in
   fp16. Steps: fp16 storage, fp32 accumulate, **fused** updates B = fp16(bA + cA·A),
   X = fp16(aX + B·X) (baddbmm-style). Rank-1 fingerprint (deviation of U from G's direction, which is
   pure rounding): data relF 5.06e-4, elementwise rel. p50/p90/p99 3.55e-4 / 8.55e-4 / 1.37e-3;
   fused fp16 emulation 5.00e-4, 3.57e-4 / 8.46e-4 / 1.38e-3; unfused (a·X and B@X rounded
   separately) 6.8e-3 — rejected; bf16 3.9e-3 — rejected; f32 8e-7 — rejected.
   **Not bit-exact:** at r=1 the emulation equals the recovered fp16 matrix in median 43% of entries
   (p90 54%), differences ±1–4 fp16 ulp in 96/152 modules, up to 32 ulp; i.e. some per-step scalar
   (A = X X^T or B) rounds differently from ours.
5. **Final normalisation is explicit**: ||U||_F/(lr·sqrt(r)·(r/32)^0.175) − 1 is within ±1.3e-7 at
   every rank (implied exponent 0.175000 at every r ≠ 32), while the fp16 NS output's own norm is far
   from sqrt(r) where unconverged: ||X16|| = 0.876–0.896 (r1), 2.63–3.04 (r8), 4.16–4.68 (r32, vs
   5.66), 4.30–4.73 (r64, vs 8.0). So U = −lr·sqrt(r)·(r/32)^0.175·X/||X||_F, not a consequence of
   orthogonality. lr enters only as that scalar: lr 1e-1 and 1e-3 give identical residual tables.
6. **Full-matrix residuals** (median / max over 152 modules of ||Û−U||_F/||U||_F; `final_eval.txt`):

   | rule | r1 | r2 | r4 | r8 | r16 | r32 | r64 |
   |---|---|---|---|---|---|---|---|
   | exact polar | 5.1e-4 | 4.4e-2 | 7.4e-2 | 8.2e-2 | 8.5e-2 | 4.9e-1 | 7.2e-1 |
   | Muon quintic x5, f64 | 5.1e-4 | 1.7e-1 | 1.8e-1 | 1.9e-1 | 2.0e-1 | 2.2e-1 | 2.3e-1 |
   | PE5 no safety, f64 | 5.1e-4 | 6.5e-2 | 1.1e-1 | 1.1e-1 | 9.8e-2 | 1.0e-1 | 1.0e-1 |
   | PE6 safety, f64 | 5.1e-4 | 4.5e-2 | 7.4e-2 | 8.2e-2 | 8.5e-2 | 1.8e-1 | 2.2e-1 |
   | PE5 safety, f64 (= f32) | 5.1e-4 | 6.5e-3 | 6.5e-3 | 5.2e-3 | 9.6e-3 | 3.1e-2 | 4.0e-2 |
   | PE5, X0 fp16, exact after | **4.6e-4** | 6.6e-3 | 6.5e-3 | **5.2e-3** | **5.6e-3** | **1.4e-2** | **1.8e-2** |
   | PE5, full fp16 fused | 5.7e-4 | **6.3e-3** | **6.3e-3** | 5.7e-3 | 6.6e-3 | 1.9e-2 | 2.5e-2 |
   | PE5, full bf16 fused | 4.1e-3 | 2.4e-2 | 2.9e-2 | 2.2e-2 | 6.4e-2 | 2.4e-1 | 3.1e-1 |

   max over modules for the X0-fp16 rule: 5.8e-4, 3.8e-2, 2.8e-2, 1.6e-2, 1.0e-2, 1.9e-2, 2.5e-2.
   Median max|Û−U|/max|U|: 6.3e-4, 6.4e-3, 7.5e-3, 5.8e-3, 6.6e-3, 1.7e-2, 2.3e-2.
   Even/odd module halves agree to ≤10% at every rank (no rule parameter is fitted to modules).
7. **Noise floor** (`fs_noise.py`): deviation from the deterministic part (X0 fp16, exact after) —
   data vs our full-fp16 emulation: r1 4.6e-4 vs 4.7e-4, r32 1.38e-2 vs 1.28e-2, r64 1.79e-2 vs
   1.68e-2 (consistent with independent fp16 step noise of equal size), but r2 5.4e-3 vs 2.9e-3,
   r4 6.1e-3 vs 3.5e-3, r8 5.2e-3 vs 3.0e-3, r16 5.8e-3 vs 3.7e-3: **at r=2..16 the data deviates
   1.5–2x more than our emulation's own rounding noise**, mostly in the diagonal (singular values
   near x≈0.3–1 where the curve is steepest) and off-diagonal, not out-of-span. Unexplained;
   rounding-placement variants did not fix it consistently (A@A rounded: matches r2, too low at r8;
   b·A and c·AA rounded separately: matches r8, too high at r2).
8. **Loss scale (w10 / w0.1 with their own oracle G):** same accuracy as w=1. Median relF, X0-fp16
   rule: w10 4.9e-3, w0.1 5.3e-3, w1 5.2e-3 (fp16 emulation 5.3e-3 / 5.4e-3 / 5.7e-3). The 2.4%
   U(w10) vs U(w1) difference is the gradient's own 1.2% change, propagated.
9. **beta1:** only 1 of 152 rank-8 tensors differs between beta1 0.9 and 0
   (`layers.27.self_attn.v_proj`); in it, after removing the per-tensor scale, exactly 5 fp16
   entries differ, all in row 856 (±1, 3, 5 ulp) — the footprint of a single X0 entry rounding
   differently. Plain EMA m = (1−β)G in float32 predicts ~818 X0 flips across ~150 tensors at
   β=0.9 (Nesterov variants 120–550): **rejected**. Bias-corrected EMA m̂ = ((1−β)G)/(1−β) returns G
   except for 1-ulp float32 changes in 13% of entries; on numpy those produce 0 X0 flips but shift
   the float32 norm in 7/152 tensors, and the expected X0 flips from such norm shifts is ≈5.7 — so
   "bias-corrected momentum, scale-invariant map, rare 1-ulp norm differences" is **consistent**
   with 1 flip in 1 tensor (and with β=0 ≡ β=0.5 bit-identity), but not proven. It also implies
   the normaliser's +eps is absent or too small to matter (with +1e-7 and plain EMA, β=0.5 would
   already change 1 tensor).

### Best rule (pseudocode)

```
G32 = float32(G)                       # G is bf16-valued, shape (d_out, r)
X   = fp16( G32 / (||G32||_F * 1.01) ) # eps, if any, does not show
X   = X^T                              # wide (r, d_out); orientation not identifiable
for (a,b,c) in PE[0:5]:                # coefficients above, each (a/1.01, b/1.01^3, c/1.01^5)
    A = fp16(X X^T)                    # fp32 accumulate
    B = fp16(b*A + c*A@A)              # fused, fp32 inside
    X = fp16(a*X + B@X)                # fused, fp32 inside
U = - lr * sqrt(r) * (r/32)**0.175 * float32(X^T) / ||X||_F
```

### Caveats / open

- Not bit-exact even at r=1 (median 43% of entries identical); r≥2 bit-exactness would also need the
  GPU's accumulation order. Residual floor reached: 4.6e-4 (r1), ~5–7e-3 (r2–16), 1.4–1.8e-2
  (r32–64, dominated by fp16 rounding amplified on tiny singular directions).
- The extra 1.5–2x deviation at r=2..16 (item 7) is the one unexplained piece.
- (r/32)^0.175 is measured, not explained. beta1 mechanism (item 9) is a consistent story, not a test.
- Single text, single seed, step 1 only, B0 = 0.

## 2026-11-10 — CAND-01: candidate-rule toolkit + synthetic identifiability (no Tinker calls)

**Question.** Given (G, D) pairs for lora_B at step 1, which candidate rules G -> U can the data
tell apart, at which rank and noise level? Built and validated on synthetic data only.

**Files.** `pipeline/candidates.py` (maps, diagnostics, harness, generators, sweep),
`pipeline/test_candidates.py` (pytest wrapper), outputs in `runs/2026-11-10_candidates/`
(`identifiability.md`, `misfit.json`, `verify.json`, `selftest_output.txt`).
Run: `python candidates.py selftest` (23/23 pass, 24 s); `python candidates.py identifiability`
(780 jobs, 2868 s on 4 cores); real data: `python candidates.py analyze DIR` (dir of *.npz with G, D
[, A or K=AA^T]).

**Setup.** 18 fixed hidden rules (frob, sign, row/col norm, polar, NS quintic 3/5/8, cubic 5/15,
s^0.5, L^-1/4 G R^-1/4, polar(L^-1/2 G), polar(G R^-1/2), polar(L^-1/4 G R^-1/4), quintic5 in
f32 and simulated bf16, cpolar(.3,-.4,-.2,.1); + polar(G (AA^T)^-1/2) for sketch data) and 5 fitted
families (s^p; L^a G R^c; L^a polar(L^b G R^c) R^d; 5-step NS coefficients; polar(G K^e) K^f).
n=12 modules per cell, d_out in {1024, 2048}, 8 train / 4 held-out. Three G generators:
`parametric` (geometric spectrum, cond 10–100, lognormal row σ=1 / column σ=0.5 weights; best case),
`sketch` (G = Δ (X Aᵀ) with Gaussian A like the measured init; cond(AAᵀ) 1.1–1.8),
`sketch_anisoA` (same, A rows scaled by logspace(0,-2,r): the "upload a chosen A" experiment).
Confusable at noise eps := noise-free held-out misfit m < 1.118·eps (err ratio < 1.5 under i.i.d.
noise on D). Noisy verification at eps 1e-6/1e-3/1e-2, r 1/4/16/64.

**Results.**
- r=1, every generator, every noise level: 5 classes only — {frob, polar, col_norm, all exact NS
  variants, s^p, polar(G R^-1/2), polar(G K^-1/2), f32 NS} | {sign, row_norm, polar(L^-1/2 G)} |
  {L^-1/4 G R^-1/4, polar(L^-1/4 G R^-1/4)} | cpolar(.3,-.4,-.2,.1) | bf16 NS. Analytically: at r=1
  every rule D = G S collapses to G, and cpolar = g·|g|^(2(a+b)); only a+b is identifiable.
- r>=2, parametric: the only confusable pairs are polar~ns_cubic15 (m 7e-11 at r=2,4; 1e-4 at
  r=8,16; 1.9e-2 at r=64) and quintic5 f64~f32 (m ~1e-6). Everything else m >= 1e-2.
- sketch (natural-like A): additionally polar(G (AAᵀ)^-1/2) ~ polar (m 4.7e-4 r=2, 7.9e-3 r=4,
  2.7e-2 r=16) and at r=2 cubic5 ~ cubic15 ~ polar. With anisotropic A (sketch_anisoA) the A-rule vs
  polar misfit rises to 0.27–0.60 and cubic15 vs polar to 8e-4..0.7.
- bf16-simulated NS differs from f64 NS by m = 2e-2..2e-1 (grows with r in sketch data); fp16 3.6e-3
  (selftest, r=16): reduced-precision arithmetic is itself a ~1e-3..1e-1 "noise" floor for f64 models.
- Verification: at r>=2 every case where the true rule was not ranked 1st is a tie with a pair listed
  above (ratio 1.00000). Family parameters recovered to < 0.02 in every r>=2 verification cell,
  including eps=1e-2 (cpolar 4 exponents, NS coefficients 3.4445/-4.7750/2.0315, p, apolar e).
- NS coefficient fit is very rugged: 400 random starts missed the Muon basin; 20000 vectorised
  starts find it. Noise on G (not D) is amplified by polar-type maps: 1e-3 noise on G gave 3.8e-2
  D-error for polar at r=16 (sketch).

**Caveats.** Synthetic G only; generator realism not checked against real gradients. Bit-level
reduced-precision behaviour of the real kernel is not modelled (rounding placement is a guess).
Threshold 1.5 is a choice. fp16 mode was added after the sweep, so the table has no fp16 rows.
