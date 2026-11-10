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

---

## D2 collection (S0–S6) — 2026-11-10 (implementation agent; collection + sanity stats, no fitting)

Common setup: attention-only LoRA, Qwen/Qwen3.5-4B, seed 0, `tk.TEXTS` batches T0/T1/T2, weight 1,
Dimuon lr 1e-3, no clip unless stated. Library `pipeline/s_d2.py` (on top of `tk.py`); drivers
`s_smoke.py`, `s_S0.py`, `s_fam.py`, `s_S1v2.py`, `s_S6b.py`, `s_drift.py`; regime tagging
`s_regime.py`; analysis `s_an_S0.py`, `s_an_D2.py`, `s_an_S6b.py` (outputs `runs/D2/an_*.txt`).
**Data layout (`runs/D2/`, 11 GB, npz gitignored)** — content-addressed, not per-file like D1:
`states/<sid>.npz` (`A/<m>`, `B/<m>` float32 as exported; sid = sha256 prefix of all tensors),
`grads/<key>.npz` (`GA/<m>`, `GB/<m>`, bf16-snapped oracle gradient; key = `<sid>_t<text>_w<w>_<setting>`),
`manifest.json` → `steps[step_id]` = {before, after, text, weight, params, client kind
(`new` | `fresh` = weights-only reload into fresh Dimuon state | `opt` = restored with optimizer |
`continuing`), `hist` (step ids whose optimizer state this step inherits), `grad_key`, `regime`,
`grad_regime_ok`, stats}. `s_d2.load_step(step_id)` returns before/after weights + G.
**Use only `grad_key` from the manifest (it is regime-matched, see DRIFT-01); never pair a step with
a gradient by sid alone.** 135 step records; 117 have a same-regime gradient; the other 18 are
listed in `an_D2.txt` (zero-gradient steps, the straddling S1_r32 records, three unused).

### S0 — save/restore and the nonzero-B oracle

- Oracle setting changed for nonzero states: **O1 = AdamW lr 1e6, eps 1** (wd 0, no clip, fresh
  state). With lr 1 / eps 100 (D1 setting, "O3") the float32 difference W1−W0 cannot resolve small
  lora_A gradients once A ≈ 0.03 is in the weights. lr 1e6 is accepted by the server.
- At state 2 of the b0.9 trunk (nonzero A and B), batch T1: **O1 (lr 1e6, eps 1) and O2 (lr 1e6,
  eps 10) give bit-identical snapped G for all 3,506,176 lora_A and 3,670,016 lora_B elements.**
  O1 max snap residual 1.7e-4, O2 5.8e-4 (bf16 half-gap ≈ 2e-3). O3 vs O1: B differs at 4 elements
  (|G| ≤ 8.6e-9); A differs at 303,364 elements (median |G| there 4.8e-6), rel. Frobenius 5.8e-5;
  O3 has 593,412 elements with snap residual > 1e-3. → O3 is not usable for lora_A; O1 is.
- O1 at init (B = 0, T0) vs D1 `r8_oracle` G: 0 of 3,670,016 elements differ; G_A exactly 0.
- **Reload is bit-exact**: 54/54 clients created from a save_state checkpoint (weights-only into
  fresh AdamW or Dimuon, or with optimizer) exported weights bit-identical to the source state.
- **Restore-with-optimizer is bit-exact**: continue-in-original vs save → restore with optimizer →
  continue: identical after-state (sid) at step 2 (from ck1) and step 3 (from ck2), rank 8. Also
  holds in the later regime: S3_r32_b0.9 and S6_r32_ck2_lr1e-3 (restored) are bit-identical to the
  continuing trunk's step (0 differing elements).
- Fresh Dimuon client loaded from the init checkpoint, step on T0: bit-identical to D1
  `r8_dim_lr1e-3` (0 differing elements).
- Side facts: an r8 save_state (weights + optimizer) is 43.1 MB vs 28.7 MB for the weights export
  (r32: 172.3 MB vs ~115 MB) — the optimizer state is half the size of the float32 weights.
  Re-running a trunk from a new client reproduced recorded states bit-for-bit in 3/3 checks
  (`regen_check` events).

### DRIFT-01 — the server changed numerics at ~19:31 UTC (important for every analyst)

- Symptom: trunks created after 19:32 did not reproduce D1's step 1 (rank 8: rel. Frobenius 2.05e-2
  vs D1 for b1 0 / 0.5 / 0.99, all 3.67M elements different; rank 32 similar).
- `s_drift.py` at 19:50, 19:54, 20:02 (identical results each time): new r8 Dimuon client, step 1 on
  T0 vs D1 → rel. F 2.0498e-2; new AdamW client, O1 at init on T0 vs the 18:3x gradient → G_B
  differs at 2,928,419 of 3,670,016 elements, rel. F 1.105e-2. Loss is bit-identical
  (30.40399169921875) in both regimes: the forward pass did not change, the backward did.
- Bracket: an oracle stepped at 19:30:45–52 reproduces D1's r32 gradient bit-for-bit (R1); a
  Dimuon step bracketed by 19:32:38–19:32:53 does not (R2). Steps between are "U".
- Each regime is internally deterministic (R2 repeats bit-identical at 19:50/19:54/20:02; R2 trunk
  re-runs at 20:00 and 20:38 reproduce 19:36 states bit-for-bit).
- Handling: `s_regime.py` tags every step/gradient R1/R2/U by its bracketing export times; R2
  gradients carry setting suffix `O1r2`; R2 init gradients were recomputed; `grad_key` always points
  to a same-regime gradient or is null. S1 rank 32 was re-collected entirely in R2 (`S1v2_r32*`).
  Whether the Dimuon map itself also changed is not tested here (the analyst can, with R2 pairs).

### Storage incident (affects reruns)

Free checkpoint storage is capped (~10 GB; the owner's 10 old checkpoints hold 6.92 GB). The cap
blocked client creation (402) and, when a save_state crossed it, poisoned the ServiceClient so 72 of
my jobs failed. Fixes in `s_d2.py`: every export is deleted right after download; save_state
checkpoints are deleted when their family is done; every export/save waits until listed bytes +
in-flight reservations < 9.5 GB. 46 D1-era exports were still listed ~15–20 min past their TTL
(expires_at 19:07–19:12) and kept storage at the cap; I deleted those (already-expired, contents
on disk in `runs/D1`). Also: creating the SDK REST client lazily inside a worker thread deadlocked
once (fixed: created up front). At the end none of my checkpoints remain (listing = owner's 10).

### S1 — fresh-state step from a nonzero-B state

- Trunks T0×5 (b1 0.9): `S1_r8` (R1), `S1v2_r32` (R2; the original `S1_r32` straddles the change,
  only its steps 1, 3 have same-regime G). At states 1, 2, 5: fresh-state Dimuon branch on T0 and on
  T1 (`<T>_fresh_ck{k}_t{t}`), oracle G at every state for T0 and at 1, 2, 5 for T1.
- Every Dimuon update (all experiments, both regimes) has per-tensor Frobenius norm equal to the
  step-1 value: r8 0.00221914 (0.00221913–0.00221915), r32 0.00565685 (…684–…686), **for lora_A and
  for lora_B separately**, at lr 1e-3; lr 1e-4/1e-2 scale it exactly (S6).
- lora_A moves at every step except a step from B = 0 (all step-1 records: max |ΔA| = 0).
- Fresh-state T0 step vs continuing T0 step from the same state: at state 1, **ΔA bit-identical**
  (0 diffs, r8 and r32) while ΔB differs (rel. F 0.73 r8, 0.75 r32); at state 2 both differ
  (A 0.63 / B 0.74 r8; 0.65 / 0.77 r32 R2).
- |G_A| (global) at T0 states 1→5: r8 6.0, 8.0, 10.9, 14.0, 17.6; r32 (R2) 2.66, 3.56, 4.28, 5.38,
  6.66. |G_B| ≈ 218–235 (r8), 100–112 (r32). On T1 |G_A| 0.17–1.1.

### S2 — continuing runs, T0,T0,T1,T2,T0,T1

- Runs: r8 b1 0.0 / 0.5 / 0.99 (R2), r8 0.9 (R1), r32 0.0 / 0.9 (R2). Oracle at every state for the
  batch used; checkpoints of every state 0–5 were saved during collection and deleted after.
- Step-1 updates: b1 0 and b1 0.5 bit-identical (as in D1); b1 0.9 / 0.99 differ from them slightly.
- **beta1 = 0 control (fresh-state step vs continuing step, same weights, same batch):** never
  bit-identical. r8: step 2 A 0 diffs / B rel. F 0.733; steps 3–6 A 0.61–0.63, B 0.64–0.68.
  r32: step 2 A 0 diffs / B 0.732; steps 3–6 A 0.54–0.60, B 0.62–0.67. Same in R1 (S3x below:
  A 0.59, B 0.64). **So with beta1 = 0 the step still depends on stored optimizer state.**
- Losses at the T0 steps decrease (r8 b0.9: 30.40, 29.58, —, —, 28.07); T1/T2 losses ~45 / ~14.

### S3 — beta1 changed at step 4 (restored with optimizer from b0.9 state 3, batch T2)

- b1 0.9 restored = trunk step 4 bit-identically (r8 R1, r32 R2).
- r8: b1 0.0 vs trunk A rel. F 1.25, B 1.11; b1 0.99 vs trunk A 0.32, B 0.49; b0 vs b0.99 A 1.39,
  B 1.37. r32 (R2): b0 A 1.18 / B 1.14; b0.99 A 0.54 / B 0.53. Norms unchanged (0.00221914 /
  0.00565685).
- Extra control `S3x_r8_fresh_ck3`: fresh-state b1 0 step from the same weights: vs restored b1 0
  step A rel. F 0.59, B 0.64 — consistent with S2's control.

### S4 — zero-gradient step (all per-token weights 0)

- Oracle on the zero batch (state 2, r8): G = 0 exactly (max |G| 0); loss 0.
- Continuing b1 0.9 client after 2 normal T0 steps (r8 and r32, R1): **the weights move at full
  step size** — per-tensor Frobenius 0.00221914 (r8) / 0.00565685 (r32) for A and B, at each of 3
  consecutive zero steps. Per-tensor cosine with the previous update: zero1 vs step 2 median 0.95
  (r8; min 0.61) / 0.94 (r32); zero2 vs zero1 median 0.99998 (r8) / 0.983 (r32); zero3 vs zero2
  0.99998 / 0.984.
- Fresh-state client at the same weights (r8 R1, r32 R2): zero step → max |Δ| = 0 (no movement).
- Continuing **b1 0** client after 2 steps: zero step → max |Δ| = 0.
- Shrinkage: in every zero step with fresh state or b1 0, weights are bit-unchanged, so there is no
  decay applied independently of the update. Whether the continuing zero-step update equals
  "momentum alone" exactly is left to the analyst (states + previous updates are stored).

### S5 — gradient clipping (rank 8, R1)

- Global oracle norm (A and B) at init T0: N0 = 284.19 (G_A = 0); at state 1 T0: N1 = 232.04.
- Step 1 with clip 2×N0: bit-identical to unclipped; 0.01×N0: bit-identical; **0.5×N0: 14 of
  3.67M B elements differ, max 3.1e-9, rel. F 1.7e-7**.
- Step 2 (restored with optimizer from state 1): clip 2×N1 bit-identical; 0.5×N1: **A bit-identical,
  B rel. F 7.5e-2** (max 2.1e-5); 0.01×N1: A bit-identical, B rel. F 0.126. Norms unchanged.

### S6 — lr linearity beyond step 1 (restored with optimizer)

- r8 R1, from state 2 (step 3, T1) and state 3 (step 4, T2): U(lr)·(1e-3/lr) vs U(1e-3): lr 1e-4
  A 1.9e-4 / B 4.9e-7–6.7e-7; lr 1e-2 A 1.9e-5 / B 8.7e-8–1.0e-7 (float32 rounding of A ≈ 0.03
  limits A; consistent with exact proportionality). r8 R2 (`S6b_r8`): same numbers.
- r32 R2 from state 2: lr 1e-4 A 1.5e-4 / B 4.9e-7; lr 3e-3 A 1.4e-5 / B 1.0e-7; **lr 1e-2 A 4.2e-2
  / B 1.8e-2 — not proportional**, every one of the 304 tensors affected (A per-tensor median 3.7e-2,
  max 0.108), per-tensor norms still exactly 10× (0.0565685). Reproducible: a repeat (re-run trunk
  + restore) is bit-identical. r32 at step 1 with lr 1e-2: proportional (B 4.9e-8). Cause not
  investigated.

### Usage

145 training clients logged (36 new, 87 from_state, 22 from_state_with_optimizer; +3 refused
with 402, not logged), 200 forward_backward+optim_step pairs, 338 exports (all deleted after
download), 61 save_state (all deleted), 46 already-expired D1-era exports deleted. Some clients
were wasted on the storage failures and one deadlock (~20).
