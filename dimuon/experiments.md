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

---

## FRESH-01 — fresh-state steps from B != 0, clipping, lr dependence — 2026-11-10 (analysis agent; no Tinker calls)

Offline, float64 analysis on D2 (`runs/D2`). Scripts `pipeline/fr_*.py` (`fr_core.py` helpers: lazy npz,
PE5 in f64 and in the FIRSTSTEP-01 fp16 emulation, `finalise` = explicit norm lr·sqrt(r)·(r/32)^0.175).
Outputs `runs/FR/` (`baseline`, `step1scan`, `coupling`, `orient`, `orient_r32`, `grid`, `clip`, `clip2`,
`lr`, `lr2`, `lr3` .txt/.json). Figure `figures/fr_fresh_state.png`. Momentum/hidden state in continuing
runs is left to the `ms_` analysis; continuing steps are used here only for clipping and lr.

**Q1a — fresh-state step from B != 0 = per-factor PE5 of the factor's own gradient (same law as step 1).**
`fr_step1scan.py` (all 45 hist-free steps with a same-regime G, every 4th module) and `fr_baseline.py`
(17 steps, all 152 modules). Relative Frobenius residual of finalise(PE5_fp16(G_X)) vs dX, per tensor:
- r8 (S1_r8 ck1/2/5 × T0/T1, S3x ck3 T2, S2_r8_b0 ck1/4/5): A median 0.0048–0.0060, B 0.0052–0.0064;
  max over tensors A ≤ 0.025, B ≤ 0.022. Step-1 reference (B, from B=0): median 0.0056–0.0059, max 0.013–0.016.
- r32 (S1v2_r32 ck1/2/5, S2_r32_b0 ck1–5): on T0 A 0.019–0.020, B 0.018–0.019 (step-1 reference B 0.0186);
  on T1/T2 A 0.0046–0.011, B 0.0057–0.0080. max A ≤ 0.032, B ≤ 0.028.
- f64 PE5 instead of fp16: r8 about equal, r32 worse (A/B 0.029–0.031 vs 0.019 on T0) — same precision
  fingerprint as step 1. Best-scalar residual = residual (norm law exact for both factors). cos ≈ 1.
- Null (PE5 of the other text's gradient at the same state): 1.38–1.41.
- Bit level (`fr_grid.py`, lora_B): fresh-state updates are (scalar × fp16 matrix) in 91–100% of sampled
  entries (step 1: 100%; misses track float32 subtraction error 1e-7..7e-7 near the 1e-6 test tolerance).
  **Continuing updates (hist ≥ 1, incl. beta1 = 0, S5, S6) are NOT: 1.5% on grid = chance.** So a
  continuing step has a non-scalar float32 operation after the fp16 orthogonaliser (relevant to `ms_`).
- Orientation (`fr_orient.py`, 7 r8 + 11 r32 tensors): fp16 PE5 with the r×r Gram vs with the d×d Gram give
  equal residuals (r32 e.g. 0.0259 vs 0.0259, 0.0168 vs 0.0170; f64 0.030–0.037). Not identifiable, does
  not matter at this resolution; the fp16 advantage comes from rounding X0, not from the Gram side.
- **Three step/gradient pairs do not match** (exclude them): `S1v2_r32_fresh_ck1_t1` (A 0.14, B 0.067),
  `S2_r8_b0_fresh_ck2/step3` (A 0.025, B 0.015), `S2_r8_b0_fresh_ck3/step4` (A 0.029, B 0.018). In them
  1.5% of ||dW|| lies outside span(G) (0.1% in matched steps), uniformly over all 152 modules, while the
  update is still a scaled fp16 matrix (grid 0.95–0.985): the fresh law applied to a gradient that differs
  from the stored oracle G. All are R2, T1/T2, 19:35 and 20:00; clock_cycle cluster does not explain it.
  Probably the same kind of backward-numerics difference as DRIFT-01; not resolvable offline.

**Q1b — no coupling** (`fr_coupling.py`, 9 matched steps, exponent fitted on even modules, reported on odd):
A1 PE5((BᵀB)^p G_A), A2 (BᵀB)^p PE5(G_A), A3 PE5((AAᵀ)^p G_A), B1 PE5(G_B (AAᵀ)^p), B2 PE5(G_B)(AAᵀ)^p,
B3 PE5(G_B (BᵀB)^p): **p* = 0 for all six families in all 9 steps**; |p| = 0.05 already raises the median
residual 1.1–15× (e.g. r32 ck5 B3 0.018 → 0.088). Identifiable: cond(BᵀB) median 1.6 (r8 ck1), 3.8 (r8 ck5,
max 9.2), 55 (r32 ck2), 85 (r32 ck5, max 165); cond(AAᵀ) 1.18 (r8), 1.49 (r32). Joint orthogonalisation of
[G_Bᵀ, w G_A] (shared r×r Gram): B best only as w → 0 and A best only as w → ∞ (A still 0.06–0.67 at
w = 100) — i.e. only the decoupled limit fits. No ‖A‖/‖B‖ balancing: both norms are exactly the law value.

**Q1c.** Fresh dA and dB are PE5 of G_A and G_B at the current state, nothing else (Q1a/b). At state 1 the
continuing dA is bit-identical to the fresh one because A has no history (G_A = 0 at step 1), so whatever A
state the continuing step keeps is scale-equivalent to "empty"; B's step-1 history makes dB differ. At state 2
both factors have history, so both differ. Continuing dB at step 2 is not PE5(g2 + a·g1) nor PE5(g2 + a·U1)
for any per-tensor a (residual 0.76–0.78, no better than PE5(g2) 0.78; `fr_clip.py`).

**Q2 — clipping** (`fr_clip.py`, `fr_clip2.py`; rank 8, R1, step 2 from state 1, T0).
- Global oracle norm g2 232.033 (N1 232.04). At clip 0.5·N1 = 116: **152/152 B tensors change, only 1 B tensor
  has its own ‖g2‖ > 116** (max 179.6) → not per-tensor clipping; a global norm (over all LoRA tensors or over
  all B tensors — indistinguishable here, ‖G_A‖ ≈ 6 vs ‖G_B‖ ≈ 232). At 0.01·N1: 152/152 change.
- A is bit-identical at every level (0 elements): A's stored state is empty at step 2, and the map is scale
  invariant in a lone gradient (as at step 1, where clip 0.01 is bit-identical and 0.5 changes 14 elements).
  → **clipping multiplies the current gradient by c = clip/N before it is combined with stored state.**
  2·N1 → c = 1 → bit-identical (as observed).
- Not scale-invariant at step 2: the relative weight of (c·g2) vs B's stored state. ΔB vs unclipped: c=0.5 relF
  0.0755 (p90 0.096), c=0.01 0.129 (p90 0.153); far smaller than relF(PE5 g1, PE5 g2) = 0.19 would suggest for
  a g1/g2 mixture, and the two increments are nearly orthogonal (cos(D1−D.5, D.5−D.01) median 0.07), ratio of
  norms 0.72 (linear-in-c would be 1.02). D(1)−D(0.01) has cos −0.29 with the polar tangent of g2⊥g1 (sign as
  expected for a −orth(Y + c·g2) update), 0.000 with T(g1), 0.000 with T(random). So g2 enters additively with
  the right sign but most of the c-dependence is not explained. **No quantitative model**: it needs the
  continuing map, which is not PE5 of any g1/g2/U1 combination.

**Q3 — lr** (`fr_lr.py`, `fr_lr2.py`, `fr_lr3.py`; r32 R2 step 3 from state 2, T1).
- Re-measured: U(lr)·1e-3/lr vs U(1e-3): A 1.6e-4 (1e-4), 1.5e-5 (3e-3), **0.037 (1e-2; p90 0.054, max 0.108)**;
  B 4.9e-7, 1.0e-7, **0.017 (p90 0.022, max 0.041)**. r8 R2 at 1e-2: A 2.0e-5, B 8.6e-8 (proportional). All 304
  tensors affected. Reproduced bit-identically on a different clock_cycle cluster (7202958 vs 7533163).
- Structure of E = U(1e-2)/10 − U(1e-3): 93–94% of ‖E‖ outside the span of U(1e-3); inside, singular values
  change only 0.1% (rotation, not rescaling). |E| has the same absolute rms in every |U| decile (additive,
  magnitude-independent, unbiased: mean E·sign(U)/|U| ≤ 0.004). corr(|E|, |W|): A 1e-4, B 0.05; corr(E, W) ≤ 0.03.
  Tangent-space test: cos(E, T(Z)) ≤ 0.035 for Z ∈ {g3, g2, U1, U2, W, U(1e-3), random}; least squares on all of
  them explains < 0.2% of ‖E‖².
- **Rules out:** precision of W + lr·U (exported fp32, float32 ulp/|U| = 7e-6 (A), 2e-8 (B); and no |W|
  correlation); weight decay / any W-proportional term; saturation/element clipping (no |U| dependence);
  a smooth lr-weighted mix of state and gradient (lr 1e-4 and 3e-3 are proportional to rounding level).
- **Consistent with:** the same algorithm taking a different numerical path (rounding) at lr 1e-2, r32: per
  tensor relE correlates with the tensor's fp16-vs-f64 PE5 gap on g3 (corr of logs A +0.65, B +0.24); relE is
  ~2.7× that gap. Trigger unknown. Candidates for the switch, by scale: per-tensor ‖lr·U_A‖/‖A‖ = 0.049 (r32 1e-2)
  vs 0.019 (r8 1e-2) and 0.015 (r32 3e-3); global ‖lr·U‖ over 304 tensors 0.99 vs 0.39 vs 0.30.

**Caveats.** Fresh-state law residuals are not bit-exact (same floor as FIRSTSTEP-01). Clip analysis is one
state, one batch, rank 8. The lr anomaly is one state, one rank. Q2/Q3 conclusions about the continuing map
are limited by not knowing that map.

**Proposed Tinker follow-ups (not run).** (1) Recompute the oracle G at the three mismatched states
(48f8…/T2, 7bde…/T1, 6b6b…/T1) twice each and compare with the stored G — decides gradient mismatch vs map.
(2) r32 state 2 T1 restored-with-optimizer lr ∈ {4e-3, 5e-3, 6e-3, 7e-3, 8e-3}, plus r8 state 2 lr ∈ {2e-2, 3e-2,
5e-2}: locate the threshold and see whether it is at fixed ‖lr·U_A‖/‖A‖ or fixed global ‖lr·U‖. (3) Fresh-state
(weights-only) r32 step from state 2 at lr 1e-2 vs 1e-3: does the anomaly need optimizer state? (4) Clip at a
continuing step with threshold between ‖G_B‖_global and ‖G‖_global at a state with large G_A (r8 state 5:
‖G_A‖ 17.6) — global over all tensors vs over B only; and a dense clip series c ∈ {1, .9, .7, .5, .3, .1, .01, 1e-4}
at step 2 to map the c-path. ~15 clients.

---

## MULTISTEP-01 — the continuing-step rule: state, momentum, R2 — 2026-11-10 (analysis agent; no Tinker calls)

**Questions.** (1) What hidden state makes a beta1 = 0 continuing step differ from its fresh twin?
(2) Momentum form / storage. (3) Does the FIRSTSTEP-01 step-1 law still hold in regime R2?
All offline on D2 (`runs/D2`), regime-matched gradients via `grad_key` only. float64 analysis.
Score = relative Frobenius residual after unit-normalising both sides (norm is the known constant),
per tensor; tables give median / p90 / max. Fitted parameters (2 eps values) chosen on the EVEN module
half, every score below is on the ODD half (76 modules x A/B) unless marked "exploratory".

**Files.** `pipeline/ms_common.py` (loaders, PE5 in f64, relF), `ms_model.py` (the rule),
`ms_fit.py` (torch fitter for conditioned-PE families), `ms_freediag.py` / `ms_symfree.py`
(hypothesis-free free-diagonal probes), `ms_b0.py` (beta1=0 grid + held-out eval), `ms_freshfloor.py`,
`ms_mom.py` (momentum chains), `ms_zero.py`, `ms_zero32_proxy.py`, `ms_r2step1.py`, `ms_figs.py`.
Outputs `runs/MS/` (`b0_grid_r{8,32}.txt`, `b0_finegrid_r8.txt`, `b0_eval_r{8,32}.{txt,json}`,
`fresh_floor.{txt,json}`, `mom_*.{txt,json}`, `zero_steps.{txt,json}`, `zero32_proxy.{txt,json}`,
`r2_step1.{txt,json}`, `freediag/`, `symfree/`). Figure `figures/ms_b0_residuals.png`.
Throwaway exploration scripts (e1–e29) live in the scratchpad, not the repo; their numbers are
marked exploratory below.

### Result: the rule

```
state per tensor: m  (momentum, STORED IN bf16), starts at 0          tall orientation n x r
                                                                        (lora_B as is, lora_A^T)
step t, gradient g_t (bf16-valued), beta = beta1 of THIS call:
  S     = m                                   # the stored momentum BEFORE this step
  m32   = beta * S + (1 - beta) * g_t         # float32; this is what is orthogonalised
  m     = bf16(m32)                           # stored for next step
  if S == 0 (fresh state, or a factor whose previous momentum is 0):  Dl = Dr = 1
  else:
      l_i = sum_j S_ij^2 ;  l = l + 1e-4 * max(l)          # row energies of previous momentum
      c_j = sum_i S_ij^2 ;  c = c + 1e-4 * max(c)          # column energies
      Dl = l^(-1/4) ;  Dr = c^(-1/4)
  X  = PE5( Dl * m32 * Dr )                   # FIRSTSTEP-01 Polar Express, 5 steps, safety 1.01
  V  = Dl * X * Dr
  U_t = - lr * sqrt(r) * (r/32)^0.175 * V / ||V||_F
```
i.e. U ∝ −L^{-1/4} PE(L^{-1/4} m_t R^{-1/4}) R^{-1/4} with L, R the row/column Gram diagonals of the
PREVIOUS momentum (a one-step-lagged, diagonal two-sided preconditioner, no second-moment EMA),
momentum m_t = beta m_{t-1} + (1-beta) g_t (EMA form, not beta m + g, not Nesterov).

### Q1 — hidden state at beta1 = 0 (S2_r{8,32}_b0 vs fresh twins; all R2)

- **Hypothesis-free first (exploratory, r8).** Step-2 lora_B continuing update has 66% of its norm
  OUTSIDE col-span(g_2) (and 65% outside span[g_1, g_2]) → not a right r x r map, needs a left
  (d_out-side) operator; its singular values are not flat (0.6–1.45 x mean) → not a bare PE output.
  Best free per-row rescaling of PE(g_t) still leaves 0.20; the fitted row scale correlates with
  the PREVIOUS gradient's row energy (log–log corr −0.97 at steps 3–6) and not the current one (−0.3),
  for A and B. Free diagonals on all four sides U ≈ diag·PE(diag·g·diag)·diag reach 0.009–0.021 on
  every tensor tried (14 tensors; the same freedom fitted to a WRONG target — another step's update —
  gives 1.08–1.18, so the freedom alone does not fit anything). A torch fit of exponents gave
  a = −0.262, b = −0.253, c = −0.259, e = −0.253: all −1/4. A free symmetric fit (one diagonal per side)
  decomposed by NNLS: row stat = energy of g_{t−1} (older gradients and current gradient coefficients
  0 to ≤1e-3), column stat likewise.
- **Damping.** eps relative to the MAX energy, 1e-4, is sharply identified (fit half, r8, steps 2/4/6:
  median relF 6.4e-3 A / 6.2e-3 B at 1e-4·max; 8e-5 → 1.8e-2 / 2.6e-2; 1.25e-4 → 1.9e-2 / 2.6e-2).
  Best eps relative to the mean: 4.2e-2 / 4.7e-2; best absolute eps: 6.0e-2 (A). Column eps is not
  identifiable (0, 1e-6, 1e-5, 1e-4 ·max indistinguishable); 1e-4·max used by symmetry. Rank 32 grid
  picks the same values (1.26e-2 / 1.18e-2 median).
- **Held-out table (odd half; `b0_eval_r*.txt`), median relF per step 2..6:**

  | rule (median relF, steps 2, 3, 4, 5, 6) | r8 A | r8 B | r32 A | r32 B |
  |---|---|---|---|---|
  | fresh twin vs continuing [null] | 0, .601, .602, .616, .593 | .745, .692, .638, .654, .693 | 0, .558, .541, .594, .511 | .743, .675, .631, .666, .679 |
  | PE5(g_t) alone [baseline] | .005, .601, .602, .616, .593 | .745, .692, .638, .654, .693 | .030, .557, .541, .594, .511 | .742, .675, .631, .665, .679 |
  | symmetric −1/4, stats from CURRENT g_t | .605, .569, .595, .572, .561 | .150, .786, .810, .795, .790 | .599, .588, .576, .606, .550 | .129, .737, .772, .793, .740 |
  | symmetric, exponents −1/2 | .005, .395, .402, .400, .375 | .533, .487, .459, .463, .473 | .030, .415, .404, .438, .371 | .542, .500, .437, .467, .493 |
  | pre-only row-normalise by prev (L^-1/2 g, damp 1e-2·mean) | –, .243, .248, .275, .201 | .275, .245, .244, .268, .248 | –, .289, .283, .346, .219 | .366, .292, .279, .327, .290 |
  | L^-1/4 only (no column side) | .005, .210, .219, .228, .174 | .208, .219, .215, .225, .225 | .030, .259, .259, .320, .180 | .322, .258, .240, .293, .254 |
  | symmetric −1/4, eps → 0 | .005, .140, .162, .145, .130 | .172, .300, .252, .335, .296 | .030, .108, .099, .105, .099 | .167, .274, .241, .260, .279 |
  | **MODEL** (−1/4, 1e-4·max), PE5 f64 | **.005, .028, .035, .012, .005** | **.004, .016, .020, .013, .005** | **.030, .014, .013, .031, .005** | **.027, .007, .011, .029, .007** |
  | **MODEL, X0 rounded to fp16** | .004, .028, .035, .012, .005 | .004, .016, .020, .013, .005 | **.014, .007, .007, .015, .004** | **.014, .005, .006, .013, .005** |
  | fresh twin vs PE5(g_t) f64 [floor, no state] | .005, .026, .033, .005, .005 | .005, .017, .018, .005, .005 | .030, .015, .014, .031, .005 | .031, .008, .012, .031, .008 |

  ("–": the row-normalise rule divides by a zero previous A gradient at step 2.) The model sits on the
  floor of "PE5 applied to this oracle gradient" (the fresh twin vs PE5(g_t), which involves no state
  at all) at every step and rank, with one exception: **r8 step 5 (A .012, B .013 vs floor .005)**.
  At rank 32 the X0-fp16 version halves the model residual (steps 2/5: .030 → .014, = the FIRSTSTEP
  r32 step-1 floor .014), so the r32 "floor" of the f64 column is fp16 rounding of X0. The r8 floor
  at steps 3–4 (1.7–3.3%) is NOT reduced by X0 fp16 nor by the full fp16 fused emulation (r8 step 4 A
  0.0376 f64 vs 0.0374 fp16, exploratory e28); it is a property of PE on those gradients, not of state.
- **Leakage tests (exploratory, r8 steps 3–6, fit half):** adding w·g_t² or w·g_{t−2}² to the
  statistic (w = 1e-4…0.1) never helps; adding w·g_{t−1} to the orthogonalised vector (momentum leak
  at beta = 0, w = ±1e-4…1e-2) never helps; an extra absolute eps (1e-12…1e-8) on energies, on root
  energies, or in the PE normaliser never helps. EMA second moments of g² with beta2 0.5–0.999
  (pre-row only form) 0.43–0.51 vs 0.22 for the previous gradient alone (step 4 B).
- **Why A is bit-identical at step 2 but B is not:** at step 1 G_A = 0 exactly (B = 0), so A's stored
  momentum after step 1 is 0 — the same as a fresh state — and the rule takes the S = 0 branch in
  both cases. B's stored momentum is (1−beta)g_1 ≠ 0, so the continuing step is preconditioned and the
  fresh one is not. Zero-gradient step with beta1 = 0: m32 = 0 → PE(0) = 0 → no movement (as observed).

### Q2 — momentum form, state source, storage precision (`mom_*.txt`, odd half)

- **Source of the statistic = previous stored momentum, not previous gradient.** S2_r8_b0.9 medians
  per step 2..6, ema/prev-momentum 0.005–0.006 at every step; ema/prev-gradient 0.005, 0.105 (B),
  0.54/0.65, 0.55/0.75, 0.12/0.44. Same picture at beta 0.99 and 0.5.
- **EMA, not sum; not Nesterov.** At constant beta EMA and sum are proportional (identical scores).
  S3 (beta 0.9 → 0.99 at step 4): EMA 0.0045 A / 0.0045 B (r8), 0.014 / 0.0045 (r32); sum 0.26 / 0.44
  (r8), 0.52 / 0.50 (r32). Nesterov variants (fed = beta m + (1−beta) g after the update, or g + beta m)
  already fail at step 2: 0.071 (r8 b0.9, B) and 0.25–0.29 at steps 3–6.
- **Stored in bf16, orthogonalised in float32.** Rounding the stored momentum to bf16 while feeding the
  float32 value: r32 b0.9 steps 2..6 B 0.095 → 0.027, 0.0087 → 0.0040, 0.0086 → 0.0040, 0.0105 → 0.0043,
  0.0088 → 0.0041; A step 3 0.048 → 0.0092, step 4 0.032 → 0.0075, step 6 0.0072 → 0.0038; r8 b0.9 all
  steps 0.005–0.006 → 0.0043–0.0046. Rounding the FED value too is rejected (step 1 B r32 0.031 → 0.21).
  Consistent with D2's size fact (optimizer state = half the float32 weight bytes).
- **Zero-gradient steps (S4_r8_cont, R1; all 304 tensors), model with bf16 storage, no refit:**
  relF zero1/zero2/zero3 = 4.3e-3 / 4.7e-3 / 4.4e-3 (A), 4.3e-3 / 4.4e-3 / 4.2e-3 (B).
  cos(zero1, step2): predicted 0.81646 A / 0.99510 B, measured 0.81657 / 0.99510.
  cos(zero2, zero1): predicted 0.999994 / 0.999990, measured 0.999983 / 0.999981. (The D2 entry's
  "0.95" was A and B pooled.) The prev-gradient variant predicts zero2 = unpreconditioned PE(m)
  (relF 0.61 / 0.75) — rejected.
- **Rank-32 zero steps (S4_r32_cont)** cannot be simulated (no same-regime step-2 gradient). Measured
  cos(zero2,zero1) 0.982 A / 0.984 B; successive changes are nearly orthogonal (cos(Δ23, Δ12) −0.07),
  i.e. rounding-like, not a drift. A proxy built from the R2 trunk S2_r32_b0.9 (same texts) reproduces
  this only with bf16 storage: cos(zero2,zero1) 1.000000 (f32 momentum, f64 or fp16 PE) vs 0.9853 /
  0.9847 (bf16 storage, f64 / fp16 PE); rank-8 proxy 0.99999 (measured 0.99998). bf16 rounding of a
  0.9-scaled momentum perturbs the tiny singular directions that PE amplifies ~1000x; at rank 32 that
  is a 17% change of direction per zero step.
- Momentum EMA uses beta of the CURRENT call (S3 restored with a new beta: fits at the floor).

### Q3 — step-1 law in R2 (`r2_step1.txt`)

Same rule, same residual: r8 PE5 (X0 fp16) median 4.85e-3 (R2) vs 5.19e-3 (R1); fp16 fused emulation
5.55e-3 vs 5.68e-3; r32 1.40e-2 vs 1.39e-2 (fused 1.88e-2 both). Norm law within 1e-7 in both; A
unchanged in both. Only the backward pass changed at ~19:31.

### Caveats / open

- r8 beta1=0 step 5 sits at 1.2–1.3% vs a 0.5% floor (only exception). Not explained by leakage,
  abs eps or precision variants tried.
- The per-step floor itself (fresh twin vs PE5(g_t)) at r8 steps 3–4 (T1/T2 batches) is 1.7–3.3%;
  FIRSTSTEP's PE emulation (f64, X0-fp16 or full fp16) does not capture it. Not a state effect.
- epsR is not identified; the "max" normaliser and exponents −1/4 are fitted on rank 8/32, Qwen3.5-4B,
  three texts only. Bias correction is invisible (every quantity is scale-invariant) — not tested.
- Rank-32 zero-step reproduction is via a proxy trunk, not the S4_r32 chain itself.
- r32 lr 1e-2 non-proportionality (S6) not investigated; plausibly the same bf16/PE sensitivity.

### Measurements that would settle the rest
1. S4 at rank 32 in the current regime WITH an oracle at step 2 (2 normal steps, 3 zero steps) → direct
   test of bf16 storage via zero-step predictions (~4 clients).
2. Upload chosen weights so the previous gradient has a few rows of exactly known, widely spread energy
   (incl. rows ≪ 1e-4·max), then one continuing step → pins epsL form and epsR.
3. beta1 changed at a ZERO-gradient step (m = beta' m exactly) → separates EMA (1−beta) placement from
   any bias correction.

---

## PREDICT-01 — out-of-sample predictions of the MULTISTEP-01 rule (P1–P8) — 2026-11-10 (measurement agent)

**Setup.** Qwen/Qwen3.5-4B attention-only LoRA unless stated, seed 0, Dimuon lr 1e-3 b0.9 unless stated, texts
T0/T1/T2 (`tk.TEXTS`). Library `pipeline/p3_lib.py` = `s_d2.py` re-pointed to `runs/P3/` (own manifest; oracle O1 =
AdamW lr 1e6 eps 1, setting `P3`; repeats `P3b`, `P3x1..`). Rule `pipeline/p3_rule.py` (exactly the stated rule;
PE5 with X0 rounded to fp16 = "x0f16" unless stated; momentum stored bf16). Drivers `p3_collect.py`, `p3_solo.py`;
predictions `p3_predict.py` → `runs/P3/pred/<exp>/` (npz + json with write time and sha; computed from oracle
GRADIENTS and call parameters only, never from measured states); scoring `p3_score.py` → `runs/P3/score_*.txt|json`;
figures `p3_figs.py`. Residual = ‖U_pred − U_meas‖_F/‖U_meas‖_F per tensor (prediction includes the norm law),
median / p90 / max, A and B separately. P1a, P1b, P3 (zero step) and P4 predictions were written BEFORE the branch
existed (pre-registered, times in the json); all others blind (gradients only). Exploratory scratch scripts
(`e_*.py`) are in the scratchpad, numbers marked "exploratory".

**Regime.** Same regime as D2 R2 throughout: P6 step 1 → `3362e501…` (= D2 R2), init oracle bit-identical to D2
`0ff0…_O1r2`; P2/P5 trunks reproduce D2 R2 states `6b6b…`, `aab5…`, `7bde…`, `48f8…` bit-for-bit; end-of-session
new client step 1 → `3362e501…` again (`score_regime.txt`).

### Finding 0 (affects everything): neither the oracle gradient nor the Dimuon step is a function of (state, batch, params) alone
- **Gradient.** Repeated oracles at the same state/text: r32 state 1/T1 — P3 (= D2's stored G, bit-identical) vs P3b:
  23.4M of 28.7M elements differ; r8 b0 state 3/T2 — P3 vs P3b 5.85M of 7.18M differ (P3b = D2's stored G, 0 diffs);
  r32 state 1/T0: the concurrent oracle differs from a solo one at 20.2M elements. At other states repeats are
  bit-identical (r8 b0 state 2/T1 ×3 incl. D2; P6 state 1/T1 ×3: one concurrent, two solo on different pools).
  The init gradient (B = 0) never varied. Loss is always bit-identical. The server metric `clock_cycle:unique` falls in
  two clusters (~7.20M, ~7.53M; gpt-oss ~5.43M/7.96M); in the P5 cases the gradient version follows the cluster, but
  the cluster does NOT predict it in general.
- **Step.** From one restored state (P6 ck1, T1) the unclipped step 2 gave two different results: trunk (X,
  `d7d2…`) and two repeats (Y, `92f2…`, one of them strictly solo). Every lora_A tensor falls cleanly in X or Y
  (149–152 of 152 bit-identical to one of them); the state-1 oracle that both share is verified identical, so the split
  is in the optim step path. X/Y does not follow c, lr, pool, or co-scheduling (P1a c 0.5 and 0.3 shared a cycle; P4
  lr 5e-3 and 6e-3 shared a cycle and gave the other variant). X vs Y differ by relF 3.3% (A, state 1); the
  difference lies 59–98% inside span(g_A); a free r-side diagonal fits X only to 1.5–3.9% (Y: 0.3%); bf16/fp16-fed
  momentum or bf16 PE do not reproduce X (exploratory). **X is not characterised.** The rule fits Y at the floor.
- Consequence: the per-step "floors" quoted in FRESH-01/MULTISTEP-01 for T1/T2 (1.7–3.3%) and the S6 "lr
  anomaly" are at least partly this variance (see P4, P5). Run oracles solo; even solo does not guarantee one variant.

### P1 — clipping (`score_p1a.txt`, `score_p1b.txt`, `score_solo.txt`; fig `p3_clip_storage.png` a)
- P1a, state 1 (T1, N_all 94.55): pre-registered rule with g → c·g, c = clip/N. Fitted c (B, every 4th tensor):
  **0.7021, 0.5005, 0.3011, 0.1000, 0.0100** for nominal 0.7, 0.5, 0.3, 0.1, 0.01. Residual (rule, c) B median
  1.25e-2 / 1.15e-2 / 1.01e-2 / 4.5e-3 / 4.5e-3 vs null (unclipped prediction) 0.167 / 0.272 / 0.443 / 0.760 / 1.06;
  unclipped c = 1 1.31e-2. A: 2.83e-2 (X-variant runs) or 4.9e-3 (Y-variant runs). N_all vs N_B indistinguishable
  here (‖G_A‖ small).
- P1b, state 6 (T0): N_A 21.48, N_B 242.81, N_all 243.76; t1 243.28, t2 72.21, t3 10.74. Per-type and per-tensor
  N rejected: t2 A residual 0.230 (per-type predicts A unclipped) vs 6.8e-3 (all); t3 A 0.232 vs 8.1e-3. All vs B-only
  by fitted c on 4 disjoint quarters of the A tensors: t2 0.29561–0.29609 (all → 0.29626, B-only → 0.29741); t3
  0.04392–0.04412 (all → 0.04406 inside the range, B-only → 0.04423 outside). B quarters are noisier (t3 0.04313–0.04412).
  **Favours N = global norm over all LoRA tensors (A and B).** The t1 bit-identity test is inconclusive because of
  Finding 0: concurrent t1 run = solo unclipped run bit-for-bit (as if c = 1), solo t1 run differs from it (A relF
  1.2%, fits "all" 6.5e-3 vs "none" 7.7e-3); predicted size of the c = 0.998 effect is only A 3.0e-3 / B 1.7e-3.
- **Verdict: g → c·g before the momentum update holds (fitted c within 0.3% of clip/N_all at every level).
  N over all LoRA tensors: favoured, not decisively.**

### P2 — zero-gradient steps at rank 32 (`score_p2.txt` pre-registered oracle, `score_p2_solo.txt`; fig b)
- Trunk T0, T0, zero ×3 (b0.9), re-run solo (`P2s_r32`): every state identical to the concurrent run. With the solo
  oracle at state 1 (the concurrent one was a different gradient version): bf16 storage, x0f16 — median relF
  A/B: step1 –/1.40e-2, step2 1.42e-2/1.43e-2, zero1 1.52e-2/1.48e-2, zero2 1.49e-2/1.45e-2, zero3 1.46e-2/1.42e-2
  (floor = step 1, 1.40e-2). f32 storage: step2 B 9.5e-2, zero1 0.19/0.19, zero2 0.25/0.26, zero3 0.30/0.30.
  f64 PE instead of x0f16: 2.5–2.9e-2 (bf16 storage).
- Cosines (median, measured | bf16 | f32): cos(zero1,step2) A 0.80841 | 0.80856 | 0.82251, B 0.98125 | 0.98142 |
  0.99701; cos(zero2,zero1) A 0.98181 | 0.98199 | 1.000000, B 0.98424 | 0.98439 | 1.000000; cos(zero3,zero2)
  A 0.98305 | 0.98326 | 1.000000, B 0.98526 | 0.98543 | 1.000000.
- With the concurrent (wrong-version) oracle the same predictions were 0.13–0.35 (kept in `score_p2.txt`).
- **Verdict: bf16 storage holds (at the step-1 floor, zero-step cosines to ≤2e-4); f32 storage fails.**

### P3 — beta changed at a zero-gradient step (`score_p3.txt`)
- From P6 state 2: zero step at β′ = 0 → max|ΔW| = 0 (predicted 0). β′ = 0.5 / 0.9 / 0.99 → identical within
  max|diff| 3.2–4.0e-7 (|U| entries ~1.7e-4); 0.5–2.0M differing elements; relF between them median 2–3e-5, p90
  0.4–1.1e-3, max 3e-3. The rule predicts relF median 0, p90 ~1e-5 (bf16 rounding only): **nearly, not bit-for-bit;
  the measured β′-dependence is ~100× the rule's rounding prediction (cause not identified).** Zero-step residual vs
  rule A 3.05e-2 / B 1.31e-2 (inflated by the step-2 variant issue, see P6).
- Next normal T2 step (β 0.9), oracle at the post-zero state: rule vs null "β′ ignored (0.9)": β′=0 A 3.0e-2 / B 1.7e-2
  vs 1.02 / 1.12; β′=0.5 1.4e-2 / 1.1e-2 vs 0.24 / 0.24; β′=0.99 1.8e-2 / 1.1e-2 vs 9.7e-2 / 7.9e-2; β′=0.9 1.7e-2 /
  1.2e-2. **Verdict: holds** (stored m = bf16(β′S), next step uses it with (1−β) weighting; β′=0 empties momentum →
  the next step is the fresh law).

### P4 — learning rate (`score_p4.txt`, `score_p4_solo.txt`, `score_solo.txt`)
- r32 from P2 state 2 (T1), restored with optimizer: U(lr)·1e-3/lr takes exactly two values. {4e-3, 5e-3, 6e-3, 1e-2}
  are proportional to each other and to a SOLO pair (solo U(1e-2)/10 vs solo U(1e-3): A 1.6e-5, B 9e-8 = float32
  rounding of W+ΔW); {1e-3, 8e-3} (concurrent) are the other variant, 3.7e-2 A / 1.7e-2 B away. No threshold in lr,
  per-tensor ‖lrU‖/‖W‖ (A 0.0017–0.0173) or global ‖lrU‖ (0.10–0.99). Fresh-state r32 lr 1e-2 vs 1e-3: proportional
  (A 1.6e-5, B 8e-8). With solo oracles the rule fits the proportional group at A 5.9e-3 / B 3.5e-3 (x0f16;
  f64 9.2e-3/3.9e-3; fused fp16 7.3e-3/3.9e-3) and the other group at 3.6e-2 / 1.7e-2. r8 (state 2, T2): lr 1e-2…5e-2
  all mutually proportional, all 1.7e-2 A / 7.8e-3 B from the trunk's lr-1e-3 step.
- **Verdict: there is no lr anomaly; the update is exactly proportional to lr. The D2/FRESH-01 "r32 lr 1e-2" effect is
  the two-variant step nondeterminism (Finding 0).** No PE precision/orientation variant fits the other variant better.

### P5 — the three FRESH-01 mismatches (`score_p5.txt`)
- r32 state 1/T1: fresh step re-taken → same after-state as D2 (`773d…`). Oracle P3 = D2's G (bit-identical) →
  residual A 0.140 / B 0.070 (the old mismatch); oracle P3b (different version) → **7.4e-3 / 5.0e-3**.
- r8 b0 state 3/T2: today's fresh step `81ca…` ≠ D2's `1828…`. Today's step vs PE5(G_P3b) 4.8e-3 / 5.1e-3 (vs G_P3
  3.2e-2/1.8e-2); D2's step vs PE5(G_P3) **4.5e-3 / 5.2e-3**. Each fresh step matches one of the two gradient versions.
- r8 b0 state 2/T1: all three oracles bit-identical (incl. D2), fresh step = D2's; residual 2.6e-2 / 1.6e-2 — not
  resolved (no second gradient version seen; could be the step variant).
- **Verdict: gradient mismatch, not a different map** (2 of 3 resolved to the floor by a recomputed oracle).

### P6 — 12-step horizon, rank 8 (`score_p6.txt`, `p6_table.json`; fig `p3_horizon.png`)
- Rolled from oracle gradients only (bf16 storage). Median relF A / B, steps 2…12: 2.8e-2/1.3e-2, 2.0e-2/1.2e-2,
  9.4e-3/1.0e-2, 8.7e-3/1.0e-2, 8.5e-3/1.0e-2, 6.4e-3/9.4e-3, 7.3e-3/9.4e-3, 7.0e-3/9.0e-3, 8.1e-3/9.5e-3,
  6.7e-3/8.4e-3, 7.0e-3/8.2e-3 (step 1 B 4.8e-3). Nulls: no-state 0.67–1.24, previous update 0.25–1.4.
  Fresh-twin floor: step 4 (T0) 5.0e-3/4.8e-3, step 8 (T1) 1.6e-2/1.6e-2, step 12 (T2) 2.3e-2/1.8e-2.
- f32 storage at r8: indistinguishable (B 7.9e-3 vs 8.2e-3 at step 12, slightly lower).
- **Verdict: no error accumulation over 12 steps** (residual falls after steps 2–3 and stays ≤1% median, p90 ≤1.4%).
  Steps 2–3 elevated A (2–3%) are where the step-variant/gradient-version issue was seen.

### P7 — openai/gpt-oss-20b (`score_p7.txt`)
- Rank 8 attention LoRA: 192 tensors; lora_A (8, 2880) and (8, 4096); lora_B (4096, 8), (512, 8), (2880, 8); r < every
  dimension. Init oracle: snap residual 5.0e-6, G_A exactly 0 (all 2,445,312 A elements), ‖G_B‖ 345.5.
  Step 1: A unchanged, norm/law 0.9999999–1.0000000 (same constant lr·√r·(r/32)^0.175).
- No refit. Median relF A / B: step1 –/4.95e-3; step2 4.6e-3/4.3e-3; step3 5.3e-3/4.6e-3; step4 4.6e-3/4.6e-3;
  step5 4.6e-3/4.5e-3 (p90 ≤ 8.5e-3, max ≤ 1.8e-2); nulls 0.31–1.31. β=0 at state 2: continuing 4.9e-3/4.8e-3, fresh
  4.8e-3/4.6e-3, continuing vs fresh differ by 0.75/0.79. **Verdict: holds at the step-1 floor, no refit.**

### P8 — MLP + unembed adapters, Qwen rank 8 (`score_p8.txt`)
- 498 tensors: attention as before + gate/up (lora_B 9216×8), down (2560×8), unembed_tokens (lora_B 248320×8,
  lora_A 8×2560). Norm law 1.000000 for every type at every step. Rule median relF per type, steps 1–3: 3.5e-3–8.0e-3
  (A and B), unembed B 4.8e-3 / 8.0e-3 / 3.7e-3; nulls 0.35–1.3. **Verdict: holds for MLP and unembed.**

### Changes I would make to the rule
- None to the equations. Clipping: write c = min(1, clip/N), N = ‖G‖ over all LoRA tensors (A and B), applied to g
  before the momentum update. Add to the model: the server step and the backward each have ≥2 numerical variants
  per call (Finding 0), so a pairing of a step with an oracle from another call is only valid when both are the same
  version; compute oracles solo and repeat them where residuals are high.

### Caveats
- X variant uncharacterised; which variant a call takes is not observable beforehand. Some P1b/P3/P6 residuals are
  inflated by it. Solo-oracle predictions (`*_solo`) were computed after the measured states existed (still from
  gradients only). To fit the disk I deleted the f64/fused32/fusedbf16 prediction arrays after scoring (json metadata
  kept). The clip-N verdict rests on fitted-c spreads, not on bit identity.
- Usage (`runs/calls.jsonl`, events after 21:50 UTC): 85 training clients (7 new, 38 weights-only AdamW oracles,
  9 weights-only Dimuon fresh-state, 31 restored with optimizer), 116 forward_backward + optim_step, 201 exports (all deleted after download), 31
  save_state checkpoints, all deleted; final listing = the owner's 10 checkpoints, 6.92 GB (`runs/P3/final_ckpt_listing.json`).

---

## COMPARE-01 — our frozen rule vs public sources — 2026-11-10 (comparison agent; no Tinker calls)

**Question.** What do public sources say about Dimuon? Where do they agree or disagree with our rule? Do their
statements explain anything we left open? Full write-up and source list: `runs/COMPARE/compare.md` (retrieved
2026-11-10).

**Sources.**
- Official: Tinker docs Optimizers page; SDK docstrings; the announcement (republished); "LoRA Without Regret".
  None of these gives the rule. `tinker-cookbook` @8486b8d never mentions Dimuon.
- Third-party, independent: `anpaure/dimuon` @d6570bb (full PyTorch rule) and `romeerp/dimuon-replay` @d32fbee
  plus its site (precision fingerprinting).
- Third-party, not independent: `catid/dimuon` (its `dimuon.py` is anpaure's, uncredited) and theblackcat102's blog
  (ablates anpaure's code).
- Papers checked: Polar Express 2505.16932, PoLoRA 2607.17620, Shampoo 1802.09568, Old Optimizer New Norm 2409.20325.

**Agreement.** Every equation in our rule matches anpaure, and romeerp matches it where it covers the same point:
EMA momentum in fp32 stored as bf16; row/column energies of the previous stored momentum; −1/4 on four sides;
relative 1e-4 ridge; identity when the momentum is zero; PE5 with safety 1.01 and X0 normalised in fp32 then cast to
fp16; norm lr·√r·(r/32)^0.175; global clip before the EMA; no weight decay; no A/B coupling. anpaure's relative ridge
(e/max + 1e-4) is our l + 1e-4·max up to a scalar that cancels. Running anpaure's `dimuon.py` unchanged on the P6
chain reproduces our residuals step for step: fp64 identical to ≤2%, fp16 default within 10% (`anpaure_p6.txt`).

**Disagreement (one): orthogonaliser arithmetic.** romeerp's Horner form X·(aI + R(bI + cR)), with R = XᵀX and
a separate fp16 store of cR, beats our expanded fused form (`cmp_horner.py`, D1, all 152 modules). Median relF:

| r | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| ours | 5.70e-4 | 6.34e-3 | 6.28e-3 | 5.68e-3 | 6.58e-3 | 1.87e-2 | 2.46e-2 |
| Horner | **6.3e-8** | **9.1e-8** | 1.76e-4 | 1.83e-3 | 2.79e-3 | 8.07e-3 | 1.03e-2 |

- Bit-identical modules: 149/152 at r1 and 91/152 at r2 (ours: 43% / 2% of entries). Dropping the cR store loses
  this (r1 3.96e-4).
- Resolves FIRSTSTEP-01 item 7. The deviation from "X0 fp16 exact after" is, data vs Horner: r2 6.59e-3 vs 7.09e-3,
  r4 6.53e-3 vs 6.55e-3, r8 5.19e-3 vs 4.99e-3, r16 5.62e-3 vs 5.66e-3, r32/64 equal. Our emulation gave 2.6–3.6e-3.
- Split-K fp16 partial sums on the Gram (romeerp's GPU explanation, our emulation of it) fit worse at every r ≥ 2.
- Horner also helps preconditioned steps: P6 step 2 B vs Y goes 4.6e-3 → 1.4e-3, and step 1 B 5.1e-3 → 2.0e-3.
  Beyond step 2 the ~1% chain residual and the T1/T2 fresh-twin floors (1.6–2.3%) do not move, so those are not
  orthogonaliser precision. That is consistent with PREDICT-01's gradient-version explanation.

**Open items closed or narrowed.**
- **Column damping is identified at rank 32** (`cmp_damping.py`, P2s/P4s solo, step 3, row damping fixed). Median relF
  A / B by eps_c·max: 0 → 9.4e-3 / 4.0e-3; 6e-5 → 7.7e-3 / 3.8e-3; **1e-4 → 5.9e-3 / 3.5e-3**; 1.5e-4 → 8.1e-3 / 3.9e-3;
  1e-3 → 1.3e-2 / 5.5e-3. Relative to the mean (1e-4) is worse: 9.1e-3 / 4.0e-3. This matches the public 1e-4·max.
  One state, one step.
- **Variant X: no public description reproduces it.** Every orthogonaliser variant, including Horner and split-K
  2/8/32, sits at relF 2.8–3.0e-2 from X in A (= X vs Y 2.83e-2). Fits to Y: 3.8–5.8e-3. anpaure says "clients can
  compute different gradients from identical inputs", which would fit X being a backward-pass variant, but that is
  untestable offline here.
- **(r/32)^0.175: no source explains it.** All public formulas are the identical empirical expression. anpaure has it
  to 1.2 ppm over ranks 1–128; romeerp says the origin is "a mystery". Tinker exposes no alpha, so it cannot be told
  apart from (α/r)^−0.175.
- Bias correction and the norm-floor form remain unaddressed everywhere.

**Theory.** These are my inferences; the sourced pieces are cited in compare.md §4.
- PoLoRA eq. 14: steepest descent under ‖P^{1/2}ΔW Q^{1/2}‖₂ is −P^{−1/2} msign(P^{−1/2} G Q^{−1/2}) Q^{−1/2}.
  Dimuon is that form with P = diag(l)^{1/2} and Q = diag(c)^{1/2}.
- That is a diagonal, unaccumulated, one-step-lagged Shampoo preconditioner (Shampoo's L^{−1/4} G R^{−1/4}) wrapped
  around Muon.
- At r = 1 it reduces to Adam with β₂ = 0, using the previous momentum as the second moment and an ε relative to
  the max.

**Caveats.** All tests are offline on existing D1/P3 data with our oracle gradients. The Horner emulation is
CPU-style (fp32 accumulation, fp16 stores), not GPU kernels. The damping test is one rank-32 state. Third-party code
lives only in the scratchpad (not in the repo). anpaure's file was imported unchanged after reading all 104 lines.

**Files.** `pipeline/cmp_horner.py`, `cmp_variantX.py`, `cmp_anpaure.py`, `cmp_damping.py`;
`runs/COMPARE/compare.md`, `horner.{txt,json}`, `variantX.{txt,json}`, `anpaure_p6.{txt,json}`, `damping.{txt,json}`.

## COORD-CHECKS (coordinator's own re-computations, 2026-11-10)

Independent float64 numpy re-implementations written inline by the coordinator (not the agents' code),
expanded-form Polar Express 5 steps, safety 1.01, no fp16 emulation; residual = relative Frobenius per
tensor with the prediction scaled to the measured norm. Scripts were run as inline heredocs and are
reproduced by the equations in the paper; they are not saved as files (caveat).

- D2 sanity: rank-8 per-tensor update norms at `S2_r8_b0/step3` 0.00221914 (A and B) vs law 0.00221914;
  b0 fresh-vs-continuing step 2: A 0, B 0.7334; step 3: A 0.6126, B 0.6813; zero1 vs previous update
  cosine (A, B pooled) median 0.952; zero2 vs zero1 0.99998; r32 lr 1e-2 vs 10x lr 1e-3: A 0.0423, B 0.0178.
- Fresh-state law, `S1_r8_fresh_ck2_t0/step` (R1): A median 0.00504 (max 0.0190), B 0.00509 (max 0.0140).
- Full rule, `S2_r8_b0.9/step3` (R1), momentum rolled from step 1 with bf16 storage: A median 0.00450
  (max 0.0131), B 0.00449 (max 0.0116); unconditioned PE(m3): A 0.611, B 0.763.
- Second model, `P7_r8/step3` (gpt-oss-20b, 192 tensors, shapes (2880|4096|512) x 8): A 0.00534
  (max 0.0118), B 0.00461 (max 0.0150); per-tensor norms 0.00221914.
- Billing: the first billing script truncated output; corrected. Study training cost priced so far ~$0.003;
  $7.88 of sampling on thinkingmachines/Inkling-Small on 2026-11-10 is not from this study.

---

## BIASCORR-01 — what is fed to the orthogonaliser at step 1 (S = 0)? — 2026-11-10 (measurement agent)

**Question.** At step 1 is the orthogonalised tensor g, f32 (1−β)·g, a bias-corrected m/(1−β), a lerp form, or a
reduced-precision copy? Earlier evidence: rank 8, β 0 ≡ 0.5 bit-identical, β 0.9 differs in 5 fp16 entries of one
tensor; FIRSTSTEP-01 item 9 estimated ~818 flips for plain f32 EMA and rejected it.

**Setup.** Qwen/Qwen3.5-4B attention-only, seed 0, `tk.TEXTS[0]`, weight 1, Dimuon lr 1e-3, no clip, B0 = 0, ranks 1
and 2. One new client per run, strictly sequential. β ∈ {0, 0.5, 0.75, 0.9, 0.99, 0.999, 0.3, 0.7} per rank
((1−β) exact powers of two at 0.5/0.75); oracle G (AdamW lr 1, eps 100, bf16-snapped, as D1) before and after the
β runs at each rank, one extra rank-1 oracle, one repeat of r2 β 0.9. Only lora_B kept (float32; = the update).
Emulation (`bc_lib.py`): candidate M = f(g, β) → X0 = fp16(M/(1.01·‖M‖ + eps)) → COMPARE-01 Horner PE5 → norm law.
Candidates: `g`; `ema_a64` f32(1−β)·g; `ema_a32` (f32(1)−f32(β))·g; `lerp_*` torch-lerp semantics (incl. FMA);
`*_bcdiv*` / `*_bcmul64` bias-corrected (÷ f32(1−β) in f64- or f32-computed form, or × f32(1/(1−β)));
`ema_a64_bf16` / `_fp16` (fed after a bf16/fp16 cast); each with eps 1e-7 and 0. `bc_flips.py` / `bc_pairs.py`
also vary the normaliser arithmetic (norm by numpy pairwise f32 / f64 / sequential f32 / BLAS dot; M/(n·1.01),
M·(1/(n·1.01)), (M/n)/1.01).

**Finding 0 again, now at B = 0: two gradient versions at each rank.** r1: oracle1 ≡ oracle2 but r1 β 0/0.5/0.75/0.9
were stepped on a different gradient (emulation relF 9.6e-3 vs 6.2e-8); oracle3 reproduced it (6.2e-8). r2: oracle1
and oracle2 differ in all 152 tensors (732,877 entries, median relF 9.7e-3); runs fall in both (b0.5, b0.3 on
oracle1; the rest on oracle2). Versions interleave in time (r1 switched between 23:32:14 and 23:32:36 UTC; r2 went
v2, v1, v2, …, v1, v2). PREDICT-01 had "the init gradient never varied" — no longer true. Every run below is scored
against the oracle that reproduces it (`analysis.txt` lists the pairing). Within a version the step is
deterministic: r2 β 0.9 repeat bit-identical (0 entries differ).

**Data, within one gradient version (no emulation involved; `analysis.txt`, `pairs.txt`).**
- r1 (grad oracle3): β 0.5, 0.75 ≡ β 0 bit-for-bit; β 0.9 differs in 1 tensor (`layers.16.linear_attn.in_proj_v`,
  44 X16 entries, all 4096 float32 entries).
- r2 (grad oracle2): β 0.75, 0.999, 0.7 ≡ β 0 bit-for-bit; β 0.9 differs in 1 tensor (`layers.26…in_proj_v`, 59 X16
  entries), β 0.99 in 1 other tensor (`layers.22…in_proj_v`, 98 entries).
- r1 (grad oracle1, no β 0 run): β 0.3 ≡ 0.7; β 0.99 and 0.999 differ from them in 2 and 4 tensors.
- r2 (grad oracle1): β 0.5 ≡ 0.3.
- Every tensor whose float32 update changed also changed its X16; no tensor changed only in the final scalar.
  So β acts only before the fp16 cast of X0.

**Candidates × β (`table.md`; cell = X16 entries mispredicted / tensors predicted exactly, of 152).**

| candidate | r1 β0 | r1 β0.9 | r1 β0.99 | r1 β0.999 | r2 β0 | r2 β0.99 | r2 β0.999 |
|---|---|---|---|---|---|---|---|
| g, eps 1e-7 | 4091/150 | 4135/149 | 2134/149 | 2142/149 | 170840/86 | 170938/85 | 170840/86 |
| ema f32(1−β)g, eps 1e-7 | 4091/150 | 4157/147 | 8254/141 | **65966/56** | 170840/86 | 188329/73 | **255706/17** |
| ema, eps 0 | 4091/150 | 4134/149 | 2142/149 | 2163/148 | 170810/87 | 170908/86 | 170810/87 |
| bias-corr ÷f32(1−β), eps 1e-7 | 4091/150 | 4135/149 | 2134/149 | 2142/149 | 170840/86 | 170938/85 | 170840/86 |
| bias-corr ÷f32(1−β), eps 0 | 4091/150 | 4122/150 | 2134/149 | 2142/149 | 170810/87 | 170908/86 | 170810/87 |
| fed bf16(ema) | 4091/150 | 386154/0 | 389356/0 | 392316/0 | 170840/86 | 866940/0 | 861461/0 |
| fed fp16(ema) | 4091/150 | 228183/0 | 253109/0 | 265022/0 | 170845/84 | 659187/0 | 723728/0 |

(Full table, all 13 forms × all 17 runs × both eps: `table.md`. The β 0 column is the emulation floor: r1 150/152,
r2 86–87/152 tensors exact, as in COMPARE-01. Float32 entries are never predicted exactly in any tensor, by any
candidate: the final-scalar stage is not bit-exact (r1 β0 199,783 of 458,752 float32 entries equal; I tried 26
op orders for −lr·√r·(r/32)^0.175·X/‖X‖ and none is exact; for 7 r1 tensors tested, no single float32 multiplier fits). So the
discrimination is done on X16.)

**Verdict.**
- **Rejected:** (a) M = g for every β (also the sum form βS + g, S = 0): the data change deterministically at β 0.9
  (r1, r2, repeat) and 0.99 (r2); (b) M fed after a bf16 or fp16 cast: wrong in every tensor at every non-power-of-two
  β; (c) plain EMA (any of a64/a32/lerp) **with** an additive normaliser eps ≥ 1e-8: at β 0.999 it predicts 120 changed
  tensors at r2 (1e-8: 24, 1e-9: 1; data 0) and 65,966 mispredicted entries at r1 (`eps_scan.txt`).
- **Not rejected, not separable:** plain f32 EMA with eps ≲ 1e-12 (or none; 1e-10 and 1e-9 already predict 1 change at r2 β 0.999, data 0), and bias-corrected EMA (eps anything ≤ 1e-7).
  Both predict the observed pattern: no change at exactly representable (1−β), rare one-tensor changes elsewhere,
  none of them gross. Predicted changed tensors over the 8 within-version β-vs-0 comparisons, numpy/f64 norm
  variants: plain EMA 1–6, bias-corrected 0–3, data 3 (`rate_by_variant.txt`). Too few events to choose.
- **Bit level: undetermined.** No (form × normaliser) combination reproduces all three distinct flip events without
  false positives (best: `bias-corr ÷(f32(1)−f32β) after f32(1−β)g`, f64 norm: 2 TP, 0 FP, 2 FN of 4 β-vs-0 tensor
  events, none of the TP exact; `pairs.txt` over all within-version pairs: best 15 TP / 3 FP / 12 FN of 27).
  `ema_a32` (or `lerp_a32`) with pairwise norm and eps 0 reproduces the r1 β 0.9 event exactly (all 44 entries) but
  also flips 2 tensors the data does not. Which tensors flip is decided by the last ulp of the f32 norm, i.e. by the
  GPU reduction order, which we do not model; so these 1–3 tensor events cannot identify the form.
- **FIRSTSTEP-01 item 9 corrected:** its "~818 X0 flips, plain EMA rejected" used eps 1e-7 in the normaliser
  (`fs_ns16.x0_f16`). Recomputed on D1 rank 8 at β 0.9: plain EMA 818–986 X0 flips with eps 1e-7, 120–121 in 2–3
  tensors with eps ≤ 1e-8; bias-corrected 0–6 in 0–1 tensors; data 5 in 1. The rejection was of the eps, not of EMA.
- **Step 2 (offline, D2, `step2.txt`, no new calls):** the STORED momentum is not bias-corrected. Storing
  S = bf16(g1) instead of bf16((1−β)g1) gives step-2 lora_B median relF 0.114 vs 0.0046 (S2_r8_b0.9, R1), 0.107 vs
  0.0043 (r8 b0.99, R2), 0.171 vs 0.014 (r32 b0.9, R2). A bias correction, if present, is applied only to the copy
  that is orthogonalised; a scalar there cancels except at the rounding level, so step 2 cannot separate it either.
  I did not spend clients on a step-2 run.

**Caveats.** The X16 recovery and Horner emulation are CPU float32/fp16 (COMPARE-01), bit-exact for 150/152 (r1) and
86–87/152 (r2) tensors at β 0; changes in the other tensors are scored only coarsely. Two gradient versions, one
text, one seed. The "eps ≲ 1e-12 under plain EMA" bound is from one tensor count at r2 β 0.999 with numpy norms.
The scale stage (float32) is unidentified, so "entries predicted exactly in float32" is ~43–46% for every candidate
and carries no information about the form.

**What would decide it.** A gradient with large dynamic range per tensor (‖g‖ ≪ 1, e.g. loss weight 1e-6) at
β 0.999/0.9999: an additive eps would then act even under bias correction, and under plain EMA (1−β)·g would approach
eps; or a step that makes the norm's last ulp irrelevant (not available through the API).

**Usage.** 22 training clients (1 smoke = r1_oracle1, 19 in the main run, 1 extra oracle), 22 forward_backward +
22 optim_step, 22 exports — every export's server checkpoint deleted right after download (22/22 `delete` events,
0 failures). No save_state. Final listing = the owner's 10 checkpoints, 6.92 GB (`runs/BC/final_ckpt_listing.json`).
Billing for these calls had not posted at report time (last bucket 21:00 UTC).

**Files.** `pipeline/bc_lib.py` (candidates + emulation), `bc_collect.py`, `bc_extra.py` (Tinker), `bc_analyze.py`,
`bc_flips.py`, `bc_pairs.py`, `bc_eps.py`, `bc_step2.py`, `bc_table.py` (offline). `runs/BC/`: `r{1,2}_b*.npz`,
`r{1,2}_oracle*.npz` + `.json` (model_id, metrics, A sha, export path, deleted flag), `analysis.{txt,json}`,
`table.md`, `flips.{txt,json}`, `pairs.{txt,json}`, `rate*.txt`, `eps_scan.txt`, `step2.txt`, `collect.log`,
`extra.log`, `final_ckpt_listing.json` (42 MB total).

## BEHAVE-01 — behavioural comparison, Dimuon vs AdamW, losses only — 2026-11-13 (measurement agent; coordinator re-computed the block A/B/C end-point and mean losses from the json files on 2026-11-14 and they agree)

**Question.** H1: is Dimuon's best lr ≈ 30–90× AdamW's, with loss-vs-lr curves coinciding after that rescale?
H2: at constant lr, late in training on a memorisable set, does Dimuon (fixed-norm step) sit on a noisy floor that
AdamW does not, and does linear decay remove it? H3: does rank 4 → 32 help Dimuon more than AdamW?

**Setup.** Qwen/Qwen3.5-4B, LoRA attn+MLP (train_unembed=False), seed 0. Dataset (`pipeline/bh_lib.py`): a made-up
language "Velish", 8 nouns × 8 adjectives; prompt `Velish for '<adj> <noun>':`, answer ` <noun-word>-<adj-word>.`
(order swapped vs English). 48 train / 16 held-out combinations (seeded split, every word seen in train), so
held-out loss measures composition. Loss only on answer tokens (≈5 tokens/example, 83 held-out tokens). Step =
forward_backward on a fixed-order minibatch of 8 (6 minibatches, cycled) + optim_step. Train loss = loss:sum/answer
tokens of that minibatch before the step; held-out = `forward` on all 16 every 10 steps (step 0 = 7.60 for every
run). AdamW = SDK defaults β 0.9/0.95, eps 1e-12, wd 0, no clip. Dimuon β1 0.9, no clip. "Best lr" = lowest
mean train loss over all 60 steps (a speed metric; end-point losses are ~0 for several lrs and spiky, see noise).

**Noise floor (repeat of Dimuon 1e-2 r8, `R_dimuon_lr0.01_r8`).** Steps 1–11 bit-identical; diverges from step 12
(1.1969 vs 1.1974). Mean train loss 0.768 vs 0.817 (6%); last-6 train 0.0008 vs 0.022; held at step 60 0.0030 vs
0.0005. So: mean-train differences under ~10% and end-point differences under ~10–30× are within run-to-run noise.

**Block A (rank 8, 60 steps)** — mean train loss over 60 steps / held-out at 60:
| AdamW lr | mean | held60 | | Dimuon lr | mean | held60 |
|---|---|---|---|---|---|---|
| 3e-5 | 2.095 | 0.220 | | 1e-3 | 4.147 | 1.70 |
| 1e-4 | 0.943 | 0.0062 | | 3e-3 | 1.722 | 0.132 |
| **3e-4** | **0.589** | 2e-5 | | **1e-2** | **0.768** | 0.0030 |
| 1e-3 | 0.865 | 0.0019 | | 3e-2 | 0.910 | 0.040 |
| 3e-3 | 5.531 | 2.21 | | 1e-1 | 6.278 | 1.32 |
Best lr AdamW 3e-4, Dimuon 1e-2 → **measured ratio ≈ 33×** (grid resolution ½ decade, so "between ~10× and ~100×").
Held-out ranking gives the same two lrs. Dimuon lr ÷ 30 against AdamW at the same lr: 4.15 vs 2.10 (3.3e-5),
1.72 vs 0.94 (1e-4), 0.77 vs 0.59 (3.3e-4), 0.91 vs 0.87 (1e-3), 6.28 vs 5.53 (3.3e-3) — same shape and width
(both usable over ~1 decade, both diverge 10× above best), Dimuon uniformly a bit worse (best 0.768 vs 0.589, 30%,
outside the 6% repeat noise). Dimuon lr ÷ 90 puts its optimum at 1.1e-4, where AdamW is 0.94 — fits worse.
**H1: supported at the low end of the predicted range (≈30×, not 90×)**; curves coincide in shape, not level.

**Block B (60 steps), mean train loss at ranks 4 / 8 / 32:** AdamW 3e-4: 0.580 / 0.589 / 0.575; AdamW 1e-3:
1.071 / 0.865 / 0.749. Dimuon 1e-2: 0.795 / 0.768 / 0.854; Dimuon 3e-2: 0.957 / 0.910 / 1.025.
At each optimizer's best lr, 4 → 32 changes AdamW by −1% and Dimuon by +7% (both near the 6% noise).
**H3: not supported** — no sign rank helps Dimuon more; if anything AdamW gains with rank at its second lr
(1.07 → 0.75). Caveat: Dimuon's step norm scales with √r·(r/32)^0.175, so its best lr may shift with rank and only
2 lrs per rank were run; and the task saturates (loss → 1e-5), so capacity is never the limit.

**Block C (rank 8, 150 steps, const vs linear decay to 0).** Late phase = steps 91–150 (`runs/BH/blockC_stats.json`):
| run | median late | max late | frac late > 0.01 | held at 150 |
|---|---|---|---|---|
| AdamW 3e-4 const | 4.6e-6 | 1.1e-5 | 0.00 | 2e-6 |
| AdamW 3e-4 linear | 1.1e-5 | 2.0e-5 | 0.00 | 2e-5 |
| AdamW 1e-3 const | 0.100 | 4.72 | 0.87 | 3.28 |
| AdamW 1e-3 linear | 2.9e-4 | 0.18 | 0.15 | 0.0015 |
| Dimuon 1e-2 const | 0.023 | 2.60 | 0.57 | 0.011 |
| Dimuon 1e-2 linear | 1.3e-5 | 0.61 | 0.05 | 2e-6 |
| Dimuon 3e-2 const / linear | not available | | | | (both runs were cut at step 90 of 150 by a container restart; at step 90 last-6 train loss 0.91 const vs 0.006 linear, held-out 1.11 vs 0.0019) |
At each optimizer's best lr: AdamW const keeps falling smoothly (late median 4.6e-6, never above 1.1e-5); Dimuon
const sits ~5000× higher (median 0.023) with spikes to 2.6 — a noisy floor; linear decay brings Dimuon to the
AdamW level (1.3e-5). But AdamW at 3× its best lr shows the same kind of floor (median 0.10, spikes to 4.7) and
decay also removes it. **H2: partly supported** — at matched best lr Dimuon has a late noisy floor that AdamW does
not, and decay removes it; but the floor is not unique to Dimuon (AdamW gets one at 3× lr), so this run cannot
separate "Dimuon never slows down" from "Dimuon's best-by-speed lr sits closer to its stability edge". One seed; the
repeat shows late-phase values vary ~10–30× between identical runs.

**Caveats.** One model, one tiny synthetic dataset (saturates to ~1e-5), one seed, one minibatch order, ½-decade lr
grid, best lr chosen by a 60-step speed metric. Service is not deterministic after ~11 steps. Step time 6–260 s
under shared load (later block C runs were slow).

**Usage.** 27 training clients (10 A, 8 B, 1 repeat, 8 C); 1,980 forward_backward + 1,980 optim_step, plus 186
forward (eval) calls. **Nothing stored on the server:** no save_weights, save_state or export call in this task;
no files written except losses.

**Files.** `pipeline/bh_lib.py` (dataset + run loop), `bh_blockA.py`, `bh_blockBC.py`, `bh_figs.py`,
`bh_c_stats.py`; `runs/BH/{A,B,R,C}_*.json` (per-step train loss, held-out every 10, s/step, model_id),
`summary.json`, `blockC_stats.json`, `blockA.log`, `blockBC.log`; `figures/bh_curves.png`, `bh_lr.png`,
`bh_blockC.png`.

## BEHAVE-02 — does Dimuon's constant-lr floor persist below its best lr? — 2026-11-14 (measurement agent; UNREVIEWED)

**Question.** BEHAVE-01 block C: at constant lr Dimuon 1e-2 (its best) ends on a noisy floor (late median 0.023), AdamW
3e-4 (its best) settles to 5e-6, but AdamW at 1e-3 also shows a floor. Fixed-step account: Dimuon's per-tensor update
norm is fixed, so it cannot settle at constant lr; well below its best lr it still sits on a floor (lower, but well
above AdamW and not decreasing). Stability-edge account: Dimuon's best lr is simply nearer its stability edge; well
below it Dimuon settles like AdamW.

**Setup.** Identical to BEHAVE-01 (same model, Velish data, split seed 1234, fixed minibatch order, LoRA seed 0, rank 8,
attn+MLP), constant lr, 240 steps planned, held-out every 20. `pipeline/bh2_blockD.py`; `bh_lib.py` gained a
`data_seed` argument (0 = BEHAVE-01 behaviour, unchanged) and now writes the json every 10 steps. Stats:
`pipeline/bh2_analyze.py` → `runs/BH/behave02_stats.json`, figure `figures/bh_floor.png`. Last 60 = the last 60
steps the run actually reached.

**Consistency check.** First 60 steps against block A at the same lr: mean train loss AdamW 1e-4 0.931 vs 0.943,
Dimuon 1e-3 4.144 vs 4.147 (within the 6% repeat noise; not bit-identical, as before).

**Service throughput was the binding constraint.** Steps took 4 s for some clients and 30–260 s for others. Attempt 1
(00:34) gave Dimuon 1e-3 a fast lane (240 steps at 6.3 s/step); the other three stalled (one waited >13 min on its
step-0 eval). At 01:08 I killed and relaunched those three with fresh clients (partials kept in
`runs/BH/D_attempt1_partial/`). They still ran at ~30–100 s/step and were stopped at 02:47 by the deadline.
**Only Dimuon 1e-3 completed 240 steps.**

| run | steps reached | first step < 0.01 | last-60 window | median | max | frac > 0.01 | trend (median) |
|---|---|---|---|---|---|---|---|
| Dimuon 1e-3 (1/10 best) | **240 (done)** | 88 | 181–240 | **1.4e-5** | 3.3e-3 | **0.00** | 121–180: 1.05e-3 → 181–240: 1.4e-5 (falling) |
| Dimuon 3e-3 (1/3 best) | 110 (partial) | 46 | 51–110 | 6.3e-5 | 0.033 | 0.02 | 51–80: 1.6e-4 → 81–110: 2.5e-5 (falling) |
| AdamW 1e-4 (1/3 best) | 170 (partial) | 34 | 111–170 | 1.3e-5 | 4.3e-5 | 0.00 | 111–140: 1.6e-5 → 141–170: 9.5e-6 (falling) |
| AdamW 3e-5 (1/10 best) | 60 (partial) | not reached | 1–60 | 0.90 | 7.8 | 1.00 | not converged — no reading |
| *ref.* AdamW 3e-4 const (BEHAVE-01 C) | 150 | 22 | 91–150 | 4.6e-6 | 1.1e-5 | 0.00 | |
| *ref.* Dimuon 1e-2 const (BEHAVE-01 C) | 150 | 24 | 91–150 | 0.023 | 2.6 | 0.57 | |

Held-out at the last eval: Dimuon 1e-3 0.0013 (step 240), Dimuon 3e-3 0.0018 (step 100), AdamW 1e-4 1.1e-4 (step 160),
AdamW 3e-5 0.21 (step 60). Last 10 train losses of Dimuon 1e-3 are ~5e-7 to 1.2e-6.

**Matched window for the partial Dimuon 3e-3 run.** Dimuon 1e-2's floor in BEHAVE-01 developed late. Over steps
61–90 its median was 9e-4, and over 51–110 it was 4.9e-4 with max 0.55 and 22% of steps above 0.01. Dimuon 3e-3 over
the same 51–110 window has median 6.3e-5, max 0.033 and 2% of steps above 0.01. AdamW 1e-3 over 51–110 has median
0.042 and 77% of steps above 0.01. So 3e-3 is already much quieter than 1e-2 in the same window, but 110 steps cannot
rule out a floor that appears later.

**Verdict: stability-edge account supported (one seed, one fully completed Dimuon run).** Dimuon at 1/10 of its best
lr converges (below 0.01 at step 88) and then keeps falling: median 1.05e-3 over steps 121–180 and 1.4e-5 over
181–240, never above 0.01 in the last 60 steps. That is within ~3× of AdamW at 1/3 of its best lr (1.3e-5 over
111–170) and of the same order as AdamW at its best (4.6e-6). The fixed-step account predicted a floor well above
AdamW's that stops decreasing; at lr 1e-3 neither happens. A floor that scales with lr may still exist below these
values. With a fixed update norm the attainable loss should fall with lr, and 240 steps at 1e-3 cannot separate "no
floor" from "floor below ~1e-5". So what is refuted is the floor as an explanation of the 0.023 seen at 1e-2, not
fixed-norm steps as such. Dimuon 3e-3 (partial) points the same way.

**Caveats.** One seed, one data order, one task that saturates. Three of four runs are partial because of service
throughput: AdamW 3e-5 did not converge in its 60 steps, so it gives no reading, and Dimuon 3e-3 stopped at 110 of 240.
The "best-lr" comparison mixes a done Dimuon run with a partial AdamW run (170 steps). The Dimuon 1e-3 curve shows
some isolated late spikes (max 3.3e-3); these are small next to 1e-2's 2.6.
**Block E (seed-1 replication of the lr ratio) was not run**: there was no time once block D throughput collapsed.
Code is ready (`pipeline/bh2_blockE.py`; `data_seed=1` gives split seed 1235 with 3/16 held-out overlap, shuffled
minibatch order, LoRA seed 1).

**Usage.** 7 training clients (4 in attempt 1, 3 relaunched), about 590 forward_backward + optim_step calls
(240 + 60 + 10 attempt-1 partials, then 170 + 110 + 60 relaunched, minus a few in flight at the kill), about 35 forward
evals. No save_weights, save_state or export calls; losses only.

**Files.** `runs/BH/D_*.json`, `runs/BH/D_attempt1_partial/`, `runs/BH/blockD.log`, `runs/BH/behave02_stats.json`,
`figures/bh_floor.png`, `pipeline/bh2_blockD.py`, `bh2_blockE.py`, `bh2_analyze.py`, `bh_lib.py` (data_seed + write
every 10 steps).
