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
