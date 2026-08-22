# Pod-side scripts (run on the RunPod GPU, not the laptop)

Rescued from /tmp on 2026-08-22 — these are the exact scripts that produced the runs in
`mats_2027/runs/`. Paths inside assume the pod layout: fork cloned to `/root/nla`,
`HF_HOME=/root/hf`, data in `/root/`.

| script | run | role |
|---|---|---|
| `pod_setup.sh` | all | clone fork, pin torch 2.9.1+cu128 / transformers 5.3.0 / sglang 0.5.10.post1 |
| `nla_tests.py` | SMOKE-01 | marker check (A), embedding-norm vs injection_scale (B), **extract last-position activations → acts.npy** (C) |
| `nla_roundtrip.py` | SMOKE-01 | AV→AR round trip per activation (cos 0.997 reproduction) |
| `nla_variance.py` | SMOKE-01 | K=8 resamples of the same activation, T=1 |
| `gen_k8.py` | NOISE-01 | K=8 explanations for all 6 pilot activations |
| `score_ablations.py` | NOISE-01 | AR-only scoring of every ablation variant (batch size 1) |
| `run_extract.sh` | POS-01 | `stage2_generate.py --extract-only` + bit-for-bit check vs acts.npy |
| `launch_av.sh` | POS-01 | sglang AV server (mem-fraction 0.6, triton attention, pytorch sampling) |
| `run_generate.sh` | POS-01 | waits for /health, then `stage2_generate.py --skip-extract --k 4` |

Order on a 48 GB card: extract (base alone) → launch AV → generate → kill AV → AR score.
Base (24 GB) + AV server (~29 GB) do not fit together.
