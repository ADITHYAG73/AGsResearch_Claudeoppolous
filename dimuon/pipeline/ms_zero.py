"""ms_zero.py -- MULTISTEP-01 Q2: zero-gradient steps (S4). Predict zero1..zero3 of S4_r8_cont
(beta1 0.9, R1) from the two preceding oracle gradients with the candidate rule, and reproduce the
measured cosines cos(zero1, step2), cos(zero2, zero1), cos(zero3, zero2).
Also: rank-32 zero steps (S4_r32_cont) have no same-regime step-2 gradient, so they cannot be
simulated; for them we report the measured cosines and test, model-free, how zero_{k+1} relates to
zero_k (pure rescaling? diagonal rescaling?)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ms_common import step, state, grad, names, get, relF, cos, summ, fmt, OUT
from ms_model import direction, mom_update
from fs_sim import bf16

res = {}
# ---- rank 8, full simulation
recs = ["S4_r8_cont/step1", "S4_r8_cont/step2", "S4_r8_cont/zero1", "S4_r8_cont/zero2", "S4_r8_cont/zero3"]
S = [step(s) for s in recs]
for a, b in zip(recs, recs[1:]):
    assert step(b)["before"] == step(a)["after"], (a, b)
G = [grad(S[0]["grad_key"]), grad(S[1]["grad_key"])]
W = [state(S[0]["before"])] + [state(s["after"]) for s in S]
ms = names(W[0])
variants = {"ema/prevmom": ("ema", "prevmom", "f32"), "ema/prevmom/bf16s": ("ema", "prevmom", "bf16s"),
            "ema/prevgrad": ("ema", "prevgrad", "f32")}
acc = {}
for m in ms:
    for kind in "AB":
        Us = [get(W[i + 1], kind, m) - get(W[i], kind, m) for i in range(5)]
        gs = [get(G[0], kind, m, "G"), get(G[1], kind, m, "G")] + [np.zeros_like(Us[0])] * 3
        meas = [cos(Us[2], Us[1]), cos(Us[3], Us[2]), cos(Us[4], Us[3])]
        for i, c in enumerate(meas):
            acc.setdefault(("measured", kind, f"cos{i}"), []).append(c)
        for vn, (form, src, prec) in variants.items():
            mom, pm, pg = None, None, None
            preds = []
            for i in range(5):
                Sst = pm if src == "prevmom" else pg
                mom, fed = mom_update(mom, gs[i], 0.9, form)
                if prec == "bf16s":
                    mom = bf16(mom).astype(np.float64)
                d = direction(fed, Sst, 1e-4, 1e-4, "relmax")
                preds.append(d)
                pm, pg = mom.copy(), gs[i]
                if i >= 1 and np.any(Us[i]):
                    acc.setdefault((vn, kind, f"relF_{recs[i].split('/')[1]}"), []).append(relF(d, Us[i]))
            for i, (x, y) in enumerate([(2, 1), (3, 2), (4, 3)]):
                acc.setdefault((vn, kind, f"cos{i}"), []).append(cos(preds[x], preds[y]))
lab = {"cos0": "cos(zero1,step2)", "cos1": "cos(zero2,zero1)", "cos2": "cos(zero3,zero2)"}
for key, v in sorted(acc.items()):
    name = lab.get(key[2], key[2])
    s = summ(v)
    if key[2].startswith("cos"):
        print(f"r8 {key[0]:14s} {key[1]} {name:18s} median {np.median(v):.6f}  min {np.min(v):.4f}")
    else:
        print(f"r8 {key[0]:14s} {key[1]} {name:18s} relF {fmt(s)}")
    res["r8|" + "|".join(key)] = s

# ---- rank 32, model-free
recs32 = ["S4_r32_cont/step2", "S4_r32_cont/zero1", "S4_r32_cont/zero2", "S4_r32_cont/zero3"]
W32 = [state(step(recs32[0])["before"])] + [state(step(s)["after"]) for s in recs32]
acc32 = {}
for m in names(W32[0]):
    for kind in "AB":
        Us = [get(W32[i + 1], kind, m) - get(W32[i], kind, m) for i in range(4)]
        for i in range(3):
            acc32.setdefault((kind, f"cos({recs32[i+1].split('/')[1]},{recs32[i].split('/')[1]})"), []).append(cos(Us[i + 1], Us[i]))
        # does zero3 extrapolate geometrically from zero1 -> zero2 ? (constant per-step change)
        d1 = Us[2] / np.linalg.norm(Us[2]) - Us[1] / np.linalg.norm(Us[1])
        d2 = Us[3] / np.linalg.norm(Us[3]) - Us[2] / np.linalg.norm(Us[2])
        acc32.setdefault((kind, "cos(delta23, delta12)"), []).append(cos(d2, d1))
        acc32.setdefault((kind, "|delta23|/|delta12|"), []).append(np.linalg.norm(d2) / np.linalg.norm(d1))
for key, v in sorted(acc32.items()):
    print(f"r32 measured {key[0]} {key[1]:24s} median {np.median(v):.5f}  p10 {np.quantile(v, .1):.4f}  min {np.min(v):.4f}")
    res["r32|" + "|".join(key)] = summ(v)
json.dump(res, open(OUT / "zero_steps.json", "w"), indent=1)
