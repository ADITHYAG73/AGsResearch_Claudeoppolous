"""ms_zero32_proxy.py -- is cos(zero2, zero1) = 0.98 at rank 32 explained by fp16 rounding inside
the orthogonaliser? S4_r32_cont has no step-2 gradient, so use the analogous R2 trunk S2_r32_b0.9
(steps 1-2 on T0, beta 0.9): build m2 and m1 with the model, then emulate two consecutive zero
steps (m3 = 0.9 m2 with stats m2; m4 = 0.9 m3 with stats m3) in float64 and in the full fp16
fused emulation, float32 momentum. Report cos(zero2, zero1) for each, rank 32 and rank 8."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ms_common import step, grad, names, cos, OUT, get, state
from fs_sim import ns, PE5 as P5
from ms_common import pe
f32 = np.float32


def D(S):
    l = (S ** 2).sum(1); c = (S ** 2).sum(0)
    return (l + 1e-4 * l.max()) ** -.25, (c + 1e-4 * c.max()) ** -.25


def dirn(m, S, prec):
    Dl, Dr = D(S)
    M = Dl[:, None] * m * Dr[None, :]
    X = pe(M) if prec == "f64" else ns(M, P5, "f16", cast_first=False)
    return -Dl[:, None] * X * Dr[None, :]


out = {}
for chain in ["S2_r32_b0.9", "S2_r8_b0.9"]:
    g1 = grad(step(f"{chain}/step1")["grad_key"]); g2 = grad(step(f"{chain}/step2")["grad_key"])
    ms = names(state(step(f"{chain}/step1")["before"]))[1::8]
    for kind in "AB":
        for prec in ["f64", "f16"]:
            cs = []
            for m in ms:
                a, b = get(g1, kind, m, "G"), get(g2, kind, m, "G")
                m1 = (f32(0.1) * a.astype(f32))
                m2 = (f32(0.9) * m1 + f32(0.1) * b.astype(f32))
                m3 = f32(0.9) * m2; m4 = f32(0.9) * m3
                if not np.any(m1):
                    continue
                z1 = dirn(m3.astype(np.float64), m2.astype(np.float64), prec)
                z2 = dirn(m4.astype(np.float64), m3.astype(np.float64), prec)
                cs.append(cos(z1, z2))
            out[f"{chain}|{kind}|{prec}"] = dict(median=float(np.median(cs)), min=float(np.min(cs)), n=len(cs)) if cs else None
            if cs:
                print(f"{chain} {kind} {prec}: simulated cos(zero2,zero1) median {np.median(cs):.6f} min {np.min(cs):.5f} (n={len(cs)})", flush=True)
json.dump(out, open(OUT / "zero32_proxy.json", "w"), indent=1)
