"""fr_orient.py -- FRESH-01 Q1a: on which side is the Newton-Schulz Gram formed?
In exact arithmetic X <- aX + b(XX^T)X + c(XX^T)^2 X is the same map whichever orientation of the
gradient is used; only the fp16 rounding differs.  Compare, per tensor, the FIRSTSTEP-01 fp16
emulation with the r x r Gram (small) against the same emulation with the d x d Gram (big: the
matrix used as is, A = X X^T over its long side).  lora_A (r x d_in) used as is has the small
Gram; lora_B (d_out x r) used as is has the big one.  Few tensors (d x d Gram is expensive).
-> runs/FR/orient.json, orient.txt"""
import json
import numpy as np
from fr_core import *  # noqa

f32 = np.float32
def H(x): return np.asarray(x, f32).astype(np.float16).astype(f32)


def pe5_f16_big(M):
    """fp16 fused PE5 with the Gram on the LONG side: X tall (d x r), A = X X^T (d x d)."""
    tall = M.shape[0] >= M.shape[1]
    G = np.asarray(M if tall else M.T, f32)
    X = H(G / (np.sqrt(np.sum(G * G, dtype=f32)) * f32(1.01)))
    for a, b, c in PE5:
        A = H(X @ X.T)
        B = H(f32(b) * A + f32(c) * (A @ A))
        X = H(f32(a) * X + B @ X)
    X = X.astype(np.float64)
    return X if tall else X.T


if __name__ == "__main__":
    out, lines = {}, []
    import sys
    steps = sys.argv[1:] or ["S1_r8_fresh_ck2_t0/step", "S1_r8_fresh_ck5_t1/step", "S1_r8/step1"]
    tag = "_r32" if any("r32" in x for x in steps) else ""
    for s in steps:
        r, W0, W1, G = step_io(s)
        rk = r["rank"]
        rows = []
        for m in modules(W0):
            for F, key, gk in (("A", "A/" + m, "GA/" + m), ("B", "B/" + mB(m), "GB/" + mB(m))):
                g = G[gk]
                if not np.any(g) or max(g.shape) > 2560: continue
                if F == "B" and g.shape[0] != 1024: continue
                if F == "A" and sum(x["F"] == "A" for x in rows) >= 3: continue
                if F == "B" and sum(x["F"] == "B" for x in rows) >= 4: continue
                D = W1[key] - W0[key]
                rows.append(dict(F=F, m=m, small=relF(finalise(pe5_f16(g), rk, 1e-3), D),
                                 big=relF(finalise(pe5_f16_big(g), rk, 1e-3), D),
                                 f64=relF(finalise(pe5_f64(g), rk, 1e-3), D)))
                l = f"{s:28s} {F} {key:42s} small-Gram fp16 {rows[-1]['small']:.5f}  big-Gram fp16 {rows[-1]['big']:.5f}  f64 {rows[-1]['f64']:.5f}"
                print(l, flush=True); lines.append(l)
        out[s] = rows
    json.dump(out, open(OUT / f"orient{tag}.json", "w"), indent=1)
    open(OUT / f"orient{tag}.txt", "w").write("\n".join(lines) + "\n")
