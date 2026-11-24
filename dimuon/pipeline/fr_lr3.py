"""fr_lr3.py -- FRESH-01 Q3: is the per-tensor size of the lr=1e-2 deviation predicted by the
tensor's sensitivity to fp16 rounding inside PE5?  Proxy sensitivity s_t = relF(PE5_fp16(g3),
PE5_f64(g3)) (current gradient; the continuing step's true input is unknown).  Also the
sensitivity to a 1-float32-ulp-scale random perturbation of the input before the fp16 cast.
-> runs/FR/lr3.json, lr3.txt"""
import json
import numpy as np
from fr_core import *  # noqa
lr = json.load(open(OUT / "lr.json"))["32"]
relE = {(x["F"], x["m"]): x["rel_0.01"] for x in lr}
r0 = rec("S6_r32_ck2_lr0.001/step3"); G = grad(r0["grad_key"]); W = state(r0["before"])
rng = np.random.default_rng(1)
rows = []
for i, m in enumerate(modules(W)):
    if i % 2: continue
    for F, gk in (("A", "GA/" + m), ("B", "GB/" + mB(m))):
        g = G[gk]
        p16 = pe5_f16(g); s = relF(p16, pe5_f64(g))
        gp = g * (1 + 2 ** -24 * rng.standard_normal(g.shape))   # ~1 f32 ulp relative noise
        s2 = relF(pe5_f16(gp), p16)
        rows.append(dict(F=F, m=m, relE=relE[(F, m)], s_f16_f64=s, s_ulp=s2))
lines = []
for F in "AB":
    R = [x for x in rows if x["F"] == F]
    e = np.log([x["relE"] for x in R])
    for k in ("s_f16_f64", "s_ulp"):
        v = np.array([x[k] for x in R]); ok = v > 0
        c = np.corrcoef(e[ok], np.log(v[ok]))[0, 1] if ok.sum() > 3 else float("nan")
        l = f"{F}: relE {fmt(summ(np.exp(e)))};  {k} {fmt(summ(v))};  corr(log relE, log {k}) = {c:+.3f}"
        print(l); lines.append(l)
json.dump(rows, open(OUT / "lr3.json", "w"), indent=0)
open(OUT / "lr3.txt", "w").write("\n".join(lines) + "\n")
