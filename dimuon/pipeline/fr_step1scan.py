"""fr_step1scan.py -- FRESH-01 consistency scan: every recorded step whose optimizer state is fresh
(hist == []) and that has a same-regime gradient, residual of the per-factor PE5-f16 law
(median over a module subset, stride 4). Purpose: find step/gradient pairs that do not match
(e.g. mixed server numerics) before interpreting anything.  -> runs/FR/step1scan.json"""
import json, sys
import numpy as np
from fr_core import *  # noqa

stride = int(sys.argv[1]) if len(sys.argv) > 1 else 4
out = {}
for s, r in sorted(man()["steps"].items()):
    if r.get("hist") or not r.get("grad_key") or not r.get("grad_regime_ok"):
        continue
    if r["before"] == r["after"]:
        continue
    _, W0, W1, G = step_io(s)
    lr, rank = r["params"]["learning_rate"], r["rank"]
    res = {"A": [], "B": []}
    for i, m in enumerate(modules(W0)):
        if i % stride: continue
        for F, key, gk in (("A", "A/" + m, "GA/" + m), ("B", "B/" + mB(m), "GB/" + mB(m))):
            g = G[gk]
            if not np.any(g): continue
            res[F].append(relF(finalise(pe5_f16(g), rank, lr), W1[key] - W0[key]))
    out[s] = dict(rank=rank, regime=r["regime"], text=r["text"], lr=lr, clip=r["params"]["grad_clip_norm"],
                  t=r["t_before"], **{F: (summ(v) if v else None) for F, v in res.items()})
    a = out[s]["A"]; b = out[s]["B"]
    print(f"{s:36s} r{rank:<3d} {r['regime']} T{r['text']} {r['t_before'][11:]}  "
          f"A {a['median'] if a else float('nan'):.4f}  B {b['median']:.4f} (B max {b['max']:.3f})", flush=True)
json.dump(out, open(OUT / "step1scan.json", "w"), indent=1)
