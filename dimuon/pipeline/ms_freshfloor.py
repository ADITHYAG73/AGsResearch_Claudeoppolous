"""ms_freshfloor.py -- noise-floor check for MULTISTEP-01: the fresh-state twin of each continuing
beta1=0 step is (by the model) PE5(g_t) with no preconditioner. Its residual against PE5(g_t) is
therefore the floor of 'PE5 + this oracle gradient' at that step, independent of any hidden state.
Scored on the test half (odd module index)."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ms_common import step, state, grad, names, get, pe, relF, summ, fmt, OUT
out = {}
for r in [8, 32]:
    T = f"S2_r{r}_b0"
    for k in range(2, 7):
        f = step(f"{T}_fresh_ck{k-1}/step{k}")
        W0, Wf, G = state(f["before"]), state(f["after"]), grad(f["grad_key"])
        for kind in "AB":
            v = [relF(-pe(get(G, kind, m, "G")), get(Wf, kind, m) - get(W0, kind, m)) for m in names(W0)[1::2]]
            out[f"r{r}_step{k}_{kind}"] = summ(v)
            print(f"r{r} step{k} {kind}: fresh twin vs PE5(g_t)  {fmt(summ(v))}", flush=True)
json.dump(out, open(OUT / "fresh_floor.json", "w"), indent=1)
