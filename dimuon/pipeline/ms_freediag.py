"""ms_freediag.py -- hypothesis-free probe: fit U ~ diag(postL) PE5(diag(preL) g diag(preR)) diag(postR)
with all four diagonals FREE (log-parametrised), per tensor, for continuing beta1=0 steps.
Saves fitted log-diagonals plus candidate statistics (row/col energies of current and previous
gradient) so their functional relation can be read off afterwards.

usage: ms_freediag.py RANK STEP KIND STRIDE OFFSET [CHAIN]
CHAIN defaults to S2_r{RANK}_b0. With CHAIN=null the target is replaced by the update of a
different step (step 3 <-> 5), as a null for how much the free diagonals can fit on their own.
"""
import sys, time, json
import numpy as np
import torch
sys.path.insert(0, __import__("os").path.dirname(__file__))
from ms_common import step, state, grad, names, get, unit, OUT
from ms_fit import pe_t

torch.set_default_dtype(torch.float64)
r, k, kind, stride, off = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
chain = sys.argv[6] if len(sys.argv) > 6 else f"S2_r{r}_b0"
null = chain == "null"
T = f"S2_r{r}_b0"
Gp = grad(step(f"{T}/step{k-1}")["grad_key"])
Gc = grad(step(f"{T}/step{k}")["grad_key"])
c = step(f"{T}/step{k}")
W0, Wc = state(c["before"]), state(c["after"])
if null:
    kn = 5 if k != 5 else 3
    cn = step(f"{T}/step{kn}")
    Wn0, Wn1 = state(cn["before"]), state(cn["after"])
ms = names(W0)
od = OUT / "freediag"
od.mkdir(exist_ok=True, parents=True)
for m in ms[off::stride]:
    g = get(Gc, kind, m, "G")
    gp = get(Gp, kind, m, "G")
    U = (get(Wn1, kind, m) - get(Wn0, kind, m)) if null else (get(Wc, kind, m) - get(W0, kind, m))
    U = unit(U)
    gt, Ut = torch.tensor(g), torch.tensor(U)
    n, rr = g.shape
    e = (gp ** 2).mean(1)
    P = {"pL": torch.tensor(-0.5 * np.log(e + 0.0136 * e.mean()), requires_grad=True),
         "pR": torch.zeros(rr, requires_grad=True),
         "qL": torch.zeros(n, requires_grad=True),
         "qR": torch.zeros(rr, requires_grad=True)}
    opt = torch.optim.LBFGS(list(P.values()), lr=1, max_iter=400, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-14)

    def f():
        opt.zero_grad()
        M = gt * torch.exp(P["pL"] - P["pL"].mean())[:, None] * torch.exp(P["pR"] - P["pR"].mean())[None, :]
        X = -pe_t(M) * torch.exp(P["qL"] - P["qL"].mean())[:, None] * torch.exp(P["qR"] - P["qR"].mean())[None, :]
        L = torch.linalg.norm(X / torch.linalg.norm(X) - Ut) ** 2
        L.backward()
        return L
    t = time.time()
    for _ in range(4):
        opt.step(f)
    res = float(f().detach()) ** 0.5
    tag = f"r{r}_s{k}_{kind}_{'null_' if null else ''}{m}"
    np.savez(od / f"{tag}.npz", **{kk: v.detach().numpy() for kk, v in P.items()},
             g=g, gp=gp, U=U, relF=res)
    print(json.dumps(dict(tag=tag, relF=round(res, 5), sec=round(time.time() - t))), flush=True)
