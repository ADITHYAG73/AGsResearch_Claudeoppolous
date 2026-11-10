"""ms_symfree.py -- free symmetric-diagonal probe: U ~ Dl PE5(Dl g Dr) Dr with Dl (n), Dr (r) free.
Saves the implied row statistic sL = Dl^-4 and column statistic sR = Dr^-4 (gauge: up to scale)
together with row/col energies of every earlier gradient in the chain, so the state can be read off.
usage: ms_symfree.py RANK STEP KIND STRIDE OFFSET [CHAIN]"""
import sys, time, json, os
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from ms_common import step, state, grad, names, get, unit, OUT
from ms_fit import pe_t
torch.set_default_dtype(torch.float64)
r, k, kind, stride, off = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
T = sys.argv[6] if len(sys.argv) > 6 else f"S2_r{r}_b0"
c = step(f"{T}/step{k}")
G = {j: grad(step(f"{T}/step{j}")["grad_key"]) for j in range(1, k + 1)}
W0, Wc = state(c["before"]), state(c["after"])
od = OUT / "symfree"; od.mkdir(parents=True, exist_ok=True)
for m in names(W0)[off::stride]:
    g = get(G[k], kind, m, "G"); U = unit(get(Wc, kind, m) - get(W0, kind, m))
    gs = [get(G[j], kind, m, "G") for j in range(1, k)]
    e = (gs[-1] ** 2).sum(1)
    init = -0.25 * np.log(e + 1e-4 * e.max())
    lL = torch.tensor(init - init.mean(), requires_grad=True)
    lR = torch.tensor(-0.25 * np.log((gs[-1] ** 2).sum(0) + 1e-30), requires_grad=True)
    gt, Ut = torch.tensor(g), torch.tensor(U)
    opt = torch.optim.LBFGS([lL, lR], lr=1, max_iter=500, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-12, tolerance_change=1e-15)
    def f():
        opt.zero_grad()
        Dl = torch.exp(lL - lL.mean())[:, None]; Dr = torch.exp(lR - lR.mean())[None, :]
        X = -Dl * pe_t(Dl * gt * Dr) * Dr
        L = torch.linalg.norm(X / torch.linalg.norm(X) - Ut) ** 2; L.backward(); return L
    t = time.time()
    for _ in range(4): opt.step(f)
    res = float(f().detach()) ** 0.5
    tag = f"{T}_s{k}_{kind}_{m}"
    np.savez(od / f"{tag}.npz", lL=lL.detach().numpy(), lR=lR.detach().numpy(), g=g,
             gs=np.stack(gs), U=U, relF=res)
    print(json.dumps(dict(tag=tag, relF=round(res, 5), sec=round(time.time() - t))), flush=True)
