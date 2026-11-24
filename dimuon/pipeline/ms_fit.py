"""ms_fit.py -- torch (float64, CPU) fitter for conditioned-PE models of a continuing Dimuon step.

Model family (per tensor, tall orientation n x r, current gradient g, stats from history):
    M  = Lpre * g * Rpre                     (diagonal pre-scaling, rows/cols)
    X  = PE5(M)                              (Polar Express, safety 1.01, float64)
    U ~ Lpost * X * Rpost                    (diagonal post-scaling), compared after unit-normalising
with Lpre = (vL + dL*mean(vL))^b, Rpre = (vR + dR*mean(vR))^c, Lpost = (...)^a, Rpost = (...)^e,
vL / vR = row / column statistics supplied by the caller (e.g. previous-gradient row energies).
Free parameters are fitted on one half of the tensors (sum of squared relF) and scored on the other.
"""
import numpy as np
import torch
from ms_common import PE5

torch.set_default_dtype(torch.float64)


def pe_t(M, coefs=PE5, sf=1.01):
    X = M / (torch.linalg.norm(M) * sf)
    I = torch.eye(M.shape[1])
    for a, b, c in coefs:
        K = X.T @ X
        X = X @ (a * I + b * K + c * (K @ K))
    return X


def dpow(v, p, d):
    v = v + d * v.mean()
    return torch.where(v > 0, v.clamp_min(1e-300) ** p, torch.zeros_like(v))


def predict(th, item, spec):
    """th: dict of torch scalars; item: dict(g, vL, vR); spec: which params are active."""
    g = item["g"]
    vL, vR = item["vL"], item["vR"]
    dL = torch.exp(th["ldL"]) if "ldL" in th else torch.tensor(0.0)
    dR = torch.exp(th["ldR"]) if "ldR" in th else torch.tensor(0.0)
    M = g
    if "b" in th:
        M = M * dpow(vL, th["b"], dL)[:, None]
    if "c" in th:
        M = M * dpow(vR, th["c"], dR)[None, :]
    if "kc" in th or "ke" in th:
        K = item["K"]
        dK = torch.exp(th["ldK"]) if "ldK" in th else torch.tensor(0.0)
        w, Q = torch.linalg.eigh((K + K.T) / 2 + dK * torch.trace(K) / K.shape[0] * torch.eye(K.shape[0]))
        w = w.clamp_min(1e-300 * w.max())
    if "kc" in th:
        M = M @ ((Q * w ** th["kc"]) @ Q.T)
    X = -pe_t(M)
    if "ke" in th:
        X = X @ ((Q * w ** th["ke"]) @ Q.T)
    if "a" in th:
        X = X * dpow(vL, th["a"], dL)[:, None]
    if "e" in th:
        X = X * dpow(vR, th["e"], dR)[None, :]
    return X


def relF_t(P, U):
    return torch.linalg.norm(P / torch.linalg.norm(P) - U / torch.linalg.norm(U))


def to_t(items):
    return [{k: torch.as_tensor(v) for k, v in it.items()} for it in items]


def fit(items, init, fixed=None, iters=150, lr=0.05):
    """init: dict name->float (free); fixed: dict name->float (held). Returns params dict."""
    fixed = fixed or {}
    th = {k: torch.tensor(float(v), requires_grad=True) for k, v in init.items()}
    opt = torch.optim.LBFGS(list(th.values()), lr=1, max_iter=iters, line_search_fn="strong_wolfe")
    T = to_t(items)

    def closure():
        opt.zero_grad()
        allp = dict(th, **{k: torch.tensor(float(v)) for k, v in fixed.items()})
        L = sum(relF_t(predict(allp, it, None), it["U"]) ** 2 for it in T) / len(T)
        L.backward()
        return L
    if th:
        opt.step(closure)
    out = {k: float(v.detach()) for k, v in th.items()}
    out.update(fixed)
    return out


def evaluate(items, params):
    T = to_t(items)
    p = {k: torch.tensor(float(v)) for k, v in params.items()}
    with torch.no_grad():
        return np.array([float(relF_t(predict(p, it, None), it["U"])) for it in T])
