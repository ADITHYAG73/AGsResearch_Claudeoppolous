"""p3_rule.py -- PREDICT-01: the MULTISTEP-01 rule as a forward simulator + blind scoring.

Rule (per factor, tall orientation n x r: lora_B as stored, lora_A^T), beta/lr/clip of THIS call:
    g'   = c * g                         c = min(1, clip / N)  (N chosen by the caller)
    m32  = beta * S + (1 - beta) * g'    float32;  S = stored momentum before the step (0 if fresh)
    m    = bf16(m32)                     stored for the next step  (storage="f32": m = m32)
    if S != 0:  l_i = sum_j S_ij^2, l += 1e-4 max(l), Dl = l^-1/4;  same for columns -> Dr
    else:       Dl = Dr = 1
    V    = Dl PE5(Dl m32 Dr) Dr ;   U = -lr sqrt(r) (r/32)^0.175 V / ||V||_F
PE5 = Polar Express 5 steps, safety 1.01. Default PE precision "x0f16" (X0 rounded to fp16, rest
f64: MULTISTEP-01's best); "fused16" = FIRSTSTEP-01 fp16 fused emulation; "f64"; "f32"; "bf16".

Prediction files are written from GRADIENTS ONLY (runs/P3/grads) + call parameters; scoring reads
the measured states afterwards (pred/<exp>/... carry the wall time they were written).
"""
from __future__ import annotations
import hashlib, json, pathlib, sys, time
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from ms_common import pe, PE5  # noqa: E402
from fs_sim import bf16, ns, rank_scale  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
P3 = ROOT / "runs" / "P3"
PRED = P3 / "pred"
import os
SUFFIX = os.environ.get("P3_SUFFIX", "")   # e.g. "_solo": predictions made with solo oracles


def man():
    return json.load(open(P3 / "manifest.json"))


class Lazy:
    def __init__(self, path):
        self.z = np.load(path); self.files = self.z.files

    def __getitem__(self, k):
        return np.asarray(self.z[k], np.float64)


def grad(key):
    return Lazy(P3 / "grads" / f"{key}.npz")


def state(sid):
    return Lazy(P3 / "states" / f"{sid}.npz")


def tensors(z):
    """[(kind, module)] from a grads npz (GA/<m>.lora_A) or a state npz (A/<m>.lora_A)."""
    out = []
    for k in z.files:
        p, rest = k.split("/", 1)
        kind = p[-1]
        out.append((kind, rest.rsplit(".lora_", 1)[0]))
    return sorted(set(out))


def tall(x, kind):
    return x.T if kind == "A" else x


def gget(G, kind, m):
    return tall(G[f"G{kind}/{m}.lora_{kind}"], kind)


def wget(W, kind, m):
    return tall(W[f"{kind}/{m}.lora_{kind}"], kind)


def pe_apply(M, mode="x0f16"):
    if mode == "f64":
        return pe(M)
    if mode == "x0f16":
        return pe(M, x0prec="f16")
    if mode in ("fused16", "fusedbf16", "fused32"):
        p = {"fused16": "f16", "fusedbf16": "bf16", "fused32": "f32"}[mode]
        return ns(M, PE5, p, cast_first=False, fused=True)
    if mode == "wide_x0f16":   # d x d Gram side (X <- (aI + bXX^T + c(XX^T)^2) X), X0 in fp16
        X = M / (np.linalg.norm(M) * 1.01)
        X = np.asarray(X, np.float32).astype(np.float16).astype(np.float64)
        for a, b, c in PE5:
            K = X @ X.T
            X = a * X + (b * K + c * K @ K) @ X
        return X
    raise ValueError(mode)


def precond(S):
    if S is None or not np.any(S):
        return None, None
    S2 = S * S
    l, c = S2.sum(1), S2.sum(0)
    l = l + 1e-4 * l.max(); c = c + 1e-4 * c.max()
    return l ** -0.25, c ** -0.25


def rule(S, g, beta, lr, r, c=1.0, storage="bf16", mode="x0f16"):
    """one step for one tensor (tall). Returns (U predicted with its norm, stored momentum)."""
    g32 = np.asarray(g, np.float32)
    if c != 1.0:
        g32 = (np.float32(c) * g32).astype(np.float32)
    S32 = np.zeros_like(g32) if S is None else np.asarray(S, np.float32)
    m32 = (np.float32(beta) * S32 + np.float32(1.0 - beta) * g32).astype(np.float32)
    Dl, Dr = precond(None if S is None else S32.astype(np.float64))
    M = m32.astype(np.float64)
    if not np.any(M):
        V = np.zeros_like(M)
    elif Dl is None:
        V = pe_apply(M, mode)
    else:
        V = Dl[:, None] * pe_apply(Dl[:, None] * M * Dr[None, :], mode) * Dr[None, :]
    n = np.linalg.norm(V)
    U = -rank_scale(r, lr) * V / n if n > 0 else V
    m = bf16(m32).astype(np.float32) if storage == "bf16" else m32
    return U, m


def nostate(g, lr, r, mode="x0f16"):
    V = pe_apply(np.asarray(g, np.float64), mode)
    n = np.linalg.norm(V)
    return -rank_scale(r, lr) * V / n if n > 0 else V


def relF(P, Dm):
    d = np.linalg.norm(Dm)
    return float(np.linalg.norm(P - Dm) / d) if d > 0 else (0.0 if not np.any(P) else float("inf"))


def cos(X, Y):
    nx, ny = np.linalg.norm(X), np.linalg.norm(Y)
    return float(np.sum(X * Y) / (nx * ny)) if nx > 0 and ny > 0 else float("nan")


def summ(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if v.size == 0:
        return dict(n=0)
    return dict(n=int(v.size), med=float(np.median(v)), p90=float(np.quantile(v, .9)),
                max=float(v.max()))


def fmt(s):
    if not s or s.get("n", 0) == 0:
        return "n/a"
    return f"{s['med']:.2e} / {s['p90']:.2e} / {s['max']:.2e}"


# --------------------------------------------------------------------- prediction files
def write_pred(exp, name, U, meta):
    exp = exp + SUFFIX
    d = PRED / exp; d.mkdir(parents=True, exist_ok=True)
    arrs = {f"{k}/{m}": np.asarray(u, np.float32) for (k, m), u in U.items()}
    np.savez(d / f"{name}.npz", **arrs)
    h = hashlib.sha256(open(d / f"{name}.npz", "rb").read()).hexdigest()[:16]
    meta = dict(meta, written=time.time(), written_utc=time.strftime("%H:%M:%S", time.gmtime()),
                sha=h)
    json.dump(meta, open(d / f"{name}.json", "w"), indent=1, default=str)


def read_pred(exp, name):
    exp = exp + SUFFIX
    z = np.load(PRED / exp / f"{name}.npz")
    out = {}
    for k in z.files:
        kind, m = k.split("/", 1)
        out[(kind, m)] = np.asarray(z[k], np.float64)
    return out, json.load(open(PRED / exp / f"{name}.json"))


def measured(step_rec):
    W0, W1 = state(step_rec["before"]), state(step_rec["after"])
    return {(k, m): wget(W1, k, m) - wget(W0, k, m) for k, m in tensors(W0)}


def score(P, Um, extra=None):
    """per-tensor relF of prediction dict P vs measured dict Um, split A/B."""
    res = {"A": [], "B": []}
    for key, u in Um.items():
        if key in P:
            res[key[0]].append(relF(P[key], u))
    return {k: summ(v) for k, v in res.items()}


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
