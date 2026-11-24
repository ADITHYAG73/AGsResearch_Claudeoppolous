"""fr_core.py -- shared helpers for FRESH-01 (fresh-state steps, clipping, lr dependence).

Lazy per-tensor loading from runs/D2 (npz members are read one at a time), Polar-Express-5 maps
in float64 and in the FIRSTSTEP-01 fp16 emulation, residual summaries.
Conventions: for module m, A = W["A/m"] (r x d_in), B = W["B/m"] (d_out x r),
GA = G["GA/m"], GB = G["GB/m"]; dA = A1 - A0, dB = B1 - B0.
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fs_spec import PE_MEM, safety  # noqa: E402
from fs_sim import ns as _ns_sim, rank_scale  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
D2 = ROOT / "runs" / "D2"
OUT = ROOT / "runs" / "FR"; OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures"; FIG.mkdir(exist_ok=True)
PE5 = safety(PE_MEM[:5], 1.01)
_MAN = None


def man():
    global _MAN
    if _MAN is None:
        _MAN = json.load(open(D2 / "manifest.json"))
    return _MAN


def rec(step_id):
    return man()["steps"][step_id]


class Lazy:
    """npz opened lazily; z[key] reads only that member, as float64."""
    def __init__(self, path):
        self.z = np.load(path)
        self.files = self.z.files

    def __getitem__(self, k):
        return np.asarray(self.z[k], np.float64)


def state(sid):
    return Lazy(D2 / "states" / f"{sid}.npz")


def grad(key):
    return Lazy(D2 / "grads" / f"{key}.npz")


def modules(W):
    return sorted(k[2:] for k in W.files if k.startswith("A/"))


def step_io(step_id):
    r = rec(step_id)
    W0, W1 = state(r["before"]), state(r["after"])
    G = grad(r["grad_key"]) if r.get("grad_key") else None
    return r, W0, W1, G


def mB(m):  # module name of the B tensor that pairs with lora_A name m
    return m.replace("lora_A", "lora_B")


# --------------------------------------------------------------------- maps
def polar(M):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def pe5_f64(M, coefs=PE5, sf=1.01):
    """float64 PE5 on the small-gram orientation (whichever side is r)."""
    X = M if M.shape[0] <= M.shape[1] else M.T
    X = X / (np.linalg.norm(X) * sf)
    for a, b, c in coefs:
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X if M.shape[0] <= M.shape[1] else X.T


def pe5_f16(M):
    """FIRSTSTEP-01 rule: X0 = fp16(G32/(||G||*1.01)), fused fp16 NS on the r x r gram."""
    tall = M.shape[0] >= M.shape[1]
    G = M if tall else M.T          # fs_sim.ns wants (d x r) and works on its transpose
    X = _ns_sim(G, PE5, p="f16", cast_first=False, fused=True)
    return X if tall else X.T


def finalise(X, r, lr):
    """-lr*sqrt(r)*(r/32)^0.175 * X/||X||_F in float32 (FIRSTSTEP-01 item 5)."""
    X32 = X.astype(np.float32)
    n = np.sqrt(np.sum(X32 * X32, dtype=np.float32))
    return (-(np.float32(rank_scale(r, lr)) / n) * X32).astype(np.float64)


def relF(P, D):
    return float(np.linalg.norm(P - D) / np.linalg.norm(D))


def best_scaled_relF(P, D):
    """residual after the optimal scalar on P (direction-only comparison)."""
    a = float(np.sum(P * D) / np.sum(P * P))
    return relF(a * P, D)


def summ(x):
    x = np.asarray(x, float)
    return dict(n=int(x.size), median=float(np.median(x)), p90=float(np.quantile(x, .9)),
                max=float(np.max(x)), min=float(np.min(x)))


def fmt(s):
    return f"med {s['median']:.3g}  p90 {s['p90']:.3g}  max {s['max']:.3g}  (n={s['n']})"
