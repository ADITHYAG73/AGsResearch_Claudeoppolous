"""ms_common.py -- shared helpers for MULTISTEP-01 (offline analysis of D2 multi-step data).

Conventions: every matrix is handled in TALL orientation (n x r): lora_B as stored (d_out x r),
lora_A transposed (d_in x r). Polar Express (PE5, safety 1.01) is applied in float64 through the
r x r Gram matrix: X <- X (a I + b K + c K^2), K = X^T X  (identical to a X + b X X^T X + c X (X^T X)^2).
"""
import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
D2 = ROOT / "runs" / "D2"
OUT = ROOT / "runs" / "MS"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fs_spec import PE_MEM, safety  # noqa: E402

PE5 = safety(PE_MEM[:5], 1.01)
_MAN = None


def man():
    global _MAN
    if _MAN is None:
        _MAN = json.load(open(D2 / "manifest.json"))
    return _MAN


def step(sid):
    return man()["steps"][sid]


def npz(path):
    return np.load(path)


def state(sid):
    return np.load(D2 / "states" / f"{sid}.npz")


def grad(key):
    return np.load(D2 / "grads" / f"{key}.npz")


def names(z):
    """module base names (without .lora_A/.lora_B), in file order."""
    return [k[2:-7] for k in z.files if k.startswith("A/")]


def tall(x, kind):
    x = np.asarray(x, np.float64)
    return x.T if kind == "A" else x


def get(z, kind, m, pref=""):
    """kind 'A' or 'B'; pref '' for states, 'G' for gradients. Returns tall float64."""
    return tall(z[f"{pref}{kind}/{m}.lora_{kind}"], kind)


def pe(M, coefs=PE5, sf=1.01, x0prec="f64"):
    M = np.asarray(M, np.float64)
    n = np.linalg.norm(M)
    if n == 0:
        return np.zeros_like(M)
    X = M / (n * sf)
    if x0prec == "f16":
        X = np.asarray(X, np.float32).astype(np.float16).astype(np.float64)
    for a, b, c in coefs:
        K = X.T @ X
        X = X @ (a * np.eye(K.shape[0]) + b * K + c * (K @ K))
    return X


def unit(U):
    n = np.linalg.norm(U)
    return U / n if n > 0 else U


def relF(Uhat, U):
    """relative Frobenius residual after both are scaled to unit norm (norm is a known constant)."""
    return float(np.linalg.norm(unit(Uhat) - unit(U)))


def cos(X, Y):
    nx, ny = np.linalg.norm(X), np.linalg.norm(Y)
    return float(np.sum(X * Y) / (nx * ny)) if nx > 0 and ny > 0 else float("nan")


def summ(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0)
    return dict(n=int(v.size), med=float(np.median(v)), p90=float(np.quantile(v, .9)), max=float(v.max()))


def fmt(s):
    if not s or s.get("n", 0) == 0:
        return "   n/a"
    return f"{s['med']:.2e} / {s['p90']:.2e} / {s['max']:.2e}"


def halves(ms):
    """deterministic split: even-index modules = fit, odd = test."""
    return ms[0::2], ms[1::2]
