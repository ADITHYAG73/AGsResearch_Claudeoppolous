"""bc_lib.py -- BIASCORR-01: candidate forms of the step-1 momentum / normalisation arithmetic.

At step 1 the stored momentum S = 0. A candidate maps (g, beta) -> M, the float32 tensor that is
normalised and orthogonalised. Everything after M is the COMPARE-01 nested (Horner) emulation:
    X0 = fp16( M / (||M||_f32 * 1.01 + eps) )        (tall d x r)
    5 Polar-Express steps, Horner form with separate fp16 store of c*R  (cmp_horner.horner)
    U  = -lr*sqrt(r)*(r/32)^0.175 * X / ||X||_F       (float32)
g is the bf16-valued oracle gradient (float32 array).
"""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fs_spec import PE_MEM, safety
from fs_sim import rank_scale, bf16

f32, f16 = np.float32, np.float16
PE5 = safety(PE_MEM[:5], 1.01)


def H(x):
    return np.asarray(x, f32).astype(f16).astype(f32)


def a64(beta):            # 1 - beta in python float64, cast to float32 (torch alpha=1-beta)
    return f32(1.0 - beta)


def a32(beta):            # 1 - beta computed in float32
    return f32(f32(1.0) - f32(beta))


def _lerp(g, w, fma):
    """torch.lerp(S=0, g, w) CUDA semantics: w<0.5: S + w*(g-S); else g - (g-S)*(1-w), 1-w in f32."""
    w = f32(w)
    if abs(w) < 0.5:
        return (w * g).astype(f32)
    omw = f32(f32(1.0) - w)
    if fma:  # one rounding: fma(-(g), omw, g)
        return (g.astype(np.float64) - g.astype(np.float64) * np.float64(omw)).astype(f32)
    return (g - (g * omw).astype(f32)).astype(f32)


def _bc(beta, kind):      # bias correction 1 - beta^1
    return f32(1.0 - beta) if kind == "f64" else f32(f32(1.0) - f32(beta))


CANDS = {
    "g": lambda g, b: g,
    "ema_a64": lambda g, b: (a64(b) * g).astype(f32),
    "ema_a32": lambda g, b: (a32(b) * g).astype(f32),
    "lerp_a64": lambda g, b: _lerp(g, a64(b), False),
    "lerp_a64_fma": lambda g, b: _lerp(g, a64(b), True),
    "lerp_a32": lambda g, b: _lerp(g, a32(b), False),
    "ema_a64_bcdiv64": lambda g, b: ((a64(b) * g).astype(f32) / _bc(b, "f64")).astype(f32),
    "ema_a64_bcdiv32": lambda g, b: ((a64(b) * g).astype(f32) / _bc(b, "f32")).astype(f32),
    "ema_a32_bcdiv32": lambda g, b: ((a32(b) * g).astype(f32) / _bc(b, "f32")).astype(f32),
    "ema_a64_bcmul64": lambda g, b: ((a64(b) * g).astype(f32) * f32(1.0 / (1.0 - b))).astype(f32),
    "lerp_a64_bcdiv64": lambda g, b: (_lerp(g, a64(b), False) / _bc(b, "f64")).astype(f32),
    "ema_a64_bf16": lambda g, b: bf16((a64(b) * g).astype(f32)),
    "ema_a64_fp16": lambda g, b: H((a64(b) * g).astype(f32)),
}


def x0(M, eps=1e-7, sf=1.01):
    M = np.asarray(M, f32)
    n = np.sqrt(np.sum(M * M, dtype=f32))
    return H(M / (n * f32(sf) + f32(eps)))


def horner_from_x0(X):
    r = X.shape[1]
    I = np.eye(r, dtype=f32)
    for a, b, c in PE5:
        R = H(X.T @ X)
        cR = H(f32(c) * R)
        inner = H(cR + f32(b) * I)
        P = H(H(R @ inner) + f32(a) * I)
        X = H(X @ P)
    return X                 # fp16 values in float32


def fin(X, r, lr=1e-3):
    X32 = np.asarray(X, f32)
    n = np.sqrt(np.sum(X32 * X32, dtype=f32))
    return (-(f32(rank_scale(r, lr)) / n) * X32).astype(f32)


def predict(g, beta, cand, eps=1e-7, lr=1e-3):
    """Return (X fp16-valued float32 matrix, U float32) predicted for one tensor g (d x r)."""
    M = CANDS[cand](np.asarray(g, f32), beta)
    X = horner_from_x0(x0(M, eps))
    return X, fin(X, g.shape[1], lr)


def x16_of(U, Xref):
    """fp16 matrix behind measured U (float32), sign/power-of-two fixed by Xref (fs_grid.recover)."""
    from fs_grid import recover
    Xd, c, resid, ongrid = recover(np.asarray(U, np.float64), Xref=np.asarray(Xref, np.float64))
    return Xd.astype(f32), ongrid
