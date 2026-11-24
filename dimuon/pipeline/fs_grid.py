"""fs_grid.py -- recover the fp16 matrix X16 behind U = c * X16 (FIRSTSTEP-01).
U is float32 but every module is (one float32 scalar) x (an fp16-representable matrix).
Given a reference X (e.g. a simulation) that fixes the power-of-two ambiguity, return X16 and c."""
import numpy as np

def _ongrid(x, bits=11):
    m, e = np.frexp(x); q = np.ldexp(np.round(np.ldexp(m, bits)), -bits); return np.abs(q - m) / np.abs(m)

def recover(U, Xref=None, bits=11, nsamp=400):
    u = U.ravel(); nz = u[u != 0]
    i = np.argmax(np.abs(nz)); sub = nz[np.random.default_rng(0).choice(len(nz), min(nsamp, len(nz)), replace=False)]
    best = (0, None)
    for k in range(2 ** (bits - 1), 2 ** bits):
        cc = abs(nz[i]) / (k / 2 ** bits)
        f = np.mean(_ongrid(sub / cc, bits) < 1e-6)
        if f > best[0]: best = (f, cc)
    c = best[1]
    if Xref is not None:  # choose sign and power of two so that U/c ~ Xref
        s = np.sign(np.vdot(U, Xref)); ratio = np.linalg.norm(U) / np.linalg.norm(Xref)
        p = np.round(np.log2(ratio / c)); c = s * c * 2.0 ** p
    X = U / c
    Xq = np.asarray(X, np.float32).astype(np.float16).astype(np.float64)
    return Xq, c, float(np.max(np.abs(Xq - X) / np.maximum(np.abs(X), 1e-30))), best[0]
