"""fs_ns16.py -- fp16 Newton-Schulz emulation with selectable rounding points (FIRSTSTEP-01).
X0 = fp16( G_f32 / (||G||_f32 * 1.01 + 1e-7) ) ; wide orientation (r x d_out).
variant flags: rAA (round A@A before combining), rB_terms (round b*A and c*AA separately),
rBX (round B@X before adding a*X), rA (round A=X X^T to fp16; default True)."""
import numpy as np
f32 = np.float32
def H(x): return np.asarray(x, f32).astype(np.float16).astype(f32)

def x0_f16(G, sf=1.01, eps=1e-7):
    G32 = np.asarray(G, f32)
    n = np.sqrt(np.sum(G32 * G32, dtype=f32))
    return H(G32 / (n * f32(sf) + f32(eps)))

def ns16(G, coefs, rA=True, rAA=False, rB_terms=False, rBX=False, X0=None):
    X = (x0_f16(G) if X0 is None else np.asarray(X0, f32)).T.copy()
    for a, b, c in coefs:
        A = X @ X.T
        A = H(A) if rA else A.astype(f32)
        AA = A @ A
        if rAA: AA = H(AA)
        if rB_terms: B = H(H(f32(b) * A) + H(f32(c) * AA))
        else: B = H(f32(b) * A + f32(c) * AA)
        BX = B @ X
        if rBX: BX = H(BX)
        X = H(f32(a) * X + BX)
    return X.T.astype(np.float64)
