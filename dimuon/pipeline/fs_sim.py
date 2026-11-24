"""fs_sim.py -- full-matrix simulation of a Newton-Schulz / Polar-Express step-1 update (FIRSTSTEP-01).

U_hat = -lr * R(r) * X / ||X||_F,  X = NS(G)   (shape d_out x r)
NS: work on the wide orientation (r x d_out), X0 = G / (||G||_F * k_sf + eps), then for each
(a,b,c): A = X X^T ; B = b A + c A A ; X = a X + B X.
Precision modes: 'f64' (all float64), 'f32', 'f16', 'bf16' (store dtype after every op; matmuls
accumulate in float32 and are rounded to the store dtype, like torch on GPU).
"""
import numpy as np
from fs_spec import PE_MEM, safety

def bf16(x):
    x32 = np.ascontiguousarray(x, dtype=np.float32)
    b = x32.view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.astype(np.uint32).view(np.float32).reshape(x32.shape)

def rnd(x, p):
    if p == 'f64': return np.asarray(x, np.float64)
    if p == 'f32': return np.asarray(x, np.float32)
    if p == 'f16': return np.asarray(x, np.float32).astype(np.float16).astype(np.float32)
    if p == 'bf16': return bf16(x)
    raise ValueError(p)

def mm(X, Y, p):
    if p == 'f64': return X @ Y
    return rnd(np.asarray(X, np.float32) @ np.asarray(Y, np.float32), p)

def ns(G, coefs, p='f16', k_sf=1.01, eps=1e-7, norm_prec=None, cast_first=True, fused=True):
    """Return X (d_out x r) in float64 holding the values of the working dtype."""
    G = np.asarray(G, np.float64)
    X = G.T  # wide: r x d_out
    if p == 'f64':
        X = X / (np.linalg.norm(X) * k_sf + eps)
    else:
        if cast_first:
            X = rnd(X, p)
            nrm = rnd(np.sqrt(np.sum(np.asarray(X, np.float32) ** 2, dtype=np.float32)), p)  # norm accumulates in f32, output in p
            den = rnd(rnd(nrm * np.float32(k_sf), p) + np.float32(eps), p)
            X = rnd(X / den, p)
        else:  # normalise in float32 then cast
            X32 = np.asarray(X, np.float32)
            X = rnd(X32 / (np.linalg.norm(X32) * np.float32(k_sf) + np.float32(eps)), p)
    for a, b, c in coefs:
        A = mm(X, X.T, p)
        if p == 'f64':
            B = b * A + c * (A @ A); X = a * X + B @ X
        else:
            A32 = np.asarray(A, np.float32); X32 = np.asarray(X, np.float32)
            if fused:  # torch.baddbmm/addmm style: one rounding per op, fp32 inside
                B = rnd(np.float32(b) * A32 + np.float32(c) * (A32 @ A32), p)
                X = rnd(np.float32(a) * X32 + np.asarray(B, np.float32) @ X32, p)
            else:      # b*A + c*(A@A); a*X + B@X with every intermediate rounded
                B = rnd(rnd(np.float32(b) * A, p) + rnd(np.float32(c) * mm(A, A, p), p), p)
                X = rnd(rnd(np.float32(a) * X, p) + mm(B, X, p), p)
    return np.asarray(X, np.float64).T

def rank_scale(r, lr):
    return lr * np.sqrt(r) * (r / 32.0) ** 0.175

def update(G, lr, coefs, p='f16', final='fro32', **kw):
    X = ns(G, coefs, p, **kw)
    r = G.shape[1]
    if final == 'fro64':
        return -rank_scale(r, lr) * X / np.linalg.norm(X)
    if final == 'fro32':
        X32 = X.astype(np.float32)
        n = np.sqrt(np.sum(X32 * X32, dtype=np.float32))
        return (-(np.float32(rank_scale(r, lr)) / n) * X32).astype(np.float64)
    raise ValueError(final)

PE5 = safety(PE_MEM[:5], 1.01)
