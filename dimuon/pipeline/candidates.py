#!/usr/bin/env python
"""
candidates.py -- analysis toolkit for the first Dimuon step on a LoRA factor.

Data model: for each LoRA module we have
    G : gradient of lora_B at step 1, shape (d_out, r)
    D : the optimizer's one-step change of lora_B (B1 - B0), shape (d_out, r)
    K : optional (r, r) Gram matrix A A^T of lora_A (lets A-dependent rules be scored)
At step 1 any momentum buffer is a scalar multiple of G (m = G or (1-beta) G, Nesterov
beta*m + G), so the update direction is U = phi(G) and D = -lr * scale * U.  All per-module
scoring below is invariant to that scalar; the "global" score shares ONE scalar across modules
(fit on the training modules), so it also tests how |D| depends on shape / rank / |G|.

Contents
  1. candidate maps G -> U         (sign, Frobenius, row/col norm, polar, Newton-Schulz in
                                    float64/float32/simulated bf16, conditioned polar
                                    L^a polar(L^b G R^c [K^e]) R^d [K^f], diagonal
                                    preconditioning L^a G R^c, spectral U f(s) V^T,
                                    renormalisation modes)
  2. diagnostics(G, D)             hypothesis-free: spectra, subspace angles, transfer function,
                                    right-operator test D = L^q G S, row/col profiles, bf16
                                    lattice test; norm_scaling() across modules
  3. score()/fit_candidate()       optimal-scalar scoring, least-squares fitting of free
                                    parameters, held-out split, ranked table
  4. synthetic generators + self-test + identifiability sweep

CLI
  python candidates.py selftest                 # ~1-2 min, asserts, prints tables
  python candidates.py identifiability [--out DIR] [--workers 4]   # full sweep (~10 min)
  python candidates.py analyze PATH [--seed 0]  # real data: dir of *.npz with G, D [, A | K]

numpy float64 throughout unless a precision mode is requested.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pathlib
import sys
import time
import zlib
from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable, Sequence

import numpy as np
from scipy import linalg as sla
from scipy import optimize

# =============================================================================================
# 0. precision helpers
# =============================================================================================


def bf16_round(x) -> np.ndarray:
    """Round to the nearest bfloat16 (round-to-nearest-even), returned as float32."""
    x32 = np.ascontiguousarray(x, dtype=np.float32)
    b = x32.view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.astype(np.uint32).view(np.float32).reshape(x32.shape)


def _rnd(x, precision: str):
    if precision == "float64":
        return np.asarray(x, dtype=np.float64)
    if precision == "float32":
        return np.asarray(x, dtype=np.float32)
    if precision == "bf16":
        return bf16_round(x)
    raise ValueError(precision)


def _mm(X, Y, precision: str):
    """Matmul with fp32 accumulate and output rounded to the working precision (torch bf16 style)."""
    if precision == "float64":
        return X @ Y
    out = np.asarray(X, np.float32) @ np.asarray(Y, np.float32)
    return bf16_round(out) if precision == "bf16" else out


# =============================================================================================
# 1. candidate maps
# =============================================================================================

QUINTIC_MUON = (3.4445, -4.7750, 2.0315)
CUBIC = (1.5, -0.5)


def frob(G):
    return G / np.linalg.norm(G)


def sign(G):
    return np.sign(G)


def row_norm(G, eps=0.0):
    n = np.linalg.norm(G, axis=1, keepdims=True)
    return G / np.maximum(n + eps, 1e-300)


def col_norm(G, eps=0.0):
    n = np.linalg.norm(G, axis=0, keepdims=True)
    return G / np.maximum(n + eps, 1e-300)


def svd_thin(G):
    return np.linalg.svd(G, full_matrices=False)


def spectral_map(G, f: Callable[[np.ndarray], np.ndarray], tol=1e-12, svd=None):
    """U diag(f(s)) V^T. Singular values below tol*s_max are treated as exactly zero (f=0)."""
    U, s, Vt = svd if svd is not None else svd_thin(G)
    fs = np.zeros_like(s)
    keep = s > tol * (s[0] if s.size else 0.0)
    fs[keep] = f(s[keep])
    return (U * fs) @ Vt


def polar(G, svd=None):
    """Exact (pseudo-)polar factor U V^T."""
    return spectral_map(G, np.ones_like, svd=svd)


def spec_power(G, p, svd=None):
    """U diag(s^p) V^T; p=1 is G itself, p=0 is polar, 0<p<1 is soft orthogonalisation."""
    return spectral_map(G, lambda s: s ** p, svd=svd)


def _expand(coeffs, steps):
    if len(coeffs) and isinstance(coeffs[0], (tuple, list, np.ndarray)):
        return [tuple(c) for c in coeffs]
    return [tuple(coeffs)] * int(steps)


def ns_transfer(x, coeffs=QUINTIC_MUON, steps=5):
    """Scalar map on singular values: x <- sum_k c_k x^(2k+1), iterated."""
    x = np.asarray(x, dtype=np.float64)
    for c in _expand(coeffs, steps):
        x = sum(ck * x ** (2 * k + 1) for k, ck in enumerate(c))
    return x


def newton_schulz(G, coeffs=QUINTIC_MUON, steps=5, precision="float64", prenorm="fro", eps=1e-7):
    """Muon-style odd polynomial iteration X <- c0 X + c1 (XX^T) X + c2 (XX^T)^2 X + ...

    coeffs: one tuple (reused for `steps` steps) or a list of per-step tuples.
    Works on the wide orientation (r x d) as Keller Jordan's implementation does; prenorm 'fro'
    divides by ||G||_F + eps first (Muon), 'spec' by the exact spectral norm, None = no scaling.
    precision: 'float64' | 'float32' | 'bf16' (bf16 inputs, fp32-accumulated matmuls, every
    intermediate rounded to bf16). Returns float64.
    """
    cl = _expand(coeffs, steps)
    tall = G.shape[0] > G.shape[1]
    X = _rnd(G.T if tall else G, precision)
    if prenorm == "fro":
        n = float(np.linalg.norm(np.asarray(X, np.float64)))
    elif prenorm == "spec":
        n = float(np.linalg.norm(np.asarray(X, np.float64), 2))
    else:
        n, eps = 1.0, 0.0
    X = _rnd(X / _rnd(n + eps, precision), precision)
    for c in cl:
        A = _mm(X, X.T, precision)
        B = None
        P = None
        for k in range(1, len(c)):
            P = A if P is None else _mm(P, A, precision)
            term = _rnd(c[k] * P, precision)
            B = term if B is None else _rnd(B + term, precision)
        Xn = _rnd(c[0] * X, precision)
        if B is not None:
            Xn = _rnd(Xn + _mm(B, X, precision), precision)
        X = Xn
    X = np.asarray(X, np.float64)
    return X.T if tall else X


def ns_spectral(G, coeffs=QUINTIC_MUON, steps=5, prenorm="fro", svd=None, eps=1e-7):
    """Exact-arithmetic Newton-Schulz via the SVD (fast path; equals newton_schulz float64,
    including the +eps on the norm, which matters when ||G|| is not >> eps)."""
    U, s, Vt = svd if svd is not None else svd_thin(G)
    if prenorm == "fro":
        n = math.sqrt(float((s ** 2).sum())) + eps
    elif prenorm == "spec":
        n = s[0] + eps
    else:
        n = 1.0
    return (U * ns_transfer(s / n, coeffs, steps)) @ Vt


def energies(G):
    """Row energies (sum of squares along columns) and column energies."""
    G2 = G * G
    return G2.sum(1), G2.sum(0)


def _epow(e, p, damping=0.0):
    if p == 0:
        return np.ones_like(e)
    e = e + damping * e.mean()
    e = np.maximum(e, 1e-30 * max(e.max(), 1e-300))
    return e ** p


def spd_pow(K, p, floor=1e-12):
    w, Q = np.linalg.eigh((K + K.T) / 2)
    w = np.maximum(w, floor * w.max())
    return (Q * w ** p) @ Q.T


def polar_fast(M):
    """M (M^T M)^{-1/2} via an r x r eigendecomposition (pinv cutoff 1e-14 on eigenvalues)."""
    C = M.T @ M
    w, Q = np.linalg.eigh(C)
    wmax = w.max() if w.size else 0.0
    inv = np.where(w > 1e-14 * wmax, 1.0 / np.sqrt(np.maximum(w, 1e-300)), 0.0)
    return M @ ((Q * inv) @ Q.T)


def conditioned_polar(G, a=0.0, b=0.0, c=0.0, d=0.0, damping=0.0, K=None, e=0.0, f=0.0,
                      inner: Callable | None = None):
    """U = L^a polar(L^b G R^c K^e) R^d K^f.

    L = diag(row energies + damping*mean), R = diag(column energies + damping*mean).
    Special cases: all zero -> polar;  b=-1/2 -> polar of row-normalised G;  c=-1/2 -> polar of
    column-normalised G;  b=c=-1/4 -> diagonal-Shampoo-preconditioned polar;  e=-1/2 with
    K = A A^T -> polar of G (AA^T)^{-1/2} (a 'Riemannian'/scaled-LoRA style preconditioner).
    `inner` replaces the exact polar (e.g. lambda M: newton_schulz(M, ...)).
    """
    rowE, colE = energies(G)
    lb, rc = _epow(rowE, b, damping), _epow(colE, c, damping)
    M = lb[:, None] * G * rc[None, :]
    if K is not None and e != 0:
        M = M @ spd_pow(K, e)
    P = (inner or polar_fast)(M)
    U = _epow(rowE, a, damping)[:, None] * P * _epow(colE, d, damping)[None, :]
    if K is not None and f != 0:
        U = U @ spd_pow(K, f)
    return U


def diag_precond(G, a=0.0, c=0.0, damping=0.0):
    """L^a G R^c (no orthogonalisation). a=-1/2 -> row_norm; c=-1/2 -> col_norm; a=c=-1/4 ->
    Adafactor/diagonal-Shampoo-like; a=c=0 -> G."""
    rowE, colE = energies(G)
    return _epow(rowE, a, damping)[:, None] * G * _epow(colE, c, damping)[None, :]


def renorm(U, mode="none"):
    """Fixed-norm renormalisation (only matters for the global-scale score)."""
    if mode == "none":
        return U
    d, r = U.shape
    if mode == "fro":
        return U / np.linalg.norm(U)
    if mode == "fro_sqrt_r":        # same Frobenius norm as an exact polar factor
        return U * (math.sqrt(r) / np.linalg.norm(U))
    if mode == "spec":
        return U / np.linalg.norm(U, 2)
    if mode == "rms":               # entry RMS = 1
        return U * (math.sqrt(U.size) / np.linalg.norm(U))
    if mode == "spec_muon_shape":   # spectral 1 then * sqrt(max(1, d_out/r))
        return U / np.linalg.norm(U, 2) * math.sqrt(max(1.0, d / r))
    raise ValueError(mode)


RENORM_MODES = ("none", "fro", "fro_sqrt_r", "spec", "rms", "spec_muon_shape")


# ---------------------------------------------------------------------------------------------
# candidate registry
# ---------------------------------------------------------------------------------------------


@dataclass
class Candidate:
    name: str
    fn: Callable  # fn(module, theta) -> U
    pnames: tuple = ()
    starts: list = field(default_factory=list)
    needs_K: bool = False
    note: str = ""
    n_refine: int = 3
    spectral_f: Callable | None = None  # f(s, theta) for U f(s) V^T rules: O(r) exact objective
    spectral_f_batch: Callable | None = None  # f(s (r,), TH (N,k)) -> (N, r), for start grids

    def __call__(self, mod, theta=()):
        return self.fn(mod, tuple(theta))


def _fixed(name, f, note="", needs_K=False):
    return Candidate(name, lambda m, th: f(m), (), [], needs_K, note)


def fixed_candidates(include_precision=True, include_K=False):
    c = [
        _fixed("frob", lambda m: frob(m.G), "G/||G||_F (= spec_power p=1)"),
        _fixed("sign", lambda m: sign(m.G), "sign(G) (= Adam/Lion first step)"),
        _fixed("row_norm", lambda m: row_norm(m.G)),
        _fixed("col_norm", lambda m: col_norm(m.G)),
        _fixed("polar", lambda m: polar(m.G, m.svd)),
        _fixed("ns_quintic5", lambda m: ns_spectral(m.G, QUINTIC_MUON, 5, svd=m.svd)),
        _fixed("ns_quintic3", lambda m: ns_spectral(m.G, QUINTIC_MUON, 3, svd=m.svd)),
        _fixed("ns_quintic8", lambda m: ns_spectral(m.G, QUINTIC_MUON, 8, svd=m.svd)),
        _fixed("ns_cubic5", lambda m: ns_spectral(m.G, CUBIC, 5, svd=m.svd)),
        _fixed("ns_cubic15", lambda m: ns_spectral(m.G, CUBIC, 15, svd=m.svd)),
        _fixed("spec_p0.5", lambda m: spec_power(m.G, 0.5, m.svd)),
        _fixed("dpre_-.25_-.25", lambda m: diag_precond(m.G, -0.25, -0.25), "Adafactor-like L^-1/4 G R^-1/4"),
        _fixed("cpolar_rowwhite", lambda m: conditioned_polar(m.G, b=-0.5), "polar(L^-1/2 G)"),
        _fixed("cpolar_colwhite", lambda m: conditioned_polar(m.G, c=-0.5), "polar(G R^-1/2)"),
        _fixed("cpolar_dshampoo", lambda m: conditioned_polar(m.G, b=-0.25, c=-0.25), "polar(L^-1/4 G R^-1/4)"),
    ]
    if include_precision:
        c += [
            _fixed("ns_quintic5_f32", lambda m: newton_schulz(m.G, QUINTIC_MUON, 5, "float32")),
            _fixed("ns_quintic5_bf16", lambda m: newton_schulz(m.G, QUINTIC_MUON, 5, "bf16")),
        ]
    if include_K:
        c += [_fixed("apolar_-.5", lambda m: conditioned_polar(m.G, K=m.K, e=-0.5), "polar(G (AA^T)^-1/2)", True)]
    return c


_G3 = (-0.6, -0.1, 0.4)  # start grid deliberately does NOT contain the hidden values used in tests


def _ns_fit_fn(m, th, steps=5):
    out = ns_spectral(m.G, tuple(th), steps, svd=m.svd)
    return out if np.all(np.isfinite(out)) else np.full_like(m.G, np.nan)


def _ns5_batch(s, TH):
    x = np.broadcast_to(s / (math.sqrt(float((s ** 2).sum())) + 1e-7), (TH.shape[0], s.size)).copy()
    c0, c1, c2 = (TH[:, j:j + 1] for j in range(3))
    for _ in range(5):
        x2 = x * x
        x = x * (c0 + x2 * (c1 + c2 * x2))
    return x


def _objective_batch(cand, mods, TH):
    TH = np.asarray(TH, float)
    tot = np.zeros(TH.shape[0])
    for m in mods:
        s = m.svd[1]
        with np.errstate(all="ignore"):
            F = cand.spectral_f_batch(s, TH)
            F[:, ~(s > 1e-12 * s[0])] = 0.0
            t, D2 = m.tdiag
            nf = (F * F).sum(1)
            v = 1.0 - (F @ t) ** 2 / (nf * D2)
        v[~np.isfinite(v) | (nf == 0)] = 1.0
        tot += np.maximum(v, 0.0)
    return tot / len(mods)


def family_candidates(include_K=False):
    fam = [
        Candidate("FIT spec_power", lambda m, th: spec_power(m.G, th[0], m.svd), ("p",),
                  [(-0.3,), (0.4,), (1.1,)], spectral_f=lambda s, th: s ** th[0]),
        Candidate("FIT diag_precond", lambda m, th: diag_precond(m.G, *th), ("a", "c"),
                  list(itertools.product(_G3, _G3))),
        Candidate("FIT cpolar", lambda m, th: conditioned_polar(m.G, *th), ("a", "b", "c", "d"),
                  list(itertools.product(_G3, _G3, _G3, _G3)), n_refine=2),
        # the 5-step NS objective is extremely rugged (basin around the Muon coefficients is
        # < 0.1 wide; 400 random starts missed it), so: 20000 seeded random starts in a box + 2
        # generic ones, evaluated with the O(r) spectral objective, refine the best 8.
        Candidate("FIT ns5_coeffs", _ns_fit_fn, ("c0", "c1", "c2"),
                  [(1.5, -0.5, 0.0), (2.0, -1.5, 0.5)]
                  + [tuple(x) for x in np.random.default_rng(12345).uniform((1, -8, 0), (5, 0, 4), (20000, 3))],
                  n_refine=8,
                  spectral_f=lambda s, th: ns_transfer(s / (math.sqrt(float((s ** 2).sum())) + 1e-7), th, 5),
                  spectral_f_batch=_ns5_batch),
    ]
    if include_K:
        fam.append(Candidate("FIT apolar", lambda m, th: conditioned_polar(m.G, K=m.K, e=th[0], f=th[1]),
                             ("e", "f"), list(itertools.product(_G3, _G3)), True))
    return fam


# =============================================================================================
# module container
# =============================================================================================


@dataclass
class Module:
    G: np.ndarray
    D: np.ndarray
    K: np.ndarray | None = None
    name: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.G = np.asarray(self.G, np.float64)
        self.D = np.asarray(self.D, np.float64)
        assert self.G.shape == self.D.shape, (self.G.shape, self.D.shape)

    @cached_property
    def svd(self):
        return svd_thin(self.G)

    @cached_property
    def tdiag(self):
        """diag(U_G^T D V_G) and ||D||_F^2: sufficient statistics for U f(s) V^T rules."""
        U, s, Vt = self.svd
        return np.einsum("ij,ij->j", U, self.D @ Vt.T), float((self.D ** 2).sum())


def load_modules(path) -> list[Module]:
    """Directory of *.npz (keys G, D, optional A (r x d_in) or K (r x r)), or one .npz with keys
    '<name>/G', '<name>/D' [, '<name>/A' | '<name>/K']."""
    path = pathlib.Path(path)
    mods = []

    def mk(name, get):
        K = get("K")
        A = get("A")
        if K is None and A is not None:
            A = np.asarray(A, np.float64)
            K = A @ A.T
        return Module(get("G"), get("D"), K, name)

    if path.is_dir():
        for f in sorted(path.glob("*.npz")):
            z = np.load(f)
            mods.append(mk(f.stem, lambda k, z=z: z[k] if k in z.files else None))
    else:
        z = np.load(path)
        names = sorted({k.rsplit("/", 1)[0] for k in z.files if "/" in k})
        for n in names:
            mods.append(mk(n, lambda k, n=n: z[f"{n}/{k}"] if f"{n}/{k}" in z.files else None))
    return mods


# =============================================================================================
# 2. hypothesis-free diagnostics
# =============================================================================================


def _slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 2 or np.ptp(x) == 0:
        return float("nan"), float("nan"), float("nan")
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    r2 = 1 - (res ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-300)
    return float(coef[0]), float(coef[1]), float(r2)


def right_operator(G, D, q=0.0):
    """Fit D ~ L^q G S (S r x r) by least squares. Returns out-of-span fraction and S's structure.
    Every rule of the form D = G S (polar, NS, spectral, column scalings, A-preconditioning) has
    out_of_span = 0 at q=0; conditioned polar has it at q = a+b."""
    rowE, _ = energies(G)
    Dq = D / _epow(rowE, q)[:, None]
    S, *_ = np.linalg.lstsq(G, Dq, rcond=None)
    res = Dq - G @ S
    oos = float((res ** 2).sum() / max((Dq ** 2).sum(), 1e-300))
    M = G.T @ G
    nS = np.linalg.norm(S)
    return dict(
        out_of_span=oos,
        S=S,
        S_asym=float(np.linalg.norm(S - S.T) / max(nS, 1e-300)),
        S_commutes_GtG=float(np.linalg.norm(S @ M - M @ S) / max(nS * np.linalg.norm(M), 1e-300)),
        S_offdiag=float(np.linalg.norm(S - np.diag(np.diag(S))) / max(nS, 1e-300)),
    )


def left_exponent_scan(G, D, grid=np.linspace(-1.5, 1.5, 61)):
    """Find q minimising the out-of-span fraction of L^-q D w.r.t. col(G). Hypothesis-light
    estimate of a+b in L^a polar(L^b G R^c) R^d. Meaningless for r = d."""
    vals = np.array([right_operator(G, D, q)["out_of_span"] for q in grid])
    i = int(np.argmin(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    r = optimize.minimize_scalar(lambda q: right_operator(G, D, q)["out_of_span"], bounds=(lo, hi),
                                 method="bounded", options=dict(xatol=1e-6))
    return dict(q_best=float(r.x), out_of_span_at_best=float(r.fun),
                out_of_span_at_0=float(right_operator(G, D, 0.0)["out_of_span"]))


def bf16_lattice(x, scale=1.0):
    """Median relative distance of x/scale to the bf16 grid. ~0 (1e-8) if x/scale was a bf16
    number cast to fp32; ~1e-3 for generic fp32 values."""
    y = np.asarray(x, np.float64).ravel() / scale
    y = y[y != 0]
    if y.size == 0:
        return float("nan")
    return float(np.median(np.abs(bf16_round(y).astype(np.float64) - y) / np.abs(y)))


def diagnostics(G, D, K=None):
    """Hypothesis-free description of one (G, D) pair."""
    G = np.asarray(G, np.float64)
    D = np.asarray(D, np.float64)
    d, r = G.shape
    Ug, sg, Vgt = svd_thin(G)
    Ud, sd, Vdt = svd_thin(D)
    out = dict(d_out=d, r=r)
    out["sv_G"] = sg
    out["sv_D"] = sd
    out["sv_D_rel"] = sd / sd[0]
    out["fro_D"] = float(np.linalg.norm(D))
    out["spec_D"] = float(sd[0])
    out["fro_over_spec_D"] = out["fro_D"] / out["spec_D"]  # = sqrt(r) for an exact polar
    out["cond_G"] = float(sg[0] / max(sg[-1], 1e-300))
    out["cond_D"] = float(sd[0] / max(sd[-1], 1e-300))
    # left subspaces (d-dim, r-dim subspaces): principal angles
    ang = sla.subspace_angles(G, D) if r < d else np.zeros(r)
    out["left_principal_angles_deg"] = np.degrees(np.sort(ang))
    # right singular vectors: |V_G^T V_D|
    out["right_overlap"] = np.abs(Vgt @ Vdt.T)
    # transfer function in G's singular basis
    T = Ug.T @ D @ Vgt.T
    dg = np.diag(T)
    out["transfer_offdiag_frac"] = float(np.linalg.norm(T - np.diag(dg)) / max(np.linalg.norm(T), 1e-300))
    out["outside_colG_frac"] = float(1 - (T ** 2).sum() / max((D ** 2).sum(), 1e-300)) if r < d else 0.0
    out["transfer_diag"] = dg
    sgn = np.sign(dg.sum()) or 1.0
    out["transfer_vs_s"] = np.vstack([sg, sg / math.sqrt((sg ** 2).sum()), dg * sgn]).T
    out["transfer_loglog_slope"] = _slope(np.log(sg), np.log(np.abs(dg)))[0] if r > 1 else float("nan")
    # right operator / left exponent
    ro = right_operator(G, D)
    out.update({k: v for k, v in ro.items() if k != "S"})
    if r < d:
        out.update({f"lscan_{k}": v for k, v in left_exponent_scan(G, D).items()})
    # row / column profiles
    rnG, rnD = np.linalg.norm(G, axis=1), np.linalg.norm(D, axis=1)
    cnG, cnD = np.linalg.norm(G, axis=0), np.linalg.norm(D, axis=0)
    out["row_slope"], _, out["row_r2"] = _slope(np.log(rnG), np.log(rnD))
    out["row_cv_D"] = float(rnD.std() / rnD.mean())
    out["col_slope"], _, out["col_r2"] = _slope(np.log(cnG), np.log(cnD))
    out["col_cv_D"] = float(cnD.std() / cnD.mean())
    # elementwise
    nz = (G != 0) & (D != 0)
    out["sign_agree_frac"] = float(np.mean(np.sign(G[nz]) == np.sign(D[nz] * sgn)))
    out["abs_D_cv"] = float(np.abs(D).std() / np.abs(D).mean())  # 0 for sign(G)
    out["bf16_lattice_D"] = bf16_lattice(D)
    if K is not None:
        w = np.linalg.eigvalsh(K)
        out["K_cond"] = float(w.max() / max(w.min(), 1e-300))
    return out


def print_diagnostics(dg, name=""):
    f = lambda v: np.array2string(np.asarray(v), precision=4, max_line_width=150)
    print(f"--- diagnostics {name}  d_out={dg['d_out']} r={dg['r']}")
    for k in ("sv_G", "sv_D", "sv_D_rel", "left_principal_angles_deg", "transfer_diag"):
        print(f"  {k:28s} {f(dg[k])}")
    for k, v in dg.items():
        if isinstance(v, float):
            print(f"  {k:28s} {v:.6g}")


def norm_scaling(mods: Sequence[Module]):
    """Across modules: regress log||D|| (Frobenius, spectral) on [1, log d_out, log r, log||G||_F].
    Exact polar: fro coefficient on log r = 0.5, others 0; spec all 0. Frobenius-renormalised: 0."""
    X = np.array([[1.0, math.log(m.G.shape[0]), math.log(m.G.shape[1]), math.log(np.linalg.norm(m.G))]
                  for m in mods])
    out = {}
    for key, fn in (("fro", lambda D: np.linalg.norm(D)), ("spec", lambda D: np.linalg.norm(D, 2))):
        y = np.array([math.log(fn(m.D)) for m in mods])
        cols = [0] + [j for j in (1, 2, 3) if np.ptp(X[:, j]) > 1e-12]
        coef, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
        res = y - X[:, cols] @ coef
        full = dict(zip(["const", "log_d_out", "log_r", "log_G_fro"], [float("nan")] * 4))
        for j, cj in zip(cols, coef):
            full[["const", "log_d_out", "log_r", "log_G_fro"][j]] = float(cj)
        full["resid_sd"] = float(res.std())
        out[key] = full
    return out


def column_response(D1, D2, k):
    """Controlled input: G2 = G1 diag(k) (e.g. lora_A rows scaled by k). Estimate q in
    D2 ~ D1 diag(k^q) (up to a scalar) and how much D2 differs beyond that.
    col_norm -> q=0 exactly; frob -> q=1 exactly (up to scalar); polar/NS -> neither."""
    k = np.asarray(k, float)
    q, _, r2 = _slope(np.log(k), np.log(np.linalg.norm(D2, axis=0) / np.linalg.norm(D1, axis=0)))
    pred = D1 * (k ** q)[None, :]
    c = np.vdot(pred, D2) / (np.linalg.norm(pred) * np.linalg.norm(D2))
    return dict(q=q, r2=r2, resid_after_colscale=float(math.sqrt(max(0.0, 1 - c * c))))


# =============================================================================================
# 3. scoring / fitting harness
# =============================================================================================


def rel_err(U, D):
    """(signed cosine, relative Frobenius error after optimal scalar)."""
    nu, nd = np.linalg.norm(U), np.linalg.norm(D)
    if not np.isfinite(nu) or nu == 0 or nd == 0:
        return 0.0, 1.0
    Un, Dn = U / nu, D / nd
    c = float(np.vdot(Un, Dn))
    return c, float(np.linalg.norm(Dn - c * Un))


def _objective(cand, mods, th):
    tot = 0.0
    if cand.spectral_f is not None:
        # exact: ||Dn - c Un||^2 = 1 - c^2 with c = <f,t>/(|f| |D|), since u_i v_i^T are orthonormal
        for m in mods:
            s = m.svd[1]
            with np.errstate(all="ignore"):
                f = np.where(s > 1e-12 * s[0], cand.spectral_f(s, th), 0.0)
            t, D2 = m.tdiag
            nf = float(f @ f)
            if not np.isfinite(nf) or nf == 0:
                tot += 1.0
                continue
            tot += max(0.0, 1.0 - float(f @ t) ** 2 / (nf * D2))
        return tot / len(mods)
    for m in mods:
        try:
            with np.errstate(all="ignore"):
                U = cand(m, th)
            e = rel_err(U, m.D)[1]
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            e = 1.0
        tot += e * e if np.isfinite(e) else 1.0
    return tot / len(mods)


def fit_candidate(cand: Candidate, mods: Sequence[Module], n_refine=None, maxiter=400):
    """Least squares over the free parameters: minimise mean squared relative error (after
    per-module optimal scaling) over `mods`. Grid of starts, then Nelder-Mead on log objective."""
    if not cand.pnames:
        return (), _objective(cand, mods, ())
    if cand.spectral_f_batch is not None:
        vals = _objective_batch(cand, mods, cand.starts)
        starts = sorted(zip(vals.tolist(), cand.starts), key=lambda t: t[0])
    else:
        starts = sorted(((_objective(cand, mods, s), s) for s in cand.starts), key=lambda t: t[0])
    best_th, best_f = tuple(starts[0][1]), starts[0][0]
    n_refine = cand.n_refine if n_refine is None else n_refine
    f = lambda th: math.log(_objective(cand, mods, th) + 1e-300)
    for _, s in starts[:n_refine]:
        r = optimize.minimize(f, np.array(s, float), method="Nelder-Mead",
                              options=dict(xatol=1e-7, fatol=1e-9, maxiter=maxiter * len(s),
                                           maxfev=maxiter * len(s)))
        val = math.exp(r.fun)
        if val < best_f:
            best_f, best_th = val, tuple(float(x) for x in r.x)
    return best_th, best_f


def split(n, test_frac=1 / 3, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    nt = max(1, int(round(n * test_frac)))
    if n - nt < 1:
        return list(idx), list(idx)
    return sorted(idx[nt:].tolist()), sorted(idx[:nt].tolist())


def score(mods: Sequence[Module], cands: Sequence[Candidate], train=None, test=None, seed=0,
          renorm_modes=("none",)):
    """Score every candidate. Free parameters are fit on `train` and evaluated on `test`.
    Per-module score: RMS over modules of relative Frobenius error after the optimal per-module
    scalar, plus mean signed cosine. Global score: ONE scalar (fit on train) shared by all
    modules, for each renorm mode; reported as sqrt(sum||aU-D||^2 / sum||D||^2) on test."""
    if train is None:
        train, test = split(len(mods), seed=seed)
    tr = [mods[i] for i in train]
    te = [mods[i] for i in test]
    hasK = all(m.K is not None for m in mods)
    rows = []
    for c in cands:
        if c.needs_K and not hasK:
            continue
        th, ftr = fit_candidate(c, tr)
        errs, coss = [], []
        Ute = []
        for m in te:
            with np.errstate(all="ignore"):
                U = c(m, th)
            cs, e = rel_err(U, m.D)
            errs.append(e)
            coss.append(cs)
            Ute.append(U)
        row = dict(name=c.name, theta=dict(zip(c.pnames, th)), train_err=math.sqrt(ftr),
                   test_err=float(np.sqrt(np.mean(np.square(errs)))), test_err_max=float(np.max(errs)),
                   test_cos=float(np.mean(coss)))
        if renorm_modes:
            Utr = [c(m, th) for m in tr]
            g = {}
            for mode in renorm_modes:
                try:
                    a_num = sum(np.vdot(renorm(U, mode), m.D) for U, m in zip(Utr, tr))
                    a_den = sum(np.linalg.norm(renorm(U, mode)) ** 2 for U in Utr)
                    a = a_num / a_den
                    num = sum(np.linalg.norm(a * renorm(U, mode) - m.D) ** 2 for U, m in zip(Ute, te))
                    den = sum(np.linalg.norm(m.D) ** 2 for m in te)
                    g[mode] = float(math.sqrt(num / den))
                except (ValueError, ZeroDivisionError, FloatingPointError):
                    g[mode] = float("nan")
            row["global_err"] = g
            row["global_best"] = min(g, key=lambda k: g[k] if np.isfinite(g[k]) else 9e9)
        rows.append(row)
    rows.sort(key=lambda r: r["test_err"])
    return rows


def print_table(rows, title="", show_global=True, file=sys.stdout):
    print(f"\n=== {title}", file=file)
    print(f"{'#':>3} {'candidate':22s} {'test_err':>10} {'train_err':>10} {'max_test':>10} {'cos':>8}"
          f"  {'global(best renorm)':>26}  params", file=file)
    best = rows[0]["test_err"]
    for i, r in enumerate(rows):
        th = " ".join(f"{k}={v:+.4f}" for k, v in r["theta"].items())
        tie = "~" if r["test_err"] <= 1.5 * max(best, 1e-300) and i > 0 else " "
        g = ""
        if show_global and "global_err" in r:
            g = f"{r['global_err'][r['global_best']]:.3e} [{r['global_best']}]"
        print(f"{i+1:>3}{tie}{r['name']:22s} {r['test_err']:10.3e} {r['train_err']:10.3e} "
              f"{r['test_err_max']:10.3e} {r['test_cos']:+8.4f}  {g:>26}  {th}", file=file)


# =============================================================================================
# 4. synthetic data
# =============================================================================================


def make_G(d, r, rng, kind="parametric", a_row_scales=None, d_in=2560, n_tok=None):
    """Synthetic lora_B gradients.

    parametric: U0 diag(s) V0^T with geometric spectrum (condition 10-100), then rows scaled by
        lognormal(sigma=1) and columns by lognormal(sigma=0.5). Best case for identifiability.
    sketch: G = Delta (X A^T), the actual structure of a lora_B gradient: Delta (d_out x T)
        per-token output grads with low-rank structure and heavy output channels, X (T x d_in)
        inputs with a few massive channels, A ~ N(0, 1.16^2/(r d_in)) like the measured init.
        Returns (G, K=A A^T). a_row_scales multiplies the rows of A (the 'upload chosen A'
        experiment); it scales G's columns exactly.
    """
    if kind == "parametric":
        U0 = np.linalg.qr(rng.standard_normal((d, r)))[0]
        V0 = np.linalg.qr(rng.standard_normal((r, r)))[0]
        kappa = 10 ** rng.uniform(1, 2)
        s = kappa ** (-np.linspace(0, 1, r)) if r > 1 else np.ones(1)
        Z = (U0 * s) @ V0.T
        G = np.exp(rng.standard_normal(d))[:, None] * Z * np.exp(0.5 * rng.standard_normal(r))[None, :]
        return G * 1e-2, None
    if kind == "sketch":
        T = n_tok or int(rng.integers(40, 180))
        k = 8
        X = rng.standard_normal((T, k)) @ rng.standard_normal((k, d_in)) + 0.3 * rng.standard_normal((T, d_in))
        X[:, rng.choice(d_in, 3, replace=False)] *= 20.0  # massive activation channels
        Dl = (rng.standard_normal((d, k)) @ rng.standard_normal((k, T)) + 0.5 * rng.standard_normal((d, T)))
        Dl *= np.exp(0.8 * rng.standard_normal(d))[:, None] * np.exp(0.5 * rng.standard_normal(T))[None, :]
        A = rng.standard_normal((r, d_in)) * (1.16 / math.sqrt(r * d_in))
        if a_row_scales is not None:
            A = A * np.asarray(a_row_scales, float)[:, None]
        G = Dl @ (X @ A.T)
        return G / np.linalg.norm(G) * 1e-2, A @ A.T
    raise ValueError(kind)


def add_noise(X, eps, rng):
    return X + eps * np.linalg.norm(X) / math.sqrt(X.size) * rng.standard_normal(X.shape)


def hidden_rules(include_K=False):
    """Hidden ('true') rules used in the synthetic study. name -> fn(module) -> U."""
    R = {c.name: (lambda c: (lambda m: c(m, ())))(c) for c in fixed_candidates(True, include_K)}
    R["cpolar(.3,-.4,-.2,.1)"] = lambda m: conditioned_polar(m.G, 0.3, -0.4, -0.2, 0.1)
    return R


FAMILY_TRUTH = {  # hidden rule -> (family, true params) for exponent-recovery checks
    "frob": ("FIT spec_power", (1.0,)),
    "polar": ("FIT spec_power", (0.0,)),
    "spec_p0.5": ("FIT spec_power", (0.5,)),
    "row_norm": ("FIT diag_precond", (-0.5, 0.0)),
    "col_norm": ("FIT diag_precond", (0.0, -0.5)),
    "dpre_-.25_-.25": ("FIT diag_precond", (-0.25, -0.25)),
    "cpolar_rowwhite": ("FIT cpolar", (0.0, -0.5, 0.0, 0.0)),
    "cpolar_colwhite": ("FIT cpolar", (0.0, 0.0, -0.5, 0.0)),
    "cpolar_dshampoo": ("FIT cpolar", (0.0, -0.25, -0.25, 0.0)),
    "cpolar(.3,-.4,-.2,.1)": ("FIT cpolar", (0.3, -0.4, -0.2, 0.1)),
    "ns_quintic5": ("FIT ns5_coeffs", QUINTIC_MUON),
    "apolar_-.5": ("FIT apolar", (-0.5, 0.0)),
}


def make_modules(rule, r, n, rng, kind="parametric", eps=0.0, noise_on="D",
                 dims=(1024, 2048), a_row_scales=None, lr=1e-3):
    mods = []
    for i in range(n):
        d = int(rng.choice(dims))
        G, K = make_G(d, r, rng, kind if kind != "sketch_anisoA" else "sketch",
                      a_row_scales=a_row_scales if kind == "sketch_anisoA" else None)
        m0 = Module(G, np.zeros_like(G), K)
        D = -lr * rule(m0)
        if eps > 0 and noise_on in ("D", "both"):
            D = add_noise(D, eps, rng)
        Gobs = add_noise(G, eps, rng) if eps > 0 and noise_on in ("G", "both") else G
        mods.append(Module(Gobs, D, K, f"{kind}_{i}", dict(d=d, r=r)))
    return mods


# =============================================================================================
# identifiability sweep
# =============================================================================================

RANKS = (1, 2, 4, 8, 16, 64)
NOISES = (1e-6, 1e-3, 1e-2)


def _misfit_job(args):
    """Noise-free misfit of every candidate against one hidden rule: m(T, C) on held-out modules."""
    rule_name, r, kind, seed, n = args
    rng = np.random.default_rng(seed)
    incK = kind.startswith("sketch")
    rules = hidden_rules(include_K=incK)
    ascale = np.logspace(0, -2, r) if kind == "sketch_anisoA" else None
    mods = make_modules(rules[rule_name], r, n, rng, kind, 0.0, a_row_scales=ascale)
    cands = fixed_candidates(True, incK) + family_candidates(incK)
    rows = score(mods, cands, seed=seed, renorm_modes=RENORM_MODES)
    return dict(rule=rule_name, r=r, kind=kind,
                m={row["name"]: row["test_err"] for row in rows},
                theta={row["name"]: row["theta"] for row in rows if row["theta"]},
                global_best={row["name"]: (row["global_best"], row["global_err"][row["global_best"]])
                             for row in rows})


def _noisy_job(args):
    """Noisy verification: does the true rule rank first, and how close is the nearest rival?"""
    rule_name, r, kind, eps, seed, n = args
    rng = np.random.default_rng(seed)
    incK = kind.startswith("sketch")
    rules = hidden_rules(include_K=incK)
    ascale = np.logspace(0, -2, r) if kind == "sketch_anisoA" else None
    mods = make_modules(rules[rule_name], r, n, rng, kind, eps, a_row_scales=ascale)
    cands = fixed_candidates(True, incK) + family_candidates(incK)
    rows = score(mods, cands, seed=seed, renorm_modes=("none",))
    fixed = [x for x in rows if not x["name"].startswith("FIT")]
    names = [x["name"] for x in fixed]
    errT = next((x["test_err"] for x in fixed if x["name"] == rule_name), float("nan"))
    rank = names.index(rule_name) + 1 if rule_name in names else None
    rivals = sorted(((x["test_err"] / errT, x["name"]) for x in fixed if x["name"] != rule_name))
    fam = FAMILY_TRUTH.get(rule_name)
    rec = None
    if fam:
        frow = next((x for x in rows if x["name"] == fam[0]), None)
        if frow:
            rec = dict(family=fam[0], true=list(fam[1]), fitted=list(frow["theta"].values()),
                       fam_test_err=frow["test_err"])
    return dict(rule=rule_name, r=r, kind=kind, eps=eps, errT=errT, rank=rank,
                nearest=rivals[:3], recovery=rec)


def _pool_map(fn, jobs, workers):
    if workers <= 1:
        return [fn(j) for j in jobs]
    import multiprocessing as mp
    import os
    from concurrent.futures import ProcessPoolExecutor
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"  # spawned children read this before importing numpy: no BLAS oversubscription
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(fn, jobs, chunksize=1))


def identifiability(out_dir=None, workers=4, ranks=RANKS, kinds=("parametric", "sketch", "sketch_anisoA"),
                    n=12, verify_ranks=(1, 4, 16, 64), verify_kinds=("parametric", "sketch")):
    t0 = time.time()
    jobs = []
    for kind in kinds:
        names = list(hidden_rules(include_K=kind.startswith("sketch")))
        for r in ranks:
            for j, nm in enumerate(names):
                jobs.append((nm, r, kind, 1000 * r + j, n))
    mis = _pool_map(_misfit_job, jobs, workers)
    print(f"[identifiability] misfit matrix: {len(mis)} jobs in {time.time()-t0:.0f}s", flush=True)
    vjobs = []
    for kind in verify_kinds:
        names = list(hidden_rules(include_K=kind.startswith("sketch")))
        for r in verify_ranks:
            for eps in NOISES:
                for j, nm in enumerate(names):
                    vjobs.append((nm, r, kind, eps, 7 + 1000 * r + j, n))
    ver = _pool_map(_noisy_job, vjobs, workers)
    print(f"[identifiability] noisy verification: {len(ver)} jobs, total {time.time()-t0:.0f}s", flush=True)
    report = render_identifiability(mis, ver)
    if out_dir:
        out = pathlib.Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "identifiability.md").write_text(report)
        (out / "misfit.json").write_text(json.dumps(mis, indent=1, default=float))
        (out / "verify.json").write_text(json.dumps(ver, indent=1, default=float))
        print(f"[identifiability] wrote {out}/identifiability.md, misfit.json, verify.json")
    return mis, ver, report


def render_identifiability(mis, ver, thresh_ratio=1.5):
    """Confusable at noise eps iff noise-free misfit m < eps*sqrt(ratio^2-1) (i.i.d. noise on D:
    err_C^2 ~ m^2 + eps^2, err_T ~ eps, so err_C/err_T < ratio)."""
    k = math.sqrt(thresh_ratio ** 2 - 1)
    L = []
    L.append("# Identifiability of candidate first-step rules (synthetic)\n")
    L.append(f"Confusable pair (T, C) at noise eps: noise-free held-out misfit m(T,C) < {k:.3f}*eps "
             f"(equivalently err_C/err_T < {thresh_ratio} under i.i.d. noise on D). Fixed candidates only; "
             "families are reported as parameter recovery.\n")
    kinds = sorted({x["kind"] for x in mis}, key=["parametric", "sketch", "sketch_anisoA"].index)
    for kind in kinds:
        sub = [x for x in mis if x["kind"] == kind]
        ranks = sorted({x["r"] for x in sub})
        rules = list(dict.fromkeys(x["rule"] for x in sub))
        L.append(f"\n## generator = {kind}\n")
        for eps in NOISES:
            L.append(f"\n### eps = {eps:g}: fixed candidates confusable with the true rule\n")
            L.append("| true rule | " + " | ".join(f"r={r}" for r in ranks) + " |")
            L.append("|---|" + "---|" * len(ranks))
            for T in rules:
                cells = []
                for r in ranks:
                    x = next(y for y in sub if y["rule"] == T and y["r"] == r)
                    conf = [c for c, m in x["m"].items() if c != T and not c.startswith("FIT") and m < k * eps]
                    cells.append(", ".join(conf) if conf else "-")
                L.append(f"| {T} | " + " | ".join(cells) + " |")
        L.append(f"\n### noise-free misfit to the NEAREST other fixed candidate (log10)\n")
        L.append("| true rule | " + " | ".join(f"r={r}" for r in ranks) + " |")
        L.append("|---|" + "---|" * len(ranks))
        for T in rules:
            cells = []
            for r in ranks:
                x = next(y for y in sub if y["rule"] == T and y["r"] == r)
                o = [(m, c) for c, m in x["m"].items() if c != T and not c.startswith("FIT")]
                m, c = min(o)
                cells.append(f"{math.log10(max(m, 1e-17)):.1f} ({c})")
            L.append(f"| {T} | " + " | ".join(cells) + " |")
        L.append(f"\n### noise-free family fits (fitted params; held-out misfit)\n")
        L.append("| true rule | r | " + " | ".join(f.name for f in family_candidates(kind.startswith('sketch'))) + " |")
        L.append("|---|---|" + "---|" * len(family_candidates(kind.startswith('sketch'))))
        for T in rules:
            for r in ranks:
                x = next(y for y in sub if y["rule"] == T and y["r"] == r)
                if T not in FAMILY_TRUTH:
                    continue
                cells = []
                for f in family_candidates(kind.startswith("sketch")):
                    th = x["theta"].get(f.name, {})
                    cells.append(("(" + ",".join(f"{v:+.3f}" for v in th.values()) + f") {x['m'].get(f.name, float('nan')):.1e}")
                                 if f.name in x["m"] else "n/a")
                L.append(f"| {T} | {r} | " + " | ".join(cells) + " |")
    if ver:
        L.append("\n## Noisy verification (n=12 modules, 8 train / 4 test, noise on D)\n")
        L.append("rank = position of the true rule among fixed candidates by held-out error; "
                 "nearest = closest rival as err_C/err_T; recovery = fitted family parameters.\n")
        L.append("| kind | true rule | r | eps | err_T | rank | nearest rival (ratio) | recovered (true) |")
        L.append("|---|---|---|---|---|---|---|---|")
        for v in ver:
            near = f"{v['nearest'][0][1]} ({v['nearest'][0][0]:.3g})" if v["nearest"] else ""
            rec = ""
            if v["recovery"]:
                rr = v["recovery"]
                rec = ("(" + ",".join(f"{x:+.3f}" for x in rr["fitted"]) + ") ("
                       + ",".join(f"{x:+.3g}" for x in rr["true"]) + ")")
            L.append(f"| {v['kind']} | {v['rule']} | {v['r']} | {v['eps']:g} | {v['errT']:.2e} | {v['rank']} | {near} | {rec} |")
    return "\n".join(L) + "\n"


# =============================================================================================
# self-test
# =============================================================================================


def selftest():
    rng = np.random.default_rng(0)
    ok = []

    def check(name, cond, detail=""):
        ok.append((name, bool(cond)))
        print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    G, _ = make_G(1024, 16, rng)
    # NS matrix iteration == spectral fast path in float64
    a, b = newton_schulz(G, QUINTIC_MUON, 5), ns_spectral(G, QUINTIC_MUON, 5)
    check("NS matrix == NS spectral (f64)", rel_err(a, b)[1] < 1e-10, f"err={rel_err(a, b)[1]:.2e}")
    a = newton_schulz(G, [(1.5, -0.5)] * 3 + [QUINTIC_MUON] * 2)
    b = ns_spectral(G, [(1.5, -0.5)] * 3 + [QUINTIC_MUON] * 2)
    check("NS per-step coefficient list", rel_err(a, b)[1] < 1e-10, f"err={rel_err(a, b)[1]:.2e}")
    Gw, _ = make_G(1024, 8, rng)
    a = ns_spectral(Gw, CUBIC, 60, prenorm="spec")
    check("cubic NS (spec prenorm, 60 steps) -> polar", rel_err(a, polar(Gw))[1] < 1e-8,
          f"err={rel_err(a, polar(Gw))[1]:.2e}")
    e32 = rel_err(newton_schulz(G, QUINTIC_MUON, 5, "float32"), ns_spectral(G))[1]
    e16 = rel_err(newton_schulz(G, QUINTIC_MUON, 5, "bf16"), ns_spectral(G))[1]
    check("precision: f32 err ~1e-7, bf16 err ~1e-3..1e-2", e32 < 1e-5 and 1e-4 < e16 < 5e-2,
          f"f32={e32:.2e} bf16={e16:.2e}")
    x = np.array([1.0, 1.00390625, 1.001953125, 3.14159265, -2.5e-3], np.float64)
    x = np.append(x, 1.0 + 3 * 2.0 ** -9)  # 1+2^-8 is a tie -> even (1.0); 1+3*2^-9 rounds up
    check("bf16_round", np.allclose(bf16_round(x), [1.0, 1.0, 1.0, 3.140625, -0.0025024414, 1.0078125]),
          str(bf16_round(x)))
    check("cpolar(0,0,0,0) == polar", rel_err(conditioned_polar(G), polar(G))[1] < 1e-10)
    check("diag_precond(-1/2,0) == row_norm", rel_err(diag_precond(G, -0.5, 0), row_norm(G))[1] < 1e-12)
    check("diag_precond(0,-1/2) == col_norm", rel_err(diag_precond(G, 0, -0.5), col_norm(G))[1] < 1e-12)
    g1, _ = make_G(1024, 1, rng)
    check("rank 1: polar == frob == col_norm", max(rel_err(polar(g1), g1)[1], rel_err(col_norm(g1), g1)[1]) < 1e-12)
    check("rank 1: row_norm == sign", rel_err(row_norm(g1), sign(g1))[1] < 1e-12)
    check("rank 1: cpolar(a,b,c,d) == G*|G|^(2(a+b))",
          rel_err(conditioned_polar(g1, 0.2, -0.45, 0.3, -0.7), g1 * np.abs(g1) ** (2 * (0.2 - 0.45)))[1] < 1e-10)
    print("  quintic NS (5 steps) transfer f(x) at x=s/||G||_F:",
          {float(v): round(float(ns_transfer(v)), 4) for v in (0.01, 0.05, 0.1, 0.25, 0.5, 0.7, 1.0)})

    # diagnostics on clean pairs
    for nm, fn in (("polar", polar), ("sign", sign), ("cpolar_rowwhite", lambda g: conditioned_polar(g, b=-0.5)),
                   ("cpolar(.3,-.4,-.2,.1)", lambda g: conditioned_polar(g, 0.3, -0.4, -0.2, 0.1))):
        dg = diagnostics(G, -1e-3 * fn(G))
        print(f"  diag[{nm:22s}] outside_colG={dg['outside_colG_frac']:.2e} offdiag={dg['transfer_offdiag_frac']:.2e} "
              f"loglog_slope={dg['transfer_loglog_slope']:+.3f} lscan_q={dg.get('lscan_q_best', float('nan')):+.4f} "
              f"(oos {dg.get('lscan_out_of_span_at_best', float('nan')):.1e}) S_asym={dg['S_asym']:.2e} "
              f"row_slope={dg['row_slope']:+.3f} sv_D_rel_min={dg['sv_D_rel'][-1]:.3f}")
    dg = diagnostics(G, -1e-3 * conditioned_polar(G, 0.3, -0.4, -0.2, 0.1))
    check("left scan recovers a+b=-0.1", abs(dg["lscan_q_best"] + 0.1) < 1e-3, f"q={dg['lscan_q_best']:+.5f}")
    D16 = -1e-3 * bf16_round(ns_spectral(G))
    check("bf16 lattice test", bf16_lattice(D16, -1e-3) < 1e-6 < bf16_lattice(-1e-3 * ns_spectral(G), -1e-3),
          f"{bf16_lattice(D16, -1e-3):.1e} vs {bf16_lattice(-1e-3*ns_spectral(G), -1e-3):.1e}")
    # controlled column-scaling response
    k = np.logspace(0, -2, 16)
    cr = {nm: column_response(fn(G), fn(G * k), k) for nm, fn in (("col_norm", col_norm), ("frob", frob), ("polar", polar))}
    print("  column_response (G -> G diag(k)):", {k_: {a: round(b, 4) for a, b in v.items()} for k_, v in cr.items()})
    check("column_response separates col_norm(q=0) / frob(q=1) / polar(resid>0)",
          abs(cr["col_norm"]["q"]) < 1e-8 and abs(cr["frob"]["q"] - 1) < 1e-8 and cr["polar"]["resid_after_colscale"] > 1e-2)

    # harness on synthetic data: ranking + recovery at several noise levels
    rules = hidden_rules()
    t0 = time.time()
    for rule, r, eps in (("ns_quintic5", 16, 1e-3), ("cpolar(.3,-.4,-.2,.1)", 16, 1e-3),
                         ("cpolar_dshampoo", 4, 1e-2), ("spec_p0.5", 8, 1e-6), ("ns_quintic5_bf16", 16, 1e-3),
                         ("polar", 1, 1e-3)):
        mods = make_modules(rules[rule], r, 12, np.random.default_rng(zlib.crc32(rule.encode()) + r), eps=eps)
        rows = score(mods, fixed_candidates() + family_candidates(), renorm_modes=RENORM_MODES)
        print_table(rows[:8], f"hidden={rule} r={r} eps={eps:g} (top 8 of {len(rows)})")
        fixed = [x["name"] for x in rows if not x["name"].startswith("FIT")]
        errT = next((x["test_err"] for x in rows if x["name"] == rule), float("nan"))
        fixed_err = {x["name"]: x["test_err"] for x in rows if not x["name"].startswith("FIT")}
        if rule not in fixed:
            pass
        elif r > 1:
            better = [n for n, e in fixed_err.items() if e < errT / 1.01]
            ties = [n for n, e in fixed_err.items() if n != rule and e < 1.5 * errT]
            check(f"{rule} r={r} eps={eps:g}: no fixed candidate beats the true rule by >1%", not better,
                  f"order={fixed[:3]} confusable(<1.5x)={ties}")
        else:
            ties = [x["name"] for x in rows if not x["name"].startswith("FIT") and x["test_err"] < 1.5 * errT]
            print(f"  rank-1 ties with polar (err ratio<1.5): {ties}")
        if rule in FAMILY_TRUTH and r > 1:
            fam, tru = FAMILY_TRUTH[rule]
            th = list(next(x for x in rows if x["name"] == fam)["theta"].values())
            dev = max(abs(a - b) for a, b in zip(th, tru))
            check(f"{rule}: {fam} recovers params", dev < (0.02 if eps <= 1e-3 else 0.1),
                  f"fit={np.round(th, 4).tolist()} true={list(tru)} maxdev={dev:.1e}")
    print(f"  harness runs: {time.time()-t0:.0f}s")
    # norm scaling across shapes for polar with lr fixed
    mods = [Module(G_, -1e-3 * polar(G_)) for G_ in (make_G(d, r, rng)[0] for d in (1024, 4096, 8192) for r in (1, 4, 16, 64))]
    ns_ = norm_scaling(mods)
    check("norm_scaling: polar fro ~ r^0.5, no d_out / |G| dependence",
          abs(ns_["fro"]["log_r"] - 0.5) < 1e-6 and abs(ns_["fro"]["log_d_out"]) < 1e-6, json.dumps(ns_))
    n_fail = sum(1 for _, v in ok if not v)
    print(f"\nSELFTEST: {len(ok) - n_fail}/{len(ok)} checks passed")
    return n_fail == 0


# =============================================================================================
# CLI
# =============================================================================================


def analyze(path, seed=0):
    mods = load_modules(path)
    print(f"loaded {len(mods)} modules from {path}")
    for m in mods[:5]:
        print_diagnostics(diagnostics(m.G, m.D, m.K), m.name)
    print("\nnorm scaling:", json.dumps(norm_scaling(mods), indent=1))
    hasK = all(m.K is not None for m in mods)
    rows = score(mods, fixed_candidates(True, hasK) + family_candidates(hasK), seed=seed,
                 renorm_modes=RENORM_MODES)
    print_table(rows, f"{path}: {len(mods)} modules, held-out split seed {seed}")
    for r in rows[:5]:
        print(f"  {r['name']}: global err by renorm {json.dumps({k: round(v, 5) for k, v in r['global_err'].items()})}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest")
    p = sp.add_parser("identifiability")
    p.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parent.parent / "runs" / "candidates_identifiability"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--quick", action="store_true", help="ranks 1,4,16 and parametric only")
    p = sp.add_parser("analyze")
    p.add_argument("path")
    p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    if a.cmd == "identifiability":
        if a.quick:
            _, _, rep = identifiability(a.out, a.workers, ranks=(1, 4, 16), kinds=("parametric",),
                                        verify_ranks=(4,), verify_kinds=("parametric",))
        else:
            _, _, rep = identifiability(a.out, a.workers)
        print(rep)
    if a.cmd == "analyze":
        analyze(a.path, a.seed)


if __name__ == "__main__":
    main()
