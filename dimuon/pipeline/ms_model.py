"""ms_model.py -- MULTISTEP-01 candidate multi-step Dimuon rule and chain simulator.

Rule (tall orientation n x r; lora_B as stored, lora_A transposed):
    m_t   = momentum(m_{t-1}, g_t; beta)                   (form selectable)
    S     = state statistic matrix from the PREVIOUS step   (selectable: prev momentum / prev grad)
    l_i   = sum_j S_ij^2 + epsL ,   c_j = sum_i S_ij^2 + epsR     (row / column energies of S)
    Dl    = l^-1/4 , Dr = c^-1/4      (if S == 0 everywhere: Dl = Dr = 1, i.e. fresh state)
    X     = PE5( Dl m_t Dr )                                 (Polar Express, safety 1.01)
    U_t   = - lr sqrt(r) (r/32)^0.175 * (Dl X Dr) / ||Dl X Dr||_F
eps forms: 'abs' (eps is a constant), 'relmean' (eps * mean of the energies), 'relmax'.
"""
import numpy as np
from ms_common import pe


def energies(S):
    S2 = S * S
    return S2.sum(1), S2.sum(0)


def add_eps(v, eps, form):
    if form == "abs":
        return v + eps
    if form == "relmean":
        return v + eps * v.mean()
    if form == "relmax":
        return v + eps * v.max()
    if form == "clampmax":
        return np.maximum(v, eps * v.max())
    if form == "relmax_sqrt":   # eps added to the root-energy: (sqrt(v) + eps*max sqrt(v))^2
        return (np.sqrt(v) + eps * np.sqrt(v.max())) ** 2
    raise ValueError(form)


def direction(m, S, epsL=0.0, epsR=0.0, form="relmax", pL=-0.25, pR=-0.25, x0prec="f64"):
    """unit-norm predicted update direction (sign included: -)."""
    if S is None or not np.any(S):
        X = pe(m, x0prec=x0prec)
        return -X / np.linalg.norm(X) if np.any(X) else X
    l, c = energies(S)
    l, c = add_eps(l, epsL, form), add_eps(c, epsR, form)
    Dl = np.where(l > 0, np.maximum(l, 1e-300) ** pL, 0.0)
    Dr = np.where(c > 0, np.maximum(c, 1e-300) ** pR, 0.0)
    X = Dl[:, None] * pe(Dl[:, None] * m * Dr[None, :], x0prec=x0prec) * Dr[None, :]
    n = np.linalg.norm(X)
    return -X / n if n > 0 else X


def mom_update(m, g, beta, form):
    """returns (new stored momentum, vector fed to the orthogonaliser)."""
    if m is None:
        m = np.zeros_like(g)
    if form == "ema":
        m = beta * m + (1 - beta) * g
        return m, m
    if form == "sum":
        m = beta * m + g
        return m, m
    if form == "nest_ema":
        m = beta * m + (1 - beta) * g
        return m, beta * m + (1 - beta) * g
    if form == "nest_sum":
        m = beta * m + g
        return m, g + beta * m
    raise ValueError(form)
