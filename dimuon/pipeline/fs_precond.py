"""fs_precond.py -- FIRSTSTEP-01 item 1b: does any diagonal row/column scaling of G, or right
preconditioning by K = A A^T (normalised to trace r), improve on the plain spectral rule?
Model: U ~ PE5( L^b G R^c K^e ) with X0 rounded to fp16, exact arithmetic after (pe5_x0h)."""
import sys, numpy as np
sys.path.insert(0, '/home/user/AGsResearch_Claudeoppolous/dimuon/pipeline')
from fs_load import load, D1
from fs_sim import rank_scale
from fs_spec import PE_MEM, safety
from fs_eval import _nsraw
from fs_ns16 import x0_f16
PE5 = safety(PE_MEM[:5], 1.01)
def spdpow(K, e):
    w, V = np.linalg.eigh(K); return (V * w ** e) @ V.T
def model(G, r, e=0., b=0., c=0., K=None):
    M = G.copy()
    if e: M = M @ spdpow(K, e)
    if b: M = M * (np.linalg.norm(G, axis=1) ** b)[:, None]
    if c: M = M * (np.linalg.norm(G, axis=0) ** c)[None, :]
    X = _nsraw(x0_f16(M).astype(np.float64), PE5); return -rank_scale(r, 1e-3) * X / np.linalg.norm(X)
for r in [8, 32]:
    mods = load(r)[::6]; zo = np.load(D1 / f'r{r}_oracle.npz')
    Ks = []
    for n, G, U in mods:
        A = zo['A0/' + n.replace('lora_B', 'lora_A')]; K = A @ A.T; Ks.append(K / np.trace(K) * r)
    for lab, kw in [('none', {})] + [(f'K^{e:+.2f}', dict(e=e)) for e in (-0.05, -0.02, -0.01, 0.01, 0.02, 0.05)] + \
            [(f'row^{v:+.2f}', dict(b=v)) for v in (-0.02, -0.01, 0.01, 0.02)] + [(f'col^{v:+.2f}', dict(c=v)) for v in (-0.02, -0.01, 0.01, 0.02)]:
        rel = [np.linalg.norm(model(G, r, K=K, **kw) - U) / np.linalg.norm(U) for (n, G, U), K in zip(mods, Ks)]
        print('r=%d %-10s relF med %.3e' % (r, lab, np.median(rel)))
