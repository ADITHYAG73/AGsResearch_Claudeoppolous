"""fs_noise.py -- noise-floor estimate: how far do fp16-arithmetic emulations deviate from the
deterministic part (X0 rounded to fp16, exact arithmetic after)? Compares data, our pe5_f16
emulation, and data-vs-emulation, per rank, plus the diag/offdiag/out-of-span split."""
import sys, json, numpy as np
sys.path.insert(0, '/home/user/AGsResearch_Claudeoppolous/dimuon/pipeline')
from fs_load import load, svds, RANKS, CACHE
from fs_sim import rank_scale
from fs_spec import PE_MEM, safety
from fs_eval import _nsraw
from fs_ns16 import x0_f16, ns16
PE5 = safety(PE_MEM[:5], 1.01)
fin = lambda X, r: -rank_scale(r, 1e-3) * X / np.linalg.norm(X)
out = {}
for r in RANKS:
    mods = load(r); sv = svds(r, mods); acc = []
    for (n, G, U), (P, S, Q) in list(zip(mods, sv))[::3]:
        Ui = fin(_nsraw(x0_f16(G).astype(np.float64), PE5), r); Us = fin(ns16(G, PE5), r); nU = np.linalg.norm(U)
        row = []
        for V in (U, Us):
            E = V - Ui; M = P.T @ E @ Q
            row += [np.linalg.norm(E) / nU, np.linalg.norm(np.diag(M)) / nU, np.linalg.norm(M - np.diag(np.diag(M))) / nU, np.linalg.norm(E - P @ (P.T @ E)) / nU]
        row.append(np.linalg.norm(U - Us) / nU); acc.append(row)
    m = np.median(np.array(acc), axis=0)
    out[r] = dict(data_vs_x0h=m[0], data_diag=m[1], data_off=m[2], data_perp=m[3], sim_vs_x0h=m[4], sim_diag=m[5], sim_off=m[6], sim_perp=m[7], data_vs_sim=m[8])
    print('r=%-3d data-vs-x0h %.2e (diag %.2e off %.2e perp %.2e) | sim-vs-x0h %.2e (diag %.2e off %.2e perp %.2e) | data-vs-sim %.2e' % (r, *m))
json.dump({str(k): {a: float(b) for a, b in v.items()} for k, v in out.items()}, open(CACHE / 'noise_floor.json', 'w'), indent=1)
