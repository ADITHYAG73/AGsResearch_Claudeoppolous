"""fs_load.py -- load D1 (G, U) pairs and cache float64 thin SVDs of G (FIRSTSTEP-01)."""
import numpy as np, pathlib, os
D1 = pathlib.Path(__file__).resolve().parent.parent / "runs" / "D1"
CACHE = pathlib.Path(__file__).resolve().parent.parent / "runs" / "FS_cache"
RANKS = [1, 2, 4, 8, 16, 32, 64]

def load(rank, dim="lr1e-3", oracle=None):
    """Return list of (name, G float64, U float64). dim: suffix of r{r}_dim_{dim}.npz."""
    oracle = oracle or f"r{rank}_oracle.npz"
    zo = np.load(D1 / oracle); zd = np.load(D1 / f"r{rank}_dim_{dim}.npz")
    out = []
    for k in zd.files:
        n = k[2:]
        out.append((n, zo["G/" + n].astype(np.float64), zd[k].astype(np.float64)))
    return out

def svds(rank, mods, tag=None):
    tag = tag or f"r{rank}"
    f = CACHE / f"svd_{tag}.npz"
    if f.exists():
        z = np.load(f); return [(z[f"P{i}"], z[f"S{i}"], z[f"Q{i}"]) for i in range(len(mods))]
    res = [np.linalg.svd(G, full_matrices=False) for _, G, _ in mods]
    np.savez(f, **{f"{c}{i}": a for i, r in enumerate(res) for c, a in zip("PSQ", (r[0], r[1], r[2].T))})
    return [(P, S, Vt.T) for P, S, Vt in res]
