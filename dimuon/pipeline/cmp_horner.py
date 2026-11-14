"""cmp_horner.py -- COMPARE-01: test public orthogonaliser descriptions against D1 (step 1, B=0, ranks 1..64).

Public descriptions tested (all fp16 storage, fp32 accumulate, X0 = fp16(G32/(1.01*||G32||)) ):
  ours      FIRSTSTEP-01 emulation: wide (r x d) X, A = XX^T, B = fp16(bA + cAA), X = fp16(aX + BX)
            (= anpaure/dimuon.py `_polar_express` and the Polar Express paper's reference code, except
            that the paper runs it in bf16)
  horner    romeerp/dimuon-replay `replay.py::horner_polar` (CPU path): tall X, R = fp16(X^T X),
            inner = fp16(fp16(cR) + bI), P = fp16(fp16(R@inner) + aI), X = fp16(X@P)
  horner_nocR  same without the separate fp16 store of cR (romeerp's rejected variant)
  horner_sK{k} horner with the long (d-length) Gram reduction split into k chunks whose partial sums
            are rounded to fp16 (romeerp's split-K explanation of GPU paths; our emulation of it)
Metrics per rank (every `stride`-th module): median relF(U_data, U_sim); deviation from the
deterministic part (X0 fp16, exact after) for data and for each sim (FIRSTSTEP-01 item 7); and the
fraction of entries of the recovered fp16 matrix behind U_data that equal the sim's fp16 matrix.
Output: runs/COMPARE/horner.{txt,json}
"""
import sys, json, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fs_load import load, RANKS
from fs_sim import rank_scale
from fs_spec import PE_MEM, safety
from fs_eval import _nsraw
from fs_ns16 import x0_f16, ns16
from fs_grid import recover

PE5 = safety(PE_MEM[:5], 1.01)
f32 = np.float32
OUT = pathlib.Path(__file__).resolve().parents[1] / "runs" / "COMPARE"
OUT.mkdir(parents=True, exist_ok=True)


def H(x):
    return np.asarray(x, f32).astype(np.float16).astype(f32)


def gram_tall(X, splitk=1):
    """R = X^T X for tall X (d x r), fp32 accumulate, fp16 out; optional split-K with fp16 partials."""
    if splitk == 1:
        return H(X.T @ X)
    parts = np.array_split(np.arange(X.shape[0]), splitk)
    acc = np.zeros((X.shape[1], X.shape[1]), f32)
    for p in parts:
        acc = acc + H(X[p].T @ X[p])
    return H(acc)


def horner(G, coefs=PE5, cR=True, splitk=1):
    X = x0_f16(G)                      # tall d x r, fp16 values in fp32
    r = X.shape[1]
    I = np.eye(r, dtype=f32)
    for a, b, c in coefs:
        R = gram_tall(X, splitk)
        cRv = H(f32(c) * R) if cR else f32(c) * R
        inner = H(cRv + f32(b) * I)
        P = H(H(R @ inner) + f32(a) * I)
        X = H(X @ P)
    return X.astype(np.float64)


def fin(X, r):
    X32 = np.asarray(X, f32)
    n = np.sqrt(np.sum(X32 * X32, dtype=f32))
    return (-(f32(rank_scale(r, 1e-3)) / n) * X32).astype(np.float64)


SIMS = {
    "ours": lambda G: ns16(G, PE5),
    "horner": lambda G: horner(G),
    "horner_nocR": lambda G: horner(G, cR=False),
    "horner_sK4": lambda G: horner(G, splitk=4),
    "horner_sK16": lambda G: horner(G, splitk=16),
}


def main(stride=6, ranks=RANKS):
    res = {}
    lines = []
    for r in ranks:
        mods = load(r)[::stride]
        acc = {k: dict(relF=[], dev=[], bits=[]) for k in SIMS}
        acc["data"] = dict(dev=[])
        for n, G, U in mods:
            Ui = fin(_nsraw(x0_f16(G).astype(np.float64), PE5), r)
            nU = np.linalg.norm(U)
            acc["data"]["dev"].append(np.linalg.norm(U - Ui) / nU)
            for k, f in SIMS.items():
                X = f(G)
                Us = fin(X, r)
                acc[k]["relF"].append(np.linalg.norm(Us - U) / nU)
                acc[k]["dev"].append(np.linalg.norm(Us - Ui) / nU)
                Xd, c, _, ongrid = recover(U, Xref=X)
                acc[k]["bits"].append(float(np.mean(Xd == X)))
        row = {k: {m: float(np.median(v)) for m, v in d.items()} for k, d in acc.items()}
        row["n_modules"] = len(mods)
        res[r] = row
        s = f"r={r:<3d} n={len(mods):3d} data dev-from-x0h {row['data']['dev']:.2e} | " + " | ".join(
            f"{k}: relF {row[k]['relF']:.2e} dev {row[k]['dev']:.2e} bits {row[k]['bits']:.2f}" for k in SIMS)
        print(s, flush=True)
        lines.append(s)
    (OUT / "horner.txt").write_text("\n".join(lines) + "\n")
    json.dump({str(k): v for k, v in res.items()}, open(OUT / "horner.json", "w"), indent=1)


if __name__ == "__main__":
    st = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rk = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else RANKS
    main(st, rk)
