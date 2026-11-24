"""fr_lr2.py -- FRESH-01 Q3: what direction is the lr=1e-2 deviation in? (rank 32, R2, step 3 from
state 2, T1, restored with optimizer).  E = D(1e-2)/10 - D(1e-3).  If lr enters the INPUT of an
orthogonaliser (e.g. Y + f(lr) Z), E ~ T_Q(Z), the first-order polar perturbation at
Q = polar(D(1e-3)).  Candidates Z: current gradient g3, previous updates U1 = W1-W0, U2 = W2-W1 of
the trunk, the weight itself W (weight-decay-like), the other factor's update; nulls: random.
Same test on the clip difference is in fr_clip2.py.  -> runs/FR/lr2.json, lr2.txt"""
import json
import numpy as np
from fr_core import *  # noqa
from fr_clip2 import tangent, cos

if __name__ == "__main__":
    S = {1e-3: "S6_r32_ck2_lr0.001/step3", 1e-2: "S6_r32_ck2_lr0.01/step3"}
    r0 = rec(S[1e-3]); W2 = state(r0["before"]); G = grad(r0["grad_key"])
    trunk = man()["steps"]
    s1, s2 = trunk["S2_r32_b0.9/step1"], trunk["S2_r32_b0.9/step2"]
    assert s2["after"] == r0["before"], (s2["after"], r0["before"])
    W0s, W1s = state(s1["before"]), state(s1["after"])
    Wl = {lr: state(rec(s)["after"]) for lr, s in S.items()}
    g2key = s2.get("grad_key"); G2 = grad(g2key) if g2key else None
    rng = np.random.default_rng(0)
    rows = []
    for i, m in enumerate(modules(W2)):
        if i % 2: continue
        for F, key in (("A", "A/" + m), ("B", "B/" + mB(m))):
            w2 = W2[key]; D3 = Wl[1e-3][key] - w2; D2 = (Wl[1e-2][key] - w2) * 0.1
            E = D2 - D3; Q = polar(D3)
            U1 = W1s[key] - W0s[key]; U2 = w2 - W1s[key]
            Z = dict(g3=G["G" + key], U1=U1, U2=U2, W=w2, D3=D3, rand=rng.standard_normal(w2.shape))
            if G2 is not None: Z["g2"] = G2["G" + key]
            row = dict(m=m, F=F, relE=float(np.linalg.norm(E) / np.linalg.norm(D3)))
            for n, z in Z.items():
                if not np.any(z): row["t_" + n] = None; continue
                row["t_" + n] = cos(E, tangent(Q, z))
                row["raw_" + n] = cos(E, z)
            # best linear combination of tangent directions (least squares, how much of E they span)
            T = [tangent(Q, z).ravel() for n, z in Z.items() if n not in ("rand", "D3") and np.any(z)]
            Tm = np.stack(T, 1); coef, *_ = np.linalg.lstsq(Tm, E.ravel(), rcond=None)
            row["lstsq_explained"] = float(1 - np.linalg.norm(E.ravel() - Tm @ coef) ** 2 / np.linalg.norm(E) ** 2)
            rows.append(row)
    lines = []
    keys = [k for k in rows[0] if k.startswith("t_") or k.startswith("raw_") or k == "lstsq_explained"]
    for F in "AB":
        R = [x for x in rows if x["F"] == F]
        for k in keys:
            v = [x[k] for x in R if x.get(k) is not None]
            if v:
                l = f"{F} cos(E, {k:8s}) [{fmt(summ(v))}]"; print(l); lines.append(l)
    json.dump(rows, open(OUT / "lr2.json", "w"), indent=0)
    open(OUT / "lr2.txt", "w").write("\n".join(lines) + "\n")
