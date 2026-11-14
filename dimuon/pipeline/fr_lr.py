"""fr_lr.py -- FRESH-01 Q3: structure of the lr-nonproportionality at rank 32 (S6/S6b, R2, step 3
from state 2 = aab5..., batch T1, restored with optimizer).

Dn(lr) = (W1(lr) - W0) * 1e-3/lr ;  E = Dn(1e-2) - D(1e-3).  Per tensor:
  - relF(E), split into the part inside both singular subspaces of D(1e-3) (diag = singular-value
    change, offdiag = rotation within the r-dim span) and the part outside the span
  - fp16-grid test on D(1e-2) (FIRSTSTEP-01 item 1): if D(1e-2)/c is an fp16 matrix, the
    deviation was produced BEFORE/INSIDE the orthogonaliser, not at the W + lr*U addition
  - elementwise: corr(|E|, |W0|), corr(|E|, |D|), and the float32 rounding scale of W0+lr*U
  - per-tensor relF vs |W0|, |B0|/|A0|, d_out, layer index, ||G||
Also rank 8 (S6b_r8 R2) for contrast.  -> runs/FR/lr.json, lr.txt
"""
import json, re
import numpy as np
from fr_core import *  # noqa
from fs_grid import recover

SETS = {
    32: {1e-4: "S6_r32_ck2_lr0.0001/step3", 1e-3: "S6_r32_ck2_lr0.001/step3",
         3e-3: "S6b_r32_ck2_lr0.003/step3", 1e-2: "S6_r32_ck2_lr0.01/step3"},
    8: {1e-4: "S6b_r8_ck2_lr0.0001/step3", 1e-3: "S6b_r8/step3", 1e-2: "S6b_r8_ck2_lr0.01/step3"},
}


def decompose(E, D):
    U, s, Vt = np.linalg.svd(D, full_matrices=False)
    C = U.T @ E @ Vt.T
    inside = np.linalg.norm(C)
    diag = np.linalg.norm(np.diag(C))
    tot = np.linalg.norm(E)
    return dict(diag=diag / tot, offdiag=np.sqrt(max(inside ** 2 - diag ** 2, 0)) / tot,
                outside=np.sqrt(max(tot ** 2 - inside ** 2, 0)) / tot,
                sym=np.linalg.norm((C + C.T) / 2) / max(inside, 1e-300),
                diag_rel=(np.diag(C) / s))  # relative change of each singular value


def corr(x, y):
    x = x.ravel() - x.mean(); y = y.ravel() - y.mean()
    return float(x @ y / np.sqrt((x @ x) * (y @ y)))


if __name__ == "__main__":
    lines, out = [], {}
    def P(s=""):
        print(s, flush=True); lines.append(s)
    for rank, S in SETS.items():
        r0 = rec(S[1e-3]); W0 = state(r0["before"]); G = grad(r0["grad_key"])
        W1 = {lr: state(rec(s)["after"]) for lr, s in S.items()}
        rows = []
        for m in modules(W0):
            for F, key in (("A", "A/" + m), ("B", "B/" + mB(m))):
                w0 = W0[key]; D3 = W1[1e-3][key] - w0
                row = dict(m=m, F=F, layer=int(re.search(r"layers\.(\d+)", m).group(1)),
                           kind=m.split(".")[2] + "." + m.split(".")[3], shape=list(w0.shape),
                           w0_norm=float(np.linalg.norm(w0)), g_norm=float(np.linalg.norm(G["G" + key])))
                for lr in S:
                    if lr == 1e-3: continue
                    Dn = (W1[lr][key] - w0) * (1e-3 / lr)
                    row[f"rel_{lr:g}"] = relF(Dn, D3)
                lr = 1e-2
                if lr in S:
                    D2 = W1[lr][key] - w0
                    E = D2 * 0.1 - D3
                    dec = decompose(E, D3)
                    row.update({f"E_{k}": float(v) for k, v in dec.items() if k != "diag_rel"})
                    row["sv_rel_change"] = [float(x) for x in dec["diag_rel"]]
                    row["corr_absE_absW"] = corr(np.abs(E), np.abs(w0))
                    row["corr_absE_absD"] = corr(np.abs(E), np.abs(D3))
                    row["corr_E_W"] = corr(E, w0)
                    # fp16 grid: is D(1e-2) a scaled fp16 matrix?  and D(1e-3)?
                    for tag, DD in (("lr1e-2", D2), ("lr1e-3", D3)):
                        Xq, c, mx, frac = recover(DD.astype(np.float32).astype(np.float64))
                        row[f"grid_frac_{tag}"] = frac
                    # float32 rounding scale of W0 + lr*U relative to the update
                    ulp = np.spacing(np.abs(w0).astype(np.float32)).astype(np.float64)
                    row["f32_ulp_over_upd"] = float(np.median(ulp / np.maximum(np.abs(D2), 1e-30)))
                rows.append(row)
        out[rank] = rows
        P(f"\n=== rank {rank} (state {r0['before']}, T{r0['text']}, {r0['regime']}) ===")
        for F in "AB":
            R = [x for x in rows if x["F"] == F]
            for lr in S:
                if lr == 1e-3: continue
                P(f"  {F} lr {lr:g}: relF(U(lr)*1e-3/lr, U(1e-3)) [{fmt(summ([x[f'rel_{lr:g}'] for x in R]))}]")
            if 1e-2 in S:
                for k in ("E_diag", "E_offdiag", "E_outside", "E_sym", "corr_absE_absW", "corr_absE_absD",
                          "corr_E_W", "grid_frac_lr1e-2", "grid_frac_lr1e-3", "f32_ulp_over_upd"):
                    P(f"     {k:18s} [{fmt(summ([x[k] for x in R]))}]")
                sv = np.array([x["sv_rel_change"] for x in R])
                P(f"     singular-value rel. change by index (median over tensors), sorted large->small sv: "
                  + " ".join(f"{v:+.3f}" for v in np.median(sv, 0)))
                P(f"     |sv rel change| by index (median): " + " ".join(f"{v:.3f}" for v in np.median(np.abs(sv), 0)))
                rel = np.array([x["rel_0.01"] for x in R])
                for name in ("w0_norm", "g_norm", "layer"):
                    v = np.array([x[name] for x in R], float)
                    P(f"     corr(rel, {name}) = {corr(rel, v):+.3f}")
                from collections import defaultdict
                d = defaultdict(list)
                for x in R: d[x["kind"]].append(x["rel_0.01"])
                P("     by kind: " + ", ".join(f"{k} {np.median(v):.3f}" for k, v in sorted(d.items())))
                d = defaultdict(list)
                for x in R: d[x["shape"][0] if F == "B" else x["shape"][1]].append(x["rel_0.01"])
                P("     by d: " + ", ".join(f"{k} {np.median(v):.3f}" for k, v in sorted(d.items())))
    json.dump(out, open(OUT / "lr.json", "w"), indent=0)
    open(OUT / "lr.txt", "w").write("\n".join(lines) + "\n")
