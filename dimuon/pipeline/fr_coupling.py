"""fr_coupling.py -- FRESH-01 Q1b: does a fresh-state update of one factor depend on the other
factor (or on itself) beyond PE5 of its own gradient?

All candidates use the FIRSTSTEP-01 fp16 PE5 emulation and the explicit final norm, so the only
thing that changes between candidates is the matrix that is orthogonalised (or a post-multiplier).
K_A = A A^T (r x r), K_B = B^T B (r x r).  Exponent p is picked on EVEN module indices and the
residual reported on ODD ones (held out).
  A-side: A1  PE(K_B^p G_A)       A2  K_B^p PE(G_A)        A3  PE(K_A^p G_A)
  B-side: B1  PE(G_B K_A^p)       B2  PE(G_B) K_A^p        B3  PE(G_B K_B^p)
  joint : J(w) = PE([G_B^T , w G_A]) split into its two blocks, each renormalised
          (one shared r x r Gram => coupling through the factor dimension)
Conditioning of K_A, K_B is reported per step (coupling is only identifiable if they are far
from multiples of I).  -> runs/FR/coupling.json, coupling.txt
"""
import json, sys
import numpy as np
from fr_core import *  # noqa

GOOD = ["S1_r8_fresh_ck1_t0/step", "S1_r8_fresh_ck2_t0/step", "S1_r8_fresh_ck2_t1/step",
        "S1_r8_fresh_ck5_t0/step", "S1_r8_fresh_ck5_t1/step", "S3x_r8_fresh_ck3/step4",
        "S1v2_r32_fresh_ck2_t0/step", "S1v2_r32_fresh_ck5_t0/step", "S1v2_r32_fresh_ck5_t1/step"]
PS = [-1.0, -0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
WS = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]


def spd(K, p):
    w, V = np.linalg.eigh(K)
    w = np.maximum(w, w.max() * 1e-12)
    return (V * w ** p) @ V.T


def cond(K):
    w = np.linalg.eigvalsh(K)
    return float(w.max() / w.min())


def run(step_id, stride):
    r, W0, W1, G = step_io(step_id)
    lr, rank = r["params"]["learning_rate"], r["rank"]
    fin = lambda X: finalise(X, rank, lr)
    res = {k: {p: [] for p in PS} for k in ("A1", "A2", "A3", "B1", "B2", "B3")}
    res["J"] = {w: [] for w in WS}
    idx, cA, cB = [], [], []
    for i, m in enumerate(modules(W0)):
        if i % stride: continue
        A, B = W0["A/" + m], W0["B/" + mB(m)]
        gA, gB = G["GA/" + m], G["GB/" + mB(m)]
        dA, dB = W1["A/" + m] - A, W1["B/" + mB(m)] - B
        KA, KB = A @ A.T, B.T @ B
        cA.append(cond(KA)); cB.append(cond(KB)); idx.append(i)
        PA, PB = pe5_f16(gA), pe5_f16(gB)
        for p in PS:
            KBp, KAp = spd(KB, p), spd(KA, p)
            res["A1"][p].append(relF(fin(pe5_f16(KBp @ gA)), dA))
            res["A2"][p].append(relF(fin(KBp @ PA), dA))
            res["A3"][p].append(relF(fin(pe5_f16(KAp @ gA)), dA))
            res["B1"][p].append(relF(fin(pe5_f16(gB @ KAp)), dB))
            res["B2"][p].append(relF(fin(PB @ KAp), dB))
            res["B3"][p].append(relF(fin(pe5_f16(gB @ KBp)), dB))
        for w in WS:
            X = pe5_f16(np.concatenate([gB.T, w * gA], axis=1))  # r x (d_out + d_in)
            Xb, Xa = X[:, :gB.shape[0]].T, X[:, gB.shape[0]:]
            res["J"][w].append((relF(fin(Xa), dA), relF(fin(Xb), dB)))
    return r, np.array(idx), dict(condKA=summ(cA), condKB=summ(cB)), res


if __name__ == "__main__":
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    out = {}
    lines = []
    def P(s=""):
        print(s, flush=True); lines.append(s)
    for s in GOOD:
        rr = stride * (2 if "r32" in s else 1)
        r, idx, conds, res = run(s, rr)
        tr, te = (idx // rr) % 2 == 0, (idx // rr) % 2 == 1
        P(f"\n{s} rank {r['rank']}  cond(AA^T) {fmt(conds['condKA'])}  cond(B^TB) {fmt(conds['condKB'])}")
        o = dict(conds=conds)
        for k in ("A1", "A2", "A3", "B1", "B2", "B3"):
            trm = {p: float(np.median(np.array(v)[tr])) for p, v in res[k].items()}
            pbest = min(trm, key=trm.get)
            te0 = np.array(res[k][0.0])[te]; teb = np.array(res[k][pbest])[te]
            o[k] = dict(train_median=trm, p_best=pbest, test_p0=summ(te0), test_pbest=summ(teb))
            P(f"  {k}: p*={pbest:+.2f}  held-out  p=0 [{fmt(summ(te0))}]  p* [{fmt(summ(teb))}]"
              f"   train curve " + " ".join(f"{p:+.2f}:{trm[p]:.4f}" for p in PS))
        for j, F in ((0, "A"), (1, "B")):
            trm = {w: float(np.median(np.array(v)[tr, j])) for w, v in res["J"].items()}
            wb = min(trm, key=trm.get)
            o["J" + F] = dict(train_median=trm, w_best=wb,
                              test_wbest=summ(np.array(res["J"][wb])[te, j]))
            P(f"  joint->{F}: w*={wb}  held-out [{fmt(summ(np.array(res['J'][wb])[te, j]))}]   train "
              + " ".join(f"{w}:{trm[w]:.4f}" for w in WS))
        out[s] = o
    json.dump(out, open(OUT / "coupling.json", "w"), indent=1)
    open(OUT / "coupling.txt", "w").write("\n".join(lines) + "\n")
