"""fr_clip.py -- FRESH-01 Q2: what does grad_clip_norm do at step 2 (rank 8, R1, T0 twice)?

Data: state 1 = after one unclipped step from init (B1 = U1, A1 = A0).  Step 2 restored with
optimizer, clip in {2, 0.5, 0.01} x N1 (N1 = global oracle norm at state 1 = 232.04), plus the
unclipped continuing step (S1_r8/step2, after = 97ca..., bit-identical to clip 2x).
g1 = oracle G at init (T0), g2 = oracle G at state 1 (T0).

Model family (two-term, per tensor):  dB2 = finalise(PE5( g2/|g2| + alpha * Z/|Z| ))
with Z in {g1, U1 = B1 - B0}.  alpha is fitted per tensor per clip level by a 1-D scan.
If clipping multiplies the CURRENT gradient by c = clip/N1 before it is combined with state that
was stored unclipped, then alpha(c) = alpha(1)/c (tensor by tensor).  A global c (shared by all
tensors, A and B) and a per-tensor c differ: with per-tensor clipping at 0.5 N1 no tensor's own
norm exceeds the threshold, so nothing would change.
-> runs/FR/clip.json, clip.txt
"""
import json
import numpy as np
from fr_core import *  # noqa

N1 = 232.04
STEPS = {1.0: "S1_r8/step2", 2.0: "S5_step2_c2/step2", 0.5: "S5_step2_c0.5/step2",
         0.01: "S5_step2_c0.01/step2"}
ALPHAS = np.concatenate([[0.0], np.logspace(-3, 3, 49)])
PE = pe5_f16


def fit_alpha(g2n, Zn, D, rank, lr, fine=True):
    errs = [relF(finalise(PE(g2n + a * Zn), rank, lr), D) for a in ALPHAS]
    j = int(np.argmin(errs))
    if not fine or j == 0:
        return float(ALPHAS[j]), float(errs[j])
    lo, hi = np.log(ALPHAS[max(j - 1, 1)]), np.log(ALPHAS[min(j + 1, len(ALPHAS) - 1)])
    for _ in range(20):  # golden-section on log alpha
        m1, m2 = lo + (hi - lo) * 0.382, lo + (hi - lo) * 0.618
        e1 = relF(finalise(PE(g2n + np.exp(m1) * Zn), rank, lr), D)
        e2 = relF(finalise(PE(g2n + np.exp(m2) * Zn), rank, lr), D)
        if e1 < e2: hi = m2
        else: lo = m1
    a = float(np.exp((lo + hi) / 2))
    return a, relF(finalise(pe5_f16(g2n + a * Zn), rank, lr), D)


if __name__ == "__main__":
    rec0 = rec(STEPS[1.0])
    W0 = state(rec0["before"]); Wg1 = grad(rec(rec0["hist"][0])["grad_key"]); Wg2 = grad(rec0["grad_key"])
    Winit = state(rec(rec0["hist"][0])["before"])
    W1 = {c: state(rec(s)["after"]) for c, s in STEPS.items()}
    lr, rank = 1e-3, 8
    lines = []
    def P(s=""):
        print(s, flush=True); lines.append(s)
    # global and per-tensor gradient norms at state 1
    nA = [np.linalg.norm(Wg2["GA/" + m]) for m in modules(W0)]
    nB = [np.linalg.norm(Wg2["GB/" + mB(m)]) for m in modules(W0)]
    P(f"g2 global norm {np.sqrt(np.sum(np.square(nA)) + np.sum(np.square(nB))):.3f} (N1 = {N1}); "
      f"per-tensor |g2_B| max {max(nB):.2f}, |g2_A| max {max(nA):.3f}; clip 0.5N1 = {0.5 * N1:.1f}")
    out = {"per_tensor": []}
    # which tensors changed at each clip level, vs which exceed the threshold on their own
    for c, thr in ((0.5, 0.5 * N1), (0.01, 0.01 * N1)):
        chB = [m for m in modules(W0) if np.any(np.asarray(W1[c].z["B/" + mB(m)]) != np.asarray(W1[1.0].z["B/" + mB(m)]))]
        overB = [m for m, n in zip(modules(W0), nB) if n > thr]
        overA = [m for m, n in zip(modules(W0), nA) if n > thr]
        P(f"clip {c}: B tensors changed {len(chB)}/152; tensors whose own |g2| > threshold {thr:.2f}: "
          f"B {len(overB)}, A {len(overA)}; changed-but-not-over {len(set(chB) - set(overB))}")
        out[f"changed_{c}"] = dict(changed=len(chB), overB=len(overB), overA=len(overA),
                                   changed_not_over=len(set(chB) - set(overB)))
    for i, m in enumerate(modules(W0)):
        if i % 4: continue
        k = "B/" + mB(m)
        B0 = W0[k]; g2 = Wg2["GB/" + mB(m)]; g1 = Wg1["GB/" + mB(m)]
        U1 = B0 - Winit[k]
        g2n = g2 / np.linalg.norm(g2)
        row = dict(m=m)
        for c in (1.0, 0.5, 0.01):
            D = W1[c][k] - B0
            row[f"d_vs_unclipped_{c}"] = relF(W1[c][k] - B0, W1[1.0][k] - B0) if c != 1.0 else 0.0
            row[f"own_{c}"] = relF(finalise(pe5_f16(g2), rank, lr), D)
            for zn, Z in (("g1", g1), ("U1", U1)):
                a, e = fit_alpha(g2n, Z / np.linalg.norm(Z), D, rank, lr)
                row[f"{zn}_{c}"] = (a, e)
        out["per_tensor"].append(row)
    R = out["per_tensor"]
    for c in (1.0, 0.5, 0.01):
        P(f"\nclip c = {c}:  dB vs unclipped dB  [{fmt(summ([x[f'd_vs_unclipped_{c}'] for x in R]))}]" if c != 1 else "\nclip c = 1 (unclipped)")
        P(f"  baseline PE5(g2) own gradient      [{fmt(summ([x[f'own_{c}'] for x in R]))}]")
        for zn in ("g1", "U1"):
            al = np.array([x[f"{zn}_{c}"][0] for x in R]); er = [x[f"{zn}_{c}"][1] for x in R]
            P(f"  PE5(g2n + a {zn}n), a fitted per tensor: resid [{fmt(summ(er))}]  a median {np.median(al):.4g} "
              f"(p10 {np.quantile(al, .1):.4g}, p90 {np.quantile(al, .9):.4g})")
    for zn in ("g1", "U1"):
        for c in (0.5, 0.01):
            ratio = np.array([x[f"{zn}_{c}"][0] / x[f"{zn}_1.0"][0] for x in R if x[f"{zn}_1.0"][0] > 0])
            P(f"  {zn}: a(c={c})/a(1) median {np.median(ratio):.4g} p10 {np.quantile(ratio, .1):.4g} "
              f"p90 {np.quantile(ratio, .9):.4g}   [1/c = {1 / c:g} if clip scales g2 only]")
    # A: bit-identical check
    nd = 0
    for m in modules(W0):
        for c in (0.5, 0.01, 2.0):
            nd += int(np.sum(np.asarray(W1[c].z["A/" + m]) != np.asarray(W1[1.0].z["A/" + m])))
    P(f"\nA tensors: differing elements vs unclipped across clip 0.5/0.01/2: {nd}")
    json.dump(out, open(OUT / "clip.json", "w"), indent=1)
    open(OUT / "clip.txt", "w").write("\n".join(lines) + "\n")
