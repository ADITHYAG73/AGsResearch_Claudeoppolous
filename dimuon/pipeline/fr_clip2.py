"""fr_clip2.py -- FRESH-01 Q2, model-light test of HOW the clipped gradient enters the step-2 B update.

Hypothesis H_add: the step-2 lora_B update is D(c) = N(orth(Y + c*k*g2)) where Y is stored state
(unaffected by clipping), c = clip/N1 the GLOBAL clip factor, orth ~ polar.  Then
  (i)  D(c) is close to linear in c when the g2 term is small against Y:
       |D(1)-D(.5)| : |D(.5)-D(.01)| ~ 0.5 : 0.49, and the two differences are parallel;
  (ii) Delta = D(1) - D(.01) lies along the tangent of the polar map at Q = polar(D(.01)) in the
       direction g2:  T(g2) = (I-QQ^T) g2 + Q skew(Q^T g2)   (first-order polar perturbation,
       exact for orthogonal Q with equal singular values).
Nulls: T(g1) (step-1 gradient, cos(g1,g2) ~ 0.98 so this is a hard null), T(random), raw g2.
-> runs/FR/clip2.json, clip2.txt"""
import json
import numpy as np
from fr_core import *  # noqa

STEPS = {1.0: "S1_r8/step2", 0.5: "S5_step2_c0.5/step2", 0.01: "S5_step2_c0.01/step2"}


def cos(a, b):
    return float(np.sum(a * b) / np.linalg.norm(a) / np.linalg.norm(b))


def tangent(Q, Z):
    S = Q.T @ Z
    return Z - Q @ (Q.T @ Z) + Q @ ((S - S.T) / 2)


if __name__ == "__main__":
    r0 = rec(STEPS[1.0]); W0 = state(r0["before"])
    g2s = grad(r0["grad_key"]); g1s = grad(rec(r0["hist"][0])["grad_key"])
    W1 = {c: state(rec(s)["after"]) for c, s in STEPS.items()}
    rng = np.random.default_rng(0)
    rows = []
    for m in modules(W0):
        k = "B/" + mB(m)
        D = {c: W1[c][k] - W0[k] for c in STEPS}
        g2, g1 = g2s["G" + k], g1s["G" + k]
        d1, d2 = D[1.0] - D[0.5], D[0.5] - D[0.01]
        Q = polar(D[0.01]); Dl = D[1.0] - D[0.01]
        rows.append(dict(m=m, ratio=float(np.linalg.norm(d1) / np.linalg.norm(d2)), cos_d1d2=cos(d1, d2),
                         rel_1_001=relF(D[0.01], D[1.0]),
                         t_g2=cos(Dl, tangent(Q, g2)), t_g1=cos(Dl, tangent(Q, g1)),
                         t_g2perp=cos(Dl, tangent(Q, g2 - g1 * np.sum(g1 * g2) / np.sum(g1 * g1))),
                         t_rand=cos(Dl, tangent(Q, rng.standard_normal(g2.shape))), raw_g2=cos(Dl, g2)))
    lines = []
    for key, txt in (("ratio", "|D(1)-D(.5)| / |D(.5)-D(.01)|  (H_add, linear: 1.02)"),
                     ("cos_d1d2", "cos(D(1)-D(.5), D(.5)-D(.01))  (H_add: ~1)"),
                     ("t_g2", "cos(D(1)-D(.01), T_Q(g2))"), ("t_g1", "cos(D(1)-D(.01), T_Q(g1))  [null]"),
                     ("t_g2perp", "cos(D(1)-D(.01), T_Q(g2 minus its g1 component))"),
                     ("t_rand", "cos(D(1)-D(.01), T_Q(random))  [null]"),
                     ("raw_g2", "cos(D(1)-D(.01), g2) raw  [no tangent projection]")):
        l = f"{txt:62s} [{fmt(summ([x[key] for x in rows]))}]"
        print(l); lines.append(l)
    json.dump(rows, open(OUT / "clip2.json", "w"), indent=0)
    open(OUT / "clip2.txt", "w").write("\n".join(lines) + "\n")
