"""fr_figs.py -- FRESH-01 figures from runs/FR/*.json -> figures/fr_fresh_state.png"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fr_core import OUT, FIG

B = json.load(open(OUT / "baseline.json"))
C = json.load(open(OUT / "coupling.json"))
L3 = json.load(open(OUT / "lr3.json"))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))

a = ax[0]
names = list(B)
bad = {"S1v2_r32_fresh_ck1_t1/step", "S2_r8_b0_fresh_ck3/step4"}
for j, s in enumerate(names):
    for F, col, off in (("A", "C0", -0.15), ("B", "C1", 0.15)):
        v = [x[F + "_base"] for x in B[s] if x.get(F + "_base") is not None]
        if not v: continue
        a.scatter(np.full(len(v), j + off) + np.random.default_rng(j).uniform(-.06, .06, len(v)), v, s=3,
                  color=col, alpha=.5, label=(f"lora_{F}: PE5(own G)" if j == 1 else None))
        nv = [x[F + "_null"] for x in B[s] if x.get(F + "_null") is not None]
        if nv: a.scatter([j + off], [np.median(nv)], marker="x", color="k", label=("null (other text's G)" if j == 1 and F == "A" else None))
a.set_yscale("log"); a.set_xticks(range(len(names)))
a.set_xticklabels([n.replace("/step", "").replace("_fresh", "") + (" *" if n in bad else "") for n in names], rotation=80, fontsize=6)
a.set_ylabel("relative Frobenius residual per tensor"); a.legend(fontsize=7)
a.set_title("Fresh-state steps = PE5 of each factor's own gradient\n(* = step/gradient pair that does not match)", fontsize=9)

a = ax[1]
PS = [-1.0, -0.5, -0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25, 0.5, 1.0]
for s, ls in (("S1_r8_fresh_ck5_t0/step", "-"), ("S1v2_r32_fresh_ck5_t0/step", "--")):
    for k, lab in (("A1", "A: PE5((BᵀB)^p G_A)"), ("A3", "A: PE5((AAᵀ)^p G_A)"),
                   ("B1", "B: PE5(G_B (AAᵀ)^p)"), ("B3", "B: PE5(G_B (BᵀB)^p)")):
        tm = C[s][k]["train_median"]
        a.plot(PS, [tm[str(p)] for p in PS], ls, marker=".", label=f"{lab} {'r8' if 'r8' in s else 'r32'}")
a.set_yscale("log"); a.set_xlabel("exponent p"); a.set_ylabel("median residual (train half)")
a.set_title("Coupling / preconditioning exponents: optimum at p = 0\n(state 5; cond(BᵀB) r8 ≤ 9, r32 ≤ 165)", fontsize=9)
a.legend(fontsize=6)

a = ax[2]
for F, col in (("A", "C0"), ("B", "C1")):
    R = [x for x in L3 if x["F"] == F]
    a.scatter([x["s_f16_f64"] for x in R], [x["relE"] for x in R], s=8, color=col, label=f"lora_{F}")
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel("tensor's fp16-vs-f64 PE5 gap on its gradient (rounding sensitivity proxy)")
a.set_ylabel("relF(U(1e-2)/10, U(1e-3))")
a.set_title("r32 step 3, lr 1e-2: deviation is structureless,\nscales with rounding sensitivity (corr A 0.65, B 0.24)", fontsize=9)
a.legend(fontsize=7)
plt.tight_layout()
plt.savefig(FIG / "fr_fresh_state.png", dpi=130)
print("saved", FIG / "fr_fresh_state.png")
