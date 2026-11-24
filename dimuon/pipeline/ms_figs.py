"""ms_figs.py -- MULTISTEP-01 figure: beta1=0 continuing steps, test-half median relF per step for
(a) PE5(g_t) alone, (b) the fitted model (symmetric previous-momentum preconditioner), and the
fresh-twin floor (fresh-state step vs PE5(g_t): the same map with no state, i.e. the floor of
'PE5 + this oracle gradient'). Bars of the p90 shown as thin whiskers."""
import json, sys, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from ms_common import OUT, ROOT
C = {"base": "#eb6834", "model": "#2a78d6", "floor": "#52514e"}
fl = json.load(open(OUT / "fresh_floor.json"))
fig, axs = plt.subplots(2, 2, figsize=(9, 6.4), sharex=True, sharey=True)
for i, r in enumerate([8, 32]):
    ev = json.load(open(OUT / f"b0_eval_r{r}.json"))["results"]
    for j, kind in enumerate("AB"):
        ax = axs[i, j]; ks = [2, 3, 4, 5, 6]
        series = [("PE5(g_t) alone  [baseline]", "base", lambda k: ev[f"step{k}_{kind}"]["PE5(g_t) [baseline a]"]),
                  ("model: Dl·PE5(Dl m Dr)·Dr, stats from m_{t-1}", "model", lambda k: ev[f"step{k}_{kind}"]["MODEL sym prev-grad (fitted eps)"]),
                  ("fresh twin vs PE5(g_t)  [floor]", "floor", lambda k: fl[f"r{r}_step{k}_{kind}"])]
        for lab, c, f in series:
            med = [f(k)["med"] for k in ks]; p90 = [f(k)["p90"] for k in ks]
            ls = "--" if c == "floor" else "-"
            ax.plot(ks, med, ls, color=C[c], lw=2, marker="o", ms=8, label=lab, mec="white", mew=1.5)
            ax.vlines(ks, med, p90, color=C[c], lw=1, alpha=.6)
        ax.set_yscale("log"); ax.set_ylim(2e-3, 1.5); ax.set_xticks(ks)
        ax.set_title(f"rank {r}, lora_{kind}", fontsize=10, color="#0b0b0b")
        ax.grid(True, which="major", color="#e4e3df", lw=.6); ax.set_axisbelow(True)
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]: ax.spines[s].set_color("#9b9a95")
        ax.tick_params(colors="#52514e", labelsize=8)
        if i == 1: ax.set_xlabel("step (S2 b1=0 chain; batches T0,T0,T1,T2,T0,T1)", fontsize=8, color="#52514e")
        if j == 0: ax.set_ylabel("relative Frobenius residual\n(median; whisker to p90, test half)", fontsize=8, color="#52514e")
h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, fontsize=8, frameon=False, loc="lower center", ncol=3)
fig.suptitle("Dimuon at beta1 = 0: continuing step explained by a previous-momentum diagonal preconditioner", fontsize=10.5)
fig.tight_layout(rect=(0, 0.05, 1, 1))
p = ROOT / "figures" / "ms_b0_residuals.png"; fig.savefig(p, dpi=150); print(p)
