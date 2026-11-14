"""BEHAVE-02 offline analysis: block D floor statistics, block E best-lr ratio, figure bh_floor.png.
Reads runs/BH/{D,E,C}_*.json; writes runs/BH/behave02_stats.json and figures/bh_floor.png. Partial runs included,
flagged by done=False."""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[1]
R, F = ROOT / "runs" / "BH", ROOT / "figures"
load = lambda n: json.loads((R / f"{n}.json").read_text()) if (R / f"{n}.json").exists() else None
D = {p.stem: json.loads(p.read_text()) for p in sorted(R.glob("D_*.json"))}
E = {p.stem: json.loads(p.read_text()) for p in sorted(R.glob("E_*.json"))}
ref = {n: load(n) for n in ["C_adamw_lr0.0003_r8_const", "C_dimuon_lr0.01_r8_const"]}
out = {"D": {}, "C_ref": {}, "E": {}}


def floor(r):
    t = np.array(r["train"]); n = len(t)
    below = np.nonzero(t < 0.01)[0]
    s = dict(done=r.get("done", False), n_steps=n, first_below_0p01=int(below[0]) + 1 if len(below) else None)
    if n >= 60:
        L = t[-60:]
        s.update(last60_steps=f"{n-59}-{n}", median_last60=float(np.median(L)), max_last60=float(L.max()),
                 frac_last60_gt_0p01=float((L > 0.01).mean()))
    if n >= 240:
        s.update(median_121_180=float(np.median(t[120:180])), median_181_240=float(np.median(t[180:240])))
    s["held"] = [[k, round(h, 6)] for k, h in r["held"]]
    return s


for name, r in list(D.items()) + [(k, v) for k, v in ref.items() if v]:
    s = floor(r); (out["D"] if name.startswith("D_") else out["C_ref"])[name] = s
    print(f"{name:32s} done={s['done']} n={s['n_steps']} first<0.01={s['first_below_0p01']} "
          f"med60={s.get('median_last60', float('nan')):.2e} max60={s.get('max_last60', float('nan')):.3g} "
          f"frac>0.01={s.get('frac_last60_gt_0p01', float('nan')):.2f} "
          f"med121-180={s.get('median_121_180', float('nan')):.2e} med181-240={s.get('median_181_240', float('nan')):.2e} "
          f"held_last={s['held'][-1]}")

for name, r in E.items():
    t = np.array(r["train"])
    out["E"][name] = dict(opt=r["opt"], lr=r["lr"], done=r.get("done", False), n_steps=len(t),
                          train_mean=float(t.mean()) if len(t) else None,
                          train_last6=float(t[-6:].mean()) if len(t) >= 6 else None,
                          held_final=r["held"][-1][1])
    print(name, out["E"][name])
best = {}
for opt in ["adamw", "dimuon"]:
    xs = [v for v in out["E"].values() if v["opt"] == opt and v["done"]]
    if len(xs) == 3:
        b = min(xs, key=lambda v: v["train_mean"]); best[opt] = b["lr"]
if len(best) == 2:
    out["E_best"] = dict(best, ratio=best["dimuon"] / best["adamw"])
    print("E best", out["E_best"])
(R / "behave02_stats.json").write_text(json.dumps(out, indent=1))

fig, ax = plt.subplots(figsize=(10, 5.5))
sty = {"C_adamw_lr0.0003_r8_const": ("C0", "-", 1.4, "AdamW 3e-4 (best; BEHAVE-01, 150 steps)"),
       "C_dimuon_lr0.01_r8_const": ("C3", "-", 1.4, "Dimuon 1e-2 (best; BEHAVE-01, 150 steps)"),
       "D_adamw_lr0.0001_r8_const240": ("C0", "--", 1, "AdamW 1e-4 (1/3 best)"),
       "D_adamw_lr3e-05_r8_const240": ("C0", ":", 1, "AdamW 3e-5 (1/10 best)"),
       "D_dimuon_lr0.003_r8_const240": ("C3", "--", 1, "Dimuon 3e-3 (1/3 best)"),
       "D_dimuon_lr0.001_r8_const240": ("C3", ":", 1, "Dimuon 1e-3 (1/10 best)")}
for name, (c, ls, lw, lab) in sty.items():
    r = D.get(name) or ref.get(name)
    if r:
        ax.plot(range(1, len(r["train"]) + 1), r["train"], color=c, ls=ls, lw=lw,
                label=lab + ("" if r.get("done") else f" [partial, {len(r['train'])} steps]"))
ax.axhline(0.01, color="grey", lw=0.6)
ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("train loss/token (minibatch of 8, before step)")
ax.set_title("Constant lr, rank 8: late-training floor vs lr (Qwen3.5-4B, Velish task)")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(F / "bh_floor.png", dpi=120)
