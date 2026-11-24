"""BEHAVE-01 figures + summary table from runs/BH/*.json (offline)."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "runs" / "BH"
F = ROOT / "figures"
F.mkdir(exist_ok=True)
runs = {p.stem: json.loads(p.read_text()) for p in sorted(R.glob("*.json"))
        if p.stem != "summary"}
runs = {k: v for k, v in runs.items() if v.get("done")}


def fin(r, k=6):  # mean train loss over the last k steps (one pass over the 48 examples)
    return float(np.mean(r["train"][-k:]))


def best_held(r):
    return min(h for _, h in r["held"])


rows = []
for k, r in runs.items():
    rows.append(dict(name=k, opt=r["opt"], lr=r["lr"], rank=r["rank"], steps=r["steps"],
                     schedule=r["schedule"], train_last6=fin(r), train_mean=float(np.mean(r["train"])), held_final=r["held"][-1][1],
                     held_min=best_held(r), held0=r["held"][0][1],
                     train_last30_sd=float(np.std(np.diff(r["train"][-30:]))) if r["steps"] > 60
                     else None))
(R / "summary.json").write_text(json.dumps(rows, indent=1))
for x in sorted(rows, key=lambda x: (x["name"][0], x["opt"], x["rank"], x["lr"])):
    print(f"{x['name']:34s} train6={x['train_last6']:.4f} held_fin={x['held_final']:.4f} "
          f"held_min={x['held_min']:.4f}")

# Fig 1: block A loss vs step
A = {k: v for k, v in runs.items() if k.startswith("A_")}
if A:
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j, opt in enumerate(["adamw", "dimuon"]):
        rs = sorted([v for v in A.values() if v["opt"] == opt], key=lambda v: v["lr"])
        cm = plt.cm.viridis(np.linspace(0, 0.9, max(len(rs), 1)))
        for c, v in zip(cm, rs):
            ax[0, j].plot(range(1, len(v["train"]) + 1), v["train"], color=c, lw=1,
                          label=f"lr {v['lr']:g}")
            h = np.array(v["held"])
            ax[1, j].plot(h[:, 0], h[:, 1], "o-", color=c, ms=3)
        ax[0, j].set_title(f"{opt}, rank 8: train loss/token (minibatch, before step)")
        ax[1, j].set_title(f"{opt}: held-out loss/token (16 unseen combinations)")
        ax[0, j].set_yscale("log"); ax[1, j].set_yscale("log")
        ax[0, j].legend(fontsize=8); ax[1, j].set_xlabel("step")
    fig.tight_layout(); fig.savefig(F / "bh_curves.png", dpi=120); plt.close(fig)

    # Fig 2: final loss vs lr, Dimuon lr also divided by predicted factor
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, (key, lab) in enumerate([("train_mean", "mean train loss over all 60 steps (speed)"), ("train_last6", "train loss, last 6 steps"),
                                    ("held_final", "held-out loss at step 60")]):
        for opt, mk in [("adamw", "o-"), ("dimuon", "s-")]:
            xs = sorted([x for x in rows if x["name"].startswith("A_") and x["opt"] == opt],
                        key=lambda x: x["lr"])
            lr = np.array([x["lr"] for x in xs]); y = [x[key] for x in xs]
            ax[i].plot(lr, y, mk, label=opt)
            if opt == "dimuon":
                for fac, ls in [(30, ":"), (90, "--")]:
                    ax[i].plot(lr / fac, y, ls, color="C1", alpha=.6, label=f"dimuon lr / {fac}")
        ax[i].set_xscale("log"); ax[i].set_yscale("log"); ax[i].set_xlabel("learning rate")
        ax[i].set_title(lab); ax[i].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(F / "bh_lr.png", dpi=120); plt.close(fig)

# Fig 3: block C
C = {k: v for k, v in runs.items() if k.startswith("C_")}
if C:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for j, opt in enumerate(["adamw", "dimuon"]):
        for k, v in sorted(C.items()):
            if v["opt"] != opt:
                continue
            ls = "-" if v["schedule"] == "const" else "--"
            ax[j].plot(range(1, len(v["train"]) + 1), v["train"], ls, lw=1,
                       label=f"lr {v['lr']:g} {v['schedule']}")
        ax[j].set_yscale("log"); ax[j].set_title(f"{opt}, rank 8, 150 steps: train loss/token")
        ax[j].set_xlabel("step"); ax[j].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(F / "bh_blockC.png", dpi=120); plt.close(fig)
