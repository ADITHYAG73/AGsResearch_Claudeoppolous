"""p3_figs.py -- PREDICT-01 figures from the score json files (runs/P3/score_*.json, p6_table.json).
  figures/p3_horizon.png   P6 (Qwen, 12 steps) + P7 (gpt-oss-20b, 5 steps): median rule residual per
                           step vs nulls and the fresh-twin floor
  figures/p3_clip_storage.png  (a) P1a fitted clip coefficient vs clip/N  (b) P2 rank 32 (solo oracle):
                           bf16 vs f32 momentum storage, residual per step
"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
P3 = ROOT / "runs" / "P3"; FIG = ROOT / "figures"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED = "#222222", "#777777"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 2})

T = json.load(open(P3 / "p6_table.json"))
S7 = json.load(open(P3 / "score_p7.json"))
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
for ax, kind in zip(axs, "AB"):
    steps = list(range(1, 13))
    rule = [T[f"step{t}|bf16"][kind].get("med", 0) for t in steps]
    nost = [T[f"step{t}|nostate"][kind].get("med", 0) for t in steps]
    prev = [T[f"step{t}|prev"][kind]["med"] for t in steps[1:]]
    xs = [t for t in steps if rule[t - 1] > 0]
    ax.plot(xs, [rule[t - 1] for t in xs], "-o", color=C1, ms=5, label="rule, Qwen3.5-4B r8 (P6)")
    r7 = [(int(k.split()[0][4:]), v[kind]["med"]) for k, v in S7.items()
          if k.endswith(" rule") and k.startswith("step") and v[kind].get("med", 0) > 0]
    ax.plot([a for a, _ in r7], [b for _, b in r7], "-s", color=C3, ms=5, label="rule, gpt-oss-20b r8 (P7)")
    ax.plot([t for t in steps if nost[t - 1] > 0], [n for n in nost if n > 0], ":", color=C2,
            label="null: PE5(g_t), no state")
    ax.plot(steps[1:], prev, "--", color=C4, label="null: previous update")
    fl = [(t, T[f"step{t}|floor"][kind]["med"]) for t in (4, 8, 12)]
    ax.plot([a for a, _ in fl], [b for _, b in fl], "D", color=INK, ms=6, mfc="white",
            label="floor: fresh twin vs PE5(g)")
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_title(f"lora_{kind}", color=INK)
    ax.set_xticks(steps); ax.grid(axis="y", color="#e6e6e6", lw=0.6)
axs[0].set_ylabel("median relative Frobenius residual")
axs[1].legend(frameon=False, fontsize=7.5, loc="center right")
fig.suptitle("Rule rolled forward from gradients only: residual per step (no accumulation)", color=INK)
fig.tight_layout(); fig.savefig(FIG / "p3_horizon.png", dpi=150)

# ---- clip + storage
S1 = json.load(open(P3 / "score_p1a.json"))
S2 = json.load(open(P3 / "score_p2_solo.json"))
fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
ax = axs[0]
cs = sorted(float(k[5:]) for k in S1 if k.startswith("cfit_"))
fits = [S1[f"cfit_{c:g}"]["c_fit"] for c in cs]
ax.plot([0.005, 1.2], [0.005, 1.2], color="#cccccc", lw=1, label="c_fit = clip / N_all")
ax.plot(cs, fits, "o", color=C1, ms=7, label="fitted c (B tensors), P1a")
for c, f in zip(cs, fits):
    ax.annotate(f"{f:.4f}", (c, f), textcoords="offset points", xytext=(6, -10), fontsize=7, color=INK)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("clip / N_all  (N over all LoRA tensors)")
ax.set_ylabel("c minimising residual"); ax.set_title("P1a: clipping = g -> c g, c = clip/N", color=INK)
ax.legend(frameon=False, fontsize=7.5)
ax = axs[1]
lab = ["step1", "step2", "zero1", "zero2", "zero3"]
for v, col, mk, nm in (("bf16_x0f16", C1, "o", "momentum stored in bf16"),
                       ("f32_x0f16", C2, "s", "momentum stored in f32")):
    for kind, ls in (("A", "-"), ("B", "--")):
        ys = [S2[f"{t} rule[{v}]"][kind].get("med", 0) for t in lab]
        xs = [i for i, y in enumerate(ys) if y > 0]
        ax.plot(xs, [ys[i] for i in xs], ls, marker=mk, color=col, ms=5, label=f"{nm}, lora_{kind}")
ax.set_xticks(range(5)); ax.set_xticklabels(lab); ax.set_yscale("log")
ax.set_ylabel("median relative Frobenius residual")
ax.set_title("P2: rank 32, 2 steps + 3 zero-gradient steps", color=INK)
ax.grid(axis="y", color="#e6e6e6", lw=0.6); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "p3_clip_storage.png", dpi=150)
print("ok")
