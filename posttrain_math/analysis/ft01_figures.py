#!/usr/bin/env python3
"""FT-01 figures, drawn from ft01_numbers.json (+ loss.jsonl). Runs on a pod, writes PNGs.
  F1 four cells: base vs s050 vs s100, 95% CIs
  F2 dose curve: accuracy vs fraction of training (0, 0.5, 1.0), one line per condition
  F3 IFEval: prompt-level strict, thinking off/on, per model
  F4 training loss per optimizer step
  F5 flips: answers changed by fine-tuning vs by GPU alone (A40 vs A100, same model)
  F6 per-family change, s100 minus base, per condition (heatmap)
Usage: python ft01_figures.py ft01_numbers.json loss.jsonl outdir
"""
import json, sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N, LOSS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
d = json.load(open(N))
CONDS = ["notool_off", "notool_on", "tool_off", "tool_on"]
LABEL = {"notool_off": "alone\nthinking off", "notool_on": "alone\nthinking on",
         "tool_off": "sandbox\nthinking off", "tool_on": "sandbox\nthinking on"}
M = [("Qwen3.5-4B-rerun", "base", "#4a4a4a"), ("Qwen3.5-4B-ifc4-s050", "fine-tuned 50%", "#4f8fd6"),
     ("Qwen3.5-4B-ifc4-s100", "fine-tuned 100%", "#d6604d")]
M = [m for m in M if m[0] in d["summary"]]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# F1
fig, ax = plt.subplots(figsize=(8, 4)); w = 0.8 / len(M); x = np.arange(len(CONDS))
for i, (k, lab, c) in enumerate(M):
    s = [d["summary"][k][cd] for cd in CONDS]
    y = [v["pct"] for v in s]; lo = [v["pct"] - v["ci95"][0] for v in s]; hi = [v["ci95"][1] - v["pct"] for v in s]
    ax.bar(x + (i - (len(M) - 1) / 2) * w, y, w, yerr=[lo, hi], capsize=2, color=c, label=lab)
ax.set_xticks(x, [LABEL[c] for c in CONDS]); ax.set_ylabel("correct to 6 s.f. (%)"); ax.set_ylim(0, 100)
ax.set_title("Qwen3.5-4B, 3,362 questions: before and after fine-tuning (95% CI)"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/F1_four_cells.png", dpi=200); plt.close(fig)

# F2
frac = {"Qwen3.5-4B-rerun": 0.0, "Qwen3.5-4B-ifc4-s050": 0.5, "Qwen3.5-4B-ifc4-s100": 1.0}
fig, ax = plt.subplots(figsize=(6, 4))
for cd, c in zip(CONDS, ["#999999", "#555555", "#7fb3e6", "#1f5fa8"]):
    xs = [frac[k] for k, _, _ in M]; ys = [d["summary"][k][cd]["pct"] for k, _, _ in M]
    ax.plot(xs, ys, "o-", color=c, label=LABEL[cd].replace("\n", ", "))
ax.set_xlabel("fraction of fine-tuning completed"); ax.set_ylabel("correct to 6 s.f. (%)"); ax.set_ylim(0, 100)
ax.set_title("Maths accuracy along the fine-tune"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/F2_dose_curve.png", dpi=200); plt.close(fig)

# F3
if d["ifeval"]:
    fig, ax = plt.subplots(figsize=(6, 4)); x = np.arange(2)
    for i, (k, lab, c) in enumerate(M):
        if k not in d["ifeval"]: continue
        y = [100 * d["ifeval"][k][t]["prompt_strict"] for t in ("off", "on")]
        ax.bar(x + (i - (len(M) - 1) / 2) * w, y, w, color=c, label=lab)
        # thinking-on scores are budget-bound: print how many of 541 never closed </think>
        u = d["ifeval"][k]["on"]["unclosed_think"]
        ax.text(1 + (i - (len(M) - 1) / 2) * w, y[1] + 1, f"{u}\nunclosed", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, ["thinking off", "thinking on"]); ax.set_ylabel("IFEval prompt-level strict (%)"); ax.set_ylim(0, 100)
    ax.set_title("Did the fine-tune take? IFEval (541 prompts)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{OUT}/F3_ifeval.png", dpi=200); plt.close(fig)

# F4
L = [json.loads(l) for l in open(LOSS)]
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot([r["step"] for r in L], [r["loss"] for r in L], color="#bbbbbb", lw=0.8, label="per step")
k = 10; sm = np.convolve([r["loss"] for r in L], np.ones(k) / k, mode="valid")
ax.plot(range(k, k + len(sm)), sm, color="#d6604d", label=f"{k}-step mean")
for s, t in ((78, "25%"), (156, "50%"), (313, "100%")): ax.axvline(s, ls=":", color="#888"); ax.text(s, ax.get_ylim()[1], t, ha="center", va="bottom", fontsize=8)
ax.set_xlabel("optimizer step"); ax.set_ylabel("loss on the trained turn"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/F4_loss.png", dpi=200); plt.close(fig)

# F5
fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(CONDS))
hw = d["hardware_noise_a40_vs_a100"]
series = [("GPU only (A40→A100, same model)", "#999999", lambda cd: hw.get(cd))]
for k, lab, c in M[1:]:
    series.append((f"fine-tune {lab.split()[-1]} vs base (same GPU)", c, lambda cd, k=k: d["paired_vs_base_a100"][k].get(cd)))
w2 = 0.8 / len(series)
for i, (lab, c, f) in enumerate(series):
    g = [f(cd)["gained"] if f(cd) else 0 for cd in CONDS]; l = [f(cd)["lost"] if f(cd) else 0 for cd in CONDS]
    xo = x + (i - (len(series) - 1) / 2) * w2
    ax.bar(xo, g, w2, color=c, label=lab); ax.bar(xo, [-v for v in l], w2, color=c, alpha=0.5)
ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x, [LABEL[c] for c in CONDS])
ax.set_ylabel("questions flipped  (up: wrong→right, down: right→wrong)")
ax.set_title("Answer flips: fine-tuning vs hardware noise"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/F5_flips_vs_noise.png", dpi=200); plt.close(fig)

# F6
last = M[-1][0]
fams = ["trig_int", "trig_var", "transcendental", "composite", "nested", "mixed"]
Z = np.array([[d["family_pct"][last][cd][f] - d["family_pct"]["Qwen3.5-4B-rerun"][cd][f] for cd in CONDS] for f in fams])
fig, ax = plt.subplots(figsize=(6.5, 4)); v = max(5, np.abs(Z).max())
im = ax.imshow(Z, cmap="RdBu", vmin=-v, vmax=v)
ax.set_xticks(range(4), [LABEL[c] for c in CONDS], fontsize=7, rotation=30, ha="right"); ax.set_yticks(range(len(fams)), fams)
for i in range(len(fams)):
    for j in range(4): ax.text(j, i, f"{Z[i, j]:+.1f}", ha="center", va="center", fontsize=8, color="white" if abs(Z[i, j]) > 0.6 * v else "black")
fig.colorbar(im, ax=ax, label="points, fine-tuned 100% minus base")
ax.set_title("Where the change happens, by question family")
fig.tight_layout(); fig.savefig(f"{OUT}/F6_family_change.png", dpi=200); plt.close(fig)
print("FIGURES DONE", sorted(os.listdir(OUT)))
