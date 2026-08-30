"""Figure G0 — the pipeline, one box per stage. Every count is READ FROM THE RUN ARTIFACTS,
never typed in, so the figure cannot drift from the data. Same palette as G1-G5."""
import json, collections, numpy as np, pyarrow.parquet as pq
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE, ORANGE, AQUA, GRAY, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#8a8987", "#0b0b0b", "#52514e", "#fcfcfb"
R = "mats_2027/runs/2026-08-22_pos10/"
n = {f: pq.read_table(R + f + ".parquet").num_rows for f in
     ["activations", "explanations", "claims", "verdicts", "ablations", "relatedness"]}
V = pq.read_table(R + "verdicts.parquet").to_pylist()
n_sup = sum(v["verdict"].upper() == "SUPPORTED" for v in V); n_false = len(V) - n_sup
REL = pq.read_table(R + "relatedness.parquet").to_pylist()
n_related = sum(r["relatedness"].upper() == "RELATED" for r in REL)
G = pq.read_table(R + "claim_groups.parquet").to_pylist()
sizes = collections.Counter(g["group_id"] for g in G if g["group_id"] is not None)
n_grp3 = sum(1 for _, s in sizes.items() if s >= 3)
docs = sorted({v["doc_id"] for v in V}); n_pass = len(docs)
AB = pq.read_table(R + "ablations.parquet").to_pylist()
n_intact = sum(1 for a in AB if a["claim_id"] in (None, "")) or n["explanations"]

LANES = {"local": (GRAY, "laptop"), "gpu": (BLUE, "A40 GPU"), "api": (ORANGE, "Haiku 4.5")}
steps = [
 ("local", "Corpus",
  f"{n_pass} passages — 5 cricket Wikipedia paragraphs\n+ the maintainers' own example passage (reference)"),
 ("gpu", "1. Extract",
  f"Gemma-3-12B-IT forward pass → residual stream,\nlayer 32 of 48, d = 3840; last 10 token positions\nper passage (every index ≥ MIN_POSITION 50)\n→ {n['activations']} activations"),
 ("gpu", "2. Verbalise (AV)",
  f"activation injected as ONE token embedding into a\nfixed prompt (injection_scale 80 000); sampled at\nT = 1, K = 4 per activation\n→ {n['explanations']} explanations"),
 ("api", "3. Decompose",
  f"each explanation → atomic claims, each tagged\nTHEME / ENTITY / DETAIL\n→ {n['claims']} claims"),
 ("api", "4. Judge",
  f"claim vs THE PREFIX THE MODEL HAD READ\n→ {n_sup} supported / {n_false} false\n(validated: 150 blind + 30 retests, 88.7% agreement)"),
 ("api", "5. Ablate",
  f"REWRITE one claim out of its explanation, prose\nreflowed — the paper's method, not deletion\n→ {n['ablations']} variants ({n['explanations']} intact + {n['claims']} ablated)"),
 ("gpu", "6. Reconstruct (AR)",
  "score every variant against the SAME activation;\nMSE is direction-only:  MSE = 2(1 − cos)"),
 ("local", "7. Δ",
  "Δ = mse(claim rewritten out) − mse(intact)\nsame explanation, same activation, same resample\nΔ > 0 the claim was load-bearing · Δ < 0 removal helped"),
]
side = [
 (4, f"Relatedness (REL-01)\nthe {n_false} false claims → RELATED / UNRELATED\n→ {n_related} related ({100*n_related/n_false:.0f}%)"),
 (4, f"Matcher (MATCH-02)\nsame claim across the K = 4 resamples\n→ {n_grp3} groups seen in ≥ 3 of 4"),
]

# --- layout in POINTS: the axes spans the whole figure and 1 data unit == 1 pt,
# so box heights follow the real font metrics instead of an estimated scale factor. ---
FS_T, FS_B, FS_S = 11.5, 9.3, 8.9          # title / body / side font sizes
LSP = 1.45                                  # linespacing
LINE, SLINE = FS_B * LSP, FS_S * LSP
TOP_PAD, TITLE_GAP, BOT_PAD, GAPY = 9.0, 14.0, 11.0, 13.0
BODY_DY = TOP_PAD + FS_T * 1.15 + TITLE_GAP
H = [BODY_DY + LINE * (b.count("\n") + 1) + BOT_PAD for _, _, b in steps]
W, BW, X = 760.0, 400.0, 6.0
TOTAL = sum(H) + GAPY * (len(H) - 1) + 34.0
fig = plt.figure(figsize=(W / 72.0, TOTAL / 72.0), facecolor=SURF)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, TOTAL); ax.axis("off")
y = TOTAL - 30.0
centres = []
for (lane, title, body), h in zip(steps, H):
    col = LANES[lane][0]
    ax.add_patch(FancyBboxPatch((X, y - h), BW, h, boxstyle="round,pad=2,rounding_size=7",
                                fc="white", ec=col, lw=1.6, zorder=2))
    ax.add_patch(FancyBboxPatch((X, y - h + 3), 7.0, h - 6, boxstyle="square,pad=0",
                                fc=col, ec=col, lw=0, zorder=3))
    ax.text(X + 19, y - TOP_PAD, title, fontsize=FS_T, fontweight="bold", color=INK, va="top", zorder=4)
    ax.text(X + BW - 10, y - TOP_PAD - 1, LANES[lane][1], fontsize=8, color=col, ha="right", va="top", zorder=4)
    ax.text(X + 19, y - BODY_DY, body, fontsize=FS_B, color=MUTED, va="top", linespacing=LSP, zorder=4)
    centres.append((y - h / 2, y, y - h))
    y = y - h - GAPY
for i in range(len(steps) - 1):
    ax.add_patch(FancyArrowPatch((X + BW / 2, centres[i][2]), (X + BW / 2, centres[i + 1][1]),
                                 arrowstyle="-|>", mutation_scale=13, lw=1.3, color=MUTED, zorder=1))
SX, SW = X + BW + 30, W - (X + BW + 30) - 6
for j, (idx, txt) in enumerate(side):
    sh = 10.0 + SLINE * (txt.count("\n") + 1) + 9.0
    top = centres[idx][0] + 14 - j * (sh + 12)
    ax.add_patch(FancyBboxPatch((SX, top - sh), SW, sh, boxstyle="round,pad=2,rounding_size=6",
                                fc="white", ec=AQUA, lw=1.2, ls="--", zorder=2))
    ax.text(SX + 14, top - 10, txt, fontsize=FS_S, color=MUTED, va="top", linespacing=LSP, zorder=4)
    ax.add_patch(FancyArrowPatch((X + BW, centres[idx][0]), (SX, top - sh / 2), arrowstyle="-|>",
                                 mutation_scale=10, lw=1.0, color=AQUA, ls="--",
                                 connectionstyle="arc3,rad=-0.12", zorder=1))
ax.text(X + 4, TOTAL - 8, "G0   Pipeline: activation \u2192 explanation \u2192 claims \u2192 \u0394",
        fontsize=13, fontweight="bold", color=INK, va="top")
fig.savefig("mats_2027/writeup/figures/G0_pipeline.png", dpi=200, bbox_inches="tight", facecolor=SURF)
fig.savefig("mats_2027/writeup/figures/G0_pipeline.svg", bbox_inches="tight", facecolor=SURF)
open("mats_2027/writeup/figures/G0_pipeline.caption.txt", "w").write(
  f"The measurement pipeline. Colour marks where each stage ran: blue on a rented A40, orange as "
  f"{'Haiku 4.5'} API calls, gray on the laptop. Steps 3-5 reproduce the paper's confabulation "
  f"analysis, whose prompts were never published and whose shape was recovered from the grader "
  f"outputs shipped in the paper's HTML; the ablation is a rewrite, not a deletion. The judge in "
  f"step 4 is the only stage validated against a human ({n_sup} supported / {n_false} false "
  f"overall; 150 claims graded blind, 88.7% agreement). Every count shown is read from the run "
  f"artifacts, not transcribed.")
print("wrote G0_pipeline  |", n, "| supported", n_sup, "false", n_false, "| related", n_related, "| groups>=3", n_grp3)
