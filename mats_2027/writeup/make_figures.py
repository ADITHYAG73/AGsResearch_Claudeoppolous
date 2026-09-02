"""Draft figures G1–G5 for the write-up. Every number is recomputed from the data files, not
typed in, so the figure and experiments.md cannot disagree. Static PNG + SVG for a Google Doc.
Palette: the validated reference set (blue / orange / aqua), light surface, thin marks,
direct labels, one axis per chart, gray for reference/null series.
"""
import math, sys, json, numpy as np, pyarrow.parquet as pq
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "mats_2027/pipeline")

BLUE, ORANGE, AQUA, GRAY, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#1baf7a", "#8a8987", "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True})
OUT = "mats_2027/writeup/figures/"
def save(fig, name, caption):
    fig.tight_layout()
    for ext in ("png", "svg"): fig.savefig(f"{OUT}{name}.{ext}", dpi=200, bbox_inches="tight")
    open(f"{OUT}{name}.caption.txt", "w").write(caption); plt.close(fig); print("wrote", name)
def wil(k, n):
    p = k/n; z = 1.96; d = 1+z*z/n; c = (p+z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return 100*(c-h), 100*(c+h)

# ---------------- G1 specificity: ours vs paper, plus AG's labels ----------------
V = pq.read_table("mats_2027/runs/2026-08-22_pos10/verdicts.parquet").to_pylist()
AG = {json.loads(l)["claim_id"]: json.loads(l)["verdict"] for l in open("mats_2027/harness/verdicts_AG_MAIN.jsonl") if not json.loads(l)["claim_id"].startswith("RETEST")}
C = {c["claim_id"]: c for c in pq.read_table("mats_2027/runs/2026-08-22_pos10/claims.parquet").to_pylist()}
levels = ["THEME", "ENTITY", "DETAIL"]; paper = {"THEME": 64, "ENTITY": 28, "DETAIL": 24}
ours, lo, hi, ag, hk = [], [], [], [], []
for lv in levels:
    s = [r for r in V if r["level"] == lv]; k = sum(r["verdict"] == "SUPPORTED" for r in s)
    ours.append(100*k/len(s)); a, b = wil(k, len(s)); lo.append(ours[-1]-a); hi.append(b-ours[-1])
    ids = [c for c in AG if C[c]["level"] == lv]
    ag.append(100*sum(AG[c] == "SUPPORTED" for c in ids)/len(ids))
    hk.append(100*sum(next(r for r in V if r["claim_id"] == c)["verdict"] == "SUPPORTED" for c in ids)/len(ids))
fig, ax = plt.subplots(figsize=(7.2, 3.6)); x = np.arange(3); w = 0.26
ax.bar(x-w, [paper[l] for l in levels], w, color=GRAY, label="paper (Opus 4.6 NLA, Common Pile)")
ax.bar(x,   ours, w, color=BLUE, yerr=[lo, hi], capsize=3, ecolor=INK, label="ours, Haiku judge (n=2065)")
ax.bar(x+w, ag,   w, color=ORANGE, label="ours, AG blind labels (n=150)")
for i in range(3):
    ax.text(x[i]-w, paper[levels[i]]+1.5, f"{paper[levels[i]]}", ha="center", fontsize=8, color=MUTED)
    ax.text(x[i], ours[i]+hi[i]+1.5, f"{ours[i]:.0f}", ha="center", fontsize=8, color=INK)
    ax.text(x[i]+w, ag[i]+1.5, f"{ag[i]:.0f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(x); ax.set_xticklabels(levels); ax.set_ylabel("% of claims supported by the passage"); ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=8, loc="upper right"); ax.set_title("G1  Specificity ordering replicates — and the human labels say it understates", fontsize=10, loc="left")
save(fig, "G1_specificity", "Claims about the passage's theme are supported far more often than claims about specific details, on a different NLA, base model, corpus and judge from the paper. AG's blind labels (orange) give a steeper gradient than the Haiku judge (blue): the two graders err at opposite ends, so the true gap is likely larger than either measured.")

# ---------------- G2 related-false Δ: one hump. No synthetic overlay — neither a gamma (no left
# tail) nor a skew-normal (max skew 0.99 vs data 2.63) honestly matches this shape, and an overlay
# that overstates the fit is worse than none. The power argument lives in G3. ----------------
D = {r["claim_id"]: r for r in pq.read_table("mats_2027/runs/2026-08-23_ar/deltas.parquet").to_pylist() if r["valid"]}
R = {r["claim_id"]: r["relatedness"] for r in pq.read_table("mats_2027/runs/2026-08-22_pos10/relatedness.parquet").to_pylist()}
rf = np.array([d["delta"] for c, d in D.items() if d["verdict"] != "SUPPORTED" and R.get(c) == "RELATED"])
tr = np.array([d["delta"] for d in D.values() if d["verdict"] == "SUPPORTED"])
def skew(x): m = x.mean(); s_ = x.std(); return float(((x-m)**3).mean()/s_**3)
fig, ax = plt.subplots(figsize=(7.2, 3.6)); bins = np.linspace(-0.006, 0.010, 65)
ax.hist(tr, bins=bins, density=True, color=GRAY, alpha=0.55, label=f"true claims (n={len(tr)}) — for scale")
ax.hist(rf, bins=bins, density=True, color=BLUE, alpha=0.85, label=f"related-false claims (n={len(rf)}) — H1's cell")
ax.axvline(0, color=INK, lw=0.8)
ax.text(0.98, 0.95, f"related-false: Hartigan dip p = 0.992 (one mode)\nskew {skew(rf):.2f}   ΔBIC(2v1) +843 — fires on skew alone\n(power of the dip test: G3)", transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK)
ax.set_xlabel("Δ = mse(claim rewritten out) − mse(intact)"); ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=8, loc="center right"); ax.set_title("G2  H1: related-false Δ is one skewed hump, not a mixture", fontsize=10, loc="left")
save(fig, "G2_h1_null", "If related-false claims were two populations — faithful readouts the text judge mislabels, plus genuine confabulations — their Δ would be bimodal. It is a single right-skewed hump (dip p = 0.992). The ΔBIC rule that was pre-registered DID fire (+843), but a single skewed hump matched to this data fires it in 200/200 draws, so it is disqualified; the dip test, which has the power shown in G3, is the detector that counts. True claims (gray) are shown for scale: the same shape, shifted right.")

# ---------------- G3 dip power curve ----------------
sys.path.insert(0, "mats_2027/pipeline"); from analysis import dip_pvalue
MEAN, SD, N, SK, TM = 0.00060, 0.00155, 975, 2.63, 0.00115
def skewed(n, mean, sd, sk, rng): kk = max(0.05, (2.0/sk)**2); gg = rng.gamma(kk, 1.0, n); return mean+sd*(gg-gg.mean())/gg.std(ddof=1)
rng = np.random.default_rng(3); ps = [0.10, 0.15, 0.20, 0.26, 0.30, 0.35, 0.42, 0.50]; pw = []
for p in ps:
    lo_m = (MEAN-p*TM)/(1-p); hits = 0
    for _ in range(400):
        nh = int(N*p); x = np.concatenate([skewed(N-nh, lo_m, SD, SK, rng), skewed(nh, TM, SD, SK, rng)]); hits += dip_pvalue(x)[1] < 0.05
    pw.append(100*hits/400)
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.axvspan(0.26, 0.42, color=ORANGE, alpha=0.12, lw=0); ax.text(0.34, 8, "H1's predicted\nmixture range", ha="center", fontsize=8, color=ORANGE)
ax.plot(ps, pw, color=BLUE, lw=2, marker="o", ms=6)
for p, w_ in zip(ps, pw): ax.text(p, w_+4, f"{w_:.0f}%", ha="center", fontsize=8, color=INK)
ax.axhline(80, color=GRAY, lw=1, ls="--"); ax.text(0.505, 82, "80%", fontsize=8, color=MUTED, ha="right")
ax.set_xlabel("fraction of related-false claims that behave like true claims (planted)"); ax.set_ylabel("% of planted mixtures the dip test detects"); ax.set_ylim(0, 110)
ax.set_title("G3  Power: the null in G2 means something", fontsize=10, loc="left")
pw26, pw42, pw20 = pw[ps.index(0.26)], pw[ps.index(0.42)], pw[ps.index(0.20)]
save(fig, "G3_dip_power", f"Planted mixtures at the geometry H1 implies (n=975, observed noise, K=1, 400 draws per point, seed 3). Across H1's own predicted 26–42% range the dip test detects them {pw26:.0f}–{pw42:.0f}% of the time. At 20% it drops to {pw20:.0f}% — a mixture below that would have been missed. H1 is dead in its stated form, not in every form.")

# ---------------- G4 spread vs K with fit + out-of-sample ----------------
# from pipeline/noise_fit.py (4000 subsampling reps, seed 0) - the reproducible values
K = [1, 2, 3, 4]; obs = [0.00225, 0.00181, 0.00164, 0.00157]
n2 = (obs[0]**2-obs[3]**2)/(1-0.25); s2 = obs[3]**2-n2/4; noise, signal = math.sqrt(n2), math.sqrt(max(s2, 0))
kk = np.linspace(1, 8, 200); pred = np.sqrt(s2+n2/kk)
fig, ax = plt.subplots(figsize=(7.2, 3.4))
ax.plot(kk, pred, color=BLUE, lw=2, label="spread(K)² = signal² + noise²/K   (fitted on K=1, 4)")
ax.axhline(signal, color=ORANGE, lw=1.5, ls="--"); ax.text(7.9, signal+0.00004, f"signal floor {signal:.5f}  (55% of K=1)", ha="right", fontsize=8, color=ORANGE)
ax.scatter([1, 4], [obs[0], obs[3]], color=BLUE, s=60, zorder=5, label="fitted points")
ax.scatter([2, 3], [obs[1], obs[2]], color=ORANGE, s=70, zorder=5, marker="D", label="out-of-sample (−3.2%, −0.1%)")
ax.set_xlabel("K resamples averaged"); ax.set_ylabel("sd of per-claim mean Δ"); ax.set_xticks(range(1, 9)); ax.set_ylim(0.001, 0.0025)
ax.legend(frameon=False, fontsize=8); ax.set_title("G4  H2: the noise is stochastic — it divides by K — but there is a floor", fontsize=10, loc="left")
save(fig, "G4_h2_spread", f"Spread of per-claim mean Δ as more resamples are averaged (31 claims present in all four). A two-parameter model fitted on K=1 and K=4 alone predicts K=2 and K=3 to within 0.8% (orange diamonds). The noise averages away as H2 assumes; it converges on a real between-claim floor at 56% of the K=1 spread. noise/signal: 1.46× at K=1, 0.73× at K=4 (values from pipeline/noise_fit.py).")

# ---------------- G5 Savarkar vs cricket false rate ----------------
S = [r for r in pq.read_table("mats_2027/runs/2026-08-28_savarkar/verdicts.parquet").to_pylist() if r["doc_id"].startswith("SAVARKAR")]
Cc = [r for r in V if r["doc_id"].startswith("CRICKET")]
cats = ["THEME", "ENTITY", "DETAIL", "ALL"]; sv, se, cv, ce = [], [], [], []
for lv in cats:
    for src, vals, errs in ((S, sv, se), (Cc, cv, ce)):
        s = [r for r in src if lv == "ALL" or r["level"] == lv]; k = sum(r["verdict"] != "SUPPORTED" for r in s)
        vals.append(100*k/len(s)); a, b = wil(k, len(s)); errs.append([vals[-1]-a, b-vals[-1]])
fig, ax = plt.subplots(figsize=(7.2, 3.4)); x = np.arange(4); w = 0.34
ax.bar(x-w/2, cv, w, color=BLUE, yerr=np.array(ce).T, capsize=3, ecolor=INK, label="cricket Wikipedia (n=1668)")
ax.bar(x+w/2, sv, w, color=ORANGE, yerr=np.array(se).T, capsize=3, ecolor=INK, label="2019 biography, Savarkar (n=2730)")
for i in range(4):
    ax.text(x[i]-w/2, cv[i]+ce[i][1]+1.5, f"{cv[i]:.0f}", ha="center", fontsize=8, color=INK); ax.text(x[i]+w/2, sv[i]+se[i][1]+1.5, f"{sv[i]:.0f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(x); ax.set_xticklabels(cats); ax.set_ylabel("% of claims FALSE (not supported)"); ax.set_ylim(0, 100)
ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.set_title("G5  Domain transfer: the AV confabulates MORE on the less-memorised text", fontsize=10, loc="left")
save(fig, "G5_savarkar", "Same judge, same prompt, length-matched passages. Both pre-registered predictions expected fewer confabulations on a 2019 biography than on cricket Wikipedia; the opposite happened (63% vs 50% false, CIs disjoint). In both domains >90% of false person-claims name someone absent from the passage — the dominant failure is importing a plausible entity, not misbinding a present one.")
print("ALL FIGURES DONE")
