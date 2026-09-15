"""NOISE-LAW-01: reproduce Forking Fast §3.3 / App F on the shipped S=1000 store.

Claim under test: replicate-vs-replicate TVD of o_t falls as S^-1/2 (multinomial noise).
Paper reports slope -0.4903 through S=200 and measured/null ratios 0.98-1.01.

Method (paper's, as described): split the 1000 draws per branch into disjoint blocks of
size S; estimate o_t from each block; TVD between block estimates at every position;
pool over positions and block pairs. Null: simulate multinomial counts of size S from the
full-pool per-branch histogram and run the identical statistic.
"""
import sys, json, itertools
import numpy as np
sys.path.insert(0, "reference_repos/forking-fast/data")
import loader

STORE = "reference_repos/forking-fast/data/s1000/llama/row039_everytoken.json.gz"
OUT = "forking/runs/2026-09-07_noise"
S_GRID = [5, 10, 20, 25, 50, 100, 125, 200, 250, 500]
rng = np.random.default_rng(0)

rec = loader.load_store(STORE)
cats = rec["categories"]; K = len(cats); cat_idx = {c: i for i, c in enumerate(cats)}
by_t = loader.branch_groups(rec)
idxs = sorted(by_t)
S_TOTAL = rec["s"]

# per branch: answers as int codes (draw order), and normalized weights per position
enc = {t: [np.array([cat_idx.get(a, cat_idx["Other"]) for a in b["answers"]]) for b in bs]
       for t, bs in by_t.items()}
wts = {t: loader.normalized_weights(bs) for t, bs in by_t.items()}

def hist(codes):
    h = np.bincount(codes, minlength=K).astype(float)
    return h / h.sum() if h.sum() else h

def o_t_from_block(t, lo, hi):
    return sum(w * hist(c[lo:hi]) for w, c in zip(wts[t], enc[t]))

def tvd(p, q): return 0.5 * np.abs(p - q).sum()

def measured(S):
    nb = S_TOTAL // S
    vals = []
    for t in idxs:
        ests = [o_t_from_block(t, i*S, (i+1)*S) for i in range(nb)]
        vals += [tvd(a, b) for a, b in itertools.combinations(ests, 2)]
    return float(np.mean(vals)), len(vals)

def null(S, reps=20):
    """Exact multinomial null: draw S counts per branch from that branch's full-pool
    histogram, build two independent o_t estimates, TVD. Same weights, same K."""
    vals = []
    for t in idxs:
        full = [hist(c) for c in enc[t]]
        for _ in range(reps):
            ests = []
            for _ in range(2):
                o = np.zeros(K)
                for w, p in zip(wts[t], full):
                    o += w * rng.multinomial(S, p) / S
                ests.append(o)
            vals.append(tvd(*ests))
    return float(np.mean(vals))

rows = []
for S in S_GRID:
    m, n = measured(S); z = null(S)
    rows.append(dict(S=S, tvd_measured=m, n_pairs=n, tvd_null=z, ratio=m / z))
    print(f"S={S:4d}  measured={m:.4f}  null={z:.4f}  ratio={m/z:.3f}  pairs={n}")

S_arr = np.array([r["S"] for r in rows]); m_arr = np.array([r["tvd_measured"] for r in rows])
mask = S_arr <= 200
slope_all, icpt = np.polyfit(np.log(S_arr), np.log(m_arr), 1)
slope_200, _ = np.polyfit(np.log(S_arr[mask]), np.log(m_arr[mask]), 1)
print(f"\nlog-log slope, all S: {slope_all:.4f}   through S<=200: {slope_200:.4f}   (paper: -0.4903; theory: -0.5)")

json.dump(dict(store=STORE, n_positions=len(idxs), rows=rows,
               slope_all=slope_all, slope_le200=slope_200), open(f"{OUT}/noise_law.json", "w"), indent=1)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.loglog(S_arr, m_arr, "o-", label=f"measured, slope {slope_200:.3f} (S≤200)")
ax.loglog(S_arr, [r["tvd_null"] for r in rows], "s--", alpha=.7, label="i.i.d. multinomial null")
ax.loglog(S_arr, np.exp(icpt) * S_arr ** -0.5, ":", color="k", label="reference slope −1/2")
ax.set_xlabel("samples per branch, S"); ax.set_ylabel("replicate TVD, pooled over positions")
ax.set_title("Row 39, Llama-3-8B, every-token, 1000 draws split into disjoint blocks")
ax.legend(fontsize=8); ax.grid(alpha=.3, which="both")
fig.tight_layout(); fig.savefig(f"{OUT}/noise_law.png", dpi=150)
print("wrote", OUT)
