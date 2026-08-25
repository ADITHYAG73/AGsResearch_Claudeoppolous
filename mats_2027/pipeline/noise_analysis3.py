"""NOISE-03 - paired-Delta noise and H2's KILL CONDITION, using the SEMANTIC matcher.

Supersedes noise_analysis2.py, which matched recurrence with a regex (`norm()`) and found
only 22 usable groups out of 2065 claims - 1796 of 1916 were singletons - so the 1/sqrt(K)
test could not run at all. Stage 5 (MATCH-02) replaces that with a two-pass semantic matcher:
115 groups with >=3 members, 35 with >=4.

H2: the per-claim signal is noise-limited, not absent, so averaging Delta over K resamples
should shrink the spread of per-claim means as ~1/sqrt(K) - log-log slope -0.50.
AG's KILL CONDITION (2026-08-19): "if variance does NOT fall as ~1/sqrt(K), the noise is
systematic rather than stochastic, averaging cannot rescue it, and the approach is dead."
A slope near 0 kills H2 - and is a clean negative result, not a failure.

Recurrence is matched WITHIN a position: the same claim at two positions comes from a
DIFFERENT activation, so its Delta legitimately differs. Pooling across positions would
inflate the noise estimate.
"""
import argparse, math
from collections import defaultdict, Counter
import numpy as np
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--deltas", default="mats_2027/runs/2026-08-23_ar/deltas.parquet")
ap.add_argument("--groups", default="mats_2027/runs/2026-08-22_pos10/claim_groups.parquet")
ap.add_argument("--min-k", type=int, default=3)
a = ap.parse_args()

D = {r["claim_id"]: r for r in pq.read_table(a.deltas).to_pylist() if r["valid"]}
G = pq.read_table(a.groups).to_pylist()
grp = defaultdict(list)
for r in G:
    if r["claim_id"] in D: grp[r["group_id"]].append(D[r["claim_id"]])

# A group can hold >1 claim from the SAME resample (one explanation can state a thing twice).
# Average within a resample first, or those claims get double-weighted.
def by_k(rows):
    d = defaultdict(list)
    for r in rows: d[r["k"]].append(r["delta"])
    return {k: float(np.mean(v)) for k, v in d.items()}

per = {g: by_k(rows) for g, rows in grp.items()}
sizes = Counter(len(v) for v in per.values())
print(f"{len(D)} valid ablations  ->  {len(per)} semantic groups")
print("distinct resamples per group: " + "  ".join(f"{n}x{sizes[n]}" for n in sorted(sizes)))

kept = {g: v for g, v in per.items() if len(v) >= a.min_k}
print(f"groups spanning >= {a.min_k} resamples: {len(kept)}   "
      f"(exact-string matching gave 22)\n")

sds   = np.array([np.std(list(v.values()), ddof=1) for v in kept.values()])
means = np.array([np.mean(list(v.values())) for v in kept.values()])
print("PAIRED-Delta NOISE  (same claim, same activation, across resamples)")
print(f"  median within-claim sd  (NOISE)                 : {np.median(sds):.5f}")
print(f"  sd of the claim MEANS   (EFFECT, between-claim) : {means.std(ddof=1):.5f}")
print(f"  ==> noise / effect ratio                        : "
      f"{np.median(sds)/max(means.std(ddof=1),1e-12):.2f}x")
print(f"      NOISE-01 (n=6,  regex, 1 position, K=8)     : 1.41x")
print(f"      NOISE-02 (n=22, regex, 10 positions, K=4)   : 0.12x")

full = {g: v for g, v in per.items() if len(v) == 4}
print(f"\nH2 KILL CONDITION - spread of per-claim mean Delta vs K")
print(f"  groups spanning all 4 resamples: {len(full)}   (exact-string gave 5)")
if len(full) >= 20:
    rng = np.random.default_rng(0)
    ks, spreads = [], []
    for K in (1, 2, 4):
        reps = []
        for _ in range(400):
            m = [np.mean(rng.choice(list(v.values()), K, replace=False)) for v in full.values()]
            reps.append(np.std(m, ddof=1))
        ks.append(K); spreads.append(float(np.mean(reps)))
        print(f"    K={K}  sd of claim means = {spreads[-1]:.5f}")
    slope = np.polyfit(np.log(ks), np.log(spreads), 1)[0]
    print(f"\n  log-log slope = {slope:+.3f}")
    print(f"    H2 predicts -0.50 (pure 1/sqrt(K), noise is stochastic and averages away)")
    print(f"    slope ~  0.00     noise is SYSTEMATIC per claim; averaging cannot remove it")
    print(f"  spread at K=4 is {100*spreads[-1]/spreads[0]:.0f}% of K=1  "
          f"(pure 1/sqrt(K) would be {100/math.sqrt(4):.0f}%)")
    frac = 1 - spreads[-1]/spreads[0]
    print(f"\n  VERDICT: {'H2 SUPPORTED - noise averages away as predicted' if slope < -0.35 else ('PARTIAL - some of the noise is irreducible' if slope < -0.15 else 'H2 KILLED - the noise is systematic, averaging does not help')}")
else:
    print("  too few complete groups for the K sweep")

print("\nnoise/effect by claim level")
for lv in ("THEME", "ENTITY", "DETAIL"):
    idx = [i for i, g in enumerate(kept) if grp[g][0]["level"] == lv]
    if len(idx) < 8: print(f"  {lv:7s} n={len(idx):3d}  too few"); continue
    s, m = sds[idx], means[idx]
    print(f"  {lv:7s} n={len(idx):3d}  noise {np.median(s):.5f}  effect {m.std(ddof=1):.5f}  "
          f"ratio {np.median(s)/max(m.std(ddof=1),1e-12):.2f}x")
