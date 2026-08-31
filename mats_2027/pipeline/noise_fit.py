"""H2 variance decomposition, made reproducible.

NOISE-03 (noise_analysis3.py) computed spread at K=1,2,4 only and reported a log-log slope.
The stronger claim used in the write-up - fit spread(K)^2 = signal^2 + noise^2/K on K=1 and K=4
ONLY, then predict K=2 and K=3 out of sample - was done ad hoc in-session and was never committed.
This script reproduces it end to end, and prints every quantity D4 quotes.

Definitions, stated because the write-up has to be able to answer them:
  "spread(K)" = the standard deviation, ACROSS CLAIMS, of each claim's mean Delta when that mean
                is taken over K of its resamples. It is one number per K, describing how spread
                out the claims are from each other - NOT the variation of a single claim.
  Subsampling: for each claim, choose K of its 4 resamples without replacement, average, then take
               the sd across claims; repeat REPS times and average those sds.
"""
import argparse, math
from collections import defaultdict, Counter
import numpy as np, pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--deltas", default="mats_2027/runs/2026-08-23_ar/deltas.parquet")
ap.add_argument("--groups", default="mats_2027/runs/2026-08-22_pos10/claim_groups.parquet")
ap.add_argument("--reps", type=int, default=4000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

D = {r["claim_id"]: r for r in pq.read_table(a.deltas).to_pylist() if r["valid"]}
grp = defaultdict(list)
for r in pq.read_table(a.groups).to_pylist():
    if r["group_id"] is not None and r["claim_id"] in D:
        grp[r["group_id"]].append(D[r["claim_id"]])

# average within a resample first: one explanation can state the same thing twice
per = {}
for g, rows in grp.items():
    d = defaultdict(list)
    for r in rows: d[r["k"]].append(r["delta"])
    per[g] = {k: float(np.mean(v)) for k, v in d.items()}

n_members3 = sum(1 for rows in grp.values() if len(rows) >= 3)
n_span3    = sum(1 for v in per.values() if len(v) >= 3)
full       = {g: v for g, v in per.items() if len(v) == 4}
print(f"groups with >=3 MEMBER CLAIMS      : {n_members3}")
print(f"groups spanning >=3 RESAMPLES      : {n_span3}   <- the one H2 uses")
print(f"groups spanning ALL 4 resamples    : {len(full)}   <- the K sweep runs on these\n")

rng = np.random.default_rng(a.seed)
vals = [list(v.values()) for v in full.values()]
obs = {}
for K in (1, 2, 3, 4):
    reps = [np.std([np.mean(rng.choice(v, K, replace=False)) for v in vals], ddof=1)
            for _ in range(a.reps)]
    obs[K] = float(np.mean(reps))
    print(f"  K={K}  spread of claim means = {obs[K]:.5f}")

# fit on K=1 and K=4 ONLY:  spread^2 = s^2 + n^2/K
n2 = (obs[1]**2 - obs[4]**2) / (1 - 0.25)
s2 = obs[4]**2 - n2/4
noise_fit, signal = math.sqrt(max(n2, 0)), math.sqrt(max(s2, 0))
print(f"\nfit on K=1 and K=4 only:  noise sd = {noise_fit:.5f}   signal sd = {signal:.5f}")
print("out-of-sample predictions (these K were NOT used in the fit):")
for K in (2, 3):
    pred = math.sqrt(s2 + n2/K)
    print(f"  K={K}  predicted {pred:.5f}   observed {obs[K]:.5f}   error {100*(obs[K]-pred)/pred:+.1f}%")

# independent noise estimate: RMS of within-claim sd (NOT the median - see D4 evidence)
sds = np.array([np.std(v, ddof=1) for v in vals])
rms = float(np.sqrt(np.mean(sds**2)))
print(f"\nindependent noise estimate (RMS of within-claim sd) : {rms:.5f}")
print(f"  vs noise from the fit                             : {noise_fit:.5f}")
print(f"  (median within-claim sd = {np.median(sds):.5f} - the WRONG statistic for a variance"
      f" decomposition, it is what NOISE-01/02 used)")
print(f"\nsignal floor: signal sd is {100*signal/obs[1]:.0f}% of the K=1 spread")
print(f"noise/signal: {noise_fit/signal:.2f}x at K=1  ->  {noise_fit/2/signal:.2f}x at K=4")
slope = np.polyfit(np.log([1,2,3,4]), np.log([obs[k] for k in (1,2,3,4)]), 1)[0]
print(f"log-log slope = {slope:+.3f}  (pure 1/sqrt(K) would be -0.500; the floor is why it is not)")
