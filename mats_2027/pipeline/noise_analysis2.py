"""NOISE-02 - paired-Delta noise and the 1/sqrt(K) check, on the full SCORE-01 data.

Supersedes noise_analysis.py, which was written for NOISE-01: 48 explanations, ONE token
position, K=8, and 6 recurring claims. This runs on 2065 ablations across 60 positions at
K=4.

TWO CHANGES THAT ARE NOT COSMETIC:

1. RECURRENCE IS MATCHED WITHIN A POSITION, not within a document.
   noise_analysis.py keys on (doc_id, claim_norm) because NOISE-01 had only the last
   position, so doc_id and position were the same thing. Here they are not. The same claim
   at offset -9 and at offset -5 comes from a DIFFERENT ACTIVATION, so its Delta legitimately
   differs - that is signal, not noise. Pooling across positions would inflate the noise
   estimate. Key is (doc_id, pos, claim_norm).

2. Adds the 1/sqrt(K) test, which is H2's KILL CONDITION and has never been run on real data.
   H2: the per-claim signal is noise-limited, not absent, so averaging Delta over K resamples
   should shrink the spread of per-claim means as ~1/sqrt(K) (log-log slope -0.5).
   AG's kill condition, 2026-08-19: "if variance does NOT fall as ~1/sqrt(K), the noise is
   systematic rather than stochastic, averaging cannot rescue it, and the approach is dead."
   A slope near 0 kills H2 - and would be a clean negative result, not a failure.

KNOWN LIMITATION CARRIED OVER: recurrence still uses the norm() regex (lowercase, strip
non-alphanumerics, exact match). SOURCE-01 established the paper used an LLM matcher. This
undercounts recurrence - "mentions Dravid" != "discusses Dravid" - so the claims that survive
are biased toward whatever phrasings Haiku happens to emit verbatim. Deliberately kept
identical to NOISE-01 so the two numbers are comparable.

Input: runs/2026-08-23_ar/deltas.parquet
"""
import argparse, math, re
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq

def norm(t):
    """IDENTICAL to build_ablations.py:28 - kept so NOISE-01 and NOISE-02 are comparable."""
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", default="mats_2027/runs/2026-08-23_ar/deltas.parquet")
    ap.add_argument("--min-k", type=int, default=3, help="resamples a claim must appear in")
    a = ap.parse_args()

    R = [r for r in pq.read_table(a.deltas).to_pylist() if r["valid"]]
    last = defaultdict(int)
    for r in R: last[r["doc_id"]] = max(last[r["doc_id"]], r["pos"])
    for r in R: r["off"] = r["pos"] - last[r["doc_id"]]
    D = np.array([r["delta"] for r in R])
    print(f"{len(R)} valid ablations · {len({(r['doc_id'],r['pos'],r['k']) for r in R})} explanations "
          f"· {len({(r['doc_id'],r['pos']) for r in R})} activations")
    print(f"Delta  mean {D.mean():+.5f}  sd {D.std(ddof=1):.5f}  "
          f"[{D.min():+.4f}, {D.max():+.4f}]   improved {100*(D<0).mean():.1f}%\n")

    # ---- 1. recurrence WITHIN a position ----
    rec = defaultdict(list)
    for r in R: rec[(r["doc_id"], r["pos"], norm(r["claim"]))].append(r)
    sizes = [len(v) for v in rec.values()]
    print(f"distinct (position, claim-text) groups: {len(rec)}")
    print(f"  group size distribution: " +
          "  ".join(f"{n}x{sum(1 for s in sizes if s==n)}" for n in sorted(set(sizes))))
    kept = {k: v for k, v in rec.items() if len(v) >= a.min_k}
    print(f"  groups with >= {a.min_k} resamples: {len(kept)}   "
          f"(NOISE-01 had 6, matched on doc only, at one position)\n")
    if not kept:
        print("no recurring claims - cannot compute the ratio"); return

    sds   = np.array([np.std([x["delta"] for x in v], ddof=1) for v in kept.values()])
    means = np.array([np.mean([x["delta"] for x in v]) for v in kept.values()])
    ratio = np.median(sds) / max(means.std(ddof=1), 1e-12)
    print("PAIRED-Delta NOISE (same claim, same activation, across resamples)")
    print(f"  median within-claim sd (NOISE)                : {np.median(sds):.5f}")
    print(f"  sd of the claim MEANS  (EFFECT, between-claim): {means.std(ddof=1):.5f}")
    print(f"  ==> noise / effect ratio                      : {ratio:.2f}x")
    print(f"      NOISE-01 (n=6, one position, K=8)         : 1.41x")

    # ---- 2. H2 KILL CONDITION: does the spread of claim means fall as 1/sqrt(K)? ----
    print("\nH2 KILL CONDITION - spread of per-claim mean Delta vs K")
    print("  (H2 predicts log-log slope ~ -0.50; a slope near 0 means the noise is")
    print("   SYSTEMATIC, averaging cannot remove it, and H2 is dead)")
    full = {k: v for k, v in rec.items() if len(v) == 4}     # need all 4 to subsample cleanly
    print(f"  claims present in all 4 resamples: {len(full)}")
    if len(full) >= 30:
        rng = np.random.default_rng(0)
        ks, spreads = [], []
        for K in (1, 2, 4):
            reps = []
            for _ in range(200):                              # average over subsamples of size K
                m = [np.mean(rng.choice([x["delta"] for x in v], K, replace=False))
                     for v in full.values()]
                reps.append(np.std(m, ddof=1))
            ks.append(K); spreads.append(float(np.mean(reps)))
            print(f"    K={K}  sd of claim means = {spreads[-1]:.5f}")
        slope = np.polyfit(np.log(ks), np.log(spreads), 1)[0]
        print(f"  log-log slope = {slope:+.3f}   (H2 predicts -0.50, systematic noise -> ~0)")
        # what fraction of the K=1 spread is irreducible?
        irred = spreads[-1] / spreads[0]
        print(f"  spread at K=4 is {100*irred:.0f}% of K=1 "
              f"(pure 1/sqrt(K) would be {100/math.sqrt(4):.0f}%)")
    else:
        print("  too few complete groups for the K sweep")

    # ---- 3. does the ratio depend on position or level? ----
    print("\nnoise/effect by claim level (median within-claim sd / between-claim sd)")
    for lv in ("THEME", "ENTITY", "DETAIL"):
        idx = [i for i, v in enumerate(kept.values()) if v[0]["level"] == lv]
        if len(idx) < 10: print(f"  {lv:7s} n={len(idx):4d}  too few"); continue
        s, m = sds[idx], means[idx]
        print(f"  {lv:7s} n={len(idx):4d}  noise {np.median(s):.5f}  effect {m.std(ddof=1):.5f}  "
              f"ratio {np.median(s)/max(m.std(ddof=1),1e-12):.2f}x")

if __name__ == "__main__":
    main()
