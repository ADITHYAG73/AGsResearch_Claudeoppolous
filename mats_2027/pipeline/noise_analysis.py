"""NOISE-01 analysis: the paired-Δ noise, read off real data.

Δ(claim, k) = mse(explanation k with claim removed) − mse(explanation k intact)
            = how much reconstruction WORSENS when this claim is deleted (positive = load-bearing)

For each claim that recurs across several resamples k of the SAME activation:
   noise = sd of Δ across those k.
That sd is what SIM-01 assumed to be ~0.8 (in its units). Here we get it in the AR's own
units (mse = 2(1-cos)), alongside the effect size in the same units, so the RATIO is honest.
"""
import json, sys, re
import numpy as np
from collections import defaultdict

rows = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "mats_2027/runs/2026-08-21_noise/ablation_scores.json"))
full = {(r["doc_id"], r["k"]): r["mse"] for r in rows if r["variant"] == "full"}
abl = [r for r in rows if r["variant"] != "full"]
for r in abl: r["delta"] = r["mse"] - full[(r["doc_id"], r["k"])]

print(f"{len(full)} intact explanations · {len(abl)} single-claim ablations\n")

# ---- 1. the intact baseline, and its spread across resamples (the SMOKE-01 quantity) ----
by_doc = defaultdict(list)
for (d, k), m in full.items(): by_doc[d].append(m)
print("INTACT mse per activation (mean ± sd across K=8 resamples):")
for d, ms in by_doc.items(): print(f"  {d[9:26]:17s}  {np.mean(ms):.4f} ± {np.std(ms, ddof=1):.4f}")

# ---- 2. Δ overall ----
D = np.array([r["delta"] for r in abl])
print(f"\nΔ over ALL {len(D)} ablations:  mean {D.mean():+.5f}   sd {D.std(ddof=1):.5f}   "
      f"min {D.min():+.4f}  max {D.max():+.4f}")
print(f"  fraction with Δ>0 (removal HURT reconstruction): {(D>0).mean():.1%}")
print(f"  fraction with Δ<0 (removal IMPROVED it):        {(D<0).mean():.1%}")

# ---- 3. THE NUMBER: paired-Δ noise for recurring claims ----
rec = defaultdict(list)
for r in abl: rec[(r["doc_id"], r["claim_norm"])].append((r["k"], r["delta"], r.get("level")))
print(f"\nPAIRED-Δ NOISE — same claim, across the resamples it appears in  (n≥3):")
print(f"  {'activation':17s} {'lvl':6s} {'n':>2s} {'mean Δ':>9s} {'sd Δ':>8s}  claim")
sds, means = [], []
for (d, cn), lst in sorted(rec.items(), key=lambda x: -len(x[1])):
    if len(lst) < 3: continue
    ds = np.array([x[1] for x in lst]); sds.append(ds.std(ddof=1)); means.append(ds.mean())
    print(f"  {d[9:26]:17s} {lst[0][2] or '':6s} {len(lst):2d} {ds.mean():+9.5f} {ds.std(ddof=1):8.5f}  {cn[:55]}")
if sds:
    print(f"\n  median paired sd : {np.median(sds):.5f}")
    print(f"  spread of claim MEANS (the 'effect' scale, between-claim): {np.std(means, ddof=1):.5f}")
    print(f"  ⇒ noise / effect ratio on THIS data: {np.median(sds)/max(np.std(means, ddof=1),1e-9):.2f}x")

# ---- 4. by level, a first look at whether Δ differs by claim type ----
print("\nΔ by claim level (all ablations):")
byl = defaultdict(list)
for r in abl: byl[r.get("level")].append(r["delta"])
for l, v in byl.items(): print(f"  {l:7s} n={len(v):3d}  mean {np.mean(v):+.5f}  sd {np.std(v, ddof=1):.5f}")
