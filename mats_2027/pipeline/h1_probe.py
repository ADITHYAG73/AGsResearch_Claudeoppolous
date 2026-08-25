"""H1 - EXPLORATORY K=1 PROBE.  NOT the pre-registered test.

The rule frozen on 2026-08-25 specifies per-claim Delta AVERAGED OVER K=4, which requires the
semantic matcher (parked, MATCH-01). This probe instead treats every claim INSTANCE as its own
data point - K=1, no matching, no averaging.

That makes it a HARDER test than the pre-registered one, because the resample noise is not
averaged down. So it is asymmetric and must be read that way:
  detects two lumps -> a real positive, STRONGER than the K=4 test would give
  finds nothing     -> UNINFORMATIVE. SIM-02 showed K=1 is underpowered; a null says
                       something about K=1, not about H1.

Decision rule applied unchanged from the frozen pre-registration:
    dBIC(2 vs 1) > 10   is the verdict
    Hartigan's dip      reported alongside, DESCRIPTIVE only, not a gate
"""
import sys, os
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import dip_pvalue, two_vs_one_gaussian, ascii_hist

D = {r["claim_id"]: r for r in pq.read_table("mats_2027/runs/2026-08-23_ar/deltas.parquet").to_pylist()}
R = {r["claim_id"]: r for r in pq.read_table("mats_2027/runs/2026-08-22_pos10/relatedness.parquet").to_pylist()}

cats = {"true": [], "related_false": [], "unrelated_false": []}
for cid, d in D.items():
    if not d["valid"]: continue
    if d["verdict"] == "SUPPORTED":
        cats["true"].append(d["delta"])
    elif cid in R:
        key = "related_false" if R[cid]["relatedness"] == "RELATED" else "unrelated_false"
        cats[key].append(d["delta"])

print("H1 EXPLORATORY K=1 PROBE  (NOT the pre-registered K=4 test)\n")
print(f"{'category':17s} {'n':>5s} {'mean Delta':>11s} {'sd':>9s} {'dBIC(2v1)':>10s} "
      f"{'dip':>7s} {'dip_p':>7s}  verdict")
res = {}
for k in ("true", "related_false", "unrelated_false"):
    x = np.asarray(cats[k], float)
    if len(x) < 20:
        print(f"{k:17s} {len(x):5d}   too few"); continue
    dbic, fit = two_vs_one_gaussian(x)
    dip, p = dip_pvalue(x)
    verdict = "TWO LUMPS" if dbic > 10 else "one lump"
    res[k] = (x, dbic, dip, p, fit)
    print(f"{k:17s} {len(x):5d} {x.mean():+11.5f} {x.std(ddof=1):9.5f} {dbic:+10.1f} "
          f"{dip:7.4f} {p:7.3f}  {verdict}")

x, dbic, dip, p, fit = res["related_false"]
print(f"\n--- RELATED-FALSE, the cell H1 is about  (n={len(x)}) ---")
print(f"  FROZEN RULE: dBIC > 10  ->  dBIC = {dbic:+.1f}  ->  "
      f"{'TWO POPULATIONS' if dbic > 10 else 'ONE POPULATION'}")
print(f"  descriptive: Hartigan dip = {dip:.4f}, p = {p:.3f} "
      f"({'valley detected' if p < 0.05 else 'no valley'})")
print(f"  2-component fit: means {fit['means']}  sds {fit['sds']}  weights {fit['weights']}")
print(ascii_hist(x, title="related-false Delta, K=1"))

print("\nREADING THIS:")
if dbic > 10:
    print("  Positive at K=1 is STRONGER than the pre-registered K=4 test would give,")
    print("  because K=1 carries the full resample noise. But dBIC alone can be fooled by a")
    print("  single SKEWED hump, and real Delta IS right-skewed (70% positive). SIM-02's")
    print("  skewed-null control was run at K=4/K=8, NOT at K=1 - so this needs a K=1 skewed")
    print("  null before it can be believed.")
else:
    print("  Null at K=1 is UNINFORMATIVE. SIM-02 showed K=1 detects the planted mixture")
    print("  0-2 times out of 40. This says K=1 is underpowered, not that H1 is false.")
    print("  The pre-registered K=4 test still requires the matcher.")
