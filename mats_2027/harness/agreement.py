"""Judge validation: AG's blind labels vs Haiku's, and AG vs himself.

The number that matters is NOT overall agreement - it is whether the judge's error rate
DIFFERS BY CLAIM LEVEL. The specificity result (THEME 69.1 / ENTITY 43.8 / DETAIL 36.0) is a
gradient across levels. If judge error also varies by level, signal and label share a common
cause and the gradient is partly an artefact. That is a confound, not noise.

Self-consistency on the 30 retests bounds the whole exercise: AG's own labels cannot be more
reliable than he is with himself, and the judge cannot be shown to be worse than that bound.
"""
import json, math
from collections import Counter, defaultdict
import pyarrow.parquet as pq

V = [json.loads(l) for l in open('mats_2027/harness/verdicts_AG_MAIN.jsonl')]
H = {r['claim_id']: r for r in pq.read_table('mats_2027/runs/2026-08-22_pos10/verdicts.parquet').to_pylist()}
C = {c['claim_id']: c for c in pq.read_table('mats_2027/runs/2026-08-22_pos10/claims.parquet').to_pylist()}

main   = {v['claim_id']: v['verdict'] for v in V if not v['claim_id'].startswith('RETEST::')}
retest = {v['claim_id'][8:]: v['verdict'] for v in V if v['claim_id'].startswith('RETEST::')}

def wilson(k, n):
    if n == 0: return (0, 0)
    p = k/n; z = 1.96; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d; h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (100*(c-h), 100*(c+h))

print("="*74); print("1. AG vs AG  -  self-consistency on the 30 retests"); print("="*74)
same = sum(1 for cid, v in retest.items() if main.get(cid) == v)
lo, hi = wilson(same, len(retest))
print(f"  same label both times: {same}/{len(retest)} = {100*same/len(retest):.1f}%   95% CI [{lo:.0f}, {hi:.0f}]")
for cid, v in retest.items():
    if main.get(cid) != v:
        print(f"    FLIP  {main.get(cid):12s} -> {v:12s}  [{C[cid]['level']}/{C[cid]['subtype']}] {C[cid]['text'][:56]}")

print()
print("="*74); print("2. AG vs HAIKU  -  overall"); print("="*74)
agree = sum(1 for cid, v in main.items() if H[cid]['verdict'] == v)
lo, hi = wilson(agree, len(main))
print(f"  agreement: {agree}/{len(main)} = {100*agree/len(main):.1f}%   95% CI [{lo:.0f}, {hi:.0f}]")
print(f"  (bounded above by AG's own {100*same/len(retest):.0f}% self-consistency)")
print("\n  confusion  (rows = AG, cols = Haiku)")
labs = ("SUPPORTED", "CONTRADICTED", "NOT_IN_TEXT")
print(f"  {'':14s}" + "".join(f"{l[:9]:>14s}" for l in labs))
for a in labs:
    row = f"  {a:14s}"
    for hh in labs:
        row += f"{sum(1 for cid,v in main.items() if v==a and H[cid]['verdict']==hh):14d}"
    print(row)

print()
print("="*74); print("3. THE CONFOUND CHECK  -  does agreement differ BY CLAIM LEVEL?"); print("="*74)
print(f"  {'level':8s} {'n':>4s} {'agree':>7s} {'%':>7s}  {'95% CI':>14s}   AG supported%   Haiku supported%")
rates = {}
for lv in ("THEME", "ENTITY", "DETAIL"):
    ids = [cid for cid in main if C[cid]['level'] == lv]
    a = sum(1 for cid in ids if H[cid]['verdict'] == main[cid])
    lo, hi = wilson(a, len(ids)); rates[lv] = (a, len(ids), lo, hi)
    ag_s = 100*sum(1 for cid in ids if main[cid]=='SUPPORTED')/len(ids)
    hk_s = 100*sum(1 for cid in ids if H[cid]['verdict']=='SUPPORTED')/len(ids)
    print(f"  {lv:8s} {len(ids):4d} {a:7d} {100*a/len(ids):6.1f}%  [{lo:5.0f},{hi:5.0f}]   "
          f"{ag_s:11.1f}%   {hk_s:13.1f}%")
ks = list(rates)
print("\n  pairwise difference in agreement rate (two-proportion z):")
for i in range(len(ks)):
    for j in range(i+1, len(ks)):
        a1,n1,_,_ = rates[ks[i]]; a2,n2,_,_ = rates[ks[j]]
        p1,p2 = a1/n1, a2/n2; p = (a1+a2)/(n1+n2)
        se = math.sqrt(p*(1-p)*(1/n1+1/n2)); z = (p1-p2)/se if se else 0
        print(f"    {ks[i]:7s} vs {ks[j]:7s}  {100*(p1-p2):+6.1f}pp   z = {z:+.2f}   "
              f"{'DIFFERENT' if abs(z)>1.96 else 'not distinguishable'}")

print()
print("="*74); print("4. WHERE THEY DISAGREE"); print("="*74)
dis = [(cid, main[cid], H[cid]['verdict']) for cid in main if main[cid] != H[cid]['verdict']]
print(f"  {len(dis)} disagreements. by direction:")
for (a,h),n in Counter((d[1],d[2]) for d in dis).most_common():
    print(f"    AG {a:12s} / Haiku {h:12s}  {n:3d}")
print("\n  by claim subtype (top 6):")
for st,n in Counter(C[d[0]]['subtype'] for d in dis).most_common(6):
    tot = sum(1 for cid in main if C[cid]['subtype']==st)
    print(f"    {st:14s} {n:3d} of {tot:3d}  ({100*n/tot:.0f}% of that subtype)")
