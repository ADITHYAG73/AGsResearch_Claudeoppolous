"""PATCH-01's control, made reproducible: did the text edit change the ACTIVATION at all?

Two things matter and both were got wrong at least once:
1. ALIGNMENT. The edits change token counts, so absolute position index is NOT comparable across
   conditions. PATCH-01 sampled the LAST TEN positions, so conditions are aligned by OFFSET FROM
   THE END (-9..0). Comparing by absolute pos gives nonsense (REAL appears to move to cos 0.26).
2. CENTRING. Raw cosine in a residual stream is uninformative - everything is ~0.96 to everything.
   These are MEAN-CENTRED cosines, and the centring mean is stated below, because the number moves
   if you centre on a different pool.
"""
import numpy as np, pyarrow.parquet as pq

PATCH = "mats_2027/runs/2026-08-27_patch/activations.parquet"
POS10 = "mats_2027/runs/2026-08-22_pos10/activations.parquet"
A = pq.read_table(PATCH).to_pandas(); P = pq.read_table(POS10).to_pandas()
cond = {}
for r in A.itertuples(): cond.setdefault(r.doc_id, {})[r.pos] = np.array(r.activation_vector, float)
other = {(r.doc_id, r.pos): np.array(r.activation_vector, float) for r in P.itertuples()}

# CENTRING: the mean of the 60 POS-01 activations ONLY - deliberately NOT the patch data, so the
# centring mean is not fitted to the vectors being compared. (Pooling all 130 instead shifts the
# one-token-step reference from 0.42 to 0.25: the number is sensitive to this choice, so it is
# stated rather than assumed.)
pool = np.stack(list(other.values()))
mu = pool.mean(0)
print(f"centred on the mean of {len(pool)} POS-01 activations (NOT the patch data)\n")

def cc(a, b):
    a, b = a - mu, b - mu
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

# align by offset from the end within each condition
off = {d: {i - len(m) + 1: m[p] for i, p in enumerate(sorted(m))} for d, m in cond.items()}
base = off["PATCH::ORIGINAL"]
print("edited vs ORIGINAL at the same offset from the end:")
worst = 1.0
for c in ("FAMOUS", "REAL", "INVENTED", "NEUTRAL", "DELETE", "COHERENT"):
    m = off["PATCH::" + c]; ks = [k for k in base if k in m]
    cs = [cc(base[k], m[k]) for k in ks]
    worst = min(worst, min(cs))
    print(f"  {c:9s} n={len(cs):2d}  mean {np.mean(cs):.4f}   min {min(cs):.4f}")

ks = sorted(base)
step = [cc(base[a], base[b]) for a, b in zip(ks, ks[1:])]
diff = [cc(base[k], v) for k in ks for (d, p), v in list(other.items())[:50] if d != "CRICKET::Rahul Dravid"]
print(f"\nreference scales on the same axis:")
print(f"  ONE TOKEN STEP, same passage : mean {np.mean(step):.3f}   range [{min(step):.3f}, {max(step):.3f}]")
print(f"  A DIFFERENT PASSAGE          : mean {np.mean(diff):.3f}   range [{min(diff):.3f}, {max(diff):.3f}]")
print(f"\nthe edit moves the activation {(1-worst)/(1-np.mean(step))*100:.1f}% as far as one token step does"
      f"  (worst case; using the mean edit it is {(1-0.9961)/(1-np.mean(step))*100:.1f}%)")
