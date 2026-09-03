"""Draw a RANDOM sample of graded claims for the write-up.

Neel's admissions doc: "include some randomly selected qualitative examples in the write-up,
ideally just after the executive summary. Randomly selected, not cherry-picked!"

So: uniform sample over every valid ablation, fixed seed, no filtering of any kind.
Whatever comes out goes in — including boring ones.
"""
import argparse, numpy as np, pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--seed", type=int, default=20260903)
a = ap.parse_args()

d = pq.read_table("mats_2027/runs/2026-08-23_ar/deltas.parquet").to_pandas()
d = d[d.valid == True].reset_index(drop=True)
A = {(r["doc_id"], r["pos"]): r["detokenized_text_truncated"]
     for r in pq.read_table("mats_2027/runs/2026-08-22_pos10/activations.parquet").to_pylist()}

rng = np.random.default_rng(a.seed)
idx = rng.choice(len(d), size=a.n, replace=False)
print(f"# {a.n} claims drawn uniformly from all {len(d)} valid ablations, seed {a.seed}, no filtering\n")
for k, i in enumerate(sorted(idx), 1):
    r = d.iloc[i]
    prefix = A[(r.doc_id, r.pos)]
    print(f"[{k}] {r.doc_id} · position {r.pos} · resample k={r.k}")
    print(f"    prefix ends: ...{prefix[-110:].strip()}")
    print(f"    claim ({r.level}/{r.subtype}): {r.claim}")
    print(f"    judge: {r.verdict}   Δ = {r.delta:+.5f}")
    print()
