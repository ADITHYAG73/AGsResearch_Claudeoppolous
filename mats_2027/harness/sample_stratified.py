"""Draw a stratified subset of claims for blind grading.

Design (DECOMP-01 -> Stage 4):
  PRIMARY   stratify on `level` — equal n per level, because level is the CONFOUND we are
            testing for (the paper's specificity baseline is THEME 64% / ENTITY 28% /
            DETAIL 24% true; if judge error also varies by level, signal and label share a
            common cause). Equal n = equal power per level. NOTE this is deliberately NOT
            proportional to the population (887/427/751) — re-weight when a population-wide
            rate is wanted.
  BALANCE   spread evenly over the 10 position offsets, so position cannot confound the
            level comparison. This is balancing, NOT powering: n/level/offset is too small
            for a per-offset estimate.
  NOT BAL.  passage — cannot be balanced (Test cricket has only 9 ENTITY claims vs 122 for
            Laxman). Sampled at random within cell; realised spread is printed so any
            imbalance is visible rather than hidden.

Output is a claims parquet with the SAME schema, to be fed to prepare_grading.py unchanged
(that script keeps its single responsibility: strip everything the grader must not see).
"""
import argparse, json, random
from collections import defaultdict, Counter
import pyarrow as pa, pyarrow.parquet as pq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-level", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--exclude", default=None, help="JSON list of claim_ids to leave out")
    a = ap.parse_args()

    C = pq.read_table(a.claims).to_pylist()
    drop = set(json.load(open(a.exclude))) if a.exclude else set()
    if drop: C = [c for c in C if c["claim_id"] not in drop]

    last = defaultdict(int)
    for c in C: last[c["doc_id"]] = max(last[c["doc_id"]], c["pos"])
    for c in C: c["_off"] = c["pos"] - last[c["doc_id"]]

    offsets = sorted({c["_off"] for c in C})
    rng = random.Random(a.seed)
    picked = []
    for lv in ("THEME", "ENTITY", "DETAIL"):
        per_off = a.per_level // len(offsets)
        rem = a.per_level - per_off * len(offsets)
        for i, off in enumerate(offsets):
            cell = [c for c in C if c["level"] == lv and c["_off"] == off]
            want = per_off + (1 if i < rem else 0)
            if len(cell) < want:
                print(f"  !! cell ({lv},{off:+d}) has {len(cell)} < {want} requested")
                want = len(cell)
            picked += rng.sample(cell, want)

    for c in picked: c.pop("_off", None)
    rng.shuffle(picked)
    pq.write_table(pa.Table.from_pylist(picked), a.out)

    lev = Counter(c["level"] for c in picked)
    off = Counter(c["pos"] - last[c["doc_id"]] for c in picked)
    doc = Counter(c["doc_id"] for c in picked)
    print(f"\n{len(picked)} claims -> {a.out}")
    print("  by level :", dict(lev))
    print("  by offset:", {k: off[k] for k in sorted(off)})
    print("  by passage (NOT balanced, reported for transparency):")
    for d, n in doc.most_common(): print(f"      {n:4d}  {d}")

if __name__ == "__main__":
    main()
