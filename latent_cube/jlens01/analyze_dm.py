#!/usr/bin/env python3
"""Score the directed-modulation positive control (JLENS-01 part dm).

Paper's hit definition (jacobian-lens data/experiments/README.md): a target is a hit if it reaches lens rank 1 (top-1) at
any (layer, position) in the workspace band over the scored span (here: the teacher-forced carrier tokens). The band for
Qwen3.5-4B is not published in the repo, so we report three layer windows: all 31 source layers, a mid band 8-27, and
the last 3 source layers (28-30) where the paper says readouts switch to the next output token. Also top-5.
Usage: python analyze_dm.py dm_trials.jsonl
"""
import json, sys, collections

rows = [json.loads(l) for l in open(sys.argv[1])]
WINDOWS = {"all 0-30": range(0, 31), "mid 8-27": range(8, 28), "late 28-30": range(28, 31)}

def best_rank(r, layers):
    # r["ranks"]: [layer][position][tracked]; 0 = top-1
    vals = [min(min(pos) for pos in r["ranks"][l]) for l in layers if l < len(r["ranks"]) and r["ranks"][l]]
    return min(vals) if vals else None

print(f"{len(rows)} trials")
for kind in ("math", "category"):
    print(f"\n=== {kind} ===")
    for wname, layers in WINDOWS.items():
        agg = collections.defaultdict(lambda: [0, 0, 0])   # group_kind -> [n, top1, top5]
        for r in rows:
            if r["kind"] != kind or not r["tracked"]: continue
            b = best_rank(r, layers)
            if b is None: continue
            g = agg[r["group_kind"]]; g[0] += 1; g[1] += b == 0; g[2] += b < 5
        line = "  ".join(f"{gk}: top1 {v[1]}/{v[0]} ({100*v[1]/v[0]:.0f}%) top5 {100*v[2]/v[0]:.0f}%"
                         for gk, v in sorted(agg.items()))
        print(f"  [{wname}] {line}")
    nt = sum(1 for r in rows if r["kind"] == kind and not r["tracked"])
    if nt: print(f"  ({nt} trials had no single-token target and are excluded)")
