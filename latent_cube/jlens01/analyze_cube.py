#!/usr/bin/env python3
"""Describe JLENS-01 part cube WITHOUT choosing a hit definition (that is AG's call).

For FORCED captures only (the reply never contains the term), at the forced-reply token positions, take each tracked
token's best (lowest) rank over a layer window and over those positions. Report, per hidden-term power (3 = cube terms,
2 = square terms, ...), the median best rank and the share of captures where the token reaches top-1 / top-5 / top-20.
Same for the logit lens, as the comparison AG asked for on 2026-11-02.
Usage: python analyze_cube.py OUT_DIR [--layers 8-27]
"""
import json, sys, glob, os, statistics, collections
from safetensors.torch import load_file

out = sys.argv[1]
lo, hi = (8, 27)
if "--layers" in sys.argv:
    lo, hi = map(int, sys.argv[sys.argv.index("--layers") + 1].split("-"))
tracked = json.load(open(os.path.join(out, "cube_tracked_ids.json")))
names = list(tracked)
SHOW = ["³", "²", " cube", " cubic", " square", " squared", "6", "2", "3", " six", " three", " two", " third",
        "x", " x", "sin", " sin", " Taylor", " Paris"]
SHOW = [s for s in SHOW if s in tracked]

rows = []
for jf in sorted(glob.glob(os.path.join(out, "cube", "*.json"))):
    meta = json.load(open(jf))
    if meta["variant"] != "forced": continue
    t = load_file(jf[:-5] + ".safetensors")
    P, T = meta["prompt_len"], len(meta["token_ids"])
    rows.append((meta, t["ranks_jlens"][lo:hi + 1, P:T], t["ranks_logitlens"][lo:hi + 1, P:T]))
print(f"{len(rows)} forced captures; layers {lo}-{hi}; positions = forced-reply tokens only")

for lens_name, idx in (("J-lens", 1), ("logit lens", 2)):
    print(f"\n===== {lens_name} =====")
    by_power = collections.defaultdict(list)
    for r in rows: by_power[r[0]["power"]].append(r)
    for power in sorted(by_power, key=lambda p: (p is None, p)):
        grp = by_power[power]
        print(f"\n-- hidden term power {power}: {len(grp)} captures --")
        for s in SHOW:
            k = names.index(s)
            best = [int(r[idx][:, :, k].min()) for r in grp]
            print(f"  {s!r:12} median best rank {statistics.median(best):>8.0f}   top1 {sum(b == 0 for b in best):>3}"
                  f"   top5 {sum(b < 5 for b in best):>3}   top20 {sum(b < 20 for b in best):>3}   of {len(grp)}")

# the pair AG's question is about
print("\n===== sin x vs cos x, 2nd term, forced, per template (best rank over layers & reply positions) =====")
for r in rows:
    m = r[0]
    if m["set"] == "taylor" and m["kth"] == 2 and m["func"] in ("sin x", "cos x"):
        cells = "  ".join(f"{s!r}:J{int(r[1][:, :, names.index(s)].min())}/L{int(r[2][:, :, names.index(s)].min())}"
                          for s in ["³", "²", " cube", "6", "2", " square"] if s in tracked)
        print(f"  {m['id']:24} {cells}")
