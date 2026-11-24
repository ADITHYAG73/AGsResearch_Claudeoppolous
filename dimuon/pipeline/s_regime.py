"""Tag every D2 step and gradient with the server regime it was computed in (see DRIFT-01).

R1 = before the server change, R2 = after it. Evidence (journal DRIFT-01): an oracle stepped at
19:30:45-52 UTC reproduces D1's r32 gradient bit-for-bit (R1); a Dimuon step at 19:32:47 does
not reproduce D1 (its before-export is at 19:32:38) and matches every later check (R2). Times between are 'U' (uncertain).

The time of a step / oracle step is bracketed by the exports just before and after it
(runs/calls.jsonl; latest occurrence of the export name, i.e. the run that the manifest holds).
R1 if the after-export is <= 19:30:52, R2 if the before-export is >= 19:32:38, else U.

Also: R2 steps get grad_key pointing at an R2 gradient. R2 gradients computed under the old key
suffix _O1 are hard-linked to _O1r2 (same file, new name). Missing R2 init gradients are
computed here (new AdamW client at the init state). Usage: s_regime.py [--compute]
"""
import json
import os
import sys
import numpy as np
import s_d2 as d

R1_END, R2_START = "2026-11-10T19:30:52", "2026-11-10T19:32:38"
m = d.man()
S, G = m["steps"], m["grads"]

last = {}
for line in open(d.tk.LOG):
    r = json.loads(line)
    if r["event"] == "export":
        last[r["name"]] = r["t"][:19]


def regime(t_before, t_after):
    if t_after is not None and t_after <= R1_END:
        return "R1"
    if t_before is not None and t_before >= R2_START:
        return "R2"
    return "U"


for k, s in S.items():
    i = s["index_in_chain"]
    s["t_before"] = last.get(f"{s['chain']}-s{i}")
    s["t_after"] = last.get(f"{s['chain']}-s{i + 1}")
    s["regime"] = regime(s["t_before"], s["t_after"])
for k, g in G.items():
    g["t_before"] = last.get(f"or-{k}-w0")
    g["t_after"] = last.get(f"or-{k}-w1")
    g["regime"] = "R2" if k.endswith("_O1r2") else regime(g["t_before"], g["t_after"])

# R2 gradients stored under _O1: link to _O1r2 names
for k in list(G):
    g = G[k]
    if g["regime"] == "R2" and k.endswith("_O1"):
        k2 = k[:-3] + "_O1r2"
        src, dst = d.OUT / "grads" / f"{k}.npz", d.OUT / "grads" / f"{k2}.npz"
        if not dst.exists():
            os.link(src, dst)
        G[k2] = dict(g, setting="O1r2", linked_from=k)

if "--compute" in sys.argv:
    # R2 gradient at the init state (a new AdamW client = init weights, seed 0) for each rank
    for r in (8, 32):
        sid0 = next(s["before"] for s in S.values() if s["rank"] == r and s["index_in_chain"] == 0
                    and s["client"]["kind"] == "new")
        key = d.gkey(sid0, 0, 1.0, "O1r2")
        if key in G:
            continue
        o = d.ORACLES["O1r2"]
        tc = d.new_client("new", r, optimizer="adamw", tag=f"oracle {key}")
        s0, w0 = d.export_state(tc, f"or-{key}-w0", r)
        assert s0 == sid0
        mm = d.tk.step(tc, d.tk.adamw(o["lr"], eps=o["eps"]), text=0)
        w1 = d.export_tmp(tc, f"or-{key}-w1", r)
        arrs, sq = {}, {"A": 0.0, "B": 0.0}
        for kk in w0:
            dd = w1[kk] - w0[kk]; ad = np.abs(dd)
            gs = d.snap_bf16(-np.sign(dd) * o["eps"] * ad / (o["lr"] - ad))
            p = "A" if ".lora_A." in kk else "B"
            sq[p] += float((gs ** 2).sum())
            arrs[("GA/" if p == "A" else "GB/") + d.short(kk)] = gs.astype(np.float32)
        np.savez(d.OUT / "grads" / f"{key}.npz", **arrs)
        G[key] = dict(sid=sid0, ckpt=None, text=0, weight=1.0, setting="O1r2", **o, fb=mm["fb"],
                      loaded_sid=s0, load_bit_identical=True, GA_global=sq["A"] ** 0.5,
                      GB_global=sq["B"] ** 0.5, G_global=(sq["A"] + sq["B"]) ** 0.5,
                      regime="R2", note="new AdamW client at init (no checkpoint)")
        d.save_man()
        print("computed", key, flush=True)

# point every R2 step at an R2 gradient
for k, s in S.items():
    base = d.gkey(s["before"], s["text"], s["weight"], "O1")
    if s["regime"] == "R2":
        k2 = base[:-3] + "_O1r2"
        s["grad_key"] = k2 if k2 in G else None
    elif s["regime"] == "R1":
        s["grad_key"] = base if (base in G and G[base]["regime"] == "R1") else None
    else:
        s["grad_key"] = None
    s["grad_regime_ok"] = s["grad_key"] is not None
d.save_man()

from collections import Counter
print("steps by regime:", Counter(s["regime"] for s in S.values()))
print("grads by regime:", Counter(g["regime"] for g in G.values()))
print("steps WITH same-regime gradient:", sum(s["grad_regime_ok"] for s in S.values()), "of", len(S))
for k in sorted(S):
    s = S[k]
    if not s["grad_regime_ok"] or s["regime"] == "U":
        print("  no same-regime G:", k, s["regime"], s["t_before"], s["t_after"])
