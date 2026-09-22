#!/usr/bin/env python3
"""TRIG-01 analysis. Questions fixed BEFORE the numbers were seen.

AG's framing: he can do 0/30/45/60/90 from memory and get the sign right anywhere by ASTC,
but "I can never do this in between things like 37 degree." So the analysis splits exactly
along that line - the angles a person recalls versus the angles a person must compute.

Q1 Does it answer at all, and in the format it was told to use?
Q2 How accurate, at what tolerance?
Q3 EXACT-VALUE angles (deg mod 90 in {0,30,45,60}) vs the rest. This is the whole question:
   recall and computation look identical from the outside, and only this split separates them.
Q4 Sign (ASTC). Getting the sign right is a cheap rule; getting the magnitude right is not.
   If sign accuracy is high while magnitude accuracy is low, the model has the rule and not
   the function.
Q5 The poles - tan(90), tan(270). Undefined. What it says there is informative either way.
Q6 Thinking on vs off, on the shared 5-degree grid, same angles both ways.
"""
import json, math, sys, os
import numpy as np
import pandas as pd

def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return pd.DataFrame(rows)

EXACT = {0, 30, 45, 60}                      # mod 90 -> AG's memorised set, all four quadrants
def is_exact(d): return (d % 90) in EXACT

def sign_of(x, tol=1e-9):
    return 0 if abs(x) < tol else (1 if x > 0 else -1)

def report(df, label, max_new=None):
    print(f"\n{'='*72}\n{label}   n={len(df)}\n{'='*72}")
    npole = df["pole"].sum()
    fin = df[~df["pole"]].copy()                       # finite-truth rows only
    print(f"asked {len(df)}   finite {len(fin)}   poles {npole}")

    # TRUNCATION. A generation that hit the token budget never finished its answer, so the
    # parsed number is whatever fragment was mid-sentence - not the model's answer. Seen live:
    # tan(95) reasoned correctly to -11.430052, ran out of budget, and emitted 3.05e-09.
    # Scoring that as a maths error would be wrong. Reported separately, then EXCLUDED.
    if max_new is not None and "n_gen_tokens" in fin:
        trunc = fin["n_gen_tokens"] >= max_new
        if trunc.any():
            print(f"\n*** TRUNCATED at the {max_new}-token budget: {trunc.sum()}/{len(fin)} "
                  f"({100*trunc.mean():.1f}%) - excluded from accuracy, listed below ***")
            for _, r in fin[trunc].head(6).iterrows():
                print(f"      {r['fn']}({r['deg']}) truth {r['truth']:.5f}  emitted {r['value']}")
            fin = fin[~trunc].copy()

    # Q1 parse + format
    parsed = fin["value"].notna()
    print(f"\nQ1  parsed a number        {parsed.sum()}/{len(fin)}  ({100*parsed.mean():.1f}%)")
    print(f"    exactly 5 decimals     {fin['format_ok'].sum()}/{len(fin)}  ({100*fin['format_ok'].mean():.1f}%)")

    fin = fin[parsed].copy()
    if not len(fin):
        print("    nothing parsed - stopping here"); return None
    fin["abs_err"] = (fin["value"] - fin["truth"]).abs()

    # Q2 accuracy at tolerances
    print(f"\nQ2  accuracy (n={len(fin)})")
    for t, name in [(5e-6, "exact to 5dp"), (1e-4, "within 1e-4"), (1e-2, "within 1e-2"),
                    (1e-1, "within 0.1")]:
        m = (fin["abs_err"] <= t).mean()
        print(f"    {name:<14} {100*m:5.1f}%")
    print(f"    median abs err {fin['abs_err'].median():.6f}   mean {fin['abs_err'].mean():.6f}")

    # Q3 the split that matters
    fin["exact_angle"] = fin["deg"].map(is_exact)
    print(f"\nQ3  MEMORISED vs COMPUTED angles      (tolerance: exact to 5dp)")
    print(f"    {'':<10}{'n':>5}{'exact-5dp':>12}{'within 1e-2':>14}{'median err':>13}")
    for flag, name in [(True, "memorised"), (False, "in-between")]:
        g = fin[fin["exact_angle"] == flag]
        if not len(g): continue
        print(f"    {name:<10}{len(g):>5}{100*(g['abs_err']<=5e-6).mean():>11.1f}%"
              f"{100*(g['abs_err']<=1e-2).mean():>13.1f}%{g['abs_err'].median():>13.6f}")

    # per function
    print(f"\n    by function (exact-to-5dp %):")
    for fn in ("sin", "cos", "tan"):
        g = fin[fin["fn"] == fn]
        if not len(g): continue
        gm, gi = g[g["exact_angle"]], g[~g["exact_angle"]]
        print(f"      {fn}: all {100*(g['abs_err']<=5e-6).mean():5.1f}%   "
              f"memorised {100*(gm['abs_err']<=5e-6).mean() if len(gm) else float('nan'):5.1f}%   "
              f"in-between {100*(gi['abs_err']<=5e-6).mean() if len(gi) else float('nan'):5.1f}%")

    # Q4 sign vs magnitude
    fin["sign_ok"] = [sign_of(v) == sign_of(t) for v, t in zip(fin["value"], fin["truth"])]
    print(f"\nQ4  SIGN correct (ASTC)    {100*fin['sign_ok'].mean():.1f}%"
          f"     vs magnitude exact-to-5dp {100*(fin['abs_err']<=5e-6).mean():.1f}%")
    for fn in ("sin", "cos", "tan"):
        g = fin[fin["fn"] == fn]
        if len(g): print(f"      {fn}: sign {100*g['sign_ok'].mean():5.1f}%")

    return fin

def poles(df):
    p = df[df["pole"]]
    if not len(p): return
    print(f"\nQ5  POLES - tan(90) and tan(270), undefined")
    for _, r in p.iterrows():
        print(f"    tan({r['deg']:>3}) -> {r['raw']!r}")

def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    off_p = os.path.join(d, "results_off_step1.jsonl")
    on_p  = os.path.join(d, "results_on_step5.jsonl")

    off = load(off_p) if os.path.exists(off_p) else None
    on  = load(on_p)  if os.path.exists(on_p)  else None

    fin_off = report(off, "CONDITION A - thinking OFF (one forward pass, no scratchpad)",
                     max_new=24) if off is not None else None
    if off is not None: poles(off)
    fin_on = report(on, "CONDITION B - thinking ON (chain of thought allowed)",
                    max_new=2048) if on is not None else None
    if on is not None: poles(on)

    # Q6 paired comparison on the shared grid - same angles, both conditions
    if fin_off is not None and fin_on is not None:
        shared = set(zip(fin_on["fn"], fin_on["deg"])) & set(zip(fin_off["fn"], fin_off["deg"]))
        a = fin_off.set_index(["fn", "deg"]).loc[list(shared)]
        b = fin_on.set_index(["fn", "deg"]).loc[list(shared)]
        print(f"\n{'='*72}\nQ6  PAIRED: same {len(shared)} asks, thinking off vs on\n{'='*72}")
        print(f"    exact to 5dp   off {100*(a['abs_err']<=5e-6).mean():5.1f}%   on {100*(b['abs_err']<=5e-6).mean():5.1f}%")
        print(f"    within 1e-2    off {100*(a['abs_err']<=1e-2).mean():5.1f}%   on {100*(b['abs_err']<=1e-2).mean():5.1f}%")
        print(f"    median err     off {a['abs_err'].median():.6f}   on {b['abs_err'].median():.6f}")
        if "n_gen_tokens" in b: print(f"    median tokens  off {a['n_gen_tokens'].median():.0f}   on {b['n_gen_tokens'].median():.0f}")

    # figure: error against angle
    if fin_off is not None:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        for ax, fn in zip(axes, ("sin", "cos", "tan")):
            g = fin_off[fin_off["fn"] == fn].sort_values("deg")
            ax.semilogy(g["deg"], g["abs_err"].clip(lower=1e-7), ".", ms=4, label="error")
            gm = g[g["exact_angle"]]
            ax.semilogy(gm["deg"], gm["abs_err"].clip(lower=1e-7), "o", ms=8, mfc="none",
                        mec="crimson", label="memorised angle")
            ax.axhline(5e-6, ls="--", lw=.8, color="k")
            ax.set_ylabel(f"|error|  {fn}")
            ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("degrees"); axes[0].set_title("TRIG-01  Qwen3.8-27B, thinking off - absolute error vs angle")
        fig.tight_layout(); fig.savefig(os.path.join(d, "trig_error_vs_angle.png"), dpi=140)
        print(f"\nfigure -> {os.path.join(d, 'trig_error_vs_angle.png')}")

if __name__ == "__main__":
    main()
