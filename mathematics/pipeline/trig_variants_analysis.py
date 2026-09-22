#!/usr/bin/env python3
"""TRIG-01b analysis: recall or computation?

Reading rules, fixed before the numbers were seen:

  `int` is the control (TRIG-01 measured ~97% for sin/cos in Q1).

  `over360` and `neg` are DIAGNOSTIC. The correct answer is the same real number as the
  control, so accuracy cannot drop because the value got harder - only because the reduction
  (mod 360) or the symmetry (sin odd, cos even) failed. A drop here localises the failure to
  that step.

  `dec2` is SUGGESTIVE, not diagnostic. sin(37.87 degrees) is essentially untabulated, so a
  collapse is consistent with lookup - but a fractional input may also be genuinely harder to
  compute, and this design cannot separate those two. Read a collapse as "consistent with
  lookup, not proof of it". Read a NON-collapse as strong evidence for computation, because
  there is no plausible table to be reading from.

  `half` sits between: half-degree tables do exist.
"""
import json, os, sys
import pandas as pd, numpy as np

TOL = 5e-6
ORDER = ["int", "half", "dec2", "over360", "neg"]
NOTE = {"int": "control", "half": "d+0.5 (tables exist)", "dec2": "d+0.ab (untabulated)",
        "over360": "d+360, SAME value as control", "neg": "-d, SAME magnitude as control"}

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (100*(c-h), 100*(c+h))

def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    p = os.path.join(d, "results_variants.jsonl")
    df = pd.DataFrame([json.loads(l) for l in open(p)])
    # base 0 'neg' asks "-0" which renders as "0" - degenerate duplicate of the control
    df = df[~((df.variant == "neg") & (df.base_deg == 0))]
    f = df[~df.pole & df.value.notna()].copy()
    f["abs_err"] = (f.value - f.truth).abs()
    f["ok"] = f.abs_err <= TOL
    print(f"n={len(df)} asks, {len(f)} with a finite truth and a parsed number "
          f"(poles {int(df.pole.sum())}, unparsed {int(df.value.isna().sum())})\n")

    print("=== exact to 5 decimal places, by variant ===")
    print(f"{'variant':<9}{'n':>5}{'exact5dp':>10}{'95% CI':>16}{'vs control':>12}   note")
    ctrl = {}
    for v in ORDER:
        g = f[f.variant == v]
        if not len(g): continue
        k, n = int(g.ok.sum()), len(g)
        lo, hi = wilson(k, n)
        if v == "int": ctrl["all"] = k/n
        delta = 100*(k/n - ctrl.get("all", np.nan))
        ds = "  —" if v == "int" else f"{delta:+7.1f} pts"
        print(f"{v:<9}{n:>5}{100*k/n:>9.1f}%{f'[{lo:.1f}, {hi:.1f}]':>16}{ds:>12}   {NOTE[v]}")

    print("\n=== by function (exact-to-5dp %) ===")
    print(f"{'variant':<9}" + "".join(f"{fn:>9}" for fn in ("sin", "cos", "tan")))
    for v in ORDER:
        g = f[f.variant == v]
        if not len(g): continue
        row = "".join(f"{100*g[g.fn==fn].ok.mean():>8.1f}%" if len(g[g.fn==fn]) else f"{'-':>9}"
                      for fn in ("sin", "cos", "tan"))
        print(f"{v:<9}{row}")

    print("\n=== PAIRED on the same base angle: control right, variant wrong? (sin+cos only) ===")
    sc = f[f.fn != "tan"]
    base = sc[sc.variant == "int"].set_index(["fn", "base_deg"]).ok
    for v in ORDER[1:]:
        g = sc[sc.variant == v].set_index(["fn", "base_deg"])
        idx = g.index.intersection(base.index)
        if not len(idx): continue
        b, o = base.loc[idx], g.loc[idx].ok
        both = int((b & o).sum()); lost = int((b & ~o).sum()); gained = int((~b & o).sum())
        print(f"  {v:<9} n={len(idx):4d}   control-right&variant-right {both:4d}   "
              f"LOST {lost:4d}   gained {gained:3d}")

    print("\n=== how wrong, when wrong (median |error|) ===")
    for v in ORDER:
        g = f[(f.variant == v) & (~f.ok)]
        if len(g): print(f"  {v:<9} n_wrong={len(g):4d}  median |err| {g.abs_err.median():.5f}")

    print("\n=== a few dec2 examples (the untabulated case) ===")
    for _, r in f[(f.variant == "dec2") & (f.fn == "sin")].head(8).iterrows():
        mark = "OK " if r.ok else "XX "
        print(f"  {mark} sin({r.asked}) said {r.value:<12} truth {r.truth:.5f}  err {r.abs_err:.5f}")

if __name__ == "__main__":
    main()
