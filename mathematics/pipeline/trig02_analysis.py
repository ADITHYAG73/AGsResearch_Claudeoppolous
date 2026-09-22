#!/usr/bin/env python3
"""TRIG-02: does the scratchpad make it a calculator?

Paired against TRIG-01/01b, which ran the SAME angles with thinking off. Every comparison here
is within-angle, so nothing turns on the two runs having sampled comparable inputs.

Reading rules, fixed before the numbers were seen:
  dec2      up   -> a real calculator exists; the scratchpad supplies precision the forward pass
                    cannot. stays ~3% -> chain of thought adds no precision.
  over360   up   -> the reduction step exists but has to be written down.
  tabulation gradient FLATTENS -> the scratchpad substitutes computation for recall. This is the
                    sharpest of the four, because the gradient is the strongest single piece of
                    evidence that the 5-decimal answers are retrieved.

Truncated generations (hit the token budget) are reported and EXCLUDED: a run that stopped
mid-verification did not give an answer, and scoring it as a maths error would be wrong.
"""
import json, math, os, sys
import numpy as np, pandas as pd

TOL, CLOSE, MAXNEW = 5e-6, 1e-3, 2048

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (100*(c-h), 100*(c+h))

def mcnemar(b, c):
    """exact-ish two-sided binomial test on the discordant pairs (b = off-right/on-wrong)."""
    n = b + c
    if n == 0: return 1.0
    from math import comb
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k+1)) / (2**n) * 2
    return min(1.0, p)

def load(p):
    return pd.DataFrame([json.loads(l) for l in open(p)]) if os.path.exists(p) else None

def prep(df, thinking):
    if df is None: return None
    d = df.copy()
    if thinking and "n_gen_tokens" in d:
        d["trunc"] = d["n_gen_tokens"] >= MAXNEW
    else:
        d["trunc"] = False
    d["usable"] = (~d["pole"]) & d["value"].notna() & d["truth"].notna() & (~d["trunc"])
    d["err"] = (d["value"] - d["truth"]).abs()
    d["ok"] = d["usable"] & (d["err"] <= TOL)
    d["close"] = d["usable"] & (d["err"] <= CLOSE)
    return d

def main():
    d02 = sys.argv[1] if len(sys.argv) > 1 else "."
    d01 = sys.argv[2] if len(sys.argv) > 2 else "../2026-09-22_trig"

    von = prep(load(os.path.join(d02, "results_var_think.jsonl")), True)
    voff = prep(load(os.path.join(d01, "results_variants.jsonl")), False)
    ton = prep(load(os.path.join(d02, "results_on_tabsample.jsonl")), True)
    toff = prep(load(os.path.join(d01, "results_off_step1.jsonl")), False)

    if von is not None:
        tr = von["trunc"].sum()
        print(f"PART 1 - variants, thinking ON: {len(von)} asks, {tr} truncated at {MAXNEW} tokens "
              f"({100*tr/len(von):.1f}%), excluded from accuracy")
        if tr:
            for _, r in von[von["trunc"]].head(5).iterrows():
                print(f"    {r['fn']}({r['asked']}) truth "
                      f"{('%.5f'%r['truth']) if r['truth']==r['truth'] else 'pole'} emitted {r['value']}")
        print(f"\n{'variant':<9}{'n':>5}{'ON exact5dp':>14}{'95% CI':>16}"
              f"{'OFF (paired)':>14}{'change':>10}{'McNemar p':>11}")
        for v in ["int", "half", "dec2", "over360", "neg"]:
            g = von[(von["variant"] == v) & von["usable"]]
            if not len(g): continue
            key = ["fn", "variant", "asked"]
            o = voff[voff["variant"] == v].set_index(key)
            gi = g.set_index(key)
            idx = gi.index.intersection(o.index)
            on_ok = gi.loc[idx, "ok"]; off_ok = o.loc[idx, "ok"]
            k, n = int(on_ok.sum()), len(on_ok)
            lo, hi = wilson(k, n)
            b = int((off_ok & ~on_ok).sum()); c = int((~off_ok & on_ok).sum())
            offp = 100*off_ok.mean()
            print(f"{v:<9}{n:>5}{100*k/n:>13.1f}%{f'[{lo:.1f}, {hi:.1f}]':>16}"
                  f"{offp:>13.1f}%{100*k/n-offp:>+9.1f}{mcnemar(b,c):>11.4f}")
        print("\n  (McNemar uses only the angles where the two conditions DISAGREE;"
              " 'change' is in percentage points)")

        print(f"\n{'variant':<9}{'ON <=1e-3':>12}{'OFF <=1e-3':>12}   how close it gets when not exact")
        for v in ["int", "half", "dec2", "over360", "neg"]:
            g = von[(von["variant"] == v) & von["usable"]]
            o = voff[(voff["variant"] == v) & voff["usable"]]
            if not len(g): continue
            print(f"{v:<9}{100*g['close'].mean():>11.1f}%{100*o['close'].mean():>11.1f}%")

    if ton is not None and toff is not None:
        tr = ton["trunc"].sum()
        print(f"\n\nPART 2 - tabulation gradient, thinking ON: {len(ton)} asks, {tr} truncated")
        print("If the scratchpad COMPUTES, this gradient should flatten.\n")
        cls = [("multiple of 15", lambda d: d % 15 == 0),
               ("multiple of 5",  lambda d: d % 5 == 0 and d % 15 != 0),
               ("even",           lambda d: d % 2 == 0 and d % 5 != 0),
               ("odd",            lambda d: d % 2 == 1 and d % 5 != 0)]
        print(f"{'angle class':<17}{'n':>4}{'ON exact5dp':>14}{'95% CI':>16}{'OFF (1° sweep)':>17}")
        onv, offv = [], []
        for name, test in cls:
            g = ton[ton["deg"].map(test) & ton["usable"]]
            o = toff[toff["deg"].map(test) & toff["usable"]]
            if not len(g): continue
            k, n = int(g["ok"].sum()), len(g)
            lo, hi = wilson(k, n)
            offp = 100*o["ok"].mean()
            onv.append(100*k/n); offv.append(offp)
            print(f"{name:<17}{n:>4}{100*k/n:>13.1f}%{f'[{lo:.1f}, {hi:.1f}]':>16}{offp:>16.1f}%")
        if len(onv) == 4:
            print(f"\n  spread (best class - worst class):  thinking ON {max(onv)-min(onv):.1f} pts"
                  f"   vs thinking OFF {max(offv)-min(offv):.1f} pts")
            print("  -> a much smaller ON spread means the scratchpad replaced recall with computation.")

        m5 = ton[ton["deg"].map(lambda d: d % 5 == 0) & ton["usable"]]
        n5 = ton[ton["deg"].map(lambda d: d % 5 != 0) & ton["usable"]]
        if len(m5) and len(n5):
            k1, n1 = int(m5["ok"].sum()), len(m5); k2, n2 = int(n5["ok"].sum()), len(n5)
            p = (k1+k2)/(n1+n2); se = np.sqrt(p*(1-p)*(1/n1+1/n2))
            z = (k1/n1 - k2/n2)/se if se > 0 else 0.0
            print(f"\n  multiples of 5 {100*k1/n1:.1f}% (n={n1}) vs rest {100*k2/n2:.1f}% (n={n2}),"
                  f"  z = {z:.2f}   [thinking OFF was 87.9 vs 64.2, z = 6.68]")

    # what method does it use on the inputs it could not do without a scratchpad?
    if von is not None:
        print("\n\nTRACES on the two diagnostic variants (what method does it reach for?)")
        for v in ("dec2", "over360"):
            g = von[(von["variant"] == v) & von["usable"]]
            if not len(g): continue
            got = g[g["ok"]]
            r = (got.iloc[0] if len(got) else g.iloc[0])
            print(f"\n--- {v}: {r['fn']}({r['asked']})  truth {r['truth']:.5f}  said {r['value']}"
                  f"  [{'CORRECT' if r['ok'] else 'wrong'}]  {r['n_gen_tokens']} tokens")
            t = r["text"] or ""
            print("    " + (t[:900].replace("\n", "\n    ")))
        print(f"\nmedian thinking tokens by variant:")
        print(von[von['usable']].groupby("variant")["n_gen_tokens"].median().to_string())

if __name__ == "__main__":
    main()
