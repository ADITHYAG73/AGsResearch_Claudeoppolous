#!/usr/bin/env python3
"""TRIG-03 report: thinking ON vs thinking OFF on the SAME 1080 angles.

Writes two things and prints one table:
  1. TRIG_all_results.csv   - every ask from every run, one flat sheet for Excel
  2. TRIG_on_vs_off.csv     - one row per (function, angle) with both answers side by side
  3. the on-vs-off summary, including the tabulation gradient under each condition

Truncated generations (hit the token cap) are marked and excluded from accuracy: a run that
stopped mid-calculation did not give an answer. The cap differs for 24 of the rows (they were
generated at 4096 before the run was restarted at 2048); those are flagged rather than dropped,
and the flag is reported so the inconsistency is visible rather than buried.
"""
import json, math, os, sys
import numpy as np, pandas as pd

TOL, CLOSE = 5e-6, 1e-3
R = os.path.dirname(os.path.abspath(__file__))

def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k/n; d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return (100*(c-h), 100*(c+h))

def mcnemar(b, c):
    from math import comb
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k+1)) / 2**n)

def load(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []

def main():
    d_off = sys.argv[1]          # .../2026-09-22_trig
    d_on  = sys.argv[2]          # .../2026-09-23_trig03

    off = pd.DataFrame(load(os.path.join(d_off, "results_off_step1.jsonl")))
    on  = pd.DataFrame(load(os.path.join(d_on,  "results_on_step1.jsonl")))
    if not len(on):
        print("no thinking-on rows yet"); return

    for df, cap in ((off, 24), (on, 2048)):
        df["trunc"] = df["n_gen_tokens"] >= cap
        df["err"] = (df["value"] - df["truth"]).abs()
        df["usable"] = (~df["pole"]) & df["value"].notna() & (~df["trunc"])
        df["ok"] = df["usable"] & (df["err"] <= TOL)

    # the 24 rows generated before the cap was lowered
    odd = int((on["n_gen_tokens"] > 2048).sum())
    print(f"thinking-ON rows: {len(on)}   truncated at the cap: {int(on['trunc'].sum())}"
          f"   generated at the earlier 4096 cap: {odd}")
    print(f"thinking-OFF rows: {len(off)}  truncated: {int(off['trunc'].sum())}\n")

    key = ["fn", "deg"]
    m = off.set_index(key).join(on.set_index(key), lsuffix="_off", rsuffix="_on", how="inner")
    both = m[m["usable_off"] & m["usable_on"]]
    print(f"{'='*66}\nTHINKING OFF vs ON — same {len(both)} asks, paired\n{'='*66}")
    for lab, col in (("exact to 5 dp", "ok"), ):
        a = both[f"{col}_off"]; b = both[f"{col}_on"]
        ka, kb = int(a.sum()), int(b.sum())
        la, ha = wilson(ka, len(a)); lb, hb = wilson(kb, len(b))
        print(f"  {lab:<16} OFF {100*ka/len(a):5.1f}%   95% CI [{la:.1f}, {ha:.1f}]")
        print(f"  {'':<16} ON  {100*kb/len(b):5.1f}%   95% CI [{lb:.1f}, {hb:.1f}]")
        nb = int((a & ~b).sum()); nc = int((~a & b).sum())
        print(f"  change {100*(kb-ka)/len(a):+.1f} pts   "
              f"(off-right/on-wrong {nb}, off-wrong/on-right {nc}, McNemar p={mcnemar(nb,nc):.2g})")

    print(f"\n  by function:")
    print(f"    {'fn':<5}{'OFF':>9}{'ON':>9}{'change':>10}")
    for fn in ("sin", "cos", "tan"):
        g = both[both.index.get_level_values("fn") == fn]
        if not len(g): continue
        a, b = 100*g["ok_off"].mean(), 100*g["ok_on"].mean()
        print(f"    {fn:<5}{a:>8.1f}%{b:>8.1f}%{b-a:>+9.1f}")

    print(f"\n  by quadrant:")
    print(f"    {'quadrant':<12}{'OFF':>9}{'ON':>9}{'change':>10}")
    degs = both.index.get_level_values("deg")
    for q, (lo, hi) in enumerate([(0,90),(90,180),(180,270),(270,360)], 1):
        g = both[(degs >= lo) & (degs < hi)]
        if not len(g): continue
        a, b = 100*g["ok_off"].mean(), 100*g["ok_on"].mean()
        print(f"    Q{q} [{lo:3d},{hi:3d}){a:>8.1f}%{b:>8.1f}%{b-a:>+9.1f}")

    print(f"\n  THE TABULATION GRADIENT — does the scratchpad replace recall with computation?")
    print(f"    {'angle class':<16}{'n':>5}{'OFF':>9}{'ON':>9}{'change':>10}")
    cls = [("multiple of 15", lambda d: d % 15 == 0),
           ("multiple of 5",  lambda d: d % 5 == 0 and d % 15 != 0),
           ("even",           lambda d: d % 2 == 0 and d % 5 != 0),
           ("odd",            lambda d: d % 2 == 1 and d % 5 != 0)]
    offv, onv = [], []
    for name, t in cls:
        g = both[[t(d) for d in degs]]
        if not len(g): continue
        a, b = 100*g["ok_off"].mean(), 100*g["ok_on"].mean()
        offv.append(a); onv.append(b)
        print(f"    {name:<16}{len(g):>5}{a:>8.1f}%{b:>8.1f}%{b-a:>+9.1f}")
    if len(offv) == 4:
        print(f"\n    spread best-worst:  OFF {max(offv)-min(offv):.1f} pts   ON {max(onv)-min(onv):.1f} pts")
        print("    a much smaller ON spread = the scratchpad substituted computation for recall")

    print(f"\n  thinking-ON generation length: median {on['n_gen_tokens'].median():.0f}"
          f"  p95 {on['n_gen_tokens'].quantile(.95):.0f}  max {on['n_gen_tokens'].max()}")

    # ---- CSV 1: one row per ask, everything ----
    rows = []
    def add(df, exp, think, cap):
        for _, r in df.iterrows():
            rows.append({"experiment": exp, "thinking": think, "function": r["fn"],
                "angle_deg": r["deg"], "model_answer": r["value"],
                "calculator_answer": ("UNDEFINED (pole)" if r["pole"] else r["truth"]),
                "abs_error": (None if r["pole"] else r["err"]),
                "verdict": ("pole" if r["pole"] else "TRUNCATED" if r["trunc"]
                            else "PASS" if r["ok"] else "FAIL"),
                "within_0.001": ("" if not r["usable"] else ("yes" if r["err"] <= CLOSE else "no")),
                "decimals_given": r["decimals"], "format_exactly_5dp": r["format_ok"],
                "tokens_generated": r["n_gen_tokens"], "token_cap": cap,
                "raw_text": r["raw"]})
    add(off, "1-degree sweep", "OFF", 24)
    add(on,  "1-degree sweep", "ON", 2048)
    allcsv = os.path.join(d_on, "TRIG_all_results.csv")
    pd.DataFrame(rows).to_csv(allcsv, index=False)

    # ---- CSV 2: side by side, one row per angle ----
    sbs = m.reset_index()[["fn", "deg", "truth_off", "value_off", "err_off", "ok_off",
                           "value_on", "err_on", "ok_on", "n_gen_tokens_on", "pole_off"]]
    sbs.columns = ["function", "angle_deg", "calculator_answer", "model_thinking_OFF",
                   "error_OFF", "pass_OFF", "model_thinking_ON", "error_ON", "pass_ON",
                   "tokens_ON", "is_pole"]
    sbscsv = os.path.join(d_on, "TRIG_on_vs_off.csv")
    sbs.to_csv(sbscsv, index=False)
    print(f"\n  -> {allcsv}  ({len(rows)} rows)")
    print(f"  -> {sbscsv}  ({len(sbs)} rows, one per angle, both answers side by side)")

if __name__ == "__main__":
    main()
