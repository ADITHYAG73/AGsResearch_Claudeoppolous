"""fr_baseline.py -- FRESH-01 Q1a: is a fresh-state step from B != 0 just PE5 of each factor's own
gradient?  For every fresh-state step with a same-regime gradient, per tensor:
  base   : relF(finalise(pe5_f16(G)), dW)        (simplest baseline: the step-1 law per factor)
  base64 : same with float64 PE5                  (precision effect)
  dirres : residual after the best scalar (direction only)
  null   : relF of the prediction made from the OTHER text's gradient at the same state
Output runs/FR/baseline.json + printed table.
"""
import json, sys
import numpy as np
from fr_core import *  # noqa

STEPS = [
    "S1_r8/step1",  # sanity: step 1 from B = 0 (FIRSTSTEP-01 law)
    "S1_r8_fresh_ck1_t0/step", "S1_r8_fresh_ck1_t1/step", "S1_r8_fresh_ck2_t0/step",
    "S1_r8_fresh_ck2_t1/step", "S1_r8_fresh_ck5_t0/step", "S1_r8_fresh_ck5_t1/step",
    "S1v2_r32_fresh_ck1_t0/step", "S1v2_r32_fresh_ck1_t1/step", "S1v2_r32_fresh_ck2_t0/step",
    "S1v2_r32_fresh_ck2_t1/step", "S1v2_r32_fresh_ck5_t0/step", "S1v2_r32_fresh_ck5_t1/step",
    "S2_r8_b0_fresh_ck1/step2", "S2_r8_b0_fresh_ck3/step4", "S3x_r8_fresh_ck3/step4",
    "S2_r32_b0_fresh_ck2/step3",
]


def other(sid_step):
    if "_t0/" in sid_step: return sid_step.replace("_t0/", "_t1/")
    if "_t1/" in sid_step: return sid_step.replace("_t1/", "_t0/")
    return None


def run(step_id, stride=1):
    r, W0, W1, G = step_io(step_id)
    lr, rank = r["params"]["learning_rate"], r["rank"]
    o = other(step_id)
    Go = grad(rec(o)["grad_key"]) if o and rec(o).get("grad_key") else None
    rows = []
    for i, m in enumerate(modules(W0)):
        if i % stride: continue
        mb = mB(m)
        row = dict(m=m)
        for F, key, gk in (("A", "A/" + m, "GA/" + m), ("B", "B/" + mb, "GB/" + mb)):
            D = W1[key] - W0[key]
            g = G[gk]
            row[F + "_dnorm"] = float(np.linalg.norm(D))
            row[F + "_gnorm"] = float(np.linalg.norm(g))
            if not np.any(g):
                row[F + "_base"] = None; continue
            P = finalise(pe5_f16(g), rank, lr)
            row[F + "_base"] = relF(P, D)
            row[F + "_dirres"] = best_scaled_relF(P, D)
            row[F + "_base64"] = relF(finalise(pe5_f64(g), rank, lr), D)
            row[F + "_cos"] = float(np.sum(P * D) / np.linalg.norm(P) / np.linalg.norm(D))
            if Go is not None:
                row[F + "_null"] = relF(finalise(pe5_f16(Go[gk]), rank, lr), D)
        rows.append(row)
    return r, rows


if __name__ == "__main__":
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    res = {}
    for s in STEPS:
        r, rows = run(s, stride)
        res[s] = rows
        print(f"\n{s}  rank {r['rank']} regime {r['regime']} hist {r['hist']}")
        for F in "AB":
            vals = [x[F + "_base"] for x in rows if x.get(F + "_base") is not None]
            if not vals:
                print(f"  {F}: gradient zero / no move (max |dW| {max(x[F+'_dnorm'] for x in rows):.3g})"); continue
            print(f"  {F} PE5-f16 own G : {fmt(summ(vals))}")
            print(f"  {F} PE5-f64 own G : {fmt(summ([x[F+'_base64'] for x in rows]))}")
            print(f"  {F} best-scalar   : {fmt(summ([x[F+'_dirres'] for x in rows]))}")
            print(f"  {F} cos(P,D)      : {fmt(summ([x[F+'_cos'] for x in rows]))}")
            if rows[0].get(F + "_null") is not None:
                print(f"  {F} null(other T) : {fmt(summ([x[F+'_null'] for x in rows]))}")
        sys.stdout.flush()
    json.dump(res, open(OUT / "baseline.json", "w"), indent=0)
