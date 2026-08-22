"""SIM-02 - power sweep for the H1 bimodality test.

WHY. SIM-01 (2026-08-21) concluded the bimodality test was infeasible "at any affordable K".
That verdict used a HARDCODED `noise = 0.8` in analysis.py:81, a back-of-envelope from ONE
transcript. NOISE-01 then measured the real paired-Delta noise at 1.41x the between-claim
spread - roughly 4x smaller. SIM-01's verdict therefore has to be re-derived.

WHY A SWEEP AND NOT A POINT ESTIMATE. The 1.41x rests on n=6 recurring claims, and its
denominator (spread of claim means across mixed categories) is not obviously the same
quantity as this simulation's (spread WITHIN the related-false category). Substituting one
fragile number through a shaky unit conversion is how you get a confident wrong answer.
So: sweep the ratio, find the THRESHOLD where the test flips, and report how far 1.41 sits
from that edge.

UNIT. noise_sd = ratio * S, where S is the between-claim sd of World A's related-false
population, computed from the planting constants (NOT assumed):
    S = sqrt( p(1-p)*gap^2 + within^2 )
      = sqrt( 0.35*0.65*(0.31-0.06)^2 + 0.05^2 ) = 0.1293
SIM-01's noise=0.8 therefore corresponds to ratio = 0.8/0.1293 = 6.19.

REGRESSION CHECK. At ratio 6.19 this sweep MUST reproduce SIM-01's "cannot separate" verdict.
If it does not, the unit conversion is wrong and every other cell is meaningless.

DECISION RULE - unchanged from analysis.py:120, deliberately not re-tuned:
    TWO humps  iff  dBIC(2v1) > 10  AND  dip p < 0.05
Correct behaviour = TWO for World A, ONE for World B. Any ratio where World B reads TWO is a
false-positive regime and is disqualifying regardless of what World A does.
"""
import argparse, math, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import h1_report

P, GAP, WITHIN = 0.35, 0.31 - 0.06, 0.05
S = math.sqrt(P * (1 - P) * GAP ** 2 + WITHIN ** 2)      # 0.1293

def simulate_world(world, ratio, n, K, rng):
    """Planting as analysis.py:simulate, with noise expressed as ratio * S.

    World C added 2026-08-22: a SKEWED unimodal null. World B is a clean Gaussian, so it
    cannot expose the failure mode dBIC is actually vulnerable to - a two-Gaussian mixture
    fits a single skewed hump well and reports a large dBIC. Real Delta is skewed (NOISE-01:
    81% of ablations positive), so B alone is an unrealistically easy null. C is a gamma
    (shape 4, skew 1.0) rescaled to World B's mean and sd."""
    noise = ratio * S
    if world == "A":                     # H1 TRUE: 35% true-like + 65% confab-like
        is_b = rng.random(n) < P
        m = np.where(is_b, rng.normal(0.31, WITHIN, n), rng.normal(0.06, WITHIN, n))
    elif world == "B":                   # H1 FALSE: one Gaussian hump at the paper's mean
        m = rng.normal(0.14, WITHIN, n)
    else:                                # H1 FALSE, realistic: one SKEWED hump, same mean/sd
        shape = 4.0
        g = rng.gamma(shape, 1.0, n)
        m = 0.14 + WITHIN * (g - shape) / math.sqrt(shape)
    return m + rng.normal(0, noise / math.sqrt(K), n)

def verdict(x):
    """Return the three verdicts separately. The conjunction is analysis.py's frozen rule;
    reporting the components separately shows WHICH test is binding."""
    r = h1_report({"x": x})["x"]
    return {"bic": r["dBIC_2v1"] > 10, "dip": r["dip_p"] < 0.05,
            "both": r["dBIC_2v1"] > 10 and r["dip_p"] < 0.05}, r

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratios", default="0.5,1.0,1.41,2.0,3.0,5.0,6.19")
    ap.add_argument("--ks", default="1,4,8,16")
    ap.add_argument("--ns", default="400,700,2065")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    ratios = [float(x) for x in a.ratios.split(",")]
    ks = [int(x) for x in a.ks.split(",")]
    ns = [int(x) for x in a.ns.split(",")]

    print(f"between-claim sd of World A related-false: S = {S:.4f}")
    print(f"SIM-01's noise=0.8 corresponds to ratio {0.8/S:.2f}")
    print(f"\nEach cell: hits/{a.seeds} for World A (should detect) | false positives on")
    print(f"World B (Gaussian null) | false positives on World C (SKEWED null).")
    print("Columns: BIC = dBIC>10 alone · DIP = dip p<0.05 alone · BOTH = frozen rule.\n")
    for n in ns:
        print(f"{'='*92}\nn = {n} claims\n{'='*92}")
        hdr = f"{'ratio':>6} {'K':>3}  "
        for c in ("BIC", "DIP", "BOTH"): hdr += f"{c+' A|fpB|fpC':>18}"
        print(hdr)
        for ratio in ratios:
            for K in ks:
                acc = {c: [0, 0, 0] for c in ("bic", "dip", "both")}
                for sd_ in range(a.seeds):
                    va, _ = verdict(simulate_world("A", ratio, n, K, np.random.default_rng(1000*sd_+K)))
                    vb, _ = verdict(simulate_world("B", ratio, n, K, np.random.default_rng(1000*sd_+K+7)))
                    vc, _ = verdict(simulate_world("C", ratio, n, K, np.random.default_rng(1000*sd_+K+13)))
                    for c in acc:
                        acc[c][0] += va[c]; acc[c][1] += vb[c]; acc[c][2] += vc[c]
                row = f"{ratio:6.2f} {K:3d}  "
                for c in ("bic", "dip", "both"):
                    row += f"{acc[c][0]}|{acc[c][1]}|{acc[c][2]:<12}".rjust(18)
                print(row)
            print()

if __name__ == "__main__":
    main()
