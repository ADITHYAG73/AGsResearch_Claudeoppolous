"""The analysis — written BEFORE real data exists, tested on planted answers.

Input table (one row per claim):
  claim_id · doc_id · pos · k · verdict (SUPPORTED/CONTRADICTED/NOT_IN_TEXT)
  · relatedness (DIRECT/ADJACENT/UNRELATED, false claims only) · delta (Δ per resample)

Two questions it answers:
  H2  does per-claim Δ variance shrink as 1/sqrt(K) when averaged over K resamples?
  H1  is the related-false Δ distribution ONE hump or TWO?

Run:  python analysis.py --simulate     (planted worlds A and B; code must tell them apart)
      python analysis.py --claims claims_scored.parquet
"""
import argparse, json, math
import numpy as np

# ------------------------------------------------------------------ H2
def variance_vs_k(deltas_by_claim, ks=(1, 2, 4, 8)):
    """deltas_by_claim: {claim_id: array of K_max Δ samples}. For each K, average the first K
    samples per claim and report the sd ACROSS claims of that mean. If noise is independent,
    sd(K) ≈ sd(1)/sqrt(K): log-log slope ≈ -0.5."""
    out = {}
    for K in ks:
        means = np.array([np.mean(v[:K]) for v in deltas_by_claim.values() if len(v) >= K])
        out[K] = float(means.std(ddof=1))
    ks_ok = [k for k in ks if k in out]
    slope = np.polyfit(np.log(ks_ok), np.log([out[k] for k in ks_ok]), 1)[0]
    return out, float(slope)

# ------------------------------------------------------------------ H1
def dip_pvalue(x, **_):
    """Hartigan's dip test via the `diptest` package (standard implementation; p from its
    reference table). Unimodal null. Small p ⇒ evidence of >1 mode."""
    import diptest
    d, p = diptest.diptest(np.asarray(x, float))
    return float(d), float(p)

def two_vs_one_gaussian(x):
    """Fit 1- and 2-component Gaussian mixtures by EM; return ΔBIC (positive favours 2)."""
    x = np.asarray(x, float); n = len(x)
    mu, sd = x.mean(), x.std(ddof=1) + 1e-9
    ll1 = np.sum(-0.5 * ((x - mu) / sd) ** 2 - np.log(sd * math.sqrt(2 * math.pi)))
    bic1 = -2 * ll1 + 2 * math.log(n)
    # EM for 2 components, init by quantiles
    m = np.array([np.quantile(x, .25), np.quantile(x, .75)]); s = np.array([sd, sd]) / 2; w = np.array([.5, .5])
    for _ in range(200):
        pdf = np.stack([w[j] / (s[j] * math.sqrt(2 * math.pi)) * np.exp(-0.5 * ((x - m[j]) / s[j]) ** 2) for j in range(2)])
        r = pdf / (pdf.sum(0) + 1e-300)
        nk = r.sum(1) + 1e-9
        m = (r * x).sum(1) / nk; s = np.sqrt((r * (x - m[:, None]) ** 2).sum(1) / nk) + 1e-6; w = nk / n
    ll2 = np.sum(np.log(pdf.sum(0) + 1e-300))
    bic2 = -2 * ll2 + 5 * math.log(n)
    return float(bic1 - bic2), {"means": m.round(4).tolist(), "sds": s.round(4).tolist(), "weights": w.round(3).tolist()}

def h1_report(delta_by_category):
    """delta_by_category: {'true': arr, 'related_false': arr, 'unrelated_false': arr}"""
    rep = {}
    for cat, x in delta_by_category.items():
        x = np.asarray(x, float)
        d, p = dip_pvalue(x)
        dbic, fit = two_vs_one_gaussian(x)
        rep[cat] = {"n": len(x), "mean": round(float(x.mean()), 4), "sd": round(float(x.std(ddof=1)), 4),
                    "dip": round(d, 4), "dip_p": round(p, 3), "dBIC_2v1": round(dbic, 1), "fit2": fit}
    return rep

def ascii_hist(x, bins=24, width=48, title=""):
    h, edges = np.histogram(x, bins=bins)
    top = h.max() or 1
    lines = [f"  {title}"] if title else []
    for c, e in zip(h, edges[:-1]):
        lines.append(f"  {e:+7.3f} | {'█' * int(width * c / top)}")
    return "\n".join(lines)

# ------------------------------------------------------------------ simulation
def simulate(world, n_claims=400, K=8, seed=0):
    """Plant the answer. Paper's aggregate numbers (pp of FVE): true≈0.31, related-false≈0.14,
    unrelated-false≈0.06. Per-resample noise sd chosen so that sd/effect ≈ 5 at K=1 (matches
    the ~5x we estimated from the fennec example)."""
    rng = np.random.default_rng(seed)
    noise = 0.8                    # per-resample sd, independent across resamples
    def draw(mean_per_claim):
        return {f"c{i}": mean_per_claim[i] + rng.normal(0, noise, K) for i in range(len(mean_per_claim))}
    true_m   = rng.normal(0.31, 0.05, n_claims)
    unrel_m  = rng.normal(0.06, 0.05, n_claims)
    if world == "A":     # H1 TRUE: related-false = 35% true-like + 65% confab-like
        is_bhaskar = rng.random(n_claims) < 0.35
        rel_m = np.where(is_bhaskar, rng.normal(0.31, 0.05, n_claims), rng.normal(0.06, 0.05, n_claims))
    else:                # H1 FALSE: related-false = one hump at the paper's mean
        rel_m = rng.normal(0.14, 0.05, n_claims)
    return {"true": draw(true_m), "related_false": draw(rel_m), "unrelated_false": draw(unrel_m)}

def run(deltas_by_cat, K_use):
    # H2 on all claims pooled
    pooled = {f"{c}:{k}": v for c, d in deltas_by_cat.items() for k, v in d.items()}
    sd_by_k, slope = variance_vs_k(pooled)
    # H1 on K-averaged Δ per claim
    avg = {c: np.array([np.mean(v[:K_use]) for v in d.values()]) for c, d in deltas_by_cat.items()}
    return sd_by_k, slope, h1_report(avg), avg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--claims", default=None)
    ap.add_argument("--K", type=int, default=8)
    a = ap.parse_args()

    if a.simulate:
        for world, truth in [("A", "H1 TRUE — related-false is a 35/65 MIXTURE"),
                             ("B", "H1 FALSE — related-false is ONE hump")]:
            print(f"\n{'='*72}\nWORLD {world}  (planted: {truth})\n{'='*72}")
            data = simulate(world, K=a.K)
            sd_by_k, slope, rep, avg = run(data, a.K)
            print("H2  sd of claim-mean Δ vs K:", {k: round(v, 3) for k, v in sd_by_k.items()},
                  f"  log-log slope = {slope:+.2f}  (expect -0.50 if noise is independent)")
            for cat, r in rep.items():
                print(f"H1  {cat:16s} n={r['n']} mean={r['mean']:+.3f} sd={r['sd']:.3f}  "
                      f"dip={r['dip']:.4f} p={r['dip_p']:.3f}  dBIC(2v1)={r['dBIC_2v1']:+.1f}  "
                      f"2-fit means={r['fit2']['means']} w={r['fit2']['weights']}")
            print(ascii_hist(avg["related_false"], title=f"related-false Δ, K={a.K}-averaged"))
            verdict = "TWO humps (H1 supported)" if rep["related_false"]["dBIC_2v1"] > 10 and rep["related_false"]["dip_p"] < 0.05 else "ONE hump (H1 not supported)"
            print(f"\n>>> code's verdict for world {world}: {verdict}")
        return

    raise SystemExit("real-data path: wire to claims_scored.parquet after Session B")

if __name__ == "__main__":
    main()
