"""ms_mom.py -- MULTISTEP-01 Q2: momentum form, state statistic source and storage precision.

For a chain with oracle gradients at every step, simulate the candidate rule (ms_model) from step 1
and score every recorded update (test half = odd module index) per step, A and B separately.
Variants: momentum form (ema | sum | nest_ema | nest_sum) x statistic source S
(prevmom = stored momentum before this step's update | prevgrad = g_{t-1} | prevfed = vector fed to
PE at t-1) x momentum storage (f32 | bf16).
usage: ms_mom.py CHAIN [EPSFORM EPSL EPSR]    e.g. ms_mom.py S2_r8_b0.9 relmax 1e-4 0
Chains may switch beta (S3_*: steps 1-3 from the S2 b0.9 trunk, step 4 restored with a new beta).
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ms_common import step, state, grad, names, get, relF, summ, fmt, OUT, man
from ms_model import direction, mom_update
from fs_sim import bf16


def chain_records(chain):
    """list of (step_id) in order, following hist of the last step of the chain."""
    S = man()["steps"]
    last = sorted([s for s in S if s.startswith(chain + "/")], key=lambda s: len(S[s]["hist"]))[-1]
    return S[last]["hist"] + [last]


def run(chain, epsform="relmax", eL=1e-4, eR=0.0, variants=None, kinds="AB", stride=2, off=1):
    recs = chain_records(chain)
    S = man()["steps"]
    for sid in recs:
        assert S[sid]["grad_key"], f"{sid} has no same-regime gradient"
    Gz = [grad(S[s]["grad_key"]) for s in recs]
    Wz = [(state(S[s]["before"]), state(S[s]["after"])) for s in recs]
    betas = [S[s]["params"]["beta1"] for s in recs]
    ms = names(Wz[0][0])[off::stride]
    variants = variants or [(f, src, p) for f in ["ema", "sum", "nest_ema", "nest_sum"]
                            for src in ["prevmom", "prevgrad", "prevfed"] for p in ["f32"]]
    res = {(v, i, kind): [] for v in variants for i in range(len(recs)) for kind in kinds}
    for m in ms:
        for kind in kinds:
            gs = [get(G, kind, m, "G") for G in Gz]
            Us = [get(W1, kind, m) - get(W0, kind, m) for W0, W1 in Wz]
            for v in variants:
                form, src, prec = v
                mom, prev_m, prev_g, prev_fed = None, None, None, None
                for i, (g, U) in enumerate(zip(gs, Us)):
                    Sstat = {"prevmom": prev_m, "prevgrad": prev_g, "prevfed": prev_fed}[src]
                    mom, fed = mom_update(mom, g, betas[i], form)
                    if prec == "bf16":
                        mom = bf16(mom).astype(np.float64)
                        if form in ("ema", "sum"):
                            fed = mom
                    d = direction(fed, Sstat, eL, eR, epsform)
                    res[(v, i, kind)].append(relF(d, U) if np.any(U) else np.nan)
                    prev_m, prev_g, prev_fed = mom.copy(), g, fed
    out = {}
    for (v, i, kind), vals in res.items():
        out.setdefault("/".join(v), {})[f"{recs[i]}:{kind}"] = summ(vals)
    return recs, betas, out


if __name__ == "__main__":
    chain = sys.argv[1]
    ef, eL, eR = (sys.argv[2], float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else ("relmax", 1e-4, 0.0)
    V = os.environ.get("MS_VARIANTS")
    variants = [tuple(v.split("/")) for v in V.split(",")] if V else None
    stride = int(os.environ.get("MS_STRIDE", "2"))
    recs, betas, out = run(chain, ef, eL, eR, variants=variants, stride=stride)
    print(f"chain {chain}  betas {betas}  eps {ef} {eL} {eR}")
    for v, d in out.items():
        print(f"== {v}")
        for k, s in d.items():
            print(f"   {k:34s} {fmt(s)}")
    json.dump(dict(chain=chain, recs=recs, betas=betas, eps=[ef, eL, eR], results=out),
              open(OUT / f"mom_{chain}{os.environ.get('MS_TAG', '')}.json", "w"), indent=1)
