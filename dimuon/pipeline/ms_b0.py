"""ms_b0.py -- MULTISTEP-01 Q1: hidden state at beta1 = 0 (S2_r{8,32}_b0 continuing steps 2..6).

Stage 'grid': on the FIT half (even module index, every 2nd of those) choose the eps form/values of
the symmetric previous-gradient preconditioner (ms_model.direction with S = g_{t-1}).
Stage 'eval': score on the TEST half (odd module index) for all steps, A and B separately,
next to (a) PE5(g_t) alone and (b) the fresh-state twin as null. Also scores a set of rival rules.
usage: ms_b0.py grid RANK  |  ms_b0.py eval RANK FORM EPSL EPSR
"""
import sys, os, json, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ms_common import step, state, grad, names, get, pe, relF, summ, fmt, OUT
from ms_model import direction

mode, r = sys.argv[1], int(sys.argv[2])
T = f"S2_r{r}_b0"
STEPS = [2, 3, 4, 5, 6]


def load(k):
    c = step(f"{T}/step{k}")
    f = step(f"{T}_fresh_ck{k-1}/step{k}")
    assert c["before"] == f["before"] and c["grad_key"] == f["grad_key"]
    assert c["regime"] == step(f"{T}/step{k-1}")["regime"] == "R2"
    return (state(c["before"]), state(c["after"]), state(f["after"]),
            grad(c["grad_key"]), grad(step(f"{T}/step{k-1}")["grad_key"]))


if mode == "grid":
    grid = {"relmax": [0, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
            "relmean": [0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2],
            "abs": [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]}
    res = {}
    for k in [2, 4, 6]:
        W0, Wc, Wf, G, Gp = load(k)
        ms = names(W0)[0::2][::2]
        for kind in "AB":
            data = [(get(G, kind, m, "G"), get(Gp, kind, m, "G"), get(Wc, kind, m) - get(W0, kind, m)) for m in ms]
            for form, vals in grid.items():
                for eL, eR in itertools.product(vals, [0] + vals[1:3]):
                    v = [relF(direction(g, gp, eL, eR, form), U) for g, gp, U in data]
                    res.setdefault((form, eL, eR, kind), []).extend(v)
        del W0, Wc, Wf, G, Gp
    rows = sorted(((np.median(v), np.quantile(v, .9), *key) for key, v in res.items()), key=lambda x: x[0])
    with open(OUT / f"b0_grid_r{r}.txt", "w") as fh:
        for kind in "AB":
            for row in [x for x in rows if x[5] == kind][:12]:
                s = f"r{r} {kind} form={row[2]} epsL={row[3]:g} epsR={row[4]:g}  med {row[0]:.4f} p90 {row[1]:.4f}"
                print(s); fh.write(s + "\n")

if mode == "eval":
    form, eL, eR = sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
    rules = {
        "PE5(g_t) [baseline a]": lambda g, gp: -pe(g),
        "PE5(g_t), X0 fp16": lambda g, gp: -pe(g, x0prec="f16"),
        "sym prev-grad, eps->0": lambda g, gp: direction(g, gp, 0, 0, form),
        "MODEL sym prev-grad (fitted eps)": lambda g, gp: direction(g, gp, eL, eR, form),
        "MODEL, X0 fp16": lambda g, gp: direction(g, gp, eL, eR, form, x0prec="f16"),
        "L only (pL=-1/4,pR=0)": lambda g, gp: direction(g, gp, eL, eR, form, pR=0.0),
        "pre-only L^-1/2 (row-normalise prev)": None,
        "sym CURRENT-grad stats": lambda g, gp: direction(g, g, eL, eR, form),
        "sym exponent -1/2 both sides": lambda g, gp: direction(g, gp, eL, eR, form, pL=-0.5, pR=-0.5),
    }

    def pre_only(g, gp):
        l = (gp ** 2).sum(1); l = l + 1e-2 * l.mean()
        return -pe(g * (l ** -0.5)[:, None])
    rules["pre-only L^-1/2 (row-normalise prev)"] = pre_only
    out = {}
    for k in STEPS:
        W0, Wc, Wf, G, Gp = load(k)
        ms = names(W0)[1::2]
        for kind in "AB":
            acc = {n: [] for n in rules}; null = []
            for m in ms:
                g, gp = get(G, kind, m, "G"), get(Gp, kind, m, "G")
                U = get(Wc, kind, m) - get(W0, kind, m)
                null.append(relF(get(Wf, kind, m) - get(W0, kind, m), U))
                for n, f in rules.items():
                    acc[n].append(relF(f(g, gp), U))
            out[f"step{k}_{kind}"] = {"fresh twin [null b]": summ(null), **{n: summ(v) for n, v in acc.items()}}
            print(f"--- r{r} step{k} {kind} (test half, n={len(ms)}) ---", flush=True)
            for n, s in out[f"step{k}_{kind}"].items():
                print(f"  {n:38s} {fmt(s)}", flush=True)
        del W0, Wc, Wf, G, Gp
    json.dump(dict(form=form, epsL=eL, epsR=eR, results=out), open(OUT / f"b0_eval_r{r}.json", "w"), indent=1)
