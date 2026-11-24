"""p3_predict.py -- PREDICT-01 predictions, written to runs/P3/pred/<exp>/ from oracle GRADIENTS and
call parameters only (never reads runs/P3/states). Run each target BEFORE scoring; for p1a, p1b,
p3zero, p4 it is run before the measured branch exists at all (pre-registration; each file records
its wall time).

usage: p3_predict.py TARGET     targets: p6 p1a p1b p2 p3zero p3step p4 p5 p7 p8
"""
import sys, time
import numpy as np
import p3_rule as R
from p3_rule import grad, gget, tensors, rule, nostate, write_pred

M = R.man()
ST = M["steps"]


def recs_of(chain):
    ids = [s for s in ST if s.startswith(chain + "/")]
    return sorted(ids, key=lambda s: ST[s]["index_in_chain"])


def lineage(step_id):
    return ST[step_id]["hist"] + [step_id]


import os
PREFER = os.environ.get("P3_PREFER", "")   # "solo": use an oracle computed with no concurrent client


def pick(sid, text, default):
    if PREFER == "solo":
        alt = sorted(k for k, g in M["grads"].items()
                     if g["sid"] == sid and g["text"] == text and g["setting"].startswith("P3x"))
        if alt:
            return alt[0]
    return default


def gfor(step_id):
    rec = ST[step_id]
    if rec["weight"] == 0:
        return None
    k = pick(rec["before"], rec["text"], rec["grad_key"])
    return grad(k) if k in M["grads"] else "missing"


def roll(step_ids, keys, storage="bf16", mode="x0f16", cfn=None, upto=None):
    """simulate the rule over step_ids (in order) from an empty state. cfn(step_id, kind, m) -> c.
    Returns list of dicts {key: U} and the final stored momenta {key: S}."""
    S = {k: None for k in keys}
    out = []
    shapes = {}
    for sid in step_ids:
        rec = ST[sid]
        r = rec["rank"]
        G = gfor(sid)
        assert not isinstance(G, str), f"no gradient for {sid}"
        P = {}
        for k in keys:
            if G is None:
                g = np.zeros(shapes[k], np.float32)
            else:
                g = gget(G, *k); shapes[k] = g.shape
            c = cfn(sid, *k) if cfn else 1.0
            P[k], S[k] = rule(S[k], g, rec["params"]["beta1"], rec["params"]["learning_rate"], r,
                              c=c, storage=storage, mode=mode)
        out.append(P)
    return out, S


def keys_of(step_id):
    return tensors(grad(ST[step_id]["grad_key"]))


def emit(exp, name, P, **meta):
    write_pred(exp, name, P, meta)
    print(time.strftime("%H:%M:%S"), exp, name, flush=True)


def p6():
    ids = recs_of("P6_r8")
    keys = keys_of(ids[0])
    for storage in ("bf16", "f32"):
        out, _ = roll(ids, keys, storage=storage)
        for sid, P in zip(ids, out):
            emit("P6", f"{sid.split('/')[1]}_{storage}", P, step=sid, storage=storage,
                 source="rolled from oracle gradients of steps 1..t only")
    # null: no state (PE5 of this step's gradient alone)
    for sid in ids:
        G = gfor(sid); rec = ST[sid]
        emit("P6", f"{sid.split('/')[1]}_nostate",
             {k: nostate(gget(G, *k), 1e-3, rec["rank"]) for k in keys}, step=sid)


def p1a():
    import p3_collect as C
    k1 = ST["P6_r8/step2"]["grad_key"]
    g = M["grads"][k1]
    NA, NB, N = g["GA_global"], g["GB_global"], g["G_global"]
    keys = keys_of("P6_r8/step1")
    for c in C.P1A_C + [1.0]:
        clip = c * N
        for ndef, Nx in (("all", N), ("B", NB)):
            cc = min(1.0, clip / Nx)
            cf = lambda sid, kind, m, cc=cc: cc if sid == "P6_r8/step2" else 1.0
            out, _ = roll(["P6_r8/step1", "P6_r8/step2"], keys, cfn=cf)
            emit("P1a", f"c{c}_{ndef}", out[1], c_nominal=c, clip=clip, c_used=cc, Ndef=ndef,
                 NA=NA, NB=NB, N=N)


def p1b():
    import p3_collect as C
    T = C.p1b_thresholds()
    ids = lineage("P6_r8/step7")
    keys = keys_of(ids[0])
    pre, S6 = roll(ids[:-1], keys)
    G = gfor("P6_r8/step7")
    pertensor = {k: float(np.linalg.norm(gget(G, *k))) for k in keys}
    for t in ("t1", "t2", "t3"):
        clip = T[t]
        hyp = {"all": lambda kind, k: min(1, clip / T["N"]),
               "B": lambda kind, k: min(1, clip / T["NB"]),
               "pertype": lambda kind, k: min(1, clip / (T["NA"] if kind == "A" else T["NB"])),
               "pertensor": lambda kind, k: min(1, clip / pertensor[k]),
               "none": lambda kind, k: 1.0}
        for h, f in hyp.items():
            P = {}
            for k in keys:
                P[k], _ = rule(S6[k], gget(G, *k), 0.9, 1e-3, 8, c=f(k[0], k))
            emit("P1b", f"{t}_{h}", P, clip=clip, hyp=h, **T)


def p2():
    ids = recs_of("P2_r32")
    keys = keys_of(ids[0])
    for storage in ("bf16", "f32"):
        for mode in ("x0f16", "f64"):
            out, _ = roll(ids, keys, storage=storage, mode=mode)
            for sid, P in zip(ids, out):
                emit("P2", f"{sid.split('/')[1]}_{storage}_{mode}", P, step=sid, storage=storage,
                     mode=mode)


def p3zero():
    import p3_collect as C
    ids = lineage("P6_r8/step2")
    keys = keys_of(ids[0])
    _, S2 = roll(ids, keys)
    for b in C.P3_B:
        P = {}
        for k in keys:
            P[k], _ = rule(S2[k], np.zeros_like(S2[k]), b, 1e-3, 8)
        emit("P3", f"zero_b{b}", P, beta=b, note="zero-gradient step from rolled S2")


def p3step():
    import p3_collect as C
    ids = lineage("P6_r8/step2")
    keys = keys_of(ids[0])
    _, S2 = roll(ids, keys)
    for b in C.P3_B:
        sid = f"P3_zb{b}/step3"
        G = gfor(sid)
        P, Pn = {}, {}
        for k in keys:
            _, Sz = rule(S2[k], np.zeros_like(S2[k]), b, 1e-3, 8)
            P[k], _ = rule(Sz, gget(G, *k), 0.9, 1e-3, 8)
            # null: as if the zero step had used beta 0.9 (beta' ignored)
            _, Sz9 = rule(S2[k], np.zeros_like(S2[k]), 0.9, 1e-3, 8)
            Pn[k], _ = rule(Sz9, gget(G, *k), 0.9, 1e-3, 8)
        emit("P3", f"step3_b{b}", P, beta_zero=b)
        emit("P3", f"step3_b{b}_null09", Pn, beta_zero=b)


def p4():
    import p3_collect as C
    for r, base, txt_step, lrs in ((32, "P2_r32", "P2_r32/step2", C.P4_R32),
                                   (8, "P6_r8", "P6_r8/step2", C.P4_R8 + [1e-3])):
        ids = lineage(txt_step)
        keys = keys_of(ids[0])
        _, S2 = roll(ids, keys)
        gkey = (f"{ST['P2_r32/step2']['after']}_t1_w1_P3" if r == 32
                else ST["P6_r8/step3"]["grad_key"])
        gkey = pick(M["grads"][gkey]["sid"], M["grads"][gkey]["text"], gkey)
        G = grad(gkey)
        for mode in ("x0f16", "f64", "fused16", "fused32", "fusedbf16"):
            P1, P1f = {}, {}
            for k in keys:
                P1[k], _ = rule(S2[k], gget(G, *k), 0.9, 1e-3, r, mode=mode)
                if r == 32:
                    P1f[k] = nostate(gget(G, *k), 1e-3, r, mode=mode)
            emit("P4", f"r{r}_lr1e-3_{mode}", P1, mode=mode, gkey=gkey,
                 note="prediction at lr 1e-3; at other lr the rule predicts exactly lr/1e-3 x this")
            if r == 32:
                emit("P4", f"r32_fresh_lr1e-3_{mode}", P1f, mode=mode, gkey=gkey)


def p5():
    for name, sid in (("r32_ck1_t1", "P5_fresh_r32_ck1_t1/step"),
                      ("r8b0_ck2_t1", "P5_fresh_r8b0_ck2_t1/step"),
                      ("r8b0_ck3_t2", "P5_fresh_r8b0_ck3_t2/step")):
        rec = ST[sid]
        for setting in ("P3", "P3b"):
            key = rec["grad_key"].rsplit("_", 1)[0] + "_" + setting
            G = grad(key)
            keys = tensors(G)
            emit("P5", f"{name}_{setting}",
                 {k: nostate(gget(G, *k), 1e-3, rec["rank"]) for k in keys}, gkey=key)


def roll_chain(exp, chain):
    ids = recs_of(chain)
    keys = keys_of(ids[0])
    out, _ = roll(ids, keys)
    for sid, P in zip(ids, out):
        emit(exp, sid.split("/")[1], P, step=sid)
    for sid in ids:
        G = gfor(sid); rec = ST[sid]
        emit(exp, f"{sid.split('/')[1]}_nostate",
             {k: nostate(gget(G, *k), 1e-3, rec["rank"]) for k in keys}, step=sid)
    return ids, keys


def p7():
    ids, keys = roll_chain("P7", "P7_r8")
    pre = lineage("P7_r8/step2")
    _, S2 = roll(pre, keys)
    G = gfor("P7_b0_cont/step3")
    emit("P7", "b0_cont", {k: rule(S2[k], gget(G, *k), 0.0, 1e-3, 8)[0] for k in keys})
    emit("P7", "b0_fresh", {k: rule(None, gget(G, *k), 0.0, 1e-3, 8)[0] for k in keys})


def p8():
    roll_chain("P8", "P8_r8")


if __name__ == "__main__":
    globals()[sys.argv[1]]()
