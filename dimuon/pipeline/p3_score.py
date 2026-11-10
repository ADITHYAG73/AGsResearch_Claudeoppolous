"""p3_score.py -- PREDICT-01 scoring: compare runs/P3/pred/<exp>/ with the measured updates.
Residual = ||U_pred - U_meas||_F / ||U_meas||_F per tensor (prediction includes its norm),
summarised median / p90 / max, A and B separately. Writes runs/P3/score_<exp>.{txt,json}.

usage: p3_score.py EXP     EXP in p6 p1a p1b p2 p3 p4 p5 p7 p8 regime
"""
import json, sys
import numpy as np
import p3_rule as R
from p3_rule import (read_pred, measured, score, fmt, summ, relF, cos, grad, gget, state, wget,
                     tensors, rule, P3)

M = R.man()
ST = M["steps"]
LINES, OUTJ = [], {}


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LINES.append(s)


def save(exp):
    open(P3 / f"score_{exp}{R.SUFFIX}.txt", "w").write("\n".join(LINES) + "\n")
    json.dump(OUTJ, open(P3 / f"score_{exp}{R.SUFFIX}.json", "w"), indent=1, default=str)


def row(label, sc):
    say(f"  {label:34s} A {fmt(sc['A']):32s} B {fmt(sc['B'])}")
    OUTJ[label] = sc


def have(exp, name):
    return (R.PRED / (exp + R.SUFFIX) / f"{name}.npz").exists()


def recs_of(chain):
    ids = [s for s in ST if s.startswith(chain + "/")]
    return sorted(ids, key=lambda s: ST[s]["index_in_chain"])


def maxabs_diff(Ua, Ub):
    out = {"A": 0.0, "B": 0.0}
    for k in Ua:
        out[k[0]] = max(out[k[0]], float(np.max(np.abs(Ua[k] - Ub[k]))))
    return out


def ndiff(Ua, Ub):
    out = {"A": 0, "B": 0}
    for k in Ua:
        out[k[0]] += int(np.sum(Ua[k] != Ub[k]))
    return out


def rel_pair(Ua, Ub):
    res = {"A": [], "B": []}
    for k in Ua:
        res[k[0]].append(relF(Ua[k], Ub[k]))
    return {k: summ(v) for k, v in res.items()}


def norm_check(U, lr, r):
    v = [np.linalg.norm(u) / R.rank_scale(r, lr) for u in U.values() if np.any(u)]
    return (min(v), max(v)) if v else (float("nan"), float("nan"))


def regime():
    """P6 step 1 and the init oracle vs the D2 references (R1, R2)."""
    import pathlib
    D2 = R.ROOT / "runs" / "D2"
    s1 = ST["P6_r8/step1"]
    say("P6_r8 step1 before", s1["before"], "after", s1["after"],
        "| D2 r8 b0.9 T0 step1 after: R1 3c4579d4aa8db69d, R2 3362e501604407c2")
    OUTJ["step1_after"] = s1["after"]
    G = grad(s1["grad_key"])
    for ref in ("0ff0b9c991af1568_t0_w1_O1", "0ff0b9c991af1568_t0_w1_O1r2"):
        Z = np.load(D2 / "grads" / f"{ref}.npz")
        nd, n = 0, 0
        for k in G.files:
            a, b = G[k], np.asarray(Z[k], np.float64)
            nd += int(np.sum(a != b)); n += a.size
        say(f"  init oracle vs D2 {ref}: {nd} of {n} elements differ")
        OUTJ[f"init_vs_{ref}"] = nd
    if "DRIFT_end_r8/step1" in ST:
        say("  end-of-session drift check: new client step1 after =", ST["DRIFT_end_r8/step1"]["after"],
            "identical to P6 step1:", ST["DRIFT_end_r8/step1"]["after"] == s1["after"])
    info = M["grads"][s1["grad_key"]]
    say("  init oracle: max rel snap residual %.2e, |G_A| %.3g, |G_B| %.4g" %
        (info["max_rel_snap_residual"], info["GA_global"], info["GB_global"]))


def p6():
    ids = recs_of("P6_r8")
    say("P6 -- rank 8, b0.9, 12 steps texts 0,1,2 cycled; prediction rolled from oracle gradients only")
    prev = None
    res = {}
    for i, sid in enumerate(ids):
        t = sid.split("/")[1]
        Um = measured(ST[sid])
        say(f"{t} (text {ST[sid]['text']})  norm/law {norm_check(Um, 1e-3, 8)}")
        for v in ("bf16", "f32", "nostate"):
            P, meta = read_pred("P6", f"{t}_{v}")
            sc = score(P, Um); row(f"{t} rule[{v}]", sc); res[(t, v)] = sc
        if prev is not None:
            sc = score(prev, Um); row(f"{t} null: previous measured update", sc); res[(t, "prev")] = sc
        prev = Um
    for k in (3, 7, 11):
        sid = f"P6_fresh_ck{k}/step"
        if sid in ST:
            Um = measured(ST[sid])
            P, _ = read_pred("P6", f"step{k + 1}_nostate")
            sc = score(P, Um); row(f"floor step{k + 1}: fresh twin vs PE5(g)", sc)
            res[(f"step{k + 1}", "floor")] = sc
    json.dump({f"{a}|{b}": v for (a, b), v in res.items()}, open(P3 / "p6_table.json", "w"), indent=1)


def fit_c(S, g, Um, keys, lo=1e-3, hi=1.5, beta=0.9, lr=1e-3, r=8):
    """per-run c minimising the median residual over `keys` (golden section in log c)."""
    def f(lc):
        c = float(np.exp(lc))
        return float(np.median([relF(rule(S[k], g[k], beta, lr, r, c=c)[0], Um[k]) for k in keys]))
    a, b = np.log(lo), np.log(hi)
    gr = (np.sqrt(5) - 1) / 2
    c1, c2 = b - gr * (b - a), a + gr * (b - a)
    f1, f2 = f(c1), f(c2)
    for _ in range(40):
        if f1 < f2:
            b, c2, f2 = c2, c1, f1; c1 = b - gr * (b - a); f1 = f(c1)
        else:
            a, c1, f1 = c1, c2, f2; c2 = a + gr * (b - a); f2 = f(c2)
    lc = (a + b) / 2
    return float(np.exp(lc)), f(lc)


def rolled_state(ids, keys):
    S = {k: None for k in keys}
    for sid in ids:
        G = grad(ST[sid]["grad_key"])
        for k in keys:
            _, S[k] = rule(S[k], gget(G, *k), ST[sid]["params"]["beta1"], 1e-3, 8)
    return S


def p1a():
    import p3_collect as C
    say("P1a -- clip series at step 2 (P6 state 1, T1), restored with optimizer")
    Uref = measured(ST["P6_r8/step2"])
    keys = sorted(Uref)
    S1 = rolled_state(["P6_r8/step1"], keys)
    G = grad(ST["P6_r8/step2"]["grad_key"])
    g = {k: gget(G, *k) for k in keys}
    sub = [k for k in keys if k[0] == "B"][::4]
    for c in [1.0] + C.P1A_C:
        sid = "P6_r8/step2" if c == 1.0 else f"P1a_c{c}/step2"
        Um = measured(ST[sid])
        say(f"c={c}  ({sid})  A bit-identical to unclipped: {ndiff(Um, Uref)['A'] == 0}  "
            f"[{ndiff(Um, Uref)}]  relF vs unclipped {fmt(rel_pair(Um, Uref)['B'])}")
        for nd in ("all", "B"):
            P, meta = read_pred("P1a", f"c{c}_{nd}")
            row(f"c={c} rule, N={nd} (c_used {meta['c_used']:.4f})", score(P, Um))
        P, _ = read_pred("P1a", "c1.0_all")
        row(f"c={c} null: unclipped prediction", score(P, Um))
        if c != 1.0:
            cf, fv = fit_c(S1, g, Um, sub)
            say(f"    fitted c (B, every 4th tensor) = {cf:.4f}  median relF {fv:.2e};  clip/N_all = {c:.4f}")
            OUTJ[f"cfit_{c}"] = dict(c_fit=cf, resid=fv, c_nominal=c)


def p1b():
    import p3_collect as C
    T = C.p1b_thresholds()
    say("P1b -- clip thresholds at state 6 (T0):", {k: round(v, 4) for k, v in T.items()})
    Uref = measured(ST["P6_r8/step7"])
    keys = sorted(Uref)
    S6 = rolled_state(ST["P6_r8/step7"]["hist"], keys)
    G = grad(ST["P6_r8/step7"]["grad_key"])
    g = {k: gget(G, *k) for k in keys}
    for t in ("t1", "t2", "t3"):
        Um = measured(ST[f"P1b_{t}/step7"])
        nd = ndiff(Um, Uref)
        say(f"{t} clip={T[t]:.4f}: differing elements vs unclipped trunk step 7: A {nd['A']}, B {nd['B']}")
        OUTJ[f"{t}_ndiff"] = nd
        for h in ("all", "B", "pertype", "pertensor", "none"):
            P, meta = read_pred("P1b", f"{t}_{h}")
            row(f"{t} hyp={h}", score(P, Um))
        for kind in "AB":
            sub = [k for k in keys if k[0] == kind][::4]
            if nd[kind] == 0:
                say(f"    {kind}: unchanged -> c = 1"); continue
            cf, fv = fit_c(S6, g, Um, sub)
            cand = dict(all=T[t] / T["N"], B=T[t] / T["NB"], pertype=T[t] / (T["NA"] if kind == "A" else T["NB"]))
            say(f"    {kind}: fitted c = {cf:.5f} (median relF {fv:.2e}); clip/N candidates: " +
                ", ".join(f"{a} {min(1, v):.5f}" for a, v in cand.items()))
            OUTJ[f"{t}_{kind}_cfit"] = dict(c_fit=cf, resid=fv, **cand)


def p2():
    ids = recs_of("P2_r32")
    say("P2 -- rank 32, b0.9: T0, T0, zero, zero, zero")
    Ums = [measured(ST[s]) for s in ids]
    for sid, Um in zip(ids, Ums):
        t = sid.split("/")[1]
        say(f"{t}: norm/law {norm_check(Um, 1e-3, 32)}")
        for v in ("bf16_x0f16", "f32_x0f16", "bf16_f64", "f32_f64"):
            if not have("P2", f"{t}_{v}"):
                continue
            P, _ = read_pred("P2", f"{t}_{v}")
            row(f"{t} rule[{v}]", score(P, Um))
    say("successive-update cosines (median over tensors; measured vs predicted)")
    for i in range(1, len(ids)):
        a, b = ids[i].split("/")[1], ids[i - 1].split("/")[1]
        line = f"  cos({a},{b}) measured A {np.median([cos(Ums[i][k], Ums[i-1][k]) for k in Ums[i] if k[0]=='A']):.6f}" \
               f" B {np.median([cos(Ums[i][k], Ums[i-1][k]) for k in Ums[i] if k[0]=='B']):.6f}"
        for v in ("bf16_x0f16", "f32_x0f16", "bf16_f64"):
            if not have("P2", f"{a}_{v}"):
                continue
            Pa, _ = read_pred("P2", f"{a}_{v}"); Pb, _ = read_pred("P2", f"{b}_{v}")
            ca = np.median([cos(Pa[k], Pb[k]) for k in Pa if k[0] == "A"])
            cb = np.median([cos(Pa[k], Pb[k]) for k in Pa if k[0] == "B"])
            line += f" | {v} A {ca:.6f} B {cb:.6f}"
            OUTJ[f"cos_{a}_{b}_{v}"] = (ca, cb)
        say(line)


def p3():
    import p3_collect as C
    say("P3 -- zero-gradient step at beta' from P6 state 2 (b0.9 history), then a T2 step at beta 0.9")
    Uz = {b: measured(ST[f"P3_zb{b}/zero"]) for b in C.P3_B}
    for b in C.P3_B:
        mx = max(float(np.max(np.abs(u))) for u in Uz[b].values())
        say(f"beta'={b}: max |dW| of zero step {mx:.3e}")
        if mx > 0:
            P, _ = read_pred("P3", f"zero_b{b}")
            row(f"zero b'={b} rule", score(P, Uz[b]))
    for a in C.P3_B:
        for b in C.P3_B:
            if a < b and a > 0:
                say(f"  zero step b'={a} vs b'={b}: differing elements {ndiff(Uz[a], Uz[b])}, "
                    f"max|diff| {maxabs_diff(Uz[a], Uz[b])}, relF {fmt(rel_pair(Uz[a], Uz[b])['A'])} (A) "
                    f"{fmt(rel_pair(Uz[a], Uz[b])['B'])} (B)")
                Pa, _ = read_pred("P3", f"zero_b{a}"); Pb, _ = read_pred("P3", f"zero_b{b}")
                say(f"    predicted: differing elements {ndiff(Pa, Pb)}  relF B {fmt(rel_pair(Pa, Pb)['B'])}")
    for b in C.P3_B:
        Um = measured(ST[f"P3_zb{b}/step3"])
        P, _ = read_pred("P3", f"step3_b{b}"); row(f"step3 after zero b'={b} rule", score(P, Um))
        P, _ = read_pred("P3", f"step3_b{b}_null09"); row(f"step3 after zero b'={b} null(b'=0.9)", score(P, Um))


def p4():
    import p3_collect as C
    for r, lrs, ref, pre in ((32, C.P4_R32, "P4_r32_lr0.001/step3", "P4_r32_lr"),
                             (8, [1e-3] + C.P4_R8, "P6_r8/step3", "P4_r8_lr")):
        say(f"P4 -- rank {r}, step 3 from state 2, lr series (restored with optimizer)")
        Uref = measured(ST[ref])
        W0 = state(ST[ref]["before"])
        for lr in lrs:
            sid = ref if lr == 1e-3 else f"{pre}{lr:g}/step3"
            Um = measured(ST[sid])
            Us = {k: u * (1e-3 / lr) for k, u in Um.items()}
            pr = rel_pair(Us, Uref)
            gA = np.sqrt(sum(np.sum(u ** 2) for k, u in Um.items()))
            ratioA = [np.linalg.norm(Um[k]) / np.linalg.norm(wget(W0, *k)) for k in Um if k[0] == "A"]
            ratioB = [np.linalg.norm(Um[k]) / np.linalg.norm(wget(W0, *k)) for k in Um if k[0] == "B"]
            say(f" lr {lr:g}: U*1e-3/lr vs U(1e-3): A {fmt(pr['A'])}  B {fmt(pr['B'])}; global ||lr U|| {gA:.4f};"
                f" per-tensor ||lrU||/||W|| A med {np.median(ratioA):.4f} max {np.max(ratioA):.4f}, B med {np.median(ratioB):.4f}")
            OUTJ[f"r{r}_lr{lr:g}_prop"] = pr
            for mode in ("x0f16", "f64", "fused16", "fused32", "fusedbf16"):
                if not have("P4", f"r{r}_lr1e-3_{mode}"):
                    continue
                P, _ = read_pred("P4", f"r{r}_lr1e-3_{mode}")
                row(f"r{r} lr {lr:g} rule[{mode}] (scaled)", score(P, Us))
        if r == 32:
            Uf1 = measured(ST["P4_r32_fresh_lr0.001/step3"])
            Uf2 = measured(ST["P4_r32_fresh_lr0.01/step3"])
            pr = rel_pair({k: u / 10 for k, u in Uf2.items()}, Uf1)
            say(f" FRESH r32: U(1e-2)/10 vs U(1e-3): A {fmt(pr['A'])}  B {fmt(pr['B'])}")
            OUTJ["r32_fresh_prop"] = pr
            for mode in ("x0f16", "f64", "fused16"):
                P, _ = read_pred("P4", f"r32_fresh_lr1e-3_{mode}")
                row(f"fresh lr 1e-3 PE5[{mode}]", score(P, Uf1))
                row(f"fresh lr 1e-2 PE5[{mode}] (scaled)", score(P, {k: u / 10 for k, u in Uf2.items()}))


def p5():
    D2 = R.ROOT / "runs" / "D2"
    D2M = json.load(open(D2 / "manifest.json"))
    orig = {"r32_ck1_t1": ("S1v2_r32_fresh_ck1_t1/step", "6b6b57f85d3dbb33_t1_w1_O1r2"),
            "r8b0_ck2_t1": ("S2_r8_b0_fresh_ck2/step3", "7bdecab086fd9a60_t1_w1_O1r2"),
            "r8b0_ck3_t2": ("S2_r8_b0_fresh_ck3/step4", "48f87021a2ccf9b9_t2_w1_O1r2")}
    for name, sid in (("r32_ck1_t1", "P5_fresh_r32_ck1_t1/step"),
                      ("r8b0_ck2_t1", "P5_fresh_r8b0_ck2_t1/step"),
                      ("r8b0_ck3_t2", "P5_fresh_r8b0_ck3_t2/step")):
        rec = ST[sid]
        d2step, d2key = orig[name]
        d2rec = D2M["steps"][d2step]
        say(f"P5 {name}: state {rec['before']} (D2 {d2rec['before']}, same: {rec['before'] == d2rec['before']});"
            f" fresh step after {rec['after']} (D2 {d2rec['after']}, same: {rec['after'] == d2rec['after']})")
        k1 = rec["grad_key"].rsplit("_", 1)[0]
        Ga, Gb = grad(k1 + "_P3"), grad(k1 + "_P3b")
        nd = sum(int(np.sum(Ga[k] != Gb[k])) for k in Ga.files)
        say(f"   oracle P3 vs P3b (same session): {nd} differing elements of {sum(Ga[k].size for k in Ga.files)}")
        if rec["before"] == d2rec["before"]:
            Z = np.load(D2 / "grads" / f"{d2key}.npz")
            nd2 = {"A": 0, "B": 0}; rel2 = {"A": [], "B": []}
            for k in Ga.files:
                a, b = Ga[k], np.asarray(Z[k], np.float64)
                nd2[k[1]] += int(np.sum(a != b))
                rel2[k[1]].append(relF(a, b) if np.any(b) else 0.0)
            say(f"   oracle now vs D2 stored {d2key}: differing elements {nd2}; per-tensor relF A {fmt(summ(rel2['A']))} B {fmt(summ(rel2['B']))}")
            OUTJ[f"{name}_vsD2"] = dict(ndiff=nd2, relA=summ(rel2["A"]), relB=summ(rel2["B"]))
        Um = measured(rec)
        for s in ("P3", "P3b"):
            P, _ = read_pred("P5", f"{name}_{s}")
            row(f"{name} fresh step vs PE5(G_{s})", score(P, Um))
        # the D2 fresh measurement against today's oracle
        W0 = R.Lazy(D2 / "states" / f"{d2rec['before']}.npz"); W1 = R.Lazy(D2 / "states" / f"{d2rec['after']}.npz")
        Ud2 = {(kk, m): wget(W1, kk, m) - wget(W0, kk, m) for kk, m in tensors(W0)}
        P, _ = read_pred("P5", f"{name}_P3")
        row(f"{name} D2 fresh step vs PE5(G_now)", score(P, Ud2))


def chain_score(exp, chain, lr=1e-3, r=8):
    ids = recs_of(chain)
    prev = None
    for sid in ids:
        t = sid.split("/")[1]
        Um = measured(ST[sid])
        say(f"{t} (text {ST[sid]['text']}) norm/law {norm_check(Um, lr, r)}")
        P, _ = read_pred(exp, t); row(f"{t} rule", score(P, Um))
        P, _ = read_pred(exp, f"{t}_nostate"); row(f"{t} null: no state", score(P, Um))
        if prev is not None:
            row(f"{t} null: previous update", score(prev, Um))
        prev = Um
    return ids


def p7():
    say("P7 -- openai/gpt-oss-20b, rank 8 attention LoRA, b0.9 texts 0,0,1,2,0")
    ids = recs_of("P7_r8")
    g0 = M["grads"][ST[ids[0]]["grad_key"]]
    say(f"init oracle: max rel snap residual {g0['max_rel_snap_residual']:.2e}, n>1e-3 {g0['n_snap_resid_gt_1em3']},"
        f" |G_A| {g0['GA_global']}, |G_B| {g0['GB_global']:.4g}, exact zeros {g0['n_exact_zero']}")
    for sid in ids:
        g = M["grads"][ST[sid]["grad_key"]]
        say(f"  {sid}: oracle snap resid {g['max_rel_snap_residual']:.2e} |G_A| {g['GA_global']:.4g} |G_B| {g['GB_global']:.4g}")
    W = state(ST[ids[0]]["before"])
    shapes = sorted({(k, tuple(W[f"{k}/{m}.lora_{k}"].shape)) for k, m in tensors(W)})
    say("tensor shapes (kind, stored shape):", shapes, "n tensors", len(tensors(W)))
    chain_score("P7", "P7_r8")
    for nm in ("b0_cont", "b0_fresh"):
        Um = measured(ST[f"P7_{nm}/step3"])
        P, _ = read_pred("P7", nm); row(f"{nm} rule", score(P, Um))
    Uc, Uf = measured(ST["P7_b0_cont/step3"]), measured(ST["P7_b0_fresh/step3"])
    say(f"  b0 continuing vs fresh: differing {ndiff(Uc, Uf)}, relF A {fmt(rel_pair(Uc, Uf)['A'])} B {fmt(rel_pair(Uc, Uf)['B'])}")


def p8():
    say("P8 -- Qwen3.5-4B rank 8, attention + MLP + unembed, b0.9 texts 0,1,2")
    ids = recs_of("P8_r8")
    W = state(ST[ids[0]]["before"])
    T = tensors(W)
    groups = {}
    for k, m in T:
        typ = m.split(".")[-1]
        groups.setdefault(typ, set()).add(tuple(W[f"{k}/{m}.lora_{k}"].shape) if k == "B" else None)
    say("module types and lora_B shapes:", {t: sorted(s - {None}) for t, s in groups.items()}, "n", len(T))
    g0 = M["grads"][ST[ids[0]]["grad_key"]]
    say(f"init oracle: snap resid {g0['max_rel_snap_residual']:.2e}, |G_A| {g0['GA_global']}, |G_B| {g0['GB_global']:.4g}")
    for sid in ids:
        t = sid.split("/")[1]
        Um = measured(ST[sid])
        P, _ = read_pred("P8", t); Pn, _ = read_pred("P8", f"{t}_nostate")
        for typ in sorted(groups):
            sub = {k: u for k, u in Um.items() if k[1].split(".")[-1] == typ}
            nv = [np.linalg.norm(u) / R.rank_scale(8, 1e-3) for u in sub.values() if np.any(u)]
            row(f"{t} {typ} rule (norm/law {min(nv) if nv else 0:.6f}..{max(nv) if nv else 0:.6f})", score(P, sub))
            row(f"{t} {typ} null: no state", score(Pn, sub))


def solo():
    """solo follow-ups: r32 lr pair (P4s), clip t1 vs unclipped at state 6 (P1bs), P1 state-1 repeats."""
    import p3_collect as C
    say("SOLO r32 lr pair from P2 ck2 (T1), strictly sequential:")
    a, b = measured(ST["P4s_r32_lr0.001/step3"]), measured(ST["P4s_r32_lr0.01/step3"])
    pr = rel_pair({k: u / 10 for k, u in b.items()}, a)
    say(f"  U(1e-2)/10 vs U(1e-3): A {fmt(pr['A'])}  B {fmt(pr['B'])}"); OUTJ["P4s_prop"] = pr
    for lr in C.P4_R32:
        u = measured(ST[f"P4_r32_lr{lr:g}/step3"])
        q = rel_pair({k: v * (1e-3 / lr) for k, v in u.items()}, a)
        say(f"  concurrent lr {lr:g} (scaled) vs solo lr 1e-3: A {fmt(q['A'])} B {fmt(q['B'])}  "
            f"cc {ST[f'P4_r32_lr{lr:g}/step3']['fb']['clock_cycle:unique']:.0f}")
    for nm in ("P4s_r32_lr0.001", "P4s_r32_lr0.01"):
        say(f"  {nm} cc {ST[nm + '/step3']['fb']['clock_cycle:unique']:.0f} after {ST[nm + '/step3']['after']}")
    for mode in ("x0f16", "f64", "fused16"):
        P, _ = read_pred("P4", f"r32_lr1e-3_{mode}")
        row(f"solo r32 lr1e-3 rule[{mode}]", score(P, a))
        row(f"solo r32 lr1e-2 rule[{mode}] (scaled)", score(P, {k: u / 10 for k, u in b.items()}))
    say("SOLO clip at P6 state 6 (T0): unclipped vs clip t1 (N_B < t1 < N_all)")
    T = C.p1b_thresholds()
    u0, u1 = measured(ST["P1bs_none/step7"]), measured(ST["P1bs_t1/step7"])
    ut = measured(ST["P6_r8/step7"])
    say(f"  t1={T['t1']:.4f} N_B={T['NB']:.4f} N_all={T['N']:.4f}: differing elements solo-t1 vs solo-unclipped {ndiff(u1, u0)}; "
        f"relF A {fmt(rel_pair(u1, u0)['A'])} B {fmt(rel_pair(u1, u0)['B'])}")
    say(f"  solo unclipped vs trunk step 7 (concurrent era): differing {ndiff(u0, ut)}")
    for t in ("t1", "t2", "t3"):
        uc = measured(ST[f"P1b_{t}/step7"])
        say(f"  concurrent P1b_{t} vs solo unclipped: differing {ndiff(uc, u0)}")
    R.SUFFIX = ""   # P1b predictions exist only in the pre-registered (concurrent-oracle) version
    for h in ("all", "B", "none"):
        P, _ = read_pred("P1b", f"t1_{h}")
        row(f"solo t1 hyp={h}", score(P, u1))
    P, _ = read_pred("P1b", "t1_none"); row("solo unclipped vs rule", score(P, u0))
    # the size of a c = 0.998 effect, predicted: rule(c=t1/N) vs rule(c=1)
    Pa, _ = read_pred("P1b", "t1_all"); Pn, _ = read_pred("P1b", "t1_none")
    say(f"  predicted effect of c={T['t1'] / T['N']:.5f}: relF(rule c, rule 1) A {fmt(rel_pair(Pa, Pn)['A'])} B {fmt(rel_pair(Pa, Pn)['B'])}")
    # clip-c fit uncertainty: fit c on 4 disjoint quarters of each factor, at t2 and t3
    keys = sorted(ut)
    S6 = rolled_state(ST["P6_r8/step7"]["hist"], keys)
    G = grad(ST["P6_r8/step7"]["grad_key"]); g = {k: gget(G, *k) for k in keys}
    for t in ("t2", "t3"):
        Um = measured(ST[f"P1b_{t}/step7"])
        for kind in "AB":
            ks = [k for k in keys if k[0] == kind]
            fits = [fit_c(S6, g, Um, ks[q::4])[0] for q in range(4)]
            say(f"  {t} {kind}: c fitted on 4 disjoint quarters {np.round(fits, 5)}; all-tensor N -> {T[t] / T['N']:.5f}, "
                f"B-only N -> {T[t] / T['NB']:.5f}" + (f", per-type -> {min(1, T[t] / T['NA']):.5f}" if kind == 'A' else ""))
            OUTJ[f"cfit_quarters_{t}_{kind}"] = fits


if __name__ == "__main__":
    exp = sys.argv[1]
    globals()[exp]()
    save(exp)
