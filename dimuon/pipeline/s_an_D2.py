"""Sanity statistics for D2 (no fitting). Read-only. Prints one block per experiment."""
import json
import numpy as np
import s_d2 as d

m = d.man()
S = m["steps"]
_cache = {}


def W(sid):
    if sid not in _cache:
        _cache[sid] = {k: v.astype(np.float64) for k, v in d.load_state(sid).items()}
    return _cache[sid]


def U(step_id):
    s = S[step_id]
    a, b = W(s["after"]), W(s["before"])
    return {k: a[k] - b[k] for k in a}


def flat(u, part):
    return np.concatenate([u[k].ravel() for k in sorted(u) if k.startswith(part + "/")])


def diff(u, v, label):
    out = {}
    for p in ("A", "B"):
        x, y = flat(u, p), flat(v, p)
        ny = np.linalg.norm(y)
        out[p] = dict(n_diff=int((x != y).sum()), n=x.size,
                      rel_fro=float(np.linalg.norm(x - y) / ny) if ny > 0 else None,
                      max_abs=float(np.abs(x - y).max()), fro_u=float(np.linalg.norm(x)),
                      fro_v=float(ny))
    parts = []
    for p, o in out.items():
        rf = "n/a" if o["rel_fro"] is None else f"{o['rel_fro']:.3e}"
        parts.append(f"{p}: ndiff {o['n_diff']}/{o['n']} relF {rf} maxabs {o['max_abs']:.3e} "
                     f"|u| {o['fro_u']:.6g} |v| {o['fro_v']:.6g}")
    print(f"  {label}: " + " | ".join(parts))
    return out


def line(step_id):
    s = S[step_id]["stats"]
    g = m["grads"].get(S[step_id]["grad_key"] or "", {})
    print(f"  {step_id:32s} [{S[step_id].get('regime', '?')}] t{S[step_id]['text']} w{S[step_id]['weight']:g} "
          f"b1={S[step_id]['params']['beta1']} lr={S[step_id]['params']['learning_rate']:g} "
          f"clip={S[step_id]['params']['grad_clip_norm']:.4g} | dA per-tensor F "
          f"{s['dA_fro_min']:.6g}/{s['dA_fro_median']:.6g}/{s['dA_fro_max']:.6g} "
          f"dB {s['dB_fro_min']:.6g}/{s['dB_fro_median']:.6g}/{s['dB_fro_max']:.6g} "
          f"A_moved={s['A_moved']} | loss {S[step_id]['fb'].get('loss:sum'):.5g} "
          f"|G_A| {g.get('GA_global', float('nan')):.4g} |G_B| {g.get('GB_global', float('nan')):.4g}")


def have(*ids):
    miss = [i for i in ids if i not in S]
    if miss:
        print("  MISSING", miss)
    return not miss


print("== grads: oracle quality")
for k, g in sorted(m["grads"].items()):
    print(f"  {k:36s} [{g.get('regime', '?')}] load_ok={g['load_bit_identical']} "
          f"resid {g.get('max_rel_snap_residual', float('nan')):.2e} "
          f">1e-3: {g.get('n_snap_resid_gt_1em3')} sat {g.get('n_saturated')} zeros {g.get('n_exact_zero')} "
          f"|GA| {g['GA_global']:.5g} |GB| {g['GB_global']:.5g}")
ev = [e for e in m["events"] if e["kind"] == "reload_check"]
print("== reload checks:", len(ev), "bit-identical", sum(e["bit_identical"] for e in ev),
      "| job errors logged (all from the storage-capped runs, see journal):",
      sum(e["kind"] == "error" for e in m["events"]),
      "| regen checks:", [(e["regen"], e["k"], e["bit_identical"]) for e in m["events"]
                          if e["kind"] == "regen_check"])

print("\n== S1 trunks + fresh-state branches")
for r, T in ((8, "S1_r8"), (32, "S1_r32"), (32, "S1v2_r32")):
    for i in range(1, 6):
        if f"{T}/step{i}" in S:
            line(f"{T}/step{i}")
    for k in (1, 2, 5):
        for t in (0, 1):
            sid = f"{T}_fresh_ck{k}_t{t}/step"
            if sid in S:
                line(sid)
                if t == 0 and f"{T}/step{k + 1}" in S:
                    diff(U(sid), U(f"{T}/step{k + 1}"), f"fresh T0 vs continuing T0 at state {k}")
    if have(f"S1_r{r}/step1", f"S1_r{r}/step2"):
        print("  S1 trunk states 1,2 == S2 b0.9 trunk states 1,2:",
              [S[f"S1_r{r}/step{i}"]["after"] == S.get(f"S2_r{r}_b0.9/step{i}", {}).get("after")
               for i in (1, 2)])

print("\n== S2 continuing runs")
for r, bs in ((8, (0.0, 0.5, 0.9, 0.99)), (32, (0.0, 0.9))):
    for b in bs:
        for i in range(1, 7):
            if f"S2_r{r}_b{b:g}/step{i}" in S:
                line(f"S2_r{r}_b{b:g}/step{i}")
    for i in range(1, 7):
        ids = [f"S2_r{r}_b{b:g}/step{i}" for b in bs if f"S2_r{r}_b{b:g}/step{i}" in S]
        print(f"  r{r} step{i} after-sids:", [S[x]["after"] for x in ids])
print("\n== S2 beta1=0 control: continuing vs fresh-state step, same weights, same batch")
for r in (8, 32):
    for k in range(1, 6):
        c, f = f"S2_r{r}_b0/step{k + 1}", f"S2_r{r}_b0_fresh_ck{k}/step{k + 1}"
        if c in S and f in S:
            assert S[c]["before"] == S[f]["before"]
            print(f"  r{r} step{k + 1}: after bit-identical = {S[c]['after'] == S[f]['after']}")
            diff(U(f), U(c), "fresh vs continuing")

print("\n== S3 beta1 changed at step 4 (restored with optimizer from b0.9 state 3)")
for r in (8, 32):
    base = f"S2_r{r}_b0.9/step4"
    for b in (0.9, 0.0, 0.99):
        x = f"S3_r{r}_b{b:g}/step4"
        if x in S and base in S:
            line(x)
            diff(U(x), U(base), f"S3 b1={b} vs trunk step4 (b1=0.9)")
    if r == 8 and "S3x_r8_fresh_ck3/step4" in S:
        line("S3x_r8_fresh_ck3/step4")
        assert S["S3x_r8_fresh_ck3/step4"]["before"] == S["S3_r8_b0/step4"]["before"]
        diff(U("S3_r8_b0/step4"), U("S3x_r8_fresh_ck3/step4"),
             "S3 b1=0 (restored state) vs fresh-state b1=0, same weights + batch")
    if f"S3_r{r}_b0/step4" in S and f"S3_r{r}_b0.99/step4" in S:
        diff(U(f"S3_r{r}_b0/step4"), U(f"S3_r{r}_b0.99/step4"), "S3 b0 vs b0.99")

print("\n== S4 zero-gradient steps")
zk = [k for k in m["grads"] if k.endswith("_w0_O1")]
for k in zk:
    g = d.load_grad(k)
    print(f"  oracle {k}: max|G| = {max(float(np.abs(v).max()) for v in g.values())}")
for c in ("S4_r8_cont", "S4_r32_cont", "S4_r8_b0_cont", "S4_r8_fresh_ck2", "S4_r32_fresh_ck2"):
    ids = sorted(x for x in S if x.startswith(c + "/"))
    for x in ids:
        line(x)
    if c.endswith("cont"):
        r = 8 if "_r8_" in c else 32
        tr = f"S2_r{r}_b{'0' if 'b0_' in c else '0.9'}"
        print("   first two states equal trunk:",
              [S[f"{c}/step{i}"]["after"] == S.get(f"{tr}/step{i}", {}).get("after") for i in (1, 2)])
        z = [x for x in ids if "/zero" in x]
        prev = f"{c}/step2"
        for x in z:
            u, v = U(x), U(prev)
            cos = []
            for k in u:
                nu, nv = np.linalg.norm(u[k]), np.linalg.norm(v[k])
                cos.append(float((u[k] * v[k]).sum() / (nu * nv)) if nu > 0 and nv > 0 else np.nan)
            print(f"   {x}: global |dA| {S[x]['stats']['dA_global']:.6g} |dB| "
                  f"{S[x]['stats']['dB_global']:.6g}; per-tensor cos with previous update "
                  f"min/median/max {np.nanmin(cos):.6f}/{np.nanmedian(cos):.6f}/{np.nanmax(cos):.6f}")
            prev = x

print("\n== S5 gradient clipping (rank 8)")
for st, base in (("step1", "S2_r8_b0.9/step1"), ("step2", "S2_r8_b0.9/step2")):
    for c in (2, 0.5, 0.01):
        x = f"S5_{st}_c{c:g}/{st}"
        if x in S and base in S:
            line(x)
            diff(U(x), U(base), f"clip {c}x vs unclipped")

print("\n== S6 lr linearity (restored with optimizer)")
for r, st, ck, base in ((8, "step3", "ck2", "S0_opt_ck2/step3"), (32, "step3", "ck2", None),
                        (8, "step4", "ck3", "S3_r8_b0.9/step4")):
    base = base or f"S6_r{r}_{ck}_lr0.001/{st}"
    if base not in S:
        print("  MISSING base", base); continue
    ub = U(base)
    for lr in (1e-4, 1e-2):
        x = f"S6_r{r}_{ck}_lr{lr:g}/{st}"
        if x in S:
            line(x)
            u = U(x)
            diff({k: v * (1e-3 / lr) for k, v in u.items()}, ub, f"U(lr={lr:g})*(1e-3/lr) vs U(1e-3)")
