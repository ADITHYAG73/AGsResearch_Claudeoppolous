"""S0 checks, read-only on runs/D2 + runs/D1."""
import json
import numpy as np
import s_d2 as d

m = d.man()
T = "S2_r8_b0.9"


def cmp(a, b, label):
    keys = sorted(a)
    assert keys == sorted(b), label
    out = {}
    for part in ("A", "B"):
        ks = [k for k in keys if k.split("/")[0].endswith(part)]
        x = np.concatenate([a[k].ravel().astype(np.float64) for k in ks])
        y = np.concatenate([b[k].ravel().astype(np.float64) for k in ks])
        nd = int((x != y).sum())
        rel = float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-300))
        out[part] = dict(n=x.size, n_diff=nd, rel_fro=rel, max_abs=float(np.max(np.abs(x - y))),
                         y_fro=float(np.linalg.norm(y)))
    print(label, json.dumps(out))
    return out


print("== reload checks (export of a loaded client vs the saved state's export)")
ev = [e for e in m["events"] if e["kind"] == "reload_check"]
print(" chains:", len(ev), "bit-identical:", sum(e["bit_identical"] for e in ev),
      [e["chain"] for e in ev if not e["bit_identical"]])
gr = list(m["grads"].values())
print(" oracles:", len(gr), "bit-identical:", sum(g["load_bit_identical"] for g in gr))
errs = [e for e in m["events"] if e["kind"] == "error"]
print(" errors:", errs)

print("== oracle eps/lr invariance at nonzero B (", T, "ck2, T1 )")
sid2 = m["ckpts"][f"{T}-ck2"]["sid"]
G = {s: d.load_grad(d.gkey(sid2, 1, 1.0, s)) for s in ("O1", "O2", "O3")
     if d.gkey(sid2, 1, 1.0, s) in m["grads"]}
for s in G:
    print(" ", s, {k: m["grads"][d.gkey(sid2, 1, 1.0, s)][k] for k in
                   ("max_rel_snap_residual", "n_snap_resid_gt_1em3", "n_exact_zero",
                    "GA_global", "GB_global")})
if "O2" in G:
    cmp(G["O2"], G["O1"], " O2 vs O1")
if "O3" in G:
    cmp(G["O3"], G["O1"], " O3 vs O1")
    # where O3 differs, how small is |G|?
    for part in ("GA", "GB"):
        ks = [k for k in G["O1"] if k.startswith(part)]
        x = np.concatenate([G["O3"][k].ravel() for k in ks]).astype(np.float64)
        y = np.concatenate([G["O1"][k].ravel() for k in ks]).astype(np.float64)
        dif = x != y
        if dif.any():
            print(f"  {part}: O3!=O1 at {dif.sum()} elems; |G_O1| there median "
                  f"{np.median(np.abs(y[dif])):.3g} max {np.abs(y[dif]).max():.3g}; "
                  f"|G_O1| overall median {np.median(np.abs(y)):.3g}")

print("== oracle at init (ck0, T0) vs D1 r8_oracle")
sid0 = m["ckpts"][f"{T}-ck0"]["sid"]
k0 = d.gkey(sid0, 0, 1.0, "O1")
if k0 in m["grads"]:
    g0 = d.load_grad(k0)
    D1 = np.load(d.tk.ROOT / "runs/D1/r8_oracle.npz")
    gb = [k for k in g0 if k.startswith("GB/")]
    nd = sum(int((g0[k] != D1["G/" + k[3:]]).sum()) for k in gb)
    print(" GB n_diff vs D1 G:", nd, "of", sum(g0[k].size for k in gb))
    print(" GA max|.| at B=0:", max(float(np.abs(g0[k]).max()) for k in g0 if k.startswith("GA/")))

print("== continue in original vs save + restore with optimizer + continue")
for br, orig in (("S0_opt_ck1/step2", f"{T}/step2"), ("S0_opt_ck2/step3", f"{T}/step3")):
    if br not in m["steps"]:
        print(" missing", br); continue
    a, b = m["steps"][br], m["steps"][orig]
    print(f" {br}: before same {a['before'] == b['before']} after same {a['after'] == b['after']}"
          f" loss {a['fb']['loss:sum']} vs {b['fb']['loss:sum']}")
    if a["after"] != b["after"]:
        wa, wb, w0 = d.load_state(a["after"]), d.load_state(b["after"]), d.load_state(a["before"])
        cmp(wa, wb, "  after(restored) vs after(orig)")
        ua = {k: wa[k].astype(np.float64) - w0[k] for k in wa}
        ub = {k: wb[k].astype(np.float64) - w0[k] for k in wb}
        cmp(ua, ub, "  update(restored) vs update(orig)")

print("== fresh Dimuon from init checkpoint vs D1 / new-client step 1")
if "S0_fresh_ck0/step1" in m["steps"]:
    a = m["steps"]["S0_fresh_ck0/step1"]
    print(" after == trunk step1 after:", a["after"] == m["steps"][f"{T}/step1"]["after"])
    w = d.load_state(a["after"])
    D1 = np.load(d.tk.ROOT / "runs/D1/r8_dim_lr1e-3.npz")
    print(" B n_diff vs D1 U:", sum(int((w["B/" + k[2:]] != D1[k]).sum()) for k in D1.files))
