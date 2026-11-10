import numpy as np, json, sys, pathlib, itertools
R = pathlib.Path(__file__).resolve().parents[1] / "runs" / "V12"
L = {n: dict(np.load(R / f"{n}.npz")) for n in ["dim_s0_r0","dim_s0_r1","dim_s0_r2","dim_s1","adam_s0_r0","adam_s0_r1","adam_s0_r2"]}
def keys(d, p): return sorted(k for k in d if k.startswith(p))
# V1
def cmp(a, b, pre):
    ks = keys(a, pre); assert ks == keys(b, pre)
    ident = all(np.array_equal(a[k], b[k]) for k in ks)
    mx = max(float(np.max(np.abs(a[k].astype(np.float64)-b[k]))) for k in ks)
    return ident, mx, len(ks)
print("V1 A0 s0 r0 vs r1 (dimuon):", cmp(L["dim_s0_r0"], L["dim_s0_r1"], "A0/"))
print("V1 A0 s0 r0 vs r2:", cmp(L["dim_s0_r0"], L["dim_s0_r2"], "A0/"))
print("V1 A0 dimuon s0 vs adamw s0:", cmp(L["dim_s0_r0"], L["adam_s0_r0"], "A0/"))
print("V1 A0 adam r1 vs r2:", cmp(L["adam_s0_r1"], L["adam_s0_r2"], "A0/"))
print("V1 A0 s0 vs s1:", cmp(L["dim_s0_r0"], L["dim_s1"], "A0/"))
print("V1 B0 all zero:", all(np.all(L[n][k]==0) for n in L for k in keys(L[n],"B0/")))
a, b = L["dim_s0_r0"], L["dim_s1"]
corr = [np.corrcoef(a[k].ravel(), b[k].ravel())[0,1] for k in keys(a,"A0/")]
print("  s0 vs s1 corr of A0: max |corr|", np.max(np.abs(corr)), "fro s0 mean", np.mean([np.linalg.norm(a[k]) for k in keys(a,"A0/")]), "fro s1 mean", np.mean([np.linalg.norm(b[k]) for k in keys(b,"A0/")]))
# V2
for grp in [["dim_s0_r0","dim_s0_r1","dim_s0_r2"], ["adam_s0_r0","adam_s0_r1","adam_s0_r2"]]:
    print("V2", grp[0][:4])
    ks = keys(L[grp[0]], "B1/")
    for x, y in itertools.combinations(grp, 2):
        ident = all(np.array_equal(L[x][k], L[y][k]) for k in ks)
        mx = []; rel = []; relmax = []; tot_d = 0; tot_u = 0
        for k in ks:
            u = L[x][k].astype(np.float64) - L[x]["B0/"+k[3:]]; v = L[y][k].astype(np.float64) - L[y]["B0/"+k[3:]]
            d = u - v
            mx.append(np.max(np.abs(d))); rel.append(np.linalg.norm(d)/np.linalg.norm(u)); relmax.append(np.max(np.abs(d))/np.max(np.abs(u)))
            tot_d += np.sum(d**2); tot_u += np.sum(u**2)
        n_ident = sum(np.array_equal(L[x][k], L[y][k]) for k in ks)
        print(f"  {x} vs {y}: bit-identical all={ident} ({n_ident}/{len(ks)} tensors); max|d| {max(mx):.3e}; per-tensor relFro median {np.median(rel):.3e} max {max(rel):.3e}; global relFro {np.sqrt(tot_d/tot_u):.3e}; max|d|/max|u| max {max(relmax):.3e}")
    u = {k: L[grp[0]][k].astype(np.float64) for k in ks}
    fr = np.array([np.linalg.norm(u[k]) for k in ks]); print("  update Fro: min %.4g median %.4g max %.4g; max|u| %.4g" % (fr.min(), np.median(fr), fr.max(), max(np.max(np.abs(u[k])) for k in ks)))
