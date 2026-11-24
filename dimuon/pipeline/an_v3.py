import numpy as np, pathlib
R = pathlib.Path(__file__).resolve().parents[1] / "runs"
def load(p):
    d = dict(np.load(p)); ks = sorted(k[3:] for k in d if k.startswith("B1/"))
    assert all(np.all(d["B0/"+k]==0) for k in ks)
    return {k: d["B1/"+k].astype(np.float64) for k in ks}
D = {n: load(R/"V3"/f"{n}.npz") for n in ["lr1_eps10","lr1_eps0.1","lr0.5_eps1","lr1_eps100","lr1e-3_eps1e-12","lr1_eps1_w2","lr1_eps1_b0.5_0.99"]}
D["lr1_eps1"] = load(R/"V12"/"adam_s0_r0.npz")
cfg = {"lr1_eps10":(1,10),"lr1_eps0.1":(1,.1),"lr0.5_eps1":(.5,1),"lr1_eps100":(1,100),"lr1_eps1":(1,1),"lr1_eps1_w2":(1,1),"lr1_eps1_b0.5_0.99":(1,1)}
def inv1(d, lr, eps): return -np.sign(d)*eps*np.abs(d)/(lr-np.abs(d))          # delta=-lr g/(|g|+eps)
def inv2(d, lr, eps): return -np.sign(d)*np.sqrt(eps)*np.abs(d)/np.sqrt(lr**2-d**2)  # delta=-lr g/sqrt(g^2+eps)
ks = sorted(D["lr1_eps1"])
def cat(G): return np.concatenate([G[k].ravel() for k in ks])
for inv, nm in [(inv1,"M1 |g|+eps"),(inv2,"M2 sqrt(g^2+eps)")]:
    G = {n: cat({k: inv(D[n][k], *cfg[n]) for k in ks}) for n in cfg}
    ref = G["lr1_eps100"]
    print(nm, " |G| range: max %.4g, median %.4g, frac zero %.4g" % (np.max(np.abs(ref)), np.median(np.abs(ref)), np.mean(ref==0)))
    for n in cfg:
        g = G[n] / (2.0 if n.endswith("w2") else 1.0)
        dd = g - ref; nz = np.abs(ref) > 0
        rel = np.abs(dd[nz])/np.abs(ref[nz])
        print(f"  {n:22s} relFro {np.linalg.norm(dd)/np.linalg.norm(ref):.3e}  elem rel: median {np.median(rel):.2e} p99 {np.quantile(rel,.99):.2e} max {rel.max():.2e}  sign agree {np.mean(np.sign(g)==np.sign(ref)):.6f}")
# sign regime
d = cat(D["lr1e-3_eps1e-12"]); ref = cat({k: inv1(D["lr1_eps100"][k],1,100) for k in ks})
print("eps=1e-12 lr=1e-3: |delta| unique (rounded 1e-9):", np.unique(np.round(np.abs(d),9))[:10], "count", d.size)
print("   frac |delta|==1e-3 in float32:", np.mean(np.abs(d).astype(np.float32)==np.float32(1e-3)), "sign(delta)==-sign(G):", np.mean(np.sign(d)==-np.sign(ref)))
small = np.abs(d) < 0.999e-3
print("   elements with |delta|<0.999e-3:", small.sum(), " their |G|:", np.abs(ref[small])[:10])
pred = -1e-3*ref/(np.abs(ref)+1e-12); print("   max |delta - pred M1|:", np.max(np.abs(d-pred)))
