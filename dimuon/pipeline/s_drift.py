"""DRIFT check: after ~19:30 UTC new trunks stopped reproducing D1's step 1. Re-measure now:
(a) new Dimuon r8 client, step 1 on T0 (b1 0.9) vs D1 r8_dim_lr1e-3; (b) new AdamW r8 client,
O1 oracle step at init on T0 vs the cached pre-19:30 gradient 0ff0b9c991af1568_t0_w1_O1."""
import sys, json, numpy as np
import s_d2 as d
tag = sys.argv[1]
out = {}
def dim():
    tc = d.new_client("new", 8, tag=f"drift_{tag}_dim")
    w0 = d.export_tmp(tc, f"drift_{tag}_dim-w0", 8)
    m = d.tk.step(tc, d.D(), text=0)
    w1 = d.export_tmp(tc, f"drift_{tag}_dim-w1", 8)
    D9 = np.load(d.tk.ROOT / "runs/D1/r8_dim_lr1e-3.npz")
    x = np.concatenate([w1["base_model.model.model." + k[2:] + ".weight"].ravel() for k in sorted(D9.files)])
    y = np.concatenate([D9[k].ravel() for k in sorted(D9.files)]).astype(np.float64)
    out["dim"] = dict(sid0=d.sid_of(w0), sid1=d.sid_of(w1), relF=float(np.linalg.norm(x - y) / np.linalg.norm(y)),
                      ndiff=int((x != y).sum()), loss=m["fb"]["loss:sum"], model_id=tc._tk_meta["model_id"])
def ora():
    tc = d.new_client("new", 8, optimizer="adamw", tag=f"drift_{tag}_or")
    w0 = d.export_tmp(tc, f"drift_{tag}_or-w0", 8)
    o = d.ORACLES["O1"]
    m = d.tk.step(tc, d.tk.adamw(o["lr"], eps=o["eps"]), text=0)
    w1 = d.export_tmp(tc, f"drift_{tag}_or-w1", 8)
    g = d.load_grad("0ff0b9c991af1568_t0_w1_O1")
    nd, tot, num, den = 0, 0, 0.0, 0.0
    for k in w0:
        if ".lora_B." not in k: continue
        dd = w1[k] - w0[k]; ad = np.abs(dd)
        gg = d.snap_bf16(-np.sign(dd) * o["eps"] * ad / (o["lr"] - ad))
        ref = g["GB/" + d.short(k)].astype(np.float64)
        nd += int((gg != ref).sum()); tot += gg.size; num += float(((gg - ref) ** 2).sum()); den += float((ref ** 2).sum())
    out["oracle"] = dict(sid0=d.sid_of(w0), ndiff=nd, n=tot, relF=(num / den) ** 0.5, loss=m["fb"]["loss:sum"],
                         model_id=tc._tk_meta["model_id"])
d.run([dim, ora])
print(json.dumps(out, indent=1))
json.dump(out, open(d.OUT / f"drift_{tag}.json", "w"), indent=1)
