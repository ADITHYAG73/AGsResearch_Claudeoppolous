"""fs_norm.py -- FIRSTSTEP-01 item 3: is ||U||_F an explicit renormalisation target?
Per rank: ||U||/(lr sqrt(r) (r/32)^0.175) - 1, implied exponent, and the Frobenius norm of the
fp16 NS output X16 (recovered from U by the fp16-grid scale search, power of two fixed by the
pe5_f16 simulation) compared with sqrt(r)."""
import sys, json, numpy as np
sys.path.insert(0, '/home/user/AGsResearch_Claudeoppolous/dimuon/pipeline')
from fs_load import load, RANKS, CACHE
from fs_grid import recover
from fs_ns16 import ns16
from fs_spec import PE_MEM, safety
PE5 = safety(PE_MEM[:5], 1.01)
out = {}
for r in RANKS:
    mods = load(r); dev = []; xn = []; ong = []; xs = []
    tgt = 1e-3 * np.sqrt(r) * (r / 32) ** 0.175
    for i, (n, G, U) in enumerate(mods):
        nu = np.linalg.norm(U.astype(np.float32).astype(np.float64)); dev.append(nu / tgt - 1)
        if i % 4 == 0:
            Xs = ns16(G, PE5); X16, c, md, f = recover(U, -Xs); xn.append(np.linalg.norm(X16)); ong.append(f); xs.append(np.linalg.norm(Xs))
    dev = np.array(dev)
    expo = np.log(np.median([np.linalg.norm(U) for _, _, U in mods]) / (1e-3 * np.sqrt(r))) / np.log(r / 32) if r != 32 else float('nan')
    out[r] = dict(dev_med=float(np.median(dev)), dev_min=float(dev.min()), dev_max=float(dev.max()), implied_exponent=float(expo),
                  X16_norm_min=float(min(xn)), X16_norm_max=float(max(xn)), sim_norm_min=float(min(xs)), sim_norm_max=float(max(xs)),
                  ongrid_min=float(min(ong)), sqrt_r=float(np.sqrt(r)))
    print('r=%-3d ||U||/target-1: med %+.2e [%+.2e, %+.2e] implied exp %.6f | ||X16|| %.4f..%.4f (sim %.4f..%.4f) vs sqrt(r) %.3f | fp16-grid frac min %.4f' % (
        r, np.median(dev), dev.min(), dev.max(), expo, min(xn), max(xn), min(xs), max(xs), np.sqrt(r), min(ong)))
json.dump(out, open(CACHE / 'norm.json', 'w'), indent=1)
