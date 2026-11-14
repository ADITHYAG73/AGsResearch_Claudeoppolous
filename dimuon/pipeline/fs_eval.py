"""fs_eval.py -- per-rank residual table for candidate rules vs measured U (FIRSTSTEP-01).
Writes runs/FS_cache/eval_<tag>.json with per-module relF and maxabs."""
import sys, json, numpy as np
sys.path.insert(0, '/home/user/AGsResearch_Claudeoppolous/dimuon/pipeline')
from fs_load import load, RANKS, CACHE
from fs_sim import ns, rank_scale, PE5
from fs_spec import PE_MEM, safety, MUON

def polar(G):
    P, S, Vt = np.linalg.svd(G, full_matrices=False); return P @ Vt

def rule(name):
    lr = 1e-3
    def fin(X, r): return -rank_scale(r, lr) * X / np.linalg.norm(X)
    if name == 'polar': return lambda G: fin(polar(G), G.shape[1])
    if name == 'muon5_f64': return lambda G: fin(ns(G, MUON * 5, 'f64', k_sf=1.0), G.shape[1])
    if name == 'pe5_nosafety_f64': return lambda G: fin(ns(G, PE_MEM[:5], 'f64', k_sf=1.0), G.shape[1])
    if name == 'pe5_f64': return lambda G: fin(ns(G, PE5, 'f64'), G.shape[1])
    if name == 'pe5_f32': return lambda G: fin(ns(G, PE5, 'f32'), G.shape[1])
    if name == 'pe5_f16': return lambda G: fin(ns(G, PE5, 'f16', fused=True), G.shape[1])
    if name == 'pe5_f16_in_only':  # fp16 rounding of X0 only, exact arithmetic after
        def f(G):
            X0 = (G / (np.linalg.norm(G) * 1.01 + 1e-7)).astype(np.float32).astype(np.float16).astype(np.float64)
            return fin(ns(X0, PE5, 'f64', k_sf=1.0, eps=0.0) , G.shape[1]) if False else fin(_nsraw(X0, PE5), G.shape[1])
        return f
    if name == 'pe5_f16_unfused': return lambda G: fin(ns(G, PE5, 'f16', fused=False), G.shape[1])
    if name == 'pe5_bf16': return lambda G: fin(ns(G, PE5, 'bf16', fused=True), G.shape[1])
    if name == 'pe6_f64': return lambda G: fin(ns(G, safety(PE_MEM[:6], 1.01), 'f64'), G.shape[1])
    if name == 'pe4_f64': return lambda G: fin(ns(G, safety(PE_MEM[:4], 1.01), 'f64'), G.shape[1])
    raise ValueError(name)

def _nsraw(X0, coefs):
    X = X0.T.copy()
    for a, b, c in coefs:
        A = X @ X.T; B = b * A + c * (A @ A); X = a * X + B @ X
    return X.T

def evaluate(names, ranks=RANKS, dim='lr1e-3', oracle=None, stride=1, lr=1e-3, scale_U=1.0):
    out = {}
    for r in ranks:
        mods = load(r, dim, oracle)[::stride]
        for nm in names:
            f = rule(nm); rel = []; mx = []; mxr = []
            for n, G, U in mods:
                Uh = f(G) * (lr / 1e-3); Us = U
                rel.append(np.linalg.norm(Uh - Us) / np.linalg.norm(Us)); mx.append(np.abs(Uh - Us).max()); mxr.append(np.abs(Uh-Us).max()/np.abs(Us).max())
            out.setdefault(nm, {})[r] = dict(relF_med=float(np.median(rel)), relF_max=float(np.max(rel)), relF_mean=float(np.mean(rel)),
                                              maxabs_med=float(np.median(mx)), maxabs_max=float(np.max(mx)), maxabs_over_maxU_med=float(np.median(mxr)),
                                              n=len(rel), relF_all=[float(v) for v in rel])
    return out

if __name__ == '__main__':
    names = sys.argv[1].split(',')
    tag = sys.argv[2]; stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    res = evaluate(names, stride=stride)
    json.dump(res, open(CACHE / f'eval_{tag}.json', 'w'))
    for nm in names:
        print(nm)
        for r, v in res[nm].items():
            print('  r=%-3d relF med %.2e mean %.2e max %.2e | maxabs med %.2e max %.2e (max|U| ratio %.2e)' % (r, v['relF_med'], v['relF_mean'], v['relF_max'], v['maxabs_med'], v['maxabs_max'], v['maxabs_over_maxU_med']))
