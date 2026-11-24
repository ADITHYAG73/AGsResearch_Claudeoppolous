"""fs_final.py -- FIRSTSTEP-01 final evaluation table (all 152 modules x 7 ranks + rank-8 variants).
Rules (all end with U = -lr * sqrt(r) * (r/32)^0.175 * X / ||X||_F):
  polar        X = P Q^T (exact polar factor of G)
  muon5_f64    Muon quintic (3.4445,-4.7750,2.0315) x5, X0 = G/||G||_F, float64
  pe5raw_f64   Polar-Express first 5 steps, no 1.01 safety anywhere, float64
  pe6_f64      PE 6 steps with safety, float64
  pe5_f64      PE 5 steps with safety (coeffs a/1.01, b/1.01^3, c/1.01^5; X0 = G/(1.01||G||)), float64
  pe5_x0h      as pe5_f64 but X0 rounded to fp16 (computed in f32 then cast); exact arithmetic after
  pe5_f16      BEST: X0 as pe5_x0h; every step in fp16 storage with fp32 accumulate, fused
               B = fp16(b A + c A A), X = fp16(a X + B X), A = fp16(X X^T)
  pe5_f16_rAA  as pe5_f16 but A@A rounded to fp16 before combining
  pe5_bf16     as pe5_f16 with bf16 in place of fp16 (X0 cast to bf16)
Output: runs/FS_cache/final_eval.json and printed table (median/max relF, median/max max-abs)."""
import sys, json, numpy as np
sys.path.insert(0, '/home/user/AGsResearch_Claudeoppolous/dimuon/pipeline')
from fs_load import load, RANKS, CACHE
from fs_sim import ns, rank_scale, bf16
from fs_spec import PE_MEM, safety, MUON
from fs_eval import _nsraw
from fs_ns16 import x0_f16, ns16, H
PE5 = safety(PE_MEM[:5], 1.01)

def ns_bf16(G, coefs):
    G32 = np.asarray(G, np.float32); n = np.sqrt(np.sum(G32 * G32, dtype=np.float32))
    X = bf16(G32 / (n * np.float32(1.01) + np.float32(1e-7))).T.copy()
    for a, b, c in coefs:
        A = bf16(X @ X.T); B = bf16(np.float32(b) * A + np.float32(c) * (A @ A)); X = bf16(np.float32(a) * X + B @ X)
    return X.T.astype(np.float64)

def X_of(name, G):
    if name == 'polar':
        P, S, Vt = np.linalg.svd(G, full_matrices=False); return P @ Vt
    if name == 'muon5_f64': return ns(G, MUON * 5, 'f64', k_sf=1.0)
    if name == 'pe5raw_f64': return ns(G, PE_MEM[:5], 'f64', k_sf=1.0)
    if name == 'pe6_f64': return ns(G, safety(PE_MEM[:6], 1.01), 'f64')
    if name == 'pe5_f64': return ns(G, PE5, 'f64')
    if name == 'pe5_x0h': return _nsraw(x0_f16(G).astype(np.float64), PE5)
    if name == 'pe5_f16': return ns16(G, PE5)
    if name == 'pe5_f16_rAA': return ns16(G, PE5, rAA=True)
    if name == 'pe5_bf16': return ns_bf16(G, PE5)
    raise ValueError(name)

def final(X, r, lr):  # float32 renormalisation, as the data's ||U|| suggests
    X32 = X.astype(np.float32); n = np.sqrt(np.sum(X32.astype(np.float64) ** 2))
    return (-(rank_scale(r, lr) / n) * X32.astype(np.float64))

NAMES = ['polar', 'muon5_f64', 'pe5raw_f64', 'pe6_f64', 'pe5_f64', 'pe5_x0h', 'pe5_f16', 'pe5_f16_rAA', 'pe5_bf16']
SETS = [(r, 'lr1e-3', None, 1e-3) for r in RANKS] + [
    (8, 'lr1e-3_w10', 'r8_oracle_w10.npz', 1e-3), (8, 'lr1e-3_w0.1', 'r8_oracle_w0.1.npz', 1e-3),
    (8, 'lr1e-1', None, 1e-1), (8, 'lr1e-3_b0', None, 1e-3)]

if __name__ == '__main__':
    names = sys.argv[1].split(',') if len(sys.argv) > 1 else NAMES
    out = {}
    for r, dim, orc, lr in SETS:
        key = f'r{r}_{dim}'; mods = load(r, dim, orc)
        for nm in names:
            rel, mx = [], []
            for i, (n, G, U) in enumerate(mods):
                Uh = final(X_of(nm, G), r, lr)
                rel.append(np.linalg.norm(Uh - U) / np.linalg.norm(U)); mx.append(np.abs(Uh - U).max() / np.abs(U).max())
            rel = np.array(rel); mx = np.array(mx)
            out.setdefault(key, {})[nm] = dict(relF_med=float(np.median(rel)), relF_max=float(rel.max()),
                relF_med_even=float(np.median(rel[0::2])), relF_med_odd=float(np.median(rel[1::2])),
                maxabs_rel_med=float(np.median(mx)), maxabs_rel_max=float(mx.max()), relF=rel.tolist())
            print('%-16s %-12s relF med %.2e max %.2e (even %.2e odd %.2e) | max|dU|/max|U| med %.2e max %.2e' % (
                key, nm, np.median(rel), rel.max(), np.median(rel[0::2]), np.median(rel[1::2]), np.median(mx), mx.max()), flush=True)
    json.dump(out, open(CACHE / 'final_eval.json', 'w'))
