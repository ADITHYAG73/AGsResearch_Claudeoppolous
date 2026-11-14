"""fs_fitcoef.py -- free fit of a per-step odd-quintic schedule to the spectral transfer data.
Model: d_i = ||U||_F * f(x_i)/||f(x)||_module, f = composition of N quintics, x = k*s_i/||G||_F."""
import sys, json, numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, '/home/user/AGsResearch_Claudeoppolous/dimuon/pipeline')
from fs_spec import *

def fit(N, init, k0, train_mask, A, mid, fix_k=False):
    x = A[:, 1]; d = A[:, 3]; Un = A[:, 5]
    def unpack(th):
        co = [tuple(th[3*i:3*i+3]) for i in range(N)]; k = k0 if fix_k else th[-1]; return co, k
    def res(th):
        co, k = unpack(th); fx = compose(x, co, k)
        nrm = np.sqrt(np.bincount(mid, fx**2)); p = fx / nrm[mid] * Un
        return ((p - d) / Un)[train_mask]
    th0 = np.r_[np.ravel(init)] if fix_k else np.r_[np.ravel(init), k0]
    r = least_squares(res, th0, method='trf', x_scale='jac', max_nfev=3000)
    co, k = unpack(r.x)
    return co, k, r

if __name__ == '__main__':
    A, mid = load_curve()
    rng = np.random.default_rng(0)
    nmod = mid.max() + 1
    train_mod = rng.random(nmod) < 0.5
    tr = train_mod[mid] & (A[:, 0] > 1); te = (~train_mod[mid]) & (A[:, 0] > 1)
    init = safety(PE_MEM[:5], 1.01)
    fx = compose(A[:, 1], init, 1/1.01)
    base = score(fx, A, mid)
    print('PE5 nominal per-rank rms', {k: '%.2e' % v for k, v in base.items()})
    # restricted to well-resolved region to avoid the noise-dominated x<1e-3 tail? use all
    co, k, r = fit(5, init, 1/1.01, tr, A, mid, fix_k=True)
    fx = compose(A[:, 1], co, k)
    def rms(mask):
        nrm = np.sqrt(np.bincount(mid, fx**2)); p = fx / nrm[mid] * A[:, 5]; e = (p - A[:, 3]) / A[:, 5]; return np.sqrt(np.mean(e[mask]**2))
    fx0 = compose(A[:, 1], init, 1/1.01)
    def rms0(mask):
        nrm = np.sqrt(np.bincount(mid, fx0**2)); p = fx0 / nrm[mid] * A[:, 5]; e = (p - A[:, 3]) / A[:, 5]; return np.sqrt(np.mean(e[mask]**2))
    print('fitted (k fixed 1/1.01): train %.3e test %.3e | PE5 train %.3e test %.3e' % (rms(tr), rms(te), rms0(tr), rms0(te)))
    for (a, b, c), (a0, b0, c0) in zip(co, init):
        print('  fit (%.6f, %.6f, %.6f)   PE5/1.01 (%.6f, %.6f, %.6f)  rel diff a %.1e' % (a, b, c, a0, b0, c0, a/a0-1))
    json.dump({'coefs': co, 'k': k}, open('/home/user/AGsResearch_Claudeoppolous/dimuon/runs/FS_cache/fitcoef.json', 'w'))
