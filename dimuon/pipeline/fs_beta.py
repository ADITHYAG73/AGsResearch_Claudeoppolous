"""fs_beta.py -- test explanations of the beta1 observation (FIRSTSTEP-01, item 5).
Data: U(b0) == U(b0.5) bit-identical; U(b0.9) differs by 1.5e-6 relative.
Hypothesis family: the NS input is s(beta)*G formed in float32, s depends on the momentum form;
the map is scale-invariant except for float rounding of X0 = fp16(m/(||m||*1.01+eps)).
For each form we predict dU = U(s(0.9)) - U(s(0)) with the exact-arithmetic-after-X0 model and
compare its direction/magnitude with the measured dU."""
import sys, numpy as np
sys.path.insert(0, '/home/user/AGsResearch_Claudeoppolous/dimuon/pipeline')
from fs_load import load, D1
from fs_sim import rank_scale, PE5
from fs_eval import _nsraw
from fs_ns16 import x0_f16, ns16
f32 = np.float32

def U_from(m32, r, full16=False):
    if full16:
        X = ns16(None, PE5, X0=x0_f16(m32)) if False else ns16(m32, PE5)
    else:
        X = _nsraw(x0_f16(m32).astype(np.float64), PE5)
    return -rank_scale(r, 1e-3) * X / np.linalg.norm(X)

FORMS = {  # name: function beta -> float32 tensor m given G32
    'ema (1-b)G':            lambda G, b: (f32(1) - f32(b)) * G,
    'ema (1-b)G in f64->f32':lambda G, b: (G.astype(np.float64) * (1 - b)).astype(f32),
    'nesterov ema b*m+(1-b)G with m=(1-b)G': lambda G, b: f32(b) * ((f32(1) - f32(b)) * G) + (f32(1) - f32(b)) * G,
    'nesterov plain G+b*G':  lambda G, b: G + f32(b) * G,
    'lerp m.lerp_(G,1-b)':   lambda G, b: f32(0) + (f32(1) - f32(b)) * (G - f32(0)),
    'plain m=G (no beta)':   lambda G, b: G.copy(),
    'ema + bias correction /(1-b^1), f32': lambda G, b: ((f32(1) - f32(b)) * G) / f32(1 - b ** 1),
    'ema + bias corr, b computed f32': lambda G, b: ((f32(1) - f32(b)) * G) / (f32(1) - f32(b)),
    'lerp + bias corr': lambda G, b: (f32(0) + (f32(1) - f32(b)) * (G - f32(0))) / f32(1 - b),
    'ema + bias corr as multiply by 1/(1-b)': lambda G, b: ((f32(1) - f32(b)) * G) * f32(1 / (1 - b)),
}

if __name__ == '__main__':
    z0 = np.load(D1 / 'r8_dim_lr1e-3_b0.npz'); z9 = np.load(D1 / 'r8_dim_lr1e-3.npz'); z5 = np.load(D1 / 'r8_dim_lr1e-3_b0.5.npz')
    mods = load(8)
    dD = np.concatenate([(z9['U/' + n] - z0['U/' + n]).ravel().astype(np.float64) for n, _, _ in mods])
    print('measured: ||U(.9)-U(0)||/||U(0)|| = %.3e ; U(.5)==U(0): %s ; tensors differing: %d/152' % (
        np.linalg.norm(dD) / np.linalg.norm(np.concatenate([z0['U/' + n].ravel() for n, _, _ in mods])),
        all(np.array_equal(z0['U/' + n], z5['U/' + n]) for n, _, _ in mods),
        sum(not np.array_equal(z0['U/' + n], z9['U/' + n]) for n, _, _ in mods)))
    for name, fm in FORMS.items():
        dM = []; same05 = 0; n_x0_flip9 = 0
        for n, G, U in mods:
            G32 = G.astype(f32)
            m0, m5, m9 = fm(G32, 0.0), fm(G32, 0.5), fm(G32, 0.9)
            x0a, x5, x9 = x0_f16(m0), x0_f16(m5), x0_f16(m9)
            same05 += np.array_equal(x0a, x5); n_x0_flip9 += int(np.sum(x0a != x9))
            dM.append((U_from(m9, 8) - U_from(m0, 8)).ravel())
        dM = np.concatenate(dM)
        cos = dM @ dD / (np.linalg.norm(dM) * np.linalg.norm(dD) + 1e-300)
        print('%-45s X0(b.5)==X0(b0) in %3d/152 ; X0 elems flipped at b.9: %6d ; |dU_model|/|dU_meas| %.3f ; cos %.3f' % (
            name, same05, n_x0_flip9, np.linalg.norm(dM) / np.linalg.norm(dD), cos))
