"""fs_figs.py -- FIRSTSTEP-01 figures: transfer function and residual-by-rank."""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/home/user/AGsResearch_Claudeoppolous/dimuon/pipeline')
from fs_spec import load_curve, compose, PE_MEM, safety, MUON
FIG = '/home/user/AGsResearch_Claudeoppolous/dimuon/figures/'
PE5 = safety(PE_MEM[:5], 1.01)
A, mid = load_curve(); x = A[:, 1]
f = lambda xx: compose(xx, PE5, 1 / 1.01)
fx = f(x); nrm = np.sqrt(np.bincount(mid, fx ** 2))
y = A[:, 3] / (A[:, 5] / nrm[mid])           # measured singular value, in the units of f
xx = np.logspace(-5.5, 0, 20000)
fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3]))
cols = plt.cm.viridis(np.linspace(0, 0.9, 6))
for c, r in zip(cols, [2, 4, 8, 16, 32, 64]):
    m = A[:, 0] == r
    ax[0].scatter(x[m], y[m], s=3, color=c, label=f'r={r}', zorder=3)
    ax[1].scatter(x[m], y[m] / fx[m] - 1, s=3, color=c)
ax[0].plot(xx, f(xx), 'r', lw=0.6, label='Polar-Express 5 steps, safety 1.01 (float64)', zorder=4)
ax[0].plot(xx, compose(xx, MUON * 5, 1.0), color='0.6', lw=0.6, ls='--', label='Muon quintic x5 (reference)', zorder=2)
ax[0].axhline(1, color='k', lw=0.4)
ax[0].set_xscale('log'); ax[0].set_ylabel('output singular value  f(x)\n(data rescaled per module by its own ||f||)')
ax[0].set_title('Dimuon step 1: singular values of U vs x = s_i(G)/||G||_F  (152 modules x ranks 2-64; r=1 has x=1)')
ax[0].legend(fontsize=8, loc='lower right', markerscale=3); ax[0].grid(alpha=.3)
ax[1].set_ylabel('data / model - 1'); ax[1].set_xlabel('x = s_i / ||G||_F'); ax[1].set_ylim(-0.06, 0.06); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG + 'fs_transfer_function.png', dpi=130); plt.close()

E = json.load(open('/home/user/AGsResearch_Claudeoppolous/dimuon/runs/FS_cache/final_eval.json'))
ranks = [1, 2, 4, 8, 16, 32, 64]
names = ['polar', 'muon5_f64', 'pe5raw_f64', 'pe6_f64', 'pe5_f64', 'pe5_x0h', 'pe5_f16', 'pe5_bf16']
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for nm in names:
    med = [E[f'r{r}_lr1e-3'][nm]['relF_med'] for r in ranks]
    mxx = [E[f'r{r}_lr1e-3'][nm]['maxabs_rel_med'] for r in ranks]
    lw = 2.5 if nm == 'pe5_f16' else 1
    ax[0].plot(ranks, med, marker='o', lw=lw, label=nm); ax[1].plot(ranks, mxx, marker='o', lw=lw, label=nm)
nf = json.load(open('/home/user/AGsResearch_Claudeoppolous/dimuon/runs/FS_cache/noise_floor.json'))
ax[0].plot(ranks, [nf[str(r)]['sim_vs_x0h'] * np.sqrt(2) for r in ranks], 'k:', label='expected if only fp16 step rounding differs (sqrt2 x sim noise)')
for a in ax: a.set_xscale('log', base=2); a.set_yscale('log'); a.grid(alpha=.3, which='both'); a.set_xlabel('LoRA rank r')
ax[0].set_ylabel('median over 152 modules of ||U_hat-U||_F/||U||_F'); ax[1].set_ylabel('median of max|U_hat-U| / max|U|')
ax[0].legend(fontsize=7); ax[0].set_title('Residual by rank, candidate rules'); ax[1].set_title('Max-abs residual by rank')
plt.tight_layout(); plt.savefig(FIG + 'fs_residual_by_rank.png', dpi=130); plt.close()
print('ok')
