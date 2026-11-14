"""bc_eps.py -- BIASCORR-01: how large an additive eps in the X0 normaliser is compatible with the data,
under plain EMA (M = f32(1-beta)*g) vs bias-corrected EMA (M = (f32(1-beta)*g)/f32(1-beta)).
Counts tensors whose predicted X16 at beta changes vs the same form at beta = 0 (X0 = fp16(M/(n*1.01+eps)),
numpy pairwise f32 norm), at the betas where the data shows NO change within one gradient version:
r1 (grad r1_oracle3): 0.5, 0.75; r2 (grad r2_oracle2): 0.75, 0.999, 0.7. Plus per-tensor min ||g||.
Output runs/BC/eps_scan.txt"""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bc_lib as L, bc_analyze as A
lines = []
P = lambda s: (print(s, flush=True), lines.append(s))
cases = {"1": ("r1_oracle3", [0.5, 0.75, 0.9]), "2": ("r2_oracle2", [0.75, 0.999, 0.7, 0.9, 0.99])}
for r, (o, betas) in cases.items():
    G = A.ld(o, "G")
    ns = sorted(float(np.linalg.norm(G[k])) for k in G)
    P(f"rank {r} ({o}): per-tensor ||g|| min {ns[0]:.4g}, p10 {ns[len(ns)//10]:.4g}, median {ns[len(ns)//2]:.4g}")
    for cand in ("ema_a64", "ema_a64_bcdiv64"):
        for eps in (0.0, 1e-12, 1e-10, 1e-9, 1e-8, 1e-7):
            p0 = {k: L.predict(G[k], 0.0, cand, eps)[0] for k in G}
            cnt = []
            for b in betas:
                cnt.append(sum(int(np.any(L.predict(G[k], b, cand, eps)[0] != p0[k])) for k in G))
            P(f"  {cand:16s} eps {eps:<6g} changed tensors at beta " + ", ".join(f"{b:g}: {c}" for b, c in zip(betas, cnt)))
P("data: r1 0.5: 0, 0.75: 0, 0.9: 1 | r2 0.75: 0, 0.999: 0, 0.7: 0, 0.9: 1, 0.99: 1")
(A.D / "eps_scan.txt").write_text("\n".join(lines) + "\n")
