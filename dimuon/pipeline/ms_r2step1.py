"""ms_r2step1.py -- MULTISTEP-01 Q3: does the FIRSTSTEP-01 step-1 law hold in server regime R2?
Scores step-1 lora_B updates (B0 = 0) against PE5(G) [float64, X0 rounded to fp16] and the full fp16
fused emulation, for R1 and R2 step-1 records, and checks the norm law lr*sqrt(r)*(r/32)^0.175."""
import sys, json
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from ms_common import step, state, grad, names, get, pe, relF, summ, fmt, OUT
from fs_sim import update, PE5 as PE5sim

recs = ["S2_r8_b0.9/step1", "S1_r8/step1", "S5_step1_c2/step1",            # R1, r8
        "S2_r8_b0/step1", "S2_r8_b0.5/step1", "S2_r8_b0.99/step1", "S6b_r8/step1",  # R2, r8
        "S4_r32_cont/step1", "S1_r32/step1",                               # R1, r32
        "S2_r32_b0/step1", "S2_r32_b0.9/step1", "S1v2_r32/step1"]          # R2, r32
out = {}
for sid in recs:
    s = step(sid)
    if not s.get("grad_key"):
        continue
    W0, W1, G = state(s["before"]), state(s["after"]), grad(s["grad_key"])
    lr = s["params"]["learning_rate"]
    a, b, nrm, maxA = [], [], [], 0.0
    for m in names(W0):
        g = get(G, "B", m, "G")
        U = get(W1, "B", m) - get(W0, "B", m)
        maxA = max(maxA, float(np.abs(get(W1, "A", m) - get(W0, "A", m)).max()))
        r = g.shape[1]
        a.append(relF(-pe(g, x0prec="f16"), U))
        b.append(relF(update(g, lr, PE5sim, p='f16', cast_first=False), U))
        nrm.append(np.linalg.norm(U) / (lr * np.sqrt(r) * (r / 32) ** 0.175) - 1)
    out[sid] = dict(regime=s["regime"], rank=r, beta1=s["params"]["beta1"], pe_x0f16=summ(a), fp16_fused=summ(b),
                    norm_dev_maxabs=float(np.max(np.abs(nrm))), max_abs_dA=maxA)
    print(f"{sid:22s} {s['regime']} r{r} b1={s['params']['beta1']:<4}  PE5(x0 f16) {fmt(summ(a))} | fp16 fused {fmt(summ(b))} | "
          f"norm dev {np.max(np.abs(nrm)):.1e} | max|dA| {maxA:.1e}", flush=True)
json.dump(out, open(OUT / "r2_step1.json", "w"), indent=1)
