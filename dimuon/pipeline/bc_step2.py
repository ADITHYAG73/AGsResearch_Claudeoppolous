"""bc_step2.py -- BIASCORR-01, step 2 (offline, existing D2 trunks; no Tinker calls).
Does the STORED momentum carry a bias correction? Step 2 of a b0.9 / b0.99 trunk (lora_B; lora_A has
S = 0 after step 1 so it cannot tell). Hypotheses for what is stored after step 1:
  ema      S = bf16((1-b) g1)                 (MULTISTEP-01 rule)
  bcstore  S = bf16(g1) = bf16((1-b) g1 / (1-b))   (bias-corrected value stored)
fed m2 = b*S + (1-b)*g2 (any bias correction of the FED value is a scalar and cancels).
Preconditioner from S (l + 1e-4 max, -1/4 both sides), PE5 with X0 fp16 (ms_common.pe).
Score = median / p90 relF of unit-normalised update vs measured lora_B delta over 152 modules.
Output runs/BC/step2.txt"""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ms_common as C
from fs_sim import bf16

def upd(S, m):
    l = np.sum(S * S, 1); l = l + 1e-4 * l.max()
    c = np.sum(S * S, 0); c = c + 1e-4 * c.max()
    Dl, Dr = l ** -0.25, c ** -0.25
    V = Dl[:, None] * C.pe(Dl[:, None] * m * Dr[None, :], x0prec="f16") * Dr[None, :]
    return -V / np.linalg.norm(V)

lines = []
for T in ("S2_r8_b0.9", "S2_r8_b0.99", "S2_r32_b0.9"):
    s1, s2 = C.step(f"{T}/step1"), C.step(f"{T}/step2")
    b = s2["params"]["beta1"]
    if not (s1.get("grad_key") and s2.get("grad_key")):
        lines.append(f"{T}: missing same-regime gradient"); continue
    g1z, g2z = C.grad(s1["grad_key"]), C.grad(s2["grad_key"])
    W1, W2 = C.state(s2["before"]), C.state(s2["after"])
    res = {"ema": [], "bcstore": []}
    for m in C.names(W1):
        g1 = np.asarray(C.get(g1z, "B", m, "G"), np.float32); g2 = np.asarray(C.get(g2z, "B", m, "G"), np.float32)
        dB = C.get(W2, "B", m) - C.get(W1, "B", m)
        for h in res:
            S = bf16(np.float32(1 - b) * g1) if h == "ema" else bf16(g1)
            S = S.astype(np.float64)
            m2 = b * S + (1 - b) * g2.astype(np.float64)
            res[h].append(np.linalg.norm(upd(S, m2) - dB / np.linalg.norm(dB)))
    s = f"{T} (beta {b}, regime {s2.get('regime')}): " + "; ".join(
        f"{h}: median relF {np.median(v):.2e}, p90 {np.percentile(v, 90):.2e}, max {np.max(v):.2e}" for h, v in res.items())
    print(s, flush=True); lines.append(s)
(C.ROOT / "runs" / "BC" / "step2.txt").write_text("\n".join(lines) + "\n")
