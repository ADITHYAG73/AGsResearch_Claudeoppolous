"""fr_grid.py -- FRESH-01: bit-level fingerprint. FIRSTSTEP-01 item 1 found every step-1 update to
be (one float32 scalar) x (an fp16-representable matrix).  Does that hold for fresh-state steps
from B != 0, and for continuing steps?  Tested on lora_B only: B entries are small enough that the
float32 subtraction W1 - W0 is exact to < 1e-6 relative (checked per tensor and reported); for
lora_A (|A| ~ 0.03) it is not, so A cannot be tested this way at lr <= 1e-2.
Per tensor: fraction of sampled entries on the 11-bit grid after the best per-tensor scale
(chance ~ 1.5%; 100% = scaled fp16 matrix).  -> runs/FR/grid.json, grid.txt
"""
import json
import numpy as np
from fr_core import *  # noqa
from fs_grid import recover

STEPS = ["S1_r8/step1", "S1v2_r32/step1",
         "S1_r8_fresh_ck1_t0/step", "S1_r8_fresh_ck2_t1/step", "S1_r8_fresh_ck5_t0/step",
         "S1v2_r32_fresh_ck2_t0/step", "S1v2_r32_fresh_ck5_t1/step", "S3x_r8_fresh_ck3/step4",
         "S2_r8_b0_fresh_ck3/step4", "S1v2_r32_fresh_ck1_t1/step",          # the mismatched pairs
         "S1_r8/step2", "S1_r8/step3", "S1v2_r32/step2", "S2_r8_b0/step2",   # continuing (contrast)
         "S5_step2_c0.5/step2", "S6_r32_ck2_lr0.01/step3", "S6_r32_ck2_lr0.001/step3"]

if __name__ == "__main__":
    out, lines = {}, []
    for s in STEPS:
        r = rec(s); W0, W1 = state(r["before"]), state(r["after"])
        fr, ex = [], []
        for i, m in enumerate(modules(W0)):
            if i % 4: continue
            k = "B/" + mB(m)
            w0 = W0[k]; D = W1[k] - w0
            ex.append(float(np.median(np.spacing(np.abs(W1[k]).astype(np.float32)) / np.maximum(np.abs(D), 1e-30))))
            fr.append(recover(D)[3])
        out[s] = dict(grid=summ(fr), f32_err=summ(ex), hist=len(r["hist"]))
        l = (f"{s:32s} hist {len(r['hist'])}  on-fp16-grid frac [{fmt(summ(fr))}]   "
             f"f32 subtraction err/|D| median {np.median(ex):.1e}")
        print(l, flush=True); lines.append(l)
    json.dump(out, open(OUT / "grid.json", "w"), indent=1)
    open(OUT / "grid.txt", "w").write("\n".join(lines) + "\n")
