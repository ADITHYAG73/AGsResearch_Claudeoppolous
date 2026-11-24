"""S6b analysis: U(lr) * (1e-3/lr) vs U(1e-3), same before-state, same optimizer state."""
import numpy as np
import s_d2 as d
S = d.man()["steps"]


def U(s):
    a, b = d.load_state(S[s]["after"]), d.load_state(S[s]["before"])
    return {k: a[k].astype(np.float64) - b[k] for k in a}


def rel(u, v, p):
    x = np.concatenate([u[k].ravel() for k in sorted(u) if k[0] == p])
    y = np.concatenate([v[k].ravel() for k in sorted(v) if k[0] == p])
    return np.linalg.norm(x - y) / np.linalg.norm(y)


print("regen checks:", [(e["regen"], e["k"], e["bit_identical"]) for e in d.man()["events"]
                        if e["kind"] == "regen_check"])
for x, base, lr in [("S6b_r8_ck2_lr0.0001/step3", "S6b_r8/step3", 1e-4),
                    ("S6b_r8_ck2_lr0.01/step3", "S6b_r8/step3", 1e-2),
                    ("S6_r32_ck2_lr0.01/step3", "S2_r32_b0.9/step3", 1e-2),
                    ("S6b_r32_ck2_lr0.01_rep/step3", "S2_r32_b0.9/step3", 1e-2),
                    ("S6b_r32_ck2_lr0.003/step3", "S2_r32_b0.9/step3", 3e-3),
                    ("S6_r32_ck2_lr0.0001/step3", "S2_r32_b0.9/step3", 1e-4),
                    ("S6b_r32_step1_lr0.01/step1", "S2_r32_b0.9/step1", 1e-2),
                    ("S6_r8_ck2_lr0.01/step3", "S0_opt_ck2/step3", 1e-2)]:
    assert S[x]["before"] == S[base]["before"], (x, base)
    u = {k: v * (1e-3 / lr) for k, v in U(x).items()}
    print(f"{x:32s} [{S[x]['regime']}] vs {base:20s} [{S[base]['regime']}] lr={lr:g}: "
          f"relF A {rel(u, U(base), 'A'):.3e}  B {rel(u, U(base), 'B'):.3e}")
print("lr1e-2 repeat bit-identical:", S["S6_r32_ck2_lr0.01/step3"]["after"] == S["S6b_r32_ck2_lr0.01_rep/step3"]["after"])
