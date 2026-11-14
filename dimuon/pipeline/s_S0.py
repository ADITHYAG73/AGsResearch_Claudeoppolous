"""S0 validation jobs (+ the other trunks, run concurrently). Trunk S2_r8_b0.9 from s_smoke.py."""
import s_d2 as d
T = "S2_r8_b0.9"
jobs = [
    lambda: d.chain("S1_r8", 8, d.seq_steps(0.9, texts=[0] * 5), exp="S1", save=True),
    lambda: d.chain("S1_r32", 32, d.seq_steps(0.9, texts=[0] * 5), exp="S1", save=True),
    lambda: d.chain("S2_r32_b0.9", 32, d.seq_steps(0.9), exp="S2", save=True),
    lambda: d.oracle(f"{T}-ck2", 1, setting="O2"),
    lambda: d.oracle(f"{T}-ck2", 1, setting="O3"),
    lambda: d.oracle(f"{T}-ck0", 0, setting="O1"),
    # continue-vs-restore: restored with optimizer from ck1 -> step 2 (T0); from ck2 -> step 3 (T1)
    lambda: d.chain("S0_opt_ck1", 8, [dict(text=0, weight=1.0, params=d.D(), sid="step2")],
                    exp="S0", kind="opt", ckpt=f"{T}-ck1"),
    lambda: d.chain("S0_opt_ck2", 8, [dict(text=1, weight=1.0, params=d.D(), sid="step3")],
                    exp="S0", kind="opt", ckpt=f"{T}-ck2"),
    # fresh Dimuon state loaded from the init checkpoint -> step on T0 (compare to D1)
    lambda: d.chain("S0_fresh_ck0", 8, [dict(text=0, weight=1.0, params=d.D(), sid="step1")],
                    exp="S0", kind="fresh", ckpt=f"{T}-ck0"),
]
d.run(jobs)
print(d.tk.summary())
