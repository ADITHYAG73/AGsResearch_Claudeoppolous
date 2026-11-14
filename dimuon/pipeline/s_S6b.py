"""S6b (regime R2): the rank-32 lr=1e-2 step was not proportional to lr=1e-3 (A 4.2e-2, B 1.8e-2
rel. Frobenius) while rank 8 in R1 was (1.9e-5). Separate rank / regime / repeatability:
 (a) rank 8 in R2: new trunk S6b_r8 (T0,T0 b0.9, save ck2), step 3 (T1) at lr 1e-4, 1e-2
     restored with optimizer; the trunk's own step 3 at lr 1e-3 is the baseline.
 (b) rank 32 in R2 at step 1: new client, lr 1e-2 on T0 vs S2_r32_b0.9/step1 (lr 1e-3).
 (c) rank 32 repeat of S6_r32_ck2_lr0.01 (re-run trunk S2_r32_b0.9 to ck2, bit-identity checked)
     plus lr 3e-3."""
import s_d2 as d
import s_fam as f
d.SETTING = "O1r2"
seq = d.SEQ
jobs = [
    lambda: d.chain("S6b_r8", 8, d.seq_steps(0.9, texts=seq[:3]), exp="S6", save=True, skip_save=[0, 1]),
    lambda: d.chain("S6b_r32_step1_lr0.01", 32, f.one(0, d.D(lr=1e-2), "step1"), exp="S6"),
    lambda: d.ensure_ckpts("S2_r32_b0.9", 32, 0.9, seq, [2], "S2"),
]
d.run(jobs)
jobs = [lambda lr=lr: d.chain(f"S6b_r8_ck2_lr{lr:g}", 8, f.one(seq[2], d.D(lr=lr), "step3"), exp="S6",
                               kind="opt", ckpt="S6b_r8-ck2") for lr in (1e-4, 1e-2)]
jobs += [lambda lr=lr, n=n: d.chain(f"S6b_r32_ck2_lr{lr:g}{n}", 32, f.one(seq[2], d.D(lr=lr), "step3"),
                                    exp="S6", kind="opt", ckpt="S2_r32_b0.9-ck2")
         for lr, n in ((1e-2, "_rep"), (3e-3, ""))]
jobs += [lambda: d.oracle("S6b_r8-ck2", seq[2])]
d.run(jobs)
d.drop_trunk("S6b_r8"); d.drop_trunk("S2_r32_b0.9")
print("COUNTS", d.tk.summary(), flush=True)
