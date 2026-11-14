"""BEHAVE-02 block D: const lr, rank 8, 240 steps, BEHAVE-01 data (data_seed 0, LoRA seed 0), held-out every 20.
Dimuon at 1/3 and 1/10 of its best lr (1e-2); AdamW at 1/3 and 1/10 of its best (3e-4)."""
import bh_lib as b
specs = [dict(name="D_dimuon_lr0.003_r8_const240", opt="dimuon", lr=3e-3),
         dict(name="D_adamw_lr0.0001_r8_const240", opt="adamw", lr=1e-4),
         dict(name="D_dimuon_lr0.001_r8_const240", opt="dimuon", lr=1e-3),
         dict(name="D_adamw_lr3e-05_r8_const240", opt="adamw", lr=3e-5)]
for s in specs:
    s.update(rank=8, steps=240, schedule="const", eval_every=20, seed=0, data_seed=0)
b.run_many(specs)
print(b.tk.summary())
