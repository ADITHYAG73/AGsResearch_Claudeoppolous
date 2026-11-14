"""BEHAVE-01 blocks B (H3: rank 4/32 at best two lrs from A), repeat, and C (H2: 150 steps const vs linear).
Best lrs from block A (lowest mean train loss over 60 steps): AdamW 3e-4 then 1e-3; Dimuon 1e-2 then 3e-2."""
import bh_lib as b
best = {"adamw": [3e-4, 1e-3], "dimuon": [1e-2, 3e-2]}
specs = []
for r in [4, 32]:
    for opt in best:
        for lr in best[opt]:
            specs.append(dict(name=f"B_{opt}_lr{lr:g}_r{r}", opt=opt, lr=lr, rank=r))
specs.append(dict(name="R_dimuon_lr0.01_r8", opt="dimuon", lr=1e-2, rank=8))  # repeat of A run
for opt in best:
    for lr in best[opt]:
        for sch in ["const", "linear"]:
            specs.append(dict(name=f"C_{opt}_lr{lr:g}_r8_{sch}", opt=opt, lr=lr, rank=8,
                              steps=150, schedule=sch))
b.run_many(specs)
print(b.tk.summary())
