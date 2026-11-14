"""BEHAVE-01 block A (H1): lr sweeps at rank 8, 60 steps."""
import bh_lib as b
specs = []
for lr in [3e-5, 1e-4, 3e-4, 1e-3, 3e-3]:
    specs.append(dict(name=f"A_adamw_lr{lr:g}_r8", opt="adamw", lr=lr, rank=8))
for lr in [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]:
    specs.append(dict(name=f"A_dimuon_lr{lr:g}_r8", opt="dimuon", lr=lr, rank=8))
# interleave so concurrent workers mix optimizers
specs = [s for pair in zip(specs[:5], specs[5:]) for s in pair]
b.run_many(specs)
print(b.tk.summary())
