"""BEHAVE-02 block E: seed-1 replication of block A's best-lr result. data_seed 1 (split seed 1235, shuffled
minibatch order), LoRA init seed 1, rank 8, 60 steps, const lr."""
import bh_lib as b
specs = []
for a, d in zip([1e-4, 3e-4, 1e-3], [3e-3, 1e-2, 3e-2]):
    specs.append(dict(name=f"E_adamw_lr{a:g}_r8_s1", opt="adamw", lr=a))
    specs.append(dict(name=f"E_dimuon_lr{d:g}_r8_s1", opt="dimuon", lr=d))
for s in specs:
    s.update(rank=8, steps=60, schedule="const", eval_every=10, seed=1, data_seed=1)
b.run_many(specs)
print(b.tk.summary())
