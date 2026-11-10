"""V1 (init determinism) + V2 (step repeatability). rank 8, attention-only."""
import sys, json, pathlib
from jobs import run_all
OUT = pathlib.Path(__file__).resolve().parents[1] / "runs" / "V12"
D = dict(lr=1e-3)                      # Dimuon, beta1 0.9, no clip
A = dict(lr=1.0, eps=1.0)              # AdamW oracle setting, wd 0, no clip
specs = [dict(name=f"dim_s0_r{i}", optimizer="dimuon", rank=8, seed=0, params=D) for i in range(3)]
specs += [dict(name="dim_s1", optimizer="dimuon", rank=8, seed=1, params=None)]
specs += [dict(name=f"adam_s0_r{i}", optimizer="adamw", rank=8, seed=0, params=A) for i in range(3)]
only = sys.argv[1:]  # optional subset of names
if only: specs = [s for s in specs if s["name"] in only]
print(json.dumps(run_all(specs, OUT), default=str, indent=1)[:3000])
