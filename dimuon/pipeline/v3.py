"""V3: AdamW step-1 gradient oracle. rank 8, seed 0, attention-only, wd 0, no clipping."""
import sys, json, pathlib
from jobs import run_all
OUT = pathlib.Path(__file__).resolve().parents[1] / "runs" / "V3"
S = lambda name, w=1.0, **p: dict(name=name, optimizer="adamw", rank=8, seed=0, params=p, weight=w, keep_A=False)
specs = [S("lr1_eps10", lr=1.0, eps=10.0),
         S("lr1_eps0.1", lr=1.0, eps=0.1),
         S("lr0.5_eps1", lr=0.5, eps=1.0),
         S("lr1_eps100", lr=1.0, eps=100.0),
         S("lr1e-3_eps1e-12", lr=1e-3, eps=1e-12),
         S("lr1_eps1_w2", w=2.0, lr=1.0, eps=1.0),
         S("lr1_eps1_b0.5_0.99", lr=1.0, eps=1.0, beta1=0.5, beta2=0.99)]
only = sys.argv[1:]
if only: specs = [s for s in specs if s["name"] in only]
print(json.dumps(run_all(specs, OUT), default=str)[:2000])
