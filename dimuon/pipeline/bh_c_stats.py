"""BEHAVE-01 block C late-phase statistics (offline)."""
import json, glob, pathlib
import numpy as np
R = pathlib.Path(__file__).resolve().parents[1] / "runs" / "BH"
out = {}
for f in sorted(R.glob("C_*.json")):
    r = json.loads(f.read_text())
    t = np.array(r["train"])
    if len(t) < 150:
        continue
    late = t[90:]  # steps 91-150 = last 10 epochs
    s = dict(n=len(t), median_late=float(np.median(late)), mean_late=float(late.mean()),
             max_late=float(late.max()), frac_late_gt_0p01=float((late > 0.01).mean()),
             mean_last6=float(t[-6:].mean()), held=r["held"][-1][1],
             held_late=[round(h, 5) for k, h in r["held"] if k >= 90])
    out[f.stem] = s
    print(f"{f.stem:30s} med91-150={s['median_late']:.2e} mean={s['mean_late']:.3g} max={s['max_late']:.3g} "
          f"frac>0.01={s['frac_late_gt_0p01']:.2f} last6={s['mean_last6']:.2e} held={s['held_late']}")
(R / "blockC_stats.json").write_text(json.dumps(out, indent=1))
