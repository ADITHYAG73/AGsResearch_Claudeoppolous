"""Generic one-step job runner on top of tk.py.

A job = fresh client -> export w0 -> (optional) one step -> export w1 -> save npz.
npz keys: "A0/<module>" (init lora_A), "B0/<module>" (init lora_B), "B1/<module>" (lora_B after
step), "A1/<module>" only if any lora_A changed. All float32, exactly as exported.
<module> is the PEFT tensor name with the "base_model.model.model." prefix and ".weight"
suffix stripped.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tk  # noqa: E402


def short(k):
    return k.replace("base_model.model.model.", "").replace(".weight", "")


def run_job(spec: dict, outdir: pathlib.Path):
    """spec: name, optimizer, rank, seed, params (dict for tk.dimuon/tk.adamw or None),
    text, weight, keep_A (bool)."""
    t0 = time.time()
    name = spec["name"]
    out = outdir / f"{name}.npz"
    if out.exists():
        return {"name": name, "skipped": "exists"}
    tc = tk.make_client(spec["optimizer"], rank=spec["rank"], seed=spec.get("seed", 0),
                        tag=name)
    w0 = tk.export(tc, f"{name}-w0")
    arrs = {}
    for k, v in w0.items():
        kind = "A0" if ".lora_A." in k else "B0"
        if kind == "A0" and not spec.get("keep_A", True):
            continue
        arrs[f"{kind}/{short(k)}"] = v.astype(np.float32)
    info = {"name": name, "model_id": tc._tk_meta["model_id"], **{k: v for k, v in spec.items()}}
    if spec.get("params") is not None:
        p = spec["params"]
        params = tk.dimuon(**p) if spec["optimizer"] == "dimuon" else tk.adamw(**p)
        m = tk.step(tc, params, text=spec.get("text", 0), weight=spec.get("weight", 1.0))
        info["metrics"] = m
        w1 = tk.export(tc, f"{name}-w1")
        dA = 0.0
        for k, v in w1.items():
            if ".lora_A." in k:
                d = float(np.max(np.abs(v - w0[k])))
                dA = max(dA, d)
                if d > 0:
                    arrs[f"A1/{short(k)}"] = v.astype(np.float32)
            else:
                arrs[f"B1/{short(k)}"] = v.astype(np.float32)
        info["max_abs_dA"] = dA
    np.savez_compressed(out, **arrs)
    info["secs"] = round(time.time() - t0, 1)
    with open(outdir / f"{name}.json", "w") as f:
        json.dump(info, f, indent=1, default=str)
    return info


def run_all(specs, outdir, workers=4):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def safe(s):
        try:
            r = run_job(s, outdir)
            print("done", s["name"], r.get("secs"), r.get("max_abs_dA"), flush=True)
            return r
        except Exception as e:  # report, do not retry
            traceback.print_exc()
            return {"name": s["name"], "error": repr(e)}

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(safe, specs))
    print("COUNTS", tk.summary(), flush=True)
    return res
