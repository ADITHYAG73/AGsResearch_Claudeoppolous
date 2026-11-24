"""D1: init weights, step-1 gradient G of every lora_B (AdamW oracle), one-step Dimuon updates.

Attention-only LoRA on Qwen/Qwen3.5-4B, seed 0, fixed batch tk.TEXTS[0].
Oracle (validated in V3): AdamW step 1, lr=1, eps=100, wd=0, no clip, B0=0:
    delta = -lr*g/(|g|+eps)  =>  g = -sign(delta)*eps*|delta|/(lr-|delta|),
then snapped to the bf16 grid (8 significant bits), which V3 showed makes G bit-identical across
(lr, eps, beta) settings.

Files (runs/D1/), keys use the module name with "base_model.model.model." and ".weight" stripped:
  r{r}_oracle.npz          A0/<m> init lora_A (float32), G/<m> snapped gradient (float32),
                           dB/<m> raw AdamW lora_B after the step (float32, B0 = 0)
  r{r}_dim_<tag>.npz       U/<m> Dimuon lora_B after one step (= update, B0 = 0) (float32)
  manifest.json            one entry per file
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tk  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "runs" / "D1"
OUT.mkdir(parents=True, exist_ok=True)
ORACLE = dict(lr=1.0, eps=100.0, beta1=0.9, beta2=0.95, weight_decay=0.0, grad_clip_norm=0.0)


def short(k):
    return k.replace("base_model.model.model.", "").replace(".weight", "")


def snap_bf16(x):
    m, e = np.frexp(x)
    return np.ldexp(np.round(m * 256) / 256, e)


def a0_hash(w):
    h = hashlib.sha256()
    for k in sorted(w):
        if ".lora_A." in k:
            h.update(k.encode()); h.update(w[k].astype(np.float32).tobytes())
    return h.hexdigest()


def job(spec):
    t0 = time.time()
    f = OUT / spec["file"]
    if f.exists():
        return json.load(open(OUT / (spec["file"] + ".json")))
    tc = tk.make_client(spec["optimizer"], rank=spec["rank"], seed=0, tag=spec["file"])
    w0 = tk.export(tc, spec["file"].replace(".npz", "") + "-w0")
    assert all(np.all(v == 0) for k, v in w0.items() if ".lora_B." in k), "B0 not zero"
    if spec["optimizer"] == "adamw":
        params = tk.adamw(**ORACLE)
    else:
        params = tk.dimuon(spec["lr"], beta1=spec["beta1"], grad_clip_norm=0.0)
    m = tk.step(tc, params, text=0, weight=spec["weight"])
    w1 = tk.export(tc, spec["file"].replace(".npz", "") + "-w1")
    dA = max(float(np.max(np.abs(w1[k] - w0[k]))) for k in w0 if ".lora_A." in k)
    arrs, stats = {}, {}
    if spec["optimizer"] == "adamw":
        resid, maxd = 0.0, 0.0
        for k in w0:
            if ".lora_A." in k:
                arrs["A0/" + short(k)] = w0[k].astype(np.float32)
            else:
                d = w1[k]
                maxd = max(maxd, float(np.max(np.abs(d))))
                g = -np.sign(d) * ORACLE["eps"] * np.abs(d) / (ORACLE["lr"] - np.abs(d))
                gs = snap_bf16(g)
                nz = g != 0
                if nz.any():
                    resid = max(resid, float(np.max(np.abs(gs[nz] - g[nz]) / np.abs(g[nz]))))
                arrs["G/" + short(k)] = gs.astype(np.float32)
                arrs["dB/" + short(k)] = d.astype(np.float32)
        stats = dict(max_abs_delta=maxd, max_rel_snap_residual=resid,
                     G_global_fro=float(np.sqrt(sum(np.sum(v.astype(np.float64) ** 2)
                                                    for kk, v in arrs.items() if kk.startswith("G/")))),
                     G_n_zero=int(sum(np.sum(v == 0) for kk, v in arrs.items() if kk.startswith("G/"))))
    else:
        fr = []
        for k in w0:
            if ".lora_B." in k:
                arrs["U/" + short(k)] = w1[k].astype(np.float32)
                fr.append(float(np.linalg.norm(w1[k])))
        stats = dict(U_fro_min=min(fr), U_fro_median=float(np.median(fr)), U_fro_max=max(fr))
    np.savez_compressed(f, **arrs)
    info = dict(file=spec["file"], optimizer=spec["optimizer"], rank=spec["rank"], seed=0,
                train_attn=True, train_mlp=False, train_unembed=False, model=tk.MODEL,
                text_index=0, loss_weight=spec["weight"],
                optim_params=params.model_dump(), client_model_id=tc._tk_meta["model_id"],
                fb_metrics=m["fb"], max_abs_dA=dA, A0_sha256=a0_hash(w0),
                n_lora_B=sum(".lora_B." in k for k in w0), stats=stats,
                secs=round(time.time() - t0, 1))
    json.dump(info, open(OUT / (spec["file"] + ".json"), "w"), indent=1, default=str)
    return info


specs = []
for r in [1, 2, 4, 8, 16, 32, 64]:
    specs.append(dict(file=f"r{r}_oracle.npz", optimizer="adamw", rank=r, weight=1.0))
    specs.append(dict(file=f"r{r}_dim_lr1e-3.npz", optimizer="dimuon", rank=r, lr=1e-3, beta1=0.9, weight=1.0))
for lr, tag in [(1e-4, "lr1e-4"), (1e-2, "lr1e-2"), (1e-1, "lr1e-1")]:
    specs.append(dict(file=f"r8_dim_{tag}.npz", optimizer="dimuon", rank=8, lr=lr, beta1=0.9, weight=1.0))
for w, tag in [(10.0, "w10"), (0.1, "w0.1")]:
    specs.append(dict(file=f"r8_dim_lr1e-3_{tag}.npz", optimizer="dimuon", rank=8, lr=1e-3, beta1=0.9, weight=w))
    specs.append(dict(file=f"r8_oracle_{tag}.npz", optimizer="adamw", rank=8, weight=w))
for b, tag in [(0.0, "b0"), (0.5, "b0.5")]:
    specs.append(dict(file=f"r8_dim_lr1e-3_{tag}.npz", optimizer="dimuon", rank=8, lr=1e-3, beta1=b, weight=1.0))

if __name__ == "__main__":
    only = sys.argv[1:]
    if only:
        specs = [s for s in specs if s["file"] in only]

    def safe(s):
        try:
            r = job(s); print("done", s["file"], r["secs"], flush=True); return r
        except Exception as e:
            traceback.print_exc(); return {"file": s["file"], "error": repr(e)}

    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(safe, specs))
    man = OUT / "manifest.json"
    old = json.load(open(man)) if man.exists() else {}
    for r in res:
        if "error" not in r:
            old[r["file"]] = r
    json.dump(dict(sorted(old.items())), open(man, "w"), indent=1, default=str)
    print("errors", [r for r in res if "error" in r])
    print("COUNTS", tk.summary())
