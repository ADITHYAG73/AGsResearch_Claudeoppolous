"""bc_collect.py -- BIASCORR-01 collection: step-1 Dimuon updates at several beta1, plus oracle G.

Qwen/Qwen3.5-4B, attention-only, seed 0, tk.TEXTS[0], weight 1, lr 1e-3, no clip, B0 = 0.
Strictly sequential (one client at a time; PREDICT-01 Finding 0). Every export is downloaded and its
server checkpoint deleted immediately. Only lora_B (= the update, since B0 = 0) is stored, float32,
plus a sha256 of lora_A (must equal across all clients of a rank: same init).

Files runs/BC/:  r{r}_oracle{k}.npz  (G/<m>, bf16-snapped AdamW-oracle gradient; AdamW lr 1, eps 100)
                 r{r}_b{beta}[_rep].npz (U/<m>, lora_B after one Dimuon step)
                 <file>.json (client model_id, params, server metrics, A sha, times)
Usage: TINKER_API_KEY=tml-proxy-injected python bc_collect.py [job-name ...]
"""
from __future__ import annotations
import hashlib, json, pathlib, sys, time, traceback
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tk  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "runs" / "BC"
OUT.mkdir(parents=True, exist_ok=True)
ORACLE = dict(lr=1.0, eps=100.0, beta1=0.9, beta2=0.95, weight_decay=0.0, grad_clip_norm=0.0)
BETAS = [0.0, 0.5, 0.75, 0.9, 0.99, 0.999, 0.3, 0.7]


def short(k):
    return k.replace("base_model.model.model.", "").replace(".weight", "")


def snap_bf16(x):
    m, e = np.frexp(x)
    return np.ldexp(np.round(m * 256) / 256, e)


def a_sha(w):
    h = hashlib.sha256()
    for k in sorted(w):
        if ".lora_A." in k:
            h.update(k.encode()); h.update(w[k].astype(np.float32).tobytes())
    return h.hexdigest()


def export_and_delete(tc, name):
    w, path = tk.export(tc, name, return_path=True)
    try:
        tk.rest().delete_checkpoint_from_tinker_path(path).result()
        tk._log("delete", path=path)
        deleted = True
    except Exception as e:
        tk._log("delete_failed", path=path, error=repr(e)); print("DELETE FAILED", path, e, flush=True)
        deleted = False
    return w, path, deleted


def job(j):
    f = OUT / (j["name"] + ".npz")
    if f.exists():
        print("skip", f.name, flush=True); return
    t0 = time.time()
    opt = "adamw" if j["kind"] == "oracle" else "dimuon"
    tc = tk.make_client(opt, rank=j["rank"], seed=0, tag="BC-" + j["name"])
    params = tk.adamw(**ORACLE) if opt == "adamw" else tk.dimuon(1e-3, beta1=j["beta"], grad_clip_norm=0.0)
    m = tk.step(tc, params, text=0, weight=1.0)
    w, path, deleted = export_and_delete(tc, "BC-" + j["name"])
    arrs = {}
    for k, v in w.items():
        if ".lora_B." not in k:
            continue
        if opt == "adamw":
            d = v
            g = -np.sign(d) * ORACLE["eps"] * np.abs(d) / (ORACLE["lr"] - np.abs(d))
            arrs["G/" + short(k)] = snap_bf16(g).astype(np.float32)
        else:
            arrs["U/" + short(k)] = v.astype(np.float32)
    np.savez_compressed(f, **arrs)
    info = dict(job=j, optim_params=params.model_dump(), model_id=tc._tk_meta["model_id"], metrics=m,
                A_sha256=a_sha(w), export_path=path, export_deleted=deleted,
                t_end=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), secs=round(time.time() - t0, 1))
    json.dump(info, open(OUT / (j["name"] + ".json"), "w"), indent=1, default=str)
    print("done", j["name"], info["secs"], m["fb"].get("loss:sum"), flush=True)


def jobs():
    J = []
    for r in (1, 2):
        J.append(dict(name=f"r{r}_oracle1", kind="oracle", rank=r))
        for b in BETAS:
            J.append(dict(name=f"r{r}_b{b:g}", kind="dimuon", rank=r, beta=b))
        J.append(dict(name=f"r{r}_oracle2", kind="oracle", rank=r))
    J.append(dict(name="r2_b0.9_rep", kind="dimuon", rank=2, beta=0.9))
    return J


if __name__ == "__main__":
    tk.service(); tk.rest()
    sel = sys.argv[1:]
    for j in jobs():
        if sel and j["name"] not in sel:
            continue
        try:
            job(j)
        except Exception:
            traceback.print_exc()
    print("COUNTS", tk.summary(), flush=True)
