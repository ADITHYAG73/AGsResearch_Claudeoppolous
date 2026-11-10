"""s_d2.py -- library for dataset D2 (Dimuon beyond step 1). Built on tk.py.

Storage (runs/D2/), content addressed so a state shared by several steps is stored once:
  states/<sid>.npz   A/<m>, B/<m>: every lora_A / lora_B exactly as exported (float32).
                     sid = first 16 hex of sha256 over (sorted key, float32 bytes) of all tensors.
  grads/<gkey>.npz   GA/<m>, GB/<m>: oracle gradient at a state for one batch, snapped to bf16
                     (float32 storage). gkey = "<sid>_t<text>_w<weight>_<oracle setting>".
  manifest.json      {"states": {sid: info}, "grads": {gkey: info}, "ckpts": {name: info},
                      "steps": {step_id: record}}. A step record names its before/after sid,
                      batch (text, weight), optimizer params, client kind, optimizer-state
                      lineage ("hist": the step ids whose optimizer state the step inherits),
                      and grad_key (O1 oracle at the before state for the same batch).
  <m> = PEFT tensor name with "base_model.model.model." and ".weight" stripped (as D1).

Oracle (V3, extended in S0): fresh-state AdamW client loaded from a save_state checkpoint,
one step, wd 0, no clip: delta = -lr*g/(|g|+eps) => g = -sign(delta)*eps*|delta|/(lr-|delta|),
then snapped to 8 significant bits. Default setting O1 = (lr 1e6, eps 1): with a nonzero state,
delta must be large against the existing weights for the float32 difference to resolve small g.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tk  # noqa: E402
from tinker import types  # noqa: E402

OUT = tk.ROOT / "runs" / "D2"
(OUT / "states").mkdir(parents=True, exist_ok=True)
(OUT / "grads").mkdir(parents=True, exist_ok=True)
MAN = OUT / "manifest.json"
CKPT_TTL = 86400
ORACLES = {
    "O1": dict(lr=1e6, eps=1.0),
    "O2": dict(lr=1e6, eps=10.0),
    "O3": dict(lr=1.0, eps=100.0),   # the D1 / V3 setting
}
_LOCK = threading.Lock()
_MAN = None


def short(k):
    return k.replace("base_model.model.model.", "").replace(".weight", "")


def snap_bf16(x):
    m, e = np.frexp(x)
    return np.ldexp(np.round(m * 256) / 256, e)


def man():
    global _MAN
    if _MAN is None:
        _MAN = json.load(open(MAN)) if MAN.exists() else {}
        for k in ("states", "grads", "ckpts", "steps", "events"):
            _MAN.setdefault(k, {} if k != "events" else [])
    return _MAN


def save_man():
    with _LOCK:
        tmp = MAN.with_suffix(".tmp")
        json.dump(man(), open(tmp, "w"), indent=1, default=str, sort_keys=True)
        os.replace(tmp, MAN)


def sid_of(w):
    h = hashlib.sha256()
    for k in sorted(w):
        h.update(k.encode()); h.update(w[k].astype(np.float32).tobytes())
    return h.hexdigest()[:16]


def norms(w):
    a = [float(np.linalg.norm(v)) for k, v in w.items() if ".lora_A." in k]
    b = [float(np.linalg.norm(v)) for k, v in w.items() if ".lora_B." in k]
    return dict(A_fro_median=float(np.median(a)), B_fro_median=float(np.median(b)),
                B_fro_max=max(b), A_global=float(np.sqrt(sum(x * x for x in a))),
                B_global=float(np.sqrt(sum(x * x for x in b))))


def delete_ckpt_path(path):
    """Delete one of OUR checkpoints (free storage is capped; an export is only transport)."""
    try:
        tk.rest().delete_checkpoint_from_tinker_path(path).result()
        tk._log("delete", path=path)
    except Exception as e:  # report, keep going
        tk._log("delete_failed", path=path, error=repr(e))
        print("delete failed", path, repr(e), flush=True)


# Free checkpoint storage on this account is capped (~10 GB; S0 journal entry). Owner's older
# checkpoints occupy ~6.9 GB that we must not touch, so every export / save_state first waits
# until (listed bytes + bytes reserved by in-flight operations + need) is under CAP_BYTES.
CAP_BYTES = 9.5e9
CAP_COUNT = 95
_SLOCK = threading.Lock()
_INFLIGHT = [0]


def _size(rank, kind):
    per8 = {"export": 28.75e6, "training": 43.2e6}[kind]
    return per8 * rank / 8 * 1.1


def storage_reserve(need, tag=""):
    for i in range(60):
        with _SLOCK:
            cks = tk.rest().list_user_checkpoints(limit=1000).result().checkpoints
            tot = sum(x.size_bytes or 0 for x in cks)
            if tot + _INFLIGHT[0] + need < CAP_BYTES and len(cks) < CAP_COUNT:
                _INFLIGHT[0] += need
                return
        if i % 5 == 0:
            print(f"storage wait {tag}: listed {tot / 1e9:.2f} GB in {len(cks)} ckpts, "
                  f"inflight {_INFLIGHT[0] / 1e9:.2f} GB, need {need / 1e9:.2f}", flush=True)
            tk._log("storage_wait", tag=tag, listed=tot, count=len(cks), inflight=_INFLIGHT[0])
        time.sleep(60)
    raise RuntimeError("storage: gave up waiting " + tag)


def storage_release(need):
    with _SLOCK:
        _INFLIGHT[0] -= need


def export_tmp(tc, name, rank=8):
    need = _size(rank, "export")
    storage_reserve(need, name)
    try:
        w, path = tk.export(tc, name, return_path=True)
        delete_ckpt_path(path)
    finally:
        storage_release(need)
    return w


def drop_ckpts(names):
    for n in names:
        c = man()["ckpts"].get(n)
        if c and not c.get("deleted"):
            delete_ckpt_path(c["path"])
            with _LOCK:
                c["deleted"] = True
    save_man()


def export_state(tc, name, rank):
    w = export_tmp(tc, name, rank)
    sid = sid_of(w)
    f = OUT / "states" / f"{sid}.npz"
    with _LOCK:
        new = sid not in man()["states"]
        if new:
            man()["states"][sid] = dict(rank=rank, first_name=name, n=len(w), **norms(w))
    if new or not f.exists():
        arrs = {("A/" if ".lora_A." in k else "B/") + short(k): v.astype(np.float32)
                for k, v in w.items()}
        np.savez(f, **arrs)
    save_man()
    return sid, w


def save_ckpt(tc, name, sid, hist, rank, optimizer):
    old = man()["ckpts"].get(name)
    if old and not old.get("deleted"):      # a re-run: do not leak the earlier checkpoint
        delete_ckpt_path(old["path"])
    need = _size(rank, "training")
    storage_reserve(need, name)
    try:
        r = tc.save_state(name, ttl_seconds=CKPT_TTL).result()
    finally:
        storage_release(need)
    with _LOCK:
        man()["ckpts"][name] = dict(path=r.path, sid=sid, hist=list(hist), rank=rank,
                                     optimizer=optimizer, t=time.time())
    tk._log("save_state", name=name, path=r.path, sid=sid)
    save_man()
    return r.path


def upd_stats(w0, w1):
    dA = [float(np.linalg.norm(w1[k] - w0[k])) for k in w0 if ".lora_A." in k]
    dB = [float(np.linalg.norm(w1[k] - w0[k])) for k in w0 if ".lora_B." in k]
    mA = max(float(np.max(np.abs(w1[k] - w0[k]))) for k in w0 if ".lora_A." in k)
    mB = max(float(np.max(np.abs(w1[k] - w0[k]))) for k in w0 if ".lora_B." in k)
    return dict(dA_fro_min=min(dA), dA_fro_median=float(np.median(dA)), dA_fro_max=max(dA),
                dA_global=float(np.sqrt(sum(x * x for x in dA))), dA_maxabs=mA,
                dB_fro_min=min(dB), dB_fro_median=float(np.median(dB)), dB_fro_max=max(dB),
                dB_global=float(np.sqrt(sum(x * x for x in dB))), dB_maxabs=mB,
                A_moved=bool(mA > 0), B_moved=bool(mB > 0))


def new_client(kind, rank, ckpt=None, optimizer="dimuon", tag=""):
    """Retries (90 s apart, at most 20 times) when the free checkpoint-storage cap blocks
    client creation; every refusal is logged."""
    import tinker
    for i in range(20):
        try:
            return _new_client(kind, rank, ckpt, optimizer, tag)
        except tinker.BillingError as e:
            tk._log("billing_refused", tag=tag, error=repr(e)[:200])
            print("402 storage cap, waiting", tag, flush=True)
            time.sleep(90)
    raise RuntimeError("storage cap: gave up " + tag)


def _new_client(kind, rank, ckpt=None, optimizer="dimuon", tag=""):
    """kind: 'new' (fresh client, seed 0, init weights), 'fresh' (weights from ckpt, fresh optimizer
    state of `optimizer`), 'opt' (weights + optimizer state from ckpt, same optimizer)."""
    if kind == "new":
        return tk.make_client(optimizer, rank=rank, seed=0, tag=tag)
    path = man()["ckpts"][ckpt]["path"]
    if kind == "fresh":
        return tk.client_from_state(path, optimizer=optimizer, tag=tag)
    if kind == "opt":
        tc = tk.service().create_training_client_from_state_with_optimizer(path).result()
        tk.COUNTS["clients"] += 1
        tc._tk_meta = dict(optimizer="restored", from_state=path, tag=tag)
        tk._log("client_from_state_with_optimizer", **tc._tk_meta)
        return tc
    raise ValueError(kind)


def chain(name, rank, steps, exp, kind="new", ckpt=None, save=False, save_last=False,
          skip_save=()):
    """Run a sequence of steps in ONE client. steps: list of dicts with keys text, weight,
    params (tinker params object), sid (short label). Before step k the state is exported (and
    saved with save_state if save). Returns list of step ids.

    Records for each step: before/after sid, params, batch, client kind, lineage."""
    if all(f"{name}/{st['sid']}" in man()["steps"] for st in steps):
        return [f"{name}/{st['sid']}" for st in steps]
    t0 = time.time()
    tc = new_client(kind, rank, ckpt, tag=name)
    mid = str(tc._guaranteed_model_id())
    if kind == "new":
        hist, src_sid = [], None
    else:
        c = man()["ckpts"][ckpt]
        hist = list(c["hist"]) if kind == "opt" else []
        src_sid = c["sid"]
    sid, w = export_state(tc, f"{name}-s0", rank)
    load_ok = None if src_sid is None else (sid == src_sid)
    if src_sid is not None:
        with _LOCK:
            man()["events"].append(dict(kind="reload_check", chain=name, ckpt=ckpt, how=kind,
                                        src_sid=src_sid, loaded_sid=sid, bit_identical=load_ok))
    ids = []
    for k, st in enumerate(steps):
        if save and k not in skip_save:
            save_ckpt(tc, f"{name}-ck{k}", sid, hist, rank, "dimuon")
        m = tk.step(tc, st["params"], data=tk.batch(st["text"], st["weight"]),
                    text=st["text"], weight=st["weight"])
        sid1, w1 = export_state(tc, f"{name}-s{k + 1}", rank)
        step_id = f"{name}/{st['sid']}"
        rec = dict(exp=exp, rank=rank, chain=name, index_in_chain=k, before=sid, after=sid1,
                   text=st["text"], weight=st["weight"], params=st["params"].model_dump(),
                   client=dict(kind=kind if k == 0 else "continuing", chain_start=kind,
                               from_ckpt=ckpt, model_id=mid, load_bit_identical=load_ok),
                   hist=list(hist), fb=m["fb"], optim=m["optim"],
                   grad_key=gkey(sid, st["text"], st["weight"], "O1"), stats=upd_stats(w, w1),
                   note=st.get("note", ""))
        with _LOCK:
            man()["steps"][step_id] = rec
        save_man()
        ids.append(step_id)
        hist = hist + [step_id]
        sid, w = sid1, w1
    if save_last:
        save_ckpt(tc, f"{name}-ck{len(steps)}", sid, hist, rank, "dimuon")
    print(f"chain {name} done {time.time() - t0:.0f}s", flush=True)
    return ids


def gkey(sid, text, weight, setting):
    return f"{sid}_t{text}_w{weight:g}_{setting}"


def oracle(ckpt, text, weight=1.0, setting="O1"):
    """Exact gradient of every lora_A and lora_B at ckpt's weights, for batch (text, weight)."""
    c = man()["ckpts"][ckpt]
    key = gkey(c["sid"], text, weight, setting)
    if key in man()["grads"] and (OUT / "grads" / f"{key}.npz").exists():
        return key
    t0 = time.time()
    o = ORACLES[setting]
    tc = new_client("fresh", c["rank"], ckpt, optimizer="adamw", tag=f"oracle {key}")
    sid0, w0 = export_state(tc, f"or-{key}-w0", c["rank"])
    m = tk.step(tc, tk.adamw(o["lr"], eps=o["eps"], beta1=0.9, beta2=0.95, weight_decay=0.0,
                             grad_clip_norm=0.0),
                data=tk.batch(text, weight), text=text, weight=weight)
    w1 = export_tmp(tc, f"or-{key}-w1", c["rank"])
    arrs, resid, nbad, nsat, nzero, sq = {}, 0.0, 0, 0, 0, {"A": 0.0, "B": 0.0}
    for k in w0:
        d = w1[k] - w0[k]
        ad = np.abs(d)
        sat = ad >= o["lr"]
        nsat += int(sat.sum())
        g = -np.sign(d) * o["eps"] * ad / np.where(sat, np.nan, o["lr"] - ad)
        gs = snap_bf16(g)
        nz = (g != 0) & np.isfinite(g)
        if nz.any():
            r = np.abs(gs[nz] - g[nz]) / np.abs(g[nz])
            resid = max(resid, float(r.max()))
            nbad += int((r > 1e-3).sum())
        nzero += int((g == 0).sum())
        part = "A" if ".lora_A." in k else "B"
        sq[part] += float(np.nansum(gs.astype(np.float64) ** 2))
        arrs[("GA/" if part == "A" else "GB/") + short(k)] = gs.astype(np.float32)
    np.savez(OUT / "grads" / f"{key}.npz", **arrs)
    info = dict(sid=c["sid"], ckpt=ckpt, text=text, weight=weight, setting=setting, **o,
                loaded_sid=sid0, load_bit_identical=(sid0 == c["sid"]), fb=m["fb"],
                max_rel_snap_residual=resid, n_snap_resid_gt_1em3=nbad, n_saturated=nsat,
                n_exact_zero=nzero, GA_global=float(np.sqrt(sq["A"])),
                GB_global=float(np.sqrt(sq["B"])),
                G_global=float(np.sqrt(sq["A"] + sq["B"])), secs=round(time.time() - t0, 1))
    with _LOCK:
        man()["grads"][key] = info
    save_man()
    print(f"oracle {key} done {info['secs']}s resid {resid:.2e} bad {nbad}", flush=True)
    return key


def run(jobs, workers=4):
    """jobs: list of zero-arg callables. At most `workers` concurrent clients."""
    def safe(j):
        try:
            return j()
        except Exception as e:
            traceback.print_exc()
            if "poisoned" in repr(e):        # a poisoned ServiceClient kills every later call
                tk._SC = None
                tk._RC = None
            with _LOCK:
                man()["events"].append(dict(kind="error", job=repr(j), error=repr(e)))
            save_man()
            return e
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(safe, jobs))
    save_man()
    return res


def D(lr=1e-3, beta1=0.9, clip=0.0):
    return tk.dimuon(lr, beta1=beta1, grad_clip_norm=clip)


SEQ = [0, 0, 1, 2, 0, 1]   # S2 batch sequence


def seq_steps(beta1, lr=1e-3, texts=SEQ):
    return [dict(text=t, weight=1.0, params=D(lr, beta1), sid=f"step{i + 1}")
            for i, t in enumerate(texts)]


def load_state(sid):
    z = np.load(OUT / "states" / f"{sid}.npz")
    return {k: z[k] for k in z.files}


def load_grad(key):
    z = np.load(OUT / "grads" / f"{key}.npz")
    return {k: z[k] for k in z.files}


def ensure_ckpts(T, rank, beta1, texts, ks, exp):
    """Make sure checkpoints T-ck{k} (k in ks) exist. If trunk T was never fully recorded, run it
    (saving states ks). Else, if any checkpoint is missing or deleted, re-run the trunk from a new
    client under the name T_reN saving only states ks, check each re-run state is bit-identical to
    the recorded trunk state, and alias T-ck{k} to it."""
    m = man()
    if not all(f"{T}/step{i + 1}" in m["steps"] for i in range(len(texts))):
        steps = seq_steps(beta1, texts=texts)
        chain(T, rank, steps, exp=exp, save=True,
              skip_save=[k for k in range(len(steps)) if k not in ks], save_last=len(steps) in ks)
        return
    if all(f"{T}-ck{k}" in m["ckpts"] and not m["ckpts"][f"{T}-ck{k}"].get("deleted")
           for k in ks):
        return
    n = 1 + len({c.split("-ck")[0] for c in m["ckpts"] if c.startswith(f"{T}_re")})
    R = f"{T}_re{n}"
    steps = seq_steps(beta1, texts=texts)[:max(ks)]
    chain(R, rank, steps, exp=exp + "-regen", save=True,
          skip_save=[k for k in range(len(steps)) if k not in ks], save_last=True)
    for k in ks:
        c = m["ckpts"][f"{R}-ck{k}"]
        ref = (m["steps"][f"{T}/step{k}"]["after"] if k > 0 else m["steps"][f"{T}/step1"]["before"])
        ok = c["sid"] == ref
        with _LOCK:
            m["events"].append(dict(kind="regen_check", trunk=T, regen=R, k=k, bit_identical=ok))
            if ok:
                m["ckpts"][f"{T}-ck{k}"] = dict(c, alias_of=f"{R}-ck{k}")
        save_man()
        assert ok, (R, k, c["sid"], ref)


def drop_trunk(T, n=7):
    """Delete every checkpoint of trunk T and of its re-runs (aliases share paths)."""
    names = [c for c in man()["ckpts"] if c.startswith(T + "-ck") or c.startswith(T + "_re")]
    paths = {}
    for nme in names:
        c = man()["ckpts"][nme]
        if not c.get("deleted"):
            paths.setdefault(c["path"], []).append(nme)
    for p, nms in paths.items():
        delete_ckpt_path(p)
    with _LOCK:
        for nme in names:
            man()["ckpts"][nme]["deleted"] = True
    save_man()
