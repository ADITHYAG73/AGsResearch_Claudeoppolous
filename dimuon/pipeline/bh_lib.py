"""bh_lib.py -- BEHAVE-01: behavioural comparison of Dimuon vs AdamW (losses only).

Dataset: a made-up language "Velish". 8 nouns x 8 adjectives = 64 two-word phrases; the answer is
noun-word + "-" + adjective-word (order swapped vs English). 48 train / 16 held-out combinations,
split so every noun and adjective appears in train. Held-out loss therefore measures
compositional generalisation, not memorisation of public facts. Loss only on answer tokens.

Nothing is saved on the server: no save_weights / save_state / export calls.
"""
from __future__ import annotations

import json
import pathlib
import random
import threading
import time

import tk
from tinker import types

OUT = tk.ROOT / "runs" / "BH"
OUT.mkdir(parents=True, exist_ok=True)

NOUNS = {"fox": "tarvo", "dog": "melik", "bird": "siru", "fish": "pomel", "horse": "garun",
         "cat": "lisko", "bear": "dovan", "frog": "keppa"}
ADJS = {"red": "zul", "blue": "ama", "big": "oro", "small": "ith", "old": "veska",
        "fast": "nimo", "green": "talu", "loud": "brek"}

_TOK = None
_LOCK = threading.Lock()


def tokenizer():
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(tk.MODEL)
    return _TOK


def split(seed: int = 1234):
    pairs = [(a, n) for n in NOUNS for a in ADJS]
    rng = random.Random(seed)
    while True:
        rng.shuffle(pairs)
        tr, ho = pairs[:48], pairs[48:]
        if {n for _, n in tr} == set(NOUNS) and {a for a, _ in tr} == set(ADJS):
            return tr, ho


def datum(adj: str, noun: str):
    t = tokenizer()
    p = t.encode(f"Velish for '{adj} {noun}':", add_special_tokens=False)
    a = t.encode(f" {NOUNS[noun]}-{ADJS[adj]}.", add_special_tokens=False)
    ids = p + a
    w = [0.0] * (len(p) - 1) + [1.0] * len(a)
    return types.Datum(model_input=types.ModelInput.from_ints(ids[:-1]),
                       loss_fn_inputs={"target_tokens": ids[1:], "weights": w}), len(a), len(ids)


def build():
    tr, ho = split()
    train = [datum(a, n) for a, n in tr]
    held = [datum(a, n) for a, n in ho]
    mbs = [train[i:i + 8] for i in range(0, 48, 8)]  # fixed order, 6 minibatches
    return mbs, held, tr, ho


def run(name: str, opt: str, lr: float, rank: int, steps: int = 60, schedule: str = "const",
        eval_every: int = 10, seed: int = 0):
    path = OUT / f"{name}.json"
    if path.exists() and json.loads(path.read_text()).get("done"):
        return json.loads(path.read_text())
    mbs, held, _, _ = build()
    hd = [d for d, _, _ in held]
    hn = sum(n for _, n, _ in held)
    tc = tk.make_client(opt, rank=rank, seed=seed, train_attn=True, train_mlp=True,
                        train_unembed=False, tag=f"BH:{name}")
    rec = dict(name=name, opt=opt, lr=lr, rank=rank, steps=steps, schedule=schedule, seed=seed,
               model_id=tc._tk_meta["model_id"], train=[], held=[], secs=[], done=False,
               adam=dict(beta1=0.9, beta2=0.95, eps=1e-12, wd=0.0) if opt == "adamw" else None,
               dimuon=dict(beta1=0.9) if opt == "dimuon" else None)

    def ev(k):
        r = tc.forward(hd, "cross_entropy").result()
        rec["held"].append([k, r.metrics["loss:sum"] / hn])

    ev(0)
    for k in range(steps):
        t0 = time.time()
        mb = mbs[k % len(mbs)]
        d = [x for x, _, _ in mb]
        n = sum(c for _, c, _ in mb)
        lr_k = lr * (1 - k / steps) if schedule == "linear" else lr
        fb = tc.forward_backward(d, "cross_entropy").result()
        p = tk.dimuon(lr_k) if opt == "dimuon" else tk.adamw(lr_k)
        tc.optim_step(p).result()
        rec["train"].append(fb.metrics["loss:sum"] / n)
        rec["secs"].append(round(time.time() - t0, 2))
        if (k + 1) % eval_every == 0:
            ev(k + 1)
            path.write_text(json.dumps(rec))
    rec["done"] = True
    path.write_text(json.dumps(rec))
    with _LOCK:
        print(f"{name}: train(last6 mean)={sum(rec['train'][-6:])/6:.4f} "
              f"held={rec['held'][-1][1]:.4f} s/step={sum(rec['secs'])/steps:.1f}", flush=True)
    return rec


def run_many(specs, workers: int = 4):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(run, **s) for s in specs]
        out = []
        for f in futs:
            try:
                out.append(f.result())
            except Exception as e:  # report, keep going
                print("FAILED", repr(e)[:300], flush=True)
        return out
