"""p3_lib.py -- PREDICT-01 collection library: s_d2.py redirected to runs/P3/.

Reuses s_d2 unchanged (content-addressed states/grads, checkpoint hygiene, storage waits) by
re-pointing its module globals:
  * OUT/MAN -> runs/P3 (a separate manifest; P3 never mixes with D2 records)
  * oracle settings "P3" and "P3b" = O1 (AdamW lr 1e6, eps 1) computed in THIS session; "P3b" is
    the second, independent recomputation used by P5.
  * new clients may use another base model / MLP / unembed adapters (CFG), and export sizes scale
    with CFG["size_mult"] for the storage reservation.
  * texts: for a non-Qwen model the three tk.TEXTS strings are re-tokenised with that model's
    tokenizer (TEXTS_BY_MODEL); tk.batch is patched to use them.
"""
from __future__ import annotations
import pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tk  # noqa: E402
import s_d2  # noqa: E402
from tinker import types  # noqa: E402

P3 = tk.ROOT / "runs" / "P3"
(P3 / "states").mkdir(parents=True, exist_ok=True)
(P3 / "grads").mkdir(parents=True, exist_ok=True)
(P3 / "pred").mkdir(parents=True, exist_ok=True)
s_d2.OUT = P3
s_d2.MAN = P3 / "manifest.json"
s_d2._MAN = None
s_d2.ORACLES["P3"] = dict(lr=1e6, eps=1.0)
s_d2.ORACLES["P3b"] = dict(lr=1e6, eps=1.0)
s_d2.SETTING = "P3"
s_d2.CKPT_TTL = 3600 * 6

CFG = dict(model=tk.MODEL, train_mlp=False, train_unembed=False, size_mult=1.0)

# gpt-oss-20b token ids for the same three strings (AutoTokenizer('openai/gpt-oss-20b'),
# encode(add_special_tokens=False)), frozen 2026-11-10.
TEXTS_BY_MODEL = {
    "openai/gpt-oss-20b": [
        [976, 4853, 19705, 68347, 65613, 1072, 290, 29082, 6446, 13, 20675, 922, 5506, 483, 6468,
         38449, 88122, 441, 13307, 13],
        [637, 220, 9659, 20, 11, 40833, 83400, 7269, 4242, 19921, 484, 9180, 35438, 25, 402, 290,
         8767, 101672, 3159, 11, 18020, 1200, 17869, 11, 3582, 37575, 2409, 11, 326, 290, 56509,
         1082, 328, 4842, 326, 5954, 13],
        [1314, 18795, 2406, 1883, 271, 538, 297, 464, 220, 17, 734, 309, 622, 297, 198, 271, 622,
         18795, 2406, 533, 220, 16, 8, 659, 18795, 2406, 533, 220, 17, 446],
    ],
}
_QWEN_TEXTS = [list(t) for t in tk.TEXTS]


def set_model(model=tk.MODEL, train_mlp=False, train_unembed=False, size_mult=1.0):
    CFG.update(model=model, train_mlp=train_mlp, train_unembed=train_unembed, size_mult=size_mult)
    tk.TEXTS[:] = TEXTS_BY_MODEL.get(model, _QWEN_TEXTS)


_orig_size = s_d2._size


def _size(rank, kind):
    return _orig_size(rank, kind) * CFG["size_mult"]


s_d2._size = _size


def _new_client(kind, rank, ckpt=None, optimizer="dimuon", tag=""):
    if kind == "new":
        return tk.make_client(optimizer, rank=rank, seed=0, train_attn=True,
                              train_mlp=CFG["train_mlp"], train_unembed=CFG["train_unembed"],
                              model=CFG["model"], tag=tag)
    path = s_d2.man()["ckpts"][ckpt]["path"]
    if kind == "fresh":
        return tk.client_from_state(path, optimizer=optimizer, tag=tag)
    if kind == "opt":
        tc = tk.service().create_training_client_from_state_with_optimizer(path).result()
        tk.COUNTS["clients"] += 1
        tc._tk_meta = dict(optimizer="restored", from_state=path, tag=tag)
        tk._log("client_from_state_with_optimizer", **tc._tk_meta)
        return tc
    raise ValueError(kind)


s_d2._new_client = _new_client

man, save_man, chain, oracle, run, D = s_d2.man, s_d2.save_man, s_d2.chain, s_d2.oracle, s_d2.run, s_d2.D
drop_ckpts, load_state, load_grad, gkey = s_d2.drop_ckpts, s_d2.load_state, s_d2.load_grad, s_d2.gkey


def steps_of(texts, beta1=0.9, lr=1e-3, clip=0.0, weights=None, prefix="step"):
    weights = weights or [1.0] * len(texts)
    return [dict(text=t, weight=w, params=D(lr, beta1, clip), sid=f"{prefix}{i + 1}")
            for i, (t, w) in enumerate(zip(texts, weights))]


def event(**kw):
    with s_d2._LOCK:
        man()["events"].append(dict(t=time.time(), **kw))
    save_man()


def ck(name):
    return man()["ckpts"][name]


for _i in range(1, 10):
    s_d2.ORACLES[f"P3x{_i}"] = dict(lr=1e6, eps=1.0)

# Compute pools, identified by the server metric fb["clock_cycle:unique"] (PREDICT-01 finding: the
# backward pass differs between pools once B != 0; each pool is deterministic).
CENTERS = {"a": 7.20e6, "b": 7.53e6}


def pool_of(cc):
    if cc is None:
        return "?"
    best = min(CENTERS, key=lambda p: abs(CENTERS[p] - cc))
    return best if abs(CENTERS[best] - cc) < 0.1e6 else f"u{cc / 1e6:.2f}"


def grads_at(sid, text):
    """{pool: [gkey...]} for every oracle gradient recorded at (sid, text)."""
    out = {}
    for k, g in man()["grads"].items():
        if g["sid"] == sid and g["text"] == text:
            out.setdefault(pool_of(g["fb"].get("clock_cycle:unique")), []).append(k)
    return out


def oracle_pooled(ckpt, text, want, tries=3):
    """repeat the oracle (one new client per attempt) until a gradient computed on pool `want`
    exists at (ckpt's state, text). Every attempt's gradient is kept (it is a valid gradient of
    whichever pool it ran on). Returns the key or None."""
    sid = ck(ckpt)["sid"]
    for _ in range(tries):
        have = grads_at(sid, text)
        if want in have:
            return have[want][0]
        used = {g["setting"] for g in man()["grads"].values() if g["sid"] == sid and g["text"] == text}
        setting = next(f"P3x{i}" for i in range(1, 10) if f"P3x{i}" not in used)
        k = oracle(ckpt, text, setting=setting)
        cc = man()["grads"][k]["fb"].get("clock_cycle:unique")
        event(kind="oracle_pooled", ckpt=ckpt, text=text, want=want, got=pool_of(cc), key=k)
        print(f"oracle_pooled {ckpt} t{text} want {want} got {pool_of(cc)}", flush=True)
    have = grads_at(sid, text)
    return have.get(want, [None])[0]


def our_ckpt_paths():
    return sorted({c["path"] for c in man()["ckpts"].values() if not c.get("deleted")})
