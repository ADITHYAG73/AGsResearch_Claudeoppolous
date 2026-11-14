"""tk.py -- small reusable harness for the Dimuon study on Tinker.

Usage (run with TINKER_API_KEY=tml-proxy-injected):
    import tk
    tc = tk.make_client("dimuon", rank=8, seed=0)
    w0 = tk.export(tc, "w0", outdir)                    # dict name -> float64 array
    m  = tk.step(tc, tk.dimuon(1e-3), text=0, weight=1.0)
    w1 = tk.export(tc, "w1", outdir)

Every client / forward_backward / optim_step / export is counted in tk.COUNTS and appended
to dimuon/runs/calls.jsonl, so a task can report exactly what it used.
"""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import tempfile
import time
import urllib.request
import warnings

import numpy as np

os.environ.setdefault("TINKER_API_KEY", "tml-proxy-injected")
import tinker  # noqa: E402
from tinker import types  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL = "Qwen/Qwen3.5-4B"
TTL = 3600
ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG = ROOT / "runs" / "calls.jsonl"

# Fixed token ids (Qwen/Qwen3.5-4B tokenizer, tokenizer.encode, no special tokens), frozen here
# so the batch never depends on a tokenizer download.
#   0: "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs."
#      (same text as pilot.py)
#   1: "In 1905, Albert Einstein published four papers that changed physics: on the
#       photoelectric effect, Brownian motion, special relativity, and the equivalence of mass
#       and energy."
#   2: "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n"
TEXTS = [
    [760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388, 13, 13399, 821, 3618, 440, 4097,
     19781, 42734, 492, 12914, 13],
    [623, 220, 16, 24, 15, 20, 11, 16993, 52237, 4501, 2943, 15228, 421, 5324, 20665, 25, 383,
     279, 6345, 61226, 2376, 11, 10158, 1068, 11051, 11, 3175, 1303, 41979, 11, 321, 279, 81259,
     314, 2973, 321, 4649, 13],
    [727, 15336, 1393, 1590, 198, 262, 413, 307, 361, 220, 17, 25, 198, 285, 460, 307, 198, 262,
     460, 15336, 1393, 471, 220, 16, 8, 478, 15336, 1393, 471, 220, 17, 8, 198],
]

COUNTS = {"clients": 0, "forward_backward": 0, "optim_step": 0, "exports": 0, "uploads": 0}
_SC = None
_RC = None


def _log(event: str, **kw):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    rec = {"t": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "event": event, **kw}
    with open(LOG, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def service():
    global _SC
    if _SC is None:
        _SC = tinker.ServiceClient()
    return _SC


def rest():
    global _RC
    if _RC is None:
        _RC = service().create_rest_client()
    return _RC


def make_client(optimizer: str = "dimuon", rank: int = 8, seed: int | None = 0,
                train_attn: bool = True, train_mlp: bool = False, train_unembed: bool = False,
                model: str = MODEL, tag: str = ""):
    """Create a fresh LoRA training client. optimizer is 'dimuon' or 'adamw'."""
    if optimizer == "dimuon":
        opt = types.DimuonOptimizerConfig()
    elif optimizer == "adamw":
        opt = types.AdamOptimizerConfig()
    else:
        raise ValueError(optimizer)
    t0 = time.time()
    tc = service().create_lora_training_client(
        base_model=model, rank=rank, seed=seed, train_mlp=train_mlp, train_attn=train_attn,
        train_unembed=train_unembed, optimizer=opt).result()
    COUNTS["clients"] += 1
    try:
        mid = tc._guaranteed_model_id()
    except Exception:
        mid = None
    tc._tk_meta = dict(optimizer=optimizer, rank=rank, seed=seed, train_attn=train_attn,
                       train_mlp=train_mlp, train_unembed=train_unembed, model=model, tag=tag,
                       model_id=str(mid))
    _log("client", secs=round(time.time() - t0, 1), **tc._tk_meta)
    return tc


def client_from_state(path: str, optimizer: str = "dimuon", tag: str = ""):
    """Fresh training client whose weights are loaded from a tinker path."""
    opt = types.DimuonOptimizerConfig() if optimizer == "dimuon" else types.AdamOptimizerConfig()
    tc = service().create_training_client_from_state(path, optimizer=opt).result()
    COUNTS["clients"] += 1
    tc._tk_meta = dict(optimizer=optimizer, from_state=path, tag=tag)
    _log("client_from_state", **tc._tk_meta)
    return tc


def _download(urls: dict, d: pathlib.Path):
    d.mkdir(parents=True, exist_ok=True)
    for f, u in urls.items():
        p = d / f
        p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(u, p)


def export(tc, name: str, outdir=None, return_path: bool = False):
    """save_weights_external + download. Returns {tensor name: float64 array}.

    If outdir is given the safetensors + config are kept in outdir/name, else a temp dir is
    used and deleted. With return_path=True also returns the tinker:// path.
    """
    r = tc.save_weights_external(name, ttl_seconds=TTL).result()
    urls = rest().get_external_weights_urls(r.path).result().urls
    COUNTS["exports"] += 1
    if outdir is None:
        with tempfile.TemporaryDirectory() as td:
            _download(urls, pathlib.Path(td))
            raw = load_file(str(pathlib.Path(td) / "adapter_model.safetensors"))
    else:
        d = pathlib.Path(outdir) / name
        _download(urls, d)
        raw = load_file(str(d / "adapter_model.safetensors"))
    out = {k: v.astype(np.float64) for k, v in raw.items()}
    _log("export", name=name, path=r.path, n=len(out),
         dtypes=sorted({str(v.dtype) for v in raw.values()}),
         model_id=getattr(tc, "_tk_meta", {}).get("model_id"))
    return (out, r.path) if return_path else out


def batch(text: int = 0, weight: float = 1.0, texts=None):
    """Fixed tiny batch. text: index (or list of indices) into TEXTS. weight: per-token loss
    weight applied to every target token (scales the cross-entropy gradient)."""
    idx = [text] if isinstance(text, int) else list(text)
    data = []
    for i in idx:
        ids = TEXTS[i]
        data.append(types.Datum(
            model_input=types.ModelInput.from_ints(ids[:-1]),
            loss_fn_inputs={"target_tokens": ids[1:],
                            "weights": [float(weight)] * (len(ids) - 1)}))
    return data


def dimuon(lr: float, beta1: float = 0.9, grad_clip_norm: float = 0.0):
    return types.DimuonParams(learning_rate=lr, beta1=beta1, grad_clip_norm=grad_clip_norm)


def adamw(lr: float, eps: float = 1e-12, beta1: float = 0.9, beta2: float = 0.95,
          weight_decay: float = 0.0, grad_clip_norm: float = 0.0):
    return types.AdamParams(learning_rate=lr, beta1=beta1, beta2=beta2, eps=eps,
                            weight_decay=weight_decay, grad_clip_norm=grad_clip_norm)


def step(tc, params, data=None, text=0, weight: float = 1.0):
    """forward_backward(cross_entropy) on the fixed batch, then optim_step(params)."""
    if data is None:
        data = batch(text, weight)
    fb = tc.forward_backward(data, "cross_entropy").result()
    COUNTS["forward_backward"] += 1
    os_ = tc.optim_step(params).result()
    COUNTS["optim_step"] += 1
    m = {"fb": dict(fb.metrics or {}),
         "optim": dict(getattr(os_, "metrics", None) or {})}
    _log("step", params=params.model_dump(), text=text, weight=weight, metrics=m,
         model_id=getattr(tc, "_tk_meta", {}).get("model_id"))
    return m


def upload(weights_dir, rank: int, seed=None, train_attn=True, train_mlp=False,
           train_unembed=False, model: str = MODEL) -> str:
    """Experimental: upload a PEFT dir (adapter_model.safetensors + adapter_config.json, exactly
    as export() wrote it) and return the tinker path of the resulting external checkpoint."""
    rc = rest()
    lc = types.LoraConfig(rank=rank, seed=seed, train_attn=train_attn, train_mlp=train_mlp,
                          train_unembed=train_unembed)
    st = rc.external_weights_upload_start(base_model=model, lora_config=lc).result()
    d = pathlib.Path(weights_dir)
    for fname, url in st.urls.items():
        body = (d / fname).read_bytes()
        req = urllib.request.Request(url, data=body, method="PUT")
        with urllib.request.urlopen(req) as resp:
            assert 200 <= resp.status < 300, (fname, resp.status)
    fin = rc.external_weights_upload_finish(st.upload_id).result()
    COUNTS["uploads"] += 1
    _log("upload", files=list(st.urls), path=fin.tinker_path)
    return fin.tinker_path


def lora_B(w: dict):
    return {k: v for k, v in w.items() if ".lora_B." in k}


def lora_A(w: dict):
    return {k: v for k, v in w.items() if ".lora_A." in k}


def summary():
    _log("summary", **COUNTS)
    return dict(COUNTS)
