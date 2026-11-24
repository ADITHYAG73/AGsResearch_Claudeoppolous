"""PILOT-01: can we observe one Dimuon step? Export weights before/after a single optim_step."""
import os, sys, json, time, urllib.request, pathlib
import tinker
from tinker import types

MODEL = "Qwen/Qwen3.5-4B"
OUT = pathlib.Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
sc = tinker.ServiceClient()
tc = sc.create_lora_training_client(
    base_model=MODEL, rank=4, seed=0, train_mlp=False, train_unembed=False, train_attn=True,
    optimizer=types.DimuonOptimizerConfig(),
).result()
print("client", time.time() - t0, tc.get_info(), flush=True)
rc = sc.create_rest_client()

def export(name):
    r = tc.save_weights_external(name, ttl_seconds=3600).result()
    urls = rc.get_external_weights_urls(r.path).result().urls
    d = OUT / name; d.mkdir(exist_ok=True)
    for f, u in urls.items():
        p = d / f; p.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(u, p)
        print(name, f, p.stat().st_size, flush=True)

export("w0")
tok = tc.get_tokenizer()
ids = tok.encode("The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.")
datum = types.Datum(
    model_input=types.ModelInput.from_ints(ids[:-1]),
    loss_fn_inputs={"target_tokens": ids[1:], "weights": [1.0] * (len(ids) - 1)},
)
fb = tc.forward_backward([datum], "cross_entropy").result()
print("fb", fb.metrics, flush=True)
os_ = tc.optim_step(types.DimuonParams(learning_rate=1e-3)).result()
print("optim", os_, flush=True)
export("w1")
print("done", time.time() - t0)
