"""V4: upload a modified adapter (pilot01 w0, rank 4 seed 0, lora_B := N(0,1e-3^2)), load it into a
fresh training client, export again, compare."""
import pathlib, shutil, json, sys
import numpy as np
from safetensors.numpy import load_file, save_file
import tk
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "runs" / "pilot01" / "w0"; OUT = ROOT / "runs" / "V4"; UP = OUT / "up"
UP.mkdir(parents=True, exist_ok=True)
w = load_file(str(SRC / "adapter_model.safetensors"))
rng = np.random.default_rng(123)
for k in sorted(w):
    if ".lora_B." in k:
        w[k] = (rng.standard_normal(w[k].shape) * 1e-3).astype(np.float32)
save_file(w, str(UP / "adapter_model.safetensors"))
shutil.copy(SRC / "adapter_config.json", UP / "adapter_config.json")
path = tk.upload(UP, rank=4, seed=0)
print("uploaded", path, flush=True)
tc = tk.client_from_state(path, optimizer="dimuon", tag="V4")
back = tk.export(tc, "v4-back", OUT)
res = {"path": path, "n_tensors_up": len(w), "n_tensors_back": len(back),
       "same_keys": sorted(w) == sorted(back)}
if res["same_keys"]:
    res["bit_identical"] = all(np.array_equal(w[k].astype(np.float64), back[k]) for k in w)
    res["max_abs_diff"] = max(float(np.max(np.abs(w[k].astype(np.float64) - back[k]))) for k in w)
    res["n_tensors_differing"] = sum(not np.array_equal(w[k].astype(np.float64), back[k]) for k in w)
print(json.dumps(res, indent=1)); json.dump(res, open(OUT / "result.json", "w"), indent=1)
print("COUNTS", tk.summary())
