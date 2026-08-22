"""K resamples of the SAME activation at T=1 — the variance H2 rides on."""
import sys, json, numpy as np, re
sys.path.insert(0, "/root/nla")
from huggingface_hub import snapshot_download
from nla_inference import NLAClient, NLACritic

K = 8
acts = np.load("/root/acts.npy", allow_pickle=True).item()
TARGETS = ["CRICKET::VVS Laxman", "THEIR_EXAMPLE"]

av = NLAClient(snapshot_download("kitft/nla-gemma3-12b-L32-av"),
               sglang_url="http://localhost:30000", device="cpu")
ar = NLACritic(snapshot_download("kitft/nla-gemma3-12b-L32-ar"), device="cuda")

out = {}
for name in TARGETS:
    v, T, pos = acts[name]
    print(f"\n{'='*78}\n{name}   (T={T}, pos={pos}, ||h||={np.linalg.norm(v):.1f})\n{'='*78}")
    rows = []
    for k in range(K):
        d = av.generate(v, temperature=1.0, max_new_tokens=256)
        mse, cos = ar.score(d, v)
        rows.append({"k": k, "mse": float(mse), "cos": float(cos), "explanation": d})
        first = d.strip().split("\n")[0][:110]
        print(f"  k={k}  cos={cos:.4f}  mse={mse:.4f}  len={len(d):4d}  | {first}")
    cs = np.array([r["cos"] for r in rows]); ms = np.array([r["mse"] for r in rows])
    print(f"\n  cos : mean {cs.mean():.4f}  sd {cs.std(ddof=1):.5f}  min {cs.min():.4f}  max {cs.max():.4f}")
    print(f"  mse : mean {ms.mean():.4f}  sd {ms.std(ddof=1):.5f}  spread {ms.max()-ms.min():.4f}")
    print(f"  sd/mean(mse) = {ms.std(ddof=1)/ms.mean():.3f}")
    print(f"  --> averaging K=4 would shrink that sd to ~{ms.std(ddof=1)/2:.5f}, K=9 to ~{ms.std(ddof=1)/3:.5f}")
    lens = [len(r["explanation"]) for r in rows]
    print(f"  explanation length: {min(lens)}-{max(lens)} chars")
    out[name] = rows

json.dump(out, open("/root/variance_K8.json","w"), indent=2)
print("\nsaved -> /root/variance_K8.json")
