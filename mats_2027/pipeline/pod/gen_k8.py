"""Phase 1 (AV): K=8 explanations for ALL 6 pilot activations at T=1. Needs the AV server."""
import sys, json, numpy as np
sys.path.insert(0, "/root/nla")
from huggingface_hub import snapshot_download
from nla_inference import NLAClient
K = 8
acts = np.load("/root/acts.npy", allow_pickle=True).item()
av = NLAClient(snapshot_download("kitft/nla-gemma3-12b-L32-av"), sglang_url="http://localhost:30000", device="cpu")
out = []
for name, (v, T, pos) in acts.items():
    for k in range(K):
        d = av.generate(v, temperature=1.0, max_new_tokens=256)
        out.append({"doc_id": name, "pos": int(pos), "k": k, "explanation": d, "n_chars": len(d)})
        print(f"{name[:28]:28s} k={k} {len(d):4d} chars", flush=True)
json.dump(out, open("/root/explanations_k8.json", "w"), indent=1)
print(f"\n{len(out)} explanations -> /root/explanations_k8.json")
