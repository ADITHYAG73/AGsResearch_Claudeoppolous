"""Phase 3 (AR): paired Δ for every ablation variant. Needs ONLY the AR (no server, no base).
Input : /root/ablations.json — rows {doc_id,pos,k,claim_id,variant,text}
         variant 'full' = intact explanation; 'del:<claim_id>' = that claim's sentence removed
Output: /root/ablation_scores.json — same rows + mse, cos
"""
import sys, json, numpy as np
sys.path.insert(0, "/root/nla")
from huggingface_hub import snapshot_download
from nla_inference import NLACritic
acts = np.load("/root/acts.npy", allow_pickle=True).item()
rows = json.load(open("/root/ablations.json"))
ar = NLACritic(snapshot_download("kitft/nla-gemma3-12b-L32-ar"), device="cuda")
for i, r in enumerate(rows, 1):
    v = acts[r["doc_id"]][0]
    mse, cos = ar.score(r["text"], v)
    r["mse"], r["cos"] = float(mse), float(cos)
    if i % 25 == 0: print(f"{i}/{len(rows)}", flush=True)
json.dump(rows, open("/root/ablation_scores.json", "w"), indent=1)
print(f"{len(rows)} variants scored -> /root/ablation_scores.json")
