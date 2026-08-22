"""AV -> AR round trip on every extracted activation. Needs the AV server up."""
import sys, json, torch, numpy as np
sys.path.insert(0, "/root/nla")
from huggingface_hub import snapshot_download
from nla_inference import NLAClient, NLACritic

acts = np.load("/root/acts.npy", allow_pickle=True).item()
texts = json.load(open("/root/passages.json"))

print("=== AV: verbalizing each activation ===")
AV_DIR = snapshot_download("kitft/nla-gemma3-12b-L32-av")
av = NLAClient(AV_DIR, sglang_url="http://localhost:30000", device="cpu")
descs = {}
for name, (v, T, pos) in acts.items():
    d = av.generate(v, temperature=1.0, max_new_tokens=256)
    descs[name] = d
    print(f"\n--- {name}  (T={T}, pos={pos}, ||h||={np.linalg.norm(v):.1f}) ---")
    print(d[:600] + ("..." if len(d) > 600 else ""))

print("\n\n=== AR: scoring reconstructions ===")
AR_DIR = snapshot_download("kitft/nla-gemma3-12b-L32-ar")
ar = NLACritic(AR_DIR, device="cuda")

rows = []
print(f"\n{'passage':40s} {'T':>5s} {'pos':>5s} {'||h||':>10s} {'mse':>8s} {'cos':>8s}")
print("-"*82)
for name, (v, T, pos) in acts.items():
    mse, cos = ar.score(descs[name], v)
    rows.append({"passage":name,"T":int(T),"pos":int(pos),
                 "norm":float(np.linalg.norm(v)),"mse":float(mse),"cos":float(cos),
                 "explanation":descs[name]})
    print(f"{name:40s} {T:5d} {pos:5d} {np.linalg.norm(v):10.1f} {mse:8.3f} {cos:8.3f}")

json.dump(rows, open("/root/roundtrip_results.json","w"), indent=2)
print("\n=== SUMMARY ===")
their = [r for r in rows if not r["passage"].startswith("CRICKET")]
cric  = [r for r in rows if r["passage"].startswith("CRICKET")]
for lbl, grp in [("their example", their), ("cricket (Wikipedia)", cric)]:
    if grp:
        cs = [r["cos"] for r in grp]
        print(f"  {lbl:22s} n={len(cs)}  cos mean {np.mean(cs):.3f}  min {min(cs):.3f}  max {max(cs):.3f}")
print("\nRecorded 2026-06-07 Gemma-3-12B result: cos ~ 0.997")
print("saved -> /root/roundtrip_results.json")
