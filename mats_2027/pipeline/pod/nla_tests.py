"""Smoke tests + the two measurements we owe. Run AFTER the AV server is up."""
import os, sys, gc, json, torch, numpy as np
sys.path.insert(0, "/root/nla")
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE, LAYER, MIN_P = "google/gemma-3-12b-it", 32, 50
tok = AutoTokenizer.from_pretrained(BASE)

# ---------- TEST A: marker ids on the POD's transformers 5.3.0 ----------
import yaml
from huggingface_hub import hf_hub_download
meta = yaml.safe_load(open(hf_hub_download("kitft/nla-gemma3-12b-L32-av", "nla_meta.yaml")))
t = meta["tokens"]
content = meta["prompt_templates"]["av"].format(injection_char=t["injection_char"])
ids = tok.apply_chat_template([{"role":"user","content":content}],
                              tokenize=True, add_generation_prompt=True, return_dict=False)
if not isinstance(ids, list): ids = ids["input_ids"] if hasattr(ids,"keys") else list(ids)
p = ids.index(t["injection_token_id"])
print("=== TEST A: marker check on transformers 5.3.0 ===")
print(f"  prompt len {len(ids)}, marker at {p}")
for nm, got, want in [("left",ids[p-1],t["injection_left_neighbor_id"]),
                      ("mark",ids[p],  t["injection_token_id"]),
                      ("right",ids[p+1],t["injection_right_neighbor_id"])]:
    print(f"  {nm:5s} got {got:7d} sidecar {want:7d} {'MATCH' if got==want else '*** MISMATCH ***'}")
print(f"  marker occurrences: {ids.count(t['injection_token_id'])}\n")

# ---------- load base, TEST B: embedding norms, then extract ----------
print("=== loading base model ===")
m = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("cuda").eval()

print("\n=== TEST B: embedding norms vs injection_scale (the gap Claude admitted) ===")
emb = m.model.language_model.embed_tokens.weight if hasattr(m.model,"language_model") else m.model.embed_tokens.weight
rn = emb.float().norm(dim=-1)
es, inj = 61.9677, meta["extraction"]["injection_scale"]
print(f"  raw table row norm : median {rn.median():.4f}  mean {rn.mean():.4f}  p75 {rn.quantile(.75):.4f}")
print(f"  x embed_scale      : median {rn.median()*es:.2f}   <- what the model actually sees")
print(f"  injection_scale    : {inj}")
print(f"  ratio injected/token: {inj/(rn.median().item()*es):.1f}x\n")

PASSAGES = {"THEIR_EXAMPLE": (
  "The French Revolution began in 1789 amid a fiscal crisis and widespread anger at the "
  "monarchy. As bread prices soared and the Estates-General convened, ordinary citizens in "
  "Paris grew restless, and by July the city stood on the brink of open revolt against King "
  "Louis XVI. The storming of the Bastille on 14 July became the emblem of that rupture, and "
  "within weeks the National Assembly had abolished feudal privileges across France.")}
PASSAGES.update({f"CRICKET::{k}": v for k, v in json.load(open("/root/cricket.json")).items()})

acts = {}
print("=== TEST C: activation norms per passage (layer 32, last valid position) ===")
for name, text in PASSAGES.items():
    enc = tok(text, return_tensors="pt").to("cuda")
    n = enc["input_ids"].shape[1]
    if n <= MIN_P:
        print(f"  {name:38s} SKIPPED ({n} tokens <= {MIN_P})"); continue
    with torch.no_grad():
        hs = m(**enc, output_hidden_states=True).hidden_states[LAYER+1][0]
    pos = n-1
    v = hs[pos].float().cpu().numpy()
    acts[name] = (v, text, n, pos)
    print(f"  {name:38s} T={n:4d} pos={pos:4d} ||h||={np.linalg.norm(v):9.1f}")

del m; gc.collect(); torch.cuda.empty_cache()
np.save("/root/acts.npy", {k:(v[0],v[2],v[3]) for k,v in acts.items()}, allow_pickle=True)
json.dump({k:v[1] for k,v in acts.items()}, open("/root/passages.json","w"), indent=2)
print(f"\n  saved {len(acts)} activations -> /root/acts.npy")
