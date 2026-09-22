#!/usr/bin/env python3
"""Walk an attribution graph token by token: for each prompt position, the most influential
features there, with Neuronpedia's label. Answers: WHERE in the prompt does the work happen and
what do the features at the operand digits look like? Labels cached on disk."""
import json, sys, pathlib, urllib.request, collections, np_graph
p=pathlib.Path(sys.argv[1]); g=json.load(open(p)); K=int(sys.argv[2]) if len(sys.argv)>2 else 4
model=g["metadata"]["scan"]; toks=g["metadata"]["prompt_tokens"]
src=[u for u in g["metadata"]["info"].get("source_urls",[]) if "neuronpedia.org" in u][0].rstrip("/").split("/")[-1]
cache_p=p.parent/f"labels_{model}.json"; cache=json.load(open(cache_p)) if cache_p.exists() else {}
def label(n):
    k=f'{n["layer"]}/{n["node_id"].split("_")[1]}'
    if k not in cache:
        try:
            d=json.load(urllib.request.urlopen(urllib.request.Request(f"https://www.neuronpedia.org/api/feature/{model}/{n['layer']}-{src}/{k.split('/')[1]}", headers={"x-api-key":np_graph.env_key()}), timeout=30))
            ex=[e.get("description") for e in (d.get("explanations") or [])]; cache[k]={"ex":ex[:1],"pos":(d.get("pos_str") or [])[:6]}
        except Exception as e: cache[k]={"ex":[f"lookup failed {type(e).__name__}"],"pos":[]}
        json.dump(cache, open(cache_p,"w"), ensure_ascii=False)
    c=cache[k]; return (c["ex"][0] if c["ex"] else ""), c["pos"]
F=[n for n in g["nodes"] if n["feature_type"]=="cross layer transcoder"]; E=[n for n in g["nodes"] if n["feature_type"]=="mlp reconstruction error"]
print(f"{model} | source {src} | prompt tokens: {toks}")
print(f"features {len(F)}, error nodes {len(E)}; influence explained by features: {sum(n['influence'] or 0 for n in F)/(sum(n['influence'] or 0 for n in F)+sum(n['influence'] or 0 for n in E)):.0%}")
bypos=collections.defaultdict(list)
for n in F: bypos[n["ctx_idx"]].append(n)
print("\nfeature count per token position:", {repr(toks[i]):len(bypos[i]) for i in range(len(toks))})
for i in range(len(toks)):
    if not bypos[i] or toks[i] in ("<|endoftext|>","<bos>"): continue
    top=sorted(bypos[i], key=lambda n:-(n["activation"] or 0))[:K]
    print(f"\n--- position {i}: token {toks[i]!r} — top {K} by activation ---")
    for n in top:
        ex,pos=label(n); print(f"   L{str(n['layer']):>2}  act={n['activation']:.2f}  {ex!r:48} boosts={pos}")
