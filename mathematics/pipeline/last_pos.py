#!/usr/bin/env python3
"""At the FINAL prompt position: the most influential features, with labels and what tokens each boosts.
Flags labels that look like arithmetic content rather than formatting."""
import json, sys, pathlib, re, urllib.request, np_graph
p=pathlib.Path(sys.argv[1]); g=json.load(open(p)); K=int(sys.argv[2]) if len(sys.argv)>2 else 30
model=g["metadata"]["scan"]; toks=g["metadata"]["prompt_tokens"]; last=len(toks)-1
src=[u for u in g["metadata"]["info"].get("source_urls",[]) if "neuronpedia.org" in u][0].rstrip("/").split("/")[-1]
cache_p=p.parent/f"labels_{model}.json"; cache=json.load(open(cache_p)) if cache_p.exists() else {}
def label(n):
    k=f'{n["layer"]}/{n["node_id"].split("_")[1]}'
    if k not in cache:
        try:
            d=json.load(urllib.request.urlopen(urllib.request.Request(f"https://www.neuronpedia.org/api/feature/{model}/{n['layer']}-{src}/{k.split('/')[1]}", headers={"x-api-key":np_graph.env_key()}), timeout=30))
            cache[k]={"ex":[e.get("description") for e in (d.get("explanations") or [])][:1],"pos":(d.get("pos_str") or [])[:8]}
        except Exception as e: cache[k]={"ex":[f"lookup failed {type(e).__name__}"],"pos":[]}
        json.dump(cache, open(cache_p,"w"), ensure_ascii=False)
    c=cache[k]; return (c["ex"][0] if c["ex"] else (n.get("clerp") or "")), c["pos"]
F=[n for n in g["nodes"] if n["feature_type"]=="cross layer transcoder" and n["ctx_idx"]==last]
F.sort(key=lambda n:-(n["influence"] or 0))
print(f"{model}: {len(F)} features at the final position {toks[last]!r}. NOTE 'influence' in this file is cumulative rank mass; lower = more important.")
F.sort(key=lambda n:(n["influence"] or 9))
ARITH=re.compile(r"\b(9|nine|ninet|5|five|add|sum|plus|arith|math|digit|number|calc|total|equal|answer)\b", re.I)
for n in F[:K]:
    ex,pos=label(n); digits=[t for t in pos if re.fullmatch(r"\s?\d+", t or "")]
    flag="*" if (ARITH.search(ex or "") or digits) else " "
    print(f" {flag} L{str(n['layer']):>2} act={n['activation']:7.2f} infl={n['influence']:.3f}  {ex[:44]!r:46} boosts={pos[:6]}")
