#!/usr/bin/env python3
"""Neuronpedia Circuit Tracer helper. Generates an attribution graph through the API, saves the
JSON, and prints what the model predicts next. Key is read from .env (NEURONPEDIA_APIKEY) and
never printed. Usage: np_graph.py --model qwen3-4b --tag few5 --prompt '...'"""
import argparse, json, os, re, sys, time, urllib.request, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
def env_key():
    for line in open(ROOT/".env"):
        if line.startswith("NEURONPEDIA_APIKEY"): return line.split("=",1)[1].strip().strip('"').strip("'")
    sys.exit("no NEURONPEDIA_APIKEY in .env")
def generate(prompt, model, tag, outdir):
    slug = re.sub(r"[^a-z0-9-]", "", tag.lower())[:20] + "-" + time.strftime("%m%d%H%M%S")
    body = json.dumps({"prompt": prompt, "modelId": model, "slug": slug}).encode()
    req = urllib.request.Request("https://www.neuronpedia.org/api/graph/generate", data=body,
          headers={"Content-Type": "application/json", "x-api-key": env_key()})
    t0 = time.time(); r = None
    for attempt in range(1, 5):   # their backend returns sporadic HTTP 500 (seen 2026-09-19); retry with a pause
        try: r = json.load(urllib.request.urlopen(req, timeout=280)); break
        except Exception as e:
            code = getattr(e, "code", ""); body = e.read()[:200].decode("utf-8", "replace") if hasattr(e, "read") else str(e)
            print(f"  attempt {attempt} failed: {type(e).__name__} {code} {body}", flush=True)
            if attempt == 4: raise SystemExit("gave up after 4 attempts")
            time.sleep(25)
            body = json.dumps({"prompt": prompt, "modelId": model, "slug": f"{slug}r{attempt}"}).encode()
            req = urllib.request.Request("https://www.neuronpedia.org/api/graph/generate", data=body,
                  headers={"Content-Type": "application/json", "x-api-key": env_key()})
    g = json.load(urllib.request.urlopen(r["s3url"], timeout=180))
    outdir.mkdir(parents=True, exist_ok=True); p = outdir/f"graph_{model}_{slug}.json"; json.dump(g, open(p, "w"))
    return g, r["url"], p, time.time()-t0
def top_logits(g, k=6):
    L = [n for n in g["nodes"] if n["feature_type"] == "logit"]
    return sorted(((n["token_prob"], n["clerp"]) for n in L), reverse=True)[:k]
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--prompt", required=True); ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--tag", default="g"); ap.add_argument("--out", default=str(ROOT/"mathematics/runs/2026-09-20_circuit_tracer"))
    a = ap.parse_args(); g, url, p, dt = generate(a.prompt.encode().decode("unicode_escape"), a.model, a.tag, pathlib.Path(a.out))
    print(f"[{a.tag}] {a.model} {dt:.0f}s  tokens={len(g['metadata']['prompt_tokens'])}  {url}\n  file: {p.name}")
    for pr, c in top_logits(g): print(f"   {pr:.3f}  {c}")
