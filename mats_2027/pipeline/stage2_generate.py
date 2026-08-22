"""Stage 2 — extract activations + generate explanations. RUNS ON THE POD (Session A).

Two output tables (parquet):
  activations.parquet  — one row per (doc_id, pos).  Mirrors the official Stage 0 schema:
                         n_raw_tokens · detokenized_text_truncated · activation_vector
                         (FixedSizeList float32 d_model) · activation_layer · doc_id
                         + our extras: pos, title, bucket, article_chars
  explanations.parquet — one row per (doc_id, pos, k). Joins on (doc_id, pos).

Position policy: LAST N_POS contiguous positions (default 10) — adjacency is required
for the recurrence signal to mean "stability of the readout".
Sampling: temperature 1.0, K resamples per activation.

Usage on pod:   python stage2_generate.py --corpus cricket_passages.jsonl --out /root/out
Dry run local:  python stage2_generate.py --dry-run   (uses SMOKE-01 activations, no GPU)
"""
import argparse, gc, json, os, sys, time
import numpy as np
import pyarrow as pa, pyarrow.parquet as pq

BASE, LAYER, D = "google/gemma-3-12b-it", 32, 3840
AV_REPO = "kitft/nla-gemma3-12b-L32-av"
MIN_POSITION = 50                         # official _MIN_POSITION

def act_schema():
    return pa.schema([
        ("doc_id", pa.string()), ("pos", pa.int64()),
        ("n_raw_tokens", pa.int64()),
        ("detokenized_text_truncated", pa.string()),
        ("activation_vector", pa.list_(pa.float32(), D)),      # FixedSizeList, as official
        ("activation_layer", pa.int64()),
        ("title", pa.string()), ("bucket", pa.string()), ("article_chars", pa.int64()),
        ("activation_norm", pa.float32()),
    ])

def expl_schema():
    return pa.schema([
        ("doc_id", pa.string()), ("pos", pa.int64()), ("k", pa.int64()),
        ("explanation", pa.string()), ("raw_generation", pa.string()),
        ("had_tags", pa.bool_()), ("n_chars", pa.int64()),
        ("temperature", pa.float32()), ("max_new_tokens", pa.int64()),
        ("gen_seconds", pa.float32()),
    ])

def extract(rows, n_pos, tok, model):
    """Base model forward → last n_pos activations per passage. Yields dicts."""
    import torch
    for r in rows:
        enc = tok(r["text"], return_tensors="pt").to("cuda")
        ids = enc["input_ids"][0]; T = ids.shape[0]
        special = set(tok.all_special_ids)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states[LAYER + 1][0]   # [T, D]
        for p in range(max(MIN_POSITION, T - n_pos), T):
            if ids[p].item() in special: continue
            v = hs[p].float().cpu().numpy()
            yield {
                "doc_id": r["doc_id"], "pos": p, "n_raw_tokens": p + 1,
                "detokenized_text_truncated": tok.decode(ids[: p + 1], skip_special_tokens=True),
                "activation_vector": v, "activation_layer": LAYER,
                "title": r["title"], "bucket": r["bucket"], "article_chars": r["article_chars"],
                "activation_norm": float(np.linalg.norm(v)),
            }

def generate(acts, K, temperature, max_new_tokens, av):
    """K explanations per activation, with raw output kept for the CJK/no-tag diagnostics."""
    import re
    TAG = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)
    for a in acts:
        for k in range(K):
            t0 = time.time()
            raw = av.generate(a["activation_vector"], temperature=temperature,
                              max_new_tokens=max_new_tokens, extract_explanation=False)
            m = TAG.search(raw)
            yield {
                "doc_id": a["doc_id"], "pos": a["pos"], "k": k,
                "explanation": m.group(1).strip() if m else raw,
                "raw_generation": raw, "had_tags": m is not None,
                "n_chars": len(m.group(1).strip() if m else raw),
                "temperature": temperature, "max_new_tokens": max_new_tokens,
                "gen_seconds": time.time() - t0,
            }

def write(rows, schema, path):
    tbl = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(tbl, path)
    return len(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="cricket_passages.jsonl")
    ap.add_argument("--out", default="out")
    ap.add_argument("--n-pos", type=int, default=10)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    ap.add_argument("--limit", type=int, default=None, help="first N passages only (pilot)")
    ap.add_argument("--dry-run", action="store_true", help="local: schema + writer check on SMOKE-01 data")
    ap.add_argument("--extract-only", action="store_true", help="pod: base model → activations.parquet, then exit (AV server NOT needed)")
    ap.add_argument("--skip-extract", action="store_true", help="pod: load <out>/activations.parquet and go straight to AV")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.dry_run:
        # No GPU. Use the 6 SMOKE-01 activations to prove the schemas and writers work.
        acts_np = np.load("mats_2027/runs/2026-08-20_smoke/acts.npy", allow_pickle=True).item()
        texts = json.load(open("mats_2027/runs/2026-08-20_smoke/passages.json"))
        rt = {r["passage"]: r for r in json.load(open("mats_2027/runs/2026-08-20_smoke/roundtrip_results.json"))}
        acts = [{"doc_id": n, "pos": int(pos), "n_raw_tokens": int(pos)+1,
                 "detokenized_text_truncated": texts[n], "activation_vector": v.astype(np.float32),
                 "activation_layer": LAYER, "title": n, "bucket": "smoke",
                 "article_chars": 0, "activation_norm": float(np.linalg.norm(v))}
                for n, (v, T, pos) in acts_np.items()]
        expls = [{"doc_id": n, "pos": a["pos"], "k": 0, "explanation": rt[n]["explanation"],
                  "raw_generation": rt[n]["explanation"], "had_tags": True,
                  "n_chars": len(rt[n]["explanation"]), "temperature": 1.0,
                  "max_new_tokens": 256, "gen_seconds": 0.0}
                 for n, a in zip(acts_np, acts)]
        na = write(acts, act_schema(), f"{args.out}/activations.parquet")
        ne = write(expls, expl_schema(), f"{args.out}/explanations.parquet")
        # read back + join, the way analysis will
        A = pq.read_table(f"{args.out}/activations.parquet"); E = pq.read_table(f"{args.out}/explanations.parquet")
        v0 = np.array(A.column("activation_vector")[0].as_py(), dtype=np.float32)
        print(f"DRY RUN OK: {na} activations, {ne} explanations")
        print(f"  activations schema : {A.schema.names}")
        print(f"  vector round-trips : shape {v0.shape}, norm {np.linalg.norm(v0):.1f} "
              f"(orig {acts[0]['activation_norm']:.1f})")
        print(f"  join key present   : {set(E.column('doc_id').to_pylist()) == set(A.column('doc_id').to_pylist())}")
        return

    # ---------------- real run (pod) ----------------
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = [json.loads(l) for l in open(args.corpus)]
    if args.limit: rows = rows[: args.limit]
    print(f"{len(rows)} passages · last {args.n_pos} positions · K={args.k} · T={args.temperature}")

    if args.skip_extract:
        A = pq.read_table(f"{args.out}/activations.parquet").to_pylist()
        for a in A: a["activation_vector"] = np.asarray(a["activation_vector"], dtype=np.float32)
        acts = A; n = len(acts)
        print(f"[1/3] loaded {n} activations from activations.parquet (skip-extract)")
    else:
        tok = AutoTokenizer.from_pretrained(BASE)
        print("[1/3] base model → activations"); t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to("cuda").eval()
        acts = list(extract(rows, args.n_pos, tok, model))
        del model; gc.collect(); torch.cuda.empty_cache()
        n = write(acts, act_schema(), f"{args.out}/activations.parquet")
        print(f"      {n} activations in {time.time()-t0:.0f}s → activations.parquet")
        for a in acts:
            print(f"      {a['doc_id'][:30]:30s} pos={a['pos']:4d} T={a['n_raw_tokens'] if a['pos']+1==a['n_raw_tokens'] else '?':>4} ||h||={a['activation_norm']:9.1f}")
        if args.extract_only:
            print("      --extract-only: done."); return

    print("[2/3] AV → explanations"); t0 = time.time()
    sys.path.insert(0, "/root/nla")
    from huggingface_hub import snapshot_download
    from nla_inference import NLAClient
    av = NLAClient(snapshot_download(AV_REPO), sglang_url=args.sglang_url, device="cpu")
    expls, no_tag = [], 0
    for i, e in enumerate(generate(acts, args.k, args.temperature, args.max_new_tokens, av), 1):
        expls.append(e); no_tag += (not e["had_tags"])
        if i % 50 == 0:
            print(f"      {i}/{n*args.k}  no-tag so far {no_tag}  ({time.time()-t0:.0f}s)", flush=True)
            pq.write_table(pa.Table.from_pylist(expls, schema=expl_schema()),
                           f"{args.out}/explanations.partial.parquet")      # checkpoint
    m = write(expls, expl_schema(), f"{args.out}/explanations.parquet")
    if os.path.exists(f"{args.out}/explanations.partial.parquet"):
        os.remove(f"{args.out}/explanations.partial.parquet")
    print(f"      {m} explanations in {time.time()-t0:.0f}s · no-tag {no_tag} ({100*no_tag/max(m,1):.1f}%)")

    print("[3/3] summary")
    cjk = sum(any('　' <= c <= '鿿' or '가' <= c <= '힯' for c in e["explanation"]) for e in expls)
    print(f"      CJK-containing explanations: {cjk}  (should be ~0; >0 ⇒ check injection)")
    print(f"      mean chars {np.mean([e['n_chars'] for e in expls]):.0f}  "
          f"mean gen {np.mean([e['gen_seconds'] for e in expls]):.2f}s")

if __name__ == "__main__":
    main()
