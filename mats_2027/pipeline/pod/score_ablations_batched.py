"""Phase 3 (AR): paired Delta for every ablation variant - BATCHED.

Needs ONLY the AR checkpoint. No SGLang server, no base model: the official
nla_inference.py describes NLACritic as "load critic + reconstruct + score (optional, pure
torch)". SGLang serves the AV (generation); the AR is a plain transformers forward pass plus
a Linear head.

Replaces score_ablations.py, which called NLACritic.score() once per variant (batch 1, 0% GPU
utilisation observed in NOISE-01). ABLATE-01 produced 2303 variants.

MANDATORY SELF-CHECK, runs BEFORE the full pass and aborts on failure:
  the first --verify-n variants are scored BOTH ways - the official single-item
  NLACritic.score() and the batched path - and must agree to float tolerance.
The CPU test (pipeline/tests/test_ar_batch.py) already proves the padding/masking/position-id
logic on a tiny random Llama. This on-pod check covers what CPU cannot: the real Gemma-3
checkpoint, its BOS handling, and the trained value_head.

Input : --ablations  parquet or json, rows {doc_id,pos,k,claim_id,variant,text,valid}
        --acts       .npy dict {doc_id: [vector, ...]}  (or a parquet with activation_vector)
Output: --out        same rows + mse, cos
"""
import argparse, json, sys, time
import numpy as np, torch

sys.path.insert(0, "/root/nla")


def load_rows(path):
    if path.endswith(".json"): return json.load(open(path))
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablations", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ar", default="kitft/nla-gemma3-12b-L32-ar")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--verify-n", type=int, default=20)
    ap.add_argument("--atol", type=float, default=1e-4)
    a = ap.parse_args()

    from huggingface_hub import snapshot_download
    from nla_inference import NLACritic
    from ar_batch import reconstruct_batch, score_batch

    rows = load_rows(a.ablations)
    acts = dict(np.load(a.acts))          # .npz keyed "doc_id||pos" -> float32[d_model]
    ar = NLACritic(snapshot_download(a.ar), device="cuda")
    print(f"{len(rows)} variants · {len(acts)} activations · mse_scale={ar.mse_scale:.4f} · "
          f"batch_size={a.batch_size}", flush=True)
    missing = {f'{r["doc_id"]}||{r["pos"]}' for r in rows} - set(acts)
    if missing:
        raise SystemExit(f"ABORT: {len(missing)} (doc_id,pos) keys have no activation, "
                         f"e.g. {sorted(missing)[:3]}")

    def gold_for(r):
        """Exact (doc_id, pos) lookup. NO fallbacks by design: POS-01 has 10 positions per
        passage, and silently scoring against the wrong position would corrupt every Delta
        while looking completely normal. A KeyError here is the correct failure."""
        return acts[f'{r["doc_id"]}||{r["pos"]}']

    # ---- self-check: batched must reproduce the official single-item path ----
    probe = rows[:a.verify_n]
    t0 = time.time()
    ser = [ar.score(r["text"], gold_for(r)) for r in probe]
    t_ser = time.time() - t0
    t0 = time.time()
    preds = reconstruct_batch(ar.backbone, ar.tokenizer, ar.value_head, ar.template,
                              [r["text"] for r in probe], device="cuda",
                              batch_size=a.batch_size)
    bm, bc = [], []
    for r, p in zip(probe, preds):
        m, c = score_batch(p.unsqueeze(0), gold_for(r), ar.mse_scale)
        bm.append(m[0]); bc.append(c[0])
    t_bat = time.time() - t0
    dm = max(abs(s[0] - b) for s, b in zip(ser, bm))
    dc = max(abs(s[1] - b) for s, b in zip(ser, bc))
    print(f"self-check on {len(probe)}: max|dmse|={dm:.2e}  max|dcos|={dc:.2e}  "
          f"serial {t_ser:.1f}s vs batched {t_bat:.1f}s ({t_ser/max(t_bat,1e-9):.1f}x)", flush=True)
    if dm > a.atol or dc > a.atol:
        raise SystemExit(f"ABORT: batched != serial (atol={a.atol}). Do not trust these Deltas.")
    print("self-check PASSED\n", flush=True)

    # ---- full pass, grouped by activation so gold is fetched once ----
    t0 = time.time()
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["doc_id"], rows[i]["pos"]))
    for s in range(0, len(order), a.batch_size):
        idx = order[s:s + a.batch_size]
        preds = reconstruct_batch(ar.backbone, ar.tokenizer, ar.value_head, ar.template,
                                  [rows[i]["text"] for i in idx], device="cuda",
                                  batch_size=a.batch_size)
        for i, p in zip(idx, preds):
            m, c = score_batch(p.unsqueeze(0), gold_for(rows[i]), ar.mse_scale)
            rows[i]["mse"], rows[i]["cos"] = float(m[0]), float(c[0])
        if (s // a.batch_size) % 20 == 0:
            print(f"  {s + len(idx)}/{len(order)}  {time.time()-t0:.0f}s", flush=True)

    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\n{len(rows)} scored -> {a.out}  in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
