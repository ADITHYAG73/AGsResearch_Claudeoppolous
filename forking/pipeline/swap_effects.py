"""SWAP-01 (compass Q3): per-token swap effects o_{t,w} on the shipped 2026 stores.

Question: at positions where the pooled curve o_t is flat, do individual next-token branches
still produce different final-answer distributions, beyond what sampling noise allows?

For every store, every position with >=2 kept branches, every alternative branch w vs base:
  effect   = TVD( o_{t,base}, o_{t,w} )   each from its 200 recorded draws
  p-value  = permutation test: pool the 400 outcomes, shuffle, split 200/200, TVD; 500 reps
  local curve change = TVD( o_t, o_{t+1 grid step} ) from the recorded o_t_full
Also decodes the alternative token for a token-class breakdown (punct / whitespace / digit /
word / other).
"""
import sys, json, gzip, os, unicodedata
import numpy as np
sys.path.insert(0, "reference_repos/forking-fast/data")
import loader
from transformers import AutoTokenizer

DATA = "reference_repos/forking-fast/data/s200"
OUT = "forking/runs/2026-09-13_swap"
REPS = 500
rng = np.random.default_rng(20260913)
TOK = {"llama": AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct"),
       "deepseek": AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")}

def tok_class(s):
    t = s.strip()
    if t == "": return "whitespace"
    if all(unicodedata.category(c).startswith("P") or unicodedata.category(c).startswith("S") for c in t): return "punct/symbol"
    if t.isdigit(): return "digit"
    if t.isalpha(): return "word"
    return "mixed"

def tvd(p, q): return 0.5 * np.abs(p - q).sum()

def frac(codes, K=5):
    h = np.bincount(codes, minlength=K).astype(float); return h / h.sum()

rows = []
for track in ["llama", "deepseek"]:
    files = sorted(f for f in os.listdir(f"{DATA}/{track}") if f.endswith(".json.gz"))
    for fn in files:
        rec = loader.load_store(f"{DATA}/{track}/{fn}")
        cats = rec["categories"]; ci = {c: i for i, c in enumerate(cats)}
        by_t = loader.branch_groups(rec); idxs = sorted(by_t)
        o_full = np.asarray(rec["o_t_full"], float)
        pos_of = {t: i for i, t in enumerate(idxs)}
        for t in idxs:
            bs = by_t[t]
            if len(bs) < 2: continue
            base = [b for b in bs if b["is_base"]]
            if not base: continue
            base = base[0]
            cb = np.array([ci.get(a, ci["Other"]) for a in base["answers"]])
            i = pos_of[t]
            local = tvd(o_full[i], o_full[i + 1]) if i + 1 < len(idxs) else np.nan
            for b in bs:
                if b["is_base"]: continue
                cw = np.array([ci.get(a, ci["Other"]) for a in b["answers"]])
                eff = tvd(frac(cb), frac(cw))
                pool = np.concatenate([cb, cw]); n = len(cb)
                null = np.empty(REPS)
                for r in range(REPS):
                    rng.shuffle(pool); null[r] = tvd(frac(pool[:n]), frac(pool[n:]))
                p = float((null >= eff).mean())
                rows.append(dict(track=track, row=rec["meta"]["row_id"], t=t, n_branches=len(bs),
                                 alt_tok=TOK[track].decode([b["tok_id"]]), alt_class=tok_class(TOK[track].decode([b["tok_id"]])),
                                 base_tok=TOK[track].decode([base["tok_id"]]), alt_p=b["tok_p"], base_p=base["tok_p"],
                                 effect=eff, null_mean=float(null.mean()), null_q95=float(np.quantile(null, .95)),
                                 p_value=p, local_change=local, pooled_maxcat=float(o_full[i].max())))
        print(track, fn, "done", len(rows), flush=True)

import pandas as pd
df = pd.DataFrame(rows); df.to_parquet(f"{OUT}/swap_effects.parquet"); df.to_csv(f"{OUT}/swap_effects.csv", index=False)
print(df.groupby("track").size())
