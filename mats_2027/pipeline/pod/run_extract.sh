#!/bin/bash
# Phase 1: base model -> last-10 activations (AV server NOT running). Then verify vs acts.npy.
set -x
export HF_TOKEN=$(grep '^export HF_TOKEN=' /root/.bashrc | cut -d= -f2)
export HF_HOME=/root/hf HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
cd /root
python3 stage2_generate.py --corpus pilot_corpus.jsonl --out /root/out --n-pos 10 --extract-only 2>&1
python3 - <<'PY'
import numpy as np, pyarrow.parquet as pq
old = np.load("/root/acts.npy", allow_pickle=True).item()
A = pq.read_table("/root/out/activations.parquet").to_pylist()
print(f"\n=== VERIFY: new last-position vectors vs SMOKE-01 acts.npy ===  rows={len(A)}")
last = {}
for a in A:
    if a["doc_id"] not in last or a["pos"] > last[a["doc_id"]]["pos"]: last[a["doc_id"]] = a
for k,(v,T,pos) in old.items():
    a = last[k]; w = np.asarray(a["activation_vector"], np.float32)
    cos = float(v@w/np.linalg.norm(v)/np.linalg.norm(w)); maxabs = float(np.abs(v-w).max())
    print(f"  {k[:28]:28s} pos old={pos} new={a['pos']}  cos={cos:.6f}  max|diff|={maxabs:.3g}  {'IDENTICAL' if maxabs==0 else ('close' if cos>0.9999 else '*** DIFFERENT ***')}")
from collections import Counter
print("  positions per doc:", dict(Counter(a["doc_id"][:20] for a in A)))
PY
echo "=== EXTRACT DONE ==="
