#!/bin/bash
# Phase 2: AV K=4 at every extracted position (server must be up).
export HF_TOKEN=$(grep '^export HF_TOKEN=' /root/.bashrc | cut -d= -f2)
export HF_HOME=/root/hf HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
cd /root
until curl -s localhost:30000/health >/dev/null; do sleep 10; done
echo "=== AV READY ===" ; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
python3 stage2_generate.py --corpus pilot_corpus.jsonl --out /root/out --n-pos 10 --k 4 --skip-extract 2>&1
echo "=== GENERATE DONE ==="
