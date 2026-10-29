#!/bin/bash
# 56 questions spread over every family, all four conditions, one model load.
set -u
source /opt/venv/bin/activate
export HF_HOME=/opt/hf HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
cd /root/pm
python run_eval.py --model "${1:-Qwen/Qwen3.5-9B}" --questions questions_v0.jsonl --out /root/out/smoke \
  --cond notool_off,tool_off,notool_on,tool_on --core --stride 12 --chunk 64
python score.py questions_v0.jsonl "/root/out/smoke/raw_*.jsonl"
echo "SMOKE DONE"
