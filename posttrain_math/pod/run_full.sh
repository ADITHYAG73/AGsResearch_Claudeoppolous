#!/bin/bash
# Full baseline for one model: all questions, four conditions, one model load.
# Cheap conditions first; thinking-alone last, core subset first. Safe to rerun: it resumes.
set -u
M="${1:?model id}"
source /opt/venv/bin/activate
export HF_HOME=/opt/hf HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
cd /root/pm
python run_eval.py --model "$M" --questions questions_v0.jsonl --out /root/out/base \
  --cond notool_off,tool_off,tool_on,notool_on --core-first --chunk 256
python score.py questions_v0.jsonl "/root/out/base/raw_$(basename $M)_*.jsonl"
echo "FULL DONE $M"
