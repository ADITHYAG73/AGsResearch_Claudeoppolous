#!/bin/bash
# After the main run ends, re-ask ONLY the agentic thinking-on questions that hit the 6-turn limit,
# with the limit raised to 12. Writes a separately tagged file; the 6-turn data is not modified.
# Waits as its own process, so the running job is never touched.
M="${1:?model id}"; N=$(basename "$M")
for i in $(seq 1 2000); do grep -q "FULL DONE" /root/out/full.log 2>/dev/null && break; sleep 30; done
grep -q "FULL DONE" /root/out/full.log || { echo "REASK GATE TIMEOUT"; exit 1; }
source /opt/venv/bin/activate
export HF_HOME=/opt/hf HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
cd /root/pm
IDS=$(python -c "
import json
print(','.join(r['id'] for r in map(json.loads, open('/root/out/base/raw_${N}_tool_on.jsonl')) if r['end']=='turn_limit'))")
echo "$(date +%H:%M:%S) re-asking $(echo "$IDS" | tr ',' '\n' | grep -c .) turn-limit questions with max-turns 12"
[ -z "$IDS" ] && { echo "REASK DONE $M (nothing to re-ask)"; exit 0; }
python run_eval.py --model "$M" --questions questions_v0.jsonl --out /root/out/base \
  --cond tool_on --ids "$IDS" --max-turns 12 --tag turns12 --chunk 256
python score.py questions_v0.jsonl "/root/out/base/raw_${N}_tool_on_turns12.jsonl"
echo "REASK DONE $M"
