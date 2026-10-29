#!/bin/bash
# Evaluate ONE model on the eval pod: the frozen 3,362 maths questions x 4 conditions (identical
# to BASE-01: run_eval.py, --core-first, chunk 256, greedy), the 12-turn re-ask, then IFEval.
# Results sync to S3 every 5 minutes, so a kill loses at most 5 minutes of finished work.
#   bash eval_model.sh base                      -> untouched Qwen3.5-4B at the pinned commit
#   bash eval_model.sh Qwen3.5-4B-ifc4-s050      -> checkpoint pulled from S3
# Credentials come from /root/.s3env (written over ssh, chmod 600). NO `set -x` in this file:
# it would print them (the HF token leaked that way on 2026-08-22).
set -u
WHICH="${1:?base or checkpoint name}"
source /root/.s3env
source /opt/venv/bin/activate
export HF_HOME=/opt/hf HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
cd /root/pm
mkdir -p /root/models /root/out/ft01
if [ "$WHICH" = "base" ]; then
  M=/root/models/Qwen3.5-4B-rerun
  python -c "from huggingface_hub import snapshot_download as s; s('Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', local_dir='$M')"
else
  M=/root/models/$WHICH
  # wait (bounded, 6 h) for the training pod's _COMPLETE marker
  python - <<PY
import boto3, os, time
s3 = boto3.client("s3", region_name="ap-south-1"); b = os.environ["S3_BUCKET"]
key = "posttrain_math/ft01/checkpoints/$WHICH/_COMPLETE"
for i in range(720):
    try: s3.head_object(Bucket=b, Key=key); print("checkpoint complete marker found"); break
    except Exception:
        if i % 10 == 0: print(time.strftime("%H:%M:%S"), "waiting for", key, flush=True)
        time.sleep(30)
else: raise SystemExit("TIMEOUT waiting for checkpoint")
PY
  python - <<PY
import boto3, os
s3 = boto3.client("s3", region_name="ap-south-1"); b = os.environ["S3_BUCKET"]
pre = "posttrain_math/ft01/checkpoints/$WHICH/"
n = 0
for page in s3.get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=pre):
    for o in page.get("Contents", []):
        dst = os.path.join("$M", o["Key"][len(pre):]); os.makedirs(os.path.dirname(dst), exist_ok=True)
        s3.download_file(b, o["Key"], dst); n += 1
print("downloaded", n, "files to $M")
assert n > 10, "checkpoint incomplete"
PY
fi
echo "$(date +%H:%M:%S) MODEL READY $M"; ls -la "$M" | head -30

# background S3 sync of results, every 5 min (bounded: stops after 12 h)
( for i in $(seq 1 144); do python /root/pm/ft/s3_sync.py /root/out/ft01 "posttrain_math/ft01/results/$(basename $M)" >/dev/null 2>&1; sleep 300; done ) &
SYNC=$!

python run_eval.py --model "$M" --questions questions_v0.jsonl --out /root/out/ft01 \
  --cond notool_off,tool_off,tool_on,notool_on --core-first --chunk 256
python score.py questions_v0.jsonl "/root/out/ft01/raw_$(basename $M)_*.jsonl"
echo "$(date +%H:%M:%S) MATH DONE"

N=$(basename "$M")
IDS=$(python -c "
import json
print(','.join(r['id'] for r in map(json.loads, open('/root/out/ft01/raw_${N}_tool_on.jsonl')) if r['end']=='turn_limit'))")
echo "$(date +%H:%M:%S) re-asking $(echo "$IDS" | tr ',' '\n' | grep -c .) turn-limit questions with max-turns 12"
if [ -n "$IDS" ]; then
  python run_eval.py --model "$M" --questions questions_v0.jsonl --out /root/out/ft01 \
    --cond tool_on --ids "$IDS" --max-turns 12 --tag turns12 --chunk 256
  python score.py questions_v0.jsonl "/root/out/ft01/raw_${N}_tool_on_turns12.jsonl"
fi
echo "$(date +%H:%M:%S) REASK DONE"

python ft/ifeval.py --model "$M" --out /root/out/ft01
kill $SYNC 2>/dev/null
python /root/pm/ft/s3_sync.py /root/out/ft01 "posttrain_math/ft01/results/$N"
echo "$(date +%H:%M:%S) ALL DONE $N"
