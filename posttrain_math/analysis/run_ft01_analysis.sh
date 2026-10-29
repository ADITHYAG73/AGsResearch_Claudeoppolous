#!/bin/bash
# FT-01 analysis, run ON THE LAST EVAL POD (never on the Mac). Pulls everything from S3, computes the
# numbers and figures, checks how the training rows were rendered (thinking vs not), uploads results.
# No `set -x` (credentials in the environment).
set -u
source /root/.s3env; source /opt/venv/bin/activate
pip install -q matplotlib 2>&1 | tail -1
A=/root/analysis; mkdir -p $A/ft $A/a40 $A/out $A/train
python - <<'PY'
import boto3, os
s3 = boto3.client("s3", region_name="ap-south-1"); b = os.environ["S3_BUCKET"]
def pull(pre, dst, keep=lambda k: True):
    n = 0
    for p in s3.get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=pre):
        for o in p.get("Contents", []):
            name = o["Key"].split("/")[-1]
            if keep(name): s3.download_file(b, o["Key"], os.path.join(dst, name)); n += 1
    print("pulled", n, "from", pre)
for m in ("Qwen3.5-4B-rerun", "Qwen3.5-4B-ifc4-s050", "Qwen3.5-4B-ifc4-s100"):
    pull(f"posttrain_math/ft01/results/{m}/", "/root/analysis/ft", lambda k: k.startswith(("scored_", "ifeval_")))
pull("posttrain_math/runs/2026-10-06_base/", "/root/analysis/a40", lambda k: k.startswith("scored_Qwen3.5-4B_"))
pull("posttrain_math/ft01/train_logs/", "/root/analysis/train", lambda k: k in ("loss.jsonl", "data_report.json"))
pull("posttrain_math/ft01/data/", "/root/analysis/train", lambda k: k == "train.jsonl")
PY
cd /root/pm
python analysis/ft01_numbers.py $A/ft $A/a40 $A/out/ft01_numbers.json 2>&1 | tee $A/out/ft01_numbers.txt
python analysis/ft01_figures.py $A/out/ft01_numbers.json $A/train/loss.jsonl $A/out/figures 2>&1 | tee -a $A/out/ft01_numbers.txt
# How were the training rows rendered? Count trained turns with non-empty thinking.
python - <<'PY' 2>&1 | tee $A/out/train_think_check.txt
import json, glob, collections
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(glob.glob("/root/models/Qwen3.5-4B-ifc4-s100")[0])
c = collections.Counter(); n = 0
first = json.loads(open("/root/analysis/train/train.jsonl").readline())
print("EXAMPLE trained segment start:", repr(tok.decode(first["input_ids"][first["label_start"] - 6:first["label_start"] + 40])))
for l in open("/root/analysis/train/train.jsonl"):
    r = json.loads(l); t = tok.decode(r["input_ids"][r["label_start"]:]); n += 1
    if "</think>" not in t and "<think>" not in tok.decode(r["input_ids"][max(0, r["label_start"] - 8):r["label_start"]]):
        c["no think tags near trained turn"] += 1
    else:
        body = t.split("</think>")[0] if "</think>" in t else ""
        c["empty thinking" if body.strip() == "" else "non-empty thinking"] += 1
    if n == 5000: print("FIRST 5000 ROWS (the ones trained, train.py takes rows[:K])", dict(c))
print("TRAIN ROWS", n, dict(c))
PY
python /root/pm/ft/s3_sync.py $A/out posttrain_math/ft01/analysis
echo "ANALYSIS DONE"
