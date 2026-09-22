#!/bin/bash
# IMAGE-TEST-01: nothing is installed here. If any pip line is ever needed, the image failed.
ts(){ date +%H:%M:%S; }
mkdir -p /root/out /root/models
echo "=========== $(ts) VERIFY ==========="; python3 /opt/interp/verify.py; echo "verify exit=$?"
echo "=========== $(ts) DOWNLOAD ==========="
python3 - <<'PY'
import time; from huggingface_hub import snapshot_download
t0=time.time()
p=snapshot_download("Qwen/Qwen3.8-27B", local_dir="/root/models/Qwen3.8-27B", allow_patterns=["*.json","*.txt","*.jinja","*.safetensors"])
print("DOWNLOAD OK", p, f"{time.time()-t0:.0f}s")
PY
echo "SHARDS $(ls /root/models/Qwen3.8-27B/*.safetensors | wc -l) (expect 18)"
echo "=========== $(ts) SMOKE ==========="
python3 /opt/interp/smoke.py --model /root/models/Qwen3.8-27B --out /root/out/smoke
echo "=========== $(ts) ALL DONE ==========="
