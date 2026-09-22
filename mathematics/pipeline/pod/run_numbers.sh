#!/bin/bash
ts(){ date +%H:%M:%S; }
mkdir -p /root/out /root/models
echo "=========== $(ts) VERIFY ==========="; python3 /opt/interp/verify.py | tail -4
echo "=========== $(ts) DOWNLOAD ==========="
python3 - <<'PY'
import time; from huggingface_hub import snapshot_download
t0=time.time(); snapshot_download("Qwen/Qwen3.8-27B", local_dir="/root/models/Qwen3.8-27B", allow_patterns=["*.json","*.txt","*.jinja","*.safetensors"])
print("DOWNLOAD OK", f"{time.time()-t0:.0f}s")
PY
echo "=========== $(ts) PROBE ==========="
python3 /root/numbers_probe.py --model /root/models/Qwen3.8-27B --out /root/out/numbers
echo "exit=$?"
echo "=========== $(ts) ALL DONE ==========="
