#!/bin/bash
# PARABOLA-01: AG's prompt, verbatim, on Qwen3.8-27B. Uses the patched smoke.py copied to /root
# (the image's /opt/interp/smoke.py has no token streaming). No installs.
ts(){ date +%H:%M:%S; }
mkdir -p /root/out /root/models
echo "=========== $(ts) VERIFY ==========="; python3 /opt/interp/verify.py | tail -4
echo "=========== $(ts) DOWNLOAD ==========="
python3 - <<'PY'
import time; from huggingface_hub import snapshot_download
t0=time.time(); p=snapshot_download("Qwen/Qwen3.8-27B", local_dir="/root/models/Qwen3.8-27B", allow_patterns=["*.json","*.txt","*.jinja","*.safetensors"])
print("DOWNLOAD OK", f"{time.time()-t0:.0f}s")
PY
echo "=========== $(ts) RUN ==========="
python3 /root/smoke_q38.py --model /root/models/Qwen3.8-27B --out /root/out/parabola \
  --max-new-tokens 4096 --answer "-4a" --question "$(cat /root/prompt.txt)"
echo "=========== $(ts) ALL DONE ==========="
