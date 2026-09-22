#!/bin/bash
# Launch the smoke detached, loading from the LOCAL plain-file copy. Evidence it started =
# 'launched pid' + smoke.log growing.
export HF_HOME=/root/hf HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
cd /root
nohup python3 /root/smoke_q38.py --model /root/models/Qwen3.8-27B --out /root/out/smoke "$@" > /root/out/smoke.log 2>&1 &
echo "launched pid $!"; sleep 5; tail -3 /root/out/smoke.log
