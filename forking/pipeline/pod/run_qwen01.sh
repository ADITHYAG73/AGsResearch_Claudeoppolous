#!/bin/bash
# Launch QWEN-01 detached. Standalone launcher (do not inline in an ssh command — see the
# 2026-08-27 hang). Evidence it started: the 'launched pid' line and run.log growing.
export HF_HOME=/root/hf HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
cd /root
nohup python3 /root/run_qwen.py --model Qwen/Qwen3.5-9B --device cuda --dtype bfloat16 \
  --questions /root/row080.json --out-dir /root/out --only-forks \
  --n-samples 200 --n0-samples 200 --base-max-tokens 400 --cont-max-tokens 400 --tok-depth 400 \
  --gen-batch 200 > /root/out/run.log 2>&1 &
echo "launched pid $!"
sleep 5; tail -3 /root/out/run.log
