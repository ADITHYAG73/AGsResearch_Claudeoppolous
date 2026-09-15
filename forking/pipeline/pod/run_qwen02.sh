#!/bin/bash
# usage: run_qwen02.sh <tag> <extra args...>   e.g.  run_qwen02.sh smoke --max-branches 2 --S 5 --S0 5
export HF_HOME=/root/hf HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
source /root/venv/bin/activate; cd /root
TAG=$1; shift
nohup python /root/fpa_vllm.py --questions /root/row080.json --out-dir /root/out/$TAG "$@" > /root/out/$TAG.log 2>&1 &
echo "launched $TAG pid $!"
