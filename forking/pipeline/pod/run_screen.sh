#!/bin/bash
# usage: run_screen.sh <tag> <extra args...>
export HF_HOME=/root/hf HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
source /root/venv/bin/activate; cd /root
TAG=$1; shift
nohup python /root/screen_t0.py --questions /root/tinymmlu_100.json --out-dir /root/out/$TAG "$@" > /root/out/$TAG.log 2>&1 &
echo "launched $TAG pid $!"
