#!/bin/bash
# PARABOLA-01 relaunch: weights already in /root/models. Fix: --answer=-4a (argparse read "-4a" as a flag).
ts(){ date +%H:%M:%S; }
echo "=========== $(ts) RUN2 ==========="
python3 /root/smoke_q38.py --model /root/models/Qwen3.8-27B --out /root/out/parabola \
  --max-new-tokens 4096 --answer=-4a --question "$(cat /root/prompt.txt)"
echo "exit=$?"
echo "=========== $(ts) ALL DONE2 ==========="
