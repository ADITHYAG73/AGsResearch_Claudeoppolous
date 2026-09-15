#!/bin/bash
# Chain: guarded generate test -> full stride-4 run on row 41. No idle gap between them.
export HF_HOME=/root/hf HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
source /root/venv/bin/activate; cd /root
nohup bash -c 'python /root/gen_test.py > /root/gen_test.log 2>&1 && python /root/fpa_vllm.py --questions /root/row041.json --out-dir /root/out/main --stride 4 --S 50 --S0 50 --only-forks --chunk 8 --base-max-tokens 1500 --cont-max-tokens 1500 --max-model-len 4096 > /root/out/main.log 2>&1' > /root/chain.log 2>&1 &
echo "launched chain pid $!"
