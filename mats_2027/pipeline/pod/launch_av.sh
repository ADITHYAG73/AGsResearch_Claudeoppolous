#!/bin/bash
export HF_TOKEN=$(grep '^export HF_TOKEN=' /root/.bashrc | cut -d= -f2)
export HF_HOME=/root/hf HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
cd /root/nla
exec python3 -m sglang.launch_server --model-path kitft/nla-gemma3-12b-L32-av \
  --port 30000 --disable-radix-cache --mem-fraction-static 0.6 --trust-remote-code \
  --attention-backend triton --sampling-backend pytorch
