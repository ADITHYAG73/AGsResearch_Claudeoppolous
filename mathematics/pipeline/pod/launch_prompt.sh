#!/bin/bash
mkdir -p /root/out
nohup bash /root/run_prompt.sh > /root/out/prompt.log 2>&1 < /dev/null &
echo "launched pid $!"
