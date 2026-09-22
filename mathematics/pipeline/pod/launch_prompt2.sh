#!/bin/bash
nohup bash /root/run_prompt2.sh > /root/out/prompt2.log 2>&1 < /dev/null &
echo "launched pid $!"
