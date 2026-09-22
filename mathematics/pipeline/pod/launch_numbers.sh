#!/bin/bash
mkdir -p /root/out
nohup bash /root/run_numbers.sh > /root/out/numbers.log 2>&1 < /dev/null &
echo "launched pid $!"
