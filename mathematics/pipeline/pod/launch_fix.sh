#!/bin/bash
nohup bash /root/fix_kernels.sh > /root/out/fix_kernels.log 2>&1 < /dev/null &
echo "launched kernel fix pid $!"
