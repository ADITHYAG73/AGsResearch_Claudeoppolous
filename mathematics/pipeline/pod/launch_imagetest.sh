#!/bin/bash
mkdir -p /root/out
nohup bash /root/run_imagetest.sh > /root/out/imagetest.log 2>&1 < /dev/null &
echo "launched imagetest pid $!"
