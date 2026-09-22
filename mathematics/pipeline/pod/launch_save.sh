#!/bin/bash
nohup bash /root/save_to_volume.sh > /root/out/save.log 2>&1 < /dev/null &
echo "launched save pid $!"
