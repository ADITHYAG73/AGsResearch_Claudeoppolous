#!/bin/bash
# Standalone launcher (never inline a backgrounded process in an ssh command — 2026-08-27 hang).
mkdir -p /root/out
nohup bash /root/pod_setup_q38.sh > /root/out/setup.log 2>&1 < /dev/null &
echo "launched setup pid $!"
