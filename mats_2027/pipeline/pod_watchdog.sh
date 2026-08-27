#!/bin/bash
# GPU idle watchdog. Run this ALONGSIDE every pod session, in the background.
#
# WHY IT EXISTS. 2026-08-27: an orchestration script hung on an ssh line that backgrounded
# the sglang server; ssh never returned because the server inherited its file descriptors.
# The script never reached its own health check. With no timeout on that line it would have
# hung INDEFINITELY at $0.44/hr. Caught only because AG happened to watch the RunPod console.
# Third idle-GPU scripting bug in the project. "Be more careful" is not a mechanism; this is.
#
# Usage:  bash pod_watchdog.sh <host> <port> [idle_minutes_before_alarm]
set -u
H=${1:?host}; P=${2:?port}; LIMIT_MIN=${3:-4}
K=$HOME/.ssh/id_ed25519_runpod_new
SSHO="-i $K -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 -o BatchMode=yes"
POLL=30
# PORTABLE TIMEOUT. macOS ships no GNU `timeout` - the first version of this script used it
# unconditionally, so EVERY poll failed instantly with "command not found", returned empty,
# and was reported as UNREACHABLE. 61 consecutive false alarms while the pod was perfectly
# reachable. A watchdog that cries wolf is worse than none. Detect what exists; if nothing,
# fall back to ssh keepalives, which bound a dead connection without any external binary.
if command -v timeout  >/dev/null 2>&1; then TMO="timeout 20"
elif command -v gtimeout >/dev/null 2>&1; then TMO="gtimeout 20"
else TMO=""; SSHO="$SSHO -o ServerAliveInterval=5 -o ServerAliveCountMax=3"; fi
NEEDED=$(( LIMIT_MIN * 60 / POLL ))
idle=0; unreachable=0
echo "[watchdog] $H:$P  alarm after ${LIMIT_MIN} min idle (${NEEDED} polls of ${POLL}s)"
echo "[watchdog] timeout mechanism: ${TMO:-ssh keepalives (no timeout binary found)}"
# SELF-TEST: prove one poll returns a real reading before trusting the loop. The first
# version of this script was never verified to work even once, and it never did.
PROBE=$($TMO ssh $SSHO -p "$P" root@"$H" "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader" 2>/dev/null)
if [ -z "$PROBE" ]; then
  echo "[watchdog] *** SELF-TEST FAILED - cannot read the GPU. NOT WATCHING. Fix before trusting this. ***"
  exit 1
fi
echo "[watchdog] self-test OK, first reading: ${PROBE}"
while true; do
  # timeout is the point: ssh's ConnectTimeout does NOT bound command execution.
  OUT=$($TMO ssh $SSHO -p "$P" root@"$H" \
        "nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" 2>/dev/null)
  if [ -z "$OUT" ]; then
    unreachable=$((unreachable+1))
    echo "[watchdog $(date -u +%H:%M:%S)] pod UNREACHABLE (${unreachable})"
    [ "$unreachable" -ge 6 ] && echo "[watchdog] *** UNREACHABLE FOR 3 MIN - is the pod still billing? CHECK. ***"
    sleep $POLL; continue
  fi
  unreachable=0
  UTIL=$(echo "$OUT" | awk -F',' '{gsub(/[^0-9]/,"",$1); print $1}')
  MEM=$(echo "$OUT" | awk -F',' '{gsub(/[^0-9]/,"",$2); print $2}')
  if [ "${UTIL:-0}" -eq 0 ]; then
    idle=$((idle+1))
    MINS=$(( idle * POLL / 60 ))
    if [ "$idle" -ge "$NEEDED" ]; then
      COST=$(awk -v m="$MINS" 'BEGIN{printf "%.3f", m*0.44/60}')
      echo "[watchdog $(date -u +%H:%M:%S)] *** GPU IDLE ${MINS} MIN (mem ${MEM} MiB) — \$${COST} burned. SOMETHING IS STUCK. ***"
    else
      echo "[watchdog $(date -u +%H:%M:%S)] idle ${MINS}m (mem ${MEM} MiB)"
    fi
  else
    [ "$idle" -gt 0 ] && echo "[watchdog $(date -u +%H:%M:%S)] working again: ${UTIL}% (was idle ${idle} polls)"
    idle=0
  fi
  sleep $POLL
done
