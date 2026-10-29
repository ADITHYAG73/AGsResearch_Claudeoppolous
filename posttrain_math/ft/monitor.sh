#!/bin/bash
# FT-01 unattended monitor. Polls every pod every 2 min and EXITS on the first event that needs
# Claude (the exit is what wakes Claude up). Events already handled are listed in $STATE/handled
# and never fire twice. Bounded: gives up (exit "TIMEOUT") after 8 h.
#   ALERT  : Traceback / OOM / STOP / TIMEOUT in a log; log silent > 25 min while its job runs;
#            GPU at 0% for > 10 min while a job should be using it; pod unreachable 3 polls running
#   EVENT  : checkpoint s050/s100 uploaded; training done; an eval finished (ALL DONE)
# Usage: monitor.sh <state_dir>    (pods are listed below)
STATE="${1:?state dir}"; mkdir -p "$STATE"; touch "$STATE/handled"
PSSH=/private/tmp/claude-501/-Users-adithyagiridharan-Desktop-PythonProjects-AGs-Research/5d62c3d5-0568-4062-a81c-214d5eefcc27/scratchpad/pssh.sh
# name host port log    (log = the file whose growth means "job alive")
PODS="${PODS:-base 157.157.221.29 13370 /root/out/eval.log
s050 154.54.102.56 14244 /root/out/eval.log
s100 216.81.248.113 11230 /root/out/eval.log}"
fire() { grep -qx "$1" "$STATE/handled" && return 0; echo "$(date -u +%H:%M:%S) $2"; echo "$1" >> "$STATE/handled"; exit 0; }
# counters live in files: macOS ships bash 3.2, which has no associative arrays
cnt() { cat "$STATE/$1" 2>/dev/null || echo 0; }
setc() { echo "$2" > "$STATE/$1"; }
for poll in $(seq 1 ${POLLS:-240}); do
  while read -r NAME H P LOG; do
    [ -z "$NAME" ] && continue
    OUT=$($PSSH "$H" "$P" "
      if [ -f $LOG ]; then now=\$(date +%s); m=\$(stat -c %Y $LOG); echo AGE \$((now-m)); else echo AGE 0; echo SETUP; fi
      echo GPU \$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ')
      grep -c -E 'Traceback|OutOfMemory|^STOP|TIMEOUT|Killed' $LOG 2>/dev/null | sed 's/^/ERR /'
      grep -o -E 'CHECKPOINT s[0-9]+|PIPELINE DONE|ALL DONE [^ ]+|waiting for posttrain|MODEL READY' $LOG 2>/dev/null | sort -u | sed 's/^/EV /'
      tail -1 $LOG 2>/dev/null | cut -c1-160 | sed 's/^/LAST /'" </dev/null 2>/dev/null)   # </dev/null: ssh must not eat the pod list on stdin
    if [ -z "$OUT" ]; then
      setc unreach_$NAME $(( $(cnt unreach_$NAME) + 1 ))
      [ "$(cnt unreach_$NAME)" -ge 3 ] && fire "unreach-$NAME-$poll" "ALERT $NAME unreachable for 3 polls ($H:$P)"
      continue
    fi
    setc unreach_$NAME 0
    echo "$(date -u +%H:%M:%S) $NAME $(echo "$OUT" | tr '\n' ' ' | cut -c1-260)" > "$STATE/last_$NAME"
    ERR=$(echo "$OUT" | awk '/^ERR/{print $2}'); AGE=$(echo "$OUT" | awk '/^AGE/{print $2}')
    UTIL=$(echo "$OUT" | awk '/^GPU/{split($2,a,","); print a[1]}')
    [ "${ERR:-0}" -gt 0 ] && fire "err-$NAME-$ERR" "ALERT $NAME: error in log: $(echo "$OUT" | grep '^LAST')"
    for ev in $(echo "$OUT" | awk '/^EV/{ $1=""; gsub(/ /,"_"); print }'); do
      case "$ev" in *waiting_for_posttrain*|*MODEL_READY*) continue;; esac
      fire "ev-$NAME-$ev" "EVENT $NAME: $ev"
    done
    # waiting = still polling S3 for its checkpoint (logged 'waiting for', not yet 'MODEL READY')
    WAITING=0; echo "$OUT" | grep -q 'waiting for posttrain' && ! echo "$OUT" | grep -q 'MODEL READY' && WAITING=1
    echo "$OUT" | grep -q '^SETUP' && WAITING=1   # log not created yet: pod still installing
    DONE=$(echo "$OUT" | grep -c -E 'EV (ALL DONE|PIPELINE DONE)')
    if [ "$DONE" -eq 0 ] && [ "$WAITING" -eq 0 ]; then
      [ "${AGE:-0}" -gt 1500 ] && fire "stale-$NAME-$poll" "ALERT $NAME: log silent for $((AGE/60)) min"
      if [ "${UTIL:-0}" = "0" ]; then setc idle_$NAME $(( $(cnt idle_$NAME) + 1 )); else setc idle_$NAME 0; fi
      [ "$(cnt idle_$NAME)" -ge 5 ] && fire "idle-$NAME-$poll" "ALERT $NAME: GPU 0% for 10 min while its job should run"
    fi
  done <<< "$PODS"
  sleep 120
done
echo "$(date -u +%H:%M:%S) TIMEOUT monitor ran 8 h with no event"
