#!/bin/bash
# Copy the plain model files local -> global volume, one file at a time with a timestamped
# line each (progress + measured write speed), then verify sizes and write the DONE marker
# LAST. No DONE marker = incomplete copy = next session re-downloads. Run detached.
ts(){ date +%H:%M:%S; }
LOCAL=/root/models/Qwen3.8-27B
if [ -d /workspace-global ]; then GV=/workspace-global; elif mount | grep -q " /workspace "; then GV=/workspace; else echo "NO VOLUME MOUNTED"; exit 1; fi
GVM=$GV/models/Qwen3.8-27B; mkdir -p "$GVM"; rm -f "$GVM/DONE"
t0=$(date +%s); bad=0
for f in $LOCAL/*; do [ -f "$f" ] || continue; b=$(basename "$f"); s0=$(date +%s)
  cp "$f" "$GVM/$b"; a=$(stat -c %s "$f"); c=$(stat -c %s "$GVM/$b" 2>/dev/null || echo 0)
  if [ "$a" = "$c" ]; then echo "$(ts) saved $b $((a/1048576)) MB in $(( $(date +%s)-s0 )) s"; else echo "$(ts) SIZE MISMATCH $b $a vs $c"; bad=1; fi
done
t1=$(date +%s); sz=$(du -sm $LOCAL | cut -f1)
if [ $bad = 0 ]; then echo "model=Qwen/Qwen3.8-27B files=$(ls $LOCAL | wc -l) MB=$sz saved=$(date -u +%FT%TZ)" > "$GVM/DONE"; echo "SAVE OK ${sz} MB in $((t1-t0)) s = $((sz/(t1-t0+1))) MB/s"; else echo "SAVE FAILED (no DONE marker written)"; fi
