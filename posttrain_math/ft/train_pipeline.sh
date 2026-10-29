#!/bin/bash
# Training pod pipeline. Each stage must print its DONE/PASSED line or the pipeline stops.
# No `set -x` (credentials are in the environment).
set -u
source /root/.s3env
export HF_HOME=/root/hf HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONUNBUFFERED=1
# allocator setting only (no effect on the maths): reduces fragmentation, which caused the A100 OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/pm
D=/root/ft01; mkdir -p $D
# ssh sessions do not inherit the image's Docker ENV (PIP_BREAK_SYSTEM_PACKAGES), so be explicit
pip install -q --break-system-packages boto3 2>&1 | tail -1
python - <<'PY'
import torch, transformers; print("VERSIONS torch", torch.__version__, "transformers", transformers.__version__, torch.cuda.get_device_name(0))
PY

if [ "${SKIP_PREP:-0}" = "1" ] && [ -s $D/train.jsonl ]; then
  echo "== STAGE 1 skipped: reusing $D/train.jsonl ($(wc -l < $D/train.jsonl) rows, already in S3)"
else
  echo "== STAGE 1 data prep"; python ft/prep_data.py --questions questions_v0.jsonl --out $D || exit 1
  python ft/s3_sync.py $D posttrain_math/ft01/data
fi
echo "== STAGE 2 save-path test"; python ft/train.py --data $D/train.jsonl --out $D --save-test 2>&1 | tee $D/savetest.log
grep -q "SAVE TEST PASSED" $D/savetest.log || { echo "STOP: save test failed"; exit 1; }
rm -rf $D/s000_savetest
echo "== STAGE 3 probe"; python ft/train.py --data $D/train.jsonl --out $D --probe 2>&1 | tee $D/probe.log
grep -q "^.*PROBE peak" $D/probe.log || { echo "STOP: probe failed"; exit 1; }
K=$(python - <<'PY'
import re, json
t = open("/root/ft01/probe.log").read()
m = re.search(r"PROBE peak activations\+grads ([\d.]+) GiB \+ AdamW state ([\d.]+) GiB; card ([\d.]+) GiB; (\d+) tok/s", t)
peak, adam, card, tps = float(m[1]), float(m[2]), float(m[3]), int(m[4])
rows = [json.loads(l) for l in open("/root/ft01/train.jsonl")]
mean_len = sum(len(r["input_ids"]) for r in rows) / len(rows)
need = peak + adam
# Rule fixed in the journal before launch: K = min(5000, floor(90 min x tok/s / mean length)).
k = min(len(rows), 5000, int(90 * 60 * tps / mean_len))
print(f"{k} {need:.1f} {card:.1f} {tps} {mean_len:.0f}")
PY
)
read KK NEED CARD TPS ML <<< "$K"
echo "PROBE RESULT: memory need ${NEED} GiB of ${CARD}; ${TPS} tok/s; mean row ${ML} tokens -> K=${KK}"
python -c "import sys; sys.exit(0 if $NEED < $CARD - 3 else 1)" || { echo "STOP: will not fit in memory (need $NEED GiB)"; exit 1; }
echo "== STAGE 4 train K=$KK"
python ft/train.py --data $D/train.jsonl --out $D --k $KK --s3-prefix posttrain_math/ft01/checkpoints 2>&1 | tee $D/train.log
mkdir -p $D/logs && cp $D/*.log $D/loss.jsonl $D/data_report.json $D/logs/ 2>/dev/null
python ft/s3_sync.py $D/logs posttrain_math/ft01/train_logs || true
grep -q "TRAIN DONE" $D/train.log && echo "PIPELINE DONE" || echo "STOP: training did not finish"
