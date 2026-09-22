#!/bin/bash
# MATH-SMOKE-01 pod setup. Same stack as QWEN-01 (forking/pipeline/pod):
#   image  runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404   (torch 2.8.0+cu128)  HOST CUDA 12.8
#   transformers 5.17.0 (knows qwen3_5); fla + causal-conv1d for the 48 DeltaNet layers —
#   official transformers doc: without them the model "silently falls back to slower and more
#   memory hungry PyTorch ops".
# STORAGE. Global Volume (beta, docs.runpod.io/pods/storage/types): object-storage backed,
# "no atomic rename, no file locking, no permission bits", built for write-once read-often.
# The HF cache layout uses symlinks + .incomplete renames, so it NEVER touches the volume:
#   - model lives as PLAIN FILES in $LOCAL (container disk) and that is what gets loaded
#   - first session: download HF -> $LOCAL, then copy $LOCAL -> $GV/models/... + DONE marker
#   - later sessions: DONE marker present -> copy $GV -> $LOCAL, no HF download
# Verdict + timestamp at every step; copy speeds are printed because nobody has published them.
set -x
ts(){ date +%H:%M:%S; }
MODEL=Qwen/Qwen3.8-27B
LOCAL=/root/models/Qwen3.8-27B
if   [ -d /workspace-global ]; then GV=/workspace-global
elif mount | grep -q " /workspace "; then GV=/workspace
else GV=""; fi
echo "$(ts) global/persistent volume: ${GV:-NONE}"; mount | grep -i workspace; df -hT / ${GV:-/root} | cat
GVM=$GV/models/Qwen3.8-27B
export HF_HOME=/root/hf HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p /root/hf /root/out $LOCAL
cat >> /root/.bashrc <<'ENVEOF'
export HF_HOME=/root/hf
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
ENVEOF
pip install --break-system-packages --no-cache-dir "transformers==5.17.0" accelerate numpy huggingface_hub 2>&1 | tail -2
# SEPARATE commands: on 2026-09-19 a failed causal-conv1d build took fla down with it.
pip install --break-system-packages --no-cache-dir flash-linear-attention 2>&1 | tail -1
# causal-conv1d's setup.py needs torch importable at build time and CUDA_HOME (recorded in
# forking/experiments.md QWEN-01 16:24; re-learned the hard way 2026-09-19). With both set it
# finds a prebuilt wheel in ~10 s.
pip install --break-system-packages --no-cache-dir ninja packaging wheel setuptools 2>&1 | tail -1
CUDA_HOME=/usr/local/cuda MAX_JOBS=16 pip install --break-system-packages --no-cache-dir --no-build-isolation causal-conv1d 2>&1 | tail -2
echo "=========== $(ts) VERSION CHECK ==========="
python3 - <<'PY'
import torch, transformers
print("torch       ", torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
ok=True
for m in ("fla","causal_conv1d"):
    try: __import__(m); print(m, "OK")
    except Exception as e: print(m, "MISSING:", e); ok=False
print("KERNELS", "OK" if ok else "FALLBACK (slow)")
PY
echo "=========== $(ts) WEIGHTS ==========="
if [ -n "$GV" ] && [ -f "$GVM/DONE" ]; then
  echo "$(ts) found $GVM/DONE -> copying volume -> local (no HF download)"; cat "$GVM/DONE"
  t0=$(date +%s)
  for f in "$GVM"/*; do b=$(basename "$f"); [ "$b" = DONE ] && continue; cp "$f" "$LOCAL/$b" && echo "$(ts) restored $b"; done
  t1=$(date +%s); sz=$(du -sm $LOCAL | cut -f1); echo "RESTORE OK ${sz} MB in $((t1-t0)) s = $((sz/(t1-t0+1))) MB/s"
else
  python3 - <<PY
import time; from huggingface_hub import snapshot_download
t0=time.time()
p=snapshot_download("$MODEL", local_dir="$LOCAL", allow_patterns=["*.json","*.txt","*.jinja","*.safetensors"])
print("DOWNLOAD OK", p, f"{time.time()-t0:.0f}s")
PY
  rm -rf $LOCAL/.cache
fi
ls $LOCAL | head -40; du -sh $LOCAL
n=$(ls $LOCAL/*.safetensors 2>/dev/null | wc -l); echo "SHARDS $n (expect 18)"
echo "=========== $(ts) SETUP DONE ==========="
