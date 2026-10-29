#!/bin/bash
# vLLM in a fresh venv, following forking/pipeline/pod/pod_setup_vllm.sh (proven 2026-09-15).
# Differences, both forced by the sandbox: the venv lives in /opt (model-written code runs as
# user "nobody", which cannot enter /root), and results stay in /root/out where it cannot reach.
# HOST REQUIREMENT: CUDA 13.0 (vllm 0.29.0 pins torch 2.13.0+cu130).
set -x
export HF_HOME=/opt/hf HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p /opt/hf /root/out
python3 -m venv /opt/venv && source /opt/venv/bin/activate
pip install --no-cache-dir -q --upgrade pip
pip install --no-cache-dir "vllm==0.29.0" 2>&1 | tail -2
python -c "import torchaudio, vllm" || pip install --no-cache-dir --force-reinstall --no-deps "torchaudio==2.11.0"
pip install --no-cache-dir -q numpy sympy mpmath 2>&1 | tail -1
chmod -R a+rX /opt/venv
echo "=========== VERSION CHECK ==========="
python - <<'PY'
import torch, transformers, vllm
print("torch       ", torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
print("vllm        ", vllm.__version__)
PY
echo "=========== SANDBOX SELF-TEST (runs as nobody) ==========="
cd /root/pm && python - <<'PY'
import sys; sys.argv=["x"]
import run_eval as R
r = R.run_code("import os, math, numpy, sympy, mpmath; print(os.geteuid(), math.sin(math.radians(37)))")
print("sandbox:", r["rc"], repr(r["shown"]))
assert r["rc"] == 0 and r["shown"].startswith("65534 0.6018"), "SANDBOX SELF-TEST FAILED"
r2 = R.run_code("open('/root/out/x','w').write('1')")
print("can the sandbox write to results? rc", r2["rc"], "(must be non-zero)")
assert r2["rc"] != 0, "SANDBOX CAN TOUCH RESULTS"
print("SANDBOX OK")
PY
echo "=========== SETUP DONE ==========="
