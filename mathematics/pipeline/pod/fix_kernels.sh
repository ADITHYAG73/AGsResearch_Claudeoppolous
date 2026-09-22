#!/bin/bash
# Kernel repair, run in parallel with the weight download. Recipe recorded in
# forking/experiments.md (QWEN-01, 16:24): causal-conv1d's setup.py needs torch importable at
# build time and CUDA_HOME -> build with CUDA_HOME set and --no-build-isolation. fla is pure
# Python/Triton and installs on its own; it only failed because it shared a pip command.
ts(){ date +%H:%M:%S; }
echo "$(ts) fla"; pip install --break-system-packages --no-cache-dir flash-linear-attention 2>&1 | tail -1
echo "$(ts) build deps"; pip install --break-system-packages --no-cache-dir ninja packaging wheel setuptools 2>&1 | tail -1
ls -d /usr/local/cuda* ; /usr/local/cuda/bin/nvcc --version | tail -2
echo "$(ts) causal-conv1d (compiles; MAX_JOBS=16)"
CUDA_HOME=/usr/local/cuda MAX_JOBS=16 pip install --break-system-packages --no-cache-dir --no-build-isolation causal-conv1d 2>&1 | tail -3
echo "$(ts) verify"
python3 -c "
for m in ('fla','causal_conv1d'):
    try: mod=__import__(m); print(m,'OK',getattr(mod,'__version__',''))
    except Exception as e: print(m,'MISSING',e)"
echo "$(ts) KERNEL FIX DONE"
