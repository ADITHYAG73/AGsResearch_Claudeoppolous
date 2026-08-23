#!/bin/bash
# ==========================================================================
# DO NOT REMOVE sglang[all] "because the AR does not need a server".
#
# It is load-bearing for a reason nothing in the file says. Install order:
#   1. pip install transformers ...      -> transformers 5.15.1
#   2. pip install torch==2.9.1          -> torchvision is now 0.23.0, built for
#                                           torch 2.8.0. pip warns and continues.
#   3. pip install sglang[all]==0.5.10.post1
#          -> transformers 5.3.0  (downgrade, which is why 5.3.0 is what runs)
#          -> torchvision 0.24.1  (UPGRADE - this is what repairs step 2)
#
# 2026-08-23: dropping sglang for an AR-only run left torchvision 0.23.0 against
# torch 2.9.1. transformers imports torchvision; loading a C++ extension built
# for a different torch ABI aborts the interpreter with std::bad_alloc and NO
# Python traceback - `import transformers.modeling_layers` alone was enough.
# Cost ~$0.25 and an hour. Verified against runs/2026-08-22_pos10/setup.log.
#
# Image used by the runs that worked: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# (experiments.md records it truncated; python3.12 in av_server.log identifies 2404).
# ==========================================================================
set -x
export HF_HOME=/root/hf HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p /root/hf
cat >> /root/.bashrc <<'ENVEOF'
export HF_HOME=/root/hf
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
ENVEOF
cd /root
git clone https://github.com/ADITHYAG73/natural_language_autoencoders.git nla 2>&1 | tail -3
pip install --break-system-packages --no-cache-dir transformers safetensors httpx orjson pyyaml numpy pyarrow huggingface_hub 2>&1 | tail -3
pip install --break-system-packages --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3
pip install --break-system-packages --no-cache-dir "sglang[all]==0.5.10.post1" 2>&1 | tail -5
echo "=========== VERSION CHECK ==========="
python3 -c "import torch, transformers, sglang; print('torch       ', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability()); print('transformers', transformers.__version__); print('sglang      ', sglang.__version__)"
echo "=========== SETUP DONE ==========="
