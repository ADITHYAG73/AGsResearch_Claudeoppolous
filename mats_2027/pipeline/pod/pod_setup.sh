#!/bin/bash
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
