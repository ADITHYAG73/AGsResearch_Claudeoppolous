#!/bin/bash
# QWEN-01 pod setup. Image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404 (the tag the NLA
# runs used; torch 2.8.0+cu128 preinstalled). Qwen3.5 needs a transformers that knows
# `qwen3_5` (5.17.0 works on the laptop) and the two fast kernels for its linear-attention
# layers; without them transformers falls back to reference PyTorch, "correct but much slower".
# Every step prints a verdict line; the last block runs a 5-token generate so a broken stack
# fails HERE, not an hour into the run.
set -x
export HF_HOME=/root/hf HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p /root/hf /root/out
cat >> /root/.bashrc <<'ENVEOF'
export HF_HOME=/root/hf
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
ENVEOF
cd /root
git clone https://github.com/IamAGP/forking-fast.git 2>&1 | tail -2
pip install --break-system-packages --no-cache-dir "transformers==5.17.0" accelerate numpy pandas pyarrow huggingface_hub 2>&1 | tail -2
pip install --break-system-packages --no-cache-dir flash-linear-attention causal-conv1d 2>&1 | tail -3
echo "=========== VERSION CHECK ==========="
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
echo "=========== DOWNLOAD + 5-TOKEN GENERATE TEST ==========="
python3 - <<'PY'
import torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
m="Qwen/Qwen3.5-9B"; t0=time.time()
tok=AutoTokenizer.from_pretrained(m); model=AutoModelForCausalLM.from_pretrained(m, dtype=torch.bfloat16).cuda().eval()
ids=tok.apply_chat_template([{"role":"user","content":"Say hi."}], add_generation_prompt=True, tokenize=True, enable_thinking=False)
ids=ids["input_ids"] if hasattr(ids,"input_ids") else ids
out=model.generate(torch.tensor([ids]).cuda(), max_new_tokens=5, do_sample=False)
print("GENERATE OK:", repr(tok.decode(out[0][len(ids):])), f"{time.time()-t0:.0f}s incl download", torch.cuda.max_memory_allocated()/1e9, "GB")
PY
echo "=========== SETUP DONE ==========="
