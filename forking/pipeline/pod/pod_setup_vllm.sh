#!/bin/bash
# QWEN-02 pod setup: vLLM in a FRESH venv (vLLM docs: the wheel bundles its own torch and
# "it is recommended to install vLLM with a fresh new environment"). Image torch is untouched.
# Verdict lines at every step; ends with a 5-token generate through vLLM so a broken stack
# fails here.
set -x
export HF_HOME=/root/hf HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p /root/hf /root/out
cat >> /root/.bashrc <<'ENVEOF'
export HF_HOME=/root/hf
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
ENVEOF
cd /root
git clone https://github.com/IamAGP/forking-fast.git 2>&1 | tail -1
python3 -m venv /root/venv && source /root/venv/bin/activate
pip install --no-cache-dir -q --upgrade pip
# NO --extra-index-url: vllm 0.29.0 on PyPI pins torch 2.13.0 (CUDA 13.0); an extra PyTorch index
# pulled torchaudio 2.11+cu128 alongside it on 2026-09-15 and transformers refused to import.
pip install --no-cache-dir "vllm==0.29.0" 2>&1 | tail -2
# vllm 0.29.0 requires torchaudio; the plain PyPI 2.11.0 wheel matches its torch 2.13.0. Verified 2026-09-15.
python -c "import torchaudio, vllm" || pip install --no-cache-dir --force-reinstall --no-deps "torchaudio==2.11.0"
echo "=========== VERSION CHECK ==========="
python - <<'PY'
import torch, transformers, vllm
print("torch       ", torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("transformers", transformers.__version__)
print("vllm        ", vllm.__version__)
PY
echo "=========== DOWNLOAD + vLLM 5-TOKEN GENERATE TEST ==========="
# MUST be a real file with a __main__ guard: vLLM spawns its EngineCore and re-imports the
# main module; a stdin heredoc fails with "No such file or directory: '/root/<stdin>'"
# (2026-09-15, SCREEN-01 setup) and a bare top-level LLM() fails with the "bootstrapping
# phase" error (2026-09-15, QWEN-02 setup).
cat > /root/gen_test.py <<'PY'
import time
def main():
    t0=time.time()
    from vllm import LLM, SamplingParams
    llm = LLM(model='Qwen/Qwen3.5-9B', dtype='bfloat16', gpu_memory_utilization=0.9, max_model_len=2048, enable_prefix_caching=True)
    o = llm.chat([{'role':'user','content':'Say hi.'}], SamplingParams(temperature=0, max_tokens=5), chat_template_kwargs={'enable_thinking': False}, use_tqdm=False)
    print('GENERATE OK:', repr(o[0].outputs[0].text), f'{time.time()-t0:.0f}s incl download', flush=True)
if __name__ == '__main__':
    main()
PY
python /root/gen_test.py
echo "=========== SETUP DONE ==========="
