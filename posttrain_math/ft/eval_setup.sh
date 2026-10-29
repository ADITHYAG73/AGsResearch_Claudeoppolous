#!/bin/bash
# Eval pod setup = the BASE-01 setup (pod/pod_setup.sh, unchanged) + three additions:
#   1. transformers pinned to 5.18.0, the version BASE-01 actually ran (env_Qwen3.5-4B.txt),
#      so the software matches the baseline exactly;
#   2. boto3 for S3; 3. Google's IFEval scorer deps (nltk 3.8.1: instructions_util loads the
#      punkt *pickle*, which newer nltk no longer reads).
set -u
bash /root/pm/pod/pod_setup.sh 2>&1 | grep -v -i "token"
source /opt/venv/bin/activate
pip install --no-cache-dir -q "transformers==5.18.0" boto3 absl-py langdetect "nltk==3.8.1" immutabledict 2>&1 | tail -2
python -c "import nltk; nltk.download('punkt', quiet=True)"
chmod -R a+rX /opt/venv
python - <<'PY'
import torch, transformers, vllm, boto3, nltk, langdetect
print("VERSIONS torch", torch.__version__, "| transformers", transformers.__version__, "| vllm", vllm.__version__,
      "| gpu", torch.cuda.get_device_name(0))
assert transformers.__version__ == "5.18.0" and vllm.__version__ == "0.29.0"
import sys; sys.path.insert(0, "/root/pm/ft")
from instruction_following_eval import evaluation_lib, instructions_util
print("IFEval scorer import OK; punkt sentences:", len(instructions_util.split_into_sentences("One. Two.")))
PY
echo "EVAL SETUP DONE"
