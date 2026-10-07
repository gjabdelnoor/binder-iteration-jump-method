#!/bin/bash
set -euo pipefail
cd "${STUDY_ROOT:?}"
export UV_CACHE_DIR="$HOME/.cache/uv-tnf-study"
export HF_HOME="$HOME/.cache/huggingface-tnf-study"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
./uv python install 3.12
./uv venv --python 3.12 .venv
./uv pip install --python .venv/bin/python 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu128
./uv pip install --python .venv/bin/python 'esm==3.4.1.post1' pandas pyarrow scikit-learn matplotlib nbformat nbclient ipykernel biopython
.venv/bin/python -m pip --version 2>/dev/null || true
./uv pip freeze --python .venv/bin/python > results/environment.lock.txt
.venv/bin/python - <<'PY'
import torch, json, importlib.metadata
from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer
from huggingface_hub import HfApi
print('ENVIRONMENT', json.dumps({'torch':torch.__version__,'esm':importlib.metadata.version('esm'),'gpu':torch.cuda.get_device_name(0),'cuda_available':torch.cuda.is_available()}),flush=True)
info=HfApi().model_info('biohub/ESMC-300M')
open('results/model_revision.txt','w').write(info.sha+'\n')
model=EsmcForMaskedLM.from_pretrained('biohub/ESMC-300M', revision=info.sha, device='cuda', attn_implementation='sdpa')
tokenizer=EsmcTokenizer()
x=tokenizer(['MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ'],return_tensors='pt',padding=True)
x={k:v.to('cuda') for k,v in x.items()}
with torch.inference_mode(): out=model(**x)
print('OUTPUT',type(out),getattr(out,'keys',lambda:[])(),flush=True)
print('MODEL',model,flush=True)
print('TOKENIZER',tokenizer.get_vocab(),flush=True)
print('PREFLIGHT_SUCCESS',flush=True)
PY
