#!/bin/bash
set -e

read -rp "Enter conda env name [prompt]: " ENV_NAME
ENV_NAME="${ENV_NAME:-prompt}"
PYTHON_VER="3.10"

conda create -n "$ENV_NAME" python="$PYTHON_VER" -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# === Stage 1 ===
pip install --upgrade pip
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# === Stage 2 ===
pip install transformers==4.55.4 trl==0.22.2  peft==0.17.1  accelerate==1.10.1  datasets==4.0.0  bitsandbytes==0.45.4 safetensors==0.5.3 tokenizers==0.21.4 huggingface-hub==0.34.0 hf-transfer==0.1.9 sentencepiece==0.2.0

pip install ninja==1.13.0
pip install flash-attn==2.8.3 --no-build-isolation
pip install xformers==0.0.32.post1

pip install vllm==0.11.0

pip install wandb==0.18.5  matplotlib==3.9.0 seaborn==0.13.2 plotly==6.0.1 pandas==2.3.2  numpy==1.26.4  scipy==1.15.2 scikit-learn==1.6.1 tqdm==4.66.5

pip install alpaca-eval==0.6.6 evaluate==0.4.3 rouge==1.0.1 rouge-score==0.1.2 nltk==3.9.1 fschat==0.2.36

pip install openai==2.6.1 captum==0.8.0

# IFEval scorer (testing/ifeval) needs absl-py / immutabledict / langdetect;
# rapidfuzz enables the optional fuzzy-match path in testing/test.py.
pip install absl-py==2.1.0 immutabledict==4.2.0 langdetect==1.0.9 rapidfuzz==3.10.1

echo "=== Setup done ==="
python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, device_count={torch.cuda.device_count()}')"
python -c "import transformers, trl, peft, bitsandbytes; print(f'transformers={transformers.__version__}, trl={trl.__version__}, peft={peft.__version__}, bnb={bitsandbytes.__version__}')"