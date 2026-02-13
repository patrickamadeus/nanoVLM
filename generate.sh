#!/usr/bin/env bash
set -euo pipefail

: "${WANDB_API_KEY:?Set WANDB_API_KEY before running generate.sh}"
: "${HF_TOKEN:?Set HF_TOKEN before running generate.sh}"
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python generate.py --mode dualtower --checkpoint patrickamadeus/dt8k1img-ref-step-1200 --image image.png --prompt "How many dogs are there in the image?"

# python generate.py --mode dualtower --checkpoint patrickamadeus/dt2k1img-ref-step-5000 --image image.png --prompt "Could you explain what is the animal doing?"
# python generate.py --mode nanovlm --checkpoint patrickamadeus/nanovlm-step-2000 --image image.png --prompt "Could you explain what is the animal doing?"
