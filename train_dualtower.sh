#!/usr/bin/env bash
set -euo pipefail

# : "${WANDB_API_KEY:?Set WANDB_API_KEY before running train.sh}"
# : "${HF_TOKEN:?Set HF_TOKEN before running train.sh}"
# export HF_HOME="${HF_HOME:-/workspace/huggingface}"

export HF_HOME=/nfs-stor/alham.fikri/huggingface

# python train.py --checkpoint_interval 200 --dualtower
torchrun --standalone --nproc_per_node=4 --nnodes=1 train.py --config /home/alham.fikri/patrick.amadeus/nanoVLM_edd_root/nanoVLM_edd_main/configs/dualtower.nopack.yaml

# srun torchrun --nproc_per_node=$SLURM_GPUS_PER_NODE \
#     --nnodes=$SLURM_NNODES \
#     --rdzv_id=$SLURM_JOB_ID \
#     --rdzv_backend=c10d \
#     --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
#     train.py 
#     #--relevance_min_rating 1 --image_correspondence_min_rating 1 --visual_dependency_min_rating 1 --formatting_min_rating 1