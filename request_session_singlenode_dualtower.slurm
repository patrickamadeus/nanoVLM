#!/bin/bash
#SBATCH --job-name=dualtower_train
#SBATCH --output=logs/train_nanoVLM/%j.out
#SBATCH --error=logs/train_nanoVLM/%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=44
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs/train_nanoVLM

source .venv/bin/activate

export TOKENIZERS_PARALLELISM=false

bash ./train.sh
