# apt update -y && apt upgrade -y && apt install -y tmux

## Using `uv` package manager
# uv init --bare --python 3.12
# uv sync --python 3.12
# source .venv/bin/activate

## If decide not to use the available pyproject.toml
# export UV_CACHE_DIR = ~/users/patrick/
# uv add install datasets einops hf_transfer huggingface_hub ipykernel ipywidgets jupyter numpy pillow pycocoevalcap torch torchvision transformers wandb lmms_eval termcolor accelerate
## OR
uv pip install datasets einops hf_transfer huggingface_hub ipykernel ipywidgets jupyter numpy pillow pycocoevalcap torch torchvision transformers wandb termcolor accelerate
