import argparse
import torch
from PIL import Image

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

from models.dual_tower.dual_tower import DualTowerVLM
from models.vision_language_model import VisionLanguageModel
from data.processors import get_tokenizer, get_image_processor, get_image_string
from train_utils.config_loader import load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate text from an image with nanoVLM")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument(
        "--mode", type=str, choices=["nanovlm", "dualtower"], default=None,
        help="Model architecture mode. Use 'dualtower' for DualTowerVLM checkpoints."
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to a local checkpoint (directory or safetensors/pth). If omitted, we pull from HF."
    )
    parser.add_argument(
        "--hf_model", type=str, default=None,
        help="HuggingFace repo ID to download from incase --checkpoint isnt set."
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Path to input image")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt to feed the model")
    parser.add_argument("--generations", type=int, default=None,
                        help="Num. of outputs to generate")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Maximum number of tokens per output")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k for sampling.")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p for nucleus sampling.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument(
        "--right_prefill_mode",
        type=str,
        choices=["full", "non_donor_only"],
        default=None,
        help="DualTower generation prefill strategy override.",
    )
    parser.add_argument("--measure_vram", action="store_true",
                        help="Measure and display VRAM usage during model loading and generation")
    parser.add_argument("--greedy", action="store_true",
                        help="Enable greedy decoding (no sampling)")
    return parser.parse_args()


def main():
    args = parse_args()
    yaml_cfg = {}
    if args.config is not None:
        loaded = load_yaml_config(args.config)
        yaml_cfg = loaded.get("generation", loaded)

    mode = args.mode if args.mode is not None else yaml_cfg.get("mode", "nanovlm")
    checkpoint = args.checkpoint if args.checkpoint is not None else yaml_cfg.get("checkpoint")
    hf_model = args.hf_model if args.hf_model is not None else yaml_cfg.get("hf_model", "lusxvr/nanoVLM-230M-8k")
    image_path = args.image if args.image is not None else yaml_cfg.get("image", "assets/image.png")
    prompt = args.prompt if args.prompt is not None else yaml_cfg.get("prompt", "What is this?")
    generations = args.generations if args.generations is not None else int(yaml_cfg.get("generations", 5))
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else int(yaml_cfg.get("max_new_tokens", 50))
    top_k = args.top_k if args.top_k is not None else int(yaml_cfg.get("top_k", 50))
    top_p = args.top_p if args.top_p is not None else float(yaml_cfg.get("top_p", 0.9))
    temperature = args.temperature if args.temperature is not None else float(yaml_cfg.get("temperature", 0.7))
    right_prefill_mode = (
        args.right_prefill_mode
        if args.right_prefill_mode is not None
        else yaml_cfg.get("right_prefill_mode")
    )
    greedy = bool(args.greedy or yaml_cfg.get("greedy", False))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    source = checkpoint if checkpoint else hf_model
    print(f"Loading weights from: {source}")
    
    if args.measure_vram and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    if mode == "dualtower":
        model = DualTowerVLM.from_pretrained(source, device=device).to(device)
        if right_prefill_mode is not None:
            model.right_prefill_mode = right_prefill_mode
            model.cfg.right_prefill_mode = right_prefill_mode
    else:
        model = VisionLanguageModel.from_pretrained(source).to(device)
    model.eval()
    
    if args.measure_vram and torch.cuda.is_available():
        torch.cuda.synchronize()
        model_vram_bytes = torch.cuda.memory_allocated(device)
        model_vram_mb = model_vram_bytes / (1024 ** 2)
        print(f"VRAM used after loading model: {model_vram_mb:.2f} MB")

    # Get tokenizer and image processor from model config if not provided
    tokenizer = get_tokenizer(model.cfg.lm_tokenizer, model.cfg.vlm_extra_tokens, model.cfg.lm_chat_template)
    resize_to_max_side_len = False
    if hasattr(model.cfg, "resize_to_max_side_len"):
        resize_to_max_side_len = model.cfg.resize_to_max_side_len
    image_processor = get_image_processor(model.cfg.max_img_size, model.cfg.vit_img_size, resize_to_max_side_len)

    img = Image.open(image_path).convert("RGB")
    processed_image, splitted_image_ratio = image_processor(img)
    if not hasattr(tokenizer, "global_image_token") and splitted_image_ratio[0]*splitted_image_ratio[1] == len(processed_image) - 1:
        # If the tokenizer doesn't have a global image token, but the processor generated it, remove it
        processed_image = processed_image[1:]

    image_string = get_image_string(tokenizer, [splitted_image_ratio], model.cfg.mp_image_token_length)

    messages = [{"role": "user", "content": image_string + prompt}]
    encoded_prompt = tokenizer.apply_chat_template([messages], tokenize=True, add_generation_prompt=True)
    tokens = torch.tensor(encoded_prompt).to(device)
    img_t = processed_image.to(device)

    print("\nInput:\n ", prompt, "\n\nOutput:")
    for i in range(generations):
        gen = model.generate(
            tokens,
            img_t,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            greedy=greedy,
        )
        out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        
        if args.measure_vram and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_vram_bytes = torch.cuda.max_memory_allocated(device)
            peak_vram_mb = peak_vram_bytes / (1024 ** 2)
            current_vram_bytes = torch.cuda.memory_allocated(device)
            current_vram_mb = current_vram_bytes / (1024 ** 2)
            print(f"  >> Generation {i+1}: {out}")
            print(f"     VRAM - Peak: {peak_vram_mb:.2f} MB, Current: {current_vram_mb:.2f} MB")
        else:
            print(f"  >> Generation {i+1}: {out}")

if __name__ == "__main__":
    main()
