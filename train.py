import os
import math
import time
import sys
import textwrap
import torch
import wandb
import numpy
import random
import argparse
import contextlib
import torch.optim as optim
from statistics import mean
from dataclasses import asdict
from datetime import timedelta
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names, load_from_disk, interleave_datasets
from tqdm.auto import tqdm

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

PG_CPU = None

from data.datasets import VQADataset
from data.collators import VQACollator
from data.data_utils import synchronized_dataloader_step
from data.advanced_datasets import ConstantLengthDataset
from data.processors import get_image_processor, get_tokenizer

import models.config as config
from models.activation_checkpointing import normalize_activation_checkpointing_mode
from models.language_model import LanguageModel
from models.vision_language_model import VisionLanguageModel
from models.dual_tower.dual_tower import DualTowerVLM
from train_utils.console import ctext, log_debug, log_info, log_success, log_warn, progress_color
from train_utils.config_loader import (
    ConfigError,
    apply_dataclass_overrides,
    load_yaml_mapping,
    validate_allowed_keys,
)
from train_utils.env import load_project_dotenv

#Otherwise, the tokenizer will throw a warning
load_project_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import warnings
warnings.filterwarnings("ignore", message=".*Length of IterableDataset.*")
warnings.filterwarnings(
    "ignore",
    message="Token indices sequence length is longer than the specified maximum sequence length for this model.*",
)

# Fix for "Decompressed data too large" error with certain PNGs
import PIL.PngImagePlugin
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024
VAL_DEBUG_SAMPLE_INTERVAL = 100

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def init_dist():
    dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # torch.cuda.manual_seed(0)           # seed *this* GPU only

def destroy_dist():
    dist.destroy_process_group()

def is_dist():
    return dist.is_available() and dist.is_initialized()

def is_master():
    return dist.get_rank() == 0 if is_dist() else True

def get_world_size():
    return dist.get_world_size() if is_dist() else 1

def get_rank():
    return dist.get_rank() if is_dist() else 0

def dist_gather(obj):
    """
    Gather *any* picklable object from every rank without allocating
    temporary CUDA buffers.  Returns a list [rank0_obj, rank1_obj, …].

    Falls back to a single-rank list when torch.distributed is not initialised.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]

    result = [None] * dist.get_world_size()
    dist.all_gather_object(result, obj, group=PG_CPU)  # CPU path
    return result

def dist_mean_scalar(x: float | int) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return float(x)

    t = torch.tensor(x, device=torch.cuda.current_device(), dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)           # in‑place, returns None
    t /= dist.get_world_size()
    return t.item()

def wrap_model(model):
    local_rank = int(os.environ["LOCAL_RANK"])
    return DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)

def get_run_name(train_cfg, vlm_cfg):
    batch_size = f"bs{int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)}"
    max_training_steps = f"{train_cfg.max_training_steps}"
    learning_rate = f"lr_vision_{train_cfg.lr_vision_backbone}-language_{train_cfg.lr_language_backbone}-{train_cfg.lr_mp}"
    num_gpus = f"{get_world_size()}xGPU"
    date = time.strftime("%m%d-%H%M%S")
    vit = f"{vlm_cfg.vit_model_type.split('/')[-1]}" + f"_{vlm_cfg.max_img_size}"
    mp = f"mp{vlm_cfg.mp_pixel_shuffle_factor}"
    llm = f"{vlm_cfg.lm_model_type.split('/')[-1]}"
    prefix = train_cfg.wandb_run_name_prefix.strip()
    if prefix:
        prefix = f"{prefix}_"

    return f"{prefix}nanoVLM_{vit}_{mp}_{llm}_{num_gpus}_{batch_size}_{max_training_steps}_{learning_rate}_{date}"

def get_optimizer_lrs(optimizer, train_cfg):
    lrs = {}
    for group in optimizer.param_groups:
        group_name = group.get("name")
        if group_name:
            lrs[group_name] = group["lr"]
    if lrs:
        return lrs

    # Backward-compatible fallback when groups are unnamed.
    param_group_idx = 0
    if train_cfg.lr_mp > 0:
        lrs["lr_mp"] = optimizer.param_groups[param_group_idx]["lr"]
        param_group_idx += 1
    if train_cfg.lr_vision_backbone > 0:
        lrs["lr_vision_backbone"] = optimizer.param_groups[param_group_idx]["lr"]
        param_group_idx += 1
    if train_cfg.lr_language_backbone > 0:
        lrs["lr_language_backbone"] = optimizer.param_groups[param_group_idx]["lr"]
    return lrs

def _combine_split_datasets(datasets_to_combine, stream_dataset):
    if not datasets_to_combine:
        raise ValueError("No datasets were provided to combine.")
    if len(datasets_to_combine) == 1:
        return datasets_to_combine[0]
    if stream_dataset:
        return interleave_datasets(datasets_to_combine, stopping_strategy="all_exhausted")
    return concatenate_datasets(datasets_to_combine)

def _load_dataset_split(train_cfg, dataset_name, split_name):
    if "shard_" in dataset_name:
        ds = load_from_disk(dataset_name)
        return ds

    # `default` means "no explicit config name", required for single-config datasets.
    hf_config_name = None if dataset_name in ("default", "", None) else dataset_name
    ds = load_dataset(
        train_cfg.train_dataset_path,
        hf_config_name,
        split=split_name,
        streaming=train_cfg.stream_dataset,
        on_bad_files='warn',
    )
    return ds

def get_dataloaders(train_cfg, vlm_cfg):
    log_info(f"Getting dataloaders from {ctext(train_cfg.train_dataset_path, 'cyan', attrs=('bold',))}")
    if train_cfg.max_sample_length > vlm_cfg.lm_max_length:
        raise ValueError(
            f"max_sample_length ({train_cfg.max_sample_length}) must be <= lm_max_length ({vlm_cfg.lm_max_length})"
        )
    if not train_cfg.use_packing and train_cfg.stream_dataset:
        raise ValueError("stream_dataset=True requires use_packing=True in the current dataloader pipeline.")
    # Create datasets
    image_processor = get_image_processor(vlm_cfg.max_img_size, vlm_cfg.vit_img_size, vlm_cfg.resize_to_max_side_len)
    tokenizer = get_tokenizer(
        vlm_cfg.lm_tokenizer,
        vlm_cfg.vlm_extra_tokens,
        vlm_cfg.lm_chat_template,
        model_max_length=train_cfg.max_sample_length,
    )

    dataset_names_to_load = train_cfg.train_dataset_name
    if "shards" in train_cfg.train_dataset_name:
        log_info("Loading shards")
        total_shards = 56
        dataset_names_to_load = [train_cfg.train_dataset_path + f"/shard_{i}" for i in range(total_shards)]

    if "_all_" in dataset_names_to_load:
        dataset_names_to_load = get_dataset_config_names(train_cfg.train_dataset_path)

    # Load and combine datasets from explicit train/val splits.
    combined_train_data = []
    combined_val_data = []

    for dataset_name in dataset_names_to_load:
        log_info(f"Loading dataset: {ctext(dataset_name, 'cyan', attrs=('bold',))}")
        try:
            train_ds = _load_dataset_split(train_cfg, dataset_name, train_cfg.train_split)
            val_ds = _load_dataset_split(train_cfg, dataset_name, train_cfg.val_split)

            if train_cfg.stream_dataset:
                next(iter(train_ds))
                next(iter(val_ds))
            else:
                train_ds[0]
                val_ds[0]

            combined_train_data.append(train_ds)
            combined_val_data.append(val_ds)
        except Exception as e:
            if is_master():
                log_warn(
                    f"Warning: Failed to load dataset config '{dataset_name}' from "
                    f"'{train_cfg.train_dataset_path}' with splits "
                    f"'{train_cfg.train_split}'/'{train_cfg.val_split}'. Error: {e}"
                )
            continue

    if not combined_train_data or not combined_val_data:
        raise ValueError(
            "No valid train/val datasets were loaded. Please check dataset path, "
            "config names, and split names."
        )

    train_ds = _combine_split_datasets(combined_train_data, train_cfg.stream_dataset)
    val_ds = _combine_split_datasets(combined_val_data, train_cfg.stream_dataset)

    if not train_cfg.stream_dataset:
        train_ds = train_ds.shuffle(seed=0)


    if is_dist():  # We need to shard the dataset in DDP since we are using an iterable dataset instead of the distributed sampler
        print("GET WORLD SIZE DIST", get_world_size())
        train_ds = train_ds.shard(num_shards=get_world_size(), index=get_rank())
        val_ds = val_ds.shard(num_shards=get_world_size(), index=get_rank())

    train_dataset = VQADataset(
        train_ds,
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
        train_cfg.relevance_min_rating,
        train_cfg.image_correspondence_min_rating,
        train_cfg.visual_dependency_min_rating,
        train_cfg.formatting_min_rating,
    )
    val_dataset = VQADataset(
        val_ds,
        tokenizer,
        image_processor,
        vlm_cfg.mp_image_token_length,
        train_cfg.relevance_min_rating,
        train_cfg.image_correspondence_min_rating,
        train_cfg.visual_dependency_min_rating,
        train_cfg.formatting_min_rating,
    )

    if train_cfg.use_packing:
        train_dataset = ConstantLengthDataset(
            train_dataset,
            infinite=False,
            max_sample_length=train_cfg.max_sample_length,
            seq_length=vlm_cfg.lm_max_length,
            num_of_sequences=train_cfg.batch_size * 4,
            queue_size=2,
            max_images_per_example=train_cfg.max_images_per_example,
            max_images_per_knapsack=train_cfg.max_images_per_knapsack,
        )

        val_dataset = ConstantLengthDataset(
            val_dataset,
            infinite=False,
            max_sample_length=train_cfg.max_sample_length,
            seq_length=vlm_cfg.lm_max_length,
            num_of_sequences=train_cfg.batch_size * 4,
            queue_size=2,
            max_images_per_example=train_cfg.max_images_per_example,
            max_images_per_knapsack=train_cfg.max_images_per_knapsack,
        )

    # Create collators
    collator_max_len = vlm_cfg.lm_max_length if train_cfg.use_packing else train_cfg.max_sample_length
    vqa_collator = VQACollator(tokenizer, collator_max_len)

    g = torch.Generator()
    g.manual_seed(0)

    # Create dataloaders

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,    # =per device BS in DDP
        collate_fn=vqa_collator,
        num_workers=2,
        pin_memory=False,
        persistent_workers=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        collate_fn=vqa_collator,
        num_workers=2,
        pin_memory=False,
        persistent_workers=False,
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    # Warmup dataloaders to kickstart worker processes
    log_info("Warming up dataloaders...")
    iter_train_loader = iter(train_loader)
    iter_val_loader = iter(val_loader)
    next(iter_train_loader)
    next(iter_val_loader)
    log_success("Warmup complete.")

    return train_loader, val_loader, iter_train_loader, iter_val_loader

# Cosine learning rate schedule with warmup (from Karpathy)
# https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L353
def get_lr(it, max_lr, max_steps):
    min_lr = max_lr * 0.1
    warmup_steps = max_steps * 0.03
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


def _base_model(model):
    return model.module if hasattr(model, "module") else model


def _short_text(text, limit=245):
    text = text.replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _split_debug_lines(text: str, width: int = 52) -> list[str]:
    if not text:
        return [""]
    wrapped = textwrap.wrap(text.strip(), width=width)
    return wrapped if wrapped else [""]


def _log_pred_vs_target(step: int, pred_text: str, target_text: str, *, tag: str) -> None:
    left_lines = _split_debug_lines(pred_text)
    right_lines = _split_debug_lines(target_text)
    rows = max(len(left_lines), len(right_lines))
    left_lines.extend([""] * (rows - len(left_lines)))
    right_lines.extend([""] * (rows - len(right_lines)))
    log_debug(
        f"[{tag}][step={ctext(step, 'yellow', attrs=('bold',))}] "
        f"{ctext('PRED', 'yellow', attrs=('bold',))} | "
        f"{ctext('GT', 'green', attrs=('bold',))}"
    )
    for left, right in zip(left_lines, right_lines):
        log_debug(f"{ctext(left.ljust(54), 'yellow')} | {ctext(right, 'green')}")


def _build_decode_preview(model, logits, labels):
    tokenizer = getattr(_base_model(model), "tokenizer", None)
    if tokenizer is None:
        return None
    if logits is None or logits.ndim != 3 or labels.ndim != 2:
        return None
    if logits.size(0) == 0 or labels.size(0) == 0:
        return None

    valid_mask = labels[0] != -100
    if int(valid_mask.sum().item()) == 0:
        return None

    pred_ids = torch.argmax(logits[0].detach(), dim=-1)[valid_mask].to("cpu")
    target_ids = labels[0][valid_mask].detach().to("cpu")
    pred_text = _short_text(tokenizer.decode(pred_ids.tolist(), skip_special_tokens=True))
    target_text = _short_text(tokenizer.decode(target_ids.tolist(), skip_special_tokens=True))
    return pred_text, target_text


def _apply_checkpoint_cfg_overrides(loaded_cfg, requested_cfg):
    if loaded_cfg is None:
        raise ValueError("Loaded checkpoint model is missing `cfg`.")
    loaded_cfg.lm_max_length = requested_cfg.lm_max_length
    loaded_cfg.lm_max_position_embeddings = requested_cfg.lm_max_position_embeddings
    loaded_cfg.resize_to_max_side_len = getattr(requested_cfg, "resize_to_max_side_len", False)
    return loaded_cfg


def _compile_modulelist_blocks(module_list, *, fullgraph: bool):
    compiled_count = 0
    for idx, block in enumerate(module_list):
        module_list[idx] = torch.compile(block, fullgraph=fullgraph)
        compiled_count += 1
    return compiled_count


def _apply_regional_compile(model):
    compile_fullgraph = True
    summary = {
        "strategy": "regional_submodule_compile_fullgraph",
        "fullgraph": compile_fullgraph,
        "compiled": {},
    }

    if isinstance(model, DualTowerVLM):
        summary["compiled"]["left_tower_decoder_blocks"] = _compile_modulelist_blocks(
            model.left_tower.decoder.blocks,
            fullgraph=compile_fullgraph,
        )
        summary["compiled"]["right_tower_decoder_blocks"] = _compile_modulelist_blocks(
            model.right_tower.blocks,
            fullgraph=compile_fullgraph,
        )
        # Keep vision/projector eager: packed-image count varies by batch and can
        # trigger repeated recompiles on image-batch dimension.
        summary["compiled"]["left_tower_vision_blocks"] = 0
        summary["compiled"]["left_tower_mp"] = 0
        return model, summary

    if isinstance(model, VisionLanguageModel):
        summary["compiled"]["decoder_blocks"] = _compile_modulelist_blocks(
            model.decoder.blocks,
            fullgraph=compile_fullgraph,
        )
        summary["compiled"]["vision_blocks"] = 0
        summary["compiled"]["mp"] = 0
        return model, summary

    raise ValueError(
        f"Unsupported model type for regional compile: {type(model).__name__}"
    )


def _maybe_mark_batch_dynamic(*, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor):
    maybe_mark_dynamic = getattr(torch._dynamo, "maybe_mark_dynamic", None)
    tensors = (input_ids, labels, attention_mask)
    for tensor in tensors:
        if tensor.dim() < 2:
            continue
        if maybe_mark_dynamic is not None:
            maybe_mark_dynamic(tensor, 0)
            maybe_mark_dynamic(tensor, 1)
        else:
            torch._dynamo.mark_dynamic(tensor, 0)
            torch._dynamo.mark_dynamic(tensor, 1)


def train(train_cfg, vlm_cfg, model_mode: str = "nanovlm"):
    if train_cfg.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be > 0.")
    try:
        train_cfg.checkpoint_repo_pattern.format(step=1, i=1)
    except Exception as e:
        raise ValueError(
            "checkpoint_repo_pattern must be a valid format string containing '{step}' or '{i}'."
        ) from e
    if train_cfg.use_lmms_eval:
        raise ValueError(
            "use_lmms_eval=True is not supported in this hub-only checkpoint workflow."
        )
    vlm_cfg.activation_checkpointing_mode = normalize_activation_checkpointing_mode(
        getattr(vlm_cfg, "activation_checkpointing_mode", "regular")
    )
    if (
        bool(getattr(vlm_cfg, "activation_checkpointing", False))
        and vlm_cfg.activation_checkpointing_mode == "selective"
        and not train_cfg.compile
    ):
        raise ValueError(
            "Selective activation checkpointing requires `train.compile=True`. "
            "Set `vlm.activation_checkpointing_mode: regular` or enable compile."
        )

    train_loader, val_loader, iter_train_loader, iter_val_loader = get_dataloaders(train_cfg, vlm_cfg)

    if is_dist():
        log_info(f"Rank {get_rank()} waiting for all workers to get dataloaders...")
        if is_master():
            log_info("Waiting for all workers to get dataloaders...")
        dist.barrier(device_ids=int(os.environ["LOCAL_RANK"]))
        if is_master():
            log_success("All workers have loaded dataloaders.")

    run_name = get_run_name(train_cfg, vlm_cfg)
    run = None
    if train_cfg.log_wandb and is_master():
        run = wandb.init(
            # entity=train_cfg.wandb_entity,
            project=train_cfg.wandb_project,
            config={
                "VLMConfig": asdict(vlm_cfg),
                "TrainConfig": asdict(train_cfg)
            },
            name=run_name,
        )
        run.define_metric("train/consumed_tokens")
        run.define_metric("train/*", step_metric="train/consumed_tokens")
        run.define_metric("training_stats/*", step_metric="train/consumed_tokens")
        run.define_metric("val_loss", step_metric="train/consumed_tokens")
        run.define_metric("checkpoint/pushed", step_metric="train/consumed_tokens")
        run.define_metric("grad_norm", step_metric="train/consumed_tokens")

    # Initialize model
    if train_cfg.resume_from_vlm_checkpoint:
        log_info(f"Loading VLM checkpoint: {ctext(vlm_cfg.vlm_checkpoint_path, 'cyan', attrs=('bold',))}")
        vlm_model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
        vlm_cfg = _apply_checkpoint_cfg_overrides(getattr(vlm_model, "cfg", None), vlm_cfg)
        if model_mode == "dualtower":
            model = DualTowerVLM(
                vlm_cfg,
                load_backbone=False,
            )
            model.left_tower.vision_encoder.load_state_dict(vlm_model.vision_encoder.state_dict())
            model.left_tower.MP.load_state_dict(vlm_model.MP.state_dict())
            model.left_tower.decoder.load_state_dict(vlm_model.decoder.state_dict())
            right_lm = LanguageModel.from_pretrained(vlm_cfg)
            model.right_tower.load_state_dict(right_lm.state_dict())
            del right_lm
            del vlm_model
            log_success("Initialized dualtower from VLM checkpoint (left tower) + LM backbone (right tower).")
        else:
            model = vlm_model

        # Override model's max_seq_len, max_position_embeddings, and any relevant sample length to config values
        # Use attribute names as in VLMConfig (lm_max_length, lm_max_position_embeddings, max_sample_length)
        if hasattr(model, "cfg"):
            if hasattr(vlm_cfg, "lm_max_length"):
                model.cfg.lm_max_length = vlm_cfg.lm_max_length
            if hasattr(vlm_cfg, "lm_max_position_embeddings"):
                model.cfg.lm_max_position_embeddings = vlm_cfg.lm_max_position_embeddings

        if hasattr(model, "max_seq_len") and hasattr(vlm_cfg, "lm_max_length"):
            model.max_seq_len = vlm_cfg.lm_max_length
        if hasattr(model, "max_position_embeddings") and hasattr(vlm_cfg, "lm_max_position_embeddings"):
            model.max_position_embeddings = vlm_cfg.lm_max_position_embeddings
        # Optionally handle decoder property (if model.decoder exists and has these attributes)
        if hasattr(model, "decoder"):
            if hasattr(model.decoder, "max_seq_len") and hasattr(vlm_cfg, "lm_max_length"):
                model.decoder.max_seq_len = vlm_cfg.lm_max_length
            if hasattr(model.decoder, "max_position_embeddings") and hasattr(vlm_cfg, "lm_max_position_embeddings"):
                model.decoder.max_position_embeddings = vlm_cfg.lm_max_position_embeddings

        if model_mode == "dualtower":
            # DualTowerVLM has two language decoders; propagate overrides to both.
            dual_decoders = []
            if hasattr(model, "left_tower") and hasattr(model.left_tower, "decoder"):
                dual_decoders.append(model.left_tower.decoder)
            if hasattr(model, "right_tower"):
                dual_decoders.append(model.right_tower)

            for decoder in dual_decoders:
                if hasattr(vlm_cfg, "lm_max_length") and hasattr(decoder, "max_seq_len"):
                    decoder.max_seq_len = vlm_cfg.lm_max_length
                if hasattr(vlm_cfg, "lm_max_position_embeddings") and hasattr(decoder, "max_position_embeddings"):
                    decoder.max_position_embeddings = vlm_cfg.lm_max_position_embeddings
                if hasattr(decoder, "cfg"):
                    if hasattr(vlm_cfg, "lm_max_length"):
                        decoder.cfg.lm_max_length = vlm_cfg.lm_max_length
                    if hasattr(vlm_cfg, "lm_max_position_embeddings"):
                        decoder.cfg.lm_max_position_embeddings = vlm_cfg.lm_max_position_embeddings
                if hasattr(decoder, "rotary_embd"):
                    if hasattr(vlm_cfg, "lm_max_position_embeddings") and hasattr(decoder.rotary_embd, "max_seq_len"):
                        decoder.rotary_embd.max_seq_len = vlm_cfg.lm_max_position_embeddings
                    if hasattr(vlm_cfg, "lm_max_position_embeddings") and hasattr(decoder.rotary_embd, "original_max_seq_len"):
                        decoder.rotary_embd.original_max_seq_len = vlm_cfg.lm_max_position_embeddings

        # In case model has sample length or tokenizer max length to override
        if hasattr(model, "max_sample_length") and hasattr(train_cfg, "max_sample_length"):
            model.max_sample_length = train_cfg.max_sample_length
        if hasattr(model, "tokenizer") and hasattr(model.tokenizer, "model_max_length") and hasattr(train_cfg, "max_sample_length"):
            model.tokenizer.model_max_length = train_cfg.max_sample_length

    else:
        if model_mode == "dualtower":
            model = DualTowerVLM(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)
        else:
            model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)
    
    if is_master():
        model_label = "DualTowerVLM" if model_mode == "dualtower" else "nanoVLM"
        total_params = sum(p.numel() for p in model.parameters())
        log_info(
            f"{ctext(model_label, 'cyan', attrs=('bold',))} initialized with "
            f"{ctext(f'{total_params:,}', 'cyan', attrs=('bold',))} parameters"
        )
        log_info(
            f"Training summary{' (global)' if is_dist() else ''}: "
            f"batch size {ctext(int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps), 'cyan', attrs=('bold',))}"
            f"{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}"
        )
        if is_dist():
            log_info(f"Training summary per GPU: batch size {ctext(train_loader.batch_size, 'cyan', attrs=('bold',))}")
        log_info(
            f"Validation summary{' (global)' if is_dist() else ''}: "
            f"batch size {ctext(int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps), 'cyan', attrs=('bold',))}"
            f"{', training on ' + str(get_world_size()) + ' GPUs' if is_dist() else ''}"
        )
        if is_dist():
            log_info(f"Validation summary per GPU: batch size {ctext(val_loader.batch_size, 'cyan', attrs=('bold',))}")

    # Define optimizer groups
    # Since we have pretrained vision and language backbones, but a newly initialized modality projection layer, it doesn't make sense to train them with the same learning rate
    # You could opt to fully freeze the backbones and only train the MP layer, but finetuning them with a lower learning rate makes the training as a whole easier
    param_groups = []
    dual_left_lr = (
        train_cfg.lr_left_tower
        if train_cfg.lr_left_tower is not None
        else train_cfg.lr_language_backbone
    )
    dual_right_lr = (
        train_cfg.lr_right_tower
        if train_cfg.lr_right_tower is not None
        else train_cfg.lr_language_backbone
    )

    mp_module = model.left_tower.MP if model_mode == "dualtower" and hasattr(model, "left_tower") else model.MP
    vision_module = model.left_tower.vision_encoder if model_mode == "dualtower" and hasattr(model, "left_tower") else model.vision_encoder

    if train_cfg.lr_mp > 0:
        param_groups.append({'name': 'lr_mp', 'params': list(mp_module.parameters()), 'lr': train_cfg.lr_mp})
    else:
        for p in list(mp_module.parameters()):
            p.requires_grad = False

    if train_cfg.lr_vision_backbone > 0:
        param_groups.append({'name': 'lr_vision_backbone', 'params': list(vision_module.parameters()), 'lr': train_cfg.lr_vision_backbone})
    else:
        for p in list(vision_module.parameters()):
            p.requires_grad = False

    if model_mode == "dualtower" and hasattr(model, "left_tower") and hasattr(model, "right_tower"):
        if dual_left_lr > 0:
            param_groups.append({'name': 'lr_left_tower', 'params': list(model.left_tower.decoder.parameters()), 'lr': dual_left_lr})
        else:
            for p in list(model.left_tower.decoder.parameters()):
                p.requires_grad = False

        if dual_right_lr > 0:
            param_groups.append({'name': 'lr_right_tower', 'params': list(model.right_tower.parameters()), 'lr': dual_right_lr})
        else:
            for p in list(model.right_tower.parameters()):
                p.requires_grad = False
    else:
        if train_cfg.lr_language_backbone > 0:
            param_groups.append({'name': 'lr_language_backbone', 'params': list(model.decoder.parameters()), 'lr': train_cfg.lr_language_backbone})
        else:
            for p in list(model.decoder.parameters()):
                p.requires_grad = False

    optimizer = optim.AdamW(param_groups)
    all_params = [p for group in optimizer.param_groups for p in group['params']]

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    if device.type == "mps":
        torch.backends.mps.enable_fallback_to_cpu = True
        torch.mps.empty_cache()
    
    log_info(f"Using device: {ctext(device, 'cyan', attrs=('bold',))}")
    model.to(device)
    
    if train_cfg.compile:
        model, compile_summary = _apply_regional_compile(model)
        compiled_kv = ", ".join(
            f"{name}={count}" for name, count in compile_summary["compiled"].items()
        )
        log_info(
            f"Compile enabled with strategy "
            f"{ctext(compile_summary['strategy'], 'cyan', attrs=('bold',))}"
        )
        log_info(f"Compile fullgraph={ctext(str(compile_summary['fullgraph']), 'cyan')}")
        log_info(f"Compiled modules: {ctext(compiled_kv, 'cyan')}")
    else:
        log_info("Compile disabled.")
    if is_dist():
        log_info("Wrapping model for DDP")
        model = wrap_model(model)
        log_success("Model wrapped for DDP")

    epoch_times = []
    best_val_loss = float('inf')
    best_val_step = None
    best_checkpoint_repo_id = None
    checkpoint_repo_by_step = {}
    global_step = 0
    global_consumed_tokens = 0
    epoch = 0
    
    # Training stats accumulators
    accumulated_stats = {
        'tokens_per_second': [],
        'data_load_time': [],
        'fw_bw_time': [],
        'post_process_time': [],
        'images_per_sample': [],
        'effective_token_ratio_per_instance': [],
    }
    
    while global_step < train_cfg.max_training_steps:
        epoch += 1
        epoch_start_time = time.time()
        model.train()
        total_train_loss_sum = 0.0
        total_train_loss_tokens = 0
        total_tokens_processed = 0
        optimizer.zero_grad()
        accumulated_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        accumulated_loss_tokens = torch.zeros((), device=device, dtype=torch.float32)
        accumulated_effective_ratio_sum = torch.zeros((), device=device, dtype=torch.float32)
        accumulated_effective_ratio_count = torch.zeros((), device=device, dtype=torch.float32)
        accumulated_tokens_for_update_local = 0
        data_load_start = time.time()

        log_info(f"Starting training loop for epoch {ctext(epoch, 'cyan', attrs=('bold',))}")
        remaining_steps = train_cfg.max_training_steps - global_step
        step_progress = tqdm(
            total=remaining_steps,
            desc=f"Epoch {epoch}",
            dynamic_ncols=True,
            disable=not is_master(),
            leave=False,
        )
        for i, batch in enumerate(synchronized_dataloader_step(iter_train_loader, is_dist())):
            is_update_step = (i + 1) % train_cfg.gradient_accumulation_steps == 0
            step_after_update = global_step + 1 if is_update_step else global_step
            grad_norm_value = None
            update_loss_value = None
            step_effective_token_ratio_value = None
            batch_start_time = time.time()
            images = batch["images"]
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            if train_cfg.compile:
                _maybe_mark_batch_dynamic(
                    input_ids=input_ids,
                    labels=labels,
                    attention_mask=attention_mask,
                )
            data_load_time = time.time() - data_load_start

            # When using DDP with gradient accumulation,
            # skip gradient synchronization on intermediate steps to save time.
            # Gradients only need to be synced at the end of each accumulation cycle.
            if (is_dist()
                and train_cfg.gradient_accumulation_steps > 1
                and not is_update_step):
                context = model.no_sync()
            else:
                context = contextlib.nullcontext()

            fw_bw_start = time.time()
            autocast_context = torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16 if device.type in ['cuda', 'cpu'] else torch.float16
            )
            with autocast_context:
                with context:
                    _, loss, loss_token_count = model(
                        input_ids,
                        images,
                        attention_mask=attention_mask,
                        targets=labels,
                        loss_reduction="sum",
                        return_loss_count=True,
                    )
                    loss_token_count_value = int(loss_token_count.item())
                    if loss_token_count_value <= 0:
                        breakpoint()
                        raise ValueError("Found a batch with no valid target tokens; check label masking.")
                    accumulated_loss_sum += loss.detach().to(dtype=torch.float32)
                    accumulated_loss_tokens += loss_token_count.to(dtype=torch.float32)
                    valid_tokens_per_sample = (labels != -100).sum(dim=1).to(dtype=torch.float32)
                    attention_tokens_per_sample = attention_mask.sum(dim=1).to(dtype=torch.float32).clamp_min(1.0)
                    effective_token_ratio_per_sample = valid_tokens_per_sample / attention_tokens_per_sample
                    accumulated_effective_ratio_sum += effective_token_ratio_per_sample.sum()
                    accumulated_effective_ratio_count += float(effective_token_ratio_per_sample.numel())

            loss.backward()

            fw_bw_time = time.time() - fw_bw_start
            post_process_start = time.time()
            if is_update_step:
                total_loss_tokens = accumulated_loss_tokens.clone()
                total_loss_sum = accumulated_loss_sum.clone()
                if is_dist():
                    dist.all_reduce(total_loss_tokens, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_loss_sum, op=dist.ReduceOp.SUM)

                total_loss_tokens_value = total_loss_tokens.item()
                if total_loss_tokens_value <= 0:
                    raise ValueError("Gradient accumulation produced zero total target tokens; check label masking.")

                update_loss_value = (total_loss_sum / total_loss_tokens).item()
                total_effective_ratio_sum = accumulated_effective_ratio_sum.clone()
                total_effective_ratio_count = accumulated_effective_ratio_count.clone()
                if is_dist():
                    dist.all_reduce(total_effective_ratio_sum, op=dist.ReduceOp.SUM)
                    dist.all_reduce(total_effective_ratio_count, op=dist.ReduceOp.SUM)
                total_effective_ratio_count_value = total_effective_ratio_count.item()
                if total_effective_ratio_count_value <= 0:
                    raise ValueError("Gradient accumulation produced zero instances; cannot compute effective token ratio.")
                step_effective_token_ratio_value = (total_effective_ratio_sum / total_effective_ratio_count).item()
                grad_scale = get_world_size() / total_loss_tokens_value
                for param in all_params:
                    if param.grad is not None:
                        param.grad.mul_(grad_scale)

                if train_cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=train_cfg.max_grad_norm)
                    grad_norm_value = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
                    if is_dist():
                        grad_norm_value = dist_mean_scalar(grad_norm_value)

                for group in optimizer.param_groups:
                    group_name = group.get("name")
                    max_lr = None
                    if group_name == "lr_mp":
                        max_lr = train_cfg.lr_mp
                    elif group_name == "lr_vision_backbone":
                        max_lr = train_cfg.lr_vision_backbone
                    elif group_name == "lr_language_backbone":
                        max_lr = train_cfg.lr_language_backbone
                    elif group_name == "lr_left_tower":
                        max_lr = (
                            train_cfg.lr_left_tower
                            if train_cfg.lr_left_tower is not None
                            else train_cfg.lr_language_backbone
                        )
                    elif group_name == "lr_right_tower":
                        max_lr = (
                            train_cfg.lr_right_tower
                            if train_cfg.lr_right_tower is not None
                            else train_cfg.lr_language_backbone
                        )

                    if max_lr is not None and max_lr > 0:
                        group['lr'] = get_lr(step_after_update, max_lr, train_cfg.max_training_steps)
              
                optimizer.step()
                optimizer.zero_grad()
                accumulated_loss_sum.zero_()
                accumulated_loss_tokens.zero_()
                accumulated_effective_ratio_sum.zero_()
                accumulated_effective_ratio_count.zero_()

            batch_loss = loss.item() / loss_token_count_value
            batch_effective_token_ratio = effective_token_ratio_per_sample.mean().item()
            total_train_loss_sum += float(loss.item())
            total_train_loss_tokens += loss_token_count_value

            num_tokens = torch.sum(attention_mask).item() # Sum of attention mask gives number of tokens
            total_tokens_processed += num_tokens
            accumulated_tokens_for_update_local += num_tokens
            post_process_time = time.time() - post_process_start

            images_per_sample = [len(image_pack) for image_pack in images]

            batch_end_time = time.time()
            batch_duration = batch_end_time - batch_start_time
            tokens_per_second = get_world_size() * num_tokens / batch_duration  # Multiply by world size to get global tokens/s

            # Accumulate training stats
            accumulated_stats['tokens_per_second'].append(tokens_per_second)
            accumulated_stats['data_load_time'].append(data_load_time)
            accumulated_stats['fw_bw_time'].append(fw_bw_time)
            accumulated_stats['post_process_time'].append(post_process_time)
            accumulated_stats['images_per_sample'].extend(images_per_sample)
            accumulated_stats['effective_token_ratio_per_instance'].append(batch_effective_token_ratio)

            if is_update_step:
                if is_dist():
                    update_tokens = torch.tensor(
                        float(accumulated_tokens_for_update_local),
                        device=device,
                        dtype=torch.float64,
                    )
                    dist.all_reduce(update_tokens, op=dist.ReduceOp.SUM)
                    update_tokens_global = int(update_tokens.item())
                else:
                    update_tokens_global = int(accumulated_tokens_for_update_local)
                global_consumed_tokens += update_tokens_global
                accumulated_tokens_for_update_local = 0
            
            if (
                train_cfg.eval_in_epochs
                and is_update_step
                and step_after_update % train_cfg.eval_interval == 0
            ):
                log_info(
                    f"Starting evaluation at step "
                    f"{progress_color(step_after_update, train_cfg.max_training_steps)}"
                )
                model.eval()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    total_val_loss = 0
                    total_val_tokens = 0
                    val_batches = 0
                    should_log_val_decode = (step_after_update % VAL_DEBUG_SAMPLE_INTERVAL == 0)
                    logged_val_decode = False
                    max_val_batches = 64  # Limit the number of batches to 64

                    for batch in synchronized_dataloader_step(iter_val_loader, is_dist()):
                        if val_batches >= max_val_batches:
                            break
                        images = batch["images"]
                        input_ids = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)
                        attention_mask = batch["attention_mask"].to(device)

                        with autocast_context:
                            logits, loss, loss_token_count = model(
                                input_ids,
                                images,
                                attention_mask=attention_mask,
                                targets=labels,
                                loss_reduction="sum",
                                return_loss_count=True,
                            )

                        if is_master() and should_log_val_decode and not logged_val_decode:
                            preview = _build_decode_preview(model, logits, labels)
                            if preview is not None:
                                pred_text, target_text = preview
                                _log_pred_vs_target(
                                    step_after_update,
                                    pred_text,
                                    target_text,
                                    tag="val-debug",
                                )
                                logged_val_decode = True

                        total_val_loss += loss.item()
                        total_val_tokens += int(loss_token_count.item())
                        val_batches += 1
                    
                    iter_val_loader = iter(val_loader)
                    if total_val_tokens <= 0:
                        raise ValueError("Validation produced zero target tokens; cannot compute validation loss.")
                    val_loss_stats = torch.tensor(
                        [float(total_val_loss), float(total_val_tokens)],
                        device=device,
                        dtype=torch.float64,
                    )
                    if is_dist():
                        dist.all_reduce(val_loss_stats, op=dist.ReduceOp.SUM)
                    avg_val_loss = val_loss_stats[0].item() / max(val_loss_stats[1].item(), 1.0)

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_val_step = step_after_update
                        best_checkpoint_repo_id = checkpoint_repo_by_step.get(step_after_update)
                    
                    if is_master():
                        log_info(
                            f"Validation {progress_color(step_after_update, train_cfg.max_training_steps)} | "
                            f"Val Loss: {ctext(f'{avg_val_loss:.4f}', 'yellow', attrs=('bold',))} | "
                            f"Tokens/s: {ctext(f'{tokens_per_second:.2f}', 'yellow', attrs=('bold',))}"
                        )
                        if train_cfg.log_wandb:
                            run.log(
                                {
                                    "val_loss": avg_val_loss,
                                    "train/consumed_tokens": global_consumed_tokens,
                                },
                                step=step_after_update,
                            )

                model.train()

            if (
                is_update_step
                and train_cfg.push_checkpoints_to_hub
                and step_after_update % train_cfg.checkpoint_interval == 0
                and is_master()
            ):
                checkpoint_repo_id = train_cfg.checkpoint_repo_pattern.format(
                    step=step_after_update,
                    i=step_after_update,
                )
                save_model = model.module if is_dist() else model
                log_info(
                    f"Pushing checkpoint for step "
                    f"{progress_color(step_after_update, train_cfg.max_training_steps)} "
                    f"to {ctext(checkpoint_repo_id, 'cyan')}"
                )
                save_model.push_to_hub(
                    checkpoint_repo_id,
                    private=train_cfg.hf_private,
                )
                checkpoint_repo_by_step[step_after_update] = checkpoint_repo_id
                if best_val_step == step_after_update:
                    best_checkpoint_repo_id = checkpoint_repo_id
                if train_cfg.log_wandb:
                    run.log(
                        {
                            "checkpoint/pushed": 1,
                            "train/consumed_tokens": global_consumed_tokens,
                        },
                        step=step_after_update,
                    )

            # Log training stats every N update steps (ALL RANKS must participate in collective ops)
            if (
                is_update_step
                and step_after_update % train_cfg.stats_log_interval == 0
                and len(accumulated_stats['tokens_per_second']) > 0
            ):
                # ALL RANKS: Perform collective operations for training stats
                stats = {}
                for key in ['tokens_per_second', 'data_load_time', 'fw_bw_time', 'post_process_time', 'images_per_sample', 'effective_token_ratio_per_instance']:
                    if is_dist():
                        all_values = dist_gather(accumulated_stats[key])
                        all_values_flat = [item for sublist in all_values for item in sublist]  # Flatten list of lists
                        stats[f'avg_{key}'] = mean(all_values_flat)
                    else:
                        stats[f'avg_{key}'] = mean(accumulated_stats[key])
                
                for key in ['data_load_time', 'fw_bw_time', 'post_process_time', 'images_per_sample']:
                    if is_dist():
                        all_values = dist_gather(accumulated_stats[key])
                        all_values_flat = [item for sublist in all_values for item in sublist]
                        stats[f'max_{key}'] = max(all_values_flat)
                    else:
                        stats[f'max_{key}'] = max(accumulated_stats[key])

                if is_dist():
                    all_images_values = dist_gather(accumulated_stats['images_per_sample'])
                    all_images_flat = [item for sublist in all_images_values for item in sublist]
                    stats['min_images_per_sample'] = min(all_images_flat)
                else:
                    stats['min_images_per_sample'] = min(accumulated_stats['images_per_sample'])
                
                lr_stats = get_optimizer_lrs(optimizer, train_cfg)
                stats.update(lr_stats)
                if grad_norm_value is not None:
                    stats["grad_norm"] = grad_norm_value

                # MASTER ONLY: Log to wandb
                if train_cfg.log_wandb and is_master():
                    run.log({
                        "train/consumed_tokens": global_consumed_tokens,
                        **{f"training_stats/{key}": value for key, value in stats.items()},
                    }, step=step_after_update)

                if is_master():
                    update_loss_text = update_loss_value if update_loss_value is not None else batch_loss
                    step_effective_ratio_text = (
                        step_effective_token_ratio_value
                        if step_effective_token_ratio_value is not None
                        else batch_effective_token_ratio
                    )
                    log_info(
                        f"[TRAIN] step={step_after_update} "
                        f"batch_loss={batch_loss:.4f} "
                        f"step_loss={update_loss_text:.4f} "
                        f"effective_token_ratio_per_instance={step_effective_ratio_text:.4f} "
                        f"tokens_per_second={stats['avg_tokens_per_second']:.2f} "
                        f"lr_mp={stats.get('lr_mp', 0.0):.6g} "
                        f"lr_vision={stats.get('lr_vision_backbone', 0.0):.6g} "
                        f"lr_lm={stats.get('lr_language_backbone', 0.0):.6g} "
                        f"lr_left={stats.get('lr_left_tower', 0.0):.6g} "
                        f"lr_right={stats.get('lr_right_tower', 0.0):.6g} "
                        + (f"grad_norm={stats['grad_norm']:.4f}" if 'grad_norm' in stats else "")
                    )
                
                # ALL RANKS: Reset accumulators
                for key in accumulated_stats:
                    accumulated_stats[key] = []

            # Log batch loss  
            if is_update_step:
                # ALL RANKS: gather loss from all ranks if DDP
                microbatch_loss_for_log = batch_loss
                step_loss_for_log = update_loss_value if update_loss_value is not None else batch_loss
                microbatch_effective_ratio_for_log = batch_effective_token_ratio
                step_effective_ratio_for_log = (
                    step_effective_token_ratio_value
                    if step_effective_token_ratio_value is not None
                    else batch_effective_token_ratio
                )
                if is_dist():
                    microbatch_loss_gathered = dist_mean_scalar(microbatch_loss_for_log)
                    step_loss_gathered = dist_mean_scalar(step_loss_for_log)
                    microbatch_effective_ratio_gathered = dist_mean_scalar(microbatch_effective_ratio_for_log)
                    step_effective_ratio_gathered = dist_mean_scalar(step_effective_ratio_for_log)
                else:
                    microbatch_loss_gathered = microbatch_loss_for_log
                    step_loss_gathered = step_loss_for_log
                    microbatch_effective_ratio_gathered = microbatch_effective_ratio_for_log
                    step_effective_ratio_gathered = step_effective_ratio_for_log
                    
                # MASTER ONLY: Log to wandb
                if train_cfg.log_wandb and is_master():
                    run.log({
                        "train/consumed_tokens": global_consumed_tokens,
                        "train/batch_loss": microbatch_loss_gathered,
                        "train/step_loss": step_loss_gathered,
                        "train/batch_effective_token_ratio_per_instance": microbatch_effective_ratio_gathered,
                        "train/step_effective_token_ratio_per_instance": step_effective_ratio_gathered,
                        **({"grad_norm": grad_norm_value} if grad_norm_value is not None else {})
                    }, step=step_after_update)
                
            if is_update_step:
                global_step = step_after_update
                if is_master():
                    step_progress.update(1)
                if global_step >= train_cfg.max_training_steps:
                    break
            data_load_start = time.time()

        step_progress.close()
        iter_train_loader = iter(train_loader)
        if total_train_loss_tokens <= 0:
            raise ValueError("No valid target tokens were accumulated during epoch; cannot compute epoch loss.")
        epoch_loss_stats = torch.tensor(
            [float(total_train_loss_sum), float(total_train_loss_tokens)],
            device=device,
            dtype=torch.float64,
        )
        if is_dist():
            dist.all_reduce(epoch_loss_stats, op=dist.ReduceOp.SUM)
        avg_train_loss = epoch_loss_stats[0].item() / max(epoch_loss_stats[1].item(), 1.0)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)

        # gather and sum total_tokens_processed across all ranks if DDP
        total_tokens_processed = sum(dist_gather(total_tokens_processed)) if is_dist() else total_tokens_processed  
        epoch_tokens_per_second = total_tokens_processed / epoch_duration

        if is_master():
            if train_cfg.log_wandb:
                run.log({
                    "train/consumed_tokens": global_consumed_tokens,
                    "train/epoch_loss": avg_train_loss,
                    "train/epoch_duration": epoch_duration,
                    "train/epoch_tokens_per_second": epoch_tokens_per_second,
                })

            log_info(
                f"Epoch {ctext(epoch, 'cyan', attrs=('bold',))} | "
                f"Step {progress_color(global_step, train_cfg.max_training_steps)} | "
                f"Train Loss: {ctext(f'{avg_train_loss:.4f}', 'yellow', attrs=('bold',))} | "
                f"Time: {epoch_duration:.2f}s | "
                f"T/s: {ctext(f'{epoch_tokens_per_second:.2f}', 'yellow', attrs=('bold',))}"
            )

    # Summary Statistics
    if is_master():
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        total_training_time = sum(epoch_times)
        batch_size = int(train_cfg.batch_size*get_world_size()*train_cfg.gradient_accumulation_steps)
        total_samples_processed = batch_size * global_step
        avg_time_per_sample = total_training_time / total_samples_processed
        log_success(f"Average time per epoch: {ctext(f'{avg_epoch_time:.2f}s', 'cyan', attrs=('bold',))}")
        log_success(f"Average time per sample: {ctext(f'{avg_time_per_sample:.4f}s', 'cyan', attrs=('bold',))}")
        if best_val_step is not None:
            log_success(
                f"Best validation loss {best_val_loss:.4f} observed at step {best_val_step}"
                + (f" ({best_checkpoint_repo_id})" if best_checkpoint_repo_id is not None else "")
            )
        else:
            log_warn("No validation pass was run during training.")

        if train_cfg.push_final_model_to_hub and vlm_cfg.hf_repo_name is not None:
            save_model = model.module if is_dist() else model
            log_info(f"Pushing final model to Hugging Face Hub repo {ctext(vlm_cfg.hf_repo_name, 'cyan')}")
            save_model.push_to_hub(
                vlm_cfg.hf_repo_name,
                private=train_cfg.hf_private,
            )

        if train_cfg.log_wandb:
            run.summary["avg_epoch_time"] = avg_epoch_time
            run.summary["avg_time_per_sample"] = avg_time_per_sample
            run.summary["best_val_loss"] = best_val_loss if best_val_step is not None else None
            run.summary["best_val_step"] = best_val_step
            run.summary["train/consumed_tokens"] = global_consumed_tokens
            if best_checkpoint_repo_id is not None:
                run.summary["best_checkpoint_repo"] = best_checkpoint_repo_id
            run.finish()

def main():
    global PG_CPU

    def _flag_present(argv, flag):
        return any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)

    def _validate_config_mode_overrides(argv):
        disallowed_flags = (
            "--lr_mp",
            "--lr_vision_backbone",
            "--lr_language_backbone",
            "--lr_left_tower",
            "--lr_right_tower",
            "--vlm_checkpoint_path",
            "--left_tower_mask_mode",
            "--momh_enabled",
            "--compile",
            "--log_wandb",
            "--resume_from_vlm_checkpoint",
            "--no_log_wandb",
            "--train_dataset_path",
            "--train_dataset_name",
            "--train_split",
            "--val_split",
            "--checkpoint_interval",
            "--checkpoint_repo_pattern",
            "--no_checkpoint_push",
            "--no_lmms_eval",
            "--relevance_min_rating",
            "--image_correspondence_min_rating",
            "--visual_dependency_min_rating",
            "--formatting_min_rating",
            "--packing",
            "--no_packing",
        )
        provided = [flag for flag in disallowed_flags if _flag_present(argv, flag)]
        if provided:
            provided_text = ", ".join(provided)
            raise ConfigError(
                "When --config is provided, only --nanovlm/--dualtower can override the config mode. "
                f"Remove these CLI overrides: {provided_text}."
            )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='Path to a YAML training config file. When provided, non-mode CLI overrides are disallowed.',
    )
    parser.add_argument('--lr_mp', type=float, help='Learning rate for the mapping network')
    parser.add_argument('--lr_vision_backbone', type=float, help='Learning rate for the vision backbone')
    parser.add_argument('--lr_language_backbone', type=float, help='Learning rate for the language backbone')
    parser.add_argument('--lr_left_tower', type=float, help='DualTower: learning rate for the left tower language decoder')
    parser.add_argument('--lr_right_tower', type=float, help='DualTower: learning rate for the right tower')
    parser.add_argument('--vlm_checkpoint_path', type=str, help='Path or repo ID of the VLM checkpoint for loading')
    parser.add_argument(
        '--left_tower_mask_mode',
        type=str,
        choices=['visual_only', 'visual_plus_prefix', 'full'],
        help='DualTower left-tower mask mode: visual_only, visual_plus_prefix, or full',
    )
    parser.add_argument('--momh_enabled', type=bool, help='Enable MoMH attention')
    parser.add_argument('--compile', type=bool, help='Use torch.compile to optimize the model')
    parser.add_argument('--log_wandb', type=bool, help='Log to wandb')
    parser.add_argument('--resume_from_vlm_checkpoint', type=bool, default=False, help='Resume training from VLM checkpoint specified by vlm_checkpoint_path (or default if not provided)')
    parser.add_argument('--no_log_wandb', action='store_true', help='Do not log to wandb')
    parser.add_argument('--train_dataset_path', type=str, help='Train dataset path')
    parser.add_argument('--train_dataset_name', nargs='+', type=str, help='Dataset config names to load (use "default" for single-config datasets)')
    parser.add_argument('--train_split', type=str, help='Dataset split name for training data')
    parser.add_argument('--val_split', type=str, help='Dataset split name for validation data')
    parser.add_argument('--checkpoint_interval', type=int, help='Push checkpoint to hub every N optimizer steps')
    parser.add_argument('--checkpoint_repo_pattern', type=str, help='Hub repo naming pattern with {i} or {step}, e.g. user/model-step-{i}')
    parser.add_argument('--no_checkpoint_push', action='store_true', help='Disable hub checkpoint pushing')
    parser.add_argument('--no_lmms_eval', action='store_true', help='Disable lmms-eval during training')
    parser.add_argument('--relevance_min_rating', type=int, help='Minimum relevance rating of images per sample')
    parser.add_argument('--image_correspondence_min_rating', type=int, help='Minimum image correspondence rating of images per sample')
    parser.add_argument('--visual_dependency_min_rating', type=int, help='Minimum visual dependency rating of images per sample')
    parser.add_argument('--formatting_min_rating', type=int, help='Minimum formatting rating of images per sample')
    packing_group = parser.add_mutually_exclusive_group()
    packing_group.add_argument('--packing', dest='use_packing', action='store_true', help='Enable packed training samples (default).')
    packing_group.add_argument('--no_packing', dest='use_packing', action='store_false', help='Disable sample packing and train on single samples.')
    parser.set_defaults(use_packing=None)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--dualtower', action='store_true', help='Use DualTowerVLM architecture')
    mode_group.add_argument('--nanovlm', action='store_true', help='Use nanoVLM architecture (default)')

    args = parser.parse_args()

    vlm_cfg = config.VLMConfig()
    train_cfg = config.TrainConfig()
    config_mode = None

    if args.config is not None:
        try:
            _validate_config_mode_overrides(sys.argv[1:])
            config_data = load_yaml_mapping(args.config)
            validate_allowed_keys(config_data, {"mode", "vlm", "train"}, f"training config '{args.config}'")

            if "mode" in config_data:
                mode = config_data["mode"]
                if not isinstance(mode, str):
                    raise ConfigError(f"Field 'mode' must be a string, got {type(mode).__name__}.")
                if mode not in {"nanovlm", "dualtower"}:
                    raise ConfigError("Field 'mode' must be either 'nanovlm' or 'dualtower'.")
                config_mode = mode

            if "vlm" in config_data:
                apply_dataclass_overrides(vlm_cfg, config_data["vlm"], "vlm")
            if "train" in config_data:
                apply_dataclass_overrides(train_cfg, config_data["train"], "train")
        except ConfigError as exc:
            parser.error(str(exc))
    else:
        if args.lr_mp is not None:
            train_cfg.lr_mp = args.lr_mp
        if args.lr_vision_backbone is not None:
            train_cfg.lr_vision_backbone = args.lr_vision_backbone
        if args.lr_language_backbone is not None:
            train_cfg.lr_language_backbone = args.lr_language_backbone
        if args.lr_left_tower is not None:
            train_cfg.lr_left_tower = args.lr_left_tower
        if args.lr_right_tower is not None:
            train_cfg.lr_right_tower = args.lr_right_tower
        if args.vlm_checkpoint_path is not None:
            vlm_cfg.vlm_checkpoint_path = args.vlm_checkpoint_path
        if args.left_tower_mask_mode is not None:
            vlm_cfg.left_tower_mask_mode = args.left_tower_mask_mode
        if args.momh_enabled is not None:
            vlm_cfg.momh_enabled = args.momh_enabled
        if args.compile is not None:
            train_cfg.compile = args.compile
        if args.log_wandb is not None:
            train_cfg.log_wandb = args.log_wandb
        if args.no_log_wandb is True:
            train_cfg.log_wandb = False
        if args.train_dataset_path is not None:
            train_cfg.train_dataset_path = args.train_dataset_path
        if args.train_dataset_name is not None:
            train_cfg.train_dataset_name = tuple(args.train_dataset_name)
        if args.train_split is not None:
            train_cfg.train_split = args.train_split
        if args.val_split is not None:
            train_cfg.val_split = args.val_split
        if args.checkpoint_interval is not None:
            train_cfg.checkpoint_interval = args.checkpoint_interval
        if args.checkpoint_repo_pattern is not None:
            train_cfg.checkpoint_repo_pattern = args.checkpoint_repo_pattern
        if args.no_checkpoint_push:
            train_cfg.push_checkpoints_to_hub = False
        if args.no_lmms_eval:
            train_cfg.use_lmms_eval = False
        if args.relevance_min_rating is not None:
            train_cfg.relevance_min_rating = args.relevance_min_rating
        if args.image_correspondence_min_rating is not None:
            train_cfg.image_correspondence_min_rating = args.image_correspondence_min_rating
        if args.visual_dependency_min_rating is not None:
            train_cfg.visual_dependency_min_rating = args.visual_dependency_min_rating
        if args.formatting_min_rating is not None:
            train_cfg.formatting_min_rating = args.formatting_min_rating
        if args.use_packing is not None:
            train_cfg.use_packing = args.use_packing

    if args.no_log_wandb is True:
        train_cfg.log_wandb = False
    if args.vlm_checkpoint_path is not None:
        vlm_cfg.vlm_checkpoint_path = args.vlm_checkpoint_path
    if args.left_tower_mask_mode is not None:
        vlm_cfg.left_tower_mask_mode = args.left_tower_mask_mode
    if args.momh_enabled is not None:
        vlm_cfg.momh_enabled = args.momh_enabled
    if args.resume_from_vlm_checkpoint:
        train_cfg.resume_from_vlm_checkpoint = True
    if args.checkpoint_interval is not None:
        train_cfg.checkpoint_interval = args.checkpoint_interval
    if args.checkpoint_repo_pattern is not None:
        train_cfg.checkpoint_repo_pattern = args.checkpoint_repo_pattern
    if args.no_checkpoint_push:
        train_cfg.push_checkpoints_to_hub = False
    if args.no_lmms_eval:
        train_cfg.use_lmms_eval = False
    if args.use_packing is not None:
        train_cfg.use_packing = args.use_packing

    model_mode = config_mode if config_mode is not None else "nanovlm"
    if args.dualtower:
        model_mode = "dualtower"
    if args.nanovlm:
        model_mode = "nanovlm"

    if train_cfg.resume_from_vlm_checkpoint and vlm_cfg.vlm_checkpoint_path is not None:
        # When resuming a full VLM, we don't need to load individual backbone weights from original sources.
        vlm_cfg.vlm_load_backbone_weights = False

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        init_dist()
        PG_CPU = dist.new_group(backend="gloo")   # host‑RAM, zero GPU allocations

    if is_master():
        log_info(ctext("--- Mode ---", "cyan", attrs=("bold",)))
        print(ctext(model_mode, "cyan", attrs=("bold",)))
        log_info(ctext("--- VLM Config ---", "cyan", attrs=("bold",)))
        print(ctext(vlm_cfg, "white"))
        log_info(ctext("--- Train Config ---", "cyan", attrs=("bold",)))
        print(ctext(train_cfg, "white"))

    train(train_cfg, vlm_cfg, model_mode=model_mode)

    if is_dist():
        destroy_dist()

if __name__ == "__main__":
    main()
