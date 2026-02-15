#!/usr/bin/env python3
"""
Inspect response-only supervision masks on packed nanoVLM batches.

This script intentionally keeps the flow simple:
1) load model (GPU-only)
2) load dataloader from training config (including packing settings)
3) inspect one packed batch and print which tokens are supervised by labels != -100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models.config as config
from data.processors import get_tokenizer
from models.dual_tower.dual_tower import DualTowerVLM
from models.language_model import LanguageModel
from models.vision_language_model import VisionLanguageModel
from train import get_dataloaders
from train_utils.config_loader import (
    apply_dataclass_overrides,
    load_yaml_mapping,
    validate_allowed_keys,
)


def _segment_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be [B, T], got {tuple(attention_mask.shape)}")
    valid = attention_mask.to(torch.bool)
    prev_valid = F.pad(valid[:, :-1], (1, 0), value=False)
    seg_start = valid & (~prev_valid)
    segment_id = seg_start.to(torch.int64).cumsum(dim=1) - 1
    segment_id = segment_id.masked_fill(~valid, -1)
    return segment_id


def _load_config(config_path: Path) -> tuple[str, config.VLMConfig, config.TrainConfig]:
    cfg_data = load_yaml_mapping(config_path)
    validate_allowed_keys(cfg_data, {"mode", "vlm", "train"}, f"training config '{config_path}'")

    mode = str(cfg_data.get("mode", "nanovlm")).strip().lower()
    if mode not in {"nanovlm", "dualtower"}:
        raise ValueError(f"Unsupported mode in config: {mode!r}. Expected 'nanovlm' or 'dualtower'.")

    vlm_cfg = config.VLMConfig()
    train_cfg = config.TrainConfig()
    if "vlm" in cfg_data:
        apply_dataclass_overrides(vlm_cfg, cfg_data["vlm"], "vlm")
    if "train" in cfg_data:
        apply_dataclass_overrides(train_cfg, cfg_data["train"], "train")

    return mode, vlm_cfg, train_cfg


def _load_model(*, mode: str, vlm_cfg: config.VLMConfig, train_cfg: config.TrainConfig, device: torch.device):
    if mode == "nanovlm":
        if train_cfg.resume_from_vlm_checkpoint:
            if not vlm_cfg.vlm_checkpoint_path:
                raise ValueError("resume_from_vlm_checkpoint=True requires vlm_checkpoint_path.")
            model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
        else:
            model = VisionLanguageModel(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)
    else:
        if train_cfg.resume_from_vlm_checkpoint:
            if not vlm_cfg.vlm_checkpoint_path:
                raise ValueError("resume_from_vlm_checkpoint=True requires vlm_checkpoint_path.")
            vlm_model = VisionLanguageModel.from_pretrained(vlm_cfg.vlm_checkpoint_path)
            model = DualTowerVLM(vlm_cfg, load_backbone=False)
            model.left_tower.vision_encoder.load_state_dict(vlm_model.vision_encoder.state_dict())
            model.left_tower.MP.load_state_dict(vlm_model.MP.state_dict())
            model.left_tower.decoder.load_state_dict(vlm_model.decoder.state_dict())
            right_lm = LanguageModel.from_pretrained(vlm_cfg)
            model.right_tower.load_state_dict(right_lm.state_dict())
            del right_lm
            del vlm_model
        else:
            model = DualTowerVLM(vlm_cfg, load_backbone=vlm_cfg.vlm_load_backbone_weights)

    model = model.to(device)
    model.eval()
    return model


def _token_to_text(tokenizer, token_id: int) -> str:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    if token is None:
        return "<unk>"
    return token.replace("\n", "\\n")


def _segment_spans(valid_row: torch.Tensor) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    in_span = False
    start = -1
    for idx, flag in enumerate(valid_row.tolist()):
        if flag and not in_span:
            in_span = True
            start = idx
        elif (not flag) and in_span:
            in_span = False
            spans.append((start, idx - 1))
    if in_span:
        spans.append((start, len(valid_row) - 1))
    return spans


def _short_text(tokenizer, ids: torch.Tensor, limit: int = 220) -> str:
    if ids.numel() == 0:
        return ""
    text = tokenizer.decode(ids.tolist(), skip_special_tokens=False).replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _inspect_batch(
    *,
    batch: dict[str, Any],
    tokenizer,
    row_index: int,
    max_token_preview: int,
) -> dict[str, Any]:
    input_ids = batch["input_ids"].to(torch.long)
    labels = batch["labels"].to(torch.long)
    attention_mask = batch["attention_mask"].to(torch.bool)

    if input_ids.ndim != 2 or labels.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("Expected batch tensors input_ids/labels/attention_mask with shape [B, T].")
    if not (input_ids.shape == labels.shape == attention_mask.shape):
        raise ValueError(
            f"Batch tensor shapes must match, got input_ids={tuple(input_ids.shape)}, "
            f"labels={tuple(labels.shape)}, attention_mask={tuple(attention_mask.shape)}"
        )

    batch_size, seq_len = input_ids.shape
    if row_index < 0 or row_index >= batch_size:
        raise ValueError(f"row_index must be in [0, {batch_size - 1}], got {row_index}.")

    target_mask = labels != -100
    response_masked_content = attention_mask & (~target_mask)
    segment_id = _segment_ids_from_attention_mask(attention_mask)

    token_capacity = int(attention_mask.numel())
    content_tokens = int(attention_mask.sum().item())
    target_tokens = int(target_mask.sum().item())
    masked_content_tokens = int(response_masked_content.sum().item())

    row_valid = attention_mask[row_index]
    row_target = target_mask[row_index]
    row_masked = response_masked_content[row_index]
    row_input_ids = input_ids[row_index]
    row_segment_id = segment_id[row_index]
    row_spans = _segment_spans(row_valid)

    segment_summaries = []
    for seg_idx, (start, end) in enumerate(row_spans):
        seg_slice = slice(start, end + 1)
        seg_content = int(row_valid[seg_slice].sum().item())
        seg_target = int(row_target[seg_slice].sum().item())
        seg_masked = int(row_masked[seg_slice].sum().item())
        segment_summaries.append(
            {
                "segment_index": seg_idx,
                "start": int(start),
                "end": int(end),
                "content_tokens": seg_content,
                "target_tokens": seg_target,
                "masked_content_tokens": seg_masked,
            }
        )

    non_pad_positions = torch.nonzero(row_valid, as_tuple=False).flatten().tolist()
    preview_positions = non_pad_positions[:max_token_preview]
    token_preview = []
    for pos in preview_positions:
        token_preview.append(
            {
                "pos": int(pos),
                "segment_id": int(row_segment_id[pos].item()),
                "token_id": int(row_input_ids[pos].item()),
                "token": _token_to_text(tokenizer, int(row_input_ids[pos].item())),
                "is_target": bool(row_target[pos].item()),
                "masked_by_response_only": bool(row_masked[pos].item()),
            }
        )

    row_target_ids = row_input_ids[row_target]
    row_masked_ids = row_input_ids[row_masked]
    row_content_ids = row_input_ids[row_valid]

    return {
        "shape": {"batch_size": int(batch_size), "seq_len": int(seq_len)},
        "batch_summary": {
            "token_capacity": token_capacity,
            "content_tokens": content_tokens,
            "target_tokens": target_tokens,
            "masked_content_tokens": masked_content_tokens,
            "content_ratio": (content_tokens / token_capacity) if token_capacity > 0 else 0.0,
            "target_ratio": (target_tokens / token_capacity) if token_capacity > 0 else 0.0,
            "target_among_content_ratio": (target_tokens / content_tokens) if content_tokens > 0 else 0.0,
        },
        "row_summary": {
            "row_index": int(row_index),
            "num_segments": len(row_spans),
            "content_tokens": int(row_valid.sum().item()),
            "target_tokens": int(row_target.sum().item()),
            "masked_content_tokens": int(row_masked.sum().item()),
            "segment_summaries": segment_summaries,
            "content_text_preview": _short_text(tokenizer, row_content_ids),
            "target_text_preview": _short_text(tokenizer, row_target_ids),
            "masked_content_text_preview": _short_text(tokenizer, row_masked_ids),
        },
        "token_preview": token_preview,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load model + packed dataloader and inspect response-only label masking."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.current.momh.compile-selective.yaml"),
        help="Training config YAML with top-level keys: mode, vlm, train.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Which loader split to inspect.",
    )
    parser.add_argument(
        "--row-index",
        type=int,
        default=0,
        help="Batch row index to print detailed token-level preview for.",
    )
    parser.add_argument(
        "--max-token-preview",
        type=int,
        default=180,
        help="Maximum non-pad tokens to print from the selected row.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write full inspection JSON.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script. Refusing to run without GPU.")
    device = torch.device("cuda")

    mode, vlm_cfg, train_cfg = _load_config(args.config)
    print(f"[INFO] Loaded config: {args.config}")
    print(f"[INFO] mode={mode} use_packing={train_cfg.use_packing} batch_size={train_cfg.batch_size}")

    # 1) Load model
    model = _load_model(mode=mode, vlm_cfg=vlm_cfg, train_cfg=train_cfg, device=device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model loaded on {device}: {type(model).__name__} ({total_params:,} params)")

    # 2) Load dataset + packing pipeline through training dataloader path
    train_loader, val_loader, iter_train_loader, iter_val_loader = get_dataloaders(train_cfg, vlm_cfg)
    loader = iter_train_loader if args.split == "train" else iter_val_loader
    _ = train_loader if args.split == "train" else val_loader
    batch = next(loader)
    print(f"[INFO] Loaded one '{args.split}' batch from dataloader.")

    tokenizer = get_tokenizer(
        vlm_cfg.lm_tokenizer,
        vlm_cfg.vlm_extra_tokens,
        vlm_cfg.lm_chat_template,
        model_max_length=train_cfg.max_sample_length,
    )
    report = _inspect_batch(
        batch=batch,
        tokenizer=tokenizer,
        row_index=args.row_index,
        max_token_preview=args.max_token_preview,
    )

    print("\n=== Batch Summary ===")
    print(json.dumps(report["batch_summary"], indent=2))
    print("\n=== Row Summary ===")
    print(json.dumps(report["row_summary"], indent=2))
    print("\n=== Token Preview (selected row, non-pad tokens) ===")
    for token_item in report["token_preview"]:
        state = "TGT" if token_item["is_target"] else "MASKED"
        print(
            f"pos={token_item['pos']:4d} seg={token_item['segment_id']:2d} "
            f"id={token_item['token_id']:6d} {state:6s} tok={token_item['token']}"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\n[INFO] Wrote JSON report to: {args.output_json}")


if __name__ == "__main__":
    main()
