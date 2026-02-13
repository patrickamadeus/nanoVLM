#!/usr/bin/env python3
"""
Validate MoMH masking invariants.

This script checks whether the MoMH mask rules match expectations:
- V-heads: vision -> vision only
- T-heads: text -> text only and causal
- VT-heads: full vision + causal text
- no attention to/from padding tokens

It also reports packed-segment cross-sample leakage by deriving segment IDs from
`attention_mask` gaps (0-regions) and counting allowed attention across segments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models.config as config
from data.processors import get_tokenizer
from models.momh_attention import generate_momh_mask_mod_from_modality
from train_utils.config_loader import (
    apply_dataclass_overrides,
    load_yaml_mapping,
    validate_allowed_keys,
)


def _segment_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Convert attention_mask [B, T] into segment IDs [B, T].

    A segment is a contiguous run where attention_mask == 1.
    Padding/separator positions get segment_id = -1.
    """
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be [B, T], got {tuple(attention_mask.shape)}")
    valid = attention_mask.to(torch.bool)
    prev_valid = torch.nn.functional.pad(valid[:, :-1], (1, 0), value=False)
    seg_start = valid & (~prev_valid)
    segment_id = seg_start.to(torch.int64).cumsum(dim=1) - 1
    segment_id = segment_id.masked_fill(~valid, -1)
    return segment_id


def _select_positions(length: int, max_qkv: int) -> torch.Tensor:
    if length <= 0:
        raise ValueError("length must be > 0")
    if max_qkv <= 0:
        raise ValueError("max_qkv must be > 0")
    if length <= max_qkv:
        return torch.arange(length, dtype=torch.int64)
    # Evenly sample positions across the sequence.
    return torch.linspace(0, length - 1, steps=max_qkv).round().to(torch.int64).unique()


def _run_momh_checks(
    *,
    is_vision: torch.Tensor,
    attention_mask: torch.Tensor,
    n_heads: int,
    pct_v: float,
    pct_t: float,
    max_qkv: int,
    q_offset: int = 0,
) -> dict:
    if is_vision.shape != attention_mask.shape:
        raise ValueError(
            f"is_vision and attention_mask must have same shape, got "
            f"{tuple(is_vision.shape)} vs {tuple(attention_mask.shape)}"
        )
    if is_vision.ndim != 2:
        raise ValueError(f"is_vision must be [B, T], got {tuple(is_vision.shape)}")
    if n_heads <= 0:
        raise ValueError("n_heads must be > 0")

    B, T = attention_mask.shape
    q_idx = _select_positions(T, max_qkv)
    kv_idx = _select_positions(T, max_qkv)
    q_len = int(q_idx.numel())
    kv_len = int(kv_idx.numel())

    mask_mod = generate_momh_mask_mod_from_modality(
        n_heads,
        is_vision=is_vision.to(torch.bool),
        attention_mask=attention_mask.to(torch.bool),
        q_offset=q_offset,
        pct_v=pct_v,
        pct_t=pct_t,
    )

    b = torch.arange(B, dtype=torch.int64).view(B, 1, 1, 1)
    h = torch.arange(n_heads, dtype=torch.int64).view(1, n_heads, 1, 1)
    q = q_idx.view(1, 1, q_len, 1)
    k = kv_idx.view(1, 1, 1, kv_len)
    allowed = mask_mod(b, h, q, k).to(torch.bool)  # [B, H, Q, K]

    attention_mask_bool = attention_mask.to(torch.bool)
    is_vision_bool = is_vision.to(torch.bool)
    segment_id = _segment_ids_from_attention_mask(attention_mask_bool)

    q_abs = q_idx + int(q_offset)
    k_abs = kv_idx

    q_content = attention_mask_bool[:, q_abs]  # [B, Q]
    k_content = attention_mask_bool[:, k_abs]  # [B, K]
    not_padding = q_content[:, None, :, None] & k_content[:, None, None, :]

    q_vis = is_vision_bool[:, q_abs]
    k_vis = is_vision_bool[:, k_abs]
    q_vis_b = q_vis[:, None, :, None]
    k_vis_b = k_vis[:, None, None, :]
    q_text_b = ~q_vis_b
    k_text_b = ~k_vis_b

    q_abs_b = q_abs[None, None, :, None]
    k_abs_b = k_abs[None, None, None, :]
    causal = q_abs_b >= k_abs_b

    seg_q = segment_id[:, q_abs]
    seg_k = segment_id[:, k_abs]
    same_segment = (
        (seg_q[:, None, :, None] == seg_k[:, None, None, :])
        & (seg_q[:, None, :, None] >= 0)
        & (seg_k[:, None, None, :] >= 0)
    )

    h_v = int(n_heads * pct_v)
    h_t = int(n_heads * pct_t)
    h_vt_start = h_v + h_t

    is_v_head = (torch.arange(n_heads) < h_v).view(1, n_heads, 1, 1)
    is_t_head = ((torch.arange(n_heads) >= h_v) & (torch.arange(n_heads) < h_vt_start)).view(1, n_heads, 1, 1)
    is_vt_head = (torch.arange(n_heads) >= h_vt_start).view(1, n_heads, 1, 1)

    expected_v = q_vis_b & k_vis_b & not_padding
    expected_t = q_text_b & k_text_b & causal & not_padding
    expected_vt = (k_vis_b | causal) & not_padding

    v_violation = (allowed & is_v_head) & (~expected_v)
    t_violation = (allowed & is_t_head) & (~expected_t)
    vt_violation = (allowed & is_vt_head) & (~expected_vt)
    padding_violation = allowed & (~not_padding)
    cross_segment_allowed = allowed & not_padding & (~same_segment)

    valid_tokens = int(attention_mask_bool.sum().item())
    vision_tokens = int((is_vision_bool & attention_mask_bool).sum().item())
    segments_per_row = (segment_id.max(dim=1).values + 1).clamp(min=0).to(torch.int64)

    return {
        "shape": {
            "batch_size": B,
            "seq_len": T,
            "sampled_q_len": q_len,
            "sampled_kv_len": kv_len,
        },
        "heads": {
            "n_heads": n_heads,
            "pct_v": pct_v,
            "pct_t": pct_t,
            "n_v_heads": h_v,
            "n_t_heads": h_t,
            "n_vt_heads": n_heads - h_vt_start,
        },
        "tokens": {
            "valid_tokens": valid_tokens,
            "vision_tokens": vision_tokens,
            "vision_ratio_among_valid": (vision_tokens / valid_tokens) if valid_tokens > 0 else 0.0,
            "segments_per_row": [int(x) for x in segments_per_row.tolist()],
        },
        "violations": {
            "padding_violation_count": int(padding_violation.sum().item()),
            "v_head_rule_violation_count": int(v_violation.sum().item()),
            "t_head_rule_violation_count": int(t_violation.sum().item()),
            "vt_head_rule_violation_count": int(vt_violation.sum().item()),
            "cross_segment_allowed_count": int(cross_segment_allowed.sum().item()),
        },
        "pass": {
            "padding_masking_ok": bool(int(padding_violation.sum().item()) == 0),
            "v_head_rules_ok": bool(int(v_violation.sum().item()) == 0),
            "t_head_rules_ok": bool(int(t_violation.sum().item()) == 0),
            "vt_head_rules_ok": bool(int(vt_violation.sum().item()) == 0),
            # Informational: currently expected to fail for packed rows unless same-segment masking is added.
            "segment_isolation_ok": bool(int(cross_segment_allowed.sum().item()) == 0),
        },
    }


def _load_batch_from_dataloader(config_path: Path) -> tuple[dict, config.VLMConfig]:
    # Import lazily to avoid training-stack import overhead in synthetic mode.
    from train import get_dataloaders

    cfg_data = load_yaml_mapping(config_path)
    validate_allowed_keys(cfg_data, {"mode", "vlm", "train"}, f"training config '{config_path}'")

    vlm_cfg = config.VLMConfig()
    train_cfg = config.TrainConfig()
    if "vlm" in cfg_data:
        apply_dataclass_overrides(vlm_cfg, cfg_data["vlm"], "vlm")
    if "train" in cfg_data:
        apply_dataclass_overrides(train_cfg, cfg_data["train"], "train")

    train_loader, _val_loader, iter_train_loader, _iter_val_loader = get_dataloaders(train_cfg, vlm_cfg)
    if len(train_loader) == 0:
        raise ValueError("Training dataloader is empty; cannot run MoMH mask checks.")
    batch = next(iter_train_loader)
    return batch, vlm_cfg


def _build_synthetic_batch() -> tuple[torch.Tensor, torch.Tensor]:
    # Two content segments separated by a gap (packing separator).
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]],
        dtype=torch.bool,
    )
    # Interleaved vision placeholders across both segments.
    is_vision = torch.tensor(
        [[1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0]],
        dtype=torch.bool,
    )
    return is_vision, attention_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Check MoMH masking invariants.")
    parser.add_argument(
        "--mode",
        choices=("synthetic", "dataloader"),
        default="synthetic",
        help="synthetic: fast no-data check; dataloader: validate on real training batch from config",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.small_debug.momh.yaml"),
        help="Config file used in dataloader mode.",
    )
    parser.add_argument(
        "--max-qkv",
        type=int,
        default=512,
        help="Maximum Q/KV positions sampled for checks (full if seq_len <= this).",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=8,
        help="Synthetic mode only: number of heads.",
    )
    parser.add_argument(
        "--pct-v",
        type=float,
        default=0.2,
        help="Synthetic mode only: fraction of V->V heads.",
    )
    parser.add_argument(
        "--pct-t",
        type=float,
        default=0.3,
        help="Synthetic mode only: fraction of T->T heads.",
    )
    parser.add_argument(
        "--require-segment-isolation",
        action="store_true",
        help="Exit non-zero when cross-segment allowed attention is detected.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save JSON report.",
    )
    args = parser.parse_args()

    if args.mode == "synthetic":
        is_vision, attention_mask = _build_synthetic_batch()
        n_heads = int(args.n_heads)
        pct_v = float(args.pct_v)
        pct_t = float(args.pct_t)
        source_meta = {
            "mode": "synthetic",
        }
    else:
        batch, vlm_cfg = _load_batch_from_dataloader(args.config)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"].to(torch.bool)
        tokenizer = get_tokenizer(vlm_cfg.lm_tokenizer, vlm_cfg.vlm_extra_tokens, vlm_cfg.lm_chat_template)
        image_token_id = int(tokenizer.image_token_id)
        is_vision = (input_ids == image_token_id)
        n_heads = int(vlm_cfg.lm_n_heads)
        pct_v = float(getattr(vlm_cfg, "momh_head_pct_vision", 0.2))
        pct_t = float(getattr(vlm_cfg, "momh_head_pct_text", 0.3))
        source_meta = {
            "mode": "dataloader",
            "config": str(args.config),
            "image_token_id": image_token_id,
            "momh_enabled": bool(getattr(vlm_cfg, "momh_enabled", False)),
        }

    report = _run_momh_checks(
        is_vision=is_vision,
        attention_mask=attention_mask,
        n_heads=n_heads,
        pct_v=pct_v,
        pct_t=pct_t,
        max_qkv=int(args.max_qkv),
    )
    output = {
        "source": source_meta,
        "report": report,
    }

    print(json.dumps(output, indent=2))
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    if args.require_segment_isolation and not report["pass"]["segment_isolation_ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
