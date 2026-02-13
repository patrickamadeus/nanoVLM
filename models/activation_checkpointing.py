import functools
from collections.abc import Callable
from typing import Any

import torch
import torch.utils.checkpoint as checkpoint_utils


_SUPPORTED_AC_MODES = {"regular", "selective"}
_SELECTIVE_CONTEXT_FN = None


def normalize_activation_checkpointing_mode(mode: str | None) -> str:
    normalized_mode = "regular" if mode is None else str(mode).strip().lower()
    if normalized_mode not in _SUPPORTED_AC_MODES:
        raise ValueError(
            f"Unsupported activation checkpointing mode '{mode}'. "
            f"Supported modes: {sorted(_SUPPORTED_AC_MODES)}."
        )
    return normalized_mode


def _resolve_aten_op(op_path: str):
    node = torch.ops.aten
    for part in op_path.split("."):
        node = getattr(node, part, None)
        if node is None:
            return None
    return node


def _build_selective_context_fn():
    create_contexts = getattr(checkpoint_utils, "create_selective_checkpoint_contexts", None)
    if create_contexts is None:
        raise RuntimeError(
            "Selective activation checkpointing is unavailable in this PyTorch build. "
            "Use `activation_checkpointing_mode: regular` or upgrade PyTorch."
        )

    save_ops = [
        op
        for op in (
            _resolve_aten_op("mm.default"),
            _resolve_aten_op("bmm.default"),
            _resolve_aten_op("addmm.default"),
            _resolve_aten_op("_scaled_dot_product_flash_attention.default"),
            _resolve_aten_op("_scaled_dot_product_efficient_attention.default"),
            _resolve_aten_op("_scaled_dot_product_cudnn_attention.default"),
        )
        if op is not None
    ]
    if not save_ops:
        raise RuntimeError(
            "Selective activation checkpointing requested, but no supported ATen ops "
            "were resolved for the save policy."
        )

    return functools.partial(
        create_contexts,
        save_ops,
        allow_cache_entry_mutation=True,
    )


def _get_selective_context_fn():
    global _SELECTIVE_CONTEXT_FN
    if _SELECTIVE_CONTEXT_FN is None:
        _SELECTIVE_CONTEXT_FN = _build_selective_context_fn()
    return _SELECTIVE_CONTEXT_FN


def run_activation_checkpoint(
    run_fn: Callable[..., Any],
    *args,
    mode: str | None,
):
    normalized_mode = normalize_activation_checkpointing_mode(mode)
    if normalized_mode == "regular":
        return checkpoint_utils.checkpoint(run_fn, *args, use_reentrant=False)

    return checkpoint_utils.checkpoint(
        run_fn,
        *args,
        use_reentrant=False,
        context_fn=_get_selective_context_fn(),
    )
