from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.activation_checkpointing import (  # noqa: E402
    normalize_activation_checkpointing_mode,
    run_activation_checkpoint,
)


def test_normalize_activation_checkpointing_mode_defaults_to_regular():
    assert normalize_activation_checkpointing_mode(None) == "regular"


def test_normalize_activation_checkpointing_mode_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported activation checkpointing mode"):
        normalize_activation_checkpointing_mode("fast")


def test_run_activation_checkpoint_regular_backward():
    x = torch.randn(4, 5, requires_grad=True)

    def run_fn(x_in):
        return (x_in * x_in).sum()

    loss = run_activation_checkpoint(run_fn, x, mode="regular")
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


def test_run_activation_checkpoint_selective_backward():
    x = torch.randn(4, 5, requires_grad=True)
    w = torch.randn(5, 3, requires_grad=True)

    def run_fn(x_in, w_in):
        return (x_in @ w_in).relu()

    out = run_activation_checkpoint(run_fn, x, w, mode="selective")
    out.sum().backward()
    assert x.grad is not None
    assert w.grad is not None
