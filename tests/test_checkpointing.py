import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_utils.checkpointing import (
    load_full_checkpoint_state,
    prune_old_checkpoints,
    save_full_checkpoint,
)


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)

    def save_pretrained(self, save_directory: str) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        (save_path / "model.safetensors").write_bytes(b"dummy")
        (save_path / "config.json").write_text(json.dumps({"model": "dummy"}), encoding="utf-8")


def _build_optimizer(model: nn.Module):
    return torch.optim.AdamW([{"name": "proj", "params": list(model.parameters()), "lr": 1e-3}])


def test_save_and_load_full_checkpoint_roundtrip(tmp_path: Path):
    model = _DummyModel()
    optimizer = _build_optimizer(model)

    checkpoint_path = save_full_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_root=str(tmp_path),
        run_name="unit-run",
        step=5,
        consumed_tokens=320,
        epoch=2,
        microbatches_seen_in_epoch=15,
        next_eval_tokens=1000,
        next_checkpoint_tokens=1500,
        best_val_loss=1.25,
        best_val_step=4,
        best_checkpoint_repo_id="repo/step-4",
        schedule_units={"stop_unit": "steps", "eval_unit": "steps", "checkpoint_unit": "steps"},
        train_cfg={"batch_size": 1},
        vlm_cfg={"lm_max_length": 32},
    )

    assert Path(checkpoint_path).is_dir()
    state = load_full_checkpoint_state(checkpoint_path)
    assert state["step"] == 5
    assert state["consumed_tokens"] == 320
    assert state["epoch"] == 2
    assert state["microbatches_seen_in_epoch"] == 15
    assert "optimizer_state_dict" in state
    assert "rng_state" in state


def test_prune_old_checkpoints_keeps_latest_n(tmp_path: Path):
    model = _DummyModel()
    optimizer = _build_optimizer(model)
    run_name = "retain-run"

    save_full_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_root=str(tmp_path),
        run_name=run_name,
        step=1,
        consumed_tokens=100,
        epoch=1,
        microbatches_seen_in_epoch=4,
        next_eval_tokens=None,
        next_checkpoint_tokens=None,
        best_val_loss=2.0,
        best_val_step=None,
        best_checkpoint_repo_id=None,
        schedule_units={"stop_unit": "steps", "eval_unit": "steps", "checkpoint_unit": "steps"},
        train_cfg={},
        vlm_cfg={},
    )
    save_full_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_root=str(tmp_path),
        run_name=run_name,
        step=2,
        consumed_tokens=200,
        epoch=1,
        microbatches_seen_in_epoch=8,
        next_eval_tokens=None,
        next_checkpoint_tokens=None,
        best_val_loss=1.9,
        best_val_step=2,
        best_checkpoint_repo_id=None,
        schedule_units={"stop_unit": "steps", "eval_unit": "steps", "checkpoint_unit": "steps"},
        train_cfg={},
        vlm_cfg={},
    )
    save_full_checkpoint(
        model=model,
        optimizer=optimizer,
        checkpoint_root=str(tmp_path),
        run_name=run_name,
        step=3,
        consumed_tokens=300,
        epoch=1,
        microbatches_seen_in_epoch=12,
        next_eval_tokens=None,
        next_checkpoint_tokens=None,
        best_val_loss=1.8,
        best_val_step=3,
        best_checkpoint_repo_id=None,
        schedule_units={"stop_unit": "steps", "eval_unit": "steps", "checkpoint_unit": "steps"},
        train_cfg={},
        vlm_cfg={},
    )

    removed = prune_old_checkpoints(
        checkpoint_root=str(tmp_path),
        run_name=run_name,
        keep_last_n=2,
    )
    assert len(removed) == 1

    run_dir = tmp_path / run_name
    remaining = sorted(path.name for path in run_dir.iterdir() if path.is_dir())
    assert remaining == [
        "step-00000002-tokens-000000000200",
        "step-00000003-tokens-000000000300",
    ]


def test_prune_old_checkpoints_rejects_non_positive_limit(tmp_path: Path):
    with pytest.raises(ValueError, match="keep_last_n must be > 0"):
        prune_old_checkpoints(checkpoint_root=str(tmp_path), run_name="x", keep_last_n=0)
