import random
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy
import torch


CHECKPOINT_STATE_FILENAME = "train_state.pt"
CHECKPOINT_PREFIX = "step-"


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": numpy.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch"}
    missing = required.difference(state.keys())
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"RNG state is missing required keys: {missing_text}")

    random.setstate(state["python"])
    numpy.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])

    cuda_state = state.get("torch_cuda")
    if cuda_state is None:
        return
    if not torch.cuda.is_available():
        raise ValueError("Checkpoint contains CUDA RNG state, but CUDA is not available.")
    torch.cuda.set_rng_state_all(cuda_state)


def _serialize_cfg(cfg: Any) -> Any:
    if cfg is None:
        return None
    if is_dataclass(cfg):
        return asdict(cfg)
    return cfg


def _checkpoint_dir_name(step: int, consumed_tokens: int) -> str:
    return f"{CHECKPOINT_PREFIX}{step:08d}-tokens-{consumed_tokens:012d}"


def save_full_checkpoint(
    *,
    model,
    optimizer,
    checkpoint_root: str,
    run_name: str,
    step: int,
    consumed_tokens: int,
    epoch: int,
    microbatches_seen_in_epoch: int,
    next_eval_tokens: int | None,
    next_checkpoint_tokens: int | None,
    best_val_loss: float,
    best_val_step: int | None,
    best_checkpoint_repo_id: str | None,
    schedule_units: dict[str, str],
    train_cfg,
    vlm_cfg,
) -> str:
    if step <= 0:
        raise ValueError(f"step must be > 0 for checkpointing, got {step}.")

    checkpoint_root_path = Path(checkpoint_root).expanduser().resolve()
    run_dir = checkpoint_root_path / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / _checkpoint_dir_name(step=step, consumed_tokens=consumed_tokens)
    checkpoint_dir.mkdir(parents=False, exist_ok=False)

    model.save_pretrained(str(checkpoint_dir))
    state_payload = {
        "checkpoint_format_version": 1,
        "run_name": run_name,
        "step": int(step),
        "consumed_tokens": int(consumed_tokens),
        "epoch": int(epoch),
        "microbatches_seen_in_epoch": int(microbatches_seen_in_epoch),
        "next_eval_tokens": next_eval_tokens,
        "next_checkpoint_tokens": next_checkpoint_tokens,
        "best_val_loss": float(best_val_loss),
        "best_val_step": best_val_step,
        "best_checkpoint_repo_id": best_checkpoint_repo_id,
        "schedule_units": dict(schedule_units),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        "train_cfg": _serialize_cfg(train_cfg),
        "vlm_cfg": _serialize_cfg(vlm_cfg),
    }
    torch.save(state_payload, checkpoint_dir / CHECKPOINT_STATE_FILENAME)
    return str(checkpoint_dir)


def load_full_checkpoint_state(checkpoint_path: str, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        raise ValueError(f"Checkpoint path must be a directory, got {checkpoint_dir}")

    model_weights_path = checkpoint_dir / "model.safetensors"
    model_config_path = checkpoint_dir / "config.json"
    train_state_path = checkpoint_dir / CHECKPOINT_STATE_FILENAME
    missing = [path.name for path in (model_weights_path, model_config_path, train_state_path) if not path.exists()]
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise FileNotFoundError(
            f"Checkpoint at {checkpoint_dir} is incomplete. Missing files: {missing_text}"
        )

    payload = torch.load(train_state_path, map_location=map_location, weights_only=False)
    required = {
        "optimizer_state_dict",
        "step",
        "consumed_tokens",
        "epoch",
        "rng_state",
        "schedule_units",
    }
    missing_keys = required.difference(payload.keys())
    if missing_keys:
        missing_text = ", ".join(sorted(missing_keys))
        raise ValueError(
            f"Checkpoint state at {checkpoint_dir} is missing required keys: {missing_text}"
        )
    return payload


def _checkpoint_sort_key(path: Path) -> tuple[int, int]:
    name = path.name
    if not name.startswith(CHECKPOINT_PREFIX):
        return (-1, -1)
    try:
        step_part, token_part = name[len(CHECKPOINT_PREFIX):].split("-tokens-")
        step = int(step_part)
        tokens = int(token_part)
    except Exception as exc:
        raise ValueError(f"Unexpected checkpoint directory format: {path.name}") from exc
    return (step, tokens)


def prune_old_checkpoints(*, checkpoint_root: str, run_name: str, keep_last_n: int) -> list[str]:
    if keep_last_n <= 0:
        raise ValueError(f"keep_last_n must be > 0, got {keep_last_n}.")

    run_dir = Path(checkpoint_root).expanduser().resolve() / run_name
    if not run_dir.exists():
        return []

    checkpoints = [path for path in run_dir.iterdir() if path.is_dir() and path.name.startswith(CHECKPOINT_PREFIX)]
    checkpoints.sort(key=_checkpoint_sort_key)
    if len(checkpoints) <= keep_last_n:
        return []

    to_remove = checkpoints[:-keep_last_n]
    removed = []
    for path in to_remove:
        shutil.rmtree(path)
        removed.append(str(path))
    return removed


def move_optimizer_state_to_device(optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
