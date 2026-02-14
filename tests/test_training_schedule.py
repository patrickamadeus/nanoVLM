from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.config import TrainConfig
from train import _advance_token_trigger, _validate_training_schedule_config


def test_validate_training_schedule_defaults_to_step_units():
    cfg = TrainConfig()
    out = _validate_training_schedule_config(cfg)
    assert out["stop_unit"] == "steps"
    assert out["eval_unit"] == "steps"
    assert out["checkpoint_unit"] == "steps"


def test_validate_training_schedule_requires_token_budget_when_stop_by_tokens():
    cfg = TrainConfig(stop_unit="tokens", max_training_tokens=None)
    with pytest.raises(ValueError, match="max_training_tokens must be > 0"):
        _validate_training_schedule_config(cfg)


def test_validate_training_schedule_requires_token_intervals_when_token_units():
    cfg = TrainConfig(
        eval_unit="tokens",
        eval_interval_tokens=1000,
        checkpoint_unit="tokens",
        checkpoint_interval_tokens=None,
    )
    with pytest.raises(ValueError, match="checkpoint_interval_tokens must be > 0"):
        _validate_training_schedule_config(cfg)


def test_advance_token_trigger_handles_threshold_crossing_and_catchup():
    triggered, next_threshold = _advance_token_trigger(global_tokens=950, next_trigger_tokens=1000, interval_tokens=1000)
    assert triggered is False
    assert next_threshold == 1000

    triggered, next_threshold = _advance_token_trigger(global_tokens=1000, next_trigger_tokens=1000, interval_tokens=1000)
    assert triggered is True
    assert next_threshold == 2000

    triggered, next_threshold = _advance_token_trigger(global_tokens=3100, next_trigger_tokens=2000, interval_tokens=1000)
    assert triggered is True
    assert next_threshold == 4000


def test_validate_training_schedule_rejects_non_positive_checkpoint_retention():
    cfg = TrainConfig(keep_last_n_checkpoints=0)
    with pytest.raises(ValueError, match="keep_last_n_checkpoints must be > 0"):
        _validate_training_schedule_config(cfg)


def test_validate_training_schedule_rejects_mixed_init_and_continue_modes():
    cfg = TrainConfig(
        resume_from_vlm_checkpoint=True,
        continue_from_checkpoint="checkpoints/step-00000010-tokens-000000001024",
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _validate_training_schedule_config(cfg)
