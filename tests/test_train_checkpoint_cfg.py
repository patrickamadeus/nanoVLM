from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.config import VLMConfig
from train import _apply_checkpoint_cfg_overrides


def test_apply_checkpoint_cfg_overrides_preserves_runtime_behavior_fields():
    loaded_cfg = VLMConfig(
        lm_max_length=8192,
        lm_max_position_embeddings=8192,
        momh_enabled=False,
        momh_head_pct_vision=0.1,
        momh_head_pct_text=0.1,
        activation_checkpointing=False,
        activation_checkpointing_mode="regular",
    )
    requested_cfg = VLMConfig(
        lm_max_length=4096,
        lm_max_position_embeddings=4096,
        momh_enabled=True,
        momh_head_pct_vision=0.2,
        momh_head_pct_text=0.3,
        activation_checkpointing=True,
        activation_checkpointing_mode="selective",
    )

    out_cfg = _apply_checkpoint_cfg_overrides(loaded_cfg, requested_cfg)

    assert out_cfg.lm_max_length == 4096
    assert out_cfg.lm_max_position_embeddings == 4096
    assert out_cfg.momh_enabled is True
    assert out_cfg.momh_head_pct_vision == 0.2
    assert out_cfg.momh_head_pct_text == 0.3
    assert out_cfg.activation_checkpointing is True
    assert out_cfg.activation_checkpointing_mode == "selective"
