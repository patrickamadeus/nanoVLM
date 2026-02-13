from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.config import TrainConfig, VLMConfig
from train_utils.config_loader import (
    ConfigError,
    apply_dataclass_overrides,
    load_named_section,
    load_yaml_mapping,
    validate_allowed_keys,
)


def test_apply_dataclass_overrides_updates_and_normalizes_tuple():
    train_cfg = TrainConfig()
    apply_dataclass_overrides(
        train_cfg,
        {"batch_size": 8, "train_dataset_name": ["default", "allava_laion"]},
        "train",
    )
    assert train_cfg.batch_size == 8
    assert train_cfg.train_dataset_name == ("default", "allava_laion")


def test_apply_dataclass_overrides_rejects_unknown_key():
    vlm_cfg = VLMConfig()
    with pytest.raises(ConfigError, match="Unknown keys"):
        apply_dataclass_overrides(vlm_cfg, {"not_a_field": 1}, "vlm")


def test_apply_dataclass_overrides_rejects_wrong_type():
    train_cfg = TrainConfig()
    with pytest.raises(ConfigError, match="expects int"):
        apply_dataclass_overrides(train_cfg, {"batch_size": "8"}, "train")


def test_apply_dataclass_overrides_allows_none_when_default_is_none():
    train_cfg = TrainConfig()
    apply_dataclass_overrides(train_cfg, {"lmms_eval_limit": None}, "train")
    assert train_cfg.lmms_eval_limit is None


def test_validate_allowed_keys_rejects_unknown():
    with pytest.raises(ConfigError, match="Unknown keys"):
        validate_allowed_keys({"mode": "nanovlm", "extra": 1}, {"mode"}, "test-config")


def test_load_yaml_mapping_rejects_top_level_list(tmp_path):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a YAML mapping"):
        load_yaml_mapping(config_path)


def test_load_named_section_requires_mapping(tmp_path):
    config_path = tmp_path / "eval.yaml"
    config_path.write_text("evaluation: 123\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_named_section(config_path, "evaluation")


def test_load_named_section_rejects_extra_top_level_keys(tmp_path):
    config_path = tmp_path / "eval.yaml"
    config_path.write_text("evaluation:\n  mode: nanovlm\nextra: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Unknown keys"):
        load_named_section(config_path, "evaluation")


def test_load_named_section_accepts_flat_mapping_when_enabled(tmp_path):
    config_path = tmp_path / "eval.yaml"
    config_path.write_text("mode: nanovlm\ntasks: mmstar\n", encoding="utf-8")
    section = load_named_section(config_path, "evaluation", allow_flat_mapping=True)
    assert section == {"mode": "nanovlm", "tasks": "mmstar"}
