from __future__ import annotations

from pathlib import Path
import yaml


def load_yaml_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"YAML config not found: {path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping/object at top level: {path}")
    return data


def apply_object_overrides(
    obj,
    overrides: dict,
    *,
    object_name: str,
    tuple_fields: set[str] | None = None,
) -> None:
    tuple_fields = tuple_fields or set()
    for key, value in overrides.items():
        if not hasattr(obj, key):
            if object_name == "VLMConfig" and key == "right_attn_gate_init_logit":
                print(
                    "Warning: Ignoring deprecated VLMConfig field `right_attn_gate_init_logit`; "
                    "projection-split gating uses default Linear initialization."
                )
                continue
            raise ValueError(f"Unknown {object_name} field in config: {key}")
        if key in tuple_fields and isinstance(value, list):
            value = tuple(value)
        setattr(obj, key, value)
