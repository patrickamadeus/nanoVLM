from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
import types
from typing import Any, Union, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised when an experiment config is invalid."""


_UNION_ORIGINS = {Union, types.UnionType}


def load_yaml_mapping(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file does not exist: {path}")
    if not path.is_file():
        raise ConfigError(f"Config path is not a file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"Config at {path} must be a YAML mapping at the top level, got {type(data).__name__}.")
    return data


def validate_allowed_keys(config: dict[str, Any], allowed_keys: set[str], context: str) -> None:
    unknown = sorted(set(config.keys()) - allowed_keys)
    if unknown:
        allowed_display = ", ".join(sorted(allowed_keys))
        unknown_display = ", ".join(unknown)
        raise ConfigError(
            f"Unknown keys in {context}: {unknown_display}. "
            f"Allowed keys: {allowed_display}."
        )


def load_named_section(config_path: str | Path, section_name: str, allow_flat_mapping: bool = False) -> dict[str, Any]:
    config = load_yaml_mapping(config_path)
    if section_name in config:
        validate_allowed_keys(config, {section_name}, f"{config_path}")
        section = config[section_name]
        if not isinstance(section, dict):
            raise ConfigError(
                f"Section '{section_name}' in {config_path} must be a mapping, got {type(section).__name__}."
            )
        return section

    if allow_flat_mapping:
        return config

    raise ConfigError(
        f"Config {config_path} must contain a top-level '{section_name}' mapping."
    )


def _annotation_name(annotation: Any) -> str:
    if isinstance(annotation, type):
        return annotation.__name__
    return str(annotation)


def _value_matches_primitive(value: Any, expected_type: type) -> bool:
    if expected_type is bool:
        return isinstance(value, bool)
    if expected_type is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type is float:
        return (isinstance(value, float) or isinstance(value, int)) and not isinstance(value, bool)
    return isinstance(value, expected_type)


def _normalize_value(value: Any, annotation: Any, key_path: str) -> Any:
    if annotation in (Any, object):
        return value

    if annotation is type(None):
        if value is None:
            return None
        raise ConfigError(f"Field '{key_path}' expects None, got {type(value).__name__}.")

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin in _UNION_ORIGINS:
        for option in args:
            try:
                return _normalize_value(value, option, key_path)
            except ConfigError:
                continue
        options = ", ".join(_annotation_name(option) for option in args)
        raise ConfigError(
            f"Field '{key_path}' expects one of ({options}), got {type(value).__name__}."
        )

    if isinstance(annotation, type):
        if not _value_matches_primitive(value, annotation):
            raise ConfigError(
                f"Field '{key_path}' expects {_annotation_name(annotation)}, got {type(value).__name__}."
            )
        if annotation is float and isinstance(value, int):
            return float(value)
        return value

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"Field '{key_path}' expects tuple, got {type(value).__name__}.")
        if len(args) == 2 and args[1] is Ellipsis:
            item_type = args[0]
            return tuple(_normalize_value(item, item_type, f"{key_path}[{idx}]") for idx, item in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(
                f"Field '{key_path}' expects tuple of length {len(args)}, got length {len(value)}."
            )
        return tuple(_normalize_value(item, item_type, f"{key_path}[{idx}]") for idx, (item, item_type) in enumerate(zip(value, args)))

    if origin is list:
        if not isinstance(value, list):
            raise ConfigError(f"Field '{key_path}' expects list, got {type(value).__name__}.")
        item_type = args[0] if args else Any
        return [_normalize_value(item, item_type, f"{key_path}[{idx}]") for idx, item in enumerate(value)]

    if origin is dict:
        if not isinstance(value, dict):
            raise ConfigError(f"Field '{key_path}' expects dict, got {type(value).__name__}.")
        key_type = args[0] if len(args) > 0 else Any
        value_type = args[1] if len(args) > 1 else Any
        normalized: dict[Any, Any] = {}
        for dict_key, dict_value in value.items():
            normalized_key = _normalize_value(dict_key, key_type, f"{key_path}.<key>")
            normalized[normalized_key] = _normalize_value(dict_value, value_type, f"{key_path}.{dict_key}")
        return normalized

    return value


def _field_allows_none(field_default: Any, annotation: Any) -> bool:
    if field_default is None:
        return True
    origin = get_origin(annotation)
    if origin in _UNION_ORIGINS:
        return type(None) in get_args(annotation)
    return False


def apply_dataclass_overrides(instance: Any, overrides: dict[str, Any], section_name: str) -> None:
    if not is_dataclass(instance):
        raise ConfigError(f"Instance for section '{section_name}' must be a dataclass.")
    if not isinstance(overrides, dict):
        raise ConfigError(f"Section '{section_name}' must be a mapping, got {type(overrides).__name__}.")

    field_map = {field_info.name: field_info for field_info in fields(instance)}
    validate_allowed_keys(overrides, set(field_map.keys()), f"section '{section_name}'")
    type_hints = get_type_hints(type(instance))

    for key, value in overrides.items():
        field_info = field_map[key]
        annotation = type_hints.get(key, Any)
        field_default = field_info.default if field_info.default is not MISSING else MISSING
        if value is None and _field_allows_none(field_default, annotation):
            setattr(instance, key, None)
            continue
        normalized = _normalize_value(value, annotation, f"{section_name}.{key}")
        setattr(instance, key, normalized)
