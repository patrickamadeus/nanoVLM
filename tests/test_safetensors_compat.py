from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.safetensors_compat import load_model_with_compile_key_compat


def test_load_model_with_compile_key_compat_calls_compile_first_for_wrapper_keys(monkeypatch):
    model = torch.nn.Linear(2, 2)
    compile_called = {"value": False}
    load_called = {"value": False}

    def fake_compile_callback(m):
        compile_called["value"] = True
        return m, {"compiled": {"dummy": 1}}

    def fake_load_model(m, path):
        load_called["value"] = True

    monkeypatch.setattr(
        "models.safetensors_compat._has_compiled_wrapper_keys",
        lambda _: True,
    )
    monkeypatch.setattr("models.safetensors_compat.load_model", fake_load_model)

    summary = load_model_with_compile_key_compat(
        model,
        "dummy.safetensors",
        compile_model_for_compiled_wrapper_keys=fake_compile_callback,
    )

    assert summary["compiled_wrapper_keys"] is True
    assert summary["compile_before_load_applied"] is True
    assert summary["compile_summary"] == {"compiled": {"dummy": 1}}
    assert compile_called["value"] is True
    assert load_called["value"] is True


def test_load_model_with_compile_key_compat_skips_compile_for_canonical(monkeypatch):
    model = torch.nn.Linear(2, 2)
    compile_called = {"value": False}
    load_called = {"value": False}

    def fake_compile_callback(m):
        compile_called["value"] = True
        return m

    def fake_load_model(m, path):
        load_called["value"] = True

    monkeypatch.setattr(
        "models.safetensors_compat._has_compiled_wrapper_keys",
        lambda _: False,
    )
    monkeypatch.setattr("models.safetensors_compat.load_model", fake_load_model)

    summary = load_model_with_compile_key_compat(
        model,
        "dummy.safetensors",
        compile_model_for_compiled_wrapper_keys=fake_compile_callback,
    )

    assert summary["compiled_wrapper_keys"] is False
    assert summary["compile_before_load_applied"] is False
    assert summary["compile_summary"] is None
    assert compile_called["value"] is False
    assert load_called["value"] is True
