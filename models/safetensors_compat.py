from __future__ import annotations

from typing import Any

from safetensors import safe_open
from safetensors.torch import load_model


COMPILED_WRAPPER_TOKEN = "._orig_mod."


def _has_compiled_wrapper_keys(weights_path: str) -> bool:
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        return any(COMPILED_WRAPPER_TOKEN in key for key in f.keys())


def load_model_with_compile_key_compat(
    model: Any,
    weights_path: str,
    *,
    compile_model_for_compiled_wrapper_keys=None,
) -> dict[str, Any]:
    has_compiled_wrapper_keys = _has_compiled_wrapper_keys(weights_path)
    compile_summary = None

    if has_compiled_wrapper_keys:
        if compile_model_for_compiled_wrapper_keys is None:
            raise RuntimeError(
                "Checkpoint keys with '._orig_mod.' detected, but no compile callback "
                "was provided. Compile the model before loading this checkpoint."
            )
        compile_result = compile_model_for_compiled_wrapper_keys(model)
        if isinstance(compile_result, tuple):
            model = compile_result[0]
            if len(compile_result) > 1:
                compile_summary = compile_result[1]
        elif compile_result is not None:
            model = compile_result

    load_model(model, weights_path)
    return {
        "compiled_wrapper_keys": has_compiled_wrapper_keys,
        "compile_before_load_applied": has_compiled_wrapper_keys,
        "compile_summary": compile_summary,
    }
