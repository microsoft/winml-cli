# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Runtime annotation-resolution tests for auto-optimize scripts."""

from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from typing import get_type_hints


SKILL_ROOT = Path(__file__).resolve().parents[1]
FINALIZER_PATH = SKILL_ROOT / "scripts" / "finalize_output.py"
PROMOTION_PATH = SKILL_ROOT / "scripts" / "promotion.py"


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _type_hint_name_error(function: types.FunctionType) -> str | None:
    try:
        get_type_hints(function)
    except NameError as error:
        return f"{type(error).__name__}: {error}"
    return None


def test_finalize_output_module_annotations_resolve_at_runtime() -> None:
    module = _load_module("runtime_annotations_finalize_output", FINALIZER_PATH)
    functions = [
        function
        for _, function in inspect.getmembers(module, inspect.isfunction)
        if function.__module__ == module.__name__ and function.__annotations__
    ]
    assert functions
    failures = {
        function.__name__: error
        for function in functions
        if (error := _type_hint_name_error(function)) is not None
    }

    assert not failures, failures
    assert get_type_hints(module._load_renderer)["return"] is types.ModuleType
    expected_sequence = Sequence[Path]
    assert get_type_hints(module._prepare_inputs)["companions"] == expected_sequence
    assert get_type_hints(module._prepare_reproduction)["repro_assets"] == expected_sequence
    assert get_type_hints(module._prepare_reproduction)["companions"] == expected_sequence
    assert get_type_hints(module._validate_repro_lock)["assets"] == expected_sequence
    assert get_type_hints(module.finalize_output)["companions"] == expected_sequence
    assert get_type_hints(module.finalize_output)["repro_assets"] == expected_sequence


def test_cached_promotion_finalizer_annotation_resolves_at_runtime() -> None:
    promotion = _load_module("runtime_annotations_promotion", PROMOTION_PATH)

    assert get_type_hints(promotion._finalizer) == {"return": types.ModuleType}
