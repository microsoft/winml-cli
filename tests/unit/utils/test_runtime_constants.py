# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for inference runtime constants."""

from __future__ import annotations

from typing import get_args

import pytest

from winml.modelkit.utils.constants import (
    RUNTIME_BACKENDS,
    RUNTIME_NAMES,
    RuntimeBackend,
    RuntimeName,
    resolve_runtime_api_backend,
)


def test_runtime_names_match_runtime_name_literal() -> None:
    assert get_args(RuntimeName) == RUNTIME_NAMES
    assert RUNTIME_NAMES == ("auto", "winml-ort", "ort-genai", "winml-runtime")


def test_runtime_backends_match_runtime_backend_literal() -> None:
    assert get_args(RuntimeBackend) == RUNTIME_BACKENDS
    assert RUNTIME_BACKENDS == ("ort", "cgc")


@pytest.mark.parametrize(
    ("model_path", "backend", "expected"),
    [
        ("model.onnx", None, "cgc"),
        ("model.onnx", "ort", "ort"),
        ("model.onnx", "cgc", "cgc"),
        ("model.mlir", None, "cgc"),
        ("model.mlir", "cgc", "cgc"),
    ],
)
def test_resolve_runtime_api_backend(model_path, backend, expected) -> None:
    assert resolve_runtime_api_backend("winml-runtime", model_path, backend) == expected


def test_resolve_runtime_api_backend_rejects_ort_for_mlir() -> None:
    with pytest.raises(ValueError, match="MLIR inputs require"):
        resolve_runtime_api_backend("winml-runtime", "model.mlir", "ort")


def test_resolve_runtime_api_backend_rejects_other_runtimes() -> None:
    with pytest.raises(ValueError, match="only supported"):
        resolve_runtime_api_backend("winml-ort", "model.onnx", "cgc")
