# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression tests for ``_preload_bundled_onnxruntime_dll``.

The preload is the load-order fix for the System32 vs. wheel-bundled
``onnxruntime.dll`` collision on Windows. It must run before any subpackage
import that transitively touches onnxruntime, otherwise the System32 copy
wins and produces "requested API version [N] is not available" errors on
end-user machines. These tests pin the contract without needing a real
Windows host or a real wheel.
"""

from __future__ import annotations

import logging
import os
import sys
import types
import warnings
from unittest import mock

import pytest


_MISSING = object()
_MODELKIT_LOGGERS = (
    "diffusers.utils.import_utils",
    "transformers.pipelines.base",
    "transformers.models.auto.image_processing_auto",
)
_MODELKIT_ENV_VARS = ("TOKENIZERS_PARALLELISM", "HF_HUB_DISABLE_PROGRESS_BARS")


@pytest.fixture(autouse=True)
def _restore_modelkit_import_state():
    """Keep forced package reimports from leaking into the rest of the suite."""
    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "winml.modelkit" or name.startswith("winml.modelkit.")
    }
    original_meta_path = list(sys.meta_path)
    original_warning_filters = list(warnings.filters)
    original_logger_filters = {
        name: list(logging.getLogger(name).filters) for name in _MODELKIT_LOGGERS
    }
    original_environment = {name: os.environ.get(name, _MISSING) for name in _MODELKIT_ENV_VARS}
    original_winml = sys.modules.get("winml")
    original_modelkit_attr = (
        getattr(original_winml, "modelkit", _MISSING) if original_winml is not None else _MISSING
    )

    yield

    for name in [
        module_name
        for module_name in sys.modules
        if module_name == "winml.modelkit" or module_name.startswith("winml.modelkit.")
    ]:
        sys.modules.pop(name, None)
    sys.modules.update(original_modules)
    sys.meta_path[:] = original_meta_path
    warnings.filters[:] = original_warning_filters
    for name, filters in original_logger_filters.items():
        logging.getLogger(name).filters[:] = filters
    for name, value in original_environment.items():
        if value is _MISSING:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value

    if original_winml is None:
        sys.modules.pop("winml", None)
    elif original_modelkit_attr is _MISSING:
        if hasattr(original_winml, "modelkit"):
            delattr(original_winml, "modelkit")
    else:
        original_winml.modelkit = original_modelkit_attr


def _reimport_modelkit():
    # Drop the cached package so its __init__.py (and the preload call at
    # module scope) executes again under the active mocks.
    for name in [
        m for m in sys.modules if m == "winml.modelkit" or m.startswith("winml.modelkit.")
    ]:
        sys.modules.pop(name, None)
    import importlib

    return importlib.import_module("winml.modelkit")


def test_preload_is_noop_on_non_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    with mock.patch("ctypes.WinDLL", create=True) as windll:
        _reimport_modelkit()
        windll.assert_not_called()


def test_preload_skips_when_bundled_dll_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    fake_spec = types.SimpleNamespace(origin=str(tmp_path / "onnxruntime" / "__init__.py"))
    with (
        mock.patch("importlib.util.find_spec", return_value=fake_spec),
        mock.patch("ctypes.WinDLL", create=True) as windll,
        mock.patch("os.add_dll_directory", create=True) as add_dir,
    ):
        _reimport_modelkit()
        windll.assert_not_called()
        add_dir.assert_not_called()


def test_preload_loads_dll_on_windows_when_present(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "win32")
    capi = tmp_path / "onnxruntime" / "capi"
    capi.mkdir(parents=True)
    (capi / "onnxruntime.dll").write_bytes(b"")
    fake_spec = types.SimpleNamespace(origin=str(tmp_path / "onnxruntime" / "__init__.py"))
    with (
        mock.patch("importlib.util.find_spec", return_value=fake_spec),
        mock.patch("ctypes.WinDLL", create=True) as windll,
        mock.patch("os.add_dll_directory", create=True) as add_dir,
    ):
        _reimport_modelkit()
        add_dir.assert_called_once_with(str(capi))
        windll.assert_called_once_with(str(capi / "onnxruntime.dll"))
