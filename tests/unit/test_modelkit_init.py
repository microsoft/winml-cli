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

import sys
import types
from unittest import mock


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
