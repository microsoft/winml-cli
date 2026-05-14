# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests warning suppression behavior in export_pytorch()."""

from __future__ import annotations

import sys
import types
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from winml.modelkit.export.pytorch import export_pytorch


def _install_fake_exporter(
    monkeypatch,
    export_return: dict[str, int],
    *,
    emit_warning: bool,
) -> MagicMock:
    exporter_cls = MagicMock()
    exporter = exporter_cls.return_value

    def _export(*args, **kwargs):
        if emit_warning:
            warnings.warn("tracer warning noise", UserWarning, stacklevel=2)
        return export_return

    exporter.export.side_effect = _export

    fake_htp_pkg = types.ModuleType("winml.modelkit.export.htp")
    fake_htp_pkg.__path__ = []  # mark as package
    fake_exporter_module = types.ModuleType("winml.modelkit.export.htp.exporter")
    fake_exporter_module.HTPExporter = exporter_cls

    monkeypatch.setitem(sys.modules, "winml.modelkit.export.htp", fake_htp_pkg)
    monkeypatch.setitem(sys.modules, "winml.modelkit.export.htp.exporter", fake_exporter_module)
    return exporter


def test_export_pytorch_uses_warning_context(tmp_path, monkeypatch) -> None:
    """export_pytorch should use warnings.catch_warnings() context manager."""

    class DummyModel:
        pass

    model = DummyModel()
    config = SimpleNamespace(enable_hierarchy_tags=True)
    expected = {"onnx_nodes": 1}
    exporter = _install_fake_exporter(monkeypatch, expected, emit_warning=False)

    with (
        patch("winml.modelkit.export.pytorch.warnings.catch_warnings") as mock_catch_warnings,
        patch("winml.modelkit.export.pytorch.warnings.filterwarnings") as mock_filterwarnings,
    ):
        result = export_pytorch(model, tmp_path / "model.onnx", config)

    assert result == expected
    exporter.export.assert_called_once()
    mock_catch_warnings.assert_called_once_with()
    mock_catch_warnings.return_value.__enter__.assert_called_once_with()
    mock_filterwarnings.assert_called_once_with("ignore")


def test_export_pytorch_suppresses_export_warnings(tmp_path, monkeypatch) -> None:
    """Warnings emitted during export should not leak to callers."""

    class DummyModel:
        pass

    model = DummyModel()
    config = SimpleNamespace(enable_hierarchy_tags=True)
    expected = {"onnx_nodes": 1}
    exporter = _install_fake_exporter(monkeypatch, expected, emit_warning=True)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        result = export_pytorch(model, tmp_path / "model.onnx", config)

    assert result == expected
    exporter.export.assert_called_once()
    assert captured == []
