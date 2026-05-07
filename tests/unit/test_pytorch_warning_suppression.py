# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests warning suppression behavior in export_pytorch()."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from winml.modelkit.export.pytorch import export_pytorch


def test_export_pytorch_suppresses_warnings(tmp_path, monkeypatch) -> None:
    """export_pytorch should suppress warnings around exporter invocation."""

    class DummyModel:
        pass

    model = DummyModel()
    config = SimpleNamespace(enable_hierarchy_tags=True)
    expected = {"onnx_nodes": 1}
    exporter_cls = MagicMock()
    exporter = exporter_cls.return_value
    exporter.export.return_value = expected

    fake_htp_pkg = types.ModuleType("winml.modelkit.export.htp")
    fake_htp_pkg.__path__ = []  # mark as package
    fake_exporter_module = types.ModuleType("winml.modelkit.export.htp.exporter")
    fake_exporter_module.HTPExporter = exporter_cls

    monkeypatch.setitem(sys.modules, "winml.modelkit.export.htp", fake_htp_pkg)
    monkeypatch.setitem(
        sys.modules,
        "winml.modelkit.export.htp.exporter",
        fake_exporter_module,
    )

    with (
        patch("winml.modelkit.export.pytorch.warnings.catch_warnings") as mock_catch_warnings,
        patch("winml.modelkit.export.pytorch.warnings.filterwarnings") as mock_filterwarnings,
    ):
        result = export_pytorch(model, tmp_path / "model.onnx", config)

    assert result == expected
    mock_catch_warnings.assert_called_once_with()
    mock_filterwarnings.assert_called_once_with("ignore")
