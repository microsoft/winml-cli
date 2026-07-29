# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for export-time attention compatibility handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from winml.modelkit.export import InputTensorSpec, OutputTensorSpec, WinMLExportConfig
from winml.modelkit.export.htp import HTPExporter
from winml.modelkit.export.policy import ExportCompatibilityConfig


if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class _AttentionConfig:
    """Minimal HF-style config with an attention implementation knob."""

    model_type = "fake"

    def __init__(self, implementation: str = "sdpa") -> None:
        self._attn_implementation = implementation


class _NestedAttentionModel(nn.Module):
    """Model with root and child configs to mirror HF module trees."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _AttentionConfig()
        self.proj = nn.Linear(2, 2)
        self.proj.config = _AttentionConfig()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _export_config(*, eager_attention: bool) -> WinMLExportConfig:
    return WinMLExportConfig(
        input_tensors=[InputTensorSpec(name="x", dtype="float32", shape=(1, 2))],
        output_tensors=[OutputTensorSpec(name="y")],
        compatibility=ExportCompatibilityConfig(
            transformers_attention="eager" if eager_attention else None
        ),
    )


def test_htp_exporter_uses_eager_attention_when_policy_requests_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _NestedAttentionModel()
    captured: dict[str, str] = {}

    def fake_export(*args: object, **kwargs: object) -> None:
        captured["root"] = model.config._attn_implementation
        captured["child"] = model.proj.config._attn_implementation

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    HTPExporter()._convert_model_to_onnx(
        model,
        str(tmp_path / "model.onnx"),
        {"x": torch.ones(1, 2)},
        _export_config(eager_attention=True),
        task=None,
    )

    assert captured == {"root": "eager", "child": "eager"}
    assert model.config._attn_implementation == "sdpa"
    assert model.proj.config._attn_implementation == "sdpa"


def test_htp_exporter_leaves_attention_unchanged_without_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _NestedAttentionModel()
    captured: dict[str, str] = {}

    def fake_export(*args: object, **kwargs: object) -> None:
        captured["root"] = model.config._attn_implementation
        captured["child"] = model.proj.config._attn_implementation

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    HTPExporter()._convert_model_to_onnx(
        model,
        str(tmp_path / "model.onnx"),
        {"x": torch.ones(1, 2)},
        _export_config(eager_attention=False),
        task=None,
    )

    assert captured == {"root": "sdpa", "child": "sdpa"}
    assert model.config._attn_implementation == "sdpa"
    assert model.proj.config._attn_implementation == "sdpa"
