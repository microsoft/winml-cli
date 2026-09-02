# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for export-time attention compatibility handling."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

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


class _CascadingAttentionConfig:
    """HF-style config whose setter cascades attention into child configs."""

    model_type = "fake"

    def __init__(
        self,
        implementation: str,
        *,
        sub_configs: list[_CascadingAttentionConfig]
        | dict[str, _CascadingAttentionConfig]
        | None = None,
    ) -> None:
        self._implementation = implementation
        self.sub_configs = sub_configs or []

    @property
    def _attn_implementation(self) -> str:
        return self._implementation

    @_attn_implementation.setter
    def _attn_implementation(self, value: str) -> None:
        self._implementation = value
        sub_configs = (
            self.sub_configs.values() if isinstance(self.sub_configs, dict) else self.sub_configs
        )
        for config in sub_configs:
            config._attn_implementation = value


class _NestedAttentionModel(nn.Module):
    """Model with root and child configs to mirror HF module trees."""

    def __init__(self) -> None:
        super().__init__()
        self.config = _AttentionConfig()
        self.proj = nn.Linear(2, 2)
        self.proj.config = _AttentionConfig()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _CascadingAttentionModel(nn.Module):
    """Model shaped like HF composite configs where the root setter recurses."""

    def __init__(self) -> None:
        super().__init__()
        child_config = _CascadingAttentionConfig("flash_attention_2")
        self.config = _CascadingAttentionConfig("sdpa", sub_configs=[child_config])
        self.proj = nn.Linear(2, 2)
        self.proj.config = child_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _CascadingEagerChildAttentionModel(nn.Module):
    """Composite config with an eager child that still needs restore snapshotting."""

    def __init__(self) -> None:
        super().__init__()
        child_config = _CascadingAttentionConfig("eager")
        self.config = _CascadingAttentionConfig("sdpa", sub_configs=[child_config])
        self.proj = nn.Linear(2, 2)
        self.proj.config = child_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _CascadingDetachedEagerChildAttentionModel(nn.Module):
    """Composite config whose child is reachable only through parent.sub_configs."""

    def __init__(self) -> None:
        super().__init__()
        self.child_config = _CascadingAttentionConfig("eager")
        self.config = _CascadingAttentionConfig(
            "sdpa", sub_configs={"child_config": self.child_config}
        )
        self.proj = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _ChildBeforeParentAttentionModel(nn.Module):
    """Model where module traversal sees a child config before its cascading parent."""

    def __init__(self) -> None:
        super().__init__()
        self.child_config = _CascadingAttentionConfig("eager")
        self.early = nn.Linear(2, 2)
        self.early.config = self.child_config
        self.parent_config = _CascadingAttentionConfig("sdpa", sub_configs=[self.child_config])
        self.late = nn.Linear(2, 2)
        self.late.config = self.parent_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.late(self.early(x))


def _export_config(*, eager_attention: bool) -> WinMLExportConfig:
    return WinMLExportConfig(
        input_tensors=[InputTensorSpec(name="x", dtype="float32", shape=(1, 2))],
        output_tensors=[OutputTensorSpec(name="y")],
        compatibility=ExportCompatibilityConfig(
            transformers_attention="eager" if eager_attention else None
        ),
    )


def test_htp_exporter_auto_load_threads_attention_compatibility(tmp_path: Path) -> None:
    export_config = _export_config(eager_attention=True)

    with patch("winml.modelkit.loader.load_hf_model") as mock_load:
        mock_load.side_effect = RuntimeError("stop after model loading")
        try:
            HTPExporter().export(
                output_path=str(tmp_path / "model.onnx"),
                export_config=export_config,
                model_name_or_path="fake/model",
            )
        except RuntimeError as exc:
            assert str(exc) == "stop after model loading"
        else:
            raise AssertionError("Expected the loader sentinel to stop export")

    mock_load.assert_called_once_with(
        "fake/model",
        attn_implementation="eager",
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


def test_htp_exporter_restores_nested_attention_configs_losslessly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _CascadingAttentionModel()
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
    assert model.proj.config._attn_implementation == "flash_attention_2"


def test_htp_exporter_restores_eager_child_after_parent_restore_cascades(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _CascadingEagerChildAttentionModel()
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
    assert model.proj.config._attn_implementation == "eager"


def test_htp_exporter_restores_sub_config_not_attached_to_module(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _CascadingDetachedEagerChildAttentionModel()
    captured: dict[str, str] = {}

    def fake_export(*args: object, **kwargs: object) -> None:
        captured["root"] = model.config._attn_implementation
        captured["child"] = model.child_config._attn_implementation

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
    assert model.child_config._attn_implementation == "eager"


def test_htp_exporter_restores_child_discovered_before_parent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _ChildBeforeParentAttentionModel()
    captured: dict[str, str] = {}

    def fake_export(*args: object, **kwargs: object) -> None:
        captured["parent"] = model.parent_config._attn_implementation
        captured["child"] = model.child_config._attn_implementation

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    HTPExporter()._convert_model_to_onnx(
        model,
        str(tmp_path / "model.onnx"),
        {"x": torch.ones(1, 2)},
        _export_config(eager_attention=True),
        task=None,
    )

    assert captured == {"parent": "eager", "child": "eager"}
    assert model.parent_config._attn_implementation == "sdpa"
    assert model.child_config._attn_implementation == "eager"


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
