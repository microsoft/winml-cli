# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for wmk build module mode (array config detection and orchestration)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import click
import pytest


if TYPE_CHECKING:
    from pathlib import Path

from winml.modelkit.commands.build import _load_config


class TestLoadConfigModuleMode:
    """Tests for _load_config array detection."""

    def test_single_config_returns_single(self, tmp_path: Path) -> None:
        """Single JSON object returns WinMLBuildConfig."""
        from winml.modelkit.config import WinMLBuildConfig

        cfg = tmp_path / "config.json"
        cfg.write_text(
            json.dumps(
                {
                    "loader": {"task": "fill-mask", "model_type": "bert"},
                    "export": {},
                    "optim": {},
                }
            )
        )

        result = _load_config(str(cfg), no_quant=False, no_compile=False)
        assert isinstance(result, WinMLBuildConfig)

    def test_array_config_returns_list(self, tmp_path: Path) -> None:
        """JSON array returns list[WinMLBuildConfig]."""
        from winml.modelkit.config import WinMLBuildConfig

        cfg = tmp_path / "modules.json"
        cfg.write_text(
            json.dumps(
                [
                    {
                        "loader": {
                            "task": "fill-mask",
                            "model_type": "bert",
                            "model_class": "BertAttention",
                            "module_path": "encoder.layer.0.attention",
                        },
                        "export": {},
                        "optim": {},
                    },
                    {
                        "loader": {
                            "task": "fill-mask",
                            "model_type": "bert",
                            "model_class": "BertAttention",
                            "module_path": "encoder.layer.1.attention",
                        },
                        "export": {},
                        "optim": {},
                    },
                ]
            )
        )

        result = _load_config(str(cfg), no_quant=False, no_compile=False)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(c, WinMLBuildConfig) for c in result)
        assert result[0].loader.module_path == "encoder.layer.0.attention"
        assert result[1].loader.module_path == "encoder.layer.1.attention"

    def test_array_config_applies_no_quant(self, tmp_path: Path) -> None:
        """--no-quant applies to every config in the array."""
        cfg = tmp_path / "modules.json"
        cfg.write_text(
            json.dumps(
                [
                    {
                        "loader": {
                            "task": "fill-mask",
                            "model_type": "bert",
                            "model_class": "X",
                            "module_path": "a",
                        },
                        "export": {},
                        "optim": {},
                        "quant": {"task": "fill-mask", "model_name": "X", "samples": 1},
                    },
                ]
            )
        )

        result = _load_config(str(cfg), no_quant=True, no_compile=False)
        assert isinstance(result, list)
        assert result[0].quant is None

    def test_load_config_empty_array_returns_empty_list(self, tmp_path: Path) -> None:
        """Empty JSON array returns empty list."""
        cfg = tmp_path / "empty.json"
        cfg.write_text("[]")
        result = _load_config(str(cfg), no_quant=False, no_compile=False)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_load_config_array_invalid_element_raises(self, tmp_path: Path) -> None:
        """Non-dict elements in JSON array raise UsageError."""
        cfg = tmp_path / "bad.json"
        cfg.write_text('[1, "not a dict"]')
        with pytest.raises(click.UsageError, match="Module config"):
            _load_config(str(cfg), no_quant=False, no_compile=False)

    def test_invalid_json_type_raises(self, tmp_path: Path) -> None:
        """Non-dict, non-list JSON raises UsageError."""
        cfg = tmp_path / "bad.json"
        cfg.write_text('"just a string"')

        with pytest.raises(click.UsageError, match="JSON object or array"):
            _load_config(str(cfg), no_quant=False, no_compile=False)


class TestBuildModuleOrchestration:
    """Tests for module mode build orchestration."""

    def test_build_modules_calls_build_per_instance(self, tmp_path: Path) -> None:
        """_build_modules calls build_hf_model for each config."""
        from winml.modelkit.commands.build import _build_modules
        from winml.modelkit.config import WinMLBuildConfig
        from winml.modelkit.export import WinMLExportConfig
        from winml.modelkit.loader import WinMLLoaderConfig
        from winml.modelkit.optim.config import WinMLOptimizationConfig

        configs = [
            WinMLBuildConfig(
                loader=WinMLLoaderConfig(
                    task="fill-mask",
                    model_type="bert",
                    model_class="BertAttention",
                    module_path="encoder.layer.0.attention",
                ),
                export=WinMLExportConfig(),
                optim=WinMLOptimizationConfig(),
                quant=None,
                compile=None,
            ),
            WinMLBuildConfig(
                loader=WinMLLoaderConfig(
                    task="fill-mask",
                    model_type="bert",
                    model_class="BertAttention",
                    module_path="encoder.layer.1.attention",
                ),
                export=WinMLExportConfig(),
                optim=WinMLOptimizationConfig(),
                quant=None,
                compile=None,
            ),
        ]

        mock_result = MagicMock()
        mock_result.reused = False
        mock_result.final_onnx_path = tmp_path / "model.onnx"
        mock_result.stages_completed = ["export"]
        mock_result.stages_skipped = []
        mock_result.stage_timings = {"export": 1.0}
        mock_result.elapsed = 1.0

        with (
            patch("winml.modelkit.build.build_hf_model", return_value=mock_result) as mock_build,
            patch("winml.modelkit.commands.build._instantiate_parent_model") as mock_parent,
        ):
            mock_model = MagicMock()
            mock_parent.return_value = mock_model

            results = _build_modules(
                configs=configs,
                output_dir=tmp_path,
                rebuild=False,
                ep=None,
                device=None,
            )

        assert len(results) == 2
        assert mock_build.call_count == 2
        # Verify parent instantiated once (same model_type + task)
        mock_parent.assert_called_once_with("bert", task="fill-mask")
        # Verify get_submodule called with correct paths
        mock_model.get_submodule.assert_any_call("encoder.layer.0.attention")
        mock_model.get_submodule.assert_any_call("encoder.layer.1.attention")

    def test_build_modules_rejects_missing_model_type(self, tmp_path: Path) -> None:
        """_build_modules raises if model_type is missing."""
        from winml.modelkit.commands.build import _build_modules
        from winml.modelkit.config import WinMLBuildConfig
        from winml.modelkit.export import WinMLExportConfig
        from winml.modelkit.loader import WinMLLoaderConfig
        from winml.modelkit.optim.config import WinMLOptimizationConfig

        configs = [
            WinMLBuildConfig(
                loader=WinMLLoaderConfig(
                    task="fill-mask",
                    model_type=None,
                    model_class="X",
                    module_path="a.b",
                ),
                export=WinMLExportConfig(),
                optim=WinMLOptimizationConfig(),
                quant=None,
                compile=None,
            ),
        ]

        with pytest.raises(click.UsageError, match="model_type"):
            _build_modules(configs=configs, output_dir=tmp_path, rebuild=False)

    def test_build_modules_rejects_missing_module_path(self, tmp_path: Path) -> None:
        """_build_modules raises if module_path is missing."""
        from winml.modelkit.commands.build import _build_modules
        from winml.modelkit.config import WinMLBuildConfig
        from winml.modelkit.export import WinMLExportConfig
        from winml.modelkit.loader import WinMLLoaderConfig
        from winml.modelkit.optim.config import WinMLOptimizationConfig

        configs = [
            WinMLBuildConfig(
                loader=WinMLLoaderConfig(
                    task="fill-mask",
                    model_type="bert",
                    model_class="X",
                    module_path=None,
                ),
                export=WinMLExportConfig(),
                optim=WinMLOptimizationConfig(),
                quant=None,
                compile=None,
            ),
        ]

        with pytest.raises(click.UsageError, match="module_path"):
            _build_modules(configs=configs, output_dir=tmp_path, rebuild=False)
