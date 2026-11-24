# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for winml build module mode (array config detection and orchestration)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import click
import pytest


if TYPE_CHECKING:
    from pathlib import Path

from winml.modelkit.commands.build import _load_config


def test_offdevice_modules_keep_independent_compile_targets(tmp_path: Path) -> None:
    from torch import nn

    from tests.unit.compiler.test_compiler_stages import create_simple_model
    from winml.modelkit.commands.build import _build_modules, _run_compile_stage
    from winml.modelkit.compiler import CompileResult, WinMLCompileConfig
    from winml.modelkit.config import WinMLBuildConfig
    from winml.modelkit.loader import WinMLLoaderConfig
    from winml.modelkit.utils.constants import normalize_ep_name

    parent = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
    requests = [("openvino", "4000"), ("qnn", "60")]
    configs = [
        WinMLBuildConfig(
            loader=WinMLLoaderConfig(
                model_type=type(parent).__name__,
                model_class=type(module).__name__,
                module_path=name,
            ),
            compile=WinMLCompileConfig.from_dict(
                {
                    "execution_provider": provider,
                    "compiler": "offdevice",
                    "target_platform": platform,
                    "embed_context": True,
                }
            ),
        )
        for (name, module), (provider, platform) in zip(
            parent.named_children(), requests, strict=True
        )
    ]
    source = tmp_path / "source.onnx"
    create_simple_model(source)

    def compile_model(*, model_path, output_path, config):
        output_path.write_bytes(model_path.read_bytes())
        return CompileResult(success=True, output_path=output_path)

    def build_module(**kwargs):
        kwargs["output_dir"].mkdir()
        return _run_compile_stage(
            config=kwargs["config"],
            current_path=source,
            compiled_path=kwargs["output_dir"] / source.name,
            stage_timings=[],
        )

    with (
        patch("winml.modelkit.commands.build._instantiate_parent_model", return_value=parent),
        patch("winml.modelkit.build.build_hf_model", side_effect=build_module) as build,
        patch("winml.modelkit.compiler.compile_onnx", side_effect=compile_model) as backend,
        patch(
            "winml.modelkit.session.resolve_device",
            side_effect=AssertionError("Off-device module attempted host-device resolution"),
        ),
    ):
        _build_modules(configs=configs, output_dir=tmp_path)

    for build_call, compile_call, config in zip(
        build.call_args_list, backend.call_args_list, configs, strict=True
    ):
        assert build_call.kwargs["ep"] == normalize_ep_name(config.compile.ep_config.provider)
        assert build_call.kwargs["device"] == "npu"
        assert compile_call.kwargs["config"] is config.compile


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

        result = _load_config(str(cfg), no_quant=False, no_compile=None)
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

        result = _load_config(str(cfg), no_quant=False, no_compile=None)
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
                        "quant": {"task": "fill-mask", "model_id": "X", "samples": 1},
                    },
                ]
            )
        )

        result = _load_config(str(cfg), no_quant=True, no_compile=None)
        assert isinstance(result, list)
        assert result[0].quant is None

    def test_load_config_empty_array_returns_empty_list(self, tmp_path: Path) -> None:
        """Empty JSON array returns empty list."""
        cfg = tmp_path / "empty.json"
        cfg.write_text("[]")
        result = _load_config(str(cfg), no_quant=False, no_compile=None)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_load_config_array_invalid_element_raises(self, tmp_path: Path) -> None:
        """Non-dict elements in JSON array raise UsageError."""
        cfg = tmp_path / "bad.json"
        cfg.write_text('[1, "not a dict"]')
        with pytest.raises(click.UsageError, match="Module config"):
            _load_config(str(cfg), no_quant=False, no_compile=None)

    def test_invalid_json_type_raises(self, tmp_path: Path) -> None:
        """Non-dict, non-list JSON raises UsageError."""
        cfg = tmp_path / "bad.json"
        cfg.write_text('"just a string"')

        with pytest.raises(click.UsageError, match="JSON object or array"):
            _load_config(str(cfg), no_quant=False, no_compile=None)


class TestBuildModuleOrchestration:
    """Tests for module mode build orchestration."""

    def test_build_modules_calls_build_per_instance(self, tmp_path: Path) -> None:
        """_build_modules calls build_hf_model for each config."""
        from winml.modelkit.commands.build import _build_modules
        from winml.modelkit.config import WinMLBuildConfig
        from winml.modelkit.export import WinMLExportConfig
        from winml.modelkit.loader import WinMLLoaderConfig
        from winml.modelkit.optim import WinMLOptimizationConfig

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
        from winml.modelkit.optim import WinMLOptimizationConfig

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
        from winml.modelkit.optim import WinMLOptimizationConfig

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
