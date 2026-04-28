# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ONNX-specific config generation in modelkit.config.build module.

Split from test_build.py to reduce per-file test duration in CI.

Tests cover:
- TestConfigOnnxAutoDetect: ONNX file auto-detection in config command
- TestGenerateBuildConfigOnnxPath: Comprehensive tests for generate_onnx_build_config
- TestResolveQuantCompileConfig: Tests for the standalone resolver
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

# Import models package to trigger ONNX config registration with TasksManager
import winml.modelkit.models  # noqa: F401
from winml.modelkit.commands.config import config as config_command
from winml.modelkit.compiler import WinMLCompileConfig
from winml.modelkit.config import (
    WinMLBuildConfig,
    generate_onnx_build_config,
)
from winml.modelkit.config.build import (
    resolve_quant_compile_config,
)
from winml.modelkit.export import InputTensorSpec, OutputTensorSpec, WinMLExportConfig
from winml.modelkit.loader import WinMLLoaderConfig
from winml.modelkit.optim import WinMLOptimizationConfig
from winml.modelkit.quant import WinMLQuantizationConfig
from winml.modelkit.utils.config_utils import merge_config


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_hf_config() -> MagicMock:
    """Create a mock HuggingFace config with model_type."""
    config = MagicMock(spec=["model_type", "architectures"])
    config.model_type = "bert"
    config.architectures = ["BertForMaskedLM"]
    return config


@pytest.fixture
def mock_model_class() -> MagicMock:
    """Create a mock model class."""
    model_class = MagicMock()
    model_class.__name__ = "BertForMaskedLM"
    return model_class


@pytest.fixture
def mock_loader_config() -> WinMLLoaderConfig:
    """Create a mock WinMLLoaderConfig for BERT fill-mask."""
    return WinMLLoaderConfig(
        task="fill-mask",
        model_class="BertForMaskedLM",
        model_type="bert",
    )


@pytest.fixture
def mock_export_config() -> WinMLExportConfig:
    """Create a mock WinMLExportConfig matching BERT structure."""
    return WinMLExportConfig(
        input_tensors=[
            InputTensorSpec(name="input_ids", shape=(2, 16), dtype="int64"),
            InputTensorSpec(name="attention_mask", shape=(2, 16), dtype="int64"),
        ],
        output_tensors=[OutputTensorSpec(name="logits")],
    )


# =============================================================================
# TestConfigOnnxAutoDetect - ONNX file auto-detection in config command
# =============================================================================


class TestConfigOnnxAutoDetect:
    """Test ONNX file auto-detection in winml config command."""

    def test_config_auto_detect_onnx(self, tmp_path) -> None:
        """When -m points to an existing .onnx file, generates config with export=None."""
        # Create a fake .onnx file
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake-onnx-data")
        output_file = tmp_path / "result.json"

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
        ):
            runner = CliRunner()
            result = runner.invoke(
                config_command,
                ["-m", str(onnx_file), "-o", str(output_file)],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        output_data = json.loads(output_file.read_text())
        # ONNX build: export should be None
        assert output_data["export"] is None
        # optim should be present (default)
        assert output_data["optim"] is not None

    def test_config_onnx_with_device_precision(self, tmp_path) -> None:
        """ONNX config with --device npu applies quant/compile policy."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake-onnx-data")
        output_file = tmp_path / "result.json"

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("npu", ["npu", "gpu", "cpu"]),
            ),
        ):
            runner = CliRunner()
            result = runner.invoke(
                config_command,
                ["-m", str(onnx_file), "--device", "npu", "-o", str(output_file)],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        output_data = json.loads(output_file.read_text())
        assert output_data["export"] is None
        assert output_data["quant"] is not None
        assert output_data["compile"] is not None
        assert output_data["compile"]["execution_provider"] == "qnn"

    def test_config_onnx_suffix_not_exists_uses_hf(
        self,
        tmp_path,
        mock_hf_config: MagicMock,
        mock_model_class: MagicMock,
        mock_loader_config: WinMLLoaderConfig,
        mock_export_config: WinMLExportConfig,
    ) -> None:
        """An .onnx path that doesn't exist falls through to HF config generation."""
        output_file = tmp_path / "result.json"

        with (
            patch(
                "winml.modelkit.config.build.resolve_loader_config",
                return_value=(mock_loader_config, mock_hf_config, mock_model_class),
            ),
            patch(
                "winml.modelkit.config.build._resolve_export_config_from_specs",
                return_value=mock_export_config,
            ),
            patch("winml.modelkit.models.hf.MODEL_BUILD_CONFIGS", {}),
        ):
            runner = CliRunner()
            result = runner.invoke(
                config_command,
                ["-m", "nonexistent.onnx", "-o", str(output_file)],
            )

        assert result.exit_code == 0, f"CLI failed: {result.output}"
        output_data = json.loads(output_file.read_text())
        # Should be HF config (export present)
        assert output_data["export"] is not None


# =============================================================================
# TestGenerateBuildConfigOnnxPath - Comprehensive tests for generate_onnx_build_config
# =============================================================================


class TestGenerateBuildConfigOnnxPath:
    """Comprehensive tests for generate_onnx_build_config() covering all branches.

    Tests call generate_onnx_build_config directly (not the dispatcher) and mock:
    - modelkit.onnx.is_compiled_onnx
    - modelkit.onnx.is_quantized_onnx
    - modelkit.sysinfo.resolve_device
    """

    # -----------------------------------------------------------------
    # Model state detection
    # -----------------------------------------------------------------

    def test_raw_onnx_full_pipeline(self, tmp_path) -> None:
        """Raw ONNX + device=npu resolves quant=w8a16 and compile=qnn."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("npu", ["npu", "cpu"]),
            ),
        ):
            config = generate_onnx_build_config(str(onnx_file), device="npu")

        assert config.export is None
        assert config.quant is not None
        assert config.quant.weight_type == "uint8"
        assert config.quant.activation_type == "uint16"
        assert config.compile is not None
        assert config.compile.ep_config.provider == "qnn"

    def test_raw_onnx_cpu(self, tmp_path) -> None:
        """Raw ONNX + device=cpu resolves quant=None and compile=None."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("cpu", ["cpu"]),
            ),
        ):
            config = generate_onnx_build_config(str(onnx_file), device="cpu")

        assert config.export is None
        assert config.quant is None
        assert config.compile is None

    def test_quantized_onnx_skips_quant(self, tmp_path) -> None:
        """Quantized ONNX + device=npu sets quant=None, compile=qnn."""
        onnx_file = tmp_path / "quantized.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=True),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("npu", ["npu", "cpu"]),
            ),
        ):
            config = generate_onnx_build_config(str(onnx_file), device="npu")

        assert config.quant is None
        assert config.compile is not None
        assert config.compile.ep_config.provider == "qnn"

    def test_quantized_onnx_cpu(self, tmp_path) -> None:
        """Quantized ONNX + device=cpu sets quant=None, compile=None."""
        onnx_file = tmp_path / "quantized.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=True),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("cpu", ["cpu"]),
            ),
        ):
            config = generate_onnx_build_config(str(onnx_file), device="cpu")

        assert config.quant is None
        assert config.compile is None

    def test_compiled_onnx_skips_all(self, tmp_path) -> None:
        """Compiled ONNX (EPContext) sets quant=None and compile=None."""
        onnx_file = tmp_path / "compiled.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=True),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
        ):
            config = generate_onnx_build_config(str(onnx_file))

        assert config.quant is None
        assert config.compile is None

    def test_compiled_onnx_with_device_npu(self, tmp_path) -> None:
        """Compiled ONNX + device=npu still sets quant=None and compile=None.

        The compiled detection short-circuits before resolve_quant_compile_config
        is called, so device parameter has no effect.
        """
        onnx_file = tmp_path / "compiled.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=True),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
        ):
            config = generate_onnx_build_config(str(onnx_file), device="npu")

        assert config.quant is None
        assert config.compile is None

    # -----------------------------------------------------------------
    # Config structure invariants
    # -----------------------------------------------------------------

    def test_export_always_none(self, tmp_path) -> None:
        """All ONNX model states produce export=None."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        states = [
            (False, False, "raw"),
            (False, True, "quantized"),
            (True, False, "compiled"),
        ]
        for is_compiled, is_quantized, label in states:
            with (
                patch("winml.modelkit.onnx.is_compiled_onnx", return_value=is_compiled),
                patch("winml.modelkit.onnx.is_quantized_onnx", return_value=is_quantized),
                patch(
                    "winml.modelkit.sysinfo.resolve_device",
                    return_value=("cpu", ["cpu"]),
                ),
            ):
                config = generate_onnx_build_config(str(onnx_file))

            assert config.export is None, f"export should be None for {label} model"

    def test_optim_always_present(self, tmp_path) -> None:
        """All ONNX model states produce a non-None optim config."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        states = [
            (False, False, "raw"),
            (False, True, "quantized"),
            (True, False, "compiled"),
        ]
        for is_compiled, is_quantized, label in states:
            with (
                patch("winml.modelkit.onnx.is_compiled_onnx", return_value=is_compiled),
                patch("winml.modelkit.onnx.is_quantized_onnx", return_value=is_quantized),
                patch(
                    "winml.modelkit.sysinfo.resolve_device",
                    return_value=("cpu", ["cpu"]),
                ),
            ):
                config = generate_onnx_build_config(str(onnx_file))

            assert isinstance(config.optim, WinMLOptimizationConfig), (
                f"optim should be WinMLOptimizationConfig for {label} model"
            )

    def test_task_stored_in_loader(self, tmp_path) -> None:
        """task='image-classification' is stored in config.loader.task."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("cpu", ["cpu"]),
            ),
        ):
            config = generate_onnx_build_config(
                str(onnx_file),
                task="image-classification",
            )

        assert config.loader.task == "image-classification"

    def test_task_none_by_default(self, tmp_path) -> None:
        """When no task is provided, config.loader.task is None."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("cpu", ["cpu"]),
            ),
        ):
            config = generate_onnx_build_config(str(onnx_file))

        assert config.loader.task is None

    # -----------------------------------------------------------------
    # Override behavior
    # -----------------------------------------------------------------

    def test_override_applied_last(self, tmp_path) -> None:
        """Override with a specific optim flag is present after device resolution.

        WinMLOptimizationConfig is a dict subclass, so flags are dict keys.
        """
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        override = WinMLBuildConfig(
            optim=WinMLOptimizationConfig(gelu_fusion=True),
        )

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("npu", ["npu", "cpu"]),
            ),
        ):
            config = generate_onnx_build_config(
                str(onnx_file),
                device="npu",
                override=override,
            )

        assert config.optim["gelu_fusion"] is True

    def test_override_quant_none_on_raw(self, tmp_path) -> None:
        """Raw ONNX + device=npu would resolve quant, but override sets quant=None.

        Override is applied last via merge_config, and explicit None in the
        override replaces the resolved quant config.
        """
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        # merge_config uses to_dict() which produces {"quant": None, ...}
        # only when quant is explicitly None. Build from dict to control this.
        override_dict = {"quant": None}

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("npu", ["npu", "cpu"]),
            ),
            patch(
                "winml.modelkit.config.build.merge_config",
                wraps=merge_config,
            ) as mock_merge,
        ):
            # Call with a real override that sets quant=None
            override_cfg = WinMLBuildConfig.from_dict(override_dict)
            config = generate_onnx_build_config(
                str(onnx_file),
                device="npu",
                override=override_cfg,
            )

        mock_merge.assert_called_once()
        # merge_config with quant=None override should set quant to None
        assert config.quant is None

    def test_override_on_compiled_model(self, tmp_path) -> None:
        """Compiled model + override with quant set: override is applied AFTER
        compiled detection, so override CAN set quant on a compiled model.

        This tests that merge_config runs after the compiled branch.
        merge_config reconstructs the quant field from the override dict when
        the base quant is None.
        """
        onnx_file = tmp_path / "compiled.onnx"
        onnx_file.write_bytes(b"fake")

        override = WinMLBuildConfig(
            quant=WinMLQuantizationConfig(weight_type="uint8"),
        )

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=True),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
        ):
            config = generate_onnx_build_config(
                str(onnx_file),
                override=override,
            )

        # Override is applied after compiled detection, so quant is non-None.
        # merge_config stores it as a dict (base quant is None, so from_dict
        # reconstruction depends on type annotation resolution).
        assert config.quant is not None
        if isinstance(config.quant, dict):
            assert config.quant["weight_type"] == "uint8"
        else:
            assert config.quant.weight_type == "uint8"

    def test_override_none_is_noop(self, tmp_path) -> None:
        """override=None does not change the config."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("npu", ["npu", "cpu"]),
            ),
        ):
            config = generate_onnx_build_config(
                str(onnx_file),
                device="npu",
                override=None,
            )

        # Without override, raw+npu should have quant and compile
        assert config.quant is not None
        assert config.compile is not None

    # -----------------------------------------------------------------
    # Edge cases
    # -----------------------------------------------------------------

    def test_onnx_path_as_string(self, tmp_path) -> None:
        """String path is accepted and works correctly."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("cpu", ["cpu"]),
            ),
        ):
            config = generate_onnx_build_config(str(onnx_file))

        assert config.export is None

    def test_onnx_path_as_pathlib(self, tmp_path) -> None:
        """pathlib.Path object is accepted and works correctly."""
        from pathlib import Path

        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("cpu", ["cpu"]),
            ),
        ):
            config = generate_onnx_build_config(Path(onnx_file))

        assert config.export is None

    def test_auto_device_auto_precision_defaults(self, tmp_path) -> None:
        """device=auto + precision=auto (defaults) keeps config defaults.

        resolve_quant_compile_config returns (None, None) when both are auto,
        so raw ONNX gets quant=None, compile=None.
        """
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("auto", ["npu", "gpu", "cpu"]),
            ),
        ):
            config = generate_onnx_build_config(str(onnx_file))

        # Both auto -> resolve_precision returns device="auto" -> (None, None)
        assert config.quant is None
        assert config.compile is None

    def test_compiled_does_not_call_resolve_quant_compile(self, tmp_path) -> None:
        """Compiled model short-circuits before resolve_quant_compile_config."""
        onnx_file = tmp_path / "compiled.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=True),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.config.build.resolve_quant_compile_config",
            ) as mock_resolve,
        ):
            generate_onnx_build_config(str(onnx_file), device="npu")

        mock_resolve.assert_not_called()

    def test_raw_onnx_with_gpu(self, tmp_path) -> None:
        """Raw ONNX + device=gpu resolves quant=None, compile=dml."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("gpu", ["gpu", "cpu"]),
            ),
        ):
            config = generate_onnx_build_config(str(onnx_file), device="gpu")

        # GPU auto-precision is fp16 -> no quantization, compile=dml
        assert config.quant is None
        assert config.compile is not None
        assert config.compile.ep_config.provider == "dml"

    def test_ep_override_forwarded(self, tmp_path) -> None:
        """Explicit ep parameter is forwarded to resolve_quant_compile_config."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        with (
            patch("winml.modelkit.onnx.is_compiled_onnx", return_value=False),
            patch("winml.modelkit.onnx.is_quantized_onnx", return_value=False),
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("gpu", ["gpu", "cpu"]),
            ),
        ):
            config = generate_onnx_build_config(
                str(onnx_file),
                device="gpu",
                ep="migraphx",
            )

        assert config.compile is not None
        assert config.compile.ep_config.provider == "migraphx"


# =============================================================================
# TestResolveQuantCompileConfig - Tests for the standalone resolver
# =============================================================================


class TestResolveQuantCompileConfig:
    """Tests for resolve_quant_compile_config() standalone function.

    This tests the shared device/precision resolution logic used by both
    the HF and ONNX build config paths.
    """

    def test_auto_auto_returns_none_none(self) -> None:
        """device=auto + precision=auto returns (None, None)."""
        with patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("auto", ["npu", "gpu", "cpu"]),
        ):
            quant, compile_cfg = resolve_quant_compile_config()

        assert quant is None
        assert compile_cfg is None

    def test_npu_returns_quant_and_compile(self) -> None:
        """device=npu returns (WinMLQuantizationConfig, WinMLCompileConfig)."""
        with patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("npu", ["npu", "cpu"]),
        ):
            quant, compile_cfg = resolve_quant_compile_config(device="npu")

        assert isinstance(quant, WinMLQuantizationConfig)
        assert quant.weight_type == "uint8"
        assert quant.activation_type == "uint16"
        assert isinstance(compile_cfg, WinMLCompileConfig)
        assert compile_cfg.ep_config.provider == "qnn"

    def test_gpu_returns_none_quant_and_dml_compile(self) -> None:
        """device=gpu returns (None, WinMLCompileConfig(dml))."""
        with patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("gpu", ["gpu", "cpu"]),
        ):
            quant, compile_cfg = resolve_quant_compile_config(device="gpu")

        assert quant is None
        assert isinstance(compile_cfg, WinMLCompileConfig)
        assert compile_cfg.ep_config.provider == "dml"

    def test_cpu_returns_none_none(self) -> None:
        """device=cpu returns (None, None) since CPU has no compile provider."""
        with patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("cpu", ["cpu"]),
        ):
            quant, compile_cfg = resolve_quant_compile_config(device="cpu")

        assert quant is None
        assert compile_cfg is None

    def test_ep_override_changes_provider(self) -> None:
        """Explicit ep overrides the default device-to-provider mapping."""
        with patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("gpu", ["gpu", "cpu"]),
        ):
            _quant, compile_cfg = resolve_quant_compile_config(
                device="gpu",
                ep="nv_tensorrt_rtx",
            )

        assert compile_cfg is not None
        assert compile_cfg.ep_config.provider == "nv_tensorrt_rtx"

    def test_task_forwarded_to_resolve_precision(self) -> None:
        """task parameter is forwarded to resolve_precision.

        Patch at the source module since it is imported locally inside
        resolve_quant_compile_config.
        """
        with (
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("gpu", ["gpu", "cpu"]),
            ),
            patch(
                "winml.modelkit.config.precision.resolve_precision",
                wraps=__import__(
                    "winml.modelkit.config.precision", fromlist=["resolve_precision"]
                ).resolve_precision,
            ) as mock_prec,
        ):
            resolve_quant_compile_config(device="gpu", task="text-generation")

        mock_prec.assert_called_once()
        assert mock_prec.call_args.kwargs.get("task") == "text-generation"

    def test_explicit_int8_precision_on_npu(self) -> None:
        """Explicit precision=int8 on npu produces uint8 quant."""
        with patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("npu", ["npu", "cpu"]),
        ):
            quant, _compile_cfg = resolve_quant_compile_config(
                device="npu",
                precision="int8",
            )

        assert quant is not None
        assert quant.weight_type == "uint8"
        assert quant.activation_type == "uint8"

    def test_explicit_fp32_precision_no_quant(self) -> None:
        """Explicit precision=fp32 produces no quantization."""
        with patch(
            "winml.modelkit.sysinfo.resolve_device",
            return_value=("gpu", ["gpu", "cpu"]),
        ):
            quant, _compile_cfg = resolve_quant_compile_config(
                device="gpu",
                precision="fp32",
            )

        assert quant is None
