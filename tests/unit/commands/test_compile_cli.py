# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for winml compile CLI -- device + ep flag parsing.

These tests verify that the --ep and --device flags are accepted, optional,
and that resolve_device() is called at the CLI boundary with the correct args.
No actual compilation or hardware EP registration is needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.compile import compile
from winml.modelkit.session import EPDeviceTarget


if TYPE_CHECKING:
    from pathlib import Path


def _fake_ep_device(ep: str, device: str) -> EPDeviceTarget:
    """Build a stub EPDeviceTarget for mocking resolve_device()."""
    return EPDeviceTarget(ep=ep, device=device)


@dataclass(frozen=True)
class CompileCliMocks:
    """Patched compile-command collaborators exposed to tests."""

    compile_onnx: MagicMock
    compile_config: MagicMock
    resolve_device: MagicMock


def _assert_successful_compile_call(
    result,
    compile_cli_mocks: CompileCliMocks,
    model_path: Path,
) -> None:
    """Assert a successful CLI invocation compiled the requested input model."""
    assert result.exit_code == 0, result.output
    compile_cli_mocks.compile_onnx.assert_called_once_with(
        model_path,
        output_path=None,
        config=compile_cli_mocks.compile_config,
    )


@pytest.fixture
def runner() -> CliRunner:
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def fake_onnx(tmp_path: Path) -> Path:
    """Write a minimal (fake) ONNX file so the CLI's exists= check passes."""
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fake-onnx")
    return model


@pytest.fixture
def compile_cli_mocks() -> CompileCliMocks:
    """Prevent any real EP discovery or compilation; return a successful result stub."""
    result = MagicMock()
    result.success = True
    result.output_path = None
    result.compile_time = None
    result.total_time = None
    result.errors = []

    # Default: resolve_device returns a QNN/NPU EPDeviceTarget.
    default_ep_device = _fake_ep_device("QNNExecutionProvider", "npu")
    compile_config = MagicMock()
    compile_config.ep_device = default_ep_device
    compile_config.validate = True
    compile_config.verbose = False
    compile_config.ep_config.compiler = "ort"
    compile_config.ep_config.qnn_sdk_root = None
    compile_config.ep_config.embed_context = False
    compile_config.ep_config.provider_options = {}
    compile_config.ep_config.provider_option_file_keys = set()
    compile_config.ep_config.enable_ep_context = False

    def _build_compile_config(ep_device: EPDeviceTarget) -> MagicMock:
        compile_config.ep_device = ep_device
        return compile_config

    with (
        patch(
            "winml.modelkit.commands.compile.resolve_device",
            return_value=default_ep_device,
        ) as mock_resolve,
        patch("winml.modelkit.commands.compile.is_compiled_onnx", return_value=False),
        patch("winml.modelkit.compiler.compile_onnx", return_value=result) as mock_compile_onnx,
        patch(
            "winml.modelkit.compiler.WinMLCompileConfig.for_ep_device",
            side_effect=_build_compile_config,
        ),
    ):
        yield CompileCliMocks(
            compile_onnx=mock_compile_onnx,
            compile_config=compile_config,
            resolve_device=mock_resolve,
        )


# =============================================================================
# CLI --help smoke tests
# =============================================================================


class TestCompileCliHelp:
    """Verify compile options appear in --help output."""

    def test_device_option_in_help(self, runner: CliRunner) -> None:
        result = runner.invoke(compile, ["--help"])
        assert result.exit_code == 0
        assert "--device" in result.output

    def test_ep_option_in_help(self, runner: CliRunner) -> None:
        result = runner.invoke(compile, ["--help"])
        assert result.exit_code == 0
        assert "--ep" in result.output

    def test_ep_options_option_in_help(self, runner: CliRunner) -> None:
        result = runner.invoke(compile, ["--help"])
        assert result.exit_code == 0
        assert "--ep-options" in result.output

    def test_device_choices_in_help(self, runner: CliRunner) -> None:
        """Help text must expose the device choices."""
        result = runner.invoke(compile, ["--help"])
        assert result.exit_code == 0
        # At least one of the valid choices must appear in the help text
        assert any(c in result.output for c in ["npu", "gpu", "cpu"])


# =============================================================================
# CLI invocation tests (no real compilation)
# =============================================================================


class TestCompileCliDeviceEpFlags:
    """Verify CLI accepts --ep/--device combinations and deduces correctly."""

    def test_ep_qnn_no_device_accepted(
        self, runner: CliRunner, fake_onnx: Path, compile_cli_mocks: CompileCliMocks
    ) -> None:
        """--ep qnn without --device is accepted (device deduced from ep)."""
        r = runner.invoke(compile, ["-m", str(fake_onnx), "--ep", "qnn"])
        _assert_successful_compile_call(
            r,
            compile_cli_mocks,
            fake_onnx,
        )

    def test_device_npu_no_ep_accepted(
        self, runner: CliRunner, fake_onnx: Path, compile_cli_mocks: CompileCliMocks
    ) -> None:
        """--device npu without --ep is accepted (ep deduced from device)."""
        r = runner.invoke(compile, ["-m", str(fake_onnx), "--device", "npu"])
        _assert_successful_compile_call(
            r,
            compile_cli_mocks,
            fake_onnx,
        )

    def test_neither_ep_nor_device_accepted(
        self, runner: CliRunner, fake_onnx: Path, compile_cli_mocks: CompileCliMocks
    ) -> None:
        """Omitting both --ep and --device is accepted (defaults to qnn/npu)."""
        r = runner.invoke(compile, ["-m", str(fake_onnx)])
        _assert_successful_compile_call(
            r,
            compile_cli_mocks,
            fake_onnx,
        )

    @patch("winml.modelkit.compiler.list_compilers", return_value="QAIRT")
    def test_list_uses_resolved_provider_name(
        self,
        mock_list_compilers: MagicMock,
        runner: CliRunner,
        compile_cli_mocks: CompileCliMocks,
    ) -> None:
        """--list must pass the resolved canonical provider name to list_compilers()."""
        result = runner.invoke(compile, ["--list", "--device", "npu", "--ep", "qnn"])

        assert result.exit_code == 0, result.output
        mock_list_compilers.assert_called_once_with("QNNExecutionProvider")
        assert result.output == "QAIRT\n"
        compile_cli_mocks.compile_onnx.assert_not_called()

    def test_device_npu_shows_npu_in_output(
        self, runner: CliRunner, fake_onnx: Path, compile_cli_mocks: CompileCliMocks
    ) -> None:
        """--device npu → 'Device: npu' appears in output."""
        compile_cli_mocks.resolve_device.return_value = _fake_ep_device(
            "QNNExecutionProvider",
            "npu",
        )
        r = runner.invoke(compile, ["-m", str(fake_onnx), "--device", "npu"])
        assert "Device: npu" in r.output

    def test_device_gpu_shows_gpu_in_output(
        self, runner: CliRunner, fake_onnx: Path, compile_cli_mocks: CompileCliMocks
    ) -> None:
        """--device gpu → 'Device: gpu' appears in output."""
        compile_cli_mocks.resolve_device.return_value = _fake_ep_device(
            "DmlExecutionProvider",
            "gpu",
        )
        r = runner.invoke(compile, ["-m", str(fake_onnx), "--device", "gpu"])
        assert "Device: gpu" in r.output

    def test_ep_dml_shows_gpu_in_output(
        self, runner: CliRunner, fake_onnx: Path, compile_cli_mocks: CompileCliMocks
    ) -> None:
        """--ep dml → 'Device: gpu' because dml is a GPU EP."""
        compile_cli_mocks.resolve_device.return_value = _fake_ep_device(
            "DmlExecutionProvider",
            "gpu",
        )
        r = runner.invoke(compile, ["-m", str(fake_onnx), "--ep", "dml"])
        assert "Device: gpu" in r.output

    def test_resolve_device_called_at_boundary(
        self, runner: CliRunner, fake_onnx: Path, compile_cli_mocks: CompileCliMocks
    ) -> None:
        """resolve_device() is called exactly once with the CLI args."""
        runner.invoke(compile, ["-m", str(fake_onnx), "--ep", "qnn", "--device", "npu"])
        compile_cli_mocks.resolve_device.assert_called_once_with(
            EPDeviceTarget(ep="qnn", device="npu")
        )

    def test_resolve_device_called_ep_only(
        self, runner: CliRunner, fake_onnx: Path, compile_cli_mocks: CompileCliMocks
    ) -> None:
        """--ep only: resolve_device(EPDeviceTarget(ep=..., device='auto'))."""
        runner.invoke(compile, ["-m", str(fake_onnx), "--ep", "vitisai"])
        compile_cli_mocks.resolve_device.assert_called_once_with(
            EPDeviceTarget(ep="vitisai", device="auto")
        )

    def test_resolve_device_called_device_only(
        self, runner: CliRunner, fake_onnx: Path, compile_cli_mocks: CompileCliMocks
    ) -> None:
        """--device only: resolve_device(EPDeviceTarget(ep='auto', device=...))."""
        runner.invoke(compile, ["-m", str(fake_onnx), "--device", "gpu"])
        compile_cli_mocks.resolve_device.assert_called_once_with(
            EPDeviceTarget(ep="auto", device="gpu")
        )

    def test_resolve_device_called_neither(
        self, runner: CliRunner, fake_onnx: Path, compile_cli_mocks: CompileCliMocks
    ) -> None:
        """Neither --ep nor --device: resolve_device(EPDeviceTarget(ep='auto', device='auto'))."""
        runner.invoke(compile, ["-m", str(fake_onnx)])
        compile_cli_mocks.resolve_device.assert_called_once_with(
            EPDeviceTarget(ep="auto", device="auto")
        )

    def test_device_auto_treated_as_none(
        self, runner: CliRunner, fake_onnx: Path, compile_cli_mocks: CompileCliMocks
    ) -> None:
        """--device auto: passes EPDeviceTarget(ep='auto', device='auto')."""
        runner.invoke(compile, ["-m", str(fake_onnx), "--device", "auto"])
        compile_cli_mocks.resolve_device.assert_called_once_with(
            EPDeviceTarget(ep="auto", device="auto")
        )

    def test_invalid_device_rejected(self, runner: CliRunner, fake_onnx: Path) -> None:
        """Unknown device string is rejected by Click."""
        result = runner.invoke(compile, ["-m", str(fake_onnx), "--device", "tpu"])
        assert result.exit_code != 0

    def test_invalid_ep_rejected(self, runner: CliRunner, fake_onnx: Path) -> None:
        """Unknown EP string is rejected by Click."""
        result = runner.invoke(compile, ["-m", str(fake_onnx), "--ep", "unknown_ep"])
        assert result.exit_code != 0


class TestCompileCliEpOptions:
    """Verify CLI EP options reach the compilation provider configuration."""

    def test_ep_options_forwarded_to_compile_config(
        self, runner: CliRunner, fake_onnx: Path, compile_cli_mocks: CompileCliMocks
    ) -> None:
        result = runner.invoke(
            compile,
            [
                "-m",
                str(fake_onnx),
                "--ep-options",
                "htp_performance_mode=burst",
                "--ep-options",
                "soc_model=57",
            ],
        )

        _assert_successful_compile_call(result, compile_cli_mocks, fake_onnx)
        assert compile_cli_mocks.compile_config.ep_config.provider_options == {
            "htp_performance_mode": "burst",
            "soc_model": "57",
        }

    def test_cli_ep_options_override_matching_config_keys(
        self, runner: CliRunner, fake_onnx: Path, compile_cli_mocks: CompileCliMocks
    ) -> None:
        config_path = fake_onnx.parent / "compile.json"
        config_path.write_text(
            json.dumps(
                {
                    "compile": {
                        "provider_options": {
                            "htp_performance_mode": "sustained_high_performance",
                            "soc_model": "57",
                        }
                    }
                }
            )
        )

        result = runner.invoke(
            compile,
            [
                "-m",
                str(fake_onnx),
                "--config",
                str(config_path),
                "--ep-options",
                "htp_performance_mode=burst",
            ],
        )

        _assert_successful_compile_call(result, compile_cli_mocks, fake_onnx)
        assert compile_cli_mocks.compile_config.ep_config.provider_options == {
            "htp_performance_mode": "burst",
            "soc_model": "57",
        }

    def test_config_provider_option_file_keys_reach_compile_config(
        self,
        runner: CliRunner,
        fake_onnx: Path,
        compile_cli_mocks: CompileCliMocks,
    ) -> None:
        """Compile config preserves explicit file-backed provider option metadata."""
        dependency = fake_onnx.parent / "compiler-input.bin"
        dependency.write_bytes(b"compiler input")
        config_path = fake_onnx.parent / "compile.json"
        config_path.write_text(
            json.dumps(
                {
                    "compile": {
                        "provider_options": {"compiler_input": str(dependency)},
                        "provider_option_file_keys": ["compiler_input"],
                    }
                }
            )
        )

        result = runner.invoke(
            compile,
            ["-m", str(fake_onnx), "--config", str(config_path)],
        )

        _assert_successful_compile_call(result, compile_cli_mocks, fake_onnx)
        assert compile_cli_mocks.compile_config.ep_config.provider_option_file_keys == {
            "compiler_input"
        }

    def test_invalid_ep_option_is_rejected(
        self, runner: CliRunner, fake_onnx: Path, compile_cli_mocks: CompileCliMocks
    ) -> None:
        result = runner.invoke(
            compile,
            ["-m", str(fake_onnx), "--ep-options", "missing-separator"],
        )

        assert result.exit_code == 2
        assert "Use KEY=VALUE" in result.output
        compile_cli_mocks.compile_onnx.assert_not_called()


class TestCompileCliConfigPriority:
    """Verify compile CLI device precedence across CLI and JSON config."""

    def test_config_device_used_when_cli_device_omitted(
        self, runner: CliRunner, fake_onnx: Path, compile_cli_mocks: CompileCliMocks
    ) -> None:
        """compile.device from JSON config must feed resolve_device when CLI omits --device."""
        config_path = fake_onnx.parent / "compile.json"
        config_path.write_text('{"compile": {"device": "gpu"}}')

        result = runner.invoke(compile, ["-m", str(fake_onnx), "--config", str(config_path)])

        _assert_successful_compile_call(result, compile_cli_mocks, fake_onnx)
        compile_cli_mocks.resolve_device.assert_called_once_with(
            EPDeviceTarget(ep="auto", device="gpu")
        )

    def test_cli_device_overrides_config_device(
        self, runner: CliRunner, fake_onnx: Path, compile_cli_mocks: CompileCliMocks
    ) -> None:
        """Explicit CLI --device must continue to override compile.device from JSON."""
        config_path = fake_onnx.parent / "compile.json"
        config_path.write_text('{"compile": {"device": "gpu"}}')

        result = runner.invoke(
            compile,
            ["-m", str(fake_onnx), "--config", str(config_path), "--device", "npu"],
        )

        _assert_successful_compile_call(result, compile_cli_mocks, fake_onnx)
        compile_cli_mocks.resolve_device.assert_called_once_with(
            EPDeviceTarget(ep="auto", device="npu")
        )

    def test_serialized_config_target_preserves_source(
        self, runner: CliRunner, fake_onnx: Path, compile_cli_mocks: CompileCliMocks
    ) -> None:
        """A serialized target source must survive the compile CLI config merge."""
        target = EPDeviceTarget(
            ep="OpenVINOExecutionProvider",
            device="npu",
            source="pypi",
        )
        config_path = fake_onnx.parent / "compile.json"
        config_path.write_text(
            json.dumps(
                {
                    "compile": {
                        "execution_provider": "openvino",
                        "device": "npu",
                        "ep_device": target.to_dict(),
                    }
                }
            )
        )

        result = runner.invoke(compile, ["-m", str(fake_onnx), "--config", str(config_path)])

        _assert_successful_compile_call(result, compile_cli_mocks, fake_onnx)
        compile_cli_mocks.resolve_device.assert_called_once_with(
            EPDeviceTarget(ep="openvino", device="npu", source="pypi")
        )

    def test_invalid_config_device_type_is_usage_error(
        self, runner: CliRunner, fake_onnx: Path, compile_cli_mocks: CompileCliMocks
    ) -> None:
        """Non-string JSON devices must fail as user input, not an internal exception."""
        config_path = fake_onnx.parent / "compile.json"
        config_path.write_text(json.dumps({"compile": {"device": 1}}))

        result = runner.invoke(compile, ["-m", str(fake_onnx), "--config", str(config_path)])

        assert result.exit_code == 2
        assert "compile.device must be a string" in result.output
        compile_cli_mocks.resolve_device.assert_not_called()

    def test_cli_ep_source_overrides_serialized_config_source(
        self, runner: CliRunner, fake_onnx: Path, compile_cli_mocks: CompileCliMocks
    ) -> None:
        """An explicit CLI EP replaces both EP and source from serialized config."""
        config_target = EPDeviceTarget(
            ep="OpenVINOExecutionProvider",
            device="npu",
            source="pypi",
        )
        config_path = fake_onnx.parent / "compile.json"
        config_path.write_text(
            json.dumps(
                {
                    "compile": {
                        "execution_provider": "openvino",
                        "device": "npu",
                        "ep_device": config_target.to_dict(),
                    }
                }
            )
        )

        result = runner.invoke(
            compile,
            [
                "-m",
                str(fake_onnx),
                "--config",
                str(config_path),
                "--ep",
                "qnn@nuget",
            ],
        )

        _assert_successful_compile_call(result, compile_cli_mocks, fake_onnx)
        compile_cli_mocks.resolve_device.assert_called_once_with(
            EPDeviceTarget(ep="qnn", device="npu", source="nuget")
        )
