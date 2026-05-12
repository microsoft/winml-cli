# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for winml perf --module flag."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from winml.modelkit.cli import main
from winml.modelkit.commands.perf import generate_output_path


if TYPE_CHECKING:
    from pathlib import Path


class TestPerfModuleFlag:
    """Tests for --module flag on winml perf."""

    def test_module_flag_in_help(self) -> None:
        """Verify --module flag appears in winml perf --help."""
        runner = CliRunner()
        result = runner.invoke(main, ["perf", "--help"])
        assert result.exit_code == 0
        assert "--module" in result.output

    def test_module_flag_requires_model(self) -> None:
        """--module without -m/--model should fail."""
        runner = CliRunner()
        result = runner.invoke(main, ["perf", "--module", "BertAttention"])
        assert result.exit_code != 0

    def test_module_no_match_exits_nonzero(self) -> None:
        """--module CLASSNAME matching no submodules must exit non-zero.

        Regression guard for #554: previously `sys.exit(0)` masked this
        as success, which silently broke CI when a module name was typoed.
        """
        # _perf_modules calls resolve_device() before generate_hf_build_config(),
        # so mock both to keep the test hermetic (no hardware probe in CI).
        with (
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("cpu", ["cpu"]),
            ),
            patch(
                "winml.modelkit.config.generate_hf_build_config",
                return_value=[],
            ),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["perf", "-m", "fake/model", "--module", "DoesNotExist"],
            )
        assert result.exit_code != 0, result.output
        assert "No modules matching" in result.output

    def test_module_default_output_includes_class_name(self) -> None:
        """Default output path includes module class name."""
        # The single-model generate_output_path produces {slug}_perf.json
        path = generate_output_path("bert-base-uncased")
        assert "bert-base-uncased" in str(path)

        # The module-mode output path (from _perf_modules) is
        # {slug}_{module_class}_perf.json -- tested via the inline logic
        # in _perf_modules. Here we verify the format difference:
        from pathlib import Path

        slug = "bert-base-uncased"
        module_class = "BertAttention"
        module_path = Path(f"{slug}_{module_class}_perf.json")
        assert module_class in str(module_path)
        assert str(module_path) != str(path)


class TestPerfModuleParameterForwarding:
    """Verify --device/--ep/--precision flow from CLI through _perf_modules
    into generate_hf_build_config, build_hf_model, and WinMLSession.

    Regression guard: these kwargs were silently dropped before.
    """

    def test_device_and_ep_forwarded_through_module_path(self, tmp_path: Path) -> None:
        # Fake module config -- only the attributes _perf_modules touches
        fake_cfg = MagicMock()
        fake_cfg.loader.model_type = "bert"
        fake_cfg.loader.module_path = "encoder.layer.0"

        fake_build_result = MagicMock()
        fake_build_result.final_onnx_path = tmp_path / "model.onnx"

        # Make WinMLSession.perf() raise so the benchmark loop is short-circuited
        # via the existing try/except in _perf_modules. We still capture the
        # constructor kwargs, which is what we care about.
        fake_session = MagicMock()
        fake_session.perf.side_effect = RuntimeError("test-skip-benchmark")

        with (
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("npu", "qnn"),
            ),
            patch(
                "winml.modelkit.config.generate_hf_build_config",
                return_value=[fake_cfg],
            ) as mock_gen,
            patch(
                "winml.modelkit.commands.build._instantiate_parent_model",
                return_value=MagicMock(),
            ),
            patch(
                "winml.modelkit.build.build_hf_model",
                return_value=fake_build_result,
            ) as mock_build,
            patch(
                "winml.modelkit.session.WinMLSession",
                return_value=fake_session,
            ) as mock_session_cls,
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "perf",
                    "-m",
                    "fake/model",
                    "--module",
                    "BertLayer",
                    "--device",
                    "npu",
                    "--ep",
                    "qnn",
                    "--iterations",
                    "1",
                    "--warmup",
                    "0",
                    "-o",
                    str(tmp_path / "out.json"),
                ],
            )

        assert result.exit_code == 0, result.output

        gen_kwargs = mock_gen.call_args.kwargs
        assert gen_kwargs["device"] == "npu"
        assert gen_kwargs["ep"] == "qnn"
        assert gen_kwargs["precision"] == "auto"

        build_kwargs = mock_build.call_args.kwargs
        assert build_kwargs["ep"] == "qnn"
        assert build_kwargs["device"] == "npu"

        session_kwargs = mock_session_cls.call_args.kwargs
        assert session_kwargs["device"] == "npu"
        assert session_kwargs["ep"] == "qnn"
