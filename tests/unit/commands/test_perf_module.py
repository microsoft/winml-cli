# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for winml perf --module flag."""

from __future__ import annotations

import json
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

    def test_module_with_onnx_path_rejected(self, tmp_path: Path) -> None:
        """--module on a .onnx path must fail with a clear UsageError.

        Regression guard for #553: previously the CLI tried to load the
        ONNX file as an HF config and surfaced a confusing "not a valid
        JSON file" error.
        """
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake")

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["perf", "-m", str(onnx_file), "--module", "NoSuchClass"],
        )
        assert result.exit_code == 2, result.output
        assert "--module is not supported for ONNX files" in result.output
        # Specifically must NOT blame the model file with a JSON-config error.
        assert "valid JSON" not in result.output

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

    def test_module_no_match_lists_available_classes(self) -> None:
        """SubmoduleClassNotFoundError surfaces the available class names
        plus a `Did you mean…?` suggestion."""
        from winml.modelkit.config import SubmoduleClassNotFoundError

        with (
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("cpu", ["cpu"]),
            ),
            patch(
                "winml.modelkit.config.generate_hf_build_config",
                side_effect=SubmoduleClassNotFoundError(
                    "ResNetStag",  # typo of ResNetStage
                    ["Conv2d", "Linear", "ResNetStage", "ResNetBottleNeckLayer"],
                ),
            ),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["perf", "-m", "fake/model", "--module", "ResNetStag"],
            )
        assert result.exit_code != 0, result.output
        assert "No modules matching 'ResNetStag'" in result.output
        # Close-match suggestion (difflib should pick ResNetStage).
        assert "Did you mean" in result.output
        assert "ResNetStage" in result.output
        # Full list also shown.
        assert "Available module class names" in result.output
        assert "Conv2d" in result.output
        assert "Linear" in result.output

    def test_module_default_output_includes_class_name(self) -> None:
        """Default output path includes the model slug and module class name."""
        # Single-model layout: ~/.cache/winml/perf/<slug>/<timestamp>.json
        plain = generate_output_path("bert-base-uncased")
        assert "bert-base-uncased" in str(plain)

        # Module-mode layout: ~/.cache/winml/perf/<slug>/<module_class>/<timestamp>.json
        module_path = generate_output_path("bert-base-uncased", module_class="BertAttention")
        assert "bert-base-uncased" in str(module_path)
        assert "BertAttention" in str(module_path)
        # Module-mode is nested one level deeper than plain.
        assert module_path.parent.parent == plain.parent


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

        # _perf_modules calls resolve_loader_config(model_id=...) to recover the
        # parent task (submodule configs strip it). Stub it so "fake/model" never
        # hits the HF Hub.
        fake_loader_cfg = MagicMock()
        fake_loader_cfg.task = "fill-mask"

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
                "winml.modelkit.loader.resolve_loader_config",
                return_value=(fake_loader_cfg, MagicMock(), MagicMock()),
            ),
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

    def test_running_model_path_in_module_result(self, tmp_path: Path) -> None:
        """A completed module benchmark records running_model_path in its
        per-instance result entry.

        Unlike the forwarding test above (which short-circuits the benchmark
        loop via a RuntimeError), this drives a successful run so result_entry
        is actually populated, then reads it back from the JSON report.
        """
        fake_cfg = MagicMock()
        fake_cfg.loader.model_type = "bert"
        fake_cfg.loader.module_path = "encoder.layer.0"

        fake_build_result = MagicMock()
        fake_build_result.final_onnx_path = tmp_path / "model.onnx"

        # Stats yielded by `with session.perf(...) as stats` — needs real
        # numbers since result_entry rounds/divides them.
        fake_stats = MagicMock()
        fake_stats.mean_ms = 1.0
        fake_stats.p50_ms = 1.0
        fake_stats.p90_ms = 1.0
        fake_stats.p95_ms = 1.0
        fake_stats.p99_ms = 1.0
        fake_stats.min_ms = 1.0
        fake_stats.max_ms = 1.0
        fake_stats.samples_ms = [1.0, 1.0]

        running_model_path = tmp_path / "model_cpu_ctx.onnx"
        fake_session = MagicMock()
        fake_session.perf.return_value.__enter__.return_value = fake_stats
        fake_session.running_model_path = running_model_path

        fake_loader_cfg = MagicMock()
        fake_loader_cfg.task = "fill-mask"

        out_path = tmp_path / "out.json"

        with (
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("npu", "qnn"),
            ),
            patch(
                "winml.modelkit.config.generate_hf_build_config",
                return_value=[fake_cfg],
            ),
            patch(
                "winml.modelkit.loader.resolve_loader_config",
                return_value=(fake_loader_cfg, MagicMock(), MagicMock()),
            ),
            patch(
                "winml.modelkit.commands.build._instantiate_parent_model",
                return_value=MagicMock(),
            ),
            patch(
                "winml.modelkit.build.build_hf_model",
                return_value=fake_build_result,
            ),
            patch(
                "winml.modelkit.session.WinMLSession",
                return_value=fake_session,
            ),
            patch(
                "winml.modelkit.commands.perf.generate_random_inputs",
                return_value={},
            ),
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
                    str(out_path),
                ],
            )

        assert result.exit_code == 0, result.output

        report = json.loads(out_path.read_text(encoding="utf-8"))
        instance = report["instances"][0]
        assert instance["running_model_path"] == str(running_model_path)


class TestPerfModuleQuantCompileToggles:
    """--no-quantize and --compile/--no-compile clear cfg.quant / cfg.compile
    independently in the per-module build (mirrors the single-model path)."""

    @staticmethod
    def _run(tmp_path: Path, extra_args: list[str]) -> MagicMock:
        """Invoke ``perf --module`` with mocked build and return the module cfg.

        The cfg is mutated (quant/compile cleared) before ``build_hf_model``,
        so short-circuiting the benchmark via a failing ``session.perf()``
        still lets us inspect the mutation.
        """
        fake_cfg = MagicMock()
        fake_cfg.loader.model_type = "bert"
        fake_cfg.loader.module_path = "encoder.layer.0"

        fake_build_result = MagicMock()
        fake_build_result.final_onnx_path = tmp_path / "model.onnx"

        fake_session = MagicMock()
        fake_session.perf.side_effect = RuntimeError("test-skip-benchmark")

        fake_loader_cfg = MagicMock()
        fake_loader_cfg.task = "fill-mask"

        with (
            patch(
                "winml.modelkit.sysinfo.resolve_device",
                return_value=("cpu", ["cpu"]),
            ),
            patch(
                "winml.modelkit.config.generate_hf_build_config",
                return_value=[fake_cfg],
            ),
            patch(
                "winml.modelkit.loader.resolve_loader_config",
                return_value=(fake_loader_cfg, MagicMock(), MagicMock()),
            ),
            patch(
                "winml.modelkit.commands.build._instantiate_parent_model",
                return_value=MagicMock(),
            ),
            patch(
                "winml.modelkit.build.build_hf_model",
                return_value=fake_build_result,
            ),
            patch(
                "winml.modelkit.session.WinMLSession",
                return_value=fake_session,
            ),
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
                    "--iterations",
                    "1",
                    "--warmup",
                    "0",
                    "-o",
                    str(tmp_path / "out.json"),
                    *extra_args,
                ],
            )
        assert result.exit_code == 0, result.output
        return fake_cfg

    def test_default_skips_compile_keeps_quant(self, tmp_path: Path) -> None:
        # perf defaults to --no-compile and --quantize.
        cfg = self._run(tmp_path, [])
        assert cfg.compile is None
        assert cfg.quant is not None

    def test_compile_flag_preserves_compile(self, tmp_path: Path) -> None:
        cfg = self._run(tmp_path, ["--compile"])
        assert cfg.compile is not None
        assert cfg.quant is not None

    def test_no_quantize_clears_only_quant(self, tmp_path: Path) -> None:
        # --no-quantize must not also clear compile when --compile is set.
        cfg = self._run(tmp_path, ["--no-quantize", "--compile"])
        assert cfg.quant is None
        assert cfg.compile is not None
