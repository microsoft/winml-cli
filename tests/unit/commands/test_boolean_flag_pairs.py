# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner


if TYPE_CHECKING:
    import click

from winml.modelkit.commands.build import build
from winml.modelkit.commands.compile import compile
from winml.modelkit.commands.config import config
from winml.modelkit.commands.eval import eval as eval_cmd
from winml.modelkit.commands.export import export
from winml.modelkit.commands.inspect import inspect
from winml.modelkit.commands.perf import perf
from winml.modelkit.commands.quantize import quantize
from winml.modelkit.commands.serve import serve


@pytest.mark.parametrize(
    ("command", "flags"),
    [
        (
            build,
            [
                "--use-cache",
                "--no-use-cache",
                "--rebuild",
                "--no-rebuild",
                "--quant",
                "--no-quant",
                "--analyze",
                "--no-analyze",
                "--optimize",
                "--no-optimize",
                "--trust-remote-code",
                "--no-trust-remote-code",
                "--allow-unsupported-nodes",
                "--no-allow-unsupported-nodes",
            ],
        ),
        (compile, ["--embed", "--no-embed"]),
        (
            config,
            ["--quant", "--no-quant", "--trust-remote-code", "--no-trust-remote-code"],
        ),
        (
            eval_cmd,
            [
                "--quant",
                "--no-quant",
                "--quantize",
                "--no-quantize",
                "--optimize",
                "--no-optimize",
                "--analyze",
                "--no-analyze",
                "--streaming",
                "--no-streaming",
                "--allow-unsupported-nodes",
                "--no-allow-unsupported-nodes",
            ],
        ),
        (
            export,
            [
                "--with-report",
                "--no-with-report",
                "--hierarchy",
                "--no-hierarchy",
                "--dynamo",
                "--no-dynamo",
            ],
        ),
        (inspect, ["--hierarchy", "--no-hierarchy", "-H", "-N"]),
        (
            perf,
            [
                "--quant",
                "--no-quant",
                "--quantize",
                "--no-quantize",
                "--optimize",
                "--no-optimize",
                "--analyze",
                "--no-analyze",
                "--rebuild",
                "--no-rebuild",
                "--ignore-cache",
                "--no-ignore-cache",
                "--monitor",
                "--no-monitor",
                "--allow-unsupported-nodes",
                "--no-allow-unsupported-nodes",
            ],
        ),
        (
            quantize,
            [
                "--per-channel",
                "--no-per-channel",
                "--symmetric",
                "--no-symmetric",
                "--reduce-range",
                "--no-reduce-range",
            ],
        ),
        (serve, ["--multi", "--no-multi"]),
    ],
)
def test_help_includes_boolean_flag_pairs(command, flags: list[str]) -> None:
    result = CliRunner().invoke(command, ["--help"])

    assert result.exit_code == 0, result.output
    for flag in flags:
        assert flag in result.output


# ---------------------------------------------------------------------------
# Verify default values and that both positive/negative forms set correct values
# ---------------------------------------------------------------------------


def _get_param_default(command: click.Command, param_name: str) -> object:
    """Get the default value of a Click parameter by its Python name."""
    for param in command.params:
        if param.name == param_name:
            return param.default
    msg = f"No param {param_name!r} on {command.name}"
    raise ValueError(msg)


class TestDefaultValues:
    """Verify the default values for converted flags are correct."""

    @pytest.mark.parametrize(
        ("command", "param_name", "expected_default"),
        [
            # Group A: positive flags default False
            (build, "use_cache", False),
            (build, "rebuild", False),
            (compile, "embed", False),
            (eval_cmd, "streaming", False),
            (export, "with_report", False),
            (inspect, "hierarchy", False),
            (perf, "rebuild", False),
            (perf, "ignore_cache", False),
            (perf, "monitor", False),
            (quantize, "per_channel", False),
            (quantize, "symmetric", False),
            (quantize, "reduce_range", False),
            (serve, "multi", False),
            (serve, "auto_reload", False),
            # Group C: negative-to-positive flags default True
            (build, "quant", True),
            (build, "analyze", True),
            (build, "optimize", True),
            (config, "quant", True),
            (perf, "quant", True),
            (perf, "optimize", True),
            (perf, "analyze", True),
            (eval_cmd, "quant", True),
            (eval_cmd, "optimize", True),
            (eval_cmd, "analyze", True),
            (export, "hierarchy", True),
            (export, "dynamo", True),
        ],
    )
    def test_default_value(self, command, param_name: str, expected_default) -> None:
        assert _get_param_default(command, param_name) == expected_default


class TestFlagValueParsing:
    """Verify that positive and negative flag forms set the correct values.

    Uses Click's make_context with resilient_parsing to parse flags and inspect
    ctx.params directly, without running the full command logic.
    """

    @pytest.mark.parametrize(
        ("command", "flag", "param_name", "expected_value"),
        [
            # Positive form sets True
            (build, "--use-cache", "use_cache", True),
            (build, "--rebuild", "rebuild", True),
            (build, "--quant", "quant", True),
            (build, "--analyze", "analyze", True),
            (build, "--optimize", "optimize", True),
            (compile, "--embed", "embed", True),
            (eval_cmd, "--streaming", "streaming", True),
            (export, "--with-report", "with_report", True),
            (export, "--hierarchy", "hierarchy", True),
            (export, "--dynamo", "dynamo", True),
            (inspect, "--hierarchy", "hierarchy", True),
            (inspect, "-H", "hierarchy", True),
            (perf, "--quant", "quant", True),
            (perf, "--quantize", "quant", True),
            (perf, "--optimize", "optimize", True),
            (perf, "--analyze", "analyze", True),
            (eval_cmd, "--quant", "quant", True),
            (eval_cmd, "--quantize", "quant", True),
            (eval_cmd, "--optimize", "optimize", True),
            (eval_cmd, "--analyze", "analyze", True),
            (perf, "--rebuild", "rebuild", True),
            (perf, "--ignore-cache", "ignore_cache", True),
            (perf, "--monitor", "monitor", True),
            (quantize, "--per-channel", "per_channel", True),
            (quantize, "--symmetric", "symmetric", True),
            (quantize, "--reduce-range", "reduce_range", True),
            (serve, "--multi", "multi", True),
            (serve, "--auto-reload", "auto_reload", True),
            # Negative form sets False
            (build, "--no-use-cache", "use_cache", False),
            (build, "--no-rebuild", "rebuild", False),
            (build, "--no-quant", "quant", False),
            (build, "--no-analyze", "analyze", False),
            (build, "--no-optimize", "optimize", False),
            (compile, "--no-embed", "embed", False),
            (eval_cmd, "--no-streaming", "streaming", False),
            (export, "--no-with-report", "with_report", False),
            (export, "--no-hierarchy", "hierarchy", False),
            (export, "--no-dynamo", "dynamo", False),
            (inspect, "--no-hierarchy", "hierarchy", False),
            (inspect, "-N", "hierarchy", False),
            (perf, "--no-quant", "quant", False),
            (perf, "--no-quantize", "quant", False),
            (perf, "--no-optimize", "optimize", False),
            (perf, "--no-analyze", "analyze", False),
            (eval_cmd, "--no-quant", "quant", False),
            (eval_cmd, "--no-quantize", "quant", False),
            (eval_cmd, "--no-optimize", "optimize", False),
            (eval_cmd, "--no-analyze", "analyze", False),
            (perf, "--no-rebuild", "rebuild", False),
            (perf, "--no-ignore-cache", "ignore_cache", False),
            (perf, "--no-monitor", "monitor", False),
            (quantize, "--no-per-channel", "per_channel", False),
            (quantize, "--no-symmetric", "symmetric", False),
            (quantize, "--no-reduce-range", "reduce_range", False),
            (serve, "--no-multi", "multi", False),
            (serve, "--no-auto-reload", "auto_reload", False),
            # Backward compat: --clean-onnx (deprecated alias for --no-hierarchy)
            (export, "--clean-onnx", "clean_onnx", True),
        ],
    )
    def test_flag_sets_expected_value(
        self, command, flag: str, param_name: str, expected_value
    ) -> None:
        """Verify each flag form correctly sets its parameter value via make_context."""
        import click as _click

        ctx = _click.Context(command, info_name=command.name, resilient_parsing=True)
        command.parse_args(ctx, [flag])
        assert ctx.params[param_name] is expected_value, (
            f"Expected {command.name} {flag} to set {param_name}={expected_value!r}, "
            f"got {ctx.params[param_name]!r}"
        )
