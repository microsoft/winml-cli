# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

import pytest
from click.testing import CliRunner

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
                "--quantize",
                "--no-quantize",
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
        (quantize, ["--per-channel", "--no-per-channel", "--symmetric", "--no-symmetric"]),
        (serve, ["--multi", "--no-multi"]),
    ],
)
def test_help_includes_boolean_flag_pairs(command, flags: list[str]) -> None:
    result = CliRunner().invoke(command, ["--help"])

    assert result.exit_code == 0, result.output
    for flag in flags:
        assert flag in result.output
