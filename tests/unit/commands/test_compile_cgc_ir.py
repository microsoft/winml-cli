# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Explicit Input IR / Output IR CLI routing, without native dependencies."""

from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.compile import compile


@pytest.fixture
def ir_cli(tmp_path, monkeypatch):
    source = tmp_path / "input.mlir"
    source.write_bytes(b"input")
    operation = Mock(return_value=tmp_path / "output.mlir")
    monkeypatch.setattr("winml.modelkit.compiler.compile_cgc_ir", operation)
    resolver = Mock(side_effect=AssertionError("IR compile must not resolve an ORT EP"))
    monkeypatch.setattr("winml.modelkit.commands.compile.resolve_device", resolver)
    args = [
        "-m",
        str(source),
    ]
    return source, operation, args


def test_explicit_ir_compilation_routes_without_ep_resolution(ir_cli):
    source, operation, args = ir_cli
    destination = source.parent / "compiled.mlir"
    result = CliRunner().invoke(
        compile,
        [
            *args,
            "--compiler",
            "cgc",
            "--device",
            "gpu",
            "--device-luid",
            "0x00000000_0x00000001",
            "-o",
            str(destination),
            "--overwrite",
        ],
    )
    assert result.exit_code == 0, result.output
    operation.assert_called_once_with(
        source,
        destination,
        device_luid="0x00000000_0x00000001",
        validate=True,
        overwrite=True,
    )
    assert "MLIR text" in result.output and "no inference" in result.output


def test_ir_defaults_and_skipped_validation(ir_cli):
    source, operation, args = ir_cli
    result = CliRunner().invoke(compile, [*args, "--no-validate"])
    assert result.exit_code == 0, result.output
    operation.assert_called_once_with(
        source,
        source.with_name("input_output.mlir"),
        device_luid=None,
        validate=False,
        overwrite=False,
    )
    assert "skipped" in result.output


def test_ir_output_file_precedes_output_directory(ir_cli):
    source, operation, args = ir_cli
    destination = source.parent / "selected.mlir"
    result = CliRunner().invoke(
        compile,
        [*args, "-o", str(destination), "--output-dir", str(source.parent / "unused")],
    )
    assert result.exit_code == 0, result.output
    assert operation.call_args.args[1] == destination


def test_explicit_cgc_compiler_routes_nonstandard_suffix(ir_cli):
    source, operation, _ = ir_cli
    source = source.with_suffix(".cgir")
    source.write_bytes(b"input")
    result = CliRunner().invoke(
        compile,
        ["-m", str(source), "--compiler", "cgc"],
    )
    assert result.exit_code == 0, result.output
    operation.assert_called_once()


@pytest.mark.parametrize(
    "extra",
    [
        ["--ep", "winmlcg"],
        ["--ep-options", "device_id=0"],
        ["--embed"],
        ["--no-embed"],
        ["--list"],
        ["--device", "cpu"],
        ["--device", "npu"],
        ["--device", "auto"],
    ],
)
def test_ir_rejects_unrelated_options(ir_cli, extra):
    _, operation, args = ir_cli
    result = CliRunner().invoke(compile, [*args, *extra])
    assert result.exit_code == 2, result.output
    operation.assert_not_called()


@pytest.mark.parametrize("option", ["--config", "--qnn-sdk-root"])
def test_ir_rejects_config_and_sdk(ir_cli, option):
    source, operation, args = ir_cli
    path = source.parent / "config.json"
    path.write_text("{}")
    result = CliRunner().invoke(compile, [*args, option, str(path)])
    assert result.exit_code == 2, result.output
    assert "does not support" in result.output
    operation.assert_not_called()


def test_ir_rejects_multiple_models(ir_cli):
    source, operation, args = ir_cli
    result = CliRunner().invoke(compile, [*args, "-m", str(source), "--compiler", "cgc"])
    assert result.exit_code == 2
    assert "exactly one" in result.output
    operation.assert_not_called()


def test_ir_requires_model(ir_cli):
    _, operation, _ = ir_cli
    result = CliRunner().invoke(compile, ["--compiler", "cgc"])
    assert result.exit_code == 2
    assert "exactly one" in result.output
    operation.assert_not_called()


def test_epcontext_rejects_ir_only_device_luid(ir_cli):
    source, operation, _ = ir_cli
    source = source.with_suffix(".onnx")
    source.write_bytes(b"input")
    result = CliRunner().invoke(
        compile,
        ["-m", str(source), "--device-luid", "0x00000000_0x00000001"],
    )
    assert result.exit_code == 2
    assert "only supported for CG Output IR" in result.output
    operation.assert_not_called()


def test_ir_filesystem_error_is_actionable(ir_cli):
    _, operation, args = ir_cli
    operation.side_effect = OSError("Disk write failed")
    result = CliRunner().invoke(compile, args)
    assert result.exit_code == 1
    assert "CGC compilation failed: Disk write failed" in result.output


def test_ir_options_are_hidden_in_help():
    result = CliRunner().invoke(compile, ["--help"])
    assert result.exit_code == 0
    assert "--compiler" in result.output
    for value in ("cgc", "cgir", "mlir", "--device-luid"):
        assert value not in result.output.casefold()
