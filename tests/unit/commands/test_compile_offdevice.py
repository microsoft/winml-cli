# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for opt-in virtual-target listing and compile-only dispatch."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import ANY, Mock

import pytest
from click.testing import CliRunner

from winml.modelkit import compiler
from winml.modelkit.commands.compile import compile
from winml.modelkit.utils.constants import normalize_ep_name


if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def forbid_physical_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Off-device commands must never enter the normal device resolver."""
    monkeypatch.setattr(
        "winml.modelkit.commands.compile.resolve_device",
        Mock(side_effect=AssertionError("Physical device resolution is forbidden")),
    )


@pytest.mark.parametrize(
    ("flag", "helper"),
    [
        ("--list-eps", "list_compilation_eps"),
        ("--list-targets", "list_compilation_targets"),
    ],
)
def test_listing_outputs_json_without_a_model(
    flag: str, helper: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [{"provider_name": "OpenVINOExecutionProvider"}]
    listing = Mock(return_value=rows)
    monkeypatch.setattr(compiler, helper, listing)
    result = CliRunner().invoke(compile, ["--off-device", flag, "--ep", "openvino@directory"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == rows
    listing.assert_called_once()
    assert normalize_ep_name(listing.call_args.args[0]) == rows[0]["provider_name"]
    assert listing.call_args.kwargs == {"source": "directory"}


def test_ep_inventory_needs_no_ep_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = Mock(return_value=[])
    monkeypatch.setattr(compiler, "list_compilation_eps", listing)
    result = CliRunner().invoke(compile, ["--off-device", "--list-eps"])
    assert result.exit_code == 0, result.output
    listing.assert_called_once_with(None, source=None)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--list-eps"], "require --off-device"),
        (["--off-device", "--list-eps", "--list-targets"], "mutually exclusive"),
        (["--off-device"], "requires exactly one --model"),
        (["--target", "4000"], "requires --off-device"),
        (
            ["--off-device", "--list-eps", "--target", "4000"],
            "does not accept --target",
        ),
        (["--off-device", "--list-targets"], "requires --ep"),
        (["--off-device", "--list-eps", "--device", "gpu"], "requires --device npu"),
        (
            ["--off-device", "--list-eps", "--compiler", "ort_session"],
            "requires --compiler ort",
        ),
        (
            ["--off-device", "--list-eps", "--compiler", "cgc"],
            "CG Output IR compilation does not support",
        ),
        (["--off-device", "--list-eps", "--list"], "cannot be combined with --list"),
        (
            ["--off-device", "--list-eps", "--ep-options", "device_type=NPU"],
            "does not accept provider options",
        ),
    ],
)
def test_invalid_listing_combinations_fail(arguments: list[str], message: str) -> None:
    result = CliRunner().invoke(compile, arguments)
    assert result.exit_code == 2, result.output
    assert message in result.output


def test_listing_rejects_model_input(tmp_path: Path) -> None:
    model = tmp_path / "input.onnx"
    model.touch()
    result = CliRunner().invoke(compile, ["--off-device", "--list-eps", "--model", str(model)])
    assert result.exit_code == 2, result.output
    assert "does not accept --model" in result.output


def test_listing_rejects_qnn_sdk_options(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        compile, ["--off-device", "--list-eps", "--qnn-sdk-root", str(tmp_path)]
    )
    assert result.exit_code == 2, result.output
    assert "requires --compiler ort" in result.output


def test_listing_error_is_not_a_successful_empty_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        compiler,
        "list_compilation_targets",
        Mock(side_effect=RuntimeError("No virtual NPU targets")),
    )
    result = CliRunner().invoke(compile, ["--off-device", "--list-targets", "--ep", "openvino"])
    assert result.exit_code == 1, result.output
    assert "No virtual NPU targets" in result.output


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    model = tmp_path / "model.onnx"
    model.touch()
    return model


@pytest.fixture
def compile_arguments(model_path: Path) -> list[str]:
    return [
        "--off-device",
        "--ep",
        "openvino@directory",
        "--target",
        "4000",
        "--model",
        str(model_path),
    ]


@pytest.fixture
def compile_backend(monkeypatch: pytest.MonkeyPatch) -> Mock:
    backend = Mock(side_effect=lambda _model, destination, _target, **_kwargs: destination)
    monkeypatch.setattr(compiler, "compile_off_device", backend)
    return backend


@pytest.mark.parametrize("output_mode", ["default", "file", "directory"])
def test_compilation_forwards_virtual_target_source_and_output(
    output_mode: str,
    model_path: Path,
    compile_arguments: list[str],
    compile_backend: Mock,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "output" if output_mode == "directory" else tmp_path
    destination = directory / ("compiled.onnx" if output_mode == "file" else "model_ctx.onnx")
    arguments = compile_arguments.copy()
    if output_mode == "file":
        arguments += ["--output", str(destination)]
    elif output_mode == "directory":
        arguments += ["--output-dir", str(directory)]

    result = CliRunner().invoke(compile, arguments)

    assert result.exit_code == 0, result.output
    compile_backend.assert_called_once_with(
        model_path, destination, "4000", ep_name=ANY, source="directory", overwrite=False
    )
    assert normalize_ep_name(compile_backend.call_args.kwargs["ep_name"]) == (
        "OpenVINOExecutionProvider"
    )
    assert "no inference was run" in result.output


@pytest.mark.parametrize(
    ("omitted", "message"),
    [("--ep", "requires --ep"), ("--target", "requires --target")],
)
def test_compilation_requires_explicit_selectors(
    omitted: str,
    message: str,
    compile_arguments: list[str],
    compile_backend: Mock,
) -> None:
    arguments = compile_arguments.copy()
    index = arguments.index(omitted)
    del arguments[index : index + 2]
    result = CliRunner().invoke(compile, arguments)
    assert result.exit_code == 2, result.output
    assert message in result.output
    compile_backend.assert_not_called()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--validate"], "cannot run inference validation"),
        (["--no-embed"], "requires embedded context"),
        (["--target", " "], "requires --target"),
    ],
)
def test_compilation_rejects_conflicting_options(
    extra: list[str],
    message: str,
    compile_arguments: list[str],
    compile_backend: Mock,
) -> None:
    result = CliRunner().invoke(compile, [*compile_arguments, *extra])
    assert result.exit_code == 2, result.output
    assert message in result.output
    compile_backend.assert_not_called()


def test_compilation_rejects_multiple_models(
    model_path: Path, compile_arguments: list[str], compile_backend: Mock
) -> None:
    result = CliRunner().invoke(compile, [*compile_arguments, "--model", str(model_path)])
    assert result.exit_code == 2, result.output
    assert "requires exactly one --model" in result.output
    compile_backend.assert_not_called()


@pytest.mark.parametrize(
    ("settings", "overrides", "message"),
    [
        ({"validate": True}, [], "cannot run inference validation"),
        ({"embed_context": False}, [], "requires embedded context"),
        (
            {"validate": True, "embed_context": False},
            ["--no-validate", "--embed"],
            None,
        ),
        ({"validate": False, "embed_context": True}, [], None),
        ({"provider_options": {"device_type": "GPU"}}, [], "does not accept provider options"),
        ({"provider_option_file_keys": ["compiler_input"]}, [], "does not accept provider options"),
    ],
)
def test_compilation_honors_explicit_config_and_cli_precedence(
    settings: dict[str, object],
    overrides: list[str],
    message: str | None,
    compile_arguments: list[str],
    compile_backend: Mock,
    tmp_path: Path,
) -> None:
    config = tmp_path / "compile.json"
    config.write_text(json.dumps({"compile": settings}), encoding="utf-8")
    result = CliRunner().invoke(compile, [*compile_arguments, "--config", str(config), *overrides])
    if message is not None:
        assert result.exit_code == 2, result.output
        assert message in result.output
        compile_backend.assert_not_called()
    else:
        assert result.exit_code == 0, result.output
        compile_backend.assert_called_once()
        assert "no inference was run" in result.output


@pytest.mark.parametrize("overwrite", [False, True])
def test_existing_default_output_requires_explicit_overwrite(
    overwrite: bool,
    model_path: Path,
    compile_arguments: list[str],
    compile_backend: Mock,
) -> None:
    destination = model_path.with_name("model_ctx.onnx")
    destination.write_text("existing output", encoding="utf-8")
    arguments = [*compile_arguments, *(["--overwrite"] if overwrite else [])]
    result = CliRunner().invoke(compile, arguments)
    if overwrite:
        assert result.exit_code == 0, result.output
        compile_backend.assert_called_once()
        assert compile_backend.call_args.kwargs["overwrite"] is True
    else:
        assert result.exit_code == 1, result.output
        assert "--overwrite" in result.output
        compile_backend.assert_not_called()
        assert destination.read_text(encoding="utf-8") == "existing output"


def test_compilation_failure_is_not_reported_as_success(
    compile_arguments: list[str], compile_backend: Mock
) -> None:
    compile_backend.side_effect = RuntimeError("native compilation failed")
    result = CliRunner().invoke(compile, compile_arguments)
    assert result.exit_code == 1, result.output
    assert "native compilation failed" in result.output
    assert "Success!" not in result.output


@pytest.mark.parametrize(
    ("ep_name", "target", "canonical"),
    [("qnn", "60", "QNNExecutionProvider"), ("vitisai", "X2", "VitisAIExecutionProvider")],
)
def test_compilation_forwards_provider_target_and_source(
    ep_name: str, target: str, canonical: str, model_path: Path, compile_backend: Mock
) -> None:
    result = CliRunner().invoke(
        compile,
        [
            "--off-device",
            "--ep",
            f"{ep_name}@directory",
            "--target",
            target,
            "--model",
            str(model_path),
        ],
    )
    assert result.exit_code == 0, result.output
    compile_backend.assert_called_once_with(
        model_path,
        model_path.with_name("model_ctx.onnx"),
        target,
        ep_name=ANY,
        source="directory",
        overwrite=False,
    )
    assert normalize_ep_name(compile_backend.call_args.kwargs["ep_name"]) == canonical
    assert "no inference was run" in result.output
