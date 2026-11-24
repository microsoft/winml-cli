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
    result = CliRunner().invoke(
        compile, ["--compiler", "offdevice", flag, "--ep", "openvino@directory"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == rows
    listing.assert_called_once()
    assert normalize_ep_name(listing.call_args.args[0]) == rows[0]["provider_name"]
    assert listing.call_args.kwargs == {"source": "directory"}


def test_ep_inventory_needs_no_ep_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = Mock(return_value=[])
    monkeypatch.setattr(compiler, "list_compilation_eps", listing)
    result = CliRunner().invoke(compile, ["--compiler", "offdevice", "--list-eps"])
    assert result.exit_code == 0, result.output
    listing.assert_called_once_with(None, source=None)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--list-eps"], "require --compiler offdevice"),
        (["--compiler", "offdevice", "--list-eps", "--list-targets"], "mutually exclusive"),
        (["--compiler", "offdevice"], "requires exactly one --model"),
        (["--target-platform", "4000"], "requires --compiler offdevice"),
        (
            ["--compiler", "offdevice", "--list-eps", "--target-platform", "4000"],
            "does not accept --target-platform",
        ),
        (["--compiler", "offdevice", "--list-targets"], "requires --ep"),
        (["--compiler", "offdevice", "--list-eps", "--device", "gpu"], "requires --device npu"),
        (
            ["--compiler", "offdevice", "--list-eps", "--compiler", "ort_session"],
            "require --compiler offdevice",
        ),
        (
            ["--compiler", "offdevice", "--list-eps", "--compiler", "cgc"],
            "CG Output IR compilation does not support",
        ),
        (["--compiler", "offdevice", "--list-eps", "--list"], "cannot be combined with --list"),
        (
            ["--compiler", "offdevice", "--list-eps", "--ep-options", "device_type=NPU"],
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
    result = CliRunner().invoke(
        compile, ["--compiler", "offdevice", "--list-eps", "--model", str(model)]
    )
    assert result.exit_code == 2, result.output
    assert "does not accept --model" in result.output


def test_listing_rejects_qnn_sdk_options(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        compile, ["--compiler", "offdevice", "--list-eps", "--qnn-sdk-root", str(tmp_path)]
    )
    assert result.exit_code == 2, result.output
    assert "does not accept --qnn-sdk-root" in result.output


def test_listing_error_is_not_a_successful_empty_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        compiler,
        "list_compilation_targets",
        Mock(side_effect=RuntimeError("No virtual NPU targets")),
    )
    result = CliRunner().invoke(
        compile, ["--compiler", "offdevice", "--list-targets", "--ep", "openvino"]
    )
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
        "--compiler",
        "offdevice",
        "--ep",
        "openvino@directory",
        "--target-platform",
        "4000",
        "--model",
        str(model_path),
    ]


@pytest.fixture
def compile_backend(monkeypatch: pytest.MonkeyPatch) -> Mock:
    backend = Mock(
        side_effect=lambda _model, *, output_path, **_kwargs: compiler.CompileResult(
            success=True, output_path=output_path
        )
    )
    monkeypatch.setattr(compiler, "compile_onnx", backend)
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
        model_path, output_path=destination, config=ANY, overwrite=False
    )
    config = compile_backend.call_args.kwargs["config"]
    assert normalize_ep_name(config.ep_config.provider) == "OpenVINOExecutionProvider"
    assert config.ep_config.compiler == "offdevice"
    assert config.ep_config.target_platform == "4000"
    assert config.ep_config.provider_options == {}
    assert config.ep_config.embed_context is True
    assert config.validate is True
    assert config.ep_device.source == "directory"
    assert config.ep_device.device == "npu"
    assert "Artifact validation: passed" in result.output


@pytest.mark.parametrize(
    ("omitted", "message"),
    [("--ep", "requires --ep"), ("--target-platform", "requires --target-platform")],
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
        (["--no-embed"], "requires embedded context"),
        (["--target-platform", " "], "requires --target-platform"),
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
        ({"validate": True}, [], None),
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
        assert "Artifact validation: passed" in result.output
        expected_validate = "--no-validate" not in overrides and settings.get("validate", True)
        assert compile_backend.call_args.kwargs["config"].validate == expected_validate


@pytest.mark.parametrize("validate", [False, True])
def test_offdevice_validation_request_and_reporting(
    validate: bool,
    compile_arguments: list[str],
    compile_backend: Mock,
) -> None:
    warning = "Target-device inference validation: skipped (off-device compilation)"
    compile_backend.side_effect = (
        lambda _model, *, output_path, **_kwargs: compiler.CompileResult(
            success=True,
            output_path=output_path,
            warnings=[warning],
        )
    )
    result = CliRunner().invoke(
        compile,
        [*compile_arguments, "--validate" if validate else "--no-validate"],
    )

    assert result.exit_code == 0, result.output
    assert compile_backend.call_args.kwargs["config"].validate is validate
    output = " ".join(result.output.split())
    assert "Artifact validation: passed (embedded-context checks)" in output
    assert warning in output


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
    ("ep_name", "selector", "value", "target", "canonical", "label"),
    [
        ("qnn", "--target-platform", "60", "60", "QNNExecutionProvider", "Target platform"),
        (
            "vitisai",
            "--ep-options",
            "target=X2",
            "X2",
            "VitisAIExecutionProvider",
            "Compiler backend",
        ),
    ],
)
def test_compilation_forwards_provider_target_and_source(
    ep_name: str,
    selector: str,
    value: str,
    target: str,
    canonical: str,
    label: str,
    model_path: Path,
    compile_backend: Mock,
) -> None:
    result = CliRunner().invoke(
        compile,
        [
            "--compiler",
            "offdevice",
            "--ep",
            f"{ep_name}@directory",
            selector,
            value,
            "--model",
            str(model_path),
        ],
    )
    assert result.exit_code == 0, result.output
    compile_backend.assert_called_once_with(
        model_path,
        output_path=model_path.with_name("model_ctx.onnx"),
        config=ANY,
        overwrite=False,
    )
    config = compile_backend.call_args.kwargs["config"]
    assert normalize_ep_name(config.ep_config.provider) == canonical
    assert config.ep_device.source == "directory"
    if ep_name == "vitisai":
        assert config.ep_config.target_platform is None
        assert config.ep_config.provider_options == {"target": target}
    else:
        assert config.ep_config.target_platform == target
        assert config.ep_config.provider_options == {}
    assert f"{label}: {target}" in result.output
    assert "Artifact validation: passed" in result.output


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ([], "requires --ep-options target=X2"),
        (["--ep-options", "target=X1"], "requires --ep-options target=X2"),
        (["--target-platform", "X2"], "instead of --target-platform"),
        (
            ["--ep-options", "target=X2", "--ep-options", "device_type=NPU"],
            "does not accept provider options",
        ),
    ],
)
def test_vitisai_rejects_invalid_backend_options(
    options: list[str],
    message: str,
    model_path: Path,
    compile_backend: Mock,
) -> None:
    result = CliRunner().invoke(
        compile,
        [
            "--compiler",
            "offdevice",
            "--ep",
            "vitisai",
            "--model",
            str(model_path),
            *options,
        ],
    )
    assert result.exit_code == 2, result.output
    assert message in result.output
    compile_backend.assert_not_called()


@pytest.mark.parametrize(
    ("configured_target", "cli_options"),
    [
        ("X2", []),
        ("X1", ["--ep-options", "target=X2"]),
    ],
)
def test_vitisai_backend_honors_config_and_cli_precedence(
    configured_target: str,
    cli_options: list[str],
    model_path: Path,
    compile_backend: Mock,
    tmp_path: Path,
) -> None:
    config = tmp_path / "compile.json"
    config.write_text(
        json.dumps({"compile": {"provider_options": {"target": configured_target}}}),
        encoding="utf-8",
    )
    result = CliRunner().invoke(
        compile,
        [
            "--compiler",
            "offdevice",
            "--ep",
            "vitisai",
            "--model",
            str(model_path),
            "--config",
            str(config),
            *cli_options,
        ],
    )
    assert result.exit_code == 0, result.output
    compile_backend.assert_called_once()
    config = compile_backend.call_args.kwargs["config"]
    assert config.ep_config.provider_options == {"target": "X2"}
    assert config.ep_config.target_platform is None


@pytest.mark.parametrize(
    ("cli_target", "expected"),
    [(None, "4000"), ("5010", "5010")],
)
def test_config_selects_offdevice_and_cli_overrides_platform(
    cli_target: str | None,
    expected: str,
    model_path: Path,
    compile_backend: Mock,
    tmp_path: Path,
) -> None:
    config = tmp_path / "compile.json"
    config.write_text(
        json.dumps(
            {
                "compile": {
                    "compiler": "offdevice",
                    "execution_provider": "openvino",
                    "target_platform": "4000",
                }
            }
        ),
        encoding="utf-8",
    )
    arguments = ["--model", str(model_path), "--config", str(config)]
    if cli_target is not None:
        arguments += ["--target-platform", cli_target]

    result = CliRunner().invoke(compile, arguments)

    assert result.exit_code == 0, result.output
    compile_backend.assert_called_once()
    assert compile_backend.call_args.kwargs["config"].ep_config.target_platform == expected


@pytest.mark.parametrize(
    ("configured_compiler", "cli_options", "accepted"),
    [
        ("offdevice", [], True),
        ("ort", ["--compiler", "offdevice"], True),
        ("offdevice", ["--compiler", "ort"], False),
    ],
)
def test_listing_respects_configured_compiler_and_cli_precedence(
    configured_compiler: str,
    cli_options: list[str],
    accepted: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "compile.json"
    config.write_text(
        json.dumps({"compile": {"compiler": configured_compiler}}),
        encoding="utf-8",
    )
    listing = Mock(return_value=[])
    monkeypatch.setattr(compiler, "list_compilation_eps", listing)

    result = CliRunner().invoke(
        compile, ["--config", str(config), "--list-eps", *cli_options]
    )

    if accepted:
        assert result.exit_code == 0, result.output
        assert json.loads(result.output) == []
        listing.assert_called_once_with(None, source=None)
    else:
        assert result.exit_code == 2, result.output
        assert "require --compiler offdevice" in result.output
        listing.assert_not_called()


def test_config_rejects_non_string_target_platform(
    compile_arguments: list[str],
    compile_backend: Mock,
    tmp_path: Path,
) -> None:
    config = tmp_path / "compile.json"
    config.write_text(
        json.dumps({"compile": {"target_platform": 4000}}),
        encoding="utf-8",
    )
    arguments = compile_arguments.copy()
    index = arguments.index("--target-platform")
    del arguments[index : index + 2]

    result = CliRunner().invoke(compile, [*arguments, "--config", str(config)])

    assert result.exit_code == 2, result.output
    assert "compile.target_platform must be a string" in result.output
    compile_backend.assert_not_called()


@pytest.mark.parametrize("success", [False, True])
def test_pipeline_failure_or_missing_artifact_is_not_reported_as_success(
    success: bool,
    compile_arguments: list[str],
    compile_backend: Mock,
) -> None:
    compile_backend.side_effect = None
    compile_backend.return_value = compiler.CompileResult(success=success, output_path=None)

    result = CliRunner().invoke(compile, compile_arguments)

    assert result.exit_code == 1, result.output
    assert "No compiled artifact was produced" in result.output
    assert "Success!" not in result.output
