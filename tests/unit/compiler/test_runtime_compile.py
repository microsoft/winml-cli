# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Persistence, validation and errors for Runtime Input IR compilation."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import click
import pytest

from winml.modelkit.compiler import compile_cgc_ir
from winml.modelkit.compiler.runtime import _publish, _select_gpu, _validate_input_ir
from winml.modelkit.export.cgc.foundry import (
    FoundryCompileError,
    FoundryToolboxUnavailableError,
)


@pytest.fixture
def native(tmp_path, monkeypatch):
    source = tmp_path / "input.mlir"
    source.write_bytes(b"Input IR fixture")
    output = tmp_path / "output.mlir"
    source.with_name("input_metadata.json").write_text(
        json.dumps({"inputs": [{"name": "x", "index": 0}], "outputs": [{"name": "y", "index": 0}]}),
    )
    input_schema = Mock()
    input_schema.input_count = 1
    input_schema.output_count = 1
    input_schema.inputs.return_value = [("float32", (2, 4))]
    input_schema.outputs.return_value = [("float32", (2, 4))]
    schema = Mock()
    schema.inputs.return_value = input_schema.inputs.return_value
    schema.outputs.return_value = input_schema.outputs.return_value
    model = Mock()
    model.schema.return_value = input_schema
    compiled = Mock()
    stage = Mock()
    stage.schema.return_value = schema
    builder = Mock()
    builder.add_model_stage.return_value = stage
    options = SimpleNamespace(form=None, close=Mock())
    compiler = Mock()
    compiler.supports_form.return_value = True
    compiler.create_options.return_value = options

    def write_artifact(_model, path, *, options):
        assert options.form == "device-targeted"
        path = Path(path)
        path.write_bytes(b"ML\xefRcompiled")
        path.with_suffix(".data").write_bytes(b"resources")

    compiler.compile_to_file.side_effect = write_artifact
    target = Mock(kind="gpu")
    target.model_compiler.return_value = compiler
    runtime = Mock()
    runtime.create_target_from_adapter.return_value = target
    runtime.load_model.side_effect = [model, compiled]
    runtime.create_pipeline_builder.return_value = builder
    adapter = SimpleNamespace(pointer=123, close=Mock())
    monkeypatch.setattr(
        "winml.modelkit.compiler.runtime.import_runtime",
        lambda: SimpleNamespace(
            Runtime=lambda: runtime,
            ExecutionTargetKind=SimpleNamespace(GPU="gpu"),
            CompiledModelForm=SimpleNamespace(DEVICE_TARGETED="device-targeted"),
        ),
    )
    monkeypatch.setattr("winml.modelkit.compiler.runtime._validate_input_ir", Mock())
    monkeypatch.setattr(
        "winml.modelkit.compiler.runtime._DXCoreAdapter.from_luid", lambda _luid: adapter
    )
    gpu = SimpleNamespace(device_type="GPU", name="Test GPU", luid="0x00000000_0x00000001")
    monkeypatch.setattr("winml.modelkit.compiler.runtime.enumerate_compute_adapters", lambda: [gpu])
    return SimpleNamespace(
        source=source,
        output=output,
        runtime=runtime,
        target=target,
        compiler=compiler,
        options=options,
        schema=schema,
        input_schema=input_schema,
        model=model,
        compiled=compiled,
        builder=builder,
        stage=stage,
        adapter=adapter,
    )


def test_compile_publishes_complete_bundle_and_closes_objects(native):
    result = compile_cgc_ir(native.source, native.output)
    assert result == native.output
    assert native.output.read_bytes() == b"ML\xefRcompiled"
    assert native.output.with_suffix(".data").read_bytes() == b"resources"
    metadata = json.loads(native.output.with_name("output_metadata.json").read_text())
    assert metadata["format"] == "cgc-output-ir"
    assert metadata["inputs"] == [{"name": "x", "index": 0}]
    receipt = json.loads(native.output.with_name("output_compile.json").read_text())
    assert receipt["validation"]["status"] == "passed"
    assert receipt["output_encoding"] == "mlir-bytecode"
    assert {entry["filename"] for entry in receipt["artifacts"]} == {
        "output.mlir",
        "output.data",
        "output_metadata.json",
    }
    native.compiler.compile_to_file.assert_called_once()
    for obj in (
        native.runtime,
        native.target,
        native.compiler,
        native.model,
        native.compiled,
        native.schema,
        native.input_schema,
        native.builder,
        native.stage,
        native.adapter,
    ):
        obj.close.assert_called_once()
    native.options.close.assert_called_once()


def test_no_validate_skips_reload_and_schema_build(native):
    compile_cgc_ir(native.source, native.output, validate=False)
    native.runtime.load_model.assert_called_once_with(str(native.source))
    native.builder.build.assert_not_called()
    receipt = json.loads(native.output.with_name("output_compile.json").read_text())
    assert receipt["validation"]["status"] == "skipped"


def test_overwrite_requires_opt_in_and_preserves_source(native):
    native.output.write_bytes(b"existing")
    with pytest.raises(click.ClickException, match="already exists"):
        compile_cgc_ir(native.source, native.output)
    native.compiler.compile_to_file.assert_not_called()
    compile_cgc_ir(native.source, native.output, overwrite=True)
    assert native.output.read_bytes() == b"ML\xefRcompiled"
    assert native.source.read_bytes() == b"Input IR fixture"


def test_companion_collision_preserves_entire_prior_output(native):
    sidecar = native.output.with_suffix(".data")
    sidecar.write_bytes(b"prior sidecar")
    with pytest.raises(click.ClickException, match="already exists"):
        compile_cgc_ir(native.source, native.output)
    assert sidecar.read_bytes() == b"prior sidecar"
    assert not native.output.exists()


def test_in_place_compile_is_rejected_even_with_overwrite(native):
    with pytest.raises(click.ClickException, match="different files"):
        compile_cgc_ir(native.source, native.source, overwrite=True)
    native.compiler.compile_to_file.assert_not_called()


def test_compile_failure_does_not_replace_existing_output(native):
    native.output.write_bytes(b"old")
    native.compiler.compile_to_file.side_effect = RuntimeError("compiler error")
    with pytest.raises(RuntimeError, match="compiler error"):
        compile_cgc_ir(native.source, native.output, overwrite=True)
    assert native.output.read_bytes() == b"old"
    assert not native.output.with_suffix(".data").exists()
    native.options.close.assert_called_once()
    native.runtime.close.assert_called_once()


@pytest.mark.parametrize("failure", ["mismatch", "empty", "text", "unsupported", "wrong-device"])
def test_rejects_incomplete_or_invalid_runtime_output(native, failure):
    if failure == "mismatch":
        native.schema.outputs.return_value = [("float32", (8, 8))]
    elif failure == "empty":
        native.compiler.compile_to_file.side_effect = lambda *_args, **_kwargs: None
    elif failure == "text":
        native.compiler.compile_to_file.side_effect = lambda _m, p, **_kw: Path(p).write_text(
            "unexpected"
        )
    elif failure == "unsupported":
        native.compiler.supports_form.return_value = False
    else:
        native.target.kind = "cpu"
    with pytest.raises(click.ClickException):
        compile_cgc_ir(native.source, native.output)
    assert not native.output.exists()


def test_schema_changed_input_is_rejected_before_publication(native):
    def mutate(_model, path, **_kwargs):
        Path(path).write_bytes(b"ML\xefRcompiled")
        native.source.write_bytes(b"changed")

    native.compiler.compile_to_file.side_effect = mutate
    with pytest.raises(click.ClickException, match="changed during compilation"):
        compile_cgc_ir(native.source, native.output)
    assert not native.output.exists()


@pytest.mark.parametrize("kind", ["invalid-json", "list"])
def test_invalid_metadata_is_explicit(native, kind):
    native.source.with_name("input_metadata.json").write_text(
        "{" if kind == "invalid-json" else "[]"
    )
    with pytest.raises(click.ClickException, match="metadata"):
        compile_cgc_ir(native.source, native.output)


@pytest.mark.parametrize(
    "entries", [[], [{"name": "x", "index": 2}], [{"name": "", "index": 0}], "not a list"]
)
def test_metadata_binding_mismatch_is_rejected(native, entries):
    native.source.with_name("input_metadata.json").write_text(
        json.dumps({"inputs": entries, "outputs": [{"name": "y", "index": 0}]}),
    )
    with pytest.raises(click.ClickException, match="do not match"):
        compile_cgc_ir(native.source, native.output)
    native.compiler.compile_to_file.assert_not_called()
    assert not native.output.exists()


def test_input_and_output_metadata_collision_preserves_source(native):
    destination = native.source.with_suffix(".mlirbc")
    with pytest.raises(click.ClickException, match="source artifact"):
        compile_cgc_ir(native.source, destination, overwrite=True)
    assert not destination.exists()
    metadata = json.loads(native.source.with_name("input_metadata.json").read_text())
    assert metadata["inputs"] == [{"name": "x", "index": 0}]


def test_source_metadata_change_during_compile_is_rejected(native):
    original_compile = native.compiler.compile_to_file.side_effect

    def mutate_metadata(model, path, **kwargs):
        original_compile(model, path, **kwargs)
        native.source.with_name("input_metadata.json").write_text("{}")

    native.compiler.compile_to_file.side_effect = mutate_metadata
    with pytest.raises(click.ClickException, match="metadata changed"):
        compile_cgc_ir(native.source, native.output)
    assert not native.output.exists()


def test_nonexistent_input_is_explicit(native):
    with pytest.raises(click.ClickException, match="file not found"):
        compile_cgc_ir(native.source.with_name("missing.mlir"), native.output)
    native.compiler.compile_to_file.assert_not_called()


def test_wrong_output_suffix_is_rejected(native):
    with pytest.raises(click.UsageError, match=r"\.mlir or \.mlirbc"):
        compile_cgc_ir(native.source, native.output.with_suffix(".onnx"))
    native.compiler.compile_to_file.assert_not_called()


def test_cgc_compile_without_io_metadata(native):
    native.source.with_name("input_metadata.json").unlink()
    compile_cgc_ir(native.source, native.output)
    assert native.output.is_file()
    assert not native.output.with_name("output_metadata.json").exists()


@pytest.mark.parametrize("count", [0, 2])
def test_gpu_selection_rejects_missing_or_ambiguous_adapters(monkeypatch, count):
    adapters = [
        SimpleNamespace(device_type="GPU", name=f"GPU {i}", luid=f"0x00000000_0x0000000{i}")
        for i in range(count)
    ]
    monkeypatch.setattr(
        "winml.modelkit.compiler.runtime.enumerate_compute_adapters", lambda: adapters
    )
    with pytest.raises(click.ClickException, match="unambiguous"):
        _select_gpu(None)
    if count == 2:
        assert _select_gpu("0X00000000_0X00000001") == adapters[1]


@pytest.mark.parametrize(
    ("text", "valid"),
    [
        ("module {\n cgc.entry_point @entry() {}\n}", True),
        ('module {\n "cgc.entry_point"() {}\n}', True),
        ("cgc.module attributes {interface = #cgc.interface<cgc_output>} {}", False),
        ('module { note = "cgc.entry_point" }', False),
        ("module {\n func.func @entry() {}\n}", False),
    ],
)
def test_input_ir_is_verified_after_native_parse(tmp_path, monkeypatch, text, valid):
    source = tmp_path / "arbitrary.extension"
    source.write_bytes(b"parse me")
    parser = Mock()
    parser.__enter__ = Mock(return_value=parser)
    parser.__exit__ = Mock(return_value=False)
    parser.read_cgir.return_value = text.encode()
    monkeypatch.setattr("winml.modelkit.compiler.runtime.FoundryCompiler", lambda: parser)
    if valid:
        _validate_input_ir(source)
    else:
        with pytest.raises(click.ClickException, match="requires a CGC Input IR"):
            _validate_input_ir(source)
    parser.read_cgir.assert_called_once_with(b"parse me")


@pytest.mark.parametrize(
    "error",
    [
        FoundryToolboxUnavailableError("Preview DLL missing"),
        FoundryCompileError(2, native_message="Invalid IR"),
    ],
)
def test_native_input_validation_errors_are_actionable(tmp_path, monkeypatch, error):
    source = tmp_path / "input.mlir"
    source.write_bytes(b"not IR")
    monkeypatch.setattr("winml.modelkit.compiler.runtime.FoundryCompiler", Mock(side_effect=error))
    with pytest.raises(click.ClickException, match="Cannot parse CGC Input IR") as result:
        _validate_input_ir(source)
    assert str(error) in str(result.value)


def test_atomic_publication_rolls_back_all_files(tmp_path, monkeypatch):
    staging = tmp_path / "stage"
    staging.mkdir()
    (staging / "first").write_bytes(b"new first")
    (staging / "second").write_bytes(b"new second")
    (tmp_path / "first").write_bytes(b"old first")
    (tmp_path / "second").write_bytes(b"old second")
    replace = Path.replace

    def fail_one_publish(self, target):
        if self == staging / "second":
            raise OSError("publish failure")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_one_publish)
    with pytest.raises(OSError, match="publish failure"):
        _publish(staging, tmp_path / "first", True, ())
    assert (tmp_path / "first").read_bytes() == b"old first"
    assert (tmp_path / "second").read_bytes() == b"old second"


def test_nested_runtime_layout_is_not_silently_discarded(tmp_path):
    staging = tmp_path / "stage"
    (staging / "resources").mkdir(parents=True)
    (staging / "compiled.mlir").write_bytes(b"ML\xefRcompiled")
    with pytest.raises(click.ClickException, match="nested Runtime"):
        _publish(staging, tmp_path / "compiled.mlir", False, ())
    assert not (tmp_path / "compiled.mlir").exists()
