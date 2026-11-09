# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Persistence and publication tests for Runtime-backed CGIR compilation."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import click
import pytest

from winml.modelkit.compiler import compile_cgc_ir
from winml.modelkit.compiler.runtime import _publish, _select_gpu


@pytest.fixture
def native(tmp_path, monkeypatch):
    source = tmp_path / "input.mlir"
    source.write_bytes(b"Input IR fixture")
    output = tmp_path / "output.mlir"
    source.with_name("input_metadata.json").write_text(
        json.dumps({"inputs": [{"name": "x", "index": 0}], "outputs": [{"name": "y", "index": 0}]}),
    )
    compile_action = Mock()

    def write_artifact(path: Path) -> None:
        path.write_text("module { cgc.output }", encoding="utf-8")
        path.with_suffix(".data").write_bytes(b"resources")

    compile_action.side_effect = write_artifact
    sessions = []

    class FakeSession:
        def __init__(self, model_path, *, backend, device_luid):
            self.model_path = Path(model_path)
            self.backend = backend
            self.device_luid = device_luid
            self.validate = None
            sessions.append(self)

        def compile_to_file(self, path, *, validate, output_format):
            self.validate = validate
            self.output_format = output_format
            compile_action(Path(path))

    monkeypatch.setattr("winml.modelkit.compiler.runtime.WinMLRuntimeSession", FakeSession)
    gpu = SimpleNamespace(device_type="GPU", name="Test GPU", luid="0x00000000_0x00000001")
    monkeypatch.setattr("winml.modelkit.compiler.runtime.enumerate_compute_adapters", lambda: [gpu])
    return SimpleNamespace(
        source=source,
        output=output,
        compile_action=compile_action,
        sessions=sessions,
    )


def test_compile_routes_through_runtime_session_and_publishes_bundle(native):
    result = compile_cgc_ir(native.source, native.output)

    assert result == native.output
    assert native.output.read_text(encoding="utf-8") == "module { cgc.output }"
    assert native.output.with_suffix(".data").read_bytes() == b"resources"
    metadata = json.loads(native.output.with_name("output_metadata.json").read_text())
    assert metadata["format"] == "cg-output-ir"
    receipt = json.loads(native.output.with_name("output_compile.json").read_text())
    assert receipt["validation"]["status"] == "passed"
    assert receipt["output_encoding"] == "mlir-text"
    assert {entry["filename"] for entry in receipt["artifacts"]} == {
        "output.mlir",
        "output.data",
        "output_metadata.json",
    }
    session = native.sessions[0]
    assert session.model_path == native.source
    assert session.backend == "cgc"
    assert session.device_luid == 1
    assert session.validate is True
    assert session.output_format == "text"


def test_no_validate_is_forwarded_to_runtime_session(native):
    compile_cgc_ir(native.source, native.output, validate=False)
    assert native.sessions[0].validate is False
    receipt = json.loads(native.output.with_name("output_compile.json").read_text())
    assert receipt["validation"]["status"] == "skipped"


def test_overwrite_requires_opt_in_and_preserves_source(native):
    native.output.write_bytes(b"existing")
    with pytest.raises(click.ClickException, match="already exists"):
        compile_cgc_ir(native.source, native.output)
    native.compile_action.assert_not_called()

    compile_cgc_ir(native.source, native.output, overwrite=True)
    assert native.output.read_text(encoding="utf-8") == "module { cgc.output }"
    assert native.source.read_bytes() == b"Input IR fixture"


def test_overwrite_removes_stale_output_metadata(native):
    native.source.with_name("input_metadata.json").unlink()
    native.output.write_bytes(b"old model")
    stale = native.output.with_name("output_metadata.json")
    stale.write_text('{"inputs": [{"name": "old", "index": 0}]}')

    compile_cgc_ir(native.source, native.output, overwrite=True)

    assert native.output.read_text(encoding="utf-8") == "module { cgc.output }"
    assert not stale.exists()


def test_companion_collision_preserves_prior_output(native):
    sidecar = native.output.with_suffix(".data")
    sidecar.write_bytes(b"prior sidecar")
    with pytest.raises(click.ClickException, match="already exists"):
        compile_cgc_ir(native.source, native.output)
    assert sidecar.read_bytes() == b"prior sidecar"
    assert not native.output.exists()


def test_compile_failure_does_not_replace_existing_output(native):
    native.output.write_bytes(b"old")
    native.compile_action.side_effect = RuntimeError("compiler error")
    with pytest.raises(RuntimeError, match="compiler error"):
        compile_cgc_ir(native.source, native.output, overwrite=True)
    assert native.output.read_bytes() == b"old"


@pytest.mark.parametrize("failure", ["empty", "binary"])
def test_rejects_invalid_runtime_output(native, failure):
    if failure == "empty":
        native.compile_action.side_effect = lambda path: None
    else:
        native.compile_action.side_effect = lambda path: path.write_bytes(b"\xff\xfe")
    with pytest.raises(click.ClickException):
        compile_cgc_ir(native.source, native.output)
    assert not native.output.exists()


def test_changed_input_is_rejected_before_publication(native):
    def mutate(path):
        path.write_text("module { cgc.output }", encoding="utf-8")
        native.source.write_bytes(b"changed")

    native.compile_action.side_effect = mutate
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


def test_source_metadata_change_during_compile_is_rejected(native):
    def mutate(path):
        path.write_text("module { cgc.output }", encoding="utf-8")
        native.source.with_name("input_metadata.json").write_text("{}")

    native.compile_action.side_effect = mutate
    with pytest.raises(click.ClickException, match="metadata changed"):
        compile_cgc_ir(native.source, native.output)
    assert not native.output.exists()


def test_input_and_output_metadata_collision_preserves_source(native):
    destination = native.source.with_suffix(".mlirbc")
    with pytest.raises(click.UsageError, match=r"must use \.mlir"):
        compile_cgc_ir(native.source, destination, overwrite=True)
    assert not destination.exists()


def test_nonexistent_input_and_wrong_output_suffix_are_explicit(native):
    with pytest.raises(click.ClickException, match="file not found"):
        compile_cgc_ir(native.source.with_name("missing.mlir"), native.output)
    with pytest.raises(click.UsageError, match=r"must use \.mlir"):
        compile_cgc_ir(native.source, native.output.with_suffix(".onnx"))
    native.compile_action.assert_not_called()


def test_in_place_compile_is_rejected_even_with_overwrite(native):
    with pytest.raises(click.ClickException, match="different files"):
        compile_cgc_ir(native.source, native.source, overwrite=True)
    native.compile_action.assert_not_called()


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


def test_atomic_publication_rolls_back_new_and_invalidated_files(tmp_path, monkeypatch):
    staging = tmp_path / "stage"
    staging.mkdir()
    (staging / "first").write_bytes(b"new first")
    (staging / "second").write_bytes(b"new second")
    (tmp_path / "first").write_bytes(b"old first")
    (tmp_path / "second").write_bytes(b"old second")
    stale = tmp_path / "stale"
    stale.write_bytes(b"old metadata")
    replace = Path.replace

    def fail_one_publish(self, target):
        if self == staging / "second":
            raise OSError("publish failure")
        return replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_one_publish)
    with pytest.raises(OSError, match="publish failure"):
        _publish(staging, tmp_path / "first", True, (), invalidated=(stale,))
    assert (tmp_path / "first").read_bytes() == b"old first"
    assert (tmp_path / "second").read_bytes() == b"old second"
    assert stale.read_bytes() == b"old metadata"


def test_nested_runtime_layout_is_not_silently_discarded(tmp_path):
    staging = tmp_path / "stage"
    (staging / "resources").mkdir(parents=True)
    (staging / "compiled.mlir").write_bytes(b"ML\xefRcompiled")
    with pytest.raises(click.ClickException, match="nested Runtime"):
        _publish(staging, tmp_path / "compiled.mlir", False, ())
