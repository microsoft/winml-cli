# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from onnx import ModelProto, TensorProto, helper, numpy_helper, save_model

from winml.modelkit.export import WinMLExportConfig
from winml.modelkit.export.cgc import CGCExporter, CGCExportResult, CGCOptions
from winml.modelkit.export.cgc.foundry import FoundryCompileError


def _make_model() -> ModelProto:
    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
    weight = numpy_helper.from_array(
        np.array([1.0, 2.0], dtype=np.float32),
        name="weight",
    )
    bias = helper.make_tensor(
        "bias",
        TensorProto.FLOAT,
        [2],
        [0.5, 1.5],
    )
    nodes = [
        helper.make_node("Add", ["input", "weight"], ["hidden"]),
        helper.make_node("Add", ["hidden", "bias"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "test",
        [input_info],
        [output_info],
        initializer=[weight, bias],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def test_pytorch_export_hides_onnx_intermediate(tmp_path: Path) -> None:
    output = tmp_path / "model.mlir"
    exporter = CGCExporter(CGCOptions())
    intermediate_paths: list[Path] = []
    export_stats = {"nodes": 1}

    def fake_export_onnx(**kwargs):
        intermediate_path = Path(kwargs["output_path"])
        intermediate_path.write_bytes(b"onnx")
        intermediate_paths.append(intermediate_path)
        return export_stats

    with (
        patch(
            "winml.modelkit.export.export_pytorch",
            side_effect=fake_export_onnx,
        ) as export_onnx,
        patch.object(
            exporter,
            "export_onnx",
            return_value=CGCExportResult(input_names=(), output_names=()),
        ) as export_cgc,
    ):
        result = exporter.export_pytorch(
            model=object(),
            output_path=output,
            export_config=WinMLExportConfig(),
            model_id="model-id",
            task="feature-extraction",
            verbose=True,
            enable_reporting=False,
        )

    assert result.export_stats is export_stats
    assert result.input_names == ()
    assert result.output_names == ()
    export_onnx.assert_called_once()
    export_cgc.assert_called_once_with(
        model=intermediate_paths[0],
        output_path=output,
    )
    assert intermediate_paths[0].name == "model.onnx"
    assert not intermediate_paths[0].parent.exists()


class _ExternalMlirCompiler:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def compile_onnx(self, _source, **kwargs):
        kwargs["output_data_file"].write_bytes(b"new weights")
        return b"module { cgc.test }"


def test_external_mlir_overwrite_replaces_sidecar(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    output = tmp_path / "model.mlir"
    sidecar = output.with_name(f"{output.name}.data")
    metadata = tmp_path / "model_metadata.json"
    save_model(_make_model(), str(source))
    sidecar.write_bytes(b"old weights")

    with patch(
        "winml.modelkit.export.cgc.exporter.FoundryCompiler",
        return_value=_ExternalMlirCompiler(),
    ):
        CGCExporter(CGCOptions(external_weights=True)).export(
            source,
            output,
        )

    assert output.read_text(encoding="utf-8") == "module { cgc.test }"
    assert sidecar.read_bytes() == b"new weights"
    assert metadata.is_file()


def test_external_mlir_failure_preserves_existing_bundle(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    output = tmp_path / "model.mlir"
    sidecar = output.with_name(f"{output.name}.data")
    metadata = tmp_path / "model_metadata.json"
    save_model(_make_model(), str(source))
    output.write_text("old mlir", encoding="utf-8")
    sidecar.write_bytes(b"old weights")
    metadata.write_text("old metadata", encoding="utf-8")
    exporter = CGCExporter(CGCOptions(external_weights=True))

    with (
        patch(
            "winml.modelkit.export.cgc.exporter.FoundryCompiler",
            return_value=_ExternalMlirCompiler(),
        ),
        patch.object(
            exporter,
            "_write_io_metadata",
            side_effect=RuntimeError("metadata write failed"),
        ),
        pytest.raises(RuntimeError, match="metadata write failed"),
    ):
        exporter.export(source, output)

    assert output.read_text(encoding="utf-8") == "old mlir"
    assert sidecar.read_bytes() == b"old weights"
    assert metadata.read_text(encoding="utf-8") == "old metadata"


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("weights_mode", ["external", "absent", "embedded"])
@pytest.mark.parametrize("failure", [None, "backup", "weights", "metadata", "model"])
def test_mlir_bundle_publication(
    tmp_path: Path, monkeypatch, existing: bool, weights_mode: str, failure: str | None,
) -> None:
    source = tmp_path / "source.onnx"
    model = _make_model()
    save_model(model, source)
    original_source = source.read_bytes()
    output = tmp_path / "model.mlir"
    exporter = CGCExporter(CGCOptions(external_weights=weights_mode != "embedded"))

    class Compiler(_ExternalMlirCompiler):
        def compile_onnx(self, serialized, **kwargs):
            data_path = kwargs["output_data_file"]
            if data_path is not None and weights_mode == "external":
                data_path.write_bytes(serialized)
            return b"module { cgc.test }"

    monkeypatch.setattr(
        "winml.modelkit.export.cgc.exporter.FoundryCompiler", Compiler,
    )
    if existing:
        exporter.export_onnx(source, output)
        if weights_mode == "absent":
            output.with_name(f"{output.name}.data").write_bytes(original_source)
    artifacts = exporter.output_artifacts(output)
    before = {path.name: path.read_bytes() for path in artifacts if path.exists()}
    model.graph.input[0].name = "updated_input"
    model.graph.node[0].input[0] = "updated_input"
    save_model(model, source)
    updated_source = source.read_bytes()
    expected_dir = tmp_path / "expected"
    exporter.export_onnx(source, expected_dir / output.name)
    expected = {
        path.name: path.read_bytes()
        for path in exporter.output_artifacts(expected_dir / output.name)
        if path.exists()
    }
    real_replace = Path.replace
    failure_names = {
        "weights": "model.mlir.data", "metadata": "model_metadata.json", "model": "model.mlir",
    }
    triggered = False

    def fail_once(path, target):
        nonlocal triggered
        target = Path(target)
        is_backup = path.parent == tmp_path and target.parent.name == "backup"
        is_publication = target.parent == tmp_path and path.name == target.name
        if not triggered and (
            (failure == "backup" and is_backup and path.name == output.name)
            or (is_publication and target.name == failure_names.get(failure))
        ):
            triggered = True
            raise PermissionError("publication probe")
        return real_replace(path, target)

    should_fail = (
        failure in ("metadata", "model")
        or (failure == "backup" and existing)
        or (failure == "weights" and weights_mode == "external")
    )
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "replace", fail_once)
        if should_fail:
            with pytest.raises(PermissionError, match="publication probe"):
                exporter.export_onnx(source, output)
        else:
            exporter.export_onnx(source, output)
    after = {path.name: path.read_bytes() for path in artifacts if path.exists()}
    assert triggered == should_fail
    assert after == (before if should_fail else expected)
    assert source.read_bytes() == updated_source
    assert not list(tmp_path.glob(".model.mlir.*"))


def test_export_prints_progress(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    output = tmp_path / "model.mlir"
    save_model(_make_model(), str(source))

    class FakeCompiler:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def compile_onnx(self, _source, **_kwargs):
            return b"module { cgc.test }"

    with patch(
        "winml.modelkit.export.cgc.exporter.FoundryCompiler",
        return_value=FakeCompiler(),
    ):
        CGCExporter(CGCOptions()).export(source, output)

    console_output = capsys.readouterr().out
    assert "ONNX TO CGC EXPORT PROCESS" in console_output
    assert "Input:" in console_output
    assert "Output:" in console_output
    assert "Format: CGC MLIR" in console_output
    assert "CGC EXPORT COMPLETE" in console_output


@pytest.mark.parametrize(
    ("options", "expected_names"),
    [
        (CGCOptions(), ("model.mlir", "model_metadata.json")),
        (
            CGCOptions(external_weights=True),
            ("model.mlir", "model.mlir.data", "model_metadata.json"),
        ),
    ],
)
def test_output_artifacts_depend_only_on_external_weights(
    tmp_path: Path,
    options: CGCOptions,
    expected_names: tuple[str, ...],
) -> None:
    artifacts = CGCExporter(options).output_artifacts(tmp_path / "model.mlir")
    assert tuple(path.name for path in artifacts) == expected_names


@pytest.mark.parametrize("dimension", ["batch_size", "batch", "batchSize", 1, 4])
@pytest.mark.parametrize("explicit", ["", "batch_size=3", "seq=128"])
def test_auto_freeze_matches_input_symbol_and_preserves_options(tmp_path, dimension, explicit):
    model = _make_model()
    model.graph.input[0].CopyFrom(
        helper.make_tensor_value_info("input", TensorProto.FLOAT, [dimension, 2]),
    )
    source = tmp_path / "source.onnx"
    save_model(model, source)
    exporter = CGCExporter(CGCOptions(freeze_dims=explicit))
    with patch.object(exporter, "_export_mlir"):
        exporter.export_onnx(source, tmp_path / "model.mlir")
        if explicit:
            name, size = explicit.split("=")
            expected = {name: int(size)}
            if name != "batch_size" and dimension == "batch_size":
                expected["batch_size"] = 1
            assert exporter._freeze_dims == expected
        else:
            assert exporter._freeze_dims == ({"batch_size": 1} if dimension == "batch_size" else {})
        save_model(_make_model(), source)
        exporter.export_onnx(source, tmp_path / "static.mlir")
        if explicit:
            name, size = explicit.split("=")
            assert exporter._freeze_dims == {name: int(size)}
        else:
            assert exporter._freeze_dims == {}


@pytest.mark.parametrize("location", ["output", "value_info", "initializer"])
def test_auto_freeze_ignores_non_input_symbols(tmp_path, location):
    model = _make_model()
    value = helper.make_tensor_value_info("weight", TensorProto.FLOAT, ["batch_size"])
    if location == "initializer":
        model.graph.input.append(value)
    else:
        getattr(model.graph, location).append(value)
    source = tmp_path / "source.onnx"
    save_model(model, source)
    exporter = CGCExporter(CGCOptions())
    with patch.object(exporter, "_export_mlir"):
        exporter.export_onnx(source, tmp_path / "model.mlir")
    assert exporter._freeze_dims == {}


def test_freeze_dims_are_forwarded_to_foundry(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    output = tmp_path / "model.mlir"
    save_model(_make_model(), str(source))

    class FakeCompiler:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def compile_onnx(self, _source, **kwargs):
            assert kwargs["freeze_dims"] == {"batch": 1, "seq": 128}
            return b"module { cgc.test }"

    with patch(
        "winml.modelkit.export.cgc.exporter.FoundryCompiler",
        return_value=FakeCompiler(),
    ):
        CGCExporter(
            CGCOptions(freeze_dims="batch=1,seq=128")
        ).export_onnx(source, output)


@pytest.mark.parametrize(
    "freeze_dims",
    ["batch", "=1", "batch=x", "batch=0", "batch=1,batch=2"],
)
def test_invalid_freeze_dims_are_rejected(freeze_dims: str) -> None:
    with pytest.raises(ValueError, match="freeze-dims"):
        CGCExporter(CGCOptions(freeze_dims=freeze_dims))


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            FoundryCompileError(
                9,
                native_message="Operation is not registered.",
                unsupported_op="com.example.CustomOp",
            ),
            "ONNX operator 'com.example.CustomOp' is not supported.",
        ),
        (
            FoundryCompileError(
                11,
                native_message="File does not exist.",
                missing_external_data="weights/model.data",
            ),
            "ONNX external weights file was not found: 'weights/model.data'.",
        ),
        (
            FoundryCompileError(
                3,
                native_message="Could not infer output shape.",
            ),
            "--options freeze-dims=batch=1,seq=128",
        ),
    ],
)
def test_exporter_formats_foundry_diagnostics(
    error: FoundryCompileError,
    expected: str,
) -> None:
    message = CGCExporter(CGCOptions())._format_foundry_error(error)

    assert f"[{error.result_name}]" in message
    assert error.native_message in message
    assert expected in message
