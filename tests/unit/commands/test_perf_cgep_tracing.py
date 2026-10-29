# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""CGEP tracing argument validation after config defaults are applied."""

import json
from unittest.mock import MagicMock, patch

import onnx
import pytest
from click.testing import CliRunner

from winml.modelkit.commands.perf import BenchmarkResult, perf


@pytest.fixture
def model(tmp_path):
    path = tmp_path / "model.onnx"
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 3])
    onnx.save(
        onnx.helper.make_model(
            onnx.helper.make_graph(
                [onnx.helper.make_node("Identity", ["x"], ["y"])], "test", [x], [y]
            )
        ),
        path,
    )
    return path


@pytest.fixture
def invoke(model, tmp_path):
    instances = []

    def create(config):
        instance = MagicMock()
        instance.config = config
        instance.run.return_value = BenchmarkResult(config=config)
        instances.append(instance)
        return instance

    with (
        patch("winml.modelkit.commands.perf.PerfBenchmark", side_effect=create),
        patch("winml.modelkit.commands.perf.display_console_report"),
        patch("winml.modelkit.commands.perf.write_json_report"),
        patch("winml.modelkit.session.gpu_capture.GpuCapture") as capture_type,
    ):
        capture = capture_type.return_value.__enter__.return_value
        capture.finish.return_value = {}

        def run(*args):
            result = CliRunner().invoke(
                perf,
                [
                    "-m",
                    str(model),
                    "--runtime",
                    "winml-ort",
                    "-o",
                    str(tmp_path / "out.json"),
                    *args,
                ],
                obj={},
            )
            return result, capture_type, capture, instances

        yield run


@pytest.mark.parametrize("target", ["auto", "dml", "d3d12"])
def test_untraced_cgep_rejects_lowering(invoke, target):
    result, capture_type, _, instances = invoke(
        "--ep", "winmlcg", "--device", "gpu", "--cgc-lowering-target", target
    )
    assert result.exit_code == 2, result.output
    assert "--etw-tracing" in result.output
    assert not instances
    capture_type.assert_not_called()


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("target", ["auto", "dml"])
def test_config_target_enables_cgep_trace(invoke, tmp_path, structured, target):
    compile_config = (
        {"ep_device": {"ep": "winmlcg", "device": "gpu"}}
        if structured
        else {"execution_provider": "winmlcg", "device": "gpu"}
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"compile": compile_config}), encoding="utf-8")
    result, capture_type, capture, instances = invoke(
        "--config",
        str(config),
        "--etw-tracing",
        str(tmp_path / "trace"),
        "--cgc-lowering-target",
        target,
    )
    assert result.exit_code == 0, result.output
    capture_type.assert_called_once()
    assert capture.execution_path == "winmlcgep"
    assert capture.lowering_target == target
    assert instances[0].config.device == "gpu"
    assert instances[0].config.ep == "winmlcg"
    instances[0].close.assert_called()
    capture.finish.assert_called_once()


@pytest.mark.parametrize("override", [("--ep", "cpu"), ("--device", "cpu")])
def test_cli_override_wins_over_config_before_trace_validation(invoke, tmp_path, override):
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"compile": {"execution_provider": "winmlcg", "device": "gpu"}}),
        encoding="utf-8",
    )
    result, capture_type, _, instances = invoke(
        "--config",
        str(config),
        "--etw-tracing",
        str(tmp_path / "trace"),
        *override,
    )
    assert result.exit_code == 2, result.output
    assert not instances
    capture_type.assert_not_called()


@pytest.mark.parametrize("suffix,target", [(".onnx", "d3d12"), (".mlir", "auto")])
def test_invalid_cgep_trace_fails_before_start(invoke, model, tmp_path, suffix, target):
    path = model.with_suffix(suffix)
    if path != model:
        path.touch()
    result, capture_type, _, instances = invoke(
        "-m",
        str(path),
        "--ep",
        "winmlcg",
        "--device",
        "gpu",
        "--etw-tracing",
        str(tmp_path / "trace"),
        "--cgc-lowering-target",
        target,
    )
    assert result.exit_code == 2, result.output
    assert "ONNX and auto/dml" in result.output
    assert not instances
    capture_type.assert_not_called()
