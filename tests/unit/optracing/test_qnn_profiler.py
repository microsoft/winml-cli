# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Test QNN profiler and viewer with mocked ORT (no QNN hardware needed)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from onnx import TensorProto, helper, load, save_model

from winml.modelkit.optracing.qnn.profiler import (
    QNNProfiler,
    _ort_type_to_numpy,
    _resolve_shape,
)
from winml.modelkit.optracing.qnn.viewer import (
    _DEFAULT_CONFIG,
    find_qnn_sdk,
    run_basic_viewer,
    run_qhas_viewer,
)


# =====================================================================
# Profiler: session options
# =====================================================================


def test_qnn_profiler_creates_session_options():
    """Basic level sets fallback only; EPContext entries are detail-only."""
    profiler = QNNProfiler(Path("model.onnx"), output_dir=Path("out"), level="basic")
    mock_ort = MagicMock()
    mock_options = MagicMock()
    mock_ort.SessionOptions.return_value = mock_options

    options = profiler._build_session_options(mock_ort)

    assert options is mock_options
    calls = mock_options.add_session_config_entry.call_args_list
    entries = {c.args[0]: c.args[1] for c in calls}
    assert entries["session.disable_cpu_ep_fallback"] == "1"
    assert "ep.context_enable" not in entries
    assert "ep.context_embed_mode" not in entries


def test_qnn_profiler_session_options_detail_enables_epcontext():
    """Detail level additionally enables the EPContext session entries."""
    profiler = QNNProfiler(Path("model.onnx"), output_dir=Path("out"), level="detail")
    mock_ort = MagicMock()
    mock_options = MagicMock()
    mock_ort.SessionOptions.return_value = mock_options

    profiler._build_session_options(mock_ort)

    calls = mock_options.add_session_config_entry.call_args_list
    entries = {c.args[0]: c.args[1] for c in calls}
    assert entries["session.disable_cpu_ep_fallback"] == "1"
    assert entries["ep.context_enable"] == "1"
    assert entries["ep.context_embed_mode"] == "0"


# =====================================================================
# Profiler: provider options
# =====================================================================


def test_qnn_profiler_provider_options_basic():
    """Verify provider options for basic mode (profiling_level=detailed).

    The QNN backend/device is now selected by ``add_ep_for_device``, so the
    options dict no longer carries ``backend_path``.
    """
    profiler = QNNProfiler(Path("model.onnx"), output_dir=Path("out"), level="basic")
    po = profiler._build_provider_options(Path("out/profiling.csv"))

    assert po["htp_performance_mode"] == "high_performance"
    assert po["htp_graph_finalization_optimization_mode"] == "3"
    assert po["enable_htp_fp16_precision"] == "1"
    assert po["profiling_level"] == "detailed"
    assert po["profiling_file_path"] == str(Path("out/profiling.csv"))


def test_qnn_profiler_provider_options_detail():
    """Verify provider options for detail mode (profiling_level=optrace)."""
    profiler = QNNProfiler(Path("model.onnx"), output_dir=Path("out"), level="detail")
    po = profiler._build_provider_options(Path("out/profiling.csv"))

    assert po["profiling_level"] == "optrace"


# =====================================================================
# Profiler: input generation helpers
# =====================================================================


def test_ort_type_to_numpy_known_types():
    """Verify ORT type string mapping to NumPy dtypes."""
    assert _ort_type_to_numpy("tensor(float)") == np.dtype("float32")
    assert _ort_type_to_numpy("tensor(float16)") == np.dtype("float16")
    assert _ort_type_to_numpy("tensor(int64)") == np.dtype("int64")
    assert _ort_type_to_numpy("tensor(bool)") == np.dtype("bool")


def test_ort_type_to_numpy_unknown_fallback():
    """Unknown ORT types fall back to float32."""
    assert _ort_type_to_numpy("tensor(bfloat16)") == np.dtype("float32")


def test_resolve_shape_concrete():
    """Concrete shapes pass through unchanged."""
    assert _resolve_shape([1, 3, 224, 224]) == [1, 3, 224, 224]


def test_resolve_shape_symbolic():
    """Symbolic (string / None / <=0) dimensions become default_dim."""
    assert _resolve_shape(["batch", 3, None, -1], default_dim=1) == [
        1,
        3,
        1,
        1,
    ]


def test_generate_inputs():
    """Verify random input generation from mock session."""
    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input_ids"
    mock_input.shape = [1, 128]
    mock_input.type = "tensor(int64)"
    mock_session.get_inputs.return_value = [mock_input]

    inputs = QNNProfiler._generate_inputs(mock_session)

    assert "input_ids" in inputs
    assert inputs["input_ids"].shape == (1, 128)
    assert inputs["input_ids"].dtype == np.int64


def _profiler_with_input_data(tmp_path, input_data):
    """Build a QNNProfiler carrying input_data without running init side effects."""
    return QNNProfiler(tmp_path / "m.onnx", output_dir=tmp_path, input_data=input_data)


def test_resolve_inputs_uses_provided_data_and_casts(tmp_path):
    """Provided input_data is used (and cast to the session dtype) when keys match."""
    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input_ids"
    mock_input.type = "tensor(int64)"
    mock_session.get_inputs.return_value = [mock_input]

    provided = {"input_ids": np.zeros((1, 4), dtype=np.int32)}
    profiler = _profiler_with_input_data(tmp_path, provided)

    with patch.object(QNNProfiler, "_generate_inputs") as mock_gen:
        inputs = profiler._resolve_inputs(mock_session)

    mock_gen.assert_not_called()
    assert set(inputs) == {"input_ids"}
    assert inputs["input_ids"].dtype == np.int64


def test_resolve_inputs_falls_back_on_key_mismatch(tmp_path):
    """When provided keys don't match the traced session, fall back to random."""
    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "pixel_values"
    mock_input.type = "tensor(float)"
    mock_session.get_inputs.return_value = [mock_input]

    provided = {"input_ids": np.zeros((1, 4), dtype=np.int64)}
    profiler = _profiler_with_input_data(tmp_path, provided)

    sentinel = {"pixel_values": np.zeros((1, 3), dtype=np.float32)}
    with patch.object(QNNProfiler, "_generate_inputs", return_value=sentinel) as mock_gen:
        inputs = profiler._resolve_inputs(mock_session)

    mock_gen.assert_called_once()
    assert inputs is sentinel


def test_resolve_inputs_no_data_uses_random(tmp_path):
    """With no input_data, _resolve_inputs delegates to random generation."""
    mock_session = MagicMock()
    profiler = _profiler_with_input_data(tmp_path, None)

    sentinel = {"x": np.zeros((1,), dtype=np.float32)}
    with patch.object(QNNProfiler, "_generate_inputs", return_value=sentinel) as mock_gen:
        inputs = profiler._resolve_inputs(mock_session)

    mock_gen.assert_called_once()
    assert inputs is sentinel


# =====================================================================
# Profiler: full run with mocked ORT
# =====================================================================


def test_qnn_profiler_run_basic(tmp_path):
    """End-to-end basic run with mocked ORT session."""
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"fake")
    output_dir = tmp_path / "output"

    # Create a minimal CSV so the CSV parser can parse it.
    csv_content = (
        "Msg Timestamp,Message,Time,Unit of Measurement,"
        "Timing Source,Event Level,Event Identifier\n"
        '0,ROOT,4,COUNT,HW,ROOT,"Number of HVX threads used"\n'
        '1,ROOT,100000,CYCLES,HW,ROOT,"Accelerator (execute) time (cycles)"\n'
        '2,NODE,500,CYCLES,HW,SUB-EVENT,"Conv2d:OpId_1 (cycles)"\n'
        '3,NODE,300,CYCLES,HW,SUB-EVENT,"Add:OpId_2 (cycles)"\n'
    )

    # Mock ORT so no real QNN EP is needed.
    mock_ort = MagicMock()
    mock_session = MagicMock()
    mock_ort.SessionOptions.return_value = MagicMock()
    mock_ort.InferenceSession.return_value = mock_session

    mock_input = MagicMock()
    mock_input.name = "input"
    mock_input.shape = [1, 3]
    mock_input.type = "tensor(float)"
    mock_session.get_inputs.return_value = [mock_input]
    mock_session.run.return_value = [np.array([1.0])]

    def write_csv_on_del():
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "profiling_output.csv"
        csv_path.write_text(csv_content, encoding="utf-8")

    # Simulate CSV being flushed when session is deleted.
    mock_ort.InferenceSession.return_value = mock_session

    with (
        patch.dict("sys.modules", {"onnxruntime": mock_ort}),
        patch("winml.modelkit.optracing.qnn.profiler.QNNProfiler._collect_results") as mock_collect,
    ):
        # Write the CSV before _collect_results is called.
        write_csv_on_del()

        profiler = QNNProfiler(model_path, output_dir=output_dir, level="basic")

        # Instead of running the full flow (which needs real ORT import),
        # test the collect_results path directly.
        mock_collect.return_value = MagicMock()
        # Verify session creation was called correctly via builder methods.
        profiler._build_session_options(mock_ort)
        po = profiler._build_provider_options(output_dir / "profiling_output.csv")
        assert po["profiling_level"] == "detailed"

        # Now test the CSV parsing path directly. The fixture holds a single
        # sample, so treat it as one measured iteration with no warmup.
        result = profiler._from_csv(
            output_dir / "profiling_output.csv",
            iterations=1,
            warmup=0,
            artifacts={"csv": str(output_dir / "profiling_output.csv")},
        )
        assert result.model == "model.onnx"
        assert result.tracing_level == "basic"
        assert result.ep == "QNNExecutionProvider"
        assert len(result.operators) == 2
        assert result.operators[0].name == "Conv2d"
        assert result.summary["hvx_threads"] == 4


_CSV_HEADER = (
    "Msg Timestamp,Message,Time,Unit of Measurement,Timing Source,Event Level,Event Identifier\n"
)


def _make_csv_sample(cycles: int, us: int, conv_cycles: int, add_cycles: int) -> str:
    """Build one inference sample block for a basic-mode profiling CSV."""
    return (
        '0,ROOT,4,COUNT,HW,ROOT,"Number of HVX threads used"\n'
        f'1,ROOT,{cycles},CYCLES,HW,ROOT,"Accelerator (execute) time (cycles)"\n'
        f'2,NODE,{conv_cycles},CYCLES,HW,SUB-EVENT,"Conv2d:OpId_1 (cycles)"\n'
        f'3,NODE,{add_cycles},CYCLES,HW,SUB-EVENT,"Add:OpId_2 (cycles)"\n'
        f'4,ROOT,{us},US,HW,ROOT,"Accelerator (execute) time"\n'
    )


def _write_transpose_model(path: Path) -> None:
    """Write a generated ONNX model with node input and attribute metadata."""
    input_info = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, "height", "width"]
    )
    output_info = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, "height", "width", 3]
    )
    node = helper.make_node(
        "Transpose",
        ["input"],
        ["output"],
        name="transpose_node",
        perm=[0, 2, 3, 1],
    )
    graph = helper.make_graph([node], "transpose_graph", [input_info], [output_info])
    save_model(helper.make_model(graph), path)


def _make_node_csv_sample(node_name: str) -> str:
    """Build one inference sample block for a named ONNX node."""
    return (
        '0,ROOT,4,COUNT,HW,ROOT,"Number of HVX threads used"\n'
        '1,ROOT,100,CYCLES,HW,ROOT,"Accelerator (execute) time (cycles)"\n'
        f'2,NODE,25,CYCLES,HW,SUB-EVENT,"{node_name}:OpId_7 (cycles)"\n'
        '3,ROOT,10,US,HW,ROOT,"Accelerator (execute) time"\n'
    )


def test_qnn_profiler_from_csv_skips_warmup(tmp_path):
    """The first ``warmup`` samples are dropped before computing metrics."""
    # One warmup sample with outlier timing, then two measured samples.
    csv_content = (
        _CSV_HEADER
        + _make_csv_sample(cycles=900000, us=9000, conv_cycles=5000, add_cycles=3000)
        + _make_csv_sample(cycles=100000, us=1000, conv_cycles=500, add_cycles=300)
        + _make_csv_sample(cycles=100000, us=1000, conv_cycles=500, add_cycles=300)
    )
    csv_path = tmp_path / "profiling_output.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    profiler = QNNProfiler(tmp_path / "model.onnx", output_dir=tmp_path, level="basic")
    result = profiler._from_csv(csv_path, iterations=2, warmup=1, artifacts={})

    # Only the two measured samples survive; the warmup outlier is excluded.
    assert result.num_samples == 2
    assert result.summary["accel_execute_cycles"] == 100000


def test_qnn_profiler_from_csv_sample_count_mismatch(tmp_path):
    """A measured-sample count that doesn't match ``iterations`` is an error."""
    csv_content = _CSV_HEADER + _make_csv_sample(
        cycles=100000, us=1000, conv_cycles=500, add_cycles=300
    )
    csv_path = tmp_path / "profiling_output.csv"
    csv_path.write_text(csv_content, encoding="utf-8")

    profiler = QNNProfiler(tmp_path / "model.onnx", output_dir=tmp_path, level="basic")
    with pytest.raises(ValueError):
        profiler._from_csv(csv_path, iterations=5, warmup=0, artifacts={})


def test_qnn_profiler_from_csv_omits_onnx_data_without_env(tmp_path, monkeypatch):
    """Basic profiling output is unchanged unless WINMLCLI_OP_ADD_DATA is set."""
    model_path = tmp_path / "model.onnx"
    csv_path = tmp_path / "profiling_output.csv"
    _write_transpose_model(model_path)
    csv_path.write_text(
        _CSV_HEADER + _make_node_csv_sample("transpose_node"),
        encoding="utf-8",
    )

    monkeypatch.delenv("WINMLCLI_OP_ADD_DATA", raising=False)
    profiler = QNNProfiler(model_path, output_dir=tmp_path, level="basic")
    result = profiler._from_csv(csv_path, iterations=1, warmup=0, artifacts={})

    operator = result.to_dict()["operators"][0]
    assert "onnx_op_type" not in operator
    assert "onnx_attributes" not in operator
    assert "onnx_inputs" not in operator


def test_qnn_profiler_from_csv_adds_onnx_data_when_env_is_set(tmp_path, monkeypatch):
    """Basic profiling can include ONNX node type, attributes, and input specs."""
    model_path = tmp_path / "model.onnx"
    csv_path = tmp_path / "profiling_output.csv"
    _write_transpose_model(model_path)
    csv_path.write_text(
        _CSV_HEADER + _make_node_csv_sample("transpose_node"),
        encoding="utf-8",
    )

    monkeypatch.setenv("WINMLCLI_OP_ADD_DATA", "1")
    profiler = QNNProfiler(model_path, output_dir=tmp_path, level="basic")
    result = profiler._from_csv(csv_path, iterations=1, warmup=0, artifacts={})

    operator = result.to_dict()["operators"][0]
    model = load(model_path)
    node = model.graph.node[0]
    graph_input = model.graph.input[0]
    assert operator["onnx_op_type"] == node.op_type
    assert operator["onnx_attributes"] == {"perm": list(node.attribute[0].ints)}
    assert operator["onnx_inputs"] == {
        "data": {
            "name": graph_input.name,
            "data_type": TensorProto.DataType.Name(graph_input.type.tensor_type.elem_type),
            "dims": [
                dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                for dim in graph_input.type.tensor_type.shape.dim
            ],
        }
    }


def test_qnn_profiler_empty_artifacts(tmp_path):
    """Profiler returns empty result when no artifacts exist."""
    profiler = QNNProfiler(Path("model.onnx"), output_dir=tmp_path, level="basic")
    result = profiler._collect_results(tmp_path / "nonexistent.csv", iterations=5, warmup=2)
    assert result.model == "model.onnx"
    assert len(result.operators) == 0
    assert result.num_samples == 0


# =====================================================================
# Viewer: SDK detection
# =====================================================================


def test_find_qnn_sdk_from_env(monkeypatch, tmp_path):
    """Test SDK detection from QNN_SDK_ROOT env var."""
    sdk_dir = tmp_path / "qnn_sdk"
    sdk_dir.mkdir()
    monkeypatch.setenv("QNN_SDK_ROOT", str(sdk_dir))

    result = find_qnn_sdk()
    assert result == sdk_dir


def test_find_qnn_sdk_not_found(monkeypatch):
    """Test graceful None when SDK not found."""
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
    # Patch common paths to nonexistent directories.
    with patch(
        "winml.modelkit.optracing.qnn.viewer._COMMON_SDK_PATHS",
        ["/nonexistent/path1", "/nonexistent/path2"],
    ):
        result = find_qnn_sdk()
        assert result is None


def test_find_qnn_sdk_from_common_path(monkeypatch, tmp_path):
    """Test SDK detection from common installation paths."""
    monkeypatch.delenv("QNN_SDK_ROOT", raising=False)

    # Create a fake SDK directory with bin/ subdirectory.
    sdk_version_dir = tmp_path / "2.28.0"
    (sdk_version_dir / "bin").mkdir(parents=True)

    with patch(
        "winml.modelkit.optracing.qnn.viewer._COMMON_SDK_PATHS",
        [str(tmp_path)],
    ):
        result = find_qnn_sdk()
        assert result == sdk_version_dir


# =====================================================================
# Viewer: basic viewer
# =====================================================================


def test_run_basic_viewer_no_sdk(tmp_path):
    """Basic viewer returns None when SDK is not found."""
    with patch("winml.modelkit.optracing.qnn.viewer._find_viewer_exe", return_value=None):
        result = run_basic_viewer(tmp_path / "log.qnn", tmp_path / "output.csv")
        assert result is None


def test_run_basic_viewer_success(tmp_path):
    """Basic viewer returns path on success."""
    output_csv = tmp_path / "output.csv"

    def fake_run(cmd, **kwargs):
        output_csv.write_text("header\ndata", encoding="utf-8")

    with (
        patch(
            "winml.modelkit.optracing.qnn.viewer._find_viewer_exe",
            return_value=Path("/fake/viewer.exe"),
        ),
        patch(
            "winml.modelkit.optracing.qnn.viewer.subprocess.run",
            side_effect=fake_run,
        ),
    ):
        result = run_basic_viewer(tmp_path / "log.qnn", output_csv)
        assert result == output_csv


# =====================================================================
# Viewer: QHAS viewer
# =====================================================================


def test_run_qhas_viewer_no_schematic(tmp_path):
    """QHAS viewer returns None when schematic file does not exist."""
    with patch(
        "winml.modelkit.optracing.qnn.viewer._find_viewer_exe",
        return_value=Path("/fake/viewer.exe"),
    ):
        result = run_qhas_viewer(
            tmp_path / "log.qnn",
            tmp_path / "nonexistent_schematic.bin",
            tmp_path / "output.json",
        )
        assert result is None


def test_run_qhas_viewer_writes_config(tmp_path):
    """QHAS viewer writes the optrace config JSON."""
    schematic = tmp_path / "model_schematic.bin"
    schematic.write_bytes(b"fake")
    output = tmp_path / "output.json"

    def fake_run(cmd, **kwargs):
        output.write_text("{}", encoding="utf-8")

    with (
        patch(
            "winml.modelkit.optracing.qnn.viewer._find_viewer_exe",
            return_value=Path("/fake/viewer.exe"),
        ),
        patch(
            "winml.modelkit.optracing.qnn.viewer.subprocess.run",
            side_effect=fake_run,
        ),
    ):
        run_qhas_viewer(
            tmp_path / "log.qnn",
            schematic,
            output,
        )
        config_path = tmp_path / "optrace_config.json"
        assert config_path.is_file()
        import json

        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["features"]["qhas_json"] is True


# =====================================================================
# Viewer: default config
# =====================================================================


def test_default_config_has_expected_features():
    """Verify the default QHAS config contains expected feature flags."""
    features = _DEFAULT_CONFIG["features"]
    assert features["qhas_json"] is True
    assert features["qhas_schema"] is True
    assert features["htp_json"] is True
    assert features["runtrace"] is True
    assert features["memory_info"] is True
    assert features["traceback"] is True
    assert features["enable_input_output_flow_events"] is True
    assert features["enable_sequencer_flow_events"] is True
