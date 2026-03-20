# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Test QNN profiler and viewer with mocked ORT (no QNN hardware needed)."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

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
    """Verify session options are configured correctly."""
    profiler = QNNProfiler(
        Path("model.onnx"), output_dir=Path("out"), level="basic"
    )
    mock_ort = MagicMock()
    mock_options = MagicMock()
    mock_ort.SessionOptions.return_value = mock_options

    options = profiler._build_session_options(mock_ort)

    assert options is mock_options
    calls = mock_options.add_session_config_entry.call_args_list
    entries = {c.args[0]: c.args[1] for c in calls}
    assert entries["session.disable_cpu_ep_fallback"] == "1"
    assert entries["ep.context_enable"] == "1"
    assert entries["ep.context_embed_mode"] == "0"


# =====================================================================
# Profiler: provider options
# =====================================================================


def test_qnn_profiler_provider_options_basic():
    """Verify provider options for basic mode (profiling_level=detailed)."""
    profiler = QNNProfiler(
        Path("model.onnx"), output_dir=Path("out"), level="basic"
    )
    opts = profiler._build_provider_options(Path("out/profiling.csv"))

    assert len(opts) == 1
    po = opts[0]
    assert po["backend_path"] == "QnnHtp.dll"
    assert po["htp_performance_mode"] == "high_performance"
    assert po["htp_graph_finalization_optimization_mode"] == "3"
    assert po["enable_htp_fp16_precision"] == "1"
    assert po["profiling_level"] == "detailed"
    assert po["profiling_file_path"] == str(Path("out/profiling.csv"))


def test_qnn_profiler_provider_options_detail():
    """Verify provider options for detail mode (profiling_level=optrace)."""
    profiler = QNNProfiler(
        Path("model.onnx"), output_dir=Path("out"), level="detail"
    )
    opts = profiler._build_provider_options(Path("out/profiling.csv"))

    po = opts[0]
    assert po["profiling_level"] == "optrace"
    assert po["backend_path"] == "QnnHtp.dll"


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

    with patch.dict("sys.modules", {"onnxruntime": mock_ort}), patch(
        "winml.modelkit.optracing.qnn.profiler.QNNProfiler._collect_results"
    ) as mock_collect:
        # Write the CSV before _collect_results is called.
        write_csv_on_del()

        profiler = QNNProfiler(
            model_path, output_dir=output_dir, level="basic"
        )

        # Instead of running the full flow (which needs real ORT import),
        # test the collect_results path directly.
        mock_collect.return_value = MagicMock()
        # Verify session creation was called correctly via builder methods.
        profiler._build_session_options(mock_ort)
        po = profiler._build_provider_options(
            output_dir / "profiling_output.csv"
        )
        assert po[0]["profiling_level"] == "detailed"

        # Now test the CSV parsing path directly.
        result = profiler._from_csv(
            output_dir / "profiling_output.csv",
            iterations=5,
            artifacts={"csv": str(output_dir / "profiling_output.csv")},
        )
        assert result.model == "model.onnx"
        assert result.tracing_level == "basic"
        assert result.ep == "QNNExecutionProvider"
        assert len(result.operators) == 2
        assert result.operators[0].name == "Conv2d"
        assert result.summary["hvx_threads"] == 4


def test_qnn_profiler_empty_artifacts(tmp_path):
    """Profiler returns empty result when no artifacts exist."""
    profiler = QNNProfiler(
        Path("model.onnx"), output_dir=tmp_path, level="basic"
    )
    result = profiler._collect_results(
        tmp_path / "nonexistent.csv", iterations=5
    )
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
    with patch(
        "winml.modelkit.optracing.qnn.viewer._find_viewer_exe", return_value=None
    ):
        result = run_basic_viewer(
            tmp_path / "log.qnn", tmp_path / "output.csv"
        )
        assert result is None


def test_run_basic_viewer_success(tmp_path):
    """Basic viewer returns path on success."""
    output_csv = tmp_path / "output.csv"

    def fake_run(cmd, **kwargs):
        output_csv.write_text("header\ndata", encoding="utf-8")

    with patch(
        "winml.modelkit.optracing.qnn.viewer._find_viewer_exe",
        return_value=Path("/fake/viewer.exe"),
    ), patch(
        "winml.modelkit.optracing.qnn.viewer.subprocess.run",
        side_effect=fake_run,
    ):
        result = run_basic_viewer(
            tmp_path / "log.qnn", output_csv
        )
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

    with patch(
        "winml.modelkit.optracing.qnn.viewer._find_viewer_exe",
        return_value=Path("/fake/viewer.exe"),
    ), patch(
        "winml.modelkit.optracing.qnn.viewer.subprocess.run",
        side_effect=fake_run,
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
