# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for WinMLQairtSession class.

These tests verify the QAIRT session logic without requiring the actual
QAIRT SDK or QNN EP. All external dependencies are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest

from .conftest import QNN_VENDOR_ID


# Mock the EP registration to avoid access violations from native DLLs
@pytest.fixture(autouse=True)
def mock_ep_registration():
    """Prevent WinML EP registration from loading native DLLs."""
    from winml.modelkit.session import EPDeviceTarget

    from .conftest import make_stub_winml_ep_device

    fake_ort_npu = MagicMock()
    fake_ort_npu.ep_name = "QNNExecutionProvider"
    fake_ort_npu.device.type.name = "NPU"
    fake_ort_npu.device.vendor_id = QNN_VENDOR_ID
    fake_ort_npu.device.device_id = 0x0001
    fake_qnn_target = EPDeviceTarget(ep="QNNExecutionProvider", device="npu")
    fake_qnn_ep_device = make_stub_winml_ep_device(fake_ort_npu, "QNNExecutionProvider")
    with (
        # Patch resolve_device where it is imported (in qairt_session module)
        patch(
            "winml.modelkit.session.qairt.qairt_session.resolve_device",
            return_value=fake_qnn_target,
        ),
        # auto_device is the new compound resolution step.
        patch("winml.modelkit.session.qairt.qairt_session.WinMLEPRegistry") as mock_reg,
        patch("winml.modelkit.session.session.ort.InferenceSession"),
        patch(
            "winml.modelkit.session.session.ort.SessionOptions",
            return_value=MagicMock(),
        ),
    ):
        mock_reg.instance.return_value.auto_device.return_value = fake_qnn_ep_device
        yield


@pytest.fixture
def mock_qairt_sdk_root(tmp_path: Path) -> Path:
    """Create minimal mock QAIRT SDK directory structure."""
    sdk_root = tmp_path / "qairt_sdk"
    (sdk_root / "lib" / "python").mkdir(parents=True)
    (sdk_root / "lib" / "x86_64-windows-msvc").mkdir(parents=True)
    (sdk_root / "lib" / "aarch64-windows-msvc").mkdir(parents=True)
    (sdk_root / "bin" / "aarch64-windows-msvc").mkdir(parents=True)
    # Create the utility executable
    (sdk_root / "bin" / "aarch64-windows-msvc" / "qnn-context-binary-utility.exe").touch()
    return sdk_root


class TestResolveSdkPath:
    """Test SDK path resolution logic."""

    def test_raises_when_sdk_path_not_found(self, simple_matmul_onnx: Path, monkeypatch):
        """Test FileNotFoundError raised when no SDK path configured.

        Key branch: raise FileNotFoundError when QNN_SDK_ROOT and QAIRT_SDK_ROOT not set
        """
        from winml.modelkit.session import WinMLQairtSession

        # Clear SDK environment variables
        monkeypatch.delenv("QNN_SDK_ROOT", raising=False)
        monkeypatch.delenv("QAIRT_SDK_ROOT", raising=False)

        with pytest.raises(FileNotFoundError, match="QAIRT SDK path not found"):
            WinMLQairtSession(onnx_path=simple_matmul_onnx)


class TestCompileIdempotency:
    """Test compile() idempotency behavior."""

    def test_default_construction_defers_session_creation(
        self, simple_matmul_onnx: Path, mock_qairt_sdk_root: Path, monkeypatch
    ):
        """Default QAIRT construction stays in compile workflow until compile()."""
        from winml.modelkit.session import WinMLQairtSession

        monkeypatch.setenv("QNN_SDK_ROOT", str(mock_qairt_sdk_root))

        session = WinMLQairtSession(onnx_path=simple_matmul_onnx)

        assert session._session is None

    def test_compile_runs_qairt_pipeline_once(
        self, simple_matmul_onnx: Path, mock_qairt_sdk_root: Path, monkeypatch
    ):
        """First compile runs the QAIRT pipeline; later calls stay idempotent."""
        from winml.modelkit.session import WinMLQairtSession

        monkeypatch.setenv("QNN_SDK_ROOT", str(mock_qairt_sdk_root))

        session = WinMLQairtSession(onnx_path=simple_matmul_onnx)

        with (
            patch.object(session, "_compile_to_qnn_bin") as mock_compile,
            patch.object(session, "_create_context_bin_info") as mock_create_info,
            patch.object(session, "_wrap_bin_to_onnx") as mock_wrap,
            patch.object(
                session,
                "_create_inference_session",
                side_effect=lambda: setattr(session, "_session", MagicMock()),
            ) as mock_create_session,
            patch(
                "winml.modelkit.session.qairt.qairt_session.ensure_venv",
                return_value=Path("python.exe"),
            ),
        ):
            session.compile()
            session.compile()

        mock_compile.assert_called_once_with(Path("python.exe"))
        mock_create_info.assert_called_once_with()
        mock_wrap.assert_called_once_with()
        mock_create_session.assert_called_once_with()

    def test_rejects_explicitly_disabled_ep_context(
        self, simple_matmul_onnx: Path, mock_qairt_sdk_root: Path
    ):
        """QAIRT sessions reject configs that disable EPContext generation."""
        from winml.modelkit.compiler import EPConfig
        from winml.modelkit.session import WinMLQairtSession

        ep_config = EPConfig(
            qnn_sdk_root=mock_qairt_sdk_root,
            provider_options={"backend_path": "QnnHtp.dll"},
            enable_ep_context=False,
        )

        with pytest.raises(ValueError, match="WinMLQairtSession requires enable_ep_context=True"):
            WinMLQairtSession(onnx_path=simple_matmul_onnx, ep_config=ep_config)

        assert ep_config.enable_ep_context is False
        assert ep_config.provider_options == {"backend_path": "QnnHtp.dll"}


class TestCompileToQnnBin:
    """Test _compile_to_qnn_bin subprocess handling."""

    def test_raises_on_subprocess_failure(
        self, simple_matmul_onnx: Path, mock_qairt_sdk_root: Path, monkeypatch
    ):
        """Test RuntimeError raised when subprocess returns non-zero.

        Key branch: if result.returncode != 0: raise RuntimeError
        """
        from winml.modelkit.session import WinMLQairtSession

        monkeypatch.setenv("QNN_SDK_ROOT", str(mock_qairt_sdk_root))

        session = WinMLQairtSession(onnx_path=simple_matmul_onnx)

        # Mock subprocess.run to return failure
        mock_result = CompletedProcess(
            args=[], returncode=1, stderr="QAIRT compilation failed: unsupported op"
        )

        with (
            patch("subprocess.run", return_value=mock_result),
            pytest.raises(RuntimeError, match="QAIRT compilation failed"),
        ):
            session._compile_to_qnn_bin(Path("python.exe"))


class TestWrapBinToOnnx:
    """Test _wrap_bin_to_onnx logic."""

    def test_validates_json_format(
        self, simple_matmul_onnx: Path, mock_qairt_sdk_root: Path, monkeypatch
    ):
        """Test RuntimeError when JSON doesn't have required keys.

        Key branch: if "info" not in qnn_json_obj or "graphs" not in...
        """
        from winml.modelkit.session import WinMLQairtSession

        monkeypatch.setenv("QNN_SDK_ROOT", str(mock_qairt_sdk_root))

        session = WinMLQairtSession(onnx_path=simple_matmul_onnx)

        # Create invalid JSON (missing "info" and "graphs" keys)
        invalid_json = {"version": "1.0"}
        session._bin_info_path.parent.mkdir(parents=True, exist_ok=True)
        with session._bin_info_path.open("w") as f:
            json.dump(invalid_json, f)

        with pytest.raises(RuntimeError, match="Unrecognized bin info JSON format"):
            session._wrap_bin_to_onnx()

    def test_patches_tensor_ids_from_raw_json(
        self, simple_matmul_onnx: Path, mock_qairt_sdk_root: Path, monkeypatch
    ):
        """Test that tensor IDs are patched from raw JSON graphInputs/graphOutputs.

        Key logic: The fix for parse_qnn_graph not setting id field.
        Verifies qnn_input_tensor_dic[name].id and qnn_output_tensor_dic[name].id are set.
        """
        from unittest.mock import MagicMock

        from winml.modelkit.session import WinMLQairtSession

        monkeypatch.setenv("QNN_SDK_ROOT", str(mock_qairt_sdk_root))

        session = WinMLQairtSession(onnx_path=simple_matmul_onnx)

        # Create valid JSON with known tensor IDs
        valid_json = {
            "info": {
                "buildId": "2.26.0",
                "graphs": [
                    {
                        "info": {
                            "name": "test_graph",
                            "graphInputs": [
                                {
                                    "info": {
                                        "name": "input_tensor",
                                        "id": 42,
                                        "type": {"info": {"dataType": "FLOAT32"}},
                                        "dimensions": [1, 4],
                                    }
                                }
                            ],
                            "graphOutputs": [
                                {
                                    "info": {
                                        "name": "output_tensor",
                                        "id": 99,
                                        "type": {"info": {"dataType": "FLOAT32"}},
                                        "dimensions": [1, 4],
                                    }
                                }
                            ],
                        }
                    }
                ],
            }
        }
        session._bin_info_path.parent.mkdir(parents=True, exist_ok=True)
        with session._bin_info_path.open("w") as f:
            json.dump(valid_json, f)

        # Create mock bin file
        session._bin_path.touch()

        # Mock parse_qnn_graph to return tensor objects without id set
        mock_input_tensor = MagicMock()
        mock_input_tensor.id = None
        mock_output_tensor = MagicMock()
        mock_output_tensor.id = None

        def mock_parse_qnn_graph(qnn_graph, input_dic, output_dic):
            input_dic["input_tensor"] = mock_input_tensor
            output_dic["output_tensor"] = mock_output_tensor
            return "test_graph"

        # Mock generate_wrapper_onnx_file to just create empty file
        def mock_generate_wrapper(**kwargs):
            Path(kwargs["model_file_name"]).touch()

        with (
            patch(
                "onnxruntime.tools.qnn.gen_qnn_ctx_onnx_model.parse_qnn_graph", mock_parse_qnn_graph
            ),
            patch(
                "onnxruntime.tools.qnn.gen_qnn_ctx_onnx_model.generate_wrapper_onnx_file",
                mock_generate_wrapper,
            ),
        ):
            session._wrap_bin_to_onnx()

        # Verify tensor IDs were patched from raw JSON
        assert mock_input_tensor.id == 42, f"Expected input id 42, got {mock_input_tensor.id}"
        assert mock_output_tensor.id == 99, f"Expected output id 99, got {mock_output_tensor.id}"


class TestQairtSessionPaths:
    """Test QAIRT-specific path construction."""

    def test_constructs_qairt_paths_from_onnx_path(
        self, simple_matmul_onnx: Path, mock_qairt_sdk_root: Path, monkeypatch
    ):
        """Test that _bin_path, _bin_info_path, _ctx_path are constructed correctly.

        Key logic: Paths are based on input ONNX model's stem and parent directory.
        """
        from winml.modelkit.session import WinMLQairtSession

        monkeypatch.setenv("QNN_SDK_ROOT", str(mock_qairt_sdk_root))

        session = WinMLQairtSession(onnx_path=simple_matmul_onnx)

        # Verify paths are constructed from the input model's stem
        model_stem = simple_matmul_onnx.stem  # "test_matmul"
        model_dir = simple_matmul_onnx.parent

        assert session._bin_path == model_dir / f"{model_stem}_qnn_ctx_qnn.bin"
        assert session._bin_info_path == model_dir / f"{model_stem}_cache_info.json"
        assert session._ctx_path == model_dir / f"{model_stem}_ctx.onnx"

    def test_uses_sdk_root_from_ep_config(
        self, simple_matmul_onnx: Path, mock_qairt_sdk_root: Path, monkeypatch
    ):
        """Test that SDK root from ep_config takes precedence over env var.

        Key logic: ep_config.qnn_sdk_root is used if provided.
        """
        from winml.modelkit.compiler import EPConfig
        from winml.modelkit.session import WinMLQairtSession

        # Set env var to a different path
        monkeypatch.setenv("QNN_SDK_ROOT", str(mock_qairt_sdk_root.parent / "wrong_sdk"))

        # Provide SDK root via ep_config
        ep_config = EPConfig(qnn_sdk_root=mock_qairt_sdk_root)

        session = WinMLQairtSession(onnx_path=simple_matmul_onnx, ep_config=ep_config)

        # Should use ep_config value, not env var
        assert session._qnn_sdk_root == mock_qairt_sdk_root
