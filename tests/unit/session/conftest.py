# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared fixtures for WinMLSession tests.

EP Markers:
    Use @pytest.mark.ep("qnn") or @pytest.mark.ep("openvino") to mark tests
    that require a specific execution provider. Tests will be skipped if the
    EP is not available.

    Example:
        @pytest.mark.ep("qnn")
        def test_qnn_inference(self, simple_matmul_onnx, qnn_npu_ep_device, fake_ort_npu):
            with patch("winml.modelkit.session.session.WinMLEPRegistry") as mock_reg:
                mock_reg.instance.return_value.register_ep.return_value = [fake_ort_npu]
                session = WinMLSession(onnx_path=simple_matmul_onnx, ep_device=qnn_npu_ep_device)
            ...
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from winml.modelkit.ep_path import EPEntry, PyPISource
from winml.modelkit.session import EPDeviceTarget, WinMLEP, WinMLEPDevice, wrap_ort_device
from winml.modelkit.session.session import WinMLSession


# Qualcomm vendor ID used in test fixtures — 0x4D4F is the 16-bit prefix from Qualcomm's
# device identification scheme. Centralised here so all session tests share one definition.
QNN_VENDOR_ID: int = 0x4D4F


def _stub_ep_entry(ep_name: str) -> EPEntry:
    """Build a minimal EPEntry suitable for wrapping a mocked OrtEpDevice.

    The dll_path is fictional — tests never load the DLL because they
    construct WinMLEP/WinMLEPDevice directly.
    """
    return EPEntry(
        ep_name=ep_name,
        dll_path=Path(f"C:/fake/{ep_name}.dll"),
        source=PyPISource(
            distribution="fake-dist",
            relative_dll="fake.dll",
            eps=(ep_name,),
        ),
    )


def make_stub_winml_ep_device(ort_device: object, ep_name: str) -> WinMLEPDevice:
    """Wrap an :class:`ort.OrtEpDevice` (real or mocked) into a :class:`WinMLEPDevice`.

    Tests use this when they need to hand a fully-resolved (source, device)
    pair to :class:`WinMLSession` without going through the discovery layer.
    """
    winml_device = wrap_ort_device(ort_device)  # type: ignore[arg-type]
    entry = _stub_ep_entry(ep_name)
    winml_ep = WinMLEP(source=entry, devices=(winml_device,))
    return WinMLEPDevice(ep=winml_ep, device=winml_device)


@pytest.fixture(autouse=True)
def _all_eps_available_by_default():
    """Pretend every catalog EP is registered, unless a test overrides.

    After docs/design/session/3_design_ep.md §6.4, ``default_ep_for_device``
    filters by ``available_eps()`` from ``ep_registry``. Pre-existing tests
    that asserted static-catalog defaults (e.g. ``default_ep_for_device("npu")
    == "QNNExecutionProvider"``) would otherwise become host-dependent on
    dev boxes where the catalog default isn't actually installed.

    Tests that want to exercise a specific registered subset override this
    by patching the same target inside the test body.
    """
    from winml.modelkit.session import EP_DEVICE_SPECS

    all_eps = frozenset(s.ep for s in EP_DEVICE_SPECS)
    with patch(
        "winml.modelkit.session.ep_registry.available_eps",
        return_value=all_eps,
    ):
        yield


# =============================================================================
# EP MARKERS - Skip tests if required EP is not available
# =============================================================================

# EP name mapping: marker name -> ORT provider name
EP_NAME_MAP = {
    "qnn": "QNNExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "dml": "DmlExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "tensorrt_rtx": "NvTensorRtRtxExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "rocm": "ROCMExecutionProvider",
}


def pytest_configure(config: pytest.Config) -> None:
    """Register EP markers."""
    config.addinivalue_line(
        "markers",
        "ep(name): mark test to run only when specific EP is available "
        "(qnn, openvino, dml, cuda, tensorrt, tensorrt_rtx, vitisai, coreml, rocm)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests if required EP is not available.

    Note: EP discovery is only performed when tests with @pytest.mark.ep() markers
    are found. This avoids slow WinML initialization for pure unit tests.
    """
    # Only consider non-e2e items with EP markers. E2e tests handle their own
    # EP discovery. This hook runs before -m filtering, so e2e items are still
    # in the list — skip them to avoid triggering WinML SDK initialization.
    items_with_ep_markers = [
        item
        for item in items
        if any(item.iter_markers(name="ep")) and not any(item.iter_markers(name="e2e"))
    ]
    if not items_with_ep_markers:
        return  # No non-e2e EP markers, skip expensive WinML discovery

    import onnxruntime as ort

    from winml.modelkit.session import WinMLEPRegistrationFailed, WinMLEPRegistry

    # Register WinML EPs so ort.get_ep_devices() includes them. Walk the cached
    # entries one-by-one — the legacy bulk register_to_ort() helper is gone.
    registry = WinMLEPRegistry.instance()
    for entry in registry._entries:
        try:
            registry.register_ep(entry)
        except WinMLEPRegistrationFailed:
            # Best-effort: an individual EP failure must not skip every
            # ep-marked test in the collection.
            continue

    # Use ort.get_ep_devices() for hardware-accurate availability (only returns
    # EPs backed by actual hardware, not just library-present registrations).
    try:
        available_providers = {d.ep_name for d in ort.get_ep_devices()}
    except Exception:
        available_providers = set()
    # Only add CPUExecutionProvider as guaranteed fallback. Avoid adding every
    # provider ORT reports as available — library-present EPs (e.g. DmlEP) can
    # be visible to ORT without the hardware actually being usable.
    available_providers.add("CPUExecutionProvider")

    for item in items_with_ep_markers:
        for marker in item.iter_markers(name="ep"):
            ep_name = marker.args[0] if marker.args else None
            if ep_name is None:
                continue

            provider_name = EP_NAME_MAP.get(ep_name.lower())
            if provider_name is None:
                item.add_marker(pytest.mark.skip(reason=f"Unknown EP marker: {ep_name}"))
            elif provider_name not in available_providers:
                item.add_marker(pytest.mark.skip(reason=f"EP not available: {provider_name}"))


def create_matmul_onnx(output_path: Path) -> Path:
    """
    Create a simple MatMul ONNX model for testing.

    Graph: A @ B = C
    Where A is input (1, 4), B is constant (4, 4), C is output (1, 4)

    This is the simplest possible ONNX model that can run on any EP.
    """
    # Input tensor info
    A = helper.make_tensor_value_info("A", TensorProto.FLOAT, [1, 4])  # noqa: N806

    # Output tensor info
    C = helper.make_tensor_value_info("C", TensorProto.FLOAT, [1, 4])  # noqa: N806

    # Constant weights (4x4 matrix)
    np.random.seed(42)  # Reproducible
    B_values = np.random.randn(4, 4).astype(np.float32)  # noqa: N806
    B_tensor = helper.make_tensor("B", TensorProto.FLOAT, [4, 4], B_values.flatten().tolist())  # noqa: N806

    # MatMul node
    matmul_node = helper.make_node("MatMul", ["A", "B"], ["C"], name="matmul")

    # Graph
    graph = helper.make_graph(
        nodes=[matmul_node],
        name="test_matmul",
        inputs=[A],
        outputs=[C],
        initializer=[B_tensor],
    )

    # Model with explicit IR version for ORT compatibility
    # Use opset 13 and IR version 7 for broad compatibility
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    # Set IR version explicitly (ORT supports up to IR version 9)
    model.ir_version = 7

    # Validate
    onnx.checker.check_model(model)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))

    return output_path


@pytest.fixture
def simple_matmul_onnx(tmp_path: Path) -> Path:
    """Create simple MatMul ONNX model for testing."""
    return create_matmul_onnx(tmp_path / "test_matmul.onnx")


def create_static_batch_onnx(output_path: Path, batch_size: int = 1) -> Path:
    """Create ONNX model with static batch size for re-batching tests.

    Graph: A @ B = C
    Where A is input (batch_size, 4), B is constant (4, 4), C is output (batch_size, 4)

    Args:
        output_path: Where to save the model
        batch_size: Static batch size (default: 1)
    """
    # Input tensor info with STATIC batch size
    a_input = helper.make_tensor_value_info("A", TensorProto.FLOAT, [batch_size, 4])

    # Output tensor info with STATIC batch size
    c_output = helper.make_tensor_value_info("C", TensorProto.FLOAT, [batch_size, 4])

    # Constant weights (4x4 matrix)
    np.random.seed(42)  # Reproducible
    b_values = np.random.randn(4, 4).astype(np.float32)
    b_tensor = helper.make_tensor("B", TensorProto.FLOAT, [4, 4], b_values.flatten().tolist())

    # MatMul node
    matmul_node = helper.make_node("MatMul", ["A", "B"], ["C"], name="matmul")

    # Graph
    graph = helper.make_graph(
        nodes=[matmul_node],
        name="test_static_batch_matmul",
        inputs=[a_input],
        outputs=[c_output],
        initializer=[b_tensor],
    )

    # Model
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    model.ir_version = 7

    # Validate
    onnx.checker.check_model(model)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))

    return output_path


@pytest.fixture
def static_batch1_onnx(tmp_path: Path) -> Path:
    """Create ONNX model with static batch=1 for re-batching tests."""
    return create_static_batch_onnx(tmp_path / "static_batch1.onnx", batch_size=1)


@pytest.fixture
def static_batch2_onnx(tmp_path: Path) -> Path:
    """Create ONNX model with static batch=2 for re-batching tests."""
    return create_static_batch_onnx(tmp_path / "static_batch2.onnx", batch_size=2)


@pytest.fixture
def sample_input() -> dict[str, np.ndarray]:
    """Create sample input for MatMul model."""
    np.random.seed(123)
    return {"A": np.random.randn(1, 4).astype(np.float32)}


# =============================================================================
# Task 7+: EPDeviceTarget fixtures (shared across Tasks 10-11 callsite sweeps)
# =============================================================================


@pytest.fixture
def qnn_npu_ep_device(fake_ort_npu: MagicMock) -> WinMLEPDevice:
    """Pre-resolved WinMLEPDevice for QNN NPU built around the fake_ort_npu handle."""
    return make_stub_winml_ep_device(fake_ort_npu, "QNNExecutionProvider")


@pytest.fixture
def qnn_npu_target() -> EPDeviceTarget:
    """Pure-intent target — useful for tests that exercise the deduction path."""
    return EPDeviceTarget(ep="QNNExecutionProvider", device="npu")


@pytest.fixture
def fake_ort_npu() -> MagicMock:
    d = MagicMock()
    d.ep_name = "QNNExecutionProvider"
    d.device.type.name = "NPU"
    d.device.vendor_id = QNN_VENDOR_ID
    d.device.device_id = 0x0001
    return d


# =============================================================================
# CPU EPDeviceTarget fixtures — use real OrtEpDevice so ORT inference actually runs
# =============================================================================

# Cached at module-scope so we only call get_ep_devices() once per test session.
_REAL_CPU_ORT_DEVICE: ort.OrtEpDevice | None = None


def _get_real_cpu_ort_device() -> ort.OrtEpDevice:
    """Return the CPUExecutionProvider OrtEpDevice from ort.get_ep_devices()."""
    global _REAL_CPU_ORT_DEVICE
    if _REAL_CPU_ORT_DEVICE is None:
        devices = ort.get_ep_devices()
        matches = [d for d in devices if d.ep_name == "CPUExecutionProvider"]
        if not matches:
            pytest.skip("CPUExecutionProvider not available in ort.get_ep_devices()")
        _REAL_CPU_ORT_DEVICE = matches[0]
    return _REAL_CPU_ORT_DEVICE


@pytest.fixture
def real_cpu_ort_device() -> ort.OrtEpDevice:
    """Real OrtEpDevice for CPUExecutionProvider (from ort.get_ep_devices())."""
    return _get_real_cpu_ort_device()


@pytest.fixture
def cpu_ep_device(real_cpu_ort_device: ort.OrtEpDevice) -> WinMLEPDevice:
    """Pre-resolved WinMLEPDevice for CPUExecutionProvider wrapping the real OrtEpDevice."""
    return make_stub_winml_ep_device(real_cpu_ort_device, "CPUExecutionProvider")


@pytest.fixture
def cpu_target() -> EPDeviceTarget:
    """Pure-intent target for CPU — used by tests exercising the resolver."""
    return EPDeviceTarget(ep="CPUExecutionProvider", device="cpu")


@pytest.fixture
def cpu_winml_session(
    simple_matmul_onnx: Path,
    cpu_ep_device: WinMLEPDevice,
) -> WinMLSession:
    """WinMLSession bound to CPU. cpu_ep_device wraps the real OrtEpDevice so
    add_provider_for_devices() receives a genuine handle and ORT can run.
    """
    return WinMLSession(simple_matmul_onnx, ep_device=cpu_ep_device)
