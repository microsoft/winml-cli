# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared fixtures for compiler tests.

Mocks ``resolve_device`` and ``resolve_eps`` at their ``commands/compile.py``
import site so the compile CLI's auto-EP resolution is deterministic across
hosts (e.g., OpenVINO would otherwise out-rank QNN/DML/CPU on a dev box).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.ep_path import BuiltinSource, EPEntry
from winml.modelkit.session import EPDeviceTarget, WinMLDevice, WinMLEP, WinMLEPDevice


_DEVICE_TO_EPS = {
    "npu": ["QNNExecutionProvider"],
    "gpu": ["DmlExecutionProvider"],
    "cpu": ["CPUExecutionProvider"],
}


_EP_SUPPORTED_DEVICES = {
    "qnn": {"npu", "gpu"},
    "dml": {"gpu"},
    "openvino": {"npu", "gpu", "cpu"},
    "vitisai": {"npu"},
    "migraphx": {"gpu"},
    "nv_tensorrt_rtx": {"gpu"},
    "tensorrt": {"gpu"},
    "cuda": {"gpu"},
    "cpu": {"cpu"},
    "qnnexecutionprovider": {"npu", "gpu"},
    "dmlexecutionprovider": {"gpu"},
    "openvinoexecutionprovider": {"npu", "gpu", "cpu"},
    "vitisaiexecutionprovider": {"npu"},
    "migraphxexecutionprovider": {"gpu"},
    "cpuexecutionprovider": {"cpu"},
}


def _fake_resolve_device(target: EPDeviceTarget) -> EPDeviceTarget:
    """Stub for HEAD's resolve_device(EPDeviceTarget) -> EPDeviceTarget.

    ``device='auto'`` resolves to ``npu``; ``ep='auto'`` resolves to the
    device's preferred EP per ``_DEVICE_TO_EPS``. Incompatible (device, ep)
    pairs raise ValueError matching the real resolver's policy check.
    """
    device = "npu" if target.device == "auto" else target.device.lower()
    ep = target.ep
    if ep in (None, "auto"):
        ep = _DEVICE_TO_EPS.get(device, ["CPUExecutionProvider"])[0]
    supported = _EP_SUPPORTED_DEVICES.get(ep.lower())
    if supported is not None and device not in supported:
        raise ValueError(
            f"EP {ep!r} does not support device {device!r}; supported: {sorted(supported)}"
        )
    return EPDeviceTarget(ep=ep, device=device, source=target.source)


_SHORT_TO_FULL_EP = {
    "qnn": "QNNExecutionProvider",
    "dml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "vitisai": "VitisAIExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "nvtensorrtrtx": "NvTensorRTRTXExecutionProvider",
    "nv_tensorrt_rtx": "NvTensorRTRTXExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "cpu": "CPUExecutionProvider",
}


def _fake_ep_device(target: EPDeviceTarget) -> WinMLEPDevice:
    """Build a minimal WinMLEPDevice matching an EPDeviceTarget.

    WinMLDevice wraps an ``ort.OrtEpDevice`` — for tests we fabricate a
    lightweight MagicMock with the properties the CLI actually reads.
    """
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    # Honor an explicit target.ep — expand short aliases to full form so
    # WinMLDevice.ep_name is canonical. Only fall back to device→EPs
    # mapping when target.ep is None or "auto".
    if target.ep and target.ep not in (None, "auto"):
        ep_name = _SHORT_TO_FULL_EP.get(target.ep.lower(), target.ep)
    else:
        ep_name = _DEVICE_TO_EPS.get(target.device, ["CPUExecutionProvider"])[0]

    device_type_str = target.device.upper() if target.device != "auto" else "NPU"
    fake_ort_dev = MagicMock()
    fake_ort_dev.ep_name = ep_name
    fake_ort_dev.device = SimpleNamespace(
        type=SimpleNamespace(name=device_type_str),
        metadata={"Description": f"fake-{device_type_str.lower()}"},
    )
    fake_ort_dev.ep_metadata = {"FULL_DEVICE_NAME": f"fake {device_type_str}"}
    fake_ort_dev.ep_vendor = "FakeVendor"
    device = WinMLDevice(fake_ort_dev)
    ep = WinMLEP(
        source=EPEntry(
            ep_name=device.ep_name,
            dll_path=Path(),
            source=BuiltinSource(eps=(device.ep_name,)),
        ),
        devices=(device,),
        arg0=device.ep_name,
    )
    return WinMLEPDevice(ep=ep, device=device)


@pytest.fixture(autouse=True)
def mock_compile_resolution():
    """Mock device + EP resolution for tests under ``tests/unit/compiler/``.

    ``WinMLEPRegistry.is_ep_available`` is also stubbed so the compile CLI's
    host-availability check passes for every EP — tests that exercise the
    negative path patch the registry singleton locally with a tighter mock.
    """
    mock_registry = MagicMock()
    mock_registry.is_ep_available.return_value = True
    mock_registry.auto_device.side_effect = _fake_ep_device

    # Patch resolve_device at every command that imports it — the
    # `winml.modelkit.commands.<cmd>.resolve_device` name binds at import
    # time, so a single patch on the session facade isn't enough.
    _resolve_targets = (
        "winml.modelkit.commands.compile.resolve_device",
        "winml.modelkit.commands.perf.resolve_device",
    )
    _patches = [
        patch(target, side_effect=_fake_resolve_device, create=True) for target in _resolve_targets
    ]
    _patches.append(
        patch(
            "winml.modelkit.commands.compile.available_eps_for_device",
            side_effect=lambda device: list(_DEVICE_TO_EPS.get(device, [])),
        )
    )
    _patches.append(
        patch(
            "winml.modelkit.session.ep_registry.WinMLEPRegistry.get_instance",
            return_value=mock_registry,
        )
    )
    _patches.append(
        patch(
            "winml.modelkit.session.ep_registry.WinMLEPRegistry.instance",
            return_value=mock_registry,
        )
    )

    for p in _patches:
        p.start()
    try:
        yield
    finally:
        for p in reversed(_patches):
            p.stop()
