# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for modelkit.sysinfo.device module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.sysinfo.device import (
    _DEVICE_EP_MAP,
    _EP_DEVICE_MAP,
    _get_available_devices,
    resolve_device,
    resolve_eps,
)
from winml.modelkit.utils.constants import EP_NAMES


def _make_ep_device(device_type) -> MagicMock:
    """Build a mock OrtEpDevice whose ``.device.type`` is ``device_type``."""
    mock = MagicMock()
    mock.device.type = device_type
    return mock


class TestGetAvailableDevices:
    """Tests for _get_available_devices()."""

    def test_with_npu_and_gpu(self) -> None:
        """When NPU and GPU EP devices are registered, returns ("npu", "gpu", "cpu")."""
        import onnxruntime as ort

        with patch(
            "winml.modelkit.sysinfo.device.get_registered_ep_devices",
            return_value=[
                _make_ep_device(ort.OrtHardwareDeviceType.NPU),
                _make_ep_device(ort.OrtHardwareDeviceType.GPU),
                _make_ep_device(ort.OrtHardwareDeviceType.CPU),
            ],
        ):
            devices = _get_available_devices()

        assert devices == ("npu", "gpu", "cpu")

    def test_no_npu_with_gpu(self) -> None:
        """When no NPU but GPU registered, returns ("gpu", "cpu")."""
        import onnxruntime as ort

        with patch(
            "winml.modelkit.sysinfo.device.get_registered_ep_devices",
            return_value=[
                _make_ep_device(ort.OrtHardwareDeviceType.GPU),
                _make_ep_device(ort.OrtHardwareDeviceType.CPU),
            ],
        ):
            devices = _get_available_devices()

        assert devices == ("gpu", "cpu")

    def test_no_npu_no_gpu(self) -> None:
        """When only CPU EP devices are registered, returns ("cpu",)."""
        import onnxruntime as ort

        with patch(
            "winml.modelkit.sysinfo.device.get_registered_ep_devices",
            return_value=[_make_ep_device(ort.OrtHardwareDeviceType.CPU)],
        ):
            devices = _get_available_devices()

        assert devices == ("cpu",)

    def test_returns_empty_when_enumeration_fails(self) -> None:
        """If EP enumeration raises, return empty tuple (no devices visible).

        ``resolve_device("auto")`` is responsible for the CPU fallback when no
        devices are reachable; ``_get_available_devices`` only reports what is
        actually registered.
        """
        with patch(
            "winml.modelkit.sysinfo.device.get_registered_ep_devices",
            side_effect=RuntimeError("ORT not available"),
        ):
            devices = _get_available_devices()

        assert devices == ()

    def test_priority_order_independent_of_input(self) -> None:
        """Result is always NPU > GPU > CPU regardless of enumeration order."""
        import onnxruntime as ort

        with patch(
            "winml.modelkit.sysinfo.device.get_registered_ep_devices",
            return_value=[
                _make_ep_device(ort.OrtHardwareDeviceType.CPU),
                _make_ep_device(ort.OrtHardwareDeviceType.GPU),
                _make_ep_device(ort.OrtHardwareDeviceType.NPU),
            ],
        ):
            devices = _get_available_devices()

        assert devices == ("npu", "gpu", "cpu")

    def test_duplicate_device_types_deduplicated(self) -> None:
        """Multiple EP devices on the same device type collapse to one entry."""
        import onnxruntime as ort

        with patch(
            "winml.modelkit.sysinfo.device.get_registered_ep_devices",
            return_value=[
                _make_ep_device(ort.OrtHardwareDeviceType.GPU),
                _make_ep_device(ort.OrtHardwareDeviceType.GPU),
                _make_ep_device(ort.OrtHardwareDeviceType.CPU),
            ],
        ):
            devices = _get_available_devices()

        assert devices == ("gpu", "cpu")


class TestMappingConstants:
    """Tests for _EP_DEVICE_MAP and _DEVICE_EP_MAP constants."""

    def test_ep_device_map_has_required_entries(self) -> None:
        """_EP_DEVICE_MAP has all standard EPs."""
        # NVIDIA
        assert "NvTensorRTRTXExecutionProvider" in _EP_DEVICE_MAP
        # AMD
        assert "MIGraphXExecutionProvider" in _EP_DEVICE_MAP
        assert "VitisAIExecutionProvider" in _EP_DEVICE_MAP
        # Qualcomm
        assert "QNNExecutionProvider" in _EP_DEVICE_MAP
        # Microsoft
        assert "DmlExecutionProvider" in _EP_DEVICE_MAP
        # Intel
        assert "OpenVINOExecutionProvider" in _EP_DEVICE_MAP
        # CPU
        assert "CPUExecutionProvider" in _EP_DEVICE_MAP

    def test_ep_device_map_covers_every_ep_name_literal(self) -> None:
        """Every value in the ``EPName`` Literal must have an _EP_DEVICE_MAP entry.

        Adding a new canonical EP to the Literal without wiring its device
        mapping here would silently break device resolution; this test guards
        against that drift.
        """
        missing = set(EP_NAMES) - set(_EP_DEVICE_MAP)
        assert not missing, f"_EP_DEVICE_MAP is missing entries for: {sorted(missing)}"

    def test_ep_device_map_no_extra_keys_outside_literal(self) -> None:
        """_EP_DEVICE_MAP must not contain keys absent from the ``EPName`` Literal."""
        extra = set(_EP_DEVICE_MAP) - set(EP_NAMES)
        assert not extra, f"_EP_DEVICE_MAP has unexpected canonical names: {sorted(extra)}"

    def test_ep_device_map_values_are_lowercase(self) -> None:
        """All _EP_DEVICE_MAP values should be lowercase."""
        for ep, device in _EP_DEVICE_MAP.items():
            assert device == device.lower(), f"{ep} maps to non-lowercase '{device}'"

    def test_device_ep_map_includes_multi_device_eps(self) -> None:
        """Multi-device EPs (QNN, OpenVINO) should appear in each device."""
        assert "QNNExecutionProvider" in _DEVICE_EP_MAP["npu"]
        assert "QNNExecutionProvider" in _DEVICE_EP_MAP["gpu"]
        assert "OpenVINOExecutionProvider" in _DEVICE_EP_MAP["npu"]
        assert "OpenVINOExecutionProvider" in _DEVICE_EP_MAP["gpu"]
        assert "OpenVINOExecutionProvider" in _DEVICE_EP_MAP["cpu"]

    def test_device_ep_map_derived_from_ep_device_map(self) -> None:
        """_DEVICE_EP_MAP should be consistent with _EP_DEVICE_MAP."""
        for device, eps in _DEVICE_EP_MAP.items():
            for ep in eps:
                assert ep in _EP_DEVICE_MAP, (
                    f"EP '{ep}' in _DEVICE_EP_MAP but not in _EP_DEVICE_MAP"
                )
                assert device in _EP_DEVICE_MAP[ep].split("/")

    def test_nv_tensorrt_rtx_is_gpu_ep(self) -> None:
        """NvTensorRTRTXExecutionProvider should map to gpu."""
        assert _EP_DEVICE_MAP["NvTensorRTRTXExecutionProvider"] == "gpu"
        assert "NvTensorRTRTXExecutionProvider" in _DEVICE_EP_MAP["gpu"]


def _patch_device_ep_map(mapping: dict[str, tuple[str, ...]]):
    """Patch the central _get_device_ep_map_from_ort probe with ``mapping``.

    Single mock point for resolve_device/resolve_eps tests. Each value is the
    tuple of EPs registered for that device, in the order ORT would return.
    """
    return patch(
        "winml.modelkit.sysinfo.device._get_device_ep_map_from_ort",
        return_value=mapping,
    )


class TestResolveDevice:
    """Tests for resolve_device()."""

    def test_resolve_device_auto_npu_with_ep(self) -> None:
        """Auto mode: NPU EP registered -> returns "npu"."""
        with _patch_device_ep_map(
            {
                "npu": ("QNNExecutionProvider",),
                "gpu": ("QNNExecutionProvider", "DmlExecutionProvider"),
                "cpu": ("CPUExecutionProvider",),
            }
        ):
            device, available = resolve_device("auto")

        assert device == "npu"
        assert available == ["npu", "gpu", "cpu"]

    def test_resolve_device_auto_npu_without_ep(self) -> None:
        """Auto mode: no NPU EP registered -> falls through to GPU."""
        with _patch_device_ep_map(
            {
                "gpu": ("DmlExecutionProvider",),
                "cpu": ("CPUExecutionProvider",),
            }
        ):
            device, available = resolve_device("auto")

        assert device == "gpu"
        assert available == ["gpu", "cpu"]

    def test_resolve_device_auto_cpu_fallback(self) -> None:
        """Auto mode: only CPU EP registered -> returns "cpu"."""
        with _patch_device_ep_map({"cpu": ("CPUExecutionProvider",)}):
            device, available = resolve_device("auto")

        assert device == "cpu"
        assert available == ["cpu"]

    def test_resolve_device_explicit_valid(self) -> None:
        """Explicit device "gpu" -> returns "gpu"."""
        with _patch_device_ep_map(
            {
                "gpu": ("DmlExecutionProvider",),
                "cpu": ("CPUExecutionProvider",),
            }
        ):
            device, available = resolve_device("gpu")

        assert device == "gpu"
        assert available == ["gpu", "cpu"]

    def test_resolve_device_explicit_invalid(self) -> None:
        """Unrecognized device "tpu" -> raises ValueError."""
        with pytest.raises(ValueError, match="Unknown device 'tpu'"):
            resolve_device("tpu")

    def test_resolve_device_explicit_no_ep_error_names_missing_eps(self) -> None:
        """Error message must name the compatible EPs so users know what to install."""
        with (
            _patch_device_ep_map({"cpu": ("CPUExecutionProvider",)}),
            patch(
                "winml.modelkit.sysinfo.device._get_available_eps",
                return_value=frozenset({"CPUExecutionProvider"}),
            ),
            pytest.raises(ValueError) as exc_info,
        ):
            resolve_device("npu")

        message = str(exc_info.value)
        assert "no compatible EP" in message
        # Names at least one NPU-compatible EP so the user can act on it
        assert "QNNExecutionProvider" in message or "VitisAIExecutionProvider" in message

    def test_resolve_device_case_insensitive(self) -> None:
        """Device argument should be case-insensitive."""
        with _patch_device_ep_map({"cpu": ("CPUExecutionProvider",)}):
            device, _ = resolve_device("CPU")

        assert device == "cpu"

    def test_resolve_device_no_eps_raises(self) -> None:
        """Auto mode with no registered EPs raises RuntimeError.

        Hit when ORT/WinML isn't installed (or hasn't enumerated any device).
        Failing fast is more helpful than silently writing a config that
        targets a CPU with no compatible EP.
        """
        with (
            _patch_device_ep_map({}),
            pytest.raises(RuntimeError, match="No execution providers detected"),
        ):
            resolve_device("auto")


class TestResolveDeviceWithEp:
    """Tests for resolve_device(ep=...) — EP-aware filtering of available_devices/eps."""

    def test_ep_qnn_filters_devices_to_npu_and_gpu(self) -> None:
        """ep='qnn' narrows available_devices to QNN's compatible devices (npu/gpu)."""
        with _patch_device_ep_map(
            {
                "npu": ("QNNExecutionProvider",),
                "gpu": ("QNNExecutionProvider", "DmlExecutionProvider"),
                "cpu": ("CPUExecutionProvider",),
            }
        ):
            device, available = resolve_device("auto", ep="qnn")

        assert device == "npu"
        assert available == ["npu", "gpu"]

    def test_ep_qnn_auto_picks_gpu_when_no_npu(self) -> None:
        """ep='qnn' on a GPU-only system auto-selects gpu."""
        with _patch_device_ep_map(
            {
                "gpu": ("QNNExecutionProvider",),
                "cpu": ("CPUExecutionProvider",),
            }
        ):
            device, available = resolve_device("auto", ep="qnn")

        assert device == "gpu"
        assert available == ["gpu"]

    def test_ep_dml_filters_to_gpu_only(self) -> None:
        """ep='dml' narrows available_devices to gpu (DML is gpu-only)."""
        with _patch_device_ep_map(
            {
                "gpu": ("DmlExecutionProvider",),
                "cpu": ("CPUExecutionProvider",),
            }
        ):
            device, available = resolve_device("auto", ep="dml")

        assert device == "gpu"
        assert available == ["gpu"]

    def test_ep_requested_but_not_available_raises(self) -> None:
        """If the requested EP is known but not present on the system, raise."""
        with (
            _patch_device_ep_map(
                {
                    "npu": ("QNNExecutionProvider",),
                    "gpu": ("QNNExecutionProvider", "DmlExecutionProvider"),
                    "cpu": ("CPUExecutionProvider",),
                }
            ),
            patch(
                "winml.modelkit.sysinfo.device._get_available_eps",
                return_value=frozenset(
                    {
                        "QNNExecutionProvider",
                        "DmlExecutionProvider",
                        "CPUExecutionProvider",
                    }
                ),
            ),
            pytest.raises(
                ValueError,
                match="Requested EP 'vitisai' is not available on this system",
            ),
        ):
            resolve_device("auto", ep="vitisai")

    def test_ep_unknown_raises(self) -> None:
        """Unknown ep short name raises ValueError from resolve_device."""
        with pytest.raises(ValueError, match=r"Unknown EP 'tpu'\. Expected one of:"):
            resolve_device("auto", ep="tpu")

    def test_ep_case_insensitive(self) -> None:
        """ep argument is case-insensitive."""
        with _patch_device_ep_map(
            {
                "npu": ("QNNExecutionProvider",),
                "gpu": ("QNNExecutionProvider",),
                "cpu": ("CPUExecutionProvider",),
            }
        ):
            device, available = resolve_device("auto", ep="QNN")

        assert device == "npu"
        assert available == ["npu", "gpu"]

    def test_ep_explicit_device_filtered_out_raises(self) -> None:
        """ep='qnn' + device='cpu' raises: cpu has no compatible EP within {QNN}."""
        with (
            _patch_device_ep_map(
                {
                    "npu": ("QNNExecutionProvider",),
                    "gpu": ("QNNExecutionProvider",),
                    "cpu": ("CPUExecutionProvider",),
                }
            ),
            patch(
                "winml.modelkit.sysinfo.device._get_available_eps",
                return_value=frozenset(
                    {"QNNExecutionProvider", "CPUExecutionProvider"},
                ),
            ),
            pytest.raises(ValueError, match="no compatible EP"),
        ):
            resolve_device("cpu", ep="qnn")


class TestResolveEps:
    """Tests for resolve_eps()."""

    def test_returns_priority_list_when_multiple_eps_available(self) -> None:
        """resolve_eps preserves the _EP_DEVICE_MAP iteration order.

        For ``gpu``, the priority is NV → CUDA → MIGraphX → QNN → OpenVINO →
        DML (IHV-first, native-last). When all are advertised, all are
        returned in that order regardless of the order ORT reports them.
        """
        with _patch_device_ep_map(
            {
                "gpu": (
                    "DmlExecutionProvider",  # intentionally first to prove
                    "QNNExecutionProvider",  # output order comes from
                    "NvTensorRTRTXExecutionProvider",  # _DEVICE_EP_MAP, not
                    "CUDAExecutionProvider",  # from the input ordering.
                    "MIGraphXExecutionProvider",
                    "OpenVINOExecutionProvider",
                ),
            }
        ):
            assert resolve_eps("gpu") == [
                "NvTensorRTRTXExecutionProvider",
                "CUDAExecutionProvider",
                "MIGraphXExecutionProvider",
                "QNNExecutionProvider",
                "OpenVINOExecutionProvider",
                "DmlExecutionProvider",
            ]

    def test_gpu_with_only_dml_available(self) -> None:
        """Common Windows case: DML is the only GPU EP advertised."""
        with _patch_device_ep_map(
            {
                "gpu": ("DmlExecutionProvider",),
                "cpu": ("CPUExecutionProvider",),
            }
        ):
            assert resolve_eps("gpu") == ["DmlExecutionProvider"]

    def test_npu_filters_to_installed_eps(self) -> None:
        """Only EPs that ORT/WinML actually advertises are returned."""
        with _patch_device_ep_map(
            {
                "npu": ("QNNExecutionProvider",),
                "cpu": ("CPUExecutionProvider",),
            }
        ):
            # _DEVICE_EP_MAP["npu"] includes Vitis and OpenVINO too, but only
            # QNN is in the available set.
            assert resolve_eps("npu") == ["QNNExecutionProvider"]

    def test_cpu_includes_multi_device_eps(self) -> None:
        """OpenVINO declares ``npu/gpu/cpu`` so it shows up for cpu too."""
        with _patch_device_ep_map(
            {
                "cpu": ("OpenVINOExecutionProvider", "CPUExecutionProvider"),
            }
        ):
            assert resolve_eps("cpu") == [
                "OpenVINOExecutionProvider",
                "CPUExecutionProvider",
            ]

    def test_case_insensitive(self) -> None:
        """Upper-case input is normalized before lookup."""
        with _patch_device_ep_map({"npu": ("QNNExecutionProvider",)}):
            assert resolve_eps("NPU") == ["QNNExecutionProvider"]
            assert resolve_eps("Npu") == ["QNNExecutionProvider"]

    def test_unknown_device_returns_empty(self) -> None:
        """Unknown device name returns ``[]``, not a KeyError."""
        with _patch_device_ep_map(
            {
                "npu": ("QNNExecutionProvider",),
                "cpu": ("CPUExecutionProvider",),
            }
        ):
            assert resolve_eps("tpu") == []
            assert resolve_eps("") == []

    def test_no_eps_available_returns_empty(self) -> None:
        """When ORT/WinML advertises nothing, every device resolves to []."""
        with _patch_device_ep_map({}):
            assert resolve_eps("npu") == []
            assert resolve_eps("gpu") == []
            assert resolve_eps("cpu") == []
