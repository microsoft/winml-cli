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
    resolve_check_device_ep,
    resolve_device,
    resolve_eps,
)
from winml.modelkit.utils.constants import EP_NAMES


def _make_ep_device(device_type, ep_name: str = "TestEP") -> MagicMock:
    """Build a mock OrtEpDevice with ``.device.type`` and ``.ep_name`` set.

    Both attributes matter to ``_get_device_ep_map_from_ort``: ``device.type``
    keys the result map, and ``ep_name`` populates each value tuple. Tests
    that don't care about the EP name can rely on the default.
    """
    mock = MagicMock()
    mock.device.type = device_type
    mock.ep_name = ep_name
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

        ``resolve_device("auto", ep=None)`` is responsible for the CPU fallback when no
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
            device, available = resolve_device("auto", ep=None)

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
            device, available = resolve_device("auto", ep=None)

        assert device == "gpu"
        assert available == ["gpu", "cpu"]

    def test_resolve_device_auto_cpu_fallback(self) -> None:
        """Auto mode: only CPU EP registered -> returns "cpu"."""
        with _patch_device_ep_map({"cpu": ("CPUExecutionProvider",)}):
            device, available = resolve_device("auto", ep=None)

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
            device, available = resolve_device("gpu", ep=None)

        assert device == "gpu"
        assert available == ["gpu", "cpu"]

    def test_resolve_device_explicit_invalid(self) -> None:
        """Unrecognized device "tpu" -> raises ValueError."""
        with pytest.raises(ValueError, match="Unknown device 'tpu'"):
            resolve_device("tpu", ep=None)

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
            resolve_device("npu", ep=None)

        message = str(exc_info.value)
        assert "no compatible EP" in message
        # Names at least one NPU-compatible EP so the user can act on it
        assert "QNNExecutionProvider" in message or "VitisAIExecutionProvider" in message

    def test_resolve_device_case_insensitive(self) -> None:
        """Device argument should be case-insensitive."""
        with _patch_device_ep_map({"cpu": ("CPUExecutionProvider",)}):
            device, _ = resolve_device("CPU", ep=None)

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
            resolve_device("auto", ep=None)


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
        """ep='qnn' + device='cpu' raises the policy error before availability is checked."""
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
            pytest.raises(ValueError, match="does not support device 'cpu'"),
        ):
            resolve_device("cpu", ep="qnn")

    @pytest.mark.parametrize(
        ("device", "ep"),
        [
            ("cpu", "qnn"),
            ("cpu", "dml"),
            ("cpu", "vitisai"),
            ("cpu", "migraphx"),
            ("npu", "cpu"),
            ("npu", "dml"),
            ("npu", "migraphx"),
            ("gpu", "cpu"),
            ("gpu", "vitisai"),
        ],
    )
    def test_explicit_device_ep_policy_mismatch_raises(self, device: str, ep: str) -> None:
        """Policy check rejects (device, ep) combos that ``EP_SUPPORTED_DEVICES`` forbids.

        Independent of host EP availability — raises the policy error before
        consulting the runtime device-EP map.
        """
        with (
            _patch_device_ep_map(
                {
                    "npu": ("QNNExecutionProvider",),
                    "gpu": ("QNNExecutionProvider", "DmlExecutionProvider"),
                    "cpu": ("CPUExecutionProvider",),
                }
            ),
            pytest.raises(ValueError, match=f"EP '{ep}' does not support device '{device}'"),
        ):
            resolve_device(device, ep=ep)


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


class TestResolveCheckDeviceEp:
    """Tests for resolve_check_device_ep().

    The function has two distinct code paths:

    - **Path A** — ``device == "auto"`` OR ``ep is None``. Resolves the
      concrete device via :func:`resolve_device` (system-aware: raises if the
      device/EP is not present) and the EP list via :func:`resolve_eps`. The
      returned ``available_devices`` is the *static* device set for the first
      available EP (``EP_SUPPORTED_DEVICES[available_eps[0]]``), not the
      runtime-available list — keeps the contract symmetric with Path B.
    - **Path B** — explicit device AND explicit ep. Validates only against
      ``EP_SUPPORTED_DEVICES``. Does **not** consult ORT, so it succeeds on
      hosts with no EPs installed — for callers that just want to validate a
      (device, ep) pair without running it.
    """

    def test_auto_no_ep_delegates_to_system(self) -> None:
        """device='auto', ep=None -> Path A: device + EPs come from system probe."""
        with _patch_device_ep_map(
            {
                "npu": ("QNNExecutionProvider",),
                "gpu": ("DmlExecutionProvider",),
                "cpu": ("CPUExecutionProvider",),
            }
        ):
            device, available_devices, available_eps = resolve_check_device_ep(
                device="auto", ep=None
            )

        assert device == "npu"
        # available_devices is EP_SUPPORTED_DEVICES[available_eps[0]] (static),
        # not the runtime device list. QNN supports npu/gpu, so cpu is absent
        # even though the mocked system has it.
        assert available_devices == ["npu", "gpu"]
        # When ep=None, available_eps comes from resolve_eps -- the full list
        # of EPs that target the resolved device, not a single explicit ep.
        assert available_eps == ["QNNExecutionProvider"]

    def test_auto_with_ep_returns_single_ep(self) -> None:
        """device='auto', ep='qnn' -> Path A: available_eps narrows to [ep_name]."""
        with _patch_device_ep_map(
            {
                "npu": ("QNNExecutionProvider",),
                "gpu": ("QNNExecutionProvider", "DmlExecutionProvider"),
                "cpu": ("CPUExecutionProvider",),
            }
        ):
            device, available_devices, available_eps = resolve_check_device_ep(
                device="auto", ep="qnn"
            )

        assert device == "npu"
        assert available_devices == ["npu", "gpu"]
        # Even though gpu also advertises DML, the EP filter pins this to qnn.
        assert available_eps == ["QNNExecutionProvider"]

    def test_explicit_device_no_ep_delegates(self) -> None:
        """device='npu', ep=None -> Path A (ep_name is None): goes through resolve_device."""
        with _patch_device_ep_map(
            {
                "npu": ("QNNExecutionProvider",),
                "cpu": ("CPUExecutionProvider",),
            }
        ):
            device, available_devices, available_eps = resolve_check_device_ep(
                device="npu", ep=None
            )

        assert device == "npu"
        # available_devices reflects the static EP_SUPPORTED_DEVICES for QNN
        # (npu, gpu) rather than what the mocked system advertises.
        assert available_devices == ["npu", "gpu"]
        assert available_eps == ["QNNExecutionProvider"]

    def test_explicit_device_and_ep_uses_static_mapping(self) -> None:
        """device='npu', ep='qnn' -> Path B: returns from static EP_SUPPORTED_DEVICES.

        The available_devices is the EP's supported set ('npu', 'gpu' for QNN),
        not what the system actually exposes.
        """
        with _patch_device_ep_map(
            {
                "npu": ("QNNExecutionProvider",),
                "cpu": ("CPUExecutionProvider",),
            }
        ):
            device, available_devices, available_eps = resolve_check_device_ep(
                device="npu", ep="qnn"
            )

        assert device == "npu"
        assert sorted(available_devices) == ["gpu", "npu"]
        assert available_eps == ["QNNExecutionProvider"]

    def test_path_b_does_not_require_system_eps(self) -> None:
        """Path B succeeds when the system has no EPs registered at all.

        This is the key contract for callers that only need to *validate* a
        (device, ep) pair without running it.
        """
        with _patch_device_ep_map({}):
            device, available_devices, available_eps = resolve_check_device_ep(
                device="npu", ep="qnn"
            )

        assert device == "npu"
        assert "npu" in available_devices
        assert available_eps == ["QNNExecutionProvider"]

    def test_explicit_device_unsupported_by_ep_raises(self) -> None:
        """device='cpu' + ep='qnn' -> ValueError: QNN does not support CPU."""
        with (
            _patch_device_ep_map({}),
            pytest.raises(ValueError, match="does not support device 'cpu'"),
        ):
            resolve_check_device_ep(device="cpu", ep="qnn")

    def test_explicit_unknown_ep_raises(self) -> None:
        """device='npu' + ep='tpu' -> ValueError: 'Unknown EP'."""
        with (
            _patch_device_ep_map({}),
            pytest.raises(ValueError, match="Unknown EP 'tpu'"),
        ):
            resolve_check_device_ep(device="npu", ep="tpu")

    def test_auto_unknown_ep_raises_from_delegate(self) -> None:
        """device='auto' + ep='tpu' -> Path A delegates to resolve_device, which raises.

        Confirms the error message is consistent across paths so users get the
        same diagnostic regardless of whether they passed an explicit device.
        """
        with (
            _patch_device_ep_map(
                {
                    "npu": ("QNNExecutionProvider",),
                    "cpu": ("CPUExecutionProvider",),
                }
            ),
            pytest.raises(ValueError, match="Unknown EP 'tpu'"),
        ):
            resolve_check_device_ep(device="auto", ep="tpu")

    def test_case_insensitive(self) -> None:
        """Device and EP arguments are case-insensitive."""
        with _patch_device_ep_map({}):
            device, _available_devices, available_eps = resolve_check_device_ep(
                device="NPU", ep="QNN"
            )

        assert device == "npu"
        assert available_eps == ["QNNExecutionProvider"]
