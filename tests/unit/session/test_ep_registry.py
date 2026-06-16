# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ep_registry module-level helpers and WinMLEPRegistry.register_ep.

Post-Batch-C: register_ep takes an EPEntry and returns a WinMLEP. Most of
the old tests that verified the (name -> list[OrtEpDevice]) shape now exercise
the new atomic-registration semantics — DLL load happens at the entry level,
the returned WinMLEP wraps every device the EP exposed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.ep_path import EPEntry, PyPISource
from winml.modelkit.session import WinMLEP, WinMLEPRegistrationFailed
from winml.modelkit.session.ep_registry import WinMLEPRegistry

from .conftest import QNN_VENDOR_ID


def _ep_entry(ep_name: str, dll: str = "C:/fake/qnn.dll") -> EPEntry:
    """Build a minimal EPEntry suitable for register_ep tests."""
    return EPEntry(
        ep_name=ep_name,
        dll_path=Path(dll),
        source=PyPISource(
            distribution="fake-dist",
            relative_dll="fake.dll",
            eps=(ep_name,),
        ),
    )


@pytest.fixture
def fresh_registry() -> WinMLEPRegistry:
    """Singleton with stubbed cache + cleared registration caches."""
    reg = WinMLEPRegistry.instance()
    reg._entries = [_ep_entry("QNNExecutionProvider")]
    reg._registered = {}
    return reg


def _fake_ort_device(ep_name: str, dev_type: str, dll_path: str = "C:/fake/qnn.dll") -> MagicMock:
    """Build a MagicMock matching the OrtEpDevice shape used downstream.

    ``register_ep`` filters enumerated handles by
    ``ep_metadata['library_path']`` (so multiple registrations of the same
    canonical ep_name produce distinct device sets), so every mocked
    device must carry the ``library_path`` of the entry it should bind to.
    """
    d = MagicMock()
    d.ep_name = ep_name
    d.ep_metadata = {"library_path": str(Path(dll_path))}
    d.device.type.name = dev_type
    d.device.vendor_id = QNN_VENDOR_ID
    d.device.device_id = 0x0001
    return d


def test_register_ep_happy_path(fresh_registry: WinMLEPRegistry) -> None:
    """register_ep(entry) loads the DLL, wraps every matching device, returns WinMLEP."""
    entry = _ep_entry("QNNExecutionProvider")
    qnn_devs = [
        _fake_ort_device("QNNExecutionProvider", "NPU"),
    ]
    other = _fake_ort_device("CPUExecutionProvider", "CPU", dll_path="C:/fake/other.dll")
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        # Single get_ep_devices call after the register_execution_provider_library
        # call — no defensive pre-check now that we always register.
        mock_ort.get_ep_devices.return_value = [*qnn_devs, other]
        mock_ort.register_execution_provider_library = MagicMock()
        result = fresh_registry.register_ep(entry)

    mock_ort.register_execution_provider_library.assert_called_once()
    args, _ = mock_ort.register_execution_provider_library.call_args
    # First registration of this ep_name uses the canonical arg0 (no suffix).
    assert args[0] == "QNNExecutionProvider"
    # Path is rendered via str(Path(...)) which uses OS-native separators.
    assert Path(args[1]) == Path("C:/fake/qnn.dll")
    assert isinstance(result, WinMLEP)
    assert result.source is entry
    # Filtering by library_path: only QNN devices whose library_path
    # matches the entry's dll_path land in result.devices.
    assert len(result.devices) == 1
    assert result.devices[0].device_type == "NPU"


def test_register_ep_rejects_double_dll_path(fresh_registry: WinMLEPRegistry) -> None:
    """A second register_ep for the same dll_path raises WinMLEPRegistrationFailed.

    The previous idempotent-cache semantic is gone: callers must dedup
    EPEntries upstream (Batch G's discover_all_eps dedup is the canonical
    guard). A double-register reaching register_ep signals a caller bug.
    """
    entry = _ep_entry("QNNExecutionProvider")
    qnn = _fake_ort_device("QNNExecutionProvider", "NPU")
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.get_ep_devices.return_value = [qnn]
        mock_ort.register_execution_provider_library = MagicMock()
        first = fresh_registry.register_ep(entry)
        with pytest.raises(WinMLEPRegistrationFailed, match="already registered"):
            fresh_registry.register_ep(entry)
    assert isinstance(first, WinMLEP)
    assert mock_ort.register_execution_provider_library.call_count == 1


def test_register_ep_suffix_for_repeat_ep_name(fresh_registry: WinMLEPRegistry) -> None:
    """Second registration for same ep_name (different DLL) gets ``_<n>`` arg0 suffix.

    Empirically (temp/probe_double_register.py): ORT accepts the same DLL or
    different DLLs under different arg0s; the device's self-reported
    ep_name stays canonical. Suffix keeps ORT's registration-tracking key
    unique without changing device-binding semantics.
    """
    entry_a = _ep_entry("OpenVINOExecutionProvider", dll="C:/fake/ov_pypi.dll")
    entry_b = _ep_entry("OpenVINOExecutionProvider", dll="C:/fake/ov_msix.dll")
    dev_a = _fake_ort_device("OpenVINOExecutionProvider", "NPU", dll_path="C:/fake/ov_pypi.dll")
    dev_b = _fake_ort_device("OpenVINOExecutionProvider", "GPU", dll_path="C:/fake/ov_msix.dll")
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.get_ep_devices.return_value = [dev_a, dev_b]
        mock_ort.register_execution_provider_library = MagicMock()
        first = fresh_registry.register_ep(entry_a)
        second = fresh_registry.register_ep(entry_b)

    assert mock_ort.register_execution_provider_library.call_count == 2
    arg0_first = mock_ort.register_execution_provider_library.call_args_list[0][0][0]
    arg0_second = mock_ort.register_execution_provider_library.call_args_list[1][0][0]
    assert arg0_first == "OpenVINOExecutionProvider"
    assert arg0_second == "OpenVINOExecutionProvider_1"
    # Each WinMLEP filtered by its own library_path → distinct device sets.
    assert len(first.devices) == 1
    assert first.devices[0].device_type == "NPU"
    assert len(second.devices) == 1
    assert second.devices[0].device_type == "GPU"


def test_register_ep_failure_wraps(fresh_registry: WinMLEPRegistry) -> None:
    """register_ep raises WinMLEPRegistrationFailed when ORT's register call raises."""
    entry = _ep_entry("QNNExecutionProvider")
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.register_execution_provider_library.side_effect = RuntimeError("dll boom")
        mock_ort.get_ep_devices.return_value = []
        with pytest.raises(WinMLEPRegistrationFailed):
            fresh_registry.register_ep(entry)


def test_register_ep_yields_zero_devices_raises(fresh_registry: WinMLEPRegistry) -> None:
    """register_ep raises when ORT registers the DLL but yields zero matching devices.

    Defends against silent partial-failure where the plugin loads but no
    OrtEpDevice records with a matching ``library_path`` appear (e.g.
    driver mismatch, or the DLL silently registered under a different path).
    """
    entry = _ep_entry("QNNExecutionProvider")
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        # Empty post-registration enumeration — guaranteed zero matches.
        mock_ort.get_ep_devices.return_value = []
        mock_ort.register_execution_provider_library = MagicMock()
        with pytest.raises(WinMLEPRegistrationFailed, match="no\\s+OrtEpDevices"):
            fresh_registry.register_ep(entry)


def test_register_ep_appends_to_entries_when_not_present(
    fresh_registry: WinMLEPRegistry,
) -> None:
    """register_ep appends an EPEntry to _entries when not already cached.

    Pins the Path A / Path B inconsistency fix: --list-ep can pass entries
    the registry's default discovery never saw, so a later auto_device
    call must still find them via _entries_for.
    """
    # Start with a registry whose _entries is empty so we can verify the
    # append unambiguously.
    fresh_registry._entries = []
    entry = _ep_entry("OpenVINOExecutionProvider", dll="C:/fake/openvino.dll")
    fake_dev = _fake_ort_device(
        "OpenVINOExecutionProvider", "NPU", dll_path="C:/fake/openvino.dll",
    )
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.get_ep_devices.return_value = [fake_dev]
        mock_ort.register_execution_provider_library = MagicMock()
        fresh_registry.register_ep(entry)

    assert entry in fresh_registry._entries
    assert fresh_registry._entries_for("OpenVINOExecutionProvider") == [entry]


def test_register_ep_does_not_double_append(
    fresh_registry: WinMLEPRegistry,
) -> None:
    """register_ep does not append when the EPEntry is already in _entries.

    EPEntry is a frozen dataclass with structural equality, so a freshly-
    constructed entry that matches one already in ``_entries`` (e.g.
    re-built from the same EPSource) is recognised as the same value and
    not appended twice.
    """
    entry = _ep_entry("QNNExecutionProvider")
    # fresh_registry fixture already places this entry in _entries.
    assert entry in fresh_registry._entries
    initial_len = len(fresh_registry._entries)

    fake_dev = _fake_ort_device("QNNExecutionProvider", "NPU")
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.get_ep_devices.return_value = [fake_dev]
        mock_ort.register_execution_provider_library = MagicMock()
        fresh_registry.register_ep(entry)

    assert len(fresh_registry._entries) == initial_len


def test_builtin_eps_public_method_returns_frozenset(
    fresh_registry: WinMLEPRegistry,
) -> None:
    """builtin_eps() exposes the frozenset snapshotted at __init__.

    Pins the new public surface (Finding #3 fix): callers must query
    this method instead of reaching into the private _builtin_eps attr.
    """
    sentinel = frozenset({"CPUExecutionProvider", "DmlExecutionProvider"})
    fresh_registry._builtin_eps = sentinel
    result = fresh_registry.builtin_eps()
    assert isinstance(result, frozenset)
    assert result == sentinel
