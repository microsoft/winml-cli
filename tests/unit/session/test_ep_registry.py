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
    """Singleton with stubbed catalog + cleared registration caches."""
    reg = WinMLEPRegistry.instance()
    reg._ep_paths = {"QNNExecutionProvider": "C:/fake/qnn.dll"}
    reg._registered_eps = []
    reg._registered = {}
    return reg


def _fake_ort_device(ep_name: str, dev_type: str) -> MagicMock:
    """Build a MagicMock matching the OrtEpDevice shape used downstream."""
    d = MagicMock()
    d.ep_name = ep_name
    d.device.type.name = dev_type
    d.device.vendor_id = QNN_VENDOR_ID
    d.device.device_id = 0x0001
    return d


def test_register_to_ort_failure_records_per_ep_state():
    """NFR-2: per-EP registration failures must be tracked in registration_failures."""
    registry = WinMLEPRegistry.instance()
    registry._ep_paths = {"FakeEP": "C:/nonexistent/fake.dll"}
    registry._registered_eps = []
    registry._registration_failures = {}
    registry._winml_available = True

    fake_ort = type("M", (), {})()

    def _bad_register(name, path):
        raise RuntimeError(f"cannot load {path}")

    fake_ort.register_execution_provider_library = _bad_register

    with patch.dict("sys.modules", {"onnxruntime": fake_ort}):
        registry.register_to_ort()

    assert "FakeEP" in registry.registration_failures
    assert "RuntimeError" in registry.registration_failures["FakeEP"]
    # Property returns a copy — mutating it must not corrupt internal state.
    snap = registry.registration_failures
    snap.clear()
    assert "FakeEP" in registry.registration_failures


def test_register_ep_happy_path(fresh_registry: WinMLEPRegistry) -> None:
    """register_ep(entry) loads the DLL, wraps every matching device, returns WinMLEP."""
    entry = _ep_entry("QNNExecutionProvider")
    qnn_devs = [
        _fake_ort_device("QNNExecutionProvider", "NPU"),
    ]
    other = _fake_ort_device("CPUExecutionProvider", "CPU")
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        # First call: defensive pre-check (empty → not yet loaded → proceed with DLL load).
        # Second call: enumerate devices after registration.
        mock_ort.get_ep_devices.side_effect = [[], [*qnn_devs, other]]
        mock_ort.register_execution_provider_library = MagicMock()
        result = fresh_registry.register_ep(entry)

    mock_ort.register_execution_provider_library.assert_called_once()
    args, _ = mock_ort.register_execution_provider_library.call_args
    assert args[0] == "QNNExecutionProvider"
    # Path is rendered via str(Path(...)) which uses OS-native separators.
    assert Path(args[1]) == Path("C:/fake/qnn.dll")
    assert isinstance(result, WinMLEP)
    assert result.source is entry
    # Only the matching ep_name's devices land in result.devices.
    assert len(result.devices) == 1
    assert result.devices[0].device_type == "NPU"


def test_register_ep_idempotent_on_dll_path(fresh_registry: WinMLEPRegistry) -> None:
    """A repeated register_ep on the same dll_path returns the cached WinMLEP.

    Post-Batch-C, the cache key is ``entry.dll_path`` rather than ep_name.
    """
    entry = _ep_entry("QNNExecutionProvider")
    qnn = _fake_ort_device("QNNExecutionProvider", "NPU")
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.get_ep_devices.side_effect = [[], [qnn]]
        mock_ort.register_execution_provider_library = MagicMock()
        first = fresh_registry.register_ep(entry)
        second = fresh_registry.register_ep(entry)
    assert first is second
    assert mock_ort.register_execution_provider_library.call_count == 1


def test_register_ep_failure_wraps(fresh_registry: WinMLEPRegistry) -> None:
    """register_ep raises WinMLEPRegistrationFailed when ORT's register call raises."""
    entry = _ep_entry("QNNExecutionProvider")
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.register_execution_provider_library.side_effect = RuntimeError("dll boom")
        mock_ort.get_ep_devices.return_value = []
        with pytest.raises(WinMLEPRegistrationFailed):
            fresh_registry.register_ep(entry)


def test_register_ep_skips_if_already_loaded(fresh_registry: WinMLEPRegistry) -> None:
    """register_ep skips DLL load if ORT already sees the EP (e.g. loaded by winml.py).

    Defensive check: ort.get_ep_devices() is consulted first; if the EP is
    already visible, register_execution_provider_library must NOT be called.
    """
    entry = _ep_entry("QNNExecutionProvider")
    fake_dev = _fake_ort_device("QNNExecutionProvider", "NPU")
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.get_ep_devices.return_value = [fake_dev]  # already loaded externally
        mock_ort.register_execution_provider_library = MagicMock()
        result = fresh_registry.register_ep(entry)

    # Critical: DLL register must NOT be called.
    mock_ort.register_execution_provider_library.assert_not_called()
    assert isinstance(result, WinMLEP)
    assert "QNNExecutionProvider" in fresh_registry._registered_eps


def test_register_ep_yields_zero_devices_raises(fresh_registry: WinMLEPRegistry) -> None:
    """register_ep raises when ORT registers the DLL but yields zero devices.

    Defends against silent partial-failure where the plugin loads but no
    OrtEpDevice records appear (e.g. driver mismatch).
    """
    entry = _ep_entry("QNNExecutionProvider")
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        # Empty before AND after the DLL load — guaranteed zero devices.
        mock_ort.get_ep_devices.return_value = []
        mock_ort.register_execution_provider_library = MagicMock()
        with pytest.raises(WinMLEPRegistrationFailed, match="no OrtEpDevices"):
            fresh_registry.register_ep(entry)
