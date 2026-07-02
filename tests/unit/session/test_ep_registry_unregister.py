# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests the register -> to_dict -> unregister cadence.

See ``isolated_ep_register`` for the loader-collision rationale.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.ep_path import BuiltinSource, EPEntry, PyPISource
from winml.modelkit.session import WinMLEP
from winml.modelkit.session.ep_registry import WinMLEPRegistry

from .conftest import QNN_VENDOR_ID


def _plugin_entry(ep_name: str = "OpenVINOExecutionProvider") -> EPEntry:
    """Plugin-source EPEntry — exercises the DLL-load unregister branch."""
    return EPEntry(
        ep_name=ep_name,
        dll_path=Path(f"C:/fake/{ep_name}.dll"),
        source=PyPISource(
            distribution="fake-dist",
            relative_dll="fake.dll",
            eps=(ep_name,),
        ),
    )


def _builtin_entry(ep_name: str = "CPUExecutionProvider") -> EPEntry:
    """BuiltinSource EPEntry — exercises the skip-unregister branch."""
    return EPEntry(
        ep_name=ep_name,
        dll_path=Path(),
        source=BuiltinSource(eps=(ep_name,)),
    )


def _fake_ort_device(
    ep_name: str,
    dev_type: str,
    dll_path: str,
    version: str,
) -> MagicMock:
    """Mimic the ``OrtEpDevice`` shape used by ``WinMLDevice`` + ``to_dict``."""
    d = MagicMock()
    d.ep_name = ep_name
    d.ep_metadata = {
        "library_path": dll_path,
        "version": version,
    }
    d.device.type.name = dev_type
    d.device.vendor_id = QNN_VENDOR_ID
    d.device.device_id = 0x0001
    d.device.vendor = "FakeVendor"
    d.device.metadata = {}
    d.ep_vendor = "Microsoft"
    return d


@pytest.mark.parametrize(
    "kind,expects_ort_unregister",
    [
        ("plugin", True),  # DLL-backed source -> hand arg0 to ORT
        ("builtin", False),  # in-process wrap -> skip guard
    ],
)
def test_to_dict_then_unregister_ep(
    fresh_registry: WinMLEPRegistry,
    kind: str,
    expects_ort_unregister: bool,
) -> None:
    """``register_ep`` -> ``to_dict`` -> ``unregister_ep`` cadence."""
    if kind == "plugin":
        entry = _plugin_entry()
    else:
        entry = _builtin_entry()

    fake_dev = _fake_ort_device(
        entry.ep_name,
        "NPU" if kind == "plugin" else "CPU",
        dll_path=str(entry.dll_path),
        version="9.9.9+deadbeef",
    )

    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.get_ep_devices.return_value = [fake_dev]
        mock_ort.register_execution_provider_library = MagicMock()
        mock_ort.unregister_execution_provider_library = MagicMock()

        winml_ep = fresh_registry.register_ep(entry)
        snapshot = winml_ep.to_dict()
        fresh_registry.unregister_ep(winml_ep)

    # 1. to_dict shape — the sys.py renderer reads exactly these keys.
    assert isinstance(snapshot, dict)
    assert snapshot["plugin_version"] == "9.9.9+deadbeef"
    assert isinstance(snapshot["devices"], list)
    assert len(snapshot["devices"]) == 1
    dev = snapshot["devices"][0]
    assert set(dev.keys()) == {
        "device_type",
        "hardware_name",
        "vendor",
        "facts",
        "device_facts",
    }

    # 2/3. unregister_ep hits ORT only for DLL-backed sources.
    if expects_ort_unregister:
        mock_ort.unregister_execution_provider_library.assert_called_once_with(
            winml_ep.arg0,
        )
        # 4. Cache eviction — dll_path key must be gone.
        assert entry.dll_path not in fresh_registry._registered
    else:
        mock_ort.unregister_execution_provider_library.assert_not_called()
