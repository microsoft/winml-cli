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


# Base fresh_registry_with_qnn fixture (empty discovery + cleanup) lives in
# tests/unit/session/conftest.py — shared with test_auto_device.py per
# the tests/CLAUDE.md DRY rule. Tests in this file that exercise the
# register_ep / suffix paths request `fresh_registry_with_qnn` instead,
# which pre-populates _discovered with a single QNNExecutionProvider
# entry layered on top of the shared base.


@pytest.fixture
def fresh_registry_with_qnn(fresh_registry):
    fresh_registry._discovered = [_ep_entry("QNNExecutionProvider")]
    yield fresh_registry


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


def test_register_ep_happy_path(fresh_registry_with_qnn: WinMLEPRegistry) -> None:
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
        result = fresh_registry_with_qnn.register_ep(entry)

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


def test_register_ep_is_idempotent_per_dll_path(fresh_registry_with_qnn: WinMLEPRegistry) -> None:
    """A second register_ep for the same dll_path returns the cached WinMLEP.

    The atomic primitive is idempotent on (entry.dll_path): the first call
    loads the DLL, enumerates devices and caches the WinMLEP; the second
    call short-circuits to the cached value WITHOUT re-invoking
    ``ort.register_execution_provider_library``. This lets callers
    (``auto_device``, ``_gather_ep_info``) loop over discovered entries
    across multiple invocations in the same process without the second
    call falsely failing with "DLL already registered".

    The previous "raise on re-call" semantic produced spurious
    ``WinMLEPRegistrationFailed`` errors on the second ``--list-ep`` and
    masked the real ``DeviceNotFound`` from ``auto_device`` on a second
    invocation for the same EP.
    """
    entry = _ep_entry("QNNExecutionProvider")
    qnn = _fake_ort_device("QNNExecutionProvider", "NPU")
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.get_ep_devices.return_value = [qnn]
        mock_ort.register_execution_provider_library = MagicMock()
        first = fresh_registry_with_qnn.register_ep(entry)
        second = fresh_registry_with_qnn.register_ep(entry)
    assert isinstance(first, WinMLEP)
    # Cached identity — same object on both calls.
    assert second is first
    # ORT's register call only happened once; the second call short-circuited.
    assert mock_ort.register_execution_provider_library.call_count == 1


def test_register_ep_suffix_for_repeat_ep_name(fresh_registry_with_qnn: WinMLEPRegistry) -> None:
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
        first = fresh_registry_with_qnn.register_ep(entry_a)
        second = fresh_registry_with_qnn.register_ep(entry_b)

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


def test_register_ep_failure_wraps(fresh_registry_with_qnn: WinMLEPRegistry) -> None:
    """register_ep raises WinMLEPRegistrationFailed when ORT's register call raises."""
    entry = _ep_entry("QNNExecutionProvider")
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.register_execution_provider_library.side_effect = RuntimeError("dll boom")
        mock_ort.get_ep_devices.return_value = []
        with pytest.raises(WinMLEPRegistrationFailed):
            fresh_registry_with_qnn.register_ep(entry)


def test_register_ep_yields_zero_devices_raises(fresh_registry_with_qnn: WinMLEPRegistry) -> None:
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
        with pytest.raises(WinMLEPRegistrationFailed, match=r"no\s+OrtEpDevices"):
            fresh_registry_with_qnn.register_ep(entry)


@pytest.mark.parametrize(
    "get_devices",
    [
        pytest.param(RuntimeError("enumeration boom"), id="enumeration-failure"),
        pytest.param([], id="zero-matching-devices"),
    ],
)
def test_register_ep_rolls_back_native_registration_after_enumeration_failure(
    fresh_registry_with_qnn: WinMLEPRegistry, get_devices: object
) -> None:
    """Post-registration failures unregister and leave registry state unchanged."""
    entry = _ep_entry("QNNExecutionProvider")
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.register_execution_provider_library = MagicMock()
        mock_ort.unregister_execution_provider_library = MagicMock()
        if isinstance(get_devices, Exception):
            mock_ort.get_ep_devices.side_effect = get_devices
        else:
            mock_ort.get_ep_devices.return_value = get_devices

        with pytest.raises(WinMLEPRegistrationFailed) as excinfo:
            fresh_registry_with_qnn.register_ep(entry)

    mock_ort.unregister_execution_provider_library.assert_called_once_with("QNNExecutionProvider")
    assert fresh_registry_with_qnn._registration_count == {}
    assert fresh_registry_with_qnn._registered == {}
    if isinstance(get_devices, Exception):
        assert excinfo.value.__cause__ is get_devices


def test_register_ep_wraps_get_ep_devices_failure_plugin_branch(
    fresh_registry_with_qnn: WinMLEPRegistry,
) -> None:
    """A raise from ``ort.get_ep_devices`` after DLL registration surfaces as
    ``WinMLEPRegistrationFailed`` so ``auto_device``'s retry loop can fall
    through to the next candidate instead of letting a raw ORT exception
    crash the CLI (R-06)."""
    entry = _ep_entry("QNNExecutionProvider")
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.register_execution_provider_library = MagicMock()
        mock_ort.get_ep_devices.side_effect = RuntimeError("ORT internal init failure")
        with pytest.raises(WinMLEPRegistrationFailed, match=r"get_ep_devices"):
            fresh_registry_with_qnn.register_ep(entry)


def test_register_ep_wraps_get_ep_devices_failure_builtin_branch() -> None:
    """Same R-06 contract applies to BuiltinSource entries: a raise from
    ``ort.get_ep_devices`` must surface as ``WinMLEPRegistrationFailed``,
    not as the raw ORT exception."""
    from winml.modelkit.ep_path import BuiltinSource, EPEntry

    WinMLEPRegistry._instance = None
    try:
        with (
            patch(
                "winml.modelkit.session.ep_registry.discover_all_eps",
                return_value=[],
            ),
            patch(
                "winml.modelkit.session.ep_registry.ort.get_available_providers",
                return_value=["CPUExecutionProvider"],
            ),
            patch(
                "winml.modelkit.session.ep_registry.ort.get_ep_devices",
                return_value=[
                    MagicMock(
                        ep_name="CPUExecutionProvider",
                        device_type="CPU",
                        vendor_id=0x0,
                        device_id=0x0,
                        ep_metadata={},
                    )
                ],
            ),
        ):
            reg = WinMLEPRegistry.instance()
        builtin_entry = EPEntry(
            ep_name="CPUExecutionProvider",
            dll_path=Path(),
            source=BuiltinSource(eps=("CPUExecutionProvider",)),
        )
        # Now flip get_ep_devices to raise — exercises the BuiltinSource
        # branch's helper call inside register_ep.
        with (
            patch(
                "winml.modelkit.session.ep_registry.ort.get_ep_devices",
                side_effect=RuntimeError("driver reset"),
            ),
            pytest.raises(WinMLEPRegistrationFailed, match=r"get_ep_devices"),
        ):
            reg.register_ep(builtin_entry)
    finally:
        WinMLEPRegistry._instance = None


def test_register_ep_appends_to_entries_when_not_present(
    fresh_registry_with_qnn: WinMLEPRegistry,
) -> None:
    """register_ep does NOT mutate _discovered (v2.9 — silent-mutation hack removed).

    Pre-v2.9 the registry back-inserted caller-supplied entries into
    _discovered. With unified-source synthesis (every entry, including
    built-ins, lives in _discovered at __init__), the back-insertion is
    provably unreachable from production callers. register_ep now treats
    _discovered as read-only; callers handing register_ep an entry that
    isn't in _discovered get a successful registration but the entry stays
    out of the discovery cache.
    """
    fresh_registry_with_qnn._discovered = []  # simulate "entry not in discovery"
    entry = _ep_entry("OpenVINOExecutionProvider", dll="C:/fake/openvino.dll")
    fake_dev = _fake_ort_device(
        "OpenVINOExecutionProvider",
        "NPU",
        dll_path="C:/fake/openvino.dll",
    )
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.get_ep_devices.return_value = [fake_dev]
        mock_ort.register_execution_provider_library = MagicMock()
        fresh_registry_with_qnn.register_ep(entry)

    # _discovered is unchanged — register_ep is read-only on the discovery cache.
    assert fresh_registry_with_qnn._discovered == []
    assert fresh_registry_with_qnn._entries_for("OpenVINOExecutionProvider") == []


def test_builtin_source_synthesized_into_discovered_at_init() -> None:
    """v2.9 regression guard: built-in EPs appear in _discovered as
    BuiltinSource entries after WinMLEPRegistry.__init__.

    Pre-v2.9 the original `winml perf --ep cpu --device cpu` failure was
    that `auto_device → _entries_for("CPUExecutionProvider")` returned
    `[]` because filesystem discovery never produced built-in entries.
    The unified-source fix synthesizes one EPEntry per name in
    ort.get_available_providers() that filesystem discovery didn't
    already cover. This test pins the synthesis invariant.
    """
    from winml.modelkit.ep_path import BuiltinSource

    # Stub: discover_all_eps() finds nothing on disk; ORT reports CPU
    # for BOTH get_available_providers (name list) AND get_ep_devices
    # (handle list) — synthesis intersects the two and would yield an
    # empty set if the runner's real get_ep_devices() omitted CPU.
    WinMLEPRegistry._instance = None
    try:
        with (
            patch(
                "winml.modelkit.session.ep_registry.discover_all_eps",
                return_value=[],
            ),
            patch(
                "winml.modelkit.session.ep_registry.ort.get_available_providers",
                return_value=["CPUExecutionProvider"],
            ),
            patch(
                "winml.modelkit.session.ep_registry.ort.get_ep_devices",
                return_value=[MagicMock(ep_name="CPUExecutionProvider")],
            ),
        ):
            reg = WinMLEPRegistry.instance()

        entries = reg._entries_for("CPUExecutionProvider")
        assert len(entries) == 1, (
            "Expected exactly one synthesized BuiltinSource entry for "
            f"CPUExecutionProvider, got {len(entries)}."
        )
        entry = entries[0]
        assert isinstance(entry.source, BuiltinSource), (
            f"Built-in entry's source should be BuiltinSource, got {type(entry.source).__name__}."
        )
        assert entry.ep_name == "CPUExecutionProvider"
    finally:
        WinMLEPRegistry._instance = None


def test_builtin_source_only_synthesized_when_ort_exposes_devices() -> None:
    """F-07: a misconfigured ORT may include 'CPUExecutionProvider' in
    get_available_providers() yet expose zero matching OrtEpDevices via
    get_ep_devices(). Don't synthesize a BuiltinSource entry for such an
    EP — auto_detect_device would otherwise pick CPU and then session
    build would fail with a confusing 'Built-in EP exposed no devices'
    error. Skipping at synthesis time keeps available_eps() honest.
    """
    from winml.modelkit.ep_path import BuiltinSource

    WinMLEPRegistry._instance = None
    try:
        with (
            patch(
                "winml.modelkit.session.ep_registry.discover_all_eps",
                return_value=[],
            ),
            patch(
                "winml.modelkit.session.ep_registry.ort.get_available_providers",
                return_value=["CPUExecutionProvider", "DmlExecutionProvider"],
            ),
            patch(
                "winml.modelkit.session.ep_registry.ort.get_ep_devices",
                # ORT lists both providers but only exposes a CPU device.
                # DML must NOT be synthesized.
                return_value=[
                    MagicMock(
                        ep_name="CPUExecutionProvider",
                        device_type="CPU",
                        vendor_id=0x0,
                        device_id=0x0,
                        ep_metadata={},
                    )
                ],
            ),
        ):
            reg = WinMLEPRegistry.instance()

        ep_names = {e.ep_name for e in reg._discovered if isinstance(e.source, BuiltinSource)}
        assert ep_names == {"CPUExecutionProvider"}, (
            f"Expected only CPU to be synthesized (DML has no devices), got {sorted(ep_names)}."
        )
    finally:
        WinMLEPRegistry._instance = None


def test_register_ep_builtin_source_is_object_identity_idempotent() -> None:
    """F-08: BuiltinSource register_ep returns the SAME WinMLEP on repeat calls.

    Pre-fix, the BuiltinSource branch constructed a fresh WinMLEP +
    WinMLDevice wrappers from the same handles every call, breaking
    object identity. auto_device's precedence loop may re-register the
    same built-in entry across calls; the WinMLEPDevice invariant
    (device must be `is` one of ep.devices) and downstream identity
    checks rely on cached singletons. The DLL-based path is already
    cached via self._registered; built-ins now use the same dict keyed
    by Path-of-the-marker-source (no collision since each BuiltinSource
    has a unique ep_name).
    """
    from winml.modelkit.ep_path import BuiltinSource, EPEntry

    WinMLEPRegistry._instance = None
    try:
        builtin_entry = EPEntry(
            ep_name="CPUExecutionProvider",
            dll_path=Path(),
            source=BuiltinSource(eps=("CPUExecutionProvider",)),
        )
        with (
            patch(
                "winml.modelkit.session.ep_registry.discover_all_eps",
                return_value=[],
            ),
            patch(
                "winml.modelkit.session.ep_registry.ort.get_available_providers",
                return_value=["CPUExecutionProvider"],
            ),
            patch(
                "winml.modelkit.session.ep_registry.ort.get_ep_devices",
                return_value=[
                    MagicMock(
                        ep_name="CPUExecutionProvider",
                        device_type="CPU",
                        vendor_id=0x0,
                        device_id=0x0,
                        ep_metadata={},
                    )
                ],
            ),
        ):
            reg = WinMLEPRegistry.instance()
            first = reg.register_ep(builtin_entry)
            second = reg.register_ep(builtin_entry)

        assert isinstance(first, WinMLEP)
        assert second is first, (
            "BuiltinSource register_ep must cache by ep_name and return "
            "object-identity-equal WinMLEPs on repeat calls."
        )
    finally:
        WinMLEPRegistry._instance = None
