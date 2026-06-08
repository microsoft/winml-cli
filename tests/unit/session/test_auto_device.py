# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ``WinMLEPRegistry.auto_device(target) -> WinMLEPDevice``.

Each test corresponds to one of the seven scenarios locked for Batch E:

    a. Single PyPI source, source=None on target  -> happy path
    b. Pinned source matches one candidate         -> happy path
    c. Primary source's DLL load fails, shadowed   -> retry-and-succeed
    d. Source loaded but no device class matches   -> DeviceNotFound
    e. All candidates fail to register             -> WinMLEPRegistrationFailed
    f. EPDeviceTarget("auto", "auto") passed       -> ValueError
    g. source tag does not match any candidate     -> UnknownListingPick

The harness drives ``discover_all_eps`` and ``register_ep`` via patching so we
can construct deterministic candidate lists and registration outcomes without
touching any real plugin DLL.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.ep_path import EPEntry, MSIXPackageSource, PyPISource
from winml.modelkit.session import (
    DeviceNotFound,
    EPDeviceTarget,
    UnknownListingPick,
    WinMLEP,
    WinMLEPDevice,
    WinMLEPRegistrationFailed,
    WinMLEPRegistry,
    wrap_ort_device,
)


# ---------- helpers --------------------------------------------------------


def _fake_ort_device(ep_name: str, device_type: str) -> MagicMock:
    """Construct a MagicMock OrtEpDevice with controllable ep_name + device.type."""
    d = MagicMock()
    d.ep_name = ep_name
    d.device.type.name = device_type
    d.ep_metadata = {}
    d.device.metadata = {}
    d.device.vendor = "FakeVendor"
    d.device.vendor_id = 0x8086
    d.device.device_id = 0x0001
    d.ep_vendor = "Microsoft"
    return d


def _pypi_entry(ep_name: str, dll: str = "C:/fake/openvino.dll") -> EPEntry:
    return EPEntry(
        ep_name=ep_name,
        dll_path=Path(dll),
        source=PyPISource(
            distribution="onnxruntime-ep-openvino",
            relative_dll="lib/openvino_ep.dll",
            eps=(ep_name,),
        ),
    )


def _msix_workload_entry(ep_name: str, dll: str = "C:/fake/qnn.dll") -> EPEntry:
    return EPEntry(
        ep_name=ep_name,
        dll_path=Path(dll),
        source=MSIXPackageSource(
            family_name_prefix="WindowsWorkload.EP.QNN.",
            relative_dll="ExecutionProvider/onnxruntime_providers_qnn.dll",
            eps=(ep_name,),
        ),
    )


def _winml_ep_with_device(entry: EPEntry, device_type: str) -> WinMLEP:
    """Build a WinMLEP wrapping one device of the requested type."""
    device = wrap_ort_device(_fake_ort_device(entry.ep_name, device_type))
    return WinMLEP(source=entry, devices=(device,))


@pytest.fixture
def fresh_registry() -> WinMLEPRegistry:
    """Singleton with cleared registration caches.

    Each scenario patches discover_all_eps and register_ep, so the only
    state we need to reset is the per-DLL cache that ``register_ep``
    would otherwise short-circuit.
    """
    reg = WinMLEPRegistry.instance()
    reg._registered = {}
    reg._registered_eps = []
    reg._registration_failures = {}
    return reg


# ---------- tests ----------------------------------------------------------


class TestAutoDevice:
    """One test per scenario from the Batch E plan."""

    def test_a_single_pypi_source_no_source_pin(
        self, fresh_registry: WinMLEPRegistry
    ) -> None:
        """Scenario a: single PyPI source discovered, source=None on target."""
        entry = _pypi_entry("OpenVINOExecutionProvider")
        winml_ep = _winml_ep_with_device(entry, "NPU")

        with (
            patch(
                "winml.modelkit.ep_path.discover_all_eps",
                return_value=[entry],
            ),
            patch.object(
                fresh_registry, "register_ep", return_value=winml_ep
            ) as mock_register,
        ):
            target = EPDeviceTarget(ep="openvino", device="npu")
            result = fresh_registry.auto_device(target)

        assert isinstance(result, WinMLEPDevice)
        assert result.device.device_type == "NPU"
        assert result.ep is winml_ep
        mock_register.assert_called_once_with(entry)

    def test_b_pinned_source_matches(self, fresh_registry: WinMLEPRegistry) -> None:
        """Scenario b: source='pypi' pinned, candidate exists."""
        entry = _pypi_entry("OpenVINOExecutionProvider")
        winml_ep = _winml_ep_with_device(entry, "NPU")

        with (
            patch(
                "winml.modelkit.ep_path.discover_all_eps",
                return_value=[entry],
            ),
            patch.object(
                fresh_registry, "register_ep", return_value=winml_ep
            ) as mock_register,
        ):
            target = EPDeviceTarget(ep="openvino", device="npu", source="pypi")
            result = fresh_registry.auto_device(target)

        assert isinstance(result, WinMLEPDevice)
        mock_register.assert_called_once_with(entry)

    def test_c_primary_dll_load_fails_shadowed_succeeds(
        self, fresh_registry: WinMLEPRegistry
    ) -> None:
        """Scenario c: primary registration raises; shadowed candidate registers.

        Verifies the retry loop in auto_device walks past failed candidates
        without silently swallowing the underlying error (the failure is
        captured as ``last_error`` and only re-raised if NO candidate
        succeeds — proven indirectly by the success of the shadowed one).
        """
        primary = _pypi_entry("OpenVINOExecutionProvider", dll="C:/fake/primary.dll")
        shadowed = _pypi_entry(
            "OpenVINOExecutionProvider", dll="C:/fake/shadow.dll"
        )
        shadowed_ep = _winml_ep_with_device(shadowed, "NPU")

        primary_error = WinMLEPRegistrationFailed("primary DLL boom")

        def selective_register(entry: EPEntry) -> WinMLEP:
            if entry.dll_path == primary.dll_path:
                raise primary_error
            return shadowed_ep

        with (
            patch(
                "winml.modelkit.ep_path.discover_all_eps",
                return_value=[primary, shadowed],
            ),
            patch.object(
                fresh_registry, "register_ep", side_effect=selective_register
            ) as mock_register,
        ):
            target = EPDeviceTarget(ep="openvino", device="npu")
            result = fresh_registry.auto_device(target)

        # Shadowed candidate won — but the primary was actually tried first.
        assert result.ep is shadowed_ep
        assert mock_register.call_count == 2
        called_paths = [
            call.args[0].dll_path for call in mock_register.call_args_list
        ]
        assert called_paths == [primary.dll_path, shadowed.dll_path]

    def test_d_source_loads_but_no_matching_device(
        self, fresh_registry: WinMLEPRegistry
    ) -> None:
        """Scenario d: registration succeeds but WinMLEP has no matching device class."""
        entry = _pypi_entry("OpenVINOExecutionProvider")
        # Source exposes a GPU device, but target requests NPU.
        winml_ep = _winml_ep_with_device(entry, "GPU")

        with (
            patch(
                "winml.modelkit.ep_path.discover_all_eps",
                return_value=[entry],
            ),
            patch.object(fresh_registry, "register_ep", return_value=winml_ep),
        ):
            target = EPDeviceTarget(ep="openvino", device="npu")
            with pytest.raises(DeviceNotFound, match="NPU"):
                fresh_registry.auto_device(target)

    def test_e_all_candidates_fail_registration(
        self, fresh_registry: WinMLEPRegistry
    ) -> None:
        """Scenario e: every candidate raises WinMLEPRegistrationFailed."""
        entry1 = _pypi_entry("OpenVINOExecutionProvider", dll="C:/fake/a.dll")
        entry2 = _pypi_entry("OpenVINOExecutionProvider", dll="C:/fake/b.dll")

        with (
            patch(
                "winml.modelkit.ep_path.discover_all_eps",
                return_value=[entry1, entry2],
            ),
            patch.object(
                fresh_registry,
                "register_ep",
                side_effect=WinMLEPRegistrationFailed("dll boom"),
            ),
        ):
            target = EPDeviceTarget(ep="openvino", device="npu")
            with pytest.raises(WinMLEPRegistrationFailed) as ei:
                fresh_registry.auto_device(target)
        # The aggregate raise must chain the last per-candidate failure.
        assert ei.value.__cause__ is not None
        assert "dll boom" in str(ei.value.__cause__)

    def test_f_auto_target_raises_value_error(
        self, fresh_registry: WinMLEPRegistry
    ) -> None:
        """Scenario f: EPDeviceTarget('auto', 'auto') must NOT be re-resolved here."""
        target = EPDeviceTarget(ep="auto", device="auto")
        with pytest.raises(ValueError, match="auto"):
            fresh_registry.auto_device(target)

    def test_f_auto_ep_only_raises_value_error(
        self, fresh_registry: WinMLEPRegistry
    ) -> None:
        """Companion: ep='auto' with concrete device must also raise."""
        target = EPDeviceTarget(ep="auto", device="npu")
        with pytest.raises(ValueError, match="auto"):
            fresh_registry.auto_device(target)

    def test_f_auto_device_only_raises_value_error(
        self, fresh_registry: WinMLEPRegistry
    ) -> None:
        """Companion: device='auto' with concrete EP must also raise."""
        target = EPDeviceTarget(ep="openvino", device="auto")
        with pytest.raises(ValueError, match="auto"):
            fresh_registry.auto_device(target)

    def test_g_unmatched_source_tag_raises_unknown_listing_pick(
        self, fresh_registry: WinMLEPRegistry
    ) -> None:
        """Scenario g: source tag is valid but no candidate carries that tag.

        The plan calls this scenario ``source='msix-does-not-exist'`` — but
        EPDeviceTarget validates ``source`` at construction time against
        the closed set of canonical tags, so an unknown tag string never
        reaches auto_device. We instead simulate the user pinning
        ``source='msix-workload'`` (a *valid* tag) when only PyPI sources
        have been discovered for OpenVINO.
        """
        entry = _pypi_entry("OpenVINOExecutionProvider")  # pypi only

        with patch(
            "winml.modelkit.ep_path.discover_all_eps",
            return_value=[entry],
        ):
            target = EPDeviceTarget(
                ep="openvino", device="npu", source="msix-workload"
            )
            with pytest.raises(UnknownListingPick) as ei:
                fresh_registry.auto_device(target)

        assert ei.value.ep_name == "openvino"
        assert ei.value.source_tag == "msix-workload"

    def test_g_msix_workload_pin_matches_correct_entry(
        self, fresh_registry: WinMLEPRegistry
    ) -> None:
        """Companion to g: when both PyPI and MSIX-workload exist, the pin filters.

        Verifies the source-tag filter narrows the candidate set rather than
        breaking ``auto_device`` outright.
        """
        pypi_entry = _pypi_entry("QNNExecutionProvider", dll="C:/fake/pypi-qnn.dll")
        msix_entry = _msix_workload_entry(
            "QNNExecutionProvider", dll="C:/fake/msix-qnn.dll"
        )
        msix_ep = _winml_ep_with_device(msix_entry, "NPU")

        def selective_register(entry: EPEntry) -> WinMLEP:
            if entry.dll_path == msix_entry.dll_path:
                return msix_ep
            raise AssertionError(
                f"PyPI entry should have been filtered out by source tag, "
                f"got {entry.dll_path}"
            )

        with (
            patch(
                "winml.modelkit.ep_path.discover_all_eps",
                return_value=[pypi_entry, msix_entry],
            ),
            patch.object(
                fresh_registry, "register_ep", side_effect=selective_register
            ),
        ):
            target = EPDeviceTarget(
                ep="qnn", device="npu", source="msix-workload"
            )
            result = fresh_registry.auto_device(target)

        assert result.ep is msix_ep

    def test_no_candidate_for_ep_raises_not_discovered(
        self, fresh_registry: WinMLEPRegistry
    ) -> None:
        """No EPEntry at all for the requested EP -> WinMLEPNotDiscovered.

        Not in the seven-scenario locked list, but covers the
        ``not candidates`` branch directly above the source-tag filter.
        """
        from winml.modelkit.session import WinMLEPNotDiscovered

        with patch(
            "winml.modelkit.ep_path.discover_all_eps",
            return_value=[],
        ):
            target = EPDeviceTarget(ep="openvino", device="npu")
            with pytest.raises(WinMLEPNotDiscovered):
                fresh_registry.auto_device(target)
