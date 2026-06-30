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

The harness drives the registry's cached ``_discovered`` list and patches
``register_ep`` so we can construct deterministic candidate lists and
registration outcomes without touching any real plugin DLL.
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
    WinMLDevice,
    WinMLEP,
    WinMLEPDevice,
    WinMLEPRegistrationFailed,
    WinMLEPRegistry,
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
    device = WinMLDevice(_fake_ort_device(entry.ep_name, device_type))
    return WinMLEP(source=entry, devices=(device,))


# fresh_registry fixture lives in tests/unit/session/conftest.py — shared
# with test_ep_registry.py per the tests/CLAUDE.md DRY rule.


# ---------- tests ----------------------------------------------------------


class TestAutoDevice:
    """One test per scenario from the Batch E plan."""

    def test_a_single_pypi_source_no_source_pin(
        self, fresh_registry: WinMLEPRegistry
    ) -> None:
        """Scenario a: single PyPI source discovered, source=None on target."""
        entry = _pypi_entry("OpenVINOExecutionProvider")
        winml_ep = _winml_ep_with_device(entry, "NPU")
        fresh_registry._discovered = [entry]

        with patch.object(
            fresh_registry, "register_ep", return_value=winml_ep
        ) as mock_register:
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
        fresh_registry._discovered = [entry]

        with patch.object(
            fresh_registry, "register_ep", return_value=winml_ep
        ) as mock_register:
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

        fresh_registry._discovered = [primary, shadowed]
        with patch.object(
            fresh_registry, "register_ep", side_effect=selective_register
        ) as mock_register:
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
        fresh_registry._discovered = [entry]

        with patch.object(fresh_registry, "register_ep", return_value=winml_ep):
            target = EPDeviceTarget(ep="openvino", device="npu")
            with pytest.raises(DeviceNotFound, match="NPU"):
                fresh_registry.auto_device(target)

    def test_fail_then_succeed_but_wrong_device_raises_device_not_found(
        self, fresh_registry: WinMLEPRegistry
    ) -> None:
        """T-04 regression guard: stale ``last_error`` must not surface when
        a later candidate registered cleanly but exposed no matching device.

        Sequence: candidate #1 raises ``WinMLEPRegistrationFailed``;
        candidate #2 registers cleanly but its ``devices`` tuple has no
        match for ``target.device``. The precedence loop exhausts.
        Expected exception type: ``DeviceNotFound`` (the second outcome).
        Pre-fix bug: ``WinMLEPRegistrationFailed`` (the first outcome's
        stale traceback survives because ``last_error`` is never reset
        after the successful registration).
        """
        primary = _pypi_entry("OpenVINOExecutionProvider", dll="C:/fake/primary.dll")
        shadowed = _pypi_entry(
            "OpenVINOExecutionProvider", dll="C:/fake/shadow.dll"
        )
        # Shadowed candidate registers cleanly but its only device is GPU
        # while target asks for NPU — no match → fall through to next.
        shadowed_ep = _winml_ep_with_device(shadowed, "GPU")

        primary_error = WinMLEPRegistrationFailed("primary DLL boom")

        def selective_register(entry: EPEntry) -> WinMLEP:
            if entry.dll_path == primary.dll_path:
                raise primary_error
            return shadowed_ep

        fresh_registry._discovered = [primary, shadowed]
        with patch.object(
            fresh_registry, "register_ep", side_effect=selective_register
        ):
            target = EPDeviceTarget(ep="openvino", device="npu")
            with pytest.raises(DeviceNotFound):
                fresh_registry.auto_device(target)

    def test_e_all_candidates_fail_registration(
        self, fresh_registry: WinMLEPRegistry
    ) -> None:
        """Scenario e: every candidate raises WinMLEPRegistrationFailed."""
        entry1 = _pypi_entry("OpenVINOExecutionProvider", dll="C:/fake/a.dll")
        entry2 = _pypi_entry("OpenVINOExecutionProvider", dll="C:/fake/b.dll")
        fresh_registry._discovered = [entry1, entry2]

        with patch.object(
            fresh_registry,
            "register_ep",
            side_effect=WinMLEPRegistrationFailed("dll boom"),
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
        fresh_registry._discovered = [entry]

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

        fresh_registry._discovered = [pypi_entry, msix_entry]
        with patch.object(
            fresh_registry, "register_ep", side_effect=selective_register
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

        fresh_registry._discovered = []
        target = EPDeviceTarget(ep="openvino", device="npu")
        with pytest.raises(WinMLEPNotDiscovered):
            fresh_registry.auto_device(target)

    def test_second_call_same_ep_different_device_surfaces_device_not_found(
        self, fresh_registry: WinMLEPRegistry
    ) -> None:
        """Cross-call regression: second auto_device for same EP must NOT raise
        WinMLEPRegistrationFailed("DLL already registered").

        Bug context: ``auto_device`` calls ``register_ep`` unconditionally
        inside the precedence loop. On the SECOND invocation in the same
        process, plugin entries already in ``_registered`` previously
        caused the REAL ``register_ep`` to raise
        ``WinMLEPRegistrationFailed``, masking the real reason (the EP
        simply didn't expose the requested device class on this host).

        After Option A fix: ``register_ep`` is idempotent on
        ``entry.dll_path`` → returns the cached ``WinMLEP`` → device
        class check fails → ``DeviceNotFound`` (the truth).

        Drives the REAL ``register_ep`` (only ``ort.*`` mocked) so the
        bug surfaces as it does in production.
        """
        from pathlib import Path

        entry = _pypi_entry("QNNExecutionProvider", dll="C:/fake/qnn.dll")
        fresh_registry._discovered = [entry]

        # Build a single fake ORT handle for QNN/NPU bound to the entry's
        # dll_path — register_ep filters get_ep_devices() by library_path.
        fake_qnn_npu = MagicMock()
        fake_qnn_npu.ep_name = "QNNExecutionProvider"
        fake_qnn_npu.ep_metadata = {"library_path": str(Path("C:/fake/qnn.dll"))}
        fake_qnn_npu.device.type.name = "NPU"
        fake_qnn_npu.device.vendor_id = 0x4D4F
        fake_qnn_npu.device.device_id = 0x0001

        with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
            mock_ort.get_ep_devices.return_value = [fake_qnn_npu]
            mock_ort.register_execution_provider_library = MagicMock()

            # First call: NPU — register_ep loads the DLL and succeeds.
            first = fresh_registry.auto_device(
                EPDeviceTarget(ep="qnn", device="npu")
            )
            assert isinstance(first, WinMLEPDevice)
            assert first.device.device_type == "NPU"

            # Second call: GPU — register_ep must short-circuit on the
            # cached entry (idempotent) and the device-class loop must
            # surface DeviceNotFound, NOT WinMLEPRegistrationFailed.
            with pytest.raises(DeviceNotFound, match="GPU"):
                fresh_registry.auto_device(
                    EPDeviceTarget(ep="qnn", device="gpu")
                )

            # ORT's register call must have happened exactly once across
            # both auto_device invocations (idempotency guarantee).
            assert mock_ort.register_execution_provider_library.call_count == 1


# ---------- Built-in EP path (v2.9 — unified BuiltinSource) ----------------


class TestAutoDeviceBuiltIn:
    """auto_device must handle built-in EPs (CPU/Dml/Azure) via the unified
    BuiltinSource synthesis pipeline, not raise WinMLEPNotDiscovered.

    Pre-v2.9, ``--ep cpu --device cpu`` failed with
    ``No EPEntry discovered for ep='CPUExecutionProvider'`` because
    ``_discovered`` only contained filesystem-found plugin entries.
    v2.9 synthesizes BuiltinSource entries into ``_discovered`` at
    registry ``__init__``, so the precedence loop finds them, and
    ``register_ep`` dispatches on isinstance(source, BuiltinSource) to
    wrap the pre-loaded ORT handles directly.
    """

    def test_cpu_built_in_returns_winml_ep_device(
        self, fresh_registry: WinMLEPRegistry
    ) -> None:
        """auto_device(EP=cpu, device=cpu) returns a WinMLEPDevice
        bound to the CPU EP — no WinMLEPNotDiscovered."""
        from winml.modelkit.ep_path import BuiltinSource

        cpu_entry = EPEntry(
            ep_name="CPUExecutionProvider",
            dll_path=Path(),
            source=BuiltinSource(eps=("CPUExecutionProvider",)),
        )
        fresh_registry._discovered = [cpu_entry]
        fake_cpu = _fake_ort_device("CPUExecutionProvider", "CPU")

        target = EPDeviceTarget(ep="cpu", device="cpu")
        with patch(
            "winml.modelkit.session.ep_registry.ort.get_ep_devices",
            return_value=[fake_cpu],
        ):
            result = fresh_registry.auto_device(target)

        assert isinstance(result, WinMLEPDevice)
        assert result.ep.source.ep_name == "CPUExecutionProvider"
        assert result.device.device_type == "CPU"

    def test_dml_built_in_returns_winml_ep_device(
        self, fresh_registry: WinMLEPRegistry
    ) -> None:
        """Same shape for DmlExecutionProvider/gpu."""
        from winml.modelkit.ep_path import BuiltinSource

        dml_entry = EPEntry(
            ep_name="DmlExecutionProvider",
            dll_path=Path(),
            source=BuiltinSource(eps=("DmlExecutionProvider",)),
        )
        fresh_registry._discovered = [dml_entry]
        fake_dml = _fake_ort_device("DmlExecutionProvider", "GPU")

        target = EPDeviceTarget(ep="dml", device="gpu")
        with patch(
            "winml.modelkit.session.ep_registry.ort.get_ep_devices",
            return_value=[fake_dml],
        ):
            result = fresh_registry.auto_device(target)

        assert isinstance(result, WinMLEPDevice)
        assert result.ep.source.ep_name == "DmlExecutionProvider"
        assert result.device.device_type == "GPU"

    def test_built_in_no_matching_device_class_raises(
        self, fresh_registry: WinMLEPRegistry
    ) -> None:
        """If the built-in EP doesn't expose the requested device class,
        DeviceNotFound is raised after register_ep succeeds (not
        WinMLEPNotDiscovered)."""
        from winml.modelkit.ep_path import BuiltinSource

        cpu_entry = EPEntry(
            ep_name="CPUExecutionProvider",
            dll_path=Path(),
            source=BuiltinSource(eps=("CPUExecutionProvider",)),
        )
        fresh_registry._discovered = [cpu_entry]
        # ORT exposes only CPU but user asks for NPU.
        fake_cpu = _fake_ort_device("CPUExecutionProvider", "CPU")

        target = EPDeviceTarget(ep="cpu", device="npu")
        with patch(
            "winml.modelkit.session.ep_registry.ort.get_ep_devices",
            return_value=[fake_cpu],
        ), pytest.raises(DeviceNotFound):
            fresh_registry.auto_device(target)

    def test_plugin_wins_precedence_over_built_in_same_ep_name(
        self, fresh_registry: WinMLEPRegistry
    ) -> None:
        """If a plugin EP with the same ep_name was discovered AND ORT
        also has it built-in, the plugin (earlier in _discovered) wins.

        Pinning the "built-in is lowest priority" semantic.
        """
        from winml.modelkit.ep_path import BuiltinSource

        plugin_entry = _pypi_entry("CPUExecutionProvider")
        builtin_entry = EPEntry(
            ep_name="CPUExecutionProvider",
            dll_path=Path(),
            source=BuiltinSource(eps=("CPUExecutionProvider",)),
        )
        # Plugin first (higher precedence), then built-in fallback.
        fresh_registry._discovered = [plugin_entry, builtin_entry]
        plugin_ep = _winml_ep_with_device(plugin_entry, "CPU")

        target = EPDeviceTarget(ep="cpu", device="cpu")
        with patch.object(
            fresh_registry, "register_ep", return_value=plugin_ep
        ) as mock_register:
            result = fresh_registry.auto_device(target)

        # Plugin entry was tried, not the built-in.
        mock_register.assert_called_once_with(plugin_entry)
        assert isinstance(result, WinMLEPDevice)
