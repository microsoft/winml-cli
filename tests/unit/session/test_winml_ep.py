# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the WinMLEP aggregate dataclass (Batch B).

WinMLEP is the per-source registration record produced by
:meth:`WinMLEPRegistry.register_ep`. It pairs the source :class:`EPEntry`
with every :class:`WinMLDevice` that ORT exposed after loading the DLL.

Invariants exercised here:

- ``devices`` is non-empty (``__post_init__`` raises ``ValueError`` otherwise).
- ``ep_devices()`` flattens into one :class:`WinMLEPDevice` per device.
- Each ``WinMLEPDevice.ep`` returned is the *same* WinMLEP instance.
- Each ``WinMLEPDevice.device`` is *one of* ``WinMLEP.devices`` (identity, not copy).
- Frozenness: assigning to fields raises :class:`FrozenInstanceError`.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from winml.modelkit.ep_path import EPEntry, PyPISource
from winml.modelkit.session import WinMLEP, WinMLEPDevice, wrap_ort_device


def _make_fake_ort_ep_device(ep_name: str, device_type: str) -> MagicMock:
    """Build a MagicMock OrtEpDevice with minimal fields for WinMLDevice wrapping.

    Module-level helper so tests do not need to monkey-patch conftest. Mirrors
    the shape conftest's ``_fake_ort_device`` returns but lives next to the
    consumers in this file.
    """
    d = MagicMock()
    d.ep_name = ep_name
    d.device.type.name = device_type
    d.ep_metadata = {}
    d.device.metadata = {}
    d.device.vendor = "FakeVendor"
    d.ep_vendor = "Microsoft"
    return d


def _make_entry(ep_name: str = "OpenVINOExecutionProvider") -> EPEntry:
    """Minimal EPEntry for WinMLEP construction tests."""
    return EPEntry(
        ep_name=ep_name,
        dll_path=Path(f"C:/fake/{ep_name}.dll"),
        source=PyPISource(
            distribution="fake-dist",
            relative_dll="fake.dll",
            eps=(ep_name,),
        ),
    )


class TestWinMLEPConstruction:
    """Direct dataclass construction + the non-empty-devices invariant."""

    def test_single_device_succeeds(self) -> None:
        entry = _make_entry()
        device = wrap_ort_device(_make_fake_ort_ep_device("OpenVINOExecutionProvider", "NPU"))
        ep = WinMLEP(source=entry, devices=(device,))
        assert ep.source is entry
        assert ep.devices == (device,)

    def test_multiple_devices_succeed(self) -> None:
        entry = _make_entry()
        d_npu = wrap_ort_device(_make_fake_ort_ep_device("OpenVINOExecutionProvider", "NPU"))
        d_gpu = wrap_ort_device(_make_fake_ort_ep_device("OpenVINOExecutionProvider", "GPU"))
        ep = WinMLEP(source=entry, devices=(d_npu, d_gpu))
        assert len(ep.devices) == 2
        assert ep.devices[0] is d_npu
        assert ep.devices[1] is d_gpu

    def test_empty_devices_raises_value_error(self) -> None:
        """__post_init__ enforces the invariant: ``len(devices) >= 1``."""
        entry = _make_entry()
        with pytest.raises(ValueError, match="invariant violated"):
            WinMLEP(source=entry, devices=())


class TestWinMLEPDevicesFlatten:
    """WinMLEP.ep_devices() — flatten into (ep, device) pairs."""

    def test_single_device_yields_one_pair(self) -> None:
        entry = _make_entry()
        device = wrap_ort_device(_make_fake_ort_ep_device("OpenVINOExecutionProvider", "NPU"))
        ep = WinMLEP(source=entry, devices=(device,))
        pairs = ep.ep_devices()
        assert len(pairs) == 1
        assert isinstance(pairs[0], WinMLEPDevice)

    def test_multiple_devices_yield_pairs_per_device(self) -> None:
        entry = _make_entry()
        devices = tuple(
            wrap_ort_device(_make_fake_ort_ep_device("OpenVINOExecutionProvider", t))
            for t in ("NPU", "GPU", "CPU")
        )
        ep = WinMLEP(source=entry, devices=devices)
        pairs = ep.ep_devices()
        assert len(pairs) == 3

    def test_each_pair_shares_ep_identity(self) -> None:
        """All WinMLEPDevice.ep references must point at the same WinMLEP."""
        entry = _make_entry()
        devices = tuple(
            wrap_ort_device(_make_fake_ort_ep_device("OpenVINOExecutionProvider", t))
            for t in ("NPU", "GPU")
        )
        ep = WinMLEP(source=entry, devices=devices)
        pairs = ep.ep_devices()
        for pair in pairs:
            assert pair.ep is ep

    def test_each_pair_device_is_one_of_devices(self) -> None:
        """Design invariant: pair.device must be the same object as one of ep.devices.

        Not just equal — identical via ``is``. See
        docs/design/session/3_design_classes.md section 3.6.
        """
        entry = _make_entry()
        d_npu = wrap_ort_device(_make_fake_ort_ep_device("OpenVINOExecutionProvider", "NPU"))
        d_gpu = wrap_ort_device(_make_fake_ort_ep_device("OpenVINOExecutionProvider", "GPU"))
        ep = WinMLEP(source=entry, devices=(d_npu, d_gpu))
        pairs = ep.ep_devices()
        device_ids = {id(d) for d in ep.devices}
        for pair in pairs:
            assert id(pair.device) in device_ids


class TestWinMLEPFrozenness:
    """Frozen dataclass invariants — no field reassignment allowed."""

    def test_cannot_reassign_source(self) -> None:
        entry = _make_entry()
        device = wrap_ort_device(_make_fake_ort_ep_device("OpenVINOExecutionProvider", "NPU"))
        ep = WinMLEP(source=entry, devices=(device,))
        with pytest.raises(FrozenInstanceError):
            ep.source = _make_entry("QNNExecutionProvider")  # type: ignore[misc]

    def test_cannot_reassign_devices(self) -> None:
        entry = _make_entry()
        device = wrap_ort_device(_make_fake_ort_ep_device("OpenVINOExecutionProvider", "NPU"))
        ep = WinMLEP(source=entry, devices=(device,))
        with pytest.raises(FrozenInstanceError):
            ep.devices = ()  # type: ignore[misc]

    def test_winml_ep_device_is_frozen(self) -> None:
        """WinMLEPDevice is frozen too — neither .ep nor .device can be rebound."""
        entry = _make_entry()
        device = wrap_ort_device(_make_fake_ort_ep_device("OpenVINOExecutionProvider", "NPU"))
        ep = WinMLEP(source=entry, devices=(device,))
        pair = WinMLEPDevice(ep=ep, device=device)
        with pytest.raises(FrozenInstanceError):
            pair.ep = ep  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            pair.device = device  # type: ignore[misc]
