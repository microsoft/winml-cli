# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Explicit physical-GPU coverage for shared RDP CI agents; never product policy."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from winml.modelkit.session import WinMLEPDevice


logger = logging.getLogger(__name__)


def physical_dml_test_mode() -> bool:
    return os.environ.get("WINML_E2E_PIN_SINGLE_DML_GPU") == "1"


def physical_dml_test_luid(selected: WinMLEPDevice | None = None) -> str | None:
    """Require one real DML GPU; tolerate only its same-hardware RDP duplicates.

    Other hosts retain the strict all-adapter tests. In this explicitly enabled
    CI mode, missing hardware, missing LUIDs, and unrelated extra devices still
    fail instead of making the test loop empty or silently dropping coverage.
    """
    if not physical_dml_test_mode():
        return None

    from winml.modelkit.session import EPDeviceTarget, WinMLEPRegistry
    from winml.modelkit.sysinfo import enumerate_compute_adapters, get_ep_device_luid

    native = [adapter for adapter in enumerate_compute_adapters() if adapter.device_type == "GPU"]
    assert len(native) == 1, f"Physical DML test mode requires exactly one DXCore GPU: {native}"
    physical = native[0]
    if selected is None:
        selected = WinMLEPRegistry.instance().auto_device(EPDeviceTarget(ep="dml", device="gpu"))
    devices = [device for device in selected.ep.devices if device.device_type == "GPU"]
    luids = [get_ep_device_luid(device.ort_handle) for device in devices]
    assert luids and None not in luids, "DML must publish an adapter LUID"
    assert physical.luid in luids, f"Physical GPU {physical.luid} missing from DML: {luids}"
    for device in devices:
        if get_ep_device_luid(device.ort_handle) == physical.luid:
            continue
        hardware = device.ort_handle.device
        assert (
            hardware.vendor_id == physical.vendor_id
            and hardware.device_id == physical.device_id
            and hardware.metadata.get("Description") == physical.name
        ), f"Unexpected nonphysical DML adapter: {hardware.metadata}"
    logger.warning(
        "Shared RDP test mode: exercising physical DML GPU %s; extra same-hardware LUIDs: %s",
        physical.luid,
        sorted(set(luids) - {physical.luid}),
    )
    return physical.luid


def perf_test_luid(ep: str | None, device: str | None) -> str | None:
    """Pin GPU perf tests on the shared agent, preserving the requested EP."""
    if not physical_dml_test_mode():
        return None
    # Do not resolve unrelated CPU/NPU/GenAI contract tests before invoking
    # the CLI: some intentionally pass unsupported targets to test errors.
    if device not in (None, "auto", "gpu"):
        return None
    if ep in (None, "auto") and device != "gpu":
        return None
    from winml.modelkit.session import EPDeviceTarget, resolve_device

    target = resolve_device(EPDeviceTarget(ep=ep or "auto", device=device or "auto"))
    if target.device != "gpu":
        return None
    # Device-only GPU tests use the same physical adapter even when the
    # normal EP policy chooses OpenVINO. The CLI validates EP support.
    return physical_dml_test_luid()
