# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Compiler-aware device binding with scoped provider registration."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from .ep_device import WinMLDevice, expand_ep_name
from .ep_registry import WinMLEP, WinMLEPRegistry


if TYPE_CHECKING:
    from collections.abc import Iterator

    from ..compiler import EPConfig
    from .ep_device import EPDeviceTarget
    from .ep_registry import WinMLEPDevice


@contextmanager
def bind_compilation_device(
    target: EPDeviceTarget,
    *,
    ep_config: EPConfig,
) -> Iterator[WinMLEPDevice]:
    """Bind a resolved runtime target or an explicit off-device deployment intent."""
    if ep_config.compiler != "offdevice":
        yield WinMLEPRegistry.instance().auto_device(target)
        return

    if target.ep == "auto":
        raise ValueError("Off-device compilation requires an explicit execution provider.")
    if target.device not in {"auto", "npu"}:
        raise ValueError("Off-device compilation requires an NPU deployment target.")

    from ..compiler.offdevice import _registered_compilation_devices

    provider = expand_ep_name(target.ep)
    with _registered_compilation_devices(provider, target.source) as (
        entry,
        devices,
        registration_name,
    ):
        matches = [
            device
            for device in devices
            if provider != "OpenVINOExecutionProvider"
            or device.ep_metadata["npu_platform"] == ep_config.target_platform
        ]
        try:
            if len(matches) != 1:
                if provider == "OpenVINOExecutionProvider":
                    available = ", ".join(
                        sorted({device.ep_metadata["npu_platform"] for device in devices})
                    )
                    raise ValueError(
                        f"Target {ep_config.target_platform!r} must match exactly one virtual "
                        f"NPU device (found {len(matches)}). Available platforms: {available}."
                    )
                host = (
                    "CPU compilation host"
                    if provider == "QNNExecutionProvider"
                    else "virtual NPU"
                )
                raise ValueError(
                    f"{provider} requires exactly one {host} (found {len(matches)})."
                )

            registration = WinMLEP(
                source=entry,
                devices=(WinMLDevice(matches[0]),),
                arg0=registration_name,
            )
            yield registration.ep_devices()[0]
        finally:
            matches.clear()
