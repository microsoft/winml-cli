# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Runtime EP-availability gate for e2e tests.

Single supported usage — call :func:`require_ep` at the top of a test body
with one EP name. Skips the test when the EP is not available on the host.

The accepted name follows the same convention as the CLI ``--ep`` flag:
short alias (``"qnn"``, ``"openvino"``, ``"ov"``, ``"dml"``, ...) or the
full ORT provider name (``"QNNExecutionProvider"``). Resolution is done
by :func:`winml.modelkit.utils.constants.normalize_ep_name`, so no local
mapping lives here.

Single test::

    def test_openvino_specific():
        require_ep("openvino")
        ...

Parametrized fan-out — follow the codebase convention of flat
``@pytest.mark.parametrize`` and gate inside the body::

    @pytest.mark.parametrize("ep", ["qnn", "openvino", "dml", "cpu"])
    def test_compile_happy_path(ep):
        require_ep(ep)
        ...
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from winml.modelkit.session import WinMLEPRegistry


def _registered_device_types(provider: str) -> frozenset[str]:
    """Return device classes exposed by successfully registered EP sources."""
    from winml.modelkit.session import WinMLEPRegistry

    registry = WinMLEPRegistry.get_instance()
    return _registered_device_types_for_registry(provider, registry)


@cache
def _registered_device_types_for_registry(
    provider: str, registry: WinMLEPRegistry
) -> frozenset[str]:
    """Return cached device classes for one registry instance."""
    from winml.modelkit.session import WinMLEPRegistrationFailed

    device_types: set[str] = set()
    for entry in registry.all_discovered():
        if entry.ep_name != provider:
            continue
        try:
            registered = registry.register_ep(entry)
        except WinMLEPRegistrationFailed:
            continue
        device_types.update(device.device_type.lower() for device in registered.devices)
    return frozenset(device_types)


def require_ep(ep: str, *, device: str | None = None) -> str:
    """Skip the current test unless the requested EP is available.

    Args:
        ep: EP name. CLI alias (``"qnn"``) or full ORT provider name
            (``"QNNExecutionProvider"``); both accepted.
        device: Optional required device class (``"cpu"``, ``"gpu"``,
            or ``"npu"``).

    Returns:
        The full ORT provider name (e.g. ``"QNNExecutionProvider"``) that
        satisfied the requirement. Handy when the caller wants to pass it
        on to ``onnxruntime`` / ``WinMLSession``.

    Raises:
        pytest.skip.Exception: When ``ep`` is unknown or the corresponding
            provider is not available on the host.
    """
    from winml.modelkit.utils import normalize_ep_name

    provider = normalize_ep_name(ep)
    if provider is None:
        pytest.skip(f"Unknown EP: {ep!r}")

    device_types = _registered_device_types(provider)
    required_device = device.lower() if device is not None else None
    if not device_types or (required_device is not None and required_device not in device_types):
        suffix = f" on {required_device}" if required_device is not None else ""
        pytest.skip(f"EP not available on this host{suffix}: {provider}")

    return provider


def require_device(device: str) -> None:
    """Skip the current test unless any registered EP exposes ``device``."""
    from winml.modelkit.session import WinMLEPRegistry

    required_device = device.lower()
    providers = {entry.ep_name for entry in WinMLEPRegistry.get_instance().all_discovered()}
    if not any(required_device in _registered_device_types(provider) for provider in providers):
        pytest.skip(f"No registered EP is available on {required_device}")


def require_not_ep(ep: str) -> None:
    """Skip the current test unless the requested EP is NOT available.

    Mirror of :func:`require_ep` for tests that exercise the
    "EP not registered" rejection path (e.g. ``winml compile --ep qnn``
    on a host without QNN).
    """
    from winml.modelkit.utils import normalize_ep_name

    provider = normalize_ep_name(ep)
    if provider is None:
        pytest.skip(f"Unknown EP: {ep!r}")

    if _registered_device_types(provider):
        pytest.skip(f"EP is available on this host (test requires it absent): {provider}")


def is_host(ep: str) -> bool:
    """Return True iff ``ep`` is available on this host.

    Non-skipping probe used to gate assertions whose tolerance depends on
    the active EP (e.g. only enforce a metric-magnitude bound on QNN,
    where quantization preserves accuracy, while still running the rest
    of the test on every EP for pipeline-regression coverage).
    """
    from winml.modelkit.utils import normalize_ep_name

    provider = normalize_ep_name(ep)
    if provider is None:
        return False
    return bool(_registered_device_types(provider))
