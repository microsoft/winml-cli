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

import pytest


def require_ep(ep: str) -> str:
    """Skip the current test unless the requested EP is available.

    Args:
        ep: EP name. CLI alias (``"qnn"``) or full ORT provider name
            (``"QNNExecutionProvider"``); both accepted.

    Returns:
        The full ORT provider name (e.g. ``"QNNExecutionProvider"``) that
        satisfied the requirement. Handy when the caller wants to pass it
        on to ``onnxruntime`` / ``WinMLSession``.

    Raises:
        pytest.skip.Exception: When ``ep`` is unknown or the corresponding
            provider is not available on the host.
    """
    from winml.modelkit.session import WinMLEPRegistry
    from winml.modelkit.utils import normalize_ep_name

    provider = normalize_ep_name(ep)
    if provider is None:
        pytest.skip(f"Unknown EP: {ep!r}")

    # CPU is always available via ORT itself but is not enumerated by
    # WinMLEPRegistry, which only tracks WinML-discoverable backend EPs
    # (QNN, OpenVINO, DML, ...). Short-circuit before consulting the
    # registry so `require_ep("cpu")` doesn't spuriously skip.
    if provider == "CPUExecutionProvider":
        return provider

    # Singleton — first call probes; subsequent calls are free.
    if not WinMLEPRegistry.get_instance().is_ep_available(provider):
        pytest.skip(f"EP not available on this host: {provider}")

    return provider


def require_not_ep(ep: str) -> None:
    """Skip the current test unless the requested EP is NOT available.

    Mirror of :func:`require_ep` for tests that exercise the
    "EP not registered" rejection path (e.g. ``winml compile --ep qnn``
    on a host without QNN).
    """
    from winml.modelkit.session import WinMLEPRegistry
    from winml.modelkit.utils import normalize_ep_name

    provider = normalize_ep_name(ep)
    if provider is None:
        pytest.skip(f"Unknown EP: {ep!r}")

    if provider == "CPUExecutionProvider":
        pytest.skip("CPU is always available on this host")

    if WinMLEPRegistry.get_instance().is_ep_available(provider):
        pytest.skip(f"EP is available on this host (test requires it absent): {provider}")
