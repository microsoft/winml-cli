# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression pin for P1-D — WinMLEPRegistry must propagate ORT init failures.

Previous behaviour silently synthesised an empty registry
(``provider_names = frozenset()``, ``ep_devices = []``) when
``ort.get_available_providers()`` or ``ort.get_ep_devices()`` raised.
Downstream ``auto_detect_device`` / ``auto_device`` then returned
misleading empty results, hiding the actual ORT problem.

The fix propagates the underlying exception as
:class:`WinMLEPRegistrationFailed` at construction time so the failure
surfaces at first ``WinMLEPRegistry.instance()``.
"""

from __future__ import annotations

import pytest

from winml.modelkit.session import WinMLEPRegistrationFailed
from winml.modelkit.session.ep_registry import WinMLEPRegistry


def test_ort_get_available_providers_failure_propagates(monkeypatch):
    """Simulated ORT failure at get_available_providers must raise."""
    import onnxruntime as ort

    def boom():
        raise RuntimeError("simulated ORT init failure")

    monkeypatch.setattr(ort, "get_available_providers", boom)

    with pytest.raises(WinMLEPRegistrationFailed) as excinfo:
        WinMLEPRegistry()

    assert "ORT init failed" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert "simulated ORT init failure" in str(excinfo.value.__cause__)


def test_ort_get_ep_devices_failure_propagates(monkeypatch):
    """Simulated ORT failure at get_ep_devices must raise."""
    import onnxruntime as ort

    def boom():
        raise RuntimeError("get_ep_devices exploded")

    monkeypatch.setattr(ort, "get_ep_devices", boom)

    with pytest.raises(WinMLEPRegistrationFailed) as excinfo:
        WinMLEPRegistry()

    assert "ORT init failed" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_healthy_ort_still_constructs():
    """Baseline sanity: without a mocked failure, construction succeeds."""
    # Should not raise. Registry may or may not discover EPs — that depends
    # on the host — but the constructor itself must not fail.
    WinMLEPRegistry()
