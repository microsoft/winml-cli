# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ep_registry module-level helpers and WinMLEPRegistry.register_ep."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from winml.modelkit.session import EPNotDiscovered, EPRegistrationFailed
from winml.modelkit.session.ep_registry import WinMLEPRegistry, ensure_initialized

from .conftest import QNN_VENDOR_ID


@pytest.fixture
def fresh_registry() -> WinMLEPRegistry:
    """Singleton with stubbed catalog + cleared registered list."""
    reg = WinMLEPRegistry.get_instance()
    reg._ep_paths = {"QNNExecutionProvider": "C:/fake/qnn.dll"}
    reg._registered_eps = []
    return reg


def _fake_ep_device(ep_name: str, dev_type: str) -> MagicMock:
    """Build a MagicMock matching the OrtEpDevice shape used downstream."""
    d = MagicMock()
    d.ep_name = ep_name
    d.device.type.name = dev_type
    d.device.vendor_id = QNN_VENDOR_ID
    d.device.device_id = 0x0001
    return d


def test_ensure_initialized_calls_registry_once():
    """ensure_initialized() calls register_to_ort() via singleton; idempotent across calls.

    A2-I3 (PR review): the previous loose ``call_count >= 1`` assertion would
    pass if the wrapper accidentally amplified calls (e.g., re-instantiating
    the registry on every entry). Pin the contract:

    * ``WinMLEPRegistry.get_instance()`` is hit exactly once per
      ``ensure_initialized()`` call (no extra allocations).
    * ``register_to_ort()`` is invoked once per call — the singleton's
      internal ``_registered_eps`` skip-list provides actual no-op
      idempotency, NOT the wrapper.
    * No exception is raised for any number of calls.
    """
    with patch("winml.modelkit.session.ep_registry.WinMLEPRegistry") as mock_registry_cls:
        instance = mock_registry_cls.get_instance.return_value
        instance.winml_available = True

        ensure_initialized()
        ensure_initialized()
        ensure_initialized()

        # Wrapper makes exactly one get_instance + one register_to_ort per call.
        assert mock_registry_cls.get_instance.call_count == 3
        assert instance.register_to_ort.call_count == 3


def test_ensure_initialized_failure_logs_warning(caplog):
    """NFR-2: registration failure must log at WARNING (not DEBUG) with exception class.

    The previous DEBUG-level swallow downgraded real environmental failures
    (broken Windows App SDK, etc.) to invisible "feature unavailable".
    """
    with patch("winml.modelkit.session.ep_registry.WinMLEPRegistry") as mock_registry_cls:
        instance = mock_registry_cls.get_instance.return_value
        instance.winml_available = True
        instance.register_to_ort.side_effect = RuntimeError("boom")

        with caplog.at_level(logging.WARNING):
            ensure_initialized()  # must NOT raise

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any(
            "WinML EP registration failed" in r.message and "RuntimeError" in r.message
            for r in warnings
        ), f"expected WARNING surfacing RuntimeError, got: {[r.message for r in warnings]}"


def test_ensure_initialized_allows_retry_after_failure(caplog):
    """A first-call failure does not latch — the next call retries registration."""
    with patch("winml.modelkit.session.ep_registry.WinMLEPRegistry") as mock_registry_cls:
        instance = mock_registry_cls.get_instance.return_value
        instance.winml_available = True
        instance.register_to_ort.side_effect = [RuntimeError("transient"), None]

        with caplog.at_level(logging.WARNING):
            ensure_initialized()  # fails
            ensure_initialized()  # should retry

        # register_to_ort should have been called both times.
        assert instance.register_to_ort.call_count == 2


def test_register_to_ort_failure_records_per_ep_state():
    """NFR-2: per-EP registration failures must be tracked in registration_failures."""
    from winml.modelkit.session.ep_registry import WinMLEPRegistry

    # Reset the singleton's failure dict for test isolation by using
    # get_instance + manipulating instance state directly.
    registry = WinMLEPRegistry.get_instance()
    # Inject test EP paths and force registration failure
    registry._ep_paths = {"FakeEP": "C:/nonexistent/fake.dll"}
    registry._registered_eps = []
    registry._registration_failures = {}
    registry._winml_available = True

    fake_ort = type("M", (), {})()

    def _bad_register(name, path):
        raise RuntimeError(f"cannot load {path}")

    fake_ort.register_execution_provider_library = _bad_register

    with patch.dict("sys.modules", {"onnxruntime": fake_ort}):
        registry.register_to_ort()

    assert "FakeEP" in registry.registration_failures
    assert "RuntimeError" in registry.registration_failures["FakeEP"]
    # Property returns a copy — mutating it must not corrupt internal state.
    snap = registry.registration_failures
    snap.clear()
    assert "FakeEP" in registry.registration_failures


def test_register_ep_happy_path(fresh_registry: WinMLEPRegistry) -> None:
    """register_ep returns only the devices matching the requested EP name.

    get_ep_devices is called twice:
      1. The defensive check (before DLL load) — must return empty so the DLL
         load path is taken rather than the skip-if-already-loaded path.
      2. The final return value — returns all matching devices.
    """
    qnn_devs = [
        _fake_ep_device("QNNExecutionProvider", "NPU"),
        _fake_ep_device("QNNExecutionProvider", "GPU"),
        _fake_ep_device("QNNExecutionProvider", "GPU"),
        _fake_ep_device("QNNExecutionProvider", "CPU"),
    ]
    other = _fake_ep_device("CPUExecutionProvider", "CPU")
    all_devs = [*qnn_devs, other]
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        # First call: defensive pre-check (empty → not yet loaded → proceed with DLL load).
        # Second call: final return value filtering.
        mock_ort.get_ep_devices.side_effect = [[], all_devs]
        mock_ort.register_execution_provider_library = MagicMock()
        result = fresh_registry.register_ep("QNNExecutionProvider")
    mock_ort.register_execution_provider_library.assert_called_once_with(
        "QNNExecutionProvider", "C:/fake/qnn.dll"
    )
    assert result == qnn_devs


def test_register_ep_unknown_raises(fresh_registry: WinMLEPRegistry) -> None:
    """register_ep raises EPNotDiscovered for an EP not in the catalog."""
    with pytest.raises(EPNotDiscovered):
        fresh_registry.register_ep("MysteryExecutionProvider")


def test_register_ep_idempotent(fresh_registry: WinMLEPRegistry) -> None:
    """register_ep skips DLL loading on a second call for the same EP.

    Call sequence for get_ep_devices:
      1st call (first register_ep, defensive pre-check): empty → DLL load proceeds.
      2nd call (first register_ep, return value): [qnn].
      3rd call (second register_ep, return value — ep already in _registered_eps): [qnn].
    """
    qnn = _fake_ep_device("QNNExecutionProvider", "NPU")
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.get_ep_devices.side_effect = [[], [qnn], [qnn]]
        mock_ort.register_execution_provider_library = MagicMock()
        fresh_registry.register_ep("QNNExecutionProvider")
        fresh_registry.register_ep("QNNExecutionProvider")
    assert mock_ort.register_execution_provider_library.call_count == 1


def test_register_ep_failure_wraps(fresh_registry: WinMLEPRegistry) -> None:
    """register_ep raises EPRegistrationFailed when ORT's register call raises."""
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.register_execution_provider_library.side_effect = RuntimeError("dll boom")
        mock_ort.get_ep_devices.return_value = []
        with pytest.raises(EPRegistrationFailed):
            fresh_registry.register_ep("QNNExecutionProvider")


def test_register_ep_skips_if_already_loaded(fresh_registry: WinMLEPRegistry) -> None:
    """register_ep skips DLL load if ORT already sees the EP (e.g. loaded by winml.py).

    Defensive check: ort.get_ep_devices() is consulted first; if the EP is
    already visible, register_execution_provider_library must NOT be called.
    The EP is still recorded in _registered_eps so subsequent calls are
    short-circuited by the existing idempotency guard.
    """
    fake_dev = _fake_ep_device("QNNExecutionProvider", "NPU")
    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.get_ep_devices.return_value = [fake_dev]  # already loaded externally
        mock_ort.register_execution_provider_library = MagicMock()

        result = fresh_registry.register_ep("QNNExecutionProvider")

    # Critical: DLL register must NOT be called a second time.
    mock_ort.register_execution_provider_library.assert_not_called()
    assert "QNNExecutionProvider" in fresh_registry._registered_eps
    assert result == [fake_dev]


def test_register_ep_after_external_registration_no_double_register() -> None:
    """Simulates the dual-singleton crash: winml.py registers first, then WinMLEPRegistry.

    Guards against `winml perf -m microsoft/resnet-50 --ep qnn --device npu`
    exiting 127 because ort.register_execution_provider_library is NOT idempotent
    — a second registration of the same DLL causes a native exit(127) with no
    Python traceback.
    """
    fake_dev = MagicMock()
    fake_dev.ep_name = "QNNExecutionProvider"

    with patch("winml.modelkit.session.ep_registry.ort") as mock_ort:
        mock_ort.get_ep_devices.return_value = [fake_dev]  # already loaded by WinML singleton

        registry = WinMLEPRegistry.get_instance()
        registry._ep_paths["QNNExecutionProvider"] = "C:/fake/qnn.dll"
        registry._registered_eps = []  # WinMLEPRegistry has NOT seen this registration

        result = registry.register_ep("QNNExecutionProvider")

        # Critical assertion: DLL register MUST NOT be called.
        mock_ort.register_execution_provider_library.assert_not_called()
        assert "QNNExecutionProvider" in registry._registered_eps
        assert result == [fake_dev]
