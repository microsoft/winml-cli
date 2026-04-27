# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for ep_registry module-level helpers."""

from __future__ import annotations

import logging
from unittest.mock import patch

from winml.modelkit.session.ep_registry import ensure_initialized


def test_ensure_initialized_calls_registry_once():
    """ensure_initialized() calls register_to_ort() via singleton; idempotent across calls."""
    with patch("winml.modelkit.session.ep_registry.WinMLEPRegistry") as mock_registry_cls:
        instance = mock_registry_cls.get_instance.return_value
        instance.winml_available = True

        ensure_initialized()
        ensure_initialized()
        ensure_initialized()

        assert mock_registry_cls.get_instance.call_count >= 1
        # Multiple calls must not raise


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
