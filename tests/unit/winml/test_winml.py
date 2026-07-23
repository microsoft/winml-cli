# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for winml.py WinML singleton EP registration guard."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_winml_instance() -> object:
    """Return a fresh WinML instance with a clean _registered_eps state.

    Bypasses __init__ (which imports winui3 / WinAppSDK) by constructing the
    object directly and injecting the minimum attributes that
    register_execution_providers() needs.
    """
    from winml.modelkit import winml as winml_mod

    # Reset the module-level singleton so we get a genuinely new object.
    winml_mod._winml_instance = None

    # Build a partial instance without running __init__.
    instance = object.__new__(winml_mod.WinML)
    instance._initialized = True  # prevent __init__ from running
    instance._ep_paths = {"QNNExecutionProvider": "C:/fake/qnn.dll"}
    instance._registered_eps = {
        "onnxruntime": [],
        "onnxruntime_genai": [],
    }
    # Stash in the module singleton so WinML() returns the same object.
    winml_mod._winml_instance = instance
    return instance


def test_winml_register_after_external_registration_no_double_load():
    """If another caller already registered the EP DLL, WinML.register_execution_providers
    must NOT call register_execution_provider_library a second time.

    Mirrors test_register_ep_after_external_registration_no_double_register in
    tests/unit/session/test_ep_registry.py (the inverse direction of the same
    dual-singleton crash vector).

    Scenario (HF perf path):
      1. WinMLEPRegistry.register_ep() runs first — QNN is now visible in ORT.
      2. WinML.register_execution_providers() runs second — must detect the
         pre-existing registration and skip the DLL load.
    """
    instance = _make_winml_instance()

    fake_dev = MagicMock()
    fake_dev.ep_name = "QNNExecutionProvider"

    fake_ort = SimpleNamespace(
        get_ep_devices=MagicMock(return_value=[fake_dev]),
        register_execution_provider_library=MagicMock(),
        __name__="onnxruntime",
    )

    with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
        result = instance.register_execution_providers(ort=True, ort_genai=False)

    # Critical: DLL load must NOT be attempted a second time.
    fake_ort.register_execution_provider_library.assert_not_called()
    assert "QNNExecutionProvider" in result["onnxruntime"]


def test_winml_register_no_prior_registration_loads_dll():
    """When no prior registration exists, WinML must call register_execution_provider_library."""
    instance = _make_winml_instance()

    fake_ort = SimpleNamespace(
        get_ep_devices=MagicMock(return_value=[]),  # empty — not yet loaded
        register_execution_provider_library=MagicMock(),
        __name__="onnxruntime",
    )

    with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
        result = instance.register_execution_providers(ort=True, ort_genai=False)

    fake_ort.register_execution_provider_library.assert_called_once_with(
        "QNNExecutionProvider", "C:/fake/qnn.dll"
    )
    assert "QNNExecutionProvider" in result["onnxruntime"]


def test_winml_register_idempotent_on_second_call():
    """A second call to register_execution_providers must skip via _registered_eps guard."""
    instance = _make_winml_instance()

    fake_dev = MagicMock()
    fake_dev.ep_name = "QNNExecutionProvider"

    fake_ort = SimpleNamespace(
        get_ep_devices=MagicMock(return_value=[]),
        register_execution_provider_library=MagicMock(),
        __name__="onnxruntime",
    )

    with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
        instance.register_execution_providers(ort=True, ort_genai=False)
        instance.register_execution_providers(ort=True, ort_genai=False)  # second call

    # DLL load happens exactly once regardless of call count.
    assert fake_ort.register_execution_provider_library.call_count == 1


def test_winml_register_get_ep_devices_failure_attempts_load():
    """If get_ep_devices raises, the guard is conservative and attempts the DLL load."""
    instance = _make_winml_instance()

    def _raise(*_):
        raise OSError("introspection unavailable")

    fake_ort = SimpleNamespace(
        get_ep_devices=_raise,
        register_execution_provider_library=MagicMock(),
        __name__="onnxruntime",
    )

    with patch.dict(sys.modules, {"onnxruntime": fake_ort}):
        result = instance.register_execution_providers(ort=True, ort_genai=False)

    # Conservative fallback: must still attempt DLL load.
    fake_ort.register_execution_provider_library.assert_called_once_with(
        "QNNExecutionProvider", "C:/fake/qnn.dll"
    )
    assert "QNNExecutionProvider" in result["onnxruntime"]


def test_get_registered_ep_devices_registers_before_enumerating():
    """The compatibility helper loads plugin EPs before querying ORT devices."""
    from winml.modelkit import winml as winml_mod

    events: list[str] = []
    fake_device = object()
    winml_mod.get_registered_ep_devices.cache_clear()

    with (
        patch.object(
            winml_mod,
            "register_execution_providers",
            side_effect=lambda **_kwargs: events.append("register") or {},
        ) as mock_register,
        patch(
            "winml.modelkit.session.WinMLEPRegistry.instance",
            return_value=MagicMock(),
        ),
        patch(
            "onnxruntime.get_ep_devices",
            side_effect=lambda: events.append("enumerate") or [fake_device],
        ),
    ):
        try:
            result = winml_mod.get_registered_ep_devices()
        finally:
            winml_mod.get_registered_ep_devices.cache_clear()

    mock_register.assert_called_once_with(ort=True)
    assert events == ["register", "enumerate"]
    assert result == (fake_device,)
