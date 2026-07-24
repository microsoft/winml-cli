# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import tests.e2e.require_ep as require_ep_module
from tests.e2e.require_ep import require_ep
from winml.modelkit.session import WinMLEPRegistrationFailed


@pytest.fixture(autouse=True)
def _clear_registered_ep_cache():
    for name in ("_registered_device_types", "_registered_device_types_for_registry"):
        cached = getattr(require_ep_module, name, None)
        if cached is not None and hasattr(cached, "cache_clear"):
            cached.cache_clear()
    yield
    for name in ("_registered_device_types", "_registered_device_types_for_registry"):
        cached = getattr(require_ep_module, name, None)
        if cached is not None and hasattr(cached, "cache_clear"):
            cached.cache_clear()


def test_require_ep_skips_discovered_provider_that_cannot_register() -> None:
    registry = MagicMock()
    entry = SimpleNamespace(ep_name="QNNExecutionProvider")
    registry.available_eps.return_value = frozenset({"QNNExecutionProvider"})
    registry.all_discovered.return_value = (entry,)
    registry.register_ep.side_effect = WinMLEPRegistrationFailed("registration failed")

    with (
        patch(
            "winml.modelkit.session.WinMLEPRegistry.get_instance",
            return_value=registry,
        ),
        pytest.raises(pytest.skip.Exception, match="not available"),
    ):
        require_ep("qnn")


def test_require_ep_skips_provider_without_requested_device_class() -> None:
    registry = MagicMock()
    entry = SimpleNamespace(ep_name="OpenVINOExecutionProvider")
    registry.available_eps.return_value = frozenset({"OpenVINOExecutionProvider"})
    registry.all_discovered.return_value = (entry,)
    registry.register_ep.return_value = SimpleNamespace(
        devices=(SimpleNamespace(device_type="CPU"),)
    )

    with (
        patch(
            "winml.modelkit.session.WinMLEPRegistry.get_instance",
            return_value=registry,
        ),
        pytest.raises(pytest.skip.Exception, match="not available"),
    ):
        require_ep("openvino", device="npu")


def test_require_ep_reprobes_after_registry_replacement() -> None:
    entry = SimpleNamespace(ep_name="QNNExecutionProvider")
    failing_registry = MagicMock()
    failing_registry.all_discovered.return_value = (entry,)
    failing_registry.register_ep.side_effect = WinMLEPRegistrationFailed("registration failed")
    healthy_registry = MagicMock()
    healthy_registry.all_discovered.return_value = (entry,)
    healthy_registry.register_ep.return_value = SimpleNamespace(
        devices=(SimpleNamespace(device_type="NPU"),)
    )

    with (
        patch(
            "winml.modelkit.session.WinMLEPRegistry.get_instance",
            return_value=failing_registry,
        ),
        pytest.raises(pytest.skip.Exception, match="not available"),
    ):
        require_ep("qnn", device="npu")

    with patch(
        "winml.modelkit.session.WinMLEPRegistry.get_instance",
        return_value=healthy_registry,
    ):
        assert require_ep("qnn", device="npu") == "QNNExecutionProvider"

    healthy_registry.register_ep.assert_called_once_with(entry)


def test_require_device_requires_registered_device_class() -> None:
    from tests.e2e.require_ep import require_device

    entry = SimpleNamespace(ep_name="QNNExecutionProvider")
    registry = MagicMock()
    registry.all_discovered.return_value = (entry,)
    registry.register_ep.return_value = SimpleNamespace(
        devices=(SimpleNamespace(device_type="NPU"),)
    )

    with patch(
        "winml.modelkit.session.WinMLEPRegistry.get_instance",
        return_value=registry,
    ):
        require_device("npu")
        with pytest.raises(pytest.skip.Exception, match="on gpu"):
            require_device("gpu")
