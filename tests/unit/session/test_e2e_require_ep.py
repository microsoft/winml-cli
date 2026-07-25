# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import tests.e2e.require_ep as require_ep_module
from tests.e2e.require_ep import require_ep
from winml.modelkit.ep_path import DirectorySource, EPEntry
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


def _plugin_entry(ep_name: str) -> EPEntry:
    return EPEntry(
        ep_name=ep_name,
        dll_path=Path(rf"C:\fake\{ep_name}.dll"),
        source=DirectorySource(
            root=Path(r"C:\fake"),
            dll_patterns={ep_name: f"{ep_name}.dll"},
        ),
    )


@contextmanager
def _isolated_probe(device_types: tuple[str, ...]):
    yield {"devices": [{"device_type": device_type} for device_type in device_types]}


def test_require_ep_skips_discovered_provider_that_cannot_register() -> None:
    registry = MagicMock()
    entry = _plugin_entry("QNNExecutionProvider")
    registry.all_discovered.return_value = (entry,)

    with (
        patch(
            "winml.modelkit.session.WinMLEPRegistry.get_instance",
            return_value=registry,
        ),
        patch(
            "winml.modelkit.commands.sys.isolated_ep_register",
            side_effect=WinMLEPRegistrationFailed("registration failed"),
        ),
        pytest.raises(pytest.skip.Exception, match="not available"),
    ):
        require_ep("qnn")


def test_require_ep_skips_provider_without_requested_device_class() -> None:
    registry = MagicMock()
    entry = _plugin_entry("OpenVINOExecutionProvider")
    registry.all_discovered.return_value = (entry,)

    with (
        patch(
            "winml.modelkit.session.WinMLEPRegistry.get_instance",
            return_value=registry,
        ),
        patch(
            "winml.modelkit.commands.sys.isolated_ep_register",
            return_value=_isolated_probe(("CPU",)),
        ),
        pytest.raises(pytest.skip.Exception, match="not available"),
    ):
        require_ep("openvino", device="npu")


def test_require_ep_reprobes_after_registry_replacement() -> None:
    entry = _plugin_entry("QNNExecutionProvider")
    failing_registry = MagicMock()
    failing_registry.all_discovered.return_value = (entry,)
    healthy_registry = MagicMock()
    healthy_registry.all_discovered.return_value = (entry,)

    with patch(
        "winml.modelkit.commands.sys.isolated_ep_register",
        side_effect=[
            WinMLEPRegistrationFailed("registration failed"),
            _isolated_probe(("NPU",)),
        ],
    ) as mock_probe:
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

    assert mock_probe.call_count == 2


@pytest.mark.parametrize(
    ("probe_device_types", "expect_skip"),
    [
        (("NPU",), False),
        (("CPU",), True),
    ],
    ids=["available", "unavailable-device-mismatch"],
)
def test_require_ep_plugin_probe_leaves_registry_state_unchanged(
    probe_device_types: tuple[str, ...], expect_skip: bool
) -> None:
    class FakeRegistry:
        pass

    entry = EPEntry(
        ep_name="OpenVINOExecutionProvider",
        dll_path=Path(r"C:\fake\openvino.dll"),
        source=DirectorySource(
            root=Path(r"C:\fake"),
            dll_patterns={"OpenVINOExecutionProvider": "openvino.dll"},
        ),
    )
    initial_registered = {Path(r"C:\fake\existing.dll"): object()}
    initial_registration_count = {"ExistingExecutionProvider": 2}
    registered_ep = SimpleNamespace(
        source=entry,
        arg0=entry.ep_name,
        devices=tuple(
            SimpleNamespace(device_type=device_type) for device_type in probe_device_types
        ),
    )
    registry = FakeRegistry()
    registry._registered = dict(initial_registered)
    registry._registration_count = dict(initial_registration_count)
    registry.all_discovered = MagicMock(return_value=(entry,))

    def _register_ep(discovered_entry: EPEntry) -> SimpleNamespace:
        registry._registered[discovered_entry.dll_path] = registered_ep
        registry._registration_count[discovered_entry.ep_name] = (
            registry._registration_count.get(discovered_entry.ep_name, 0) + 1
        )
        return registered_ep

    def _unregister_ep(winml_ep: SimpleNamespace) -> None:
        registry._registered.pop(winml_ep.source.dll_path, None)

    @contextmanager
    def _isolated_probe():
        yield {"devices": [{"device_type": device_type} for device_type in probe_device_types]}

    registry.register_ep = MagicMock(side_effect=_register_ep)
    registry.unregister_ep = MagicMock(side_effect=_unregister_ep)

    with (
        patch(
            "winml.modelkit.session.WinMLEPRegistry.get_instance",
            return_value=registry,
        ),
        patch(
            "winml.modelkit.commands.sys.isolated_ep_register",
            return_value=_isolated_probe(),
        ),
    ):
        if expect_skip:
            with pytest.raises(pytest.skip.Exception, match="not available"):
                require_ep("openvino", device="npu")
        else:
            assert require_ep("openvino", device="npu") == entry.ep_name

    assert registry._registered == initial_registered
    assert registry._registration_count == initial_registration_count


def test_require_device_requires_registered_device_class() -> None:
    from tests.e2e.require_ep import require_device

    entry = _plugin_entry("QNNExecutionProvider")
    registry = MagicMock()
    registry.all_discovered.return_value = (entry,)

    with (
        patch(
            "winml.modelkit.session.WinMLEPRegistry.get_instance",
            return_value=registry,
        ),
        patch(
            "winml.modelkit.commands.sys.isolated_ep_register",
            return_value=_isolated_probe(("NPU",)),
        ),
    ):
        require_device("npu")
        with pytest.raises(pytest.skip.Exception, match="on gpu"):
            require_device("gpu")
