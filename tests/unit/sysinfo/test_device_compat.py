# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Backward-compatibility tests for deprecated sysinfo device helpers."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest


def _patch_registered_ep_map(monkeypatch: pytest.MonkeyPatch) -> None:
    from winml.modelkit.sysinfo import device as device_mod

    monkeypatch.setattr(
        device_mod,
        "_get_device_ep_map_from_ort",
        lambda: {
            "npu": ("QNNExecutionProvider",),
            "gpu": ("DmlExecutionProvider",),
            "cpu": ("CPUExecutionProvider",),
        },
    )


def test_public_compat_exports_importable() -> None:
    from winml.modelkit import sysinfo
    from winml.modelkit.sysinfo import (
        get_device_ep_map,
        get_ep_device_map,
        resolve_check_device_ep,
        resolve_device,
        resolve_eps,
    )

    for name, symbol in {
        "get_device_ep_map": get_device_ep_map,
        "get_ep_device_map": get_ep_device_map,
        "resolve_check_device_ep": resolve_check_device_ep,
        "resolve_device": resolve_device,
        "resolve_eps": resolve_eps,
    }.items():
        assert getattr(sysinfo, name) is symbol
        assert name in sysinfo.__all__


def test_public_compat_mapping_helpers_return_legacy_shapes() -> None:
    from winml.modelkit.sysinfo import get_device_ep_map, get_ep_device_map

    with pytest.warns(DeprecationWarning, match="sysinfo.get_ep_device_map"):
        ep_device_map = get_ep_device_map()
    with pytest.warns(DeprecationWarning, match="sysinfo.get_device_ep_map"):
        device_ep_map = get_device_ep_map()

    assert isinstance(ep_device_map["QNNExecutionProvider"], str)
    assert "npu" in ep_device_map["QNNExecutionProvider"].split("/")
    assert "QNNExecutionProvider" in device_ep_map["npu"]

    ep_device_map["InjectedExecutionProvider"] = "cpu"
    device_ep_map["npu"].append("InjectedExecutionProvider")

    with pytest.warns(DeprecationWarning):
        assert "InjectedExecutionProvider" not in get_ep_device_map()
    with pytest.warns(DeprecationWarning):
        assert "InjectedExecutionProvider" not in get_device_ep_map()["npu"]


def test_public_compat_resolve_device_and_eps(monkeypatch: pytest.MonkeyPatch) -> None:
    from winml.modelkit.sysinfo import resolve_device, resolve_eps

    _patch_registered_ep_map(monkeypatch)

    with pytest.warns(DeprecationWarning, match="sysinfo.resolve_device"):
        assert resolve_device("auto", ep=None) == ("npu", ["npu", "gpu", "cpu"])
    with pytest.warns(DeprecationWarning, match="sysinfo.resolve_device"):
        assert resolve_device("auto", ep="qnn") == ("npu", ["npu"])
    with pytest.warns(DeprecationWarning, match="sysinfo.resolve_eps"):
        assert resolve_eps("NPU") == ["QNNExecutionProvider"]


def test_public_compat_resolve_check_device_ep(monkeypatch: pytest.MonkeyPatch) -> None:
    from winml.modelkit.sysinfo import resolve_check_device_ep
    from winml.modelkit.utils.constants import EP_SUPPORTED_DEVICES

    _patch_registered_ep_map(monkeypatch)

    with pytest.warns(DeprecationWarning, match="sysinfo.resolve_check_device_ep"):
        resolved_device, supported_devices, available_eps = resolve_check_device_ep(
            device="auto", ep="qnn"
        )

    assert resolved_device == "npu"
    assert supported_devices == list(EP_SUPPORTED_DEVICES["QNNExecutionProvider"])
    assert available_eps == ["QNNExecutionProvider"]

    with pytest.warns(DeprecationWarning, match="sysinfo.resolve_check_device_ep"):
        upper_auto_device, upper_auto_supported, upper_auto_eps = resolve_check_device_ep(
            device="AUTO", ep="qnn"
        )

    assert upper_auto_device == "npu"
    assert upper_auto_supported == list(EP_SUPPORTED_DEVICES["QNNExecutionProvider"])
    assert upper_auto_eps == ["QNNExecutionProvider"]

    monkeypatch.setattr(
        "winml.modelkit.sysinfo.device._get_device_ep_map_from_ort",
        lambda: {"cpu": ("CPUExecutionProvider",)},
    )
    with pytest.warns(DeprecationWarning, match="sysinfo.resolve_check_device_ep"):
        static_device, static_supported, static_eps = resolve_check_device_ep(
            device="npu", ep="qnn"
        )

    assert static_device == "npu"
    assert static_supported == list(EP_SUPPORTED_DEVICES["QNNExecutionProvider"])
    assert static_eps == ["QNNExecutionProvider"]

    with (
        pytest.warns(DeprecationWarning, match="sysinfo.resolve_check_device_ep"),
        pytest.raises(ValueError, match="does not support device"),
    ):
        resolve_check_device_ep(device="cpu", ep="qnn")


def _capture_deprecation(call):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        result = call()
    return result, [warning for warning in caught if warning.category is DeprecationWarning]


@pytest.mark.parametrize(
    ("name", "call_factory"),
    [
        ("get_ep_device_map", lambda sysinfo: sysinfo.get_ep_device_map),
        ("get_device_ep_map", lambda sysinfo: sysinfo.get_device_ep_map),
        ("resolve_device", lambda sysinfo: lambda: sysinfo.resolve_device("auto", ep="qnn")),
        ("resolve_eps", lambda sysinfo: lambda: sysinfo.resolve_eps("npu")),
    ],
)
def test_public_compat_deprecation_warning_points_to_caller(
    monkeypatch: pytest.MonkeyPatch, name: str, call_factory
) -> None:
    from winml.modelkit import sysinfo

    _patch_registered_ep_map(monkeypatch)

    _result, warning_records = _capture_deprecation(call_factory(sysinfo))

    assert len(warning_records) == 1
    warning = warning_records[0]
    assert f"sysinfo.{name}" in str(warning.message)
    assert Path(warning.filename) == Path(__file__)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"device": "auto", "ep": "qnn"}, ("npu", ["QNNExecutionProvider"])),
        ({"device": "auto", "ep": None}, ("npu", ["QNNExecutionProvider"])),
    ],
)
def test_resolve_check_device_ep_emits_single_caller_warning(
    monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, str | None], expected
) -> None:
    from winml.modelkit import sysinfo

    _patch_registered_ep_map(monkeypatch)

    result, warning_records = _capture_deprecation(
        lambda: sysinfo.resolve_check_device_ep(**kwargs)
    )

    resolved_device, _supported_devices, available_eps = result
    assert (resolved_device, available_eps) == expected
    assert len(warning_records) == 1
    warning = warning_records[0]
    assert "sysinfo.resolve_check_device_ep" in str(warning.message)
    assert Path(warning.filename) == Path(__file__)
