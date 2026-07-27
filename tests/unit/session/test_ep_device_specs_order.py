# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression pin for P1-E — EP_DEVICE_SPECS preference-order lock.

The catalog order in :data:`EP_DEVICE_SPECS` encodes *preference among
installed EPs*, not unconditional defaults. Two invariants matter:

1. Plugin (vendor-optimal) EPs come BEFORE built-ins in every device
   group — never let DML win GPU when a vendor plugin is installed.
2. Per-vendor `default_ep_for_device()` returns the expected first
   catalog match once availability + compatibility are simulated.

These tests do NOT touch the catalog; they only pin the observable
result of the walk. A change to the order (or a new EP being inserted)
will fail here loudly, forcing a deliberate re-review of user-visible
default behavior.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from winml.modelkit.session import EP_DEVICE_SPECS, default_ep_for_device


# ---------------------------------------------------------------------------
# Invariant tests — structural, no mocking required
# ---------------------------------------------------------------------------


def _first_index_of_ep_in_device_group(ep_full: str, device: str) -> int:
    for i, spec in enumerate(EP_DEVICE_SPECS):
        if spec.ep == ep_full and spec.device == device:
            return i
    return -1


def test_gpu_dml_is_deprioritized_below_every_plugin_gpu_ep():
    """DML must sit AFTER every plugin GPU EP in the catalog walk."""
    dml_idx = _first_index_of_ep_in_device_group("DmlExecutionProvider", "gpu")
    assert dml_idx >= 0, "DML must remain in the catalog as a cross-vendor fallback"

    plugin_gpu_eps = (
        "OpenVINOExecutionProvider",
        "MIGraphXExecutionProvider",
        "TensorrtExecutionProvider",
        "NvTensorRTRTXExecutionProvider",
    )
    for ep in plugin_gpu_eps:
        idx = _first_index_of_ep_in_device_group(ep, "gpu")
        assert idx >= 0, f"{ep} must be catalog'd for gpu"
        assert idx < dml_idx, (
            f"{ep} (idx={idx}) must be catalog'd BEFORE DML (idx={dml_idx}) "
            f"so DML is only used as a cross-vendor fallback"
        )


def test_cpu_builtin_is_last_in_cpu_group():
    """Built-in CPUExecutionProvider must trail every plugin CPU EP."""
    cpu_idx = _first_index_of_ep_in_device_group("CPUExecutionProvider", "cpu")
    assert cpu_idx >= 0

    for spec in EP_DEVICE_SPECS:
        if spec.device == "cpu" and spec.ep != "CPUExecutionProvider":
            idx = _first_index_of_ep_in_device_group(spec.ep, "cpu")
            assert idx < cpu_idx, (
                f"{spec.ep} (idx={idx}) must be catalog'd BEFORE "
                f"CPUExecutionProvider (idx={cpu_idx})"
            )


def test_no_builtin_ep_for_npu():
    """NPU must have no built-in fallback — plugin-only device class."""
    builtins = {"CPUExecutionProvider", "DmlExecutionProvider", "AzureExecutionProvider"}
    npu_builtins = [
        spec for spec in EP_DEVICE_SPECS if spec.device == "npu" and spec.ep in builtins
    ]
    assert not npu_builtins, f"NPU must be plugin-only; found built-in rows: {npu_builtins}"


# ---------------------------------------------------------------------------
# Per-vendor default_ep_for_device tests — simulate availability + vendor
# ---------------------------------------------------------------------------


def _mock_available_eps(*eps: str):
    """Patch WinMLEPRegistry.instance().available_eps() to return `eps`."""
    from winml.modelkit.session.ep_registry import WinMLEPRegistry

    frozen = frozenset(eps)
    return patch.object(
        WinMLEPRegistry,
        "instance",
        return_value=type("Reg", (), {"available_eps": lambda self: frozen})(),
    )


def _mock_vendors(*vendors: str):
    """Patch the vendor-detection cache to a controlled set."""
    from winml.modelkit import ep_path

    # _get_detected_vendors is @functools.cache — clear before + patch impl.
    ep_path._get_detected_vendors.cache_clear()
    return patch.object(ep_path, "_get_detected_vendors", return_value=frozenset(vendors))


@pytest.mark.parametrize(
    "vendor,device,available_eps,expected_ep",
    [
        # Intel host: OpenVINO wins on GPU (deprioritizing DML).
        (
            "Intel Corporation",
            "gpu",
            ("OpenVINOExecutionProvider", "DmlExecutionProvider"),
            "OpenVINOExecutionProvider",
        ),
        # AMD host with only DML available: DML falls through (no vendor plugin).
        (
            "Advanced Micro Devices",
            "gpu",
            ("DmlExecutionProvider",),
            "DmlExecutionProvider",
        ),
        # AMD host with MIGraphX + DML: MIGraphX wins.
        (
            "Advanced Micro Devices",
            "gpu",
            ("MIGraphXExecutionProvider", "DmlExecutionProvider"),
            "MIGraphXExecutionProvider",
        ),
        # NVIDIA host: NvTensorRTRTX wins on GPU.
        (
            "NVIDIA Corporation",
            "gpu",
            ("NvTensorRTRTXExecutionProvider", "DmlExecutionProvider"),
            "NvTensorRTRTXExecutionProvider",
        ),
        # CPU device always resolves via first available.
        (
            "Intel Corporation",
            "cpu",
            ("CPUExecutionProvider",),
            "CPUExecutionProvider",
        ),
        (
            "Intel Corporation",
            "cpu",
            ("OpenVINOExecutionProvider", "CPUExecutionProvider"),
            "OpenVINOExecutionProvider",
        ),
    ],
)
def test_default_ep_for_device_by_vendor(vendor, device, available_eps, expected_ep, monkeypatch):
    """Preference-order pin: (vendor, device, installed EPs) → resolved EP."""
    # Patch vendor detection so EP_CATALOG.is_compatible reflects the mock.
    with _mock_vendors(vendor), _mock_available_eps(*available_eps):
        assert default_ep_for_device(device) == expected_ep


def test_default_ep_for_device_gpu_returns_none_when_nothing_available():
    """No installed EPs → None (caller decides fallback)."""
    with _mock_vendors("Intel Corporation"), _mock_available_eps():
        assert default_ep_for_device("gpu") is None


def test_dml_only_valid_when_no_vendor_plugin_is_installed():
    """DML must NOT win on Intel host if OpenVINO is installed and compatible."""
    with (
        _mock_vendors("Intel Corporation"),
        _mock_available_eps("OpenVINOExecutionProvider", "DmlExecutionProvider"),
    ):
        result = default_ep_for_device("gpu")
    assert result == "OpenVINOExecutionProvider", (
        f"Intel host with OpenVINO installed must resolve gpu → OpenVINO, "
        f"got {result!r} — DML deprioritization broken"
    )


# EP_CATALOG.is_compatible internals are covered by ep_path's own test
# suite; this file only pins the observable default_ep_for_device
# contract that catalog compatibility gates.
