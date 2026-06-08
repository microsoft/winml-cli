# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for (EP, device) pair validation and default-device inference.

Regression tests for:
* Issue #513 - ``winml analyze --ep cpu`` silently uses NPU as device default.
* Issue #521 - ``winml compile --device cpu --ep qnn`` silently uses QNN/NPU.

Both bugs are rooted in the absence of a single EP-to-device support table.
The fix introduces ``EP_SUPPORTED_DEVICES`` in ``utils.constants``: an
ordered tuple per EP whose first element is the canonical default device.
"""

from __future__ import annotations

import click
import pytest


# =============================================================================
# EP_SUPPORTED_DEVICES policy table
# =============================================================================


class TestEpSupportedDevices:
    """The canonical EP -> supported-device-tuple map. First element = default."""

    def test_import_available(self) -> None:
        from winml.modelkit.utils.constants import EP_SUPPORTED_DEVICES

        assert isinstance(EP_SUPPORTED_DEVICES, dict)
        assert len(EP_SUPPORTED_DEVICES) > 0

    def test_keys_match_supported_eps(self) -> None:
        from winml.modelkit.utils.constants import EP_SUPPORTED_DEVICES, SUPPORTED_EPS

        assert set(EP_SUPPORTED_DEVICES) == set(SUPPORTED_EPS)

    def test_values_are_non_empty_tuples_of_lowercase_devices(self) -> None:
        from winml.modelkit.utils.constants import EP_SUPPORTED_DEVICES, SUPPORTED_DEVICES

        valid = {d.lower() for d in SUPPORTED_DEVICES}
        for ep, devs in EP_SUPPORTED_DEVICES.items():
            assert isinstance(devs, tuple) and len(devs) > 0, ep
            assert all(d in valid for d in devs), (ep, devs)

    @pytest.mark.parametrize(
        ("ep", "default"),
        [
            ("CPUExecutionProvider", "cpu"),
            ("QNNExecutionProvider", "npu"),
            ("VitisAIExecutionProvider", "npu"),
            ("DmlExecutionProvider", "gpu"),
            ("CUDAExecutionProvider", "gpu"),
            ("NvTensorRTRTXExecutionProvider", "gpu"),
            ("MIGraphXExecutionProvider", "gpu"),
            # OpenVINO defaults to NPU (first entry in the supported tuple).
            ("OpenVINOExecutionProvider", "npu"),
        ],
    )
    def test_default_device_per_ep(self, ep: str, default: str) -> None:
        from winml.modelkit.utils.constants import EP_SUPPORTED_DEVICES

        assert EP_SUPPORTED_DEVICES[ep][0] == default


# =============================================================================
# Compile resolver -- incompatible (device, ep) pair rejection (Issue #521)
# =============================================================================


class TestCompileIncompatiblePair:
    """``_resolve_compile_provider`` must reject (device, ep) combos that the
    policy table marks as unsupported."""

    @pytest.mark.parametrize(
        ("device", "ep"),
        [
            ("cpu", "qnn"),
            ("cpu", "dml"),
            ("cpu", "vitisai"),
            ("cpu", "migraphx"),
            ("npu", "cpu"),
            ("gpu", "cpu"),
            ("npu", "dml"),
            ("npu", "migraphx"),
        ],
    )
    def test_incompatible_pair_rejected(self, device: str, ep: str) -> None:
        from winml.modelkit.commands.compile import _resolve_compile_provider

        with pytest.raises((click.UsageError, click.ClickException)):
            _resolve_compile_provider(device, ep)

    @pytest.mark.parametrize(
        ("device", "ep", "expected"),
        [
            ("cpu", "cpu", "CPUExecutionProvider"),
            ("npu", "qnn", "QNNExecutionProvider"),
            ("gpu", "qnn", "QNNExecutionProvider"),
            ("npu", "vitisai", "VitisAIExecutionProvider"),
            ("gpu", "dml", "DmlExecutionProvider"),
            ("gpu", "migraphx", "MIGraphXExecutionProvider"),
            ("gpu", "openvino", "OpenVINOExecutionProvider"),
        ],
    )
    def test_compatible_pair_returns_canonical(self, device: str, ep: str, expected: str) -> None:
        from winml.modelkit.commands.compile import _resolve_compile_provider

        assert _resolve_compile_provider(device, ep) == expected


# Note: EP host-availability ("EP not registered on this host") is enforced by
# ``sysinfo.device.resolve_device``; see
# tests/unit/sysinfo/test_device.py::test_ep_requested_but_not_available_raises
# for that regression coverage.
