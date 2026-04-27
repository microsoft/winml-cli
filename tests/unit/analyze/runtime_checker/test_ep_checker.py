# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import onnxruntime as ort

from winml.modelkit.analyze.runtime_checker.ep_checker import EPChecker


def _make_checker(
    ep_name: str,
    device_type: ort.OrtHardwareDeviceType = ort.OrtHardwareDeviceType.CPU,
) -> EPChecker:
    return EPChecker(ep_name=ep_name, device_type=device_type)


def test_needs_case_isolation_for_openvino_npu() -> None:
    checker = _make_checker(
        "OpenVINOExecutionProvider", device_type=ort.OrtHardwareDeviceType.NPU
    )
    assert checker.needs_case_isolation() is True


def test_needs_case_isolation_for_openvino_cpu() -> None:
    checker = _make_checker(
        "OpenVINOExecutionProvider", device_type=ort.OrtHardwareDeviceType.CPU
    )
    assert checker.needs_case_isolation() is False


def test_needs_case_isolation_for_non_isolated_ep() -> None:
    checker = _make_checker("QNNExecutionProvider")
    assert checker.needs_case_isolation() is False
