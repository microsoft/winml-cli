# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import onnxruntime as ort

from tests.integration.analyze.runtime_checker.test_helper import (
    reshape_quick_helper,
    should_run_ep_test,
)
from winml.modelkit import winml


winml.register_execution_providers(ort=True)

from winml.modelkit.analyze.runtime_checker.ep_checker import (  # noqa: E402
    EPChecker,
)


# don't use EPChecker directly as there is a bug with pytest in subprocess
class OVNPUChecker(EPChecker):
    def __init__(self):
        super().__init__(
            ep_name="OpenVINOExecutionProvider", device_type=ort.OrtHardwareDeviceType.NPU
        )


# don't use EPChecker directly as there is a bug with pytest in subprocess
class OVCPUChecker(EPChecker):
    def __init__(self):
        super().__init__(
            ep_name="OpenVINOExecutionProvider", device_type=ort.OrtHardwareDeviceType.CPU
        )


# don't use EPChecker directly as there is a bug with pytest in subprocess
class OVGPUChecker(EPChecker):
    def __init__(self):
        super().__init__(
            ep_name="OpenVINOExecutionProvider", device_type=ort.OrtHardwareDeviceType.GPU
        )


def test_reshape_openvino_npu_quick() -> None:
    should_run_ep_test(
        "OpenVINOExecutionProvider",
        ort.OrtHardwareDeviceType.NPU,
        "OpenVINO NPU tests require OpenVINO NPU hardware",
    )
    reshape_quick_helper(
        ep_checker=OVNPUChecker(),
        truth_file=Path(__file__).parent / "reshape_openvino_npu_results.json",
    )


def test_reshape_openvino_cpu_quick() -> None:
    should_run_ep_test(
        "OpenVINOExecutionProvider",
        ort.OrtHardwareDeviceType.CPU,
        "OpenVINO CPU tests require OpenVINO CPU hardware",
    )
    reshape_quick_helper(
        ep_checker=OVCPUChecker(),
        truth_file=Path(__file__).parent / "reshape_openvino_cpu_results.json",
    )


def test_reshape_openvino_gpu_quick() -> None:
    should_run_ep_test(
        "OpenVINOExecutionProvider",
        ort.OrtHardwareDeviceType.GPU,
        "OpenVINO GPU tests require OpenVINO GPU hardware",
    )
    reshape_quick_helper(
        ep_checker=OVGPUChecker(),
        truth_file=Path(__file__).parent / "reshape_openvino_gpu_results.json",
    )
