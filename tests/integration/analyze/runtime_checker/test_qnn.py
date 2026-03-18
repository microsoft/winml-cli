from pathlib import Path

import onnxruntime as ort
import pytest

from tests.integration.analyze.runtime_checker.test_helper import (
    op_quick_helper,
    reshape_quick_helper,
    should_run_ep_test,
)
from winml.modelkit import winml
from winml.modelkit.analyze.runtime_checker.ep_checker import EPChecker


winml.register_execution_providers(ort=True)


pytestmark = pytest.mark.skipif(
    not should_run_ep_test("QNNExecutionProvider", ort.OrtHardwareDeviceType.NPU),
    reason="QNN tests require QNN hardware",
)


# don't use EPChecker directly as there is a bug with pytest in subprocess
class QNNNPUChecker(EPChecker):
    def __init__(self):
        super().__init__(ep_name="QNNExecutionProvider", device_type=ort.OrtHardwareDeviceType.NPU)


def test_reshape_qnn_quick() -> None:
    reshape_quick_helper(
        ep_checker=QNNNPUChecker(),
        truth_file=Path(__file__).parent / "reshape_qnn_results.json",
    )


def test_not_qnn() -> None:
    op_quick_helper(
        op_name="Not",
        ep_checker=QNNNPUChecker(),
        truth_file=Path(__file__).parent / "not_qnn_results.json",
    )
