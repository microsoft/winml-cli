# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Exercise every OpenVINO GPU route, including duplicate RDP handles."""

import json
import subprocess

import numpy as np
import onnxruntime as ort
import pytest

from scripts.e2e_eval.winml_cli import cli_command
from tests.e2e.require_ep import require_ep
from winml.modelkit.compiler import EPConfig
from winml.modelkit.session import EPDeviceTarget, WinMLEPDevice, WinMLEPRegistry, WinMLSession
from winml.modelkit.sysinfo import get_ep_device_luid


pytestmark = pytest.mark.e2e


def _gpu_routes():
    require_ep("openvino", device="gpu")
    selected = WinMLEPRegistry.instance().auto_device(EPDeviceTarget(ep="openvino", device="gpu"))
    return [
        WinMLEPDevice(ep=selected.ep, device=d)
        for d in selected.ep.devices
        if d.device_type == "GPU" and d.ort_handle.ep_name == "OpenVINOExecutionProvider"
    ]


@pytest.mark.parametrize("epcontext", [False, True], ids=["runtime", "epcontext"])
def test_gpu_routes_match_cpu(simple_matmul_onnx, sample_input, epcontext):
    reference = ort.InferenceSession(str(simple_matmul_onnx), providers=["CPUExecutionProvider"])
    expected = reference.run(None, sample_input)
    routes = _gpu_routes()
    assert routes
    for route in routes:
        session = WinMLSession(
            simple_matmul_onnx,
            ep_device=route,
            ep_config=EPConfig(provider="openvino", enable_ep_context=epcontext),
        )
        try:
            for _ in range(2):
                actual = session.run(sample_input)
                for output, value in zip(reference.get_outputs(), expected, strict=True):
                    np.testing.assert_allclose(actual[output.name], value, rtol=1e-2, atol=1e-2)
        finally:
            session.reset()


def test_cli_subprocess_gpu_routes(simple_matmul_onnx, tmp_path):
    routes = _gpu_routes()
    assert routes
    for index, route in enumerate(routes):
        luid = get_ep_device_luid(route.device.ort_handle)
        assert luid
        output = tmp_path / f"route-{index}.json"
        result = subprocess.run(  # noqa: S603 -- fixed E2E entry point, no shell
            [
                *cli_command(),
                "perf",
                "-m",
                str(simple_matmul_onnx),
                "--device",
                "gpu",
                "--ep",
                "openvino",
                "--device-luid",
                luid,
                "--iterations",
                "2",
                "--warmup",
                "1",
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "[E2E] OpenVINO GPU allocator workaround" in result.stderr
        assert (
            json.loads(output.read_text(encoding="utf-8"))["benchmark_info"]["ep"]
            == "OpenVINOExecutionProvider"
        )
