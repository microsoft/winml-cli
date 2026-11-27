# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Compile with the real VitisAI plugin while forbidding physical NPU binding."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from click.testing import CliRunner
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

from tests.fixtures.create_test_models import create_simple_conv_model
from winml.modelkit.cli import main
from winml.modelkit.onnx import get_external_data_files


if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def quantized_model(tmp_path: Path) -> Path:
    """Prepare a compatible integer-output model using CPU-only calibration."""
    graph = create_simple_conv_model()
    raw = tmp_path / "float.onnx"
    onnx.save(graph, raw)
    tensor = graph.graph.input[0]
    shape = [dimension.dim_value for dimension in tensor.type.tensor_type.shape.dim]

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.samples = iter(
                [{tensor.name: np.random.default_rng(0).normal(size=shape).astype(np.float32)}]
            )

        def get_next(self) -> dict[str, np.ndarray] | None:
            return next(self.samples, None)

    output = tmp_path / "model.onnx"
    quantize_static(
        str(raw),
        str(output),
        Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibration_providers=["CPUExecutionProvider"],
    )
    quantized = onnx.load(output)
    for output_info in quantized.graph.output:
        producer = next(node for node in quantized.graph.node if output_info.name in node.output)
        assert producer.op_type == "DequantizeLinear"
        zero_point = next(
            tensor for tensor in quantized.graph.initializer if tensor.name == producer.input[2]
        )
        output_info.name = producer.input[0]
        output_info.type.tensor_type.elem_type = zero_point.data_type
        quantized.graph.node.remove(producer)
    onnx.checker.check_model(quantized)
    onnx.save(quantized, output)
    return output


@pytest.mark.skipif(sys.platform != "win32", reason="Requires the Windows VitisAI plugin")
@pytest.mark.parametrize(("compatible", "overwrite"), [(True, False), (True, True), (False, True)])
def test_cli_compiles_vitisai_with_virtual_npu_only(
    compatible: bool,
    overwrite: bool,
    quantized_model: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    inference = Mock(side_effect=AssertionError("Inference is forbidden during compilation"))
    physical = Mock(side_effect=AssertionError("Physical device resolution is forbidden"))
    register = Mock(wraps=ort.register_execution_provider_library)
    unregister = Mock(wraps=ort.unregister_execution_provider_library)
    native_bind = ort.SessionOptions.add_provider_for_devices
    bound = []

    def bind(options: ort.SessionOptions, devices: list, provider_options: dict) -> None:
        assert len(devices) == 1
        device = devices[0]
        assert device.ep_name == "VitisAIExecutionProvider"
        assert device.device.type == ort.OrtHardwareDeviceType.NPU
        assert device.device.metadata.get("is_virtual") == "1"
        assert options.get_session_config_entry("session.disable_cpu_ep_fallback") == "1"
        assert provider_options == {"target": "X2", "enable_cache_file_io_in_mem": "1"}
        bound.append(
            {
                "provider_name": device.ep_name,
                "library_path": device.ep_metadata["library_path"],
                "hardware_metadata": dict(device.device.metadata),
                "provider_options": dict(provider_options),
            }
        )
        native_bind(options, devices, provider_options)

    monkeypatch.setattr(ort, "InferenceSession", inference)
    monkeypatch.setattr("winml.modelkit.commands.compile.resolve_device", physical)
    monkeypatch.setattr(ort, "register_execution_provider_library", register)
    monkeypatch.setattr(ort, "unregister_execution_provider_library", unregister)
    monkeypatch.setattr(ort.SessionOptions, "add_provider_for_devices", bind)
    runner = CliRunner()
    arguments = ["compile", "--compiler", "offdevice", "--ep", "vitisai"]
    inventory = runner.invoke(main, [*arguments, "--list-eps"])
    assert inventory.exit_code == 0, inventory.output
    row = json.loads(inventory.stdout)[0]
    if not row["payload_available"]:
        pytest.skip("VitisAI compiler payload unavailable; set WINMLCLI_EP_PATH to its plugin DLL")
    register.assert_not_called()
    model = quantized_model if compatible else quantized_model.with_name("float.onnx")
    original = hashlib.sha256(model.read_bytes()).digest()
    output = model.parent / "output" / "compiled.onnx"
    if overwrite:
        output.parent.mkdir()
        output.write_bytes(b"previous output")
    arguments += ["--ep-options", "target=X2", "--model", str(model), "--output", str(output)]
    result = runner.invoke(main, [*arguments, *(["--overwrite"] if overwrite else [])])
    assert len(bound) == 1, result.output
    assert (
        Path(bound[0]["library_path"]).resolve()
        == Path(row["payloads"][0]["library_path"]).resolve()
    )
    assert hashlib.sha256(model.read_bytes()).digest() == original
    assert list(output.parent.iterdir()) == [output]
    names = [call.args[0] for call in register.call_args_list]
    assert len(names) == 2
    assert names[0].endswith(".virtual")
    assert names[1] == "VitisAIExecutionProvider"
    assert [call.args[0] for call in unregister.call_args_list] == names[::-1]
    inference.assert_not_called()
    physical.assert_not_called()
    record_property("runtime_version", ort.__version__)
    record_property("selected_payload", json.dumps(row, sort_keys=True))
    record_property("bound_virtual_device", json.dumps(bound[0], sort_keys=True))
    if not compatible:
        assert result.exit_code == 1, result.output
        assert "Off-device compilation failed" in result.output
        assert output.read_bytes() == b"previous output"
        return
    assert result.exit_code == 0, result.output
    report = " ".join(result.output.split())
    assert "Artifact validation: passed (embedded-context checks)" in report
    assert "Target-device inference validation: skipped (off-device compilation)" in report
    assert not get_external_data_files(output)
    contexts = [
        {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
        for node in onnx.load(output, load_external_data=False).graph.node
        if node.op_type == "EPContext" and node.domain == "com.microsoft"
    ]
    assert contexts
    assert all(
        context["source"] == b"VitisAIExecutionProvider" and context.get("embed_mode", 1) == 1
        for context in contexts
    )
    main_contexts = [context for context in contexts if context.get("main_context", 1) == 1]
    assert main_contexts
    assert all(context["ep_cache_context"] for context in main_contexts)
    compiled = output.read_bytes()
    rejected = runner.invoke(main, arguments)
    assert rejected.exit_code == 1, rejected.output
    assert output.read_bytes() == compiled
    assert register.call_count == 2
    record_property("output_bytes", len(compiled))
