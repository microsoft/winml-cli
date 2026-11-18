# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Verify OpenVINO virtual NPU compilation through ORT and the CLI."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import Mock

import onnx
import onnxruntime as ort
import pytest
from click.testing import CliRunner

from tests.fixtures.create_test_models import create_simple_conv_model
from winml.modelkit.cli import main
from winml.modelkit.ep_path import discover_all_eps
from winml.modelkit.onnx import get_external_data_files


if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.skipif(sys.platform != "win32", reason="Requires the Windows OpenVINO EP")
def test_openvino_compiles_virtual_npu_without_inference_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Enumerate virtual targets and compile without loading the output for inference."""
    entries = [
        entry for entry in discover_all_eps() if entry.ep_name == "OpenVINOExecutionProvider"
    ]
    if not entries:
        pytest.skip("OpenVINO compiler payload unavailable; set WINMLCLI_EP_PATH")

    entry = entries[0]
    library_path = entry.dll_path.resolve()
    registration_name = f"{entry.ep_name}.virtual"
    ort.register_execution_provider_library(registration_name, str(library_path))
    try:
        targets = [
            device
            for device in ort.get_ep_devices()
            if device.ep_name == entry.ep_name
            and device.device.type == ort.OrtHardwareDeviceType.NPU
            and device.device.metadata.get("is_virtual") == "1"
            and Path(device.ep_metadata.get("library_path", "")).resolve() == library_path
        ]
        assert targets, "OpenVINO registered but exposed no virtual NPU targets"
        assert all(device.ep_metadata.get("npu_platform") for device in targets)
        targets.sort(
            key=lambda device: (
                device.ep_metadata["npu_platform"],
                device.device.device_id,
            )
        )
        record_property("runtime_version", ort.__version__)
        record_property("library_path", str(library_path))
        record_property(
            "compilation_targets",
            json.dumps(
                [
                    {
                        "metadata": dict(device.ep_metadata),
                        "hardware_metadata": dict(device.device.metadata),
                        "device_id": device.device.device_id,
                    }
                    for device in targets
                ],
                sort_keys=True,
            ),
        )

        model_path = tmp_path / "model.onnx"
        output_path = tmp_path / "compiled.onnx"
        onnx.save(create_simple_conv_model(), model_path)
        original_hash = hashlib.sha256(model_path.read_bytes()).digest()
        monkeypatch.setattr(
            ort,
            "InferenceSession",
            Mock(side_effect=AssertionError("Compilation must not load a runtime session")),
        )
        options = ort.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        options.add_provider_for_devices([targets[0]], {})
        compiler = ort.ModelCompiler(
            options,
            str(model_path),
            embed_compiled_data_into_model=True,
            flags=(
                ort.OrtCompileApiFlags.ERROR_IF_NO_NODES_COMPILED
                | ort.OrtCompileApiFlags.ERROR_IF_OUTPUT_FILE_EXISTS
            ),
        )
        compiler.compile_to_file(str(output_path))

        assert output_path.is_file()
        compiled = onnx.load(output_path, load_external_data=False)
        contexts = [
            node
            for node in compiled.graph.node
            if node.op_type == "EPContext" and node.domain == "com.microsoft"
        ]
        assert contexts, "Compilation produced no EPContext"
        for node in contexts:
            attributes = {
                attribute.name: onnx.helper.get_attribute_value(attribute)
                for attribute in node.attribute
            }
            assert attributes["source"] == entry.ep_name.encode("utf-8")
            assert attributes["embed_mode"] == 1
            assert attributes["ep_cache_context"]
        assert hashlib.sha256(model_path.read_bytes()).digest() == original_hash
    finally:
        ort.unregister_execution_provider_library(registration_name)


@pytest.mark.skipif(sys.platform != "win32", reason="Requires the Windows OpenVINO EP")
@pytest.mark.parametrize(
    ("external_weights", "overwrite"),
    [(False, False), (True, True)],
    ids=["embedded-input-new-output", "external-input-overwrite-output"],
)
def test_cli_compiles_advertised_virtual_npu_target(
    external_weights: bool,
    overwrite: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Run real target discovery and compilation without physical resolution or inference."""
    inference = Mock(side_effect=AssertionError("Compilation must not load a runtime session"))
    physical = Mock(side_effect=AssertionError("Physical device resolution is forbidden"))
    register = Mock(wraps=ort.register_execution_provider_library)
    unregister = Mock(wraps=ort.unregister_execution_provider_library)
    monkeypatch.setattr(ort, "InferenceSession", inference)
    monkeypatch.setattr("winml.modelkit.commands.compile.resolve_device", physical)
    monkeypatch.setattr(ort, "register_execution_provider_library", register)
    monkeypatch.setattr(ort, "unregister_execution_provider_library", unregister)
    runner = CliRunner()
    arguments = ["compile", "--off-device", "--ep", "openvino"]
    inventory = runner.invoke(main, [*arguments, "--list-eps"])
    assert inventory.exit_code == 0, inventory.output
    if not json.loads(inventory.stdout)[0]["payload_available"]:
        pytest.skip("OpenVINO compiler payload unavailable; set WINMLCLI_EP_PATH")

    listing = runner.invoke(main, [*arguments, "--list-targets"])
    assert listing.exit_code == 0, listing.output
    targets = json.loads(listing.stdout)
    assert targets, "OpenVINO exposed no virtual NPU targets"
    assert all(
        target["provider_name"] == "OpenVINOExecutionProvider"
        and target["hardware_metadata"]["is_virtual"] == "1"
        for target in targets
    )
    selected = targets[0]
    record_property("runtime_version", ort.__version__)
    record_property("selected_target", json.dumps(selected, sort_keys=True))

    model_path = tmp_path / "model.onnx"
    model = create_simple_conv_model()
    if external_weights:
        for tensor in model.graph.initializer:
            tensor.CopyFrom(
                onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(tensor), name=tensor.name)
            )
        onnx.save_model(
            model,
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.bin",
            size_threshold=0,
        )
    else:
        onnx.save(model, model_path)
    data_files = get_external_data_files(model_path)
    assert bool(data_files) == external_weights
    inputs = [model_path, *(model_path.parent / name for name in data_files)]
    input_hashes = {path: hashlib.sha256(path.read_bytes()).digest() for path in inputs}
    output_path = tmp_path / "output" / "compiled.onnx"
    if overwrite:
        output_path.parent.mkdir()
        output_path.write_bytes(b"previous output")
    arguments += [
        "--target",
        selected["npu_platform"],
        "--model",
        str(model_path),
        "--output",
        str(output_path),
    ]
    result = runner.invoke(main, [*arguments, *(["--overwrite"] if overwrite else [])])
    assert result.exit_code == 0, result.output
    assert "no inference was run" in result.output
    assert not get_external_data_files(output_path)
    contexts = [
        {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
        for node in onnx.load(output_path, load_external_data=False).graph.node
        if node.op_type == "EPContext" and node.domain == "com.microsoft"
    ]
    assert contexts, "Compilation produced no EPContext"
    assert all(
        context["source"] == b"OpenVINOExecutionProvider" and context.get("embed_mode", 1) == 1
        for context in contexts
    )
    main_contexts = [context for context in contexts if context.get("main_context", 1) == 1]
    assert main_contexts
    assert all(context["ep_cache_context"] for context in main_contexts)
    assert {path: hashlib.sha256(path.read_bytes()).digest() for path in inputs} == input_hashes
    assert list(output_path.parent.iterdir()) == [output_path]

    original_output = output_path.read_bytes()
    rejected = runner.invoke(main, arguments)
    assert rejected.exit_code == 1, rejected.output
    assert "already exists" in rejected.output
    assert output_path.read_bytes() == original_output
    assert register.call_count == unregister.call_count == 2
    aliases = [call.args[0] for call in register.call_args_list]
    assert len(set(aliases)) == 2
    assert all(alias.endswith(".virtual") for alias in aliases)
    assert [call.args[0] for call in unregister.call_args_list] == aliases
    assert all(
        Path(call.args[1]).resolve() == Path(selected["library_path"]).resolve()
        for call in register.call_args_list
    )
    inference.assert_not_called()
    physical.assert_not_called()
