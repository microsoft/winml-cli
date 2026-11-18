# Copyright (C) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Exercise real QNN HTP compilation on a CPU host through the root CLI."""

from __future__ import annotations

import hashlib
import json
import sys
from typing import TYPE_CHECKING
from unittest.mock import Mock

import onnx
import onnxruntime as ort
import pytest
from click.testing import CliRunner

from tests.fixtures.create_test_models import create_simple_conv_model
from winml.modelkit.cli import main
from winml.modelkit.onnx import get_external_data_files


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.mark.skipif(sys.platform != "win32", reason="Requires the Windows QNN HTP payload")
@pytest.mark.parametrize(
    ("target", "overwrite"), [("60", False), ("60", True), ("2147483647", True)]
)
def test_cli_compiles_qnn_htp_without_physical_device_or_inference(
    target: str,
    overwrite: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    inference = Mock(side_effect=AssertionError("Inference is forbidden"))
    physical = Mock(side_effect=AssertionError("Physical device resolution is forbidden"))
    register = Mock(wraps=ort.register_execution_provider_library)
    unregister = Mock(wraps=ort.unregister_execution_provider_library)
    monkeypatch.setattr(ort, "InferenceSession", inference)
    monkeypatch.setattr("winml.modelkit.commands.compile.resolve_device", physical)
    monkeypatch.setattr(ort, "register_execution_provider_library", register)
    monkeypatch.setattr(ort, "unregister_execution_provider_library", unregister)
    runner = CliRunner()
    arguments = ["compile", "--off-device", "--ep", "qnn"]
    inventory = runner.invoke(main, [*arguments, "--list-eps"])
    assert inventory.exit_code == 0, inventory.output
    row = json.loads(inventory.stdout)[0]
    if not row["payload_available"]:
        pytest.skip("QNN HTP compiler payload unavailable; set WINMLCLI_EP_PATH")
    assert row["targets_enumerable"] is False
    register.assert_not_called()

    model_path = tmp_path / "model.onnx"
    onnx.save(create_simple_conv_model(), model_path)
    original_model = hashlib.sha256(model_path.read_bytes()).digest()
    output = tmp_path / "output" / "compiled.onnx"
    if overwrite:
        output.parent.mkdir()
        output.write_bytes(b"previous output")
    arguments += ["--target", target, "--model", str(model_path), "--output", str(output)]
    result = runner.invoke(main, [*arguments, *(["--overwrite"] if overwrite else [])])
    if target != "60":
        assert result.exit_code == 1, result.output
        assert "Off-device compilation failed" in result.output
        assert output.read_bytes() == b"previous output"
        assert hashlib.sha256(model_path.read_bytes()).digest() == original_model
        assert list(output.parent.iterdir()) == [output]
        register.assert_called_once()
        unregister.assert_called_once_with("QNNExecutionProvider")
        inference.assert_not_called()
        physical.assert_not_called()
        return
    assert result.exit_code == 0, result.output
    assert "no inference was run" in result.output
    assert not get_external_data_files(output)
    contexts = [
        {attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
        for node in onnx.load(output, load_external_data=False).graph.node
        if node.op_type == "EPContext" and node.domain == "com.microsoft"
    ]
    assert contexts
    assert all(
        context["source"] == b"QNNExecutionProvider" and context.get("embed_mode", 1) == 1
        for context in contexts
    )
    main_contexts = [context for context in contexts if context.get("main_context", 1) == 1]
    assert main_contexts
    assert all(context["ep_cache_context"] for context in main_contexts)
    assert hashlib.sha256(model_path.read_bytes()).digest() == original_model
    assert list(output.parent.iterdir()) == [output]

    compiled_bytes = output.read_bytes()
    rejected = runner.invoke(main, arguments)
    assert rejected.exit_code == 1, rejected.output
    assert output.read_bytes() == compiled_bytes
    register.assert_called_once()
    assert register.call_args.args[0] == "QNNExecutionProvider"
    unregister.assert_called_once_with("QNNExecutionProvider")
    inference.assert_not_called()
    physical.assert_not_called()
    record_property("runtime_version", ort.__version__)
    record_property("selected_payload", json.dumps(row, sort_keys=True))
    record_property("soc_model", "60")
    record_property(
        "ep_sdk_versions", repr([context.get("ep_sdk_version") for context in contexts])
    )
    record_property("output_bytes", len(compiled_bytes))
