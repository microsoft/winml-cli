# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Native Runtime compilation of generated CGC IR and optional caller input."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from winml.modelkit.export.cgc import CGCExporter, CGCOptions
from winml.modelkit.sysinfo.dxcore_adapters import enumerate_compute_adapters


pytestmark = pytest.mark.e2e
_INPUT_ENCODINGS = ["text", "bytecode"]
if os.environ.get("WINML_TEST_CGC_INPUT_IR"):
    _INPUT_ENCODINGS.append("caller")


@pytest.fixture
def gpu_luid():
    if sys.platform != "win32":
        pytest.skip("Native Windows Runtime GPU compilation requires Windows.")
    pytest.importorskip("windowsml.runtime", reason="Preview Runtime API required.")
    adapters = [item for item in enumerate_compute_adapters() if item.device_type == "GPU"]
    if not adapters:
        pytest.skip("No DXCore GPU is available for native compilation.")
    return adapters[0].luid


@pytest.mark.parametrize("encoding", _INPUT_ENCODINGS)
def test_real_cli_input_ir_to_persistent_output_ir(tmp_path, gpu_luid, encoding):
    if encoding == "caller":
        source = Path(os.environ["WINML_TEST_CGC_INPUT_IR"])
        assert source.is_file()
    else:
        source_onnx = tmp_path / "source.onnx"
        model = helper.make_model(
            helper.make_graph(
                [helper.make_node("Relu", ["x"], ["y"])],
                "runtime_compile_test",
                [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])],
                [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=8,
        )
        onnx.save(model, source_onnx)
        source = tmp_path / "input.mlir"
        CGCExporter(CGCOptions()).export_onnx(source_onnx, source)
        if encoding == "bytecode":
            import windowsml
            from _foundry.compiler import Compiler, SerializationFormat

            with (
                Compiler(
                    library_path=Path(windowsml.__file__).parent / "lib" / "FoundryToolbox.dll"
                ) as compiler,
                compiler.create_module_from_mlir(source.read_bytes()) as module,
            ):
                source.write_bytes(module.serialize(SerializationFormat.BYTECODE))
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    output_dir = tmp_path / "compiled"
    output = output_dir / "output.mlir"
    argv = [
        sys.executable,
        "-m",
        "winml.modelkit.cli",
        "compile",
        "-m",
        str(source),
        "--input-format",
        "cgc-input-ir",
        "--target",
        "cgc-output-ir",
        "--compiler",
        "winml-runtime",
        "--device",
        "gpu",
        "--device-luid",
        gpu_luid,
        "-o",
        str(output),
    ]
    result = subprocess.run(  # noqa: S603 -- fixed CLI executable and argv, no shell
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=240,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "bytecode" in result.stdout and "schema passed" in result.stdout
    assert output.read_bytes().startswith(b"ML\xefR")
    receipt = json.loads((output_dir / "output_compile.json").read_text())
    assert receipt["input"]["sha256"] == original_hash
    assert receipt["gpu"]["luid"] == gpu_luid
    assert receipt["validation"]["status"] == "passed"
    assert receipt["compiled_form"] == "DEVICE_TARGETED"
    for item in receipt["artifacts"]:
        path = output_dir / item["filename"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    retry = subprocess.run(  # noqa: S603 -- same fixed argv for overwrite guard verification
        argv,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert retry.returncode != 0 and "--overwrite" in retry.stdout + retry.stderr
    if encoding == "text":
        invalid = tmp_path / "not-input-ir.onnx"
        invalid.write_bytes(source_onnx.read_bytes())
        wrong_args = list(argv)
        wrong_args[wrong_args.index("-m", 3) + 1] = str(invalid)
        wrong_args[wrong_args.index("-o") + 1] = str(output_dir / "invalid-output.mlir")
        wrong = subprocess.run(  # noqa: S603 -- fixed CLI argv, no shell
            wrong_args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            check=False,
        )
        assert wrong.returncode != 0
        assert "Cannot parse CGC Input IR" in wrong.stdout + wrong.stderr
        assert not (output_dir / "invalid-output.mlir").exists()
