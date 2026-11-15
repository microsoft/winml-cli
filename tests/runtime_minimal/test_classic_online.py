# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Opt-in CLIP source ONNX to native Runtime GPU execution checks."""

import hashlib
import json
import os
import platform
from contextlib import ExitStack
from pathlib import Path

import pytest


@pytest.mark.parametrize("kind", ["image", "text"])
def test_clip_online(kind, tmp_path):
    root = os.environ.get("WINML_CLIP_TEST_ROOT")
    if not root:
        pytest.skip("Set WINML_CLIP_TEST_ROOT to the native CLIP fixture directory")
    import numpy as np
    import windowsml.runtime as wr

    assert platform.machine().lower() in {"arm64", "aarch64"}
    root = Path(root)
    source = root / "models" / kind / "model.onnx"
    if kind == "image":
        feeds = [
            np.fromfile(root / "inputs" / "pixels.bin", dtype=np.float32).reshape(1, 3, 224, 224)
        ]
    else:
        feeds = [
            np.fromfile(root / "inputs" / f"{name}.bin", dtype=np.int32).reshape(1, 77)
            for name in ("input_ids", "attention_mask")
        ]
    with ExitStack() as stack:
        print("Creating Runtime and loading source ONNX", flush=True)
        runtime = stack.enter_context(wr.Runtime())
        source_model = stack.enter_context(runtime.load_model(str(source)))
        target = stack.enter_context(runtime.create_target(wr.ExecutionTargetKind.GPU))
        compiler = stack.enter_context(target.model_compiler())
        artifact = tmp_path / "compiled.mlir"
        print("Compiling source ONNX through Runtime ModelCompiler", flush=True)
        compiler.compile_to_file(source_model, str(artifact))
        model = stack.enter_context(runtime.load_model(str(artifact)))
        builder = stack.enter_context(runtime.create_pipeline_builder())
        stage = stack.enter_context(builder.add_model_stage(model, target))
        stage.execution_diagnostics_enabled = True
        output_count = source_model.schema().output_count
        for index in range(output_count):
            stage.request_output(index)
        print("Building GPU pipeline", flush=True)
        pipeline = stack.enter_context(builder.build())
        diagnostics = stage.execution_diagnostics
        shares = [
            {
                "device_kind": share.device_kind.name,
                "implementation": share.implementation.name,
                "adapter_luid": str(share.adapter_luid),
                "unit_count": share.unit_count,
            }
            for share in diagnostics.work_shares
        ]
        stage.bind_buffers(feeds)
        for iteration in range(4):
            print(f"GPU execution {iteration}", flush=True)
            pipeline.run()
            outputs = []
            for index in range(output_count):
                with stage.output(index) as tensor:
                    array = tensor.to_numpy().copy()
                    assert array.size > 0 and np.isfinite(array).all()
                    outputs.append(array)
        output_root = Path(os.environ.get("WINML_CLIP_TEST_OUTPUT", str(tmp_path)))
        output_root.mkdir(parents=True, exist_ok=True)
        np.savez(output_root / f"{kind}-outputs.npz", *outputs)
        (output_root / f"{kind}.json").write_text(
            json.dumps(
                {
                    "source": str(source),
                    "python_architecture": platform.machine(),
                    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "runtime": "windowsml.runtime",
                    "target": "GPU",
                    "runs": 4,
                    "coverage": diagnostics.coverage.name,
                    "work_shares": shares,
                    "output_shapes": [list(value.shape) for value in outputs],
                    "all_finite": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


@pytest.mark.parametrize("kind", ["image", "text"])
def test_clip_cpu_reference(kind):
    root = os.environ.get("WINML_CLIP_REFERENCE_ROOT")
    result_root = os.environ.get("WINML_CLIP_TEST_OUTPUT")
    if not root or not result_root:
        pytest.skip("Set WINML_CLIP_REFERENCE_ROOT and WINML_CLIP_TEST_OUTPUT for CPU comparison")
    import numpy as np
    import onnxruntime as ort

    root, result_root = Path(root), Path(result_root)
    source = root / "models" / kind / "model.onnx"
    session = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    feed = {}
    for spec in session.get_inputs():
        filename = "pixels" if spec.name == "pixel_values" else spec.name
        dtype = np.float32 if spec.type == "tensor(float)" else np.int32
        feed[spec.name] = np.fromfile(root / "inputs" / f"{filename}.bin", dtype=dtype)
        feed[spec.name] = feed[spec.name].reshape(spec.shape)
    reference = session.run(None, feed)
    errors = []
    matches = []
    with np.load(result_root / f"{kind}-outputs.npz") as actual:
        for index, expected in enumerate(reference):
            observed = actual[f"arr_{index}"]
            assert observed.shape == expected.shape
            matches.append(bool(np.allclose(observed, expected, rtol=1e-3, atol=1e-3)))
            errors.append(float(np.max(np.abs(observed - expected))))
    print(f"{kind} max absolute errors: {errors}", flush=True)
    (result_root / f"{kind}-parity.json").write_text(
        json.dumps(
            {
                "reference": "ORT CPU",
                "max_absolute_errors": errors,
                "rtol": 1e-3,
                "atol": 1e-3,
                "passed": all(matches),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    assert all(matches), f"{kind} Runtime output differs from ORT CPU: {errors}"
