# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Native dependency migration regressions, isolated from pytest's DLL state."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap

import pytest


def _probe(python: str, code: str) -> str:
    result = subprocess.run(  # noqa: S603
        [python, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"Native dependency probe failed ({result.returncode}) using {python}\n"
        f"{result.stdout}\n{result.stderr}"
    )
    return result.stdout.strip()


@pytest.mark.parametrize(
    "imports",
    [
        "import winml.modelkit; import onnxruntime; import pyarrow; import datasets",
        "import pyarrow; import datasets; import onnxruntime; import winml.modelkit",
        "import torch; import torchvision; import onnxruntime; import pyarrow",
        "import onnxruntime; import onnxruntime_genai; import torch; import pyarrow",
    ],
)
def test_native_import_order_and_cpu_inference(imports: str) -> None:
    """Catch ABI/load-order crashes and changes to default ONNX IR/opsets."""
    _assert_native_inference(imports, "CPUExecutionProvider", "float32")


@pytest.mark.gpu
@pytest.mark.skipif(sys.platform != "win32", reason="DirectML requires Windows")
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_directml_inference_without_cpu_fallback(dtype: str) -> None:
    """Validate native GPU kernels rather than accepting a CPU fallback."""
    _assert_native_inference("import winml.modelkit", "DmlExecutionProvider", dtype)


@pytest.mark.parametrize("module", ["onnxruntime_genai", "openvino", "soundfile"])
def test_optional_native_dependency_and_cpu_inference(module: str) -> None:
    """Check installed optional native extras without loading DLLs into pytest."""
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"Optional dependency {module} is not installed")
    _assert_native_inference(
        f"import torch; import {module}; import scipy.linalg; import sklearn; import pyarrow",
        "CPUExecutionProvider",
        "float32",
    )


def _assert_native_inference(imports: str, provider: str, dtype: str) -> None:
    _probe(
        sys.executable,
        imports
        + f"\nprovider = {provider!r}\ndtype_name = {dtype!r}\n"
        + textwrap.dedent(
            """
            import numpy as np
            import onnx
            import onnxruntime as ort
            import sys
            from onnx import TensorProto, helper

            tensor_type = TensorProto.FLOAT16 if dtype_name == "float16" else TensorProto.FLOAT
            graph = helper.make_graph(
                [helper.make_node("MatMul", ["X", "W"], ["Y"])],
                "compatibility",
                [
                    helper.make_tensor_value_info("X", tensor_type, [2, 3]),
                    helper.make_tensor_value_info("W", tensor_type, [3, 2]),
                ],
                [helper.make_tensor_value_info("Y", tensor_type, [2, 2])],
            )
            model = helper.make_model(graph)
            onnx.checker.check_model(model, full_check=True)
            options = ort.SessionOptions()
            if provider == "DmlExecutionProvider":
                assert "pyarrow" not in sys.modules
                options.enable_mem_pattern = False
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            session = ort.InferenceSession(
                model.SerializeToString(), sess_options=options, providers=[provider]
            )
            rng = np.random.default_rng(0)
            x = rng.standard_normal((2, 3)).astype(dtype_name)
            w = rng.standard_normal((3, 2)).astype(dtype_name)
            actual = session.run(None, {"X": x, "W": w})[0]
            tolerance = 5e-3 if dtype_name == "float16" else 1e-5
            np.testing.assert_allclose(actual, x @ w, rtol=tolerance, atol=tolerance / 10)
            # Exercise the historical EP-DLL-before-PyArrow crash order.
            import pyarrow
            import datasets
            """
        ),
    )


def test_parquet_bytes_are_deterministic() -> None:
    """Keep the pinned writer deterministic across independent Python processes."""
    code = """
        import hashlib
        import io
        import json

        import numpy as np
        import pandas as pd
        import pyarrow

        frame = pd.DataFrame({
            "condition": ["float32", None, "float16"],
            "compile_run_success": [(True, True), (True, False), (False, False)],
            "count": np.arange(3, dtype=np.int64),
            "value": np.linspace(0, 1, 3, dtype=np.float32),
        })
        output = io.BytesIO()
        frame.to_parquet(output, index=False)
        restored = pd.read_parquet(io.BytesIO(output.getvalue()))
        assert list(restored.columns) == list(frame.columns)
        np.testing.assert_array_equal(restored["count"], frame["count"])
        np.testing.assert_array_equal(restored["value"], frame["value"])
        print(json.dumps({
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "sha256": hashlib.sha256(output.getvalue()).hexdigest(),
        }))
    """
    expected = json.loads(_probe(sys.executable, code))
    actual = json.loads(_probe(sys.executable, code))
    assert actual == expected


@pytest.mark.parametrize(("dtype", "annotation"), [("float32", "FLOAT"), ("float16", "FLOAT16")])
def test_generated_rule_bytes_are_deterministic(dtype: str, annotation: str) -> None:
    """Keep representative model and Parquet bytes stable across independent processes."""
    code = f"""
        import hashlib
        import io
        import json

        import numpy as np
        import onnx
        import onnxruntime as ort
        import pandas as pd

        from winml.modelkit.pattern.op_input_gen.op_input_gen import get_runtime_checker_op

        schema = onnx.defs.get_schema("Abs", 22, "")
        generator = get_runtime_checker_op("Abs")(schema)
        values = np.array([[-2, 0, 3]], dtype={dtype!r})
        model = generator._create_model({{"X": values}}, {{"X": False}}, [{annotation!r}])
        onnx.checker.check_model(model, full_check=True)
        serialized = model.SerializeToString()
        session = ort.InferenceSession(serialized, providers=["CPUExecutionProvider"])
        actual = session.run(None, {{"X": values}})[0]
        np.testing.assert_array_equal(actual, np.abs(values))
        output = io.BytesIO()
        pd.DataFrame({{"model_bytes": [serialized], "dtype": [{dtype!r}]}}).to_parquet(
            output, index=False
        )
        print(json.dumps({{
            "ir": model.ir_version,
            "opsets": [(item.domain, item.version) for item in model.opset_import],
            "graph_sha256": hashlib.sha256(model.graph.SerializeToString()).hexdigest(),
            "model_sha256": hashlib.sha256(serialized).hexdigest(),
            "parquet_sha256": hashlib.sha256(output.getvalue()).hexdigest(),
        }}))
    """
    expected = json.loads(_probe(sys.executable, code))
    actual = json.loads(_probe(sys.executable, code))
    assert actual["ir"] == 11
    assert actual == expected
