# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""QAIRT SDK compilation script - executed in isolated venv-wmk subprocess.

This script is invoked by qnn_compiler._compile_qairt() and runs in a separate
Python 3.10 virtual environment with QAIRT SDK dependencies installed.

Errors are written to stderr. Return code 0 indicates success.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def setup_qairt_environment(qairt_root: Path) -> None:
    """Configure environment variables for QAIRT SDK.

    Args:
        qairt_root: Path to QAIRT SDK root directory
    """
    os.environ["QAIRT_SDK_ROOT"] = str(qairt_root)
    os.environ["QNN_SDK_ROOT"] = str(qairt_root)

    # Add library paths
    x64_lib_path = qairt_root / "lib" / "x86_64-windows-msvc"
    arm64_lib_path = qairt_root / "lib" / "aarch64-windows-msvc"
    os.environ["PATH"] = f"{x64_lib_path};{arm64_lib_path};{os.environ.get('PATH', '')}"

    # Add QAIRT Python modules to path
    sys.path.insert(0, str(qairt_root / "lib" / "python"))


def extract_input_specs(model_path: Path) -> list[dict]:
    """Extract input tensor specifications from ONNX model.

    Args:
        model_path: Path to ONNX model

    Returns:
        List of input specs with name, shape, and dtype
    """
    import numpy as np
    import onnx

    from ...onnx import load_onnx

    model = load_onnx(model_path, validate=False)

    dtype_map = {
        onnx.TensorProto.FLOAT:   np.float32,
        onnx.TensorProto.FLOAT16: np.float16,
        onnx.TensorProto.INT8:    np.int8,
        onnx.TensorProto.INT32:   np.int32,
        onnx.TensorProto.INT64:   np.int64,
        onnx.TensorProto.UINT8:   np.uint8,
    }

    specs = []
    initializer_names = {init.name for init in model.graph.initializer}

    for inp in model.graph.input:
        if inp.name in initializer_names:
            continue

        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value > 0 else 1)

        elem_type = inp.type.tensor_type.elem_type
        dtype_np = dtype_map.get(elem_type, np.float32)

        specs.append({
            "name": inp.name,
            "shape": tuple(shape),
            "dtype": np.dtype(dtype_np),
        })

    return specs


def compile_model(
    model_path: Path,
    output_dir: Path,
    layout: str | None = None,
    optimization_level: int = 3,
    hvx_threads: int = 4,
    vtcm_size_mb: int = 8,
) -> None:
    """Compile ONNX model to QNN context binary using QAIRT SDK.

    Outputs {model_stem}.bin and {model_stem}_cache_info.json to output_dir.

    Args:
        model_path: Path to input ONNX model
        output_dir: Output directory for compiled files
        layout: JSON string mapping input names to layouts, e.g. '{"input": "NCHW"}'.
        optimization_level: HTP optimization level (1-3)
        hvx_threads: Number of HVX threads
        vtcm_size_mb: VTCM size in MB
    """
    import qairt
    from qairt.api.compiler.config import CompileConfig
    from qairt.api.converter.converter_config import InputTensorConfig

    # Extract input specs from ONNX
    input_specs = extract_input_specs(model_path)
    layout_map = json.loads(layout) if layout else {}
    input_configs = [
        InputTensorConfig(name=spec["name"], shape=spec["shape"], layout=layout_map.get(spec["name"]), datatype=spec["dtype"])
        for spec in input_specs
    ]

    # Convert ONNX to DLC (Model object)
    model = qairt.convert(str(model_path), input_tensor_config=input_configs, preserve_io_datatype="all")

    # Configure HTP compilation
    try:
        from qairt.api.common.backends.htp.config import HtpGraphConfig

        graph_config = HtpGraphConfig(
            name=model.name,
            optimization_type=optimization_level,
            hvx_threads=hvx_threads,
            vtcm_size_in_mb=vtcm_size_mb,
        )
        compile_config = CompileConfig(
            backend="HTP",
            graph_custom_configs=[graph_config],
        )
    except ImportError:
        compile_config = CompileConfig(backend="HTP")

    # Compile to context binary
    compiled_model = qairt.compile(model, config=compile_config)

    # Save context binary
    output_dir.mkdir(parents=True, exist_ok=True)
    compiled_model.save(str(output_dir))


def main() -> int:
    """Main entry point for QAIRT compilation subprocess."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--qairt-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layout", type=str, default=None, help='e.g. \'{"input1": "NCHW", "input2": "NFC"}\'')
    parser.add_argument("--optimization-level", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--hvx-threads", type=int, default=4)
    parser.add_argument("--vtcm-size", type=int, default=8)

    args = parser.parse_args()

    # Setup environment before importing QAIRT modules
    setup_qairt_environment(args.qairt_root)

    # Compile and return result
    try:
        compile_model(
            args.model,
            args.output_dir,
            layout=args.layout,
            optimization_level=args.optimization_level,
            hvx_threads=args.hvx_threads,
            vtcm_size_mb=args.vtcm_size,
        )
        return 0
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
