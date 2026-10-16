# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Python-driven in-process native DXCGC loop (isolated once per benchmark)."""

import csv
import ctypes
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    host, *arguments = sys.argv[1:]
    work = Path(arguments[7])
    library = ctypes.CDLL(str(Path(host).with_name("winml-dxcgc-python.dll")))
    abi = library.DxcgcPythonHostAbiVersion
    abi.argtypes = []
    abi.restype = ctypes.c_uint
    if abi() != 3:
        raise RuntimeError("Native DXCGC host ABI mismatch; rebuild the native host")
    step_type = ctypes.CFUNCTYPE(
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_size_t,
    )
    driver_type = ctypes.CFUNCTYPE(
        ctypes.c_int, step_type, ctypes.c_void_p, ctypes.c_int, ctypes.c_int
    )
    errors, samples = [], []
    types = {
        1: "bool",
        4: "int8",
        5: "int16",
        6: "int32",
        7: "int64",
        10: "uint8",
        11: "uint16",
        12: "uint32",
        13: "uint64",
        14: "float64",
        15: "float32",
        16: "float16",
    }

    def drive(step: Any, context: Any, iterations: int, warmup: int) -> int:
        try:
            specs = []
            for line in (work / "io.tsv").read_text(encoding="utf8").splitlines():
                kind, index, dtype, shape, _name = line.split(chr(9))
                if kind == "output":
                    specs.append(
                        (index, tuple(int(x) for x in shape.split(",") if x), types[int(dtype)])
                    )
            input_specs = []
            for line in (work / "io.tsv").read_text().splitlines():
                kind, index, dtype, shape, _ = line.split(chr(9))
                if kind == "input":
                    input_specs.append(
                        (index, tuple(int(x) for x in shape.split(",") if x), types[int(dtype)])
                    )
            manifest = work / "input-sequence.json"
            sequence = []
            if manifest.exists():
                for path in json.loads(manifest.read_text()):
                    with np.load(path, allow_pickle=False) as archive:
                        sequence.append(
                            [np.array(archive[k], copy=True, order="C") for k in archive.files]
                        )
            else:
                sequence = [
                    [
                        np.fromfile(work / f"input-{index}.bin", dtype=dtype).reshape(shape)
                        for index, shape, dtype in input_specs
                    ]
                ]
            if not sequence:
                raise ValueError("Empty input sequence")
            for batch in sequence:
                if len(batch) != len(input_specs):
                    raise ValueError("Input count mismatch")
                for array, (_, input_shape, input_dtype) in zip(batch, input_specs, strict=True):
                    if array.shape != input_shape or array.dtype != np.dtype(input_dtype):
                        raise ValueError("Input shape or dtype mismatch")
            gpu = ctypes.c_double()
            last = None
            for i in range(iterations + warmup):
                batch = sequence[i % len(sequence)]
                start = time.perf_counter()
                outputs = [np.empty(shape, dtype=dtype) for _, shape, dtype in specs]
                pointers = (ctypes.c_void_p * len(outputs))(*(a.ctypes.data for a in outputs))
                sizes = (ctypes.c_size_t * len(outputs))(*(a.nbytes for a in outputs))
                inputs = (ctypes.c_void_p * len(batch))(*(a.ctypes.data for a in batch))
                input_sizes = (ctypes.c_size_t * len(batch))(*(a.nbytes for a in batch))
                code = step(
                    context,
                    ctypes.byref(gpu),
                    pointers,
                    sizes,
                    len(outputs),
                    inputs,
                    input_sizes,
                    len(batch),
                )
                elapsed = (time.perf_counter() - start) * 1000
                if code:
                    raise RuntimeError("Native inference failed")
                if i >= warmup:
                    samples.append((None if gpu.value == -1 else gpu.value, elapsed))
                last = outputs
            if last is None:
                raise ValueError("No inference iterations executed")
            for (index, _, _), array in zip(specs, last, strict=True):
                array.tofile(work / f"output-{index}.bin")
            return 0
        except Exception as error:
            errors.append(str(error))
            return 1

    callback = driver_type(drive)
    argv = (ctypes.c_wchar_p * (len(arguments) + 1))(host, *arguments)
    entry = library.RunPythonDriven
    entry.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_wchar_p), driver_type]
    entry.restype = ctypes.c_int
    code = entry(len(argv), argv, callback)
    if code:
        raise RuntimeError("; ".join(errors) or f"Native setup failed ({code})")
    # Native setup closes its CSV before this write. No file I/O inside timing.
    with (work / "samples.csv").open("w", newline="", encoding="utf8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["gpu_ms", "host_ms"])
        writer.writerows(samples)
    (work / "driver-policy.txt").write_text("python_ctypes_per_call", encoding="utf8")


if __name__ == "__main__":
    main()
