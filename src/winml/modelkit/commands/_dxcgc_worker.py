"""Python-driven in-process native DXCGC loop (isolated once per benchmark)."""
import csv
import ctypes
import sys
import time
from pathlib import Path

import numpy as np


def main():
    host, *arguments = sys.argv[1:]
    work = Path(arguments[7])
    library = ctypes.CDLL(str(Path(host).with_name("winml-dxcgc-python.dll")))
    step_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_double),
                                ctypes.POINTER(ctypes.c_void_p))
    driver_type = ctypes.CFUNCTYPE(ctypes.c_int, step_type, ctypes.c_void_p,
                                  ctypes.c_int, ctypes.c_int)
    errors, samples = [], []
    types = {1: "bool", 4: "int8", 5: "int16", 6: "int32", 7: "int64",
             10: "uint8", 11: "uint16", 12: "uint32", 13: "uint64",
             14: "float64", 15: "float32", 16: "float16"}

    def drive(step, context, iterations, warmup):
        try:
            specs = []
            for line in (work / "io.tsv").read_text(encoding="utf8").splitlines():
                kind, index, dtype, shape, _name = line.split(chr(9))
                if kind == "output":
                    specs.append((index, tuple(int(x) for x in shape.split(",") if x),
                                  types[int(dtype)]))
            gpu = ctypes.c_double()
            last = None
            for i in range(iterations + warmup):
                start = time.perf_counter()
                outputs = [np.empty(shape, dtype=dtype) for _, shape, dtype in specs]
                pointers = (ctypes.c_void_p * len(outputs))(*(a.ctypes.data for a in outputs))
                code = step(context, ctypes.byref(gpu), pointers)
                elapsed = (time.perf_counter() - start) * 1000
                if code:
                    raise RuntimeError("Native inference failed")
                if i >= warmup:
                    samples.append((gpu.value, elapsed))
                last = outputs
            for (index, _, _), array in zip(specs, last, strict=True):
                array.tofile(work / f"output-{index}.bin")
            return 0
        except BaseException as error:
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
