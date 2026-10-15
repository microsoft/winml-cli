# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Native DXCGC compile/record/submit path without Windows ML Runtime API."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import click
import numpy as np


_TYPES = {
    np.dtype("bool"): 1,
    np.dtype("int8"): 4,
    np.dtype("int16"): 5,
    np.dtype("int32"): 6,
    np.dtype("int64"): 7,
    np.dtype("uint8"): 10,
    np.dtype("uint16"): 11,
    np.dtype("uint32"): 12,
    np.dtype("uint64"): 13,
    np.dtype("float64"): 14,
    np.dtype("float32"): 15,
    np.dtype("float16"): 16,
}


def _samples(path: Path, iterations: int) -> dict[str, list[float]]:
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != iterations:
        raise ValueError(f"Expected {iterations} measurements, received {len(rows)}")
    result = {key: [float(row[key]) for row in rows] for key in ("gpu_ms", "host_ms")}
    if any(not math.isfinite(v) or v <= 0 for values in result.values() for v in values):
        raise ValueError("Native host returned invalid timings")
    return result


def _io_policy(work: Path) -> str:
    """Require acknowledgement from the actual executable, not a CLI label."""
    path = work / "io-policy.txt"
    if not path.is_file():
        raise ValueError("Native host has no I/O policy acknowledgement; rebuild the native host")
    policy = path.read_text(encoding="utf-8").strip()
    if policy != "per_call_cpu_ownership_and_binding":
        raise ValueError("Native host I/O policy mismatch; rebuild the native host")
    return policy


def run_dxcgc(ctx: click.Context) -> None:
    """Validate arguments and invoke the native host outside Python timing."""
    p = ctx.params
    allowed = {
        "model",
        "runtime",
        "dxcgc_host",
        "dxcgc_dir",
        "dxcgc_sdk_path",
        "dxcgc_sdk_version",
        "dxcgc_adapter",
        "disable_d3d_mlir_programs",
        "iterations",
        "warmup",
        "input_data",
        "output",
        "overwrite",
        "verbose",
        "quiet",
        "no_color",
    }
    unsupported = [
        name
        for name in p
        if name not in allowed
        and ctx.get_parameter_source(name) not in (None, click.core.ParameterSource.DEFAULT)
    ]
    if unsupported:
        raise click.UsageError(
            "DXCGC does not support: "
            + ", ".join("--" + name.replace("_", "-") for name in unsupported)
        )
    if sys.platform != "win32":
        raise click.ClickException("DXCGC requires Windows")
    model = Path(p["model"]).resolve()
    if model.suffix.lower() not in (".mlir", ".mlirbc") or not model.is_file():
        raise click.UsageError("DXCGC requires a local .mlir or .mlirbc file")
    spec = importlib.util.find_spec("windowsml")
    if spec is None or not spec.submodule_search_locations:
        raise click.ClickException("The installed windowsml Runtime package is required")
    runtime_dir = Path(next(iter(spec.submodule_search_locations))) / "lib"
    if p["dxcgc_dir"] is None:
        p["dxcgc_dir"] = runtime_dir
    if p["dxcgc_sdk_path"] is None:
        p["dxcgc_sdk_path"] = runtime_dir
    for name in ("dxcgc_host", "input_data"):
        if p[name] is None:
            raise click.UsageError("--" + name.replace("_", "-") + " is required for DXCGC")
    output = (p["output"] or Path("dxcgc-perf.json")).resolve()
    output_arrays = output.with_suffix(".outputs.npz")
    if (output.exists() or output_arrays.exists()) and not p["overwrite"]:
        raise click.ClickException(f"Output exists: {output}; use --overwrite")
    if {output, output_arrays} & {model, p["input_data"].resolve()}:
        raise click.UsageError("Output must not overwrite model or inputs")
    if not (p["dxcgc_dir"] / "dxcgc.dll").is_file():
        raise click.UsageError("--dxcgc-dir must contain dxcgc.dll")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="winml-dxcgc-") as temporary:
            work = Path(temporary)
            with np.load(p["input_data"], allow_pickle=False) as archive:
                arrays = [archive[name] for name in archive.files]
                if not arrays:
                    raise ValueError("Input archive is empty")
                for i, array in enumerate(arrays):
                    if array.dtype not in _TYPES:
                        raise ValueError(f"Unsupported input dtype: {array.dtype}")
                    np.ascontiguousarray(array).tofile(work / f"input-{i}.bin")
                    (work / f"input-{i}.txt").write_text(
                        " ".join(map(str, [_TYPES[array.dtype], array.ndim, *array.shape])),
                        encoding="utf-8",
                    )
                (work / "input-count.txt").write_text(str(len(arrays)), encoding="utf-8")
            command = [
                str(p["dxcgc_host"].resolve()),
                str(model),
                str(p["dxcgc_dir"].resolve()),
                str(p["dxcgc_sdk_path"].resolve()),
                str(p["dxcgc_sdk_version"]),
                str(p["dxcgc_adapter"]),
                str(p["iterations"]),
                str(p["warmup"]),
                str(work),
            ]
            if p["disable_d3d_mlir_programs"]:
                command.append("--disable-d3d-mlir-programs")
            command = [sys.executable, str(Path(__file__).with_name("_dxcgc_worker.py")), *command]
            process = subprocess.run(  # noqa: S603 - explicit local executable; no shell
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=600,
            )
            if process.returncode:
                raise ValueError(
                    f"Native DXCGC host failed ({process.returncode}): " + process.stderr[-8000:]
                )
            samples = _samples(work / "samples.csv", p["iterations"])
            policy = _io_policy(work)
            if (work / "driver-policy.txt").read_text(encoding="utf8") != "python_ctypes_per_call":
                raise ValueError("Native Python driver acknowledgement is missing")
            info = (work / "device.tsv").read_text(encoding="utf-8").splitlines()
            outputs = {}
            reverse = {value: key for key, value in _TYPES.items()}
            for line in (work / "io.tsv").read_text(encoding="utf-8").splitlines():
                kind, index, dtype, shape, _name = line.split(chr(9))
                if kind != "output":
                    continue
                shape_tuple = tuple(int(d) for d in shape.split(",") if d)
                array = np.fromfile(work / f"output-{index}.bin", dtype=reverse[int(dtype)])
                array = array.reshape(shape_tuple)
                if not np.isfinite(array).all():
                    raise ValueError("Native DXCGC produced non-finite outputs")
                outputs[index] = array
            if not outputs:
                raise ValueError("Native DXCGC produced no outputs")
            report = {
                "runtime": "dxcgc",
                "disable_d3d_mlir_programs": p["disable_d3d_mlir_programs"],
                "model": str(model),
                "adapter": info[0],
                "adapter_luid": info[2],
                "compile_ms": float(info[1]),
                "iterations": p["iterations"],
                "warmup": p["warmup"],
                "io_mode": "cpu-roundtrip",
                "io_binding_policy": policy,
                "call_driver": "python_ctypes_per_call",
                "measurement": {
                    "gpu_ms": "D3D12 timestamps around Record, excluding upload/readback",
                    "host_ms": (
                        "Per-call CPU input ownership/copy, argument rebinding, upload, "
                        "command recording, submission, "
                        "GPU fence completion, readback and caller-owned CPU output copy"
                    ),
                },
                "samples": samples,
                "mean_gpu_ms": float(np.mean(samples["gpu_ms"])),
                "mean_host_ms": float(np.mean(samples["host_ms"])),
                "outputs_finite": True,
                "provenance": {
                    str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (
                        model,
                        p["input_data"].resolve(),
                        p["dxcgc_host"].resolve(),
                        p["dxcgc_host"].resolve().with_name("winml-dxcgc-python.dll"),
                        Path(__file__).with_name("_dxcgc_worker.py"),
                        (p["dxcgc_dir"] / "dxcgc.dll").resolve(),
                    )
                },
                "limitations": [
                    "Input arrays use NPZ order; names are not used for binding",
                    "Embedded weights only; one static program; buffer bindings",
                ],
            }
            np.savez(output_arrays, **outputs)
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            click.echo(
                f"DXCGC GPU: {report['mean_gpu_ms']:.4f} ms; "
                f"host: {report['mean_host_ms']:.4f} ms; {output}"
            )
    except (OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        raise click.ClickException(str(error)) from error
