# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Prebuilt ONNX benchmarking without ORT, Torch or model-export wrappers."""

from __future__ import annotations

import json
import platform
import statistics
from importlib.metadata import version
from pathlib import Path
from typing import Any

import click

from ..utils import cli as cli_utils


def _inputs(io: dict[str, Any], input_data: Path | None, shapes: dict) -> dict:
    import numpy as np

    arrays = {}
    if input_data is not None:
        with np.load(input_data, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if set(arrays) != set(io["input_names"]):
            raise click.UsageError("--input-data names must match the ONNX inputs exactly.")
    rng = np.random.default_rng(0)
    for name, dtype, shape, symbolic in zip(
        io["input_names"],
        io["input_types"],
        io["input_shapes"],
        io["input_symbolic_shapes"],
        strict=True,
    ):
        wanted = shapes.get(
            name,
            [
                size if size is not None else shapes.get(symbol)
                for size, symbol in zip(shape, symbolic, strict=True)
            ],
        )
        if not isinstance(wanted, (list, tuple)) or len(wanted) != len(shape):
            raise click.UsageError(f"Shape override for {name!r} must match its rank.")
        if name in arrays:
            array = arrays[name]
            if array.dtype != np.dtype(dtype):
                raise click.UsageError(f"Input {name!r} requires dtype {dtype}, got {array.dtype}.")
            if len(array.shape) != len(wanted) or any(
                size is not None and size != actual
                for size, actual in zip(wanted, array.shape, strict=False)
            ):
                raise click.UsageError(f"Input {name!r} has incompatible shape {array.shape}.")
            continue
        if not isinstance(wanted, (list, tuple)) or any(
            not isinstance(size, int) or size <= 0 for size in wanted
        ):
            raise click.UsageError(f"Provide --input-data or --shape-config for {name!r}.")
        if np.issubdtype(dtype, np.integer):
            limits = io["value_ranges"].get(name)
            if limits is None:
                raise click.UsageError(
                    f"Integer input {name!r} has no value range; provide --input-data."
                )
            array = rng.integers(*limits, size=wanted, dtype=dtype)
        elif np.issubdtype(dtype, np.floating):
            low, high = io["value_ranges"].get(name, (-1.0, 1.0))
            array = rng.uniform(low, high, size=wanted).astype(dtype)
        elif np.dtype(dtype) == np.dtype(bool):
            array = np.zeros(wanted, dtype=bool)
        else:
            raise click.UsageError(f"Unsupported synthetic input dtype {dtype}; use --input-data.")
        arrays[name] = array
    return arrays


def run_runtime_classic(ctx: click.Context, model: str) -> None:
    """Run source ONNX through the existing Runtime session on native Arm64."""
    allowed = {
        "model",
        "runtime",
        "backend",
        "device",
        "device_luid",
        "iterations",
        "warmup",
        "memory",
        "input_data",
        "shape_config_path",
        "output",
        "output_format",
        "overwrite",
        "verbose",
        "quiet",
    }
    unsupported = [
        name
        for name in ctx.params
        if name not in allowed
        and ctx.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE
    ]
    if unsupported:
        raise click.UsageError(
            "Native Arm64 Runtime classic does not support these options: "
            + ", ".join(name.replace("_", "-") for name in unsupported)
        )
    params = ctx.params
    if params["iterations"] <= 0 or params["warmup"] < 0:
        raise click.UsageError("--iterations must be positive and --warmup must be nonnegative.")
    if params["output_format"] not in {"text", "json"}:
        raise click.UsageError("Native Arm64 Runtime classic supports --format text or json.")
    if params["device"].lower() not in {"auto", "gpu"}:
        raise click.UsageError("The CGC Runtime classic backend requires --device gpu or auto.")
    path = Path(model).resolve()
    if not path.is_file():
        raise click.UsageError(f"ONNX model not found: {path}")
    from .perf import generate_output_path

    output = params["output"] or generate_output_path(model)
    cli_utils.guard_output(output, params["overwrite"])
    try:
        from ..onnx import (
            get_external_data_files,
            get_io_config,  # noqa: F401 - validate optional dependencies early
        )
    except ImportError as exc:
        raise click.ClickException(
            "Install classic Runtime dependencies: pip install -e .[runtime-classic]. " + str(exc)
        ) from exc

    protected = {path, *((path.parent / item).resolve() for item in get_external_data_files(path))}
    for key in ("input_data", "shape_config_path"):
        if params[key] is not None:
            protected.add(Path(params[key]).resolve())
    if output.resolve() in protected:
        raise click.UsageError("--output must not overwrite a model, its weights or input data.")
    try:
        _execute(params, path, output, model)
    except (ImportError, ModuleNotFoundError) as exc:
        raise click.ClickException(
            "Classic Runtime requires the runtime-classic extra: "
            "pip install -e .[runtime-classic]. " + str(exc)
        ) from exc
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc


def _execute(params: dict, path: Path, output: Path, model: str) -> None:
    import numpy as np
    import psutil

    from ..onnx import get_io_config
    from ..session.runtime_session import WinMLRuntimeSession
    from ..sysinfo import open_gpu_adapter
    from .perf import BenchmarkConfig, BenchmarkResult

    shapes = {}
    if params["shape_config_path"] is not None:
        shapes = json.loads(params["shape_config_path"].read_text(encoding="utf-8"))
        if not isinstance(shapes, dict):
            raise click.UsageError("--shape-config must contain a JSON object.")
    source_io = get_io_config(path)
    feeds = _inputs(source_io, params["input_data"], shapes)
    process = psutil.Process()
    rss = {"before_load": process.memory_info().rss} if params["memory"] else {}
    luid = params["device_luid"]
    requested = None if luid is None else f"0x{luid >> 32:08X}_0x{luid & 0xFFFFFFFF:08X}"
    adapter = open_gpu_adapter(requested)
    try:
        info = adapter.info
        selected_luid = int(info.luid.replace("0x", "").replace("_", ""), 16)
    finally:
        adapter.close()
    session = WinMLRuntimeSession(
        path, backend="cgc", device_luid=selected_luid, capture_diagnostics=True
    )
    try:
        session.set_input_shapes({name: list(value.shape) for name, value in feeds.items()})
        session.compile()
        io = session.io_config
        if rss:
            rss["after_compile"] = process.memory_info().rss
        with session.perf(warmup=params["warmup"]) as window:
            for _ in range(params["warmup"] + params["iterations"]):
                outputs = session.run(feeds)
                if not all(np.isfinite(value).all() for value in outputs.values()):
                    raise click.ClickException("Runtime produced non-finite output values.")
        stats = window.stats
        config = BenchmarkConfig(
            model_id=model,
            runtime="winml-runtime",
            backend="cgc",
            device="gpu",
            device_luid=selected_luid,
            iterations=params["iterations"],
            warmup=params["warmup"],
        )
        batch = next(iter(feeds.values())).shape[0] if next(iter(feeds.values())).ndim else 1
        result = BenchmarkResult(
            config=config,
            input_names=io["input_names"],
            input_shapes=[list(feeds[name].shape) for name in io["input_names"]],
            input_types=[str(dtype) for dtype in io["input_types"]],
            output_names=io["output_names"],
            output_shapes=io["output_shapes"],
            mean_ms=stats.mean_ms,
            min_ms=stats.min_ms,
            max_ms=stats.max_ms,
            p50_ms=stats.p50_ms,
            p90_ms=stats.p90_ms,
            p95_ms=stats.p95_ms,
            p99_ms=stats.p99_ms,
            std_ms=statistics.pstdev(stats.samples_ms),
            warmup_mean_ms=(
                statistics.mean(stats.all_samples_ms[: params["warmup"]])
                if params["warmup"]
                else 0.0
            ),
            raw_samples_ms=stats.samples_ms,
            actual_device="gpu",
            actual_ep=None,
            running_model_path=str(path),
            effective_batch_size=batch,
            batches_per_sec=1000 / stats.mean_ms,
            samples_per_sec=batch * 1000 / stats.mean_ms,
        ).to_dict()
        result["benchmark_info"].update(
            python_architecture=platform.machine(),
            python_version=platform.python_version(),
            windowsml_version=version("windowsml"),
            adapter_name=info.name,
        )
        result["output_validation"] = {"all_finite": True, "accuracy_checked": False}
        result["gpu_execution"] = session.execution_diagnostics
        if rss:
            rss["after_inference"] = process.memory_info().rss
            result["process_memory"] = {"rss_checkpoints_bytes": rss}
            result["device_memory"] = {
                "available": False,
                "reason": "Device memory collection is not enabled by this profile.",
            }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if params["output_format"] == "json":
            click.echo(json.dumps(result, indent=2))
        else:
            click.echo(
                f"Runtime CGC on {info.name}: {stats.mean_ms:.3f} ms "
                f"({stats.count} measured iterations)"
            )
        click.echo(f"Results saved to: {output}", err=True)
    finally:
        session.close()
