# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Prebuilt ONNX benchmarking without ORT, Torch or model-export wrappers."""

from __future__ import annotations

import json
import math
import platform
import statistics
import time
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
        "ep",
        "device",
        "device_luid",
        "iterations",
        "warmup",
        "warmup_seconds",
        "etw_tracing",
        "cgc_lowering_target",
        "allow_cpu_fallback",
        "monitor",
        "skip_build",
        "no_compile",
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
    if params.get("ep") is not None and params.get("runtime") != "winml-ort":
        raise click.UsageError("Explicit EP selection requires --runtime winml-ort.")
    if (
        params.get("monitor")
        or not params.get("skip_build", True)
        or not params.get("no_compile", True)
    ):
        raise click.UsageError(
            "Native Arm64 Runtime classic does not support monitoring or EP build/compile options."
        )
    if params.get("cgc_lowering_target") not in (None, "auto"):
        raise click.UsageError(
            "Native Arm64 Runtime classic currently supports only automatic CGC lowering."
        )
    if not 0 <= params.get("warmup_seconds", 0) < float("inf"):
        raise click.UsageError("--warmup-seconds must be finite and nonnegative.")
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

    companions = (
        get_external_data_files(path)
        if path.suffix.lower() == ".onnx"
        else [p.name for p in path.parent.glob("*.safetensors")] + [path.stem + "_metadata.json"]
    )
    protected = {path, *((path.parent / item).resolve() for item in companions)}
    for key in ("input_data", "shape_config_path"):
        if params[key] is not None:
            protected.add(Path(params[key]).resolve())
    if output.resolve() in protected:
        raise click.UsageError("--output must not overwrite a model, its weights or input data.")
    trace = params.get("etw_tracing")
    if trace is not None:
        trace = trace.resolve()
        if trace.exists() and any(trace.iterdir()):
            raise click.UsageError("--etw-tracing requires a fresh empty directory.")
        if output.resolve().is_relative_to(trace) or any(
            p == trace or p.is_relative_to(trace) for p in protected
        ):
            raise click.UsageError("--etw-tracing must not contain model or input files.")
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
    trace = params.get("etw_tracing")
    if trace is None:
        result = _execute_session(params, path, model)
    else:
        from ..session.gpu_capture import GpuCapture

        with GpuCapture(trace.resolve()) as capture:
            capture.execution_path = (
                "winmlcgep" if params.get("runtime") == "winml-ort" else "runtime-api"
            )
            capture.allow_cpu_fallback = params.get("allow_cpu_fallback", False)
            result = _execute_session(params, path, model)
        result["runtime_execution_diagnostics"] = result["gpu_execution"]
        # Missing/lost GPU events fail here; never substitute Host samples.
        result["gpu_execution"] = capture.finish()
        gpu = result["gpu_execution"]
        samples = gpu.get("samples_ms", [])
        if (
            len(samples) != params["iterations"]
            or gpu.get("valid_runs") != params["iterations"]
            or gpu.get("events_lost", 0)
            or gpu.get("missing_runs", 0)
            or not all(math.isfinite(x) and x > 0 for x in samples)
        ):
            raise click.ClickException(
                "GPU capture did not produce every requested inference sample."
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if params["output_format"] == "json":
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(f"Runtime CGC: {result['latency_ms']['mean']:.3f} ms")
    click.echo(f"Results saved to: {output}", err=True)


def _run_iterations(session, feeds, *, warmup, iterations, warmup_seconds):
    import numpy as np

    def run():
        outputs = session.run(feeds)
        if not all(np.isfinite(value).all() for value in outputs.values()):
            raise click.ClickException("Runtime produced non-finite output values.")

    with session.perf(warmup=warmup) as window:
        started = time.perf_counter()
        count = 0
        while count < warmup or time.perf_counter() - started < warmup_seconds:
            # Runtime GPU markers consult this boundary before every run.
            window.stats.warmup = count + 1
            run()
            count += 1
        evidence = {
            "count": count,
            "duration_ms": (time.perf_counter() - started) * 1000,
            "minimum_count": warmup,
            "minimum_seconds": warmup_seconds,
        }
        for _ in range(iterations):
            run()
    return window.stats, evidence


def _execute_session(params: dict, path: Path, model: str) -> dict:
    import psutil

    from ..onnx import get_io_config
    from ..sysinfo import open_gpu_adapter
    from .perf import BenchmarkConfig, BenchmarkResult

    shapes = {}
    if params["shape_config_path"] is not None:
        shapes = json.loads(params["shape_config_path"].read_text(encoding="utf-8"))
        if not isinstance(shapes, dict):
            raise click.UsageError("--shape-config must contain a JSON object.")
    is_onnx = path.suffix.lower() == ".onnx"
    feeds = _inputs(get_io_config(path), params["input_data"], shapes) if is_onnx else None
    process = psutil.Process()
    rss = {"before_load": process.memory_info().rss} if params["memory"] else {}
    # device_luid_option validates and returns the canonical string, not an int.
    adapter = open_gpu_adapter(params["device_luid"])
    try:
        info = adapter.info
        selected_luid = int(info.luid.replace("0x", "").replace("_", ""), 16)
    finally:
        adapter.close()
    is_ort = params.get("runtime") == "winml-ort"
    ep_name, ep_source = params.get("ep") or (None, None)
    if is_ort:
        from ..session import EPDeviceTarget, WinMLEPRegistry, WinMLSession, resolve_device
        from ..session.runtime_session import _symbolic_dimensions_for_inputs
        from .perf import _ort_options_for_dimensions

        target = resolve_device(EPDeviceTarget(ep=ep_name, device="gpu", source=ep_source))
        binding = WinMLEPRegistry.instance().auto_device(target, device_luid=info.luid)
        if (
            binding.device.adapter_luid != selected_luid
            or binding.device.ep_name != "WinMLCGExecutionProvider"
        ):
            raise click.ClickException(
                "CGEP registry binding differs from the requested GPU/provider."
            )
        dimensions = _symbolic_dimensions_for_inputs(
            get_io_config(path), {name: list(value.shape) for name, value in feeds.items()}
        )
        session = WinMLSession(
            path,
            ep_device=binding,
            session_options=lambda: _ort_options_for_dimensions(dimensions),
        )
    else:
        from ..session.runtime_session import WinMLRuntimeSession

        session = WinMLRuntimeSession(
            path, backend="cgc", device_luid=selected_luid, capture_diagnostics=True
        )
    try:
        if feeds is not None and not is_ort:
            session.set_input_shapes({name: list(value.shape) for name, value in feeds.items()})
        session.compile()
        io = session.io_config
        if feeds is None:
            io = dict(io)
            io.setdefault("input_symbolic_shapes", [[None] * len(s) for s in io["input_shapes"]])
            io.setdefault("value_ranges", {})
            feeds = _inputs(io, params["input_data"], shapes)
        if rss:
            rss["after_compile"] = process.memory_info().rss
        stats, warmup_evidence = _run_iterations(
            session,
            feeds,
            warmup=params["warmup"],
            iterations=params["iterations"],
            warmup_seconds=params.get("warmup_seconds", 0),
        )
        config = BenchmarkConfig(
            model_id=model,
            runtime="winml-ort" if is_ort else "winml-runtime",
            backend=None if is_ort else "cgc",
            ep=ep_name,
            ep_source=ep_source,
            device="gpu",
            device_luid=selected_luid,
            iterations=params["iterations"],
            warmup=params["warmup"],
            warmup_seconds=params.get("warmup_seconds", 0),
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
                statistics.mean(stats.all_samples_ms[: stats.warmup]) if stats.warmup else 0.0
            ),
            raw_samples_ms=stats.samples_ms,
            actual_device="gpu",
            actual_ep=session.ep_name if is_ort else None,
            running_model_path=str(path),
            effective_batch_size=batch,
            batches_per_sec=1000 / stats.mean_ms,
            samples_per_sec=batch * 1000 / stats.mean_ms,
        ).to_dict()
        result["benchmark_info"].update(
            cgc_lowering_target=params.get("cgc_lowering_target"),
            python_architecture=platform.machine(),
            python_version=platform.python_version(),
            windowsml_version=version("windowsml"),
            adapter_name=info.name,
        )
        result["output_validation"] = {"all_finite": True, "accuracy_checked": False}
        result["warmup_measurement"] = warmup_evidence
        result["gpu_execution"] = None if is_ort else session.execution_diagnostics
        if rss:
            rss["after_inference"] = process.memory_info().rss
            result["process_memory"] = {"rss_checkpoints_bytes": rss}
            result["device_memory"] = {
                "available": False,
                "reason": "Device memory collection is not enabled by this profile.",
            }
        return result
    finally:
        session.close()
