# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""DXGI duration and effective-target validation regressions."""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from winml.modelkit.commands.perf import BenchmarkConfig, PerfBenchmark, perf


@pytest.mark.parametrize(
    "duration,iterations,expected",
    [
        (60, 200, 600),
        (1, 200, 200),
        (1, 1, 100),
        (None, 200, 200),
        (60, 700, 700),
    ],
)
def test_dxgi_honors_duration_and_iteration_minima(duration, iterations, expected):
    benchmark = PerfBenchmark(
        BenchmarkConfig(
            model_id="model.onnx",
            memory_backend="dxgi",
            warmup=0,
            iterations=iterations,
            duration=duration,
        )
    )
    count = 0

    def run(inputs):
        nonlocal count
        count += 1

    session = SimpleNamespace(run=run)
    benchmark._model = SimpleNamespace(_session=session)
    benchmark._inputs = {}
    context = SimpleNamespace(stats=object())
    with (
        patch(
            "winml.modelkit.commands.perf._native_warning_filtered_perf",
            return_value=nullcontext(context),
        ),
        patch("winml.modelkit.commands.perf.time.perf_counter", side_effect=lambda: count / 10),
    ):
        assert benchmark._run_benchmark_simple() is context.stats
    assert count == expected
    assert benchmark._steady_measurement["duration_ms"] == expected * 100


@pytest.mark.parametrize("seconds", ["inf", "+inf", "-inf", "nan"])
def test_warmup_seconds_rejects_nonfinite(seconds):
    result = CliRunner().invoke(perf, ["-m", "model.onnx", "--warmup-seconds", seconds])
    assert result.exit_code == 2
    assert "--warmup-seconds" in result.output


@pytest.mark.parametrize("seconds", ["0", "0.5"])
def test_warmup_seconds_accepts_finite(seconds):
    option = next(p for p in perf.params if p.name == "warmup_seconds")
    assert option.callback(None, option, option.type.convert(seconds, option, None)) == float(
        seconds
    )


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("override", [None, "gpu", "cpu"])
def test_dxgi_validates_effective_config_device(tmp_path, structured, override):
    config = tmp_path / "config.json"
    compile_config = (
        {"ep_device": {"ep": "winmlcg", "device": "gpu"}}
        if structured
        else {"execution_provider": "winmlcg", "device": "gpu"}
    )
    config.write_text(json.dumps({"compile": compile_config}), encoding="utf-8")
    model = tmp_path / "model.onnx"
    model.touch()
    args = [
        "-m",
        str(model),
        "--runtime",
        "winml-ort",
        "--config",
        str(config),
        "--memory-backend",
        "dxgi",
        "--etw-tracing",
        str(tmp_path / "trace"),
        "-o",
        str(tmp_path / "output.json"),
    ]
    if override:
        args.extend(["--device", override])
    # Stop at benchmark construction: this test must never query hardware/start ETW.
    constructor = Mock(side_effect=RuntimeError("reached benchmark"))
    with patch("winml.modelkit.commands.perf.PerfBenchmark", constructor):
        result = CliRunner().invoke(perf, args, obj={})
    if override == "cpu":
        assert result.exit_code == 2, result.output
        assert "DXGI memory measurement requires GPU" in result.output
        constructor.assert_not_called()
    else:
        constructor.assert_called_once()
        effective = constructor.call_args.args[0]
        assert effective.device == "gpu"
        assert effective.warmup >= 10
        assert effective.warmup_seconds >= 2
        assert effective.iterations >= 200
