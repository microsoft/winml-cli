# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for GGUF benchmarking in ``winml perf`` without native dependencies.

A fake session returning clock-generated :class:`LlamaGenerationTiming` objects
makes aggregation deterministic without importing Windows ML Tasks. The session is tested in
``tests/unit/session/test_llama_session.py``.
"""

from __future__ import annotations

import json
from io import StringIO
from itertools import accumulate, count, pairwise
from pathlib import Path
from random import Random
from statistics import fmean
from typing import Any
from unittest.mock import MagicMock

import click
import pytest
from click.testing import CliRunner
from rich.console import Console

from winml.modelkit.commands import _perf_gguf as perf_gguf
from winml.modelkit.commands._perf_gguf import (
    GGUFBenchmarkResult,
    GGUFPerfBenchmark,
    GGUFPerfConfig,
    display_gguf_report,
    gguf_output_path,
    is_gguf_model,
    run_gguf_perf,
    write_gguf_report,
)
from winml.modelkit.commands.perf import _resolve_runtime, perf
from winml.modelkit.session import (
    LlamaGenerationTiming,
    LlamaLoadError,
    LlamaNotInstalledError,
    LlamaSessionError,
)


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


def _timing(
    *,
    generated_tokens: int = 3,
    input_tokens: int = 4,
    scale: float = 1.0,
    stop_reason: str = "MAX_TOKENS",
) -> LlamaGenerationTiming:
    """Derive session spans from generated native-operation timestamps."""
    random = Random(generated_tokens + input_tokens)
    durations = [random.uniform(0.001, 0.1) * scale for _ in range(generated_tokens + 4)]
    reset_end, create_end, *after_create = accumulate(durations)
    arrivals = after_create[:generated_tokens]
    terminal, result_end = after_create[generated_tokens:]
    return LlamaGenerationTiming(
        input_tokens=input_tokens,
        generated_tokens=generated_tokens,
        reset_s=reset_end,
        stream_create_s=create_end - reset_end,
        first_token_s=arrivals[0] - reset_end if arrivals else None,
        decode_s=[end - start for start, end in pairwise(arrivals)],
        completion_s=terminal - (arrivals[-1] if arrivals else create_end),
        result_fetch_s=result_end - terminal,
        generation_s=result_end - reset_end,
        response_text="".join(chr(ord("a") + index) for index in range(generated_tokens)),
        stop_reason=stop_reason,
    )


class _FakeLlamaSession:
    """WinMLLlamaSession stand-in returning generated timings in order."""

    def __init__(
        self,
        timings: list[LlamaGenerationTiming],
        *,
        prompt_ids: list[int] | None = None,
        chat_template: bool = True,
    ) -> None:
        self._timings = list(timings)
        self._prompt_ids = prompt_ids if prompt_ids is not None else list(range(4))
        self._chat_template = chat_template
        self.chat_template_applied: bool | None = None
        self.context_length: int | None = None
        self.effective_device: str | None = None
        self.payload = None
        self.load_timings_ms = {
            name: float(index)
            for index, name in enumerate(
                (
                    "runtime_create_duration_ms",
                    "tokenizer_create_duration_ms",
                    "model_create_duration_ms",
                    "pipeline_build_duration_ms",
                    "task_create_duration_ms",
                ),
                start=1,
            )
        }
        self.calls: list[str] = []
        self.generate_kwargs: list[dict[str, Any]] = []
        self.closed = False

    def open_tokenizer(self) -> None:
        self.calls.append("open_tokenizer")

    def encode_prompt(self, prompt: str, *, apply_template: bool = True) -> list[int]:
        self.calls.append("encode_prompt")
        self.chat_template_applied = apply_template and self._chat_template
        return list(self._prompt_ids)

    def load(self, *, context_length: int | None = None) -> None:
        self.calls.append("load")
        self.context_length = context_length
        self.effective_device = "cpu"

    def generate_timed(self, prompt_token_ids: list[int], **kwargs: Any) -> LlamaGenerationTiming:
        self.generate_kwargs.append(kwargs)
        return self._timings.pop(0)

    def close(self) -> None:
        self.closed = True


def _config(tmp_path: Path, **overrides: Any) -> GGUFPerfConfig:
    values: dict[str, Any] = {
        "model_path": tmp_path / "tiny.gguf",
        "warmup": 1,
        "iterations": 2,
        "max_new_tokens": 3,
        "output_path": tmp_path / "report.json",
    }
    values.update(overrides)
    return GGUFPerfConfig(**values)


def _run_fake(
    tmp_path: Path, timings: list[LlamaGenerationTiming], **overrides: Any
) -> tuple[GGUFBenchmarkResult, _FakeLlamaSession]:
    session = _FakeLlamaSession(timings)
    ticks = count(step=0.001)
    benchmark = GGUFPerfBenchmark(
        _config(tmp_path, **overrides), session=session, clock=lambda: next(ticks)
    )
    return benchmark.run(), session


def _gguf(tmp_path: Path, name: str = "model.gguf") -> Path:
    model = tmp_path / name
    model.write_bytes(b"GGUF")
    return model


# ---------------------------------------------------------------------------
# Benchmark engine
# ---------------------------------------------------------------------------


class TestBenchmark:
    def test_sizes_context_from_prompt_and_loads_before_generating(self, tmp_path: Path) -> None:
        timings = [_timing() for _ in range(3)]
        result, session = _run_fake(tmp_path, timings)

        assert session.calls[:4] == ["open_tokenizer", "encode_prompt", "load", "encode_prompt"]
        assert session.context_length == timings[0].input_tokens + result.config.max_new_tokens
        assert all(
            kwargs["max_new_tokens"] == result.config.max_new_tokens
            for kwargs in session.generate_kwargs
        )
        assert not session.closed  # injected sessions belong to the caller

    def test_aggregates_timed_requests_only(self, tmp_path: Path) -> None:
        timings = [
            _timing(scale=100.0, stop_reason="EOS"),  # warmup, excluded
            *[_timing(scale=float(index)) for index in range(1, 3)],
        ]
        result, _session = _run_fake(tmp_path, timings)
        aggregate = result.aggregate

        assert aggregate["warmup_request_count"] == 1
        assert aggregate["timed_request_count"] == 2
        measured = timings[result.config.warmup :]
        assert aggregate["task_ttft_duration_ms"]["mean"] == pytest.approx(
            fmean(timing.first_token_s for timing in measured) * 1000
        )
        assert aggregate["task_generation_duration_ms"]["mean"] == pytest.approx(
            fmean(timing.generation_s for timing in measured) * 1000
        )
        assert aggregate["steady_state_tpot_ms"]["mean"] == pytest.approx(
            fmean(fmean(timing.decode_s) for timing in measured) * 1000
        )
        assert aggregate["steady_state_decode_tokens_per_second"]["mean"] == pytest.approx(
            fmean(len(timing.decode_s) / sum(timing.decode_s) for timing in measured)
        )
        assert aggregate["stop_reason_counts"] == {"MAX_TOKENS": 2}
        assert result.stop_reason == "MAX_TOKENS"
        assert (result.prompt_tokens, result.generated_tokens) == (
            measured[0].input_tokens,
            measured[0].generated_tokens,
        )

    def test_load_metrics_come_from_session_spans(self, tmp_path: Path) -> None:
        timings = [_timing(generated_tokens=1) for _ in range(3)]
        result, session = _run_fake(tmp_path, timings)

        for name, duration in session.load_timings_ms.items():
            assert result.load[name] == duration
        assert result.load["native_load_duration_ms"] == sum(
            session.load_timings_ms[name]
            for name in ("model_create_duration_ms", "pipeline_build_duration_ms")
        )
        assert result.load["weight_upload_duration_ms"] is None
        assert result.load["weight_upload_estimate_source"] == "unavailable"

    def test_request_timing_includes_stream_startup_and_terminal_work(self, tmp_path: Path) -> None:
        clock_step_s = 0.125
        ticks = count(step=clock_step_s)
        timing = _timing()
        session = _FakeLlamaSession([timing] * 3)
        benchmark = GGUFPerfBenchmark(_config(tmp_path), session=session, clock=lambda: next(ticks))
        result = benchmark.run()
        sample = result.requests[-1]

        assert sample.template_duration_ms == 0.0
        assert sample.tokenization_duration_ms == pytest.approx(clock_step_s * 1000)
        assert sample.generator_create_duration_ms == pytest.approx(timing.reset_s * 1000)
        assert sample.task_ttft_duration_ms == pytest.approx(timing.first_token_s * 1000)
        assert sample.task_generation_duration_ms == pytest.approx(timing.generation_s * 1000)
        assert sample.task_stream_create_duration_ms == pytest.approx(timing.stream_create_s * 1000)
        assert sample.task_stream_completion_duration_ms == pytest.approx(
            timing.completion_s * 1000
        )
        assert sample.result_fetch_duration_ms == pytest.approx(timing.result_fetch_s * 1000)
        assert sample.decode_token_durations_ms == pytest.approx(
            [duration * 1000 for duration in timing.decode_s]
        )
        assert sample.request_ttft_duration_ms == pytest.approx(
            (clock_step_s + timing.reset_s + timing.first_token_s) * 1000
        )
        assert sample.request_duration_ms == pytest.approx(
            (clock_step_s + timing.reset_s + timing.generation_s) * 1000
        )
        assert result.aggregate["cold_start_ttft_duration_ms"] == pytest.approx(
            result.load["session_load_duration_ms"] + result.requests[0].request_ttft_duration_ms
        )
        assert result.aggregate["cold_start_total_duration_ms"] == pytest.approx(
            result.load["session_load_duration_ms"] + result.requests[0].request_duration_ms
        )

    @pytest.mark.parametrize(
        "metric",
        [
            "prefill_duration_ms",
            "prefill_tokens_per_second",
            "first_token_duration_ms",
            "detokenization_duration_ms",
            "model_ttft_duration_ms",
            "model_compute_duration_ms",
            "response_eval_duration_ms",
            "response_eval_tokens_per_second",
            "sequence_fetch_duration_ms",
        ],
    )
    def test_unobservable_model_phases_are_null(self, tmp_path: Path, metric: str) -> None:
        result, _session = _run_fake(tmp_path, [_timing()] * 3)
        report = result.to_dict()

        assert report["aggregate"][metric] is None
        assert all(sample[metric] is None for sample in report["requests"])

    @pytest.mark.parametrize("generated_tokens", range(2))
    def test_no_decode_intervals_leave_decode_metrics_null(
        self, tmp_path: Path, generated_tokens: int
    ) -> None:
        timing = _timing(generated_tokens=generated_tokens, stop_reason="EOS")
        result, _session = _run_fake(tmp_path, [timing] * 3)
        report = result.to_dict()
        sample = report["requests"][-1]

        assert sample["decode_token_durations_ms"] == []
        for metric in ("steady_state_tpot_ms", "steady_state_decode_tokens_per_second"):
            assert sample[metric] is None
            assert report["aggregate"][metric] is None
        if generated_tokens == 0:
            assert sample["task_ttft_duration_ms"] is None
            assert sample["request_ttft_duration_ms"] is None
            assert report["aggregate"]["task_ttft_duration_ms"] is None
            assert report["aggregate"]["request_ttft_duration_ms"] is None
            assert report["aggregate"]["cold_start_ttft_duration_ms"] is None
        assert result.requests[-1].request_duration_ms == pytest.approx(
            (0.001 + timing.reset_s + timing.generation_s) * 1000
        )

    def test_unavailable_observations_do_not_dilute_timed_averages(self, tmp_path: Path) -> None:
        timings = [_timing(generated_tokens=index) for index in range(3)]
        result, _session = _run_fake(tmp_path, timings, warmup=0, iterations=len(timings))

        assert result.aggregate["task_ttft_duration_ms"]["mean"] == pytest.approx(
            fmean(timing.first_token_s for timing in timings[1:]) * 1000
        )
        assert result.aggregate["steady_state_tpot_ms"]["mean"] == pytest.approx(
            fmean(timings[-1].decode_s) * 1000
        )
        assert result.aggregate["cold_start_ttft_duration_ms"] is None

    def test_warmup_only_run_does_not_become_timed_data(self, tmp_path: Path) -> None:
        result, _session = _run_fake(tmp_path, [_timing()], warmup=1, iterations=0)

        assert result.aggregate["timed_request_count"] == 0
        assert result.aggregate["warmup_request_count"] == 1
        assert result.aggregate["task_generation_duration_ms"] is None
        assert result.aggregate["request_duration_ms"] is None
        assert result.aggregate["stop_reason_counts"] == {}

    def test_owned_session_is_closed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        session = _FakeLlamaSession([_timing(generated_tokens=1)] * 3)
        monkeypatch.setattr(perf_gguf, "WinMLLlamaSession", lambda *_args, **_kwargs: session)

        GGUFPerfBenchmark(_config(tmp_path)).run()

        assert session.closed

    def test_memory_profile_tracks_ram_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rss = iter([100.0, 400.0, 420.0])
        monkeypatch.setattr("winml.modelkit.commands._perf_genai._get_rss_mb", lambda: next(rss))
        timings = [_timing(generated_tokens=1) for _ in range(3)]
        result, _session = _run_fake(tmp_path, timings, memory=True)

        assert result.memory_profile is not None
        assert result.memory_profile["rss_model_load_delta_mb"] == 300.0
        assert not any(key.startswith("vram") for key in result.memory_profile)


class TestResultToDict:
    def test_reports_runtime_backend_and_format(self, tmp_path: Path) -> None:
        timings = [_timing() for _ in range(3)]
        result, _session = _run_fake(tmp_path, timings, requested_device="auto")
        info = result.to_dict()["benchmark_info"]

        assert info["runtime"] == "winml-runtime"
        assert info["backend"] == "llama.cpp"
        assert info["api"] == "windowsml.tasks"
        assert info["timing_source"] == "task-stream"
        assert info["model_format"] == "gguf"
        assert info["experimental"] is True
        assert info["ep"] is None
        assert (info["device"], info["effective_device"]) == ("auto", "cpu")
        assert info["context_length"] == 7
        assert info["chat_template_applied"] is True
        assert set(info["packages"]) == {
            "windowsml",
            "windowsml-llama-core",
            "windowsml-llama-cuda",
        }

    def test_retains_report_layout_with_task_timing_schema(self, tmp_path: Path) -> None:
        timings = [_timing() for _ in range(3)]
        result, _session = _run_fake(tmp_path, timings)
        report = result.to_dict()

        assert report["schema_version"] == 3
        assert {"load", "requests", "aggregate", "response_text"} <= set(report)
        assert len(report["requests"]) == 3
        assert report["requests"][0]["kind"] == "warmup"
        json.dumps(report)  # JSON-serializable


# ---------------------------------------------------------------------------
# Reporting and entry point
# ---------------------------------------------------------------------------


class TestReporting:
    def test_console_report_shows_task_metrics_and_device_resolution(self, tmp_path: Path) -> None:
        timings = [_timing() for _ in range(3)]
        result, _session = _run_fake(tmp_path, timings, requested_device="auto")
        buffer = StringIO()
        display_gguf_report(result, Console(file=buffer, width=200, color_system=None))
        output = buffer.getvalue()

        assert "llama.cpp, experimental" in output
        assert "auto -> cpu" in output
        assert "API:" in output
        assert "Windows ML Tasks" in output
        assert "Task TTFT" in output
        assert "Pure prefill: unavailable" in output
        assert "Decode (Task stream):" in output

    def test_console_report_warns_when_nothing_was_generated(self, tmp_path: Path) -> None:
        empty = _timing(generated_tokens=0, stop_reason="EOS")
        result, _session = _run_fake(tmp_path, [empty] * 3)
        buffer = StringIO()
        display_gguf_report(result, Console(file=buffer, width=200, color_system=None))

        output = buffer.getvalue()
        assert "stopped before generating a token" in output
        assert "Cold TTFT: n/a" in output
        assert "n/a tokens/sec" in output
        assert "n/a ms/token" in output

    def test_write_report_round_trips(self, tmp_path: Path) -> None:
        timings = [_timing() for _ in range(3)]
        result, _session = _run_fake(tmp_path, timings)
        path = tmp_path / "nested" / "report.json"
        write_gguf_report(result, path)

        assert json.loads(path.read_text(encoding="utf-8"))["benchmark_info"]["backend"] == (
            "llama.cpp"
        )

    def test_output_path_uses_model_stem(self) -> None:
        path = gguf_output_path(Path("models") / "qwen2.5-0.5b-instruct-q4_k_m.gguf")
        assert path.parent.name == "qwen2.5-0.5b-instruct-q4_k_m"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("model.gguf", True), ("MODEL.GGUF", True), ("model.onnx", False), ("org/model", False)],
    )
    def test_is_gguf_model(self, value: str, expected: bool) -> None:
        assert is_gguf_model(value) is expected


class TestRunGGUFPerf:
    @pytest.mark.parametrize(
        "error",
        (LlamaNotInstalledError, LlamaLoadError, LlamaSessionError, FileNotFoundError),
    )
    def test_session_failures_become_click_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[Exception]
    ) -> None:
        def fail(_self: GGUFPerfBenchmark) -> GGUFBenchmarkResult:
            raise error(error.__name__)

        monkeypatch.setattr(GGUFPerfBenchmark, "run", fail)
        console = Console(file=StringIO())
        with pytest.raises(click.ClickException, match=error.__name__):
            run_gguf_perf(_config(tmp_path), console=console, json_mode=False)

    def test_json_mode_prints_and_saves_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        timings = [_timing() for _ in range(3)]
        result, _session = _run_fake(tmp_path, timings)
        monkeypatch.setattr(GGUFPerfBenchmark, "run", lambda _self: result)
        config = _config(tmp_path)

        run_gguf_perf(config, console=Console(file=StringIO()), json_mode=True)

        assert json.loads(capsys.readouterr().out)["benchmark_info"]["model_format"] == "gguf"
        assert config.output_path is not None
        assert config.output_path.is_file()


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


class TestCliDispatch:
    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    @pytest.fixture
    def capture_run(self, monkeypatch: pytest.MonkeyPatch) -> dict:
        """Patch run_gguf_perf and capture the GGUFPerfConfig it receives."""
        captured: dict = {}

        def fake_run(config: GGUFPerfConfig, *, console: Console, json_mode: bool) -> MagicMock:
            captured["config"] = config
            captured["json_mode"] = json_mode
            return MagicMock()

        monkeypatch.setattr(perf_gguf, "run_gguf_perf", fake_run)
        return captured

    def test_auto_runtime_dispatches_gguf_with_generation_defaults(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        model = _gguf(tmp_path)
        result = runner.invoke(perf, ["-m", str(model)])

        assert result.exit_code == 0, result.output
        config = capture_run["config"]
        assert config.model_path == model
        assert (config.device, config.requested_device) == ("cpu", "auto")
        assert (config.iterations, config.warmup, config.max_new_tokens) == (10, 2, 128)
        assert config.apply_template is True
        assert config.output_path.parent.name == "model"

    def test_explicit_options_are_forwarded(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        model = _gguf(tmp_path)
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Name three primary colors.", encoding="utf-8")
        output = tmp_path / "out.json"
        result = runner.invoke(
            perf,
            [
                "-m",
                str(model),
                "--runtime",
                "winml-runtime",
                "--device",
                "gpu",
                "--prompt-file",
                str(prompt_file),
                "--no-apply-template",
                "--max-new-tokens",
                "16",
                "--iterations",
                "3",
                "--warmup",
                "0",
                "--no-memory",
                "-o",
                str(output),
                "--format",
                "json",
            ],
        )

        assert result.exit_code == 0, result.output
        config = capture_run["config"]
        assert (config.device, config.requested_device) == ("gpu", "gpu")
        assert config.prompt == "Name three primary colors."
        assert config.apply_template is False
        assert (config.iterations, config.warmup, config.max_new_tokens) == (3, 0, 16)
        assert config.memory is False
        assert config.output_path == output
        assert capture_run["json_mode"] is True

    @pytest.mark.parametrize("runtime", ["winml-ort", "ort-genai"])
    def test_rejects_non_runtime_api_runtimes(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, runtime: str
    ) -> None:
        result = runner.invoke(perf, ["-m", str(_gguf(tmp_path)), "--runtime", runtime])

        assert result.exit_code == 2, result.output
        assert f"--runtime {runtime} cannot load" in result.output
        assert "config" not in capture_run

    @pytest.mark.parametrize("device", ["npu", "config"])
    def test_rejects_unsupported_devices(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, device: str
    ) -> None:
        result = runner.invoke(perf, ["-m", str(_gguf(tmp_path)), "--device", device])

        assert result.exit_code == 2, result.output
        assert "not supported for GGUF models" in result.output
        assert "config" not in capture_run

    @pytest.mark.parametrize(
        ("args", "flag"),
        [
            (["--ep", "cpu"], "--ep"),
            (["--ep-options", "key=value"], "--ep-options"),
            (["--backend", "ort"], "--backend"),
            (["--module", "LlamaAttention"], "--module"),
            (["--submodel", "decoder"], "--submodel"),
        ],
    )
    def test_rejects_options_that_select_another_target(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, args: list[str], flag: str
    ) -> None:
        result = runner.invoke(perf, ["-m", str(_gguf(tmp_path)), *args])

        assert result.exit_code == 2, result.output
        assert f"{flag} cannot be used with a GGUF model" in result.output
        assert "config" not in capture_run

    def test_missing_file_is_a_usage_error(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        result = runner.invoke(perf, ["-m", str(tmp_path / "missing.gguf")])

        assert result.exit_code == 2, result.output
        assert "GGUF file not found" in result.output
        assert "config" not in capture_run

    def test_warns_about_ignored_build_options(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        result = runner.invoke(
            perf, ["-m", str(_gguf(tmp_path)), "--precision", "w8a16", "--batch-size", "2"]
        )

        assert result.exit_code == 0, result.output
        assert "ignored for GGUF models" in result.output
        assert "--precision" in result.output
        assert "--batch-size" in result.output

    def test_prompt_and_prompt_file_are_exclusive(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("hello", encoding="utf-8")
        result = runner.invoke(
            perf,
            ["-m", str(_gguf(tmp_path)), "--prompt", "hi", "--prompt-file", str(prompt_file)],
        )

        assert result.exit_code == 2, result.output
        assert "mutually exclusive" in result.output


class TestAutoRuntime:
    @pytest.mark.parametrize("name", ["model.gguf", "Model-00001-of-00003.GGUF"])
    def test_selects_runtime_api_for_gguf(self, name: str) -> None:
        assert _resolve_runtime("auto", name) == "winml-runtime"

    @pytest.mark.parametrize("runtime", ["winml-ort", "ort-genai", "winml-runtime"])
    def test_preserves_explicit_runtime(self, runtime: str) -> None:
        assert _resolve_runtime(runtime, "model.gguf") == runtime
