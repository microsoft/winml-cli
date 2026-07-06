# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the winml-genai perf runtime -- mock-based, no model, no genai.

A fake GenaiSession whose ``generate_timed`` returns canned
:class:`GenerationTiming` objects makes the aggregation deterministic, so these
tests never touch onnxruntime-genai or a real bundle.  (The og-boundary timing
itself is unit-tested in ``tests/unit/session/test_genai_session.py``.)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from rich.console import Console

from winml.modelkit.commands import _perf_genai as perf_genai
from winml.modelkit.commands._perf_genai import (
    GenaiBenchmarkResult,
    GenaiPerfBenchmark,
    GenaiPerfConfig,
    device_to_genai_ep,
    display_genai_report,
    genai_output_path,
    run_genai_perf,
    write_genai_report,
)
from winml.modelkit.commands.perf import perf
from winml.modelkit.session import (
    GenaiNotInstalledError,
    GenaiSessionError,
    GenerationTiming,
)


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


def _timing(
    prefill_s: float,
    first_token_s: float,
    decode_s: list[float],
    *,
    input_tokens: int = 3,
) -> GenerationTiming:
    """Build a GenerationTiming with ``1 + len(decode_s)`` generated tokens."""
    return GenerationTiming(
        input_tokens=input_tokens,
        generated_tokens=1 + len(decode_s),
        prefill_s=prefill_s,
        first_token_s=first_token_s,
        decode_s=list(decode_s),
    )


class _FakeSession:
    """GenaiSession stand-in returning canned GenerationTiming objects.

    Each ``generate_timed`` call returns the next entry from *timings*.  An
    empty *timings* list makes ``generate_timed`` raise ``GenaiSessionError``
    (mirroring the real empty-output guard).
    """

    def __init__(
        self,
        timings: list[GenerationTiming],
        *,
        prompt_ids: list[int] | None = None,
        context_length: int = 256,
        chat_template: bool = True,
    ) -> None:
        self._timings = list(timings)
        self._i = 0
        self._prompt_ids = prompt_ids if prompt_ids is not None else [1, 2, 3]
        self.context_length = context_length
        self._chat_template = chat_template
        self.encoded_text: str | None = None
        self.template_applied = False

    def encode(self, text: str) -> list[int]:
        self.encoded_text = text
        return list(self._prompt_ids)

    def apply_chat_template(self, prompt: str, **_kwargs: object) -> str:
        self.template_applied = True
        if not self._chat_template:
            raise GenaiSessionError("bundle ships no chat template")
        return f"<chat>{prompt}</chat>"

    def generate_timed(self, prompt: object, config: object = None) -> GenerationTiming:
        if self._i >= len(self._timings):
            raise GenaiSessionError("genai: generation produced no tokens (empty bundle output?)")
        timing = self._timings[self._i]
        self._i += 1
        return timing


def _make_bundle(tmp_path: Path, name: str = "bundle") -> Path:
    """Create a minimal genai bundle directory."""
    bundle = tmp_path / name
    bundle.mkdir()
    (bundle / "genai_config.json").write_text("{}", encoding="utf-8")
    return bundle


# ---------------------------------------------------------------------------
# device_to_genai_ep
# ---------------------------------------------------------------------------


class TestDeviceToGenaiEp:
    @pytest.mark.parametrize(
        ("device", "expected"),
        [
            ("cpu", "cpu"),
            ("npu", "qnn"),
            ("gpu", "dml"),
            # "auto" (and any unrecognized device) resolves to None = respect
            # the bundle's genai_config.json routing rather than forcing an EP.
            ("auto", None),
            ("NPU", "qnn"),
            ("unknown-device", None),
        ],
    )
    def test_mapping(self, device: str, expected: str | None) -> None:
        assert device_to_genai_ep(device) == expected


# ---------------------------------------------------------------------------
# Metric math
# ---------------------------------------------------------------------------


class TestMetricMath:
    def test_single_run_metrics(self) -> None:
        cfg = GenaiPerfConfig(bundle_dir=Path("x"), warmup=0, iterations=1, max_new_tokens=4)
        # prefill 0.4s + first token 0.6s -> TTFT 1.0s; 3 decode steps of 0.4s.
        timing = _timing(0.4, 0.6, [0.4, 0.4, 0.4], input_tokens=5)
        session = _FakeSession([timing], prompt_ids=[1, 2, 3, 4, 5], context_length=256)
        bench = GenaiPerfBenchmark(cfg, session=session)

        result = bench.run()

        assert result.prompt_tokens == 5
        assert result.generated_tokens == 4
        assert result.context_length == 256
        assert result.ttft_mean_ms == pytest.approx(1000.0)
        assert result.prefill_mean_ms == pytest.approx(400.0)
        # total = 0.4 + 0.6 + 3*0.4 = 2.2s
        assert result.total_generation_mean_ms == pytest.approx(2200.0)
        # decode: 3 steps over 1.2s -> 2.5 tok/s
        assert result.decode_tokens_per_sec == pytest.approx(2.5)
        # TPOT: mean of [0.4, 0.4, 0.4] = 0.4s
        assert result.tpot_mean_ms == pytest.approx(400.0)
        # avg per-token latency: 2200 ms / 4 tokens = 550 ms
        assert result.avg_token_latency_ms == pytest.approx(550.0)
        assert result.raw_ttft_ms == pytest.approx([1000.0])
        assert result.raw_prefill_ms == pytest.approx([400.0])
        assert result.raw_tpot_ms == pytest.approx([400.0])

    def test_warmup_runs_excluded(self) -> None:
        cfg = GenaiPerfConfig(bundle_dir=Path("x"), warmup=1, iterations=2, max_new_tokens=4)
        # warmup TTFT 5000 ms must be excluded; timed runs: TTFT 1000 / 2000 ms.
        warmup = _timing(2.5, 2.5, [1.0])
        run1 = _timing(0.4, 0.6, [0.5])  # ttft 1.0s, total 1.5s
        run2 = _timing(0.8, 1.2, [1.0])  # ttft 2.0s, total 3.0s
        session = _FakeSession([warmup, run1, run2])
        bench = GenaiPerfBenchmark(cfg, session=session)

        result = bench.run()

        assert len(result.raw_ttft_ms) == 2
        assert result.raw_ttft_ms == pytest.approx([1000.0, 2000.0])
        assert result.ttft_mean_ms == pytest.approx(1500.0)
        # totals 1500 / 3000 ms -> mean 2250 ms
        assert result.total_generation_mean_ms == pytest.approx(2250.0)

    def test_single_token_generation_has_zero_decode_rate(self) -> None:
        cfg = GenaiPerfConfig(bundle_dir=Path("x"), warmup=0, iterations=1, max_new_tokens=1)
        # Single token: no steady-state decode phase.
        session = _FakeSession([_timing(1.0, 1.0, [])])
        bench = GenaiPerfBenchmark(cfg, session=session)

        result = bench.run()

        assert result.generated_tokens == 1
        assert result.decode_tokens_per_sec == 0.0
        assert result.tpot_mean_ms == 0.0
        assert result.ttft_mean_ms == pytest.approx(2000.0)

    def test_no_tokens_raises(self) -> None:
        cfg = GenaiPerfConfig(bundle_dir=Path("x"), warmup=0, iterations=1)
        # Empty timings -> generate_timed raises GenaiSessionError (empty output).
        session = _FakeSession([])
        bench = GenaiPerfBenchmark(cfg, session=session)

        with pytest.raises(GenaiSessionError, match="no tokens"):
            bench.run()

    def test_percentiles_over_multiple_runs(self) -> None:
        cfg = GenaiPerfConfig(bundle_dir=Path("x"), warmup=0, iterations=4, max_new_tokens=2)
        # 4 runs with TTFTs of 1000, 2000, 3000, 4000 ms.
        timings = [
            _timing(0.5, 0.5, [0.1]),
            _timing(1.0, 1.0, [0.1]),
            _timing(1.5, 1.5, [0.1]),
            _timing(2.0, 2.0, [0.1]),
        ]
        session = _FakeSession(timings)
        bench = GenaiPerfBenchmark(cfg, session=session)

        result = bench.run()

        assert result.raw_ttft_ms == pytest.approx([1000.0, 2000.0, 3000.0, 4000.0])
        assert result.ttft_min_ms == pytest.approx(1000.0)
        assert result.ttft_max_ms == pytest.approx(4000.0)


# ---------------------------------------------------------------------------
# GenaiBenchmarkResult.to_dict
# ---------------------------------------------------------------------------


class TestChatTemplate:
    def test_applies_bundle_chat_template(self) -> None:
        """The prompt is wrapped in the bundle's chat template before encoding."""
        cfg = GenaiPerfConfig(
            bundle_dir=Path("x"), warmup=0, iterations=1, max_new_tokens=2, prompt="Hi"
        )
        session = _FakeSession([_timing(0.4, 0.6, [0.4])], chat_template=True)
        bench = GenaiPerfBenchmark(cfg, session=session)

        bench.run()

        assert session.encoded_text == "<chat>Hi</chat>"

    def test_falls_back_to_raw_prompt_without_template(self) -> None:
        """Bundles without a chat template benchmark the raw prompt unchanged."""
        cfg = GenaiPerfConfig(
            bundle_dir=Path("x"), warmup=0, iterations=1, max_new_tokens=2, prompt="Hi"
        )
        session = _FakeSession([_timing(0.4, 0.6, [0.4])], chat_template=False)
        bench = GenaiPerfBenchmark(cfg, session=session)

        bench.run()

        assert session.encoded_text == "Hi"

    def test_no_apply_template_benchmarks_prompt_verbatim(self) -> None:
        """--no-apply-template benchmarks the prompt as-is, even if a template exists."""
        cfg = GenaiPerfConfig(
            bundle_dir=Path("x"),
            warmup=0,
            iterations=1,
            max_new_tokens=2,
            prompt="Hi",
            apply_template=False,
        )
        session = _FakeSession([_timing(0.4, 0.6, [0.4])], chat_template=True)
        bench = GenaiPerfBenchmark(cfg, session=session)

        bench.run()

        # The bundle ships a chat template, but apply_template=False bypasses it
        # so a caller can benchmark a prompt they have already templated.
        assert session.encoded_text == "Hi"
        assert session.template_applied is False


class TestResultToDict:
    def _result(self) -> GenaiBenchmarkResult:
        cfg = GenaiPerfConfig(
            bundle_dir=Path("bundle"),
            ep="qnn",
            device="npu",
            prompt="Benchmark this exact prompt",
            max_new_tokens=4,
            iterations=1,
            warmup=0,
            compile=True,
            compile_timeout=120,
        )
        session = _FakeSession([_timing(0.4, 0.6, [0.4, 0.4, 0.4])], prompt_ids=[1, 2, 3])
        bench = GenaiPerfBenchmark(cfg, session=session)
        return bench.run()

    def test_to_dict_shape(self) -> None:
        d = self._result().to_dict()

        assert set(d) == {
            "benchmark_info",
            "ttft_ms",
            "prefill_ms",
            "decode",
            "total_generation_ms",
            "raw",
        }
        info = d["benchmark_info"]
        assert info["runtime"] == "winml-genai"
        assert info["ep"] == "qnn"
        assert info["device"] == "npu"
        assert info["max_new_tokens"] == 4
        assert info["prompt_tokens"] == 3
        assert info["generated_tokens"] == 4
        assert info["compile"] is True
        assert info["compile_timeout"] == 120
        assert info["apply_template"] is True
        assert info["prompt"] == "Benchmark this exact prompt"
        assert set(d["ttft_ms"]) == {"mean", "min", "max", "p50", "p90", "p95", "p99"}
        assert set(d["prefill_ms"]) == {"mean"}
        assert set(d["decode"]) == {"tokens_per_sec", "avg_token_latency_ms", "tpot_ms"}
        assert set(d["raw"]) == {
            "ttft_ms",
            "prefill_ms",
            "decode_tokens_per_sec",
            "tpot_ms",
            "total_ms",
        }

    def test_to_dict_is_json_serializable(self) -> None:
        # Round-trips without error.
        json.dumps(self._result().to_dict())

    def test_to_dict_ep_none_reports_config(self) -> None:
        # ep=None (respect the bundle config) is reported as the label "config"
        # so the JSON never carries a null EP.
        cfg = GenaiPerfConfig(
            bundle_dir=Path("bundle"), ep=None, device="auto", iterations=1, warmup=0
        )
        session = _FakeSession([_timing(0.4, 0.6, [0.4, 0.4])])
        info = GenaiPerfBenchmark(cfg, session=session).run().to_dict()["benchmark_info"]
        assert info["ep"] == "config"
        assert info["device"] == "auto"


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


class TestReporting:
    def _result(self) -> GenaiBenchmarkResult:
        cfg = GenaiPerfConfig(
            bundle_dir=Path("bundle"),
            ep="cpu",
            device="cpu",
            iterations=1,
            warmup=0,
        )
        session = _FakeSession([_timing(0.4, 0.6, [0.4, 0.4, 0.4])])
        bench = GenaiPerfBenchmark(cfg, session=session)
        return bench.run()

    def test_write_genai_report_writes_json(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "report.json"
        write_genai_report(self._result(), out)

        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["benchmark_info"]["runtime"] == "winml-genai"

    def test_display_genai_report_does_not_crash(self) -> None:
        display_genai_report(self._result(), Console())

    def test_display_genai_report_ep_none_does_not_crash(self) -> None:
        # ep=None renders as "<device> (config)" without error.
        cfg = GenaiPerfConfig(
            bundle_dir=Path("bundle"), ep=None, device="auto", iterations=1, warmup=0
        )
        session = _FakeSession([_timing(0.4, 0.6, [0.4, 0.4, 0.4])])
        result = GenaiPerfBenchmark(cfg, session=session).run()
        display_genai_report(result, Console())

    def test_genai_output_path_uses_bundle_name(self) -> None:
        path = genai_output_path(Path("/some/dir/my-bundle"))
        assert path.parent.name == "my-bundle"
        assert path.suffix == ".json"


# ---------------------------------------------------------------------------
# run_genai_perf entry point
# ---------------------------------------------------------------------------


class TestRunGenaiPerf:
    def test_writes_report_and_returns_result(self, tmp_path: Path, monkeypatch) -> None:
        cfg = GenaiPerfConfig(
            bundle_dir=Path("bundle"),
            iterations=1,
            warmup=0,
            max_new_tokens=4,
            output_path=tmp_path / "out.json",
        )
        session = _FakeSession([_timing(0.4, 0.6, [0.4, 0.4, 0.4])])
        # Inject the fake session by patching the benchmark's session builder.
        monkeypatch.setattr(perf_genai.GenaiPerfBenchmark, "_build_session", lambda self: session)

        result = run_genai_perf(cfg, console=Console(), json_mode=False)

        assert isinstance(result, GenaiBenchmarkResult)
        assert (tmp_path / "out.json").exists()

    def test_json_mode_emits_json(self, tmp_path: Path, monkeypatch, capsys) -> None:
        cfg = GenaiPerfConfig(
            bundle_dir=Path("bundle"),
            iterations=1,
            warmup=0,
            max_new_tokens=4,
            output_path=tmp_path / "out.json",
        )
        session = _FakeSession([_timing(0.4, 0.6, [0.4, 0.4, 0.4])])
        monkeypatch.setattr(perf_genai.GenaiPerfBenchmark, "_build_session", lambda self: session)

        run_genai_perf(cfg, console=Console(stderr=True), json_mode=True)

        out = capsys.readouterr().out
        assert json.loads(out)["benchmark_info"]["runtime"] == "winml-genai"

    def test_not_installed_becomes_click_error(self, monkeypatch) -> None:
        import click

        class _Boom:
            def __init__(self, _cfg: object) -> None:
                pass

            def run(self) -> None:
                raise GenaiNotInstalledError("onnxruntime-genai missing")

        monkeypatch.setattr(perf_genai, "GenaiPerfBenchmark", _Boom)
        with pytest.raises(click.ClickException):
            run_genai_perf(
                GenaiPerfConfig(bundle_dir=Path("x")),
                console=Console(),
                json_mode=False,
            )


# ---------------------------------------------------------------------------
# CLI dispatch (winml perf --runtime winml-genai)
# ---------------------------------------------------------------------------


class TestCliDispatch:
    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    @pytest.fixture
    def capture_run(self, monkeypatch) -> dict:
        """Patch run_genai_perf and capture the GenaiPerfConfig it receives."""
        captured: dict = {}

        def fake_run(config, *, console, json_mode):
            captured["config"] = config
            captured["json_mode"] = json_mode
            return MagicMock()

        monkeypatch.setattr(perf_genai, "run_genai_perf", fake_run)
        return captured

    def test_dispatches_and_maps_device_to_ep(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        bundle = _make_bundle(tmp_path)
        result = runner.invoke(
            perf, ["-m", str(bundle), "--runtime", "winml-genai", "--device", "npu"]
        )
        assert result.exit_code == 0, result.output
        cfg = capture_run["config"]
        assert cfg.ep == "qnn"
        assert cfg.device == "npu"
        assert cfg.bundle_dir == bundle

    def test_explicit_ep_overrides_device(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        # Explicit --ep wins over the --device mapping (precedence:
        # --ep > concrete --device > respect config).
        bundle = _make_bundle(tmp_path)
        result = runner.invoke(
            perf,
            ["-m", str(bundle), "--runtime", "winml-genai", "--device", "npu", "--ep", "cpu"],
        )
        assert result.exit_code == 0, result.output
        cfg = capture_run["config"]
        assert cfg.ep == "cpu"
        assert cfg.device == "npu"

    def test_explicit_ep_without_device(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        # --ep alone forces that EP even though --device stays at its "auto"
        # default (which on its own would mean "respect config").
        bundle = _make_bundle(tmp_path)
        result = runner.invoke(perf, ["-m", str(bundle), "--runtime", "winml-genai", "--ep", "dml"])
        assert result.exit_code == 0, result.output
        cfg = capture_run["config"]
        assert cfg.ep == "dml"
        assert cfg.device == "auto"

    def test_ep_flag_not_warned_as_ignored(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        # --ep is honored for winml-genai, so it must not appear in the
        # "options are ignored" warning.
        bundle = _make_bundle(tmp_path)
        result = runner.invoke(perf, ["-m", str(bundle), "--runtime", "winml-genai", "--ep", "qnn"])
        assert result.exit_code == 0, result.output
        assert "--ep" not in result.output

    def test_genai_iteration_defaults(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        bundle = _make_bundle(tmp_path)
        runner.invoke(perf, ["-m", str(bundle), "--runtime", "winml-genai"])
        cfg = capture_run["config"]
        assert cfg.iterations == 10
        assert cfg.warmup == 2
        # Default device ("auto") resolves to None: no EP override, so the
        # bundle's own per-stage routing in genai_config.json is respected
        # (ctx/iter on QNN, embeddings/lm_head on CPU).
        assert cfg.device == "auto"
        assert cfg.ep is None

    def test_explicit_iterations_honored(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        bundle = _make_bundle(tmp_path)
        runner.invoke(
            perf,
            [
                "-m",
                str(bundle),
                "--runtime",
                "winml-genai",
                "--iterations",
                "50",
                "--warmup",
                "5",
            ],
        )
        cfg = capture_run["config"]
        assert cfg.iterations == 50
        assert cfg.warmup == 5

    def test_prompt_and_max_new_tokens_forwarded(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        bundle = _make_bundle(tmp_path)
        runner.invoke(
            perf,
            [
                "-m",
                str(bundle),
                "--runtime",
                "winml-genai",
                "--prompt",
                "hello there",
                "--max-new-tokens",
                "64",
            ],
        )
        cfg = capture_run["config"]
        assert cfg.prompt == "hello there"
        assert cfg.max_new_tokens == 64

    def test_default_prompt_used_when_omitted(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        bundle = _make_bundle(tmp_path)
        runner.invoke(perf, ["-m", str(bundle), "--runtime", "winml-genai"])
        # The CLI --prompt default must match the GenaiPerfConfig field default.
        assert capture_run["config"].prompt == perf_genai._DEFAULT_PROMPT
        assert capture_run["config"].prompt == GenaiPerfConfig(bundle_dir=Path("x")).prompt

    def test_apply_template_defaults_true(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        bundle = _make_bundle(tmp_path)
        runner.invoke(perf, ["-m", str(bundle), "--runtime", "winml-genai"])
        assert capture_run["config"].apply_template is True

    def test_no_apply_template_forwarded(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        bundle = _make_bundle(tmp_path)
        runner.invoke(
            perf,
            ["-m", str(bundle), "--runtime", "winml-genai", "--no-apply-template"],
        )
        assert capture_run["config"].apply_template is False

    def test_compile_flag_forwarded(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        bundle = _make_bundle(tmp_path)
        runner.invoke(
            perf,
            [
                "-m",
                str(bundle),
                "--runtime",
                "winml-genai",
                "--compile",
                "--compile-timeout",
                "120",
            ],
        )
        cfg = capture_run["config"]
        assert cfg.compile is True
        assert cfg.compile_timeout == 120

    def test_warns_and_ignores_winml_only_flags(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        bundle = _make_bundle(tmp_path)
        result = runner.invoke(
            perf,
            ["-m", str(bundle), "--runtime", "winml-genai", "--batch-size", "4"],
        )
        assert result.exit_code == 0, result.output
        assert "ignored" in result.output.lower()
        assert "--batch-size" in result.output
        # Still dispatched despite the ignored flag.
        assert "config" in capture_run

    def test_module_rejected(self, runner: CliRunner, tmp_path: Path, capture_run: dict) -> None:
        bundle = _make_bundle(tmp_path)
        result = runner.invoke(
            perf,
            ["-m", str(bundle), "--runtime", "winml-genai", "--module", "Foo"],
        )
        assert result.exit_code != 0
        assert "--module" in result.output
        assert "config" not in capture_run

    def test_onnx_file_rejected(self, runner: CliRunner, tmp_path: Path, capture_run: dict) -> None:
        onnx = tmp_path / "model.onnx"
        onnx.write_bytes(b"fake")
        result = runner.invoke(perf, ["-m", str(onnx), "--runtime", "winml-genai"])
        assert result.exit_code != 0
        assert "directory" in result.output.lower()
        assert "config" not in capture_run

    def test_missing_directory_rejected(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        result = runner.invoke(perf, ["-m", str(tmp_path / "nope"), "--runtime", "winml-genai"])
        assert result.exit_code != 0
        assert "config" not in capture_run

    def test_missing_genai_config_rejected(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        result = runner.invoke(perf, ["-m", str(empty), "--runtime", "winml-genai"])
        assert result.exit_code != 0
        assert "genai_config.json" in result.output
        assert "config" not in capture_run

    def test_winml_runtime_unaffected(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        # Default runtime must not route through the genai path.
        result = runner.invoke(perf, ["--help"])
        assert result.exit_code == 0
        assert "config" not in capture_run
