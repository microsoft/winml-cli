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
    display_genai_report,
    genai_output_path,
    resolve_genai_ep,
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
        effective_ep: str | None = None,
    ) -> None:
        self._timings = list(timings)
        self._i = 0
        self._prompt_ids = prompt_ids if prompt_ids is not None else [1, 2, 3]
        self.context_length = context_length
        self._chat_template = chat_template
        self.encoded_text: str | None = None
        self.template_applied = False
        # Mirrors GenaiSession.effective_ep: the EP that actually took effect (or
        # None to mean "config").  Reported by the benchmark instead of the
        # requested ep so a no-op override is not falsely claimed.
        self.effective_ep = effective_ep

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


def _fake_resolve_loader_config(model_type: str):
    """Return a ``resolve_loader_config`` stand-in yielding *model_type*.

    Keeps the auto-build tests offline (no HF-hub access).
    """
    from types import SimpleNamespace

    def _resolve(model_id: object = None, *, task: object = None, **_kw: object):
        cfg = SimpleNamespace(model_type=model_type)
        hf = SimpleNamespace(model_type=model_type)
        return cfg, hf, object, None

    return _resolve


def _fake_build_genai_bundle(captured: dict):
    """A ``build_genai_bundle`` stand-in that writes a stub bundle and records kwargs."""

    def _build(model_id: str, output_dir: object, recipe: object, **kwargs: object) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "genai_config.json").write_text("{}", encoding="utf-8")
        captured["build"] = {"model_id": model_id, "output_dir": out, **kwargs}
        return out / "genai_config.json"

    return _build


# ---------------------------------------------------------------------------
# resolve_genai_ep
# ---------------------------------------------------------------------------


class TestResolveGenaiEp:
    """``resolve_genai_ep`` reuses the shared resolve_device/resolve_eps path.

    ``resolve_genai_ep`` imports them from ``..sysinfo`` at call time, so the
    tests patch ``winml.modelkit.sysinfo`` and assert the *device -> best
    available EP alias* mapping without probing real hardware.
    """

    def test_config_short_circuits_without_resolving(self, monkeypatch) -> None:
        # "config" means "respect the bundle": no device resolution happens.
        import winml.modelkit.sysinfo as sysinfo

        def _must_not_run(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("resolve_device must not be called for 'config'")

        monkeypatch.setattr(sysinfo, "resolve_device", _must_not_run)
        monkeypatch.setattr(sysinfo, "resolve_eps", _must_not_run)
        assert resolve_genai_ep("config") is None

    @pytest.mark.parametrize(
        ("device", "resolved_device", "eps", "expected"),
        [
            # auto picks the highest-priority available device + its best EP.
            ("auto", "npu", ["QNNExecutionProvider"], "qnn"),
            # npu that is QNN vs VitisAI/OpenVINO -- whatever ORT advertises,
            # not a static "npu -> qnn" guess.
            ("npu", "npu", ["QNNExecutionProvider", "OpenVINOExecutionProvider"], "qnn"),
            ("npu", "npu", ["VitisAIExecutionProvider"], "vitisai"),
            ("gpu", "gpu", ["DmlExecutionProvider"], "dml"),
            # cpu matches ONNX: OpenVINO-on-CPU when available, else plain CPU.
            ("cpu", "cpu", ["OpenVINOExecutionProvider", "CPUExecutionProvider"], "openvino"),
            ("cpu", "cpu", ["CPUExecutionProvider"], "cpu"),
        ],
    )
    def test_resolves_best_available_ep_alias(
        self,
        monkeypatch,
        device: str,
        resolved_device: str,
        eps: list[str],
        expected: str,
    ) -> None:
        import winml.modelkit.sysinfo as sysinfo

        monkeypatch.setattr(
            sysinfo, "resolve_device", lambda **_kwargs: (resolved_device, [resolved_device])
        )
        monkeypatch.setattr(sysinfo, "resolve_eps", lambda _device: list(eps))
        assert resolve_genai_ep(device) == expected

    def test_no_available_ep_returns_none(self, monkeypatch) -> None:
        # A device that resolves to an empty EP list falls back to None
        # (respect config) rather than raising.
        import winml.modelkit.sysinfo as sysinfo

        monkeypatch.setattr(sysinfo, "resolve_device", lambda **_kwargs: ("npu", ["npu"]))
        monkeypatch.setattr(sysinfo, "resolve_eps", lambda _device: [])
        assert resolve_genai_ep("npu") is None

    def test_unavailable_device_propagates_valueerror(self, monkeypatch) -> None:
        # resolve_device raises for an unavailable device; genai fails fast
        # (matches the ONNX path) instead of silently respecting config.
        import winml.modelkit.sysinfo as sysinfo

        def _raise(**_kwargs: object) -> object:
            raise ValueError("Device 'npu' requested but no compatible EP is available.")

        monkeypatch.setattr(sysinfo, "resolve_device", _raise)
        with pytest.raises(ValueError, match="no compatible EP"):
            resolve_genai_ep("npu")


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
        session = _FakeSession(
            [_timing(0.4, 0.6, [0.4, 0.4, 0.4])], prompt_ids=[1, 2, 3], effective_ep="qnn"
        )
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

    def test_to_dict_reports_effective_ep_over_requested_ep(self) -> None:
        # A hardware override that took no effect (session.effective_ep is None,
        # e.g. flat/all-CPU bundle) is reported as "config", never the requested
        # ep — so the JSON never claims an EP that never applied.
        cfg = GenaiPerfConfig(
            bundle_dir=Path("bundle"), ep="qnn", device="npu", iterations=1, warmup=0
        )
        session = _FakeSession([_timing(0.4, 0.6, [0.4, 0.4])], effective_ep=None)
        info = GenaiPerfBenchmark(cfg, session=session).run().to_dict()["benchmark_info"]
        assert info["ep"] == "config"
        assert info["device"] == "npu"


# ---------------------------------------------------------------------------
# _session_device (concrete device for a forced EP)
# ---------------------------------------------------------------------------


class TestSessionDevice:
    """``_session_device`` resolves the device a forced EP should target."""

    @staticmethod
    def _bench(ep: str | None, device: str) -> GenaiPerfBenchmark:
        cfg = GenaiPerfConfig(bundle_dir=Path("bundle"), ep=ep, device=device)
        return GenaiPerfBenchmark(cfg, session=_FakeSession([]))

    def test_none_without_override(self) -> None:
        assert self._bench(None, "config")._session_device() is None

    def test_concrete_device_used_verbatim(self) -> None:
        assert self._bench("openvino", "npu")._session_device() == "npu"

    def test_config_sentinel_falls_back_to_ep_primary_device(self) -> None:
        # --ep given alone (device defaults to "config"): use the EP's primary
        # supported device so device-parameterized EPs still get a device_type.
        assert self._bench("openvino", "config")._session_device() == "npu"

    def test_auto_falls_back_to_ep_primary_device(self) -> None:
        assert self._bench("qnn", "auto")._session_device() == "npu"

    def test_build_session_forwards_device(self, monkeypatch) -> None:
        captured: dict = {}

        def fake_ctor(bundle_dir, ep, *, device=None, **_kwargs):
            captured["ep"] = ep
            captured["device"] = device
            return _FakeSession([])

        monkeypatch.setattr(perf_genai, "GenaiSession", fake_ctor)
        cfg = GenaiPerfConfig(bundle_dir=Path("bundle"), ep="openvino", device="npu")
        GenaiPerfBenchmark(cfg)._build_session()
        assert captured == {"ep": "openvino", "device": "npu"}


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
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, monkeypatch
    ) -> None:
        # A concrete --device resolves to the best available EP for that device
        # via the shared resolve_device/resolve_eps path (here: npu -> qnn).
        import winml.modelkit.sysinfo as sysinfo

        monkeypatch.setattr(sysinfo, "resolve_device", lambda **_k: ("npu", ["npu"]))
        monkeypatch.setattr(sysinfo, "resolve_eps", lambda _d: ["QNNExecutionProvider"])
        bundle = _make_bundle(tmp_path)
        result = runner.invoke(
            perf, ["-m", str(bundle), "--runtime", "winml-genai", "--device", "npu"]
        )
        assert result.exit_code == 0, result.output
        cfg = capture_run["config"]
        assert cfg.ep == "qnn"
        assert cfg.device == "npu"
        assert cfg.bundle_dir == bundle

    def test_device_auto_resolves_best_ep(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, monkeypatch
    ) -> None:
        # Explicit --device auto matches the ONNX path: pick the best device +
        # its best available EP and force the whole pipeline onto it.
        import winml.modelkit.sysinfo as sysinfo

        monkeypatch.setattr(sysinfo, "resolve_device", lambda **_k: ("gpu", ["gpu"]))
        monkeypatch.setattr(sysinfo, "resolve_eps", lambda _d: ["DmlExecutionProvider"])
        bundle = _make_bundle(tmp_path)
        result = runner.invoke(
            perf, ["-m", str(bundle), "--runtime", "winml-genai", "--device", "auto"]
        )
        assert result.exit_code == 0, result.output
        cfg = capture_run["config"]
        assert cfg.ep == "dml"
        assert cfg.device == "auto"

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
        # --ep alone forces that EP even though --device is omitted (its
        # effective default for genai is "config" = respect the bundle).
        bundle = _make_bundle(tmp_path)
        result = runner.invoke(perf, ["-m", str(bundle), "--runtime", "winml-genai", "--ep", "dml"])
        assert result.exit_code == 0, result.output
        cfg = capture_run["config"]
        assert cfg.ep == "dml"
        assert cfg.device == "config"

    def test_default_device_is_config(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        # Omitting --device is genai's "respect the bundle" default: no EP
        # override, device recorded as "config".
        bundle = _make_bundle(tmp_path)
        result = runner.invoke(perf, ["-m", str(bundle), "--runtime", "winml-genai"])
        assert result.exit_code == 0, result.output
        cfg = capture_run["config"]
        assert cfg.device == "config"
        assert cfg.ep is None

    def test_explicit_device_config_respects_bundle(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        # Passing --device config explicitly is the same as omitting it: no EP
        # override (and it must not trigger device resolution).
        bundle = _make_bundle(tmp_path)
        result = runner.invoke(
            perf, ["-m", str(bundle), "--runtime", "winml-genai", "--device", "config"]
        )
        assert result.exit_code == 0, result.output
        cfg = capture_run["config"]
        assert cfg.device == "config"
        assert cfg.ep is None

    def test_onnx_runtime_rejects_device_config(self, runner: CliRunner, tmp_path: Path) -> None:
        # "config" is a winml-genai-only sentinel; the default (winml) runtime
        # rejects it with a clear message rather than a generic device error.
        result = runner.invoke(perf, ["-m", str(tmp_path / "model.onnx"), "--device", "config"])
        assert result.exit_code != 0
        assert "winml-genai" in result.output

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
        # Omitting --device means "config": no EP override, so the bundle's own
        # per-stage routing in genai_config.json is respected (ctx/iter on QNN,
        # embeddings/lm_head on CPU).
        assert cfg.device == "config"
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

    def test_submodel_rejected(self, runner: CliRunner, tmp_path: Path, capture_run: dict) -> None:
        # --submodel narrows a composite to one standalone sub-session; a genai
        # bundle is already the full composite generation pipeline, so it must be
        # rejected rather than silently ignored (the winml-genai return runs before
        # the winml-path --submodel handling).
        bundle = _make_bundle(tmp_path)
        result = runner.invoke(
            perf,
            ["-m", str(bundle), "--runtime", "winml-genai", "--submodel", "decoder"],
        )
        assert result.exit_code != 0
        assert "--submodel" in result.output
        assert "config" not in capture_run

    def test_onnx_file_rejected(self, runner: CliRunner, tmp_path: Path, capture_run: dict) -> None:
        onnx = tmp_path / "model.onnx"
        onnx.write_bytes(b"fake")
        result = runner.invoke(perf, ["-m", str(onnx), "--runtime", "winml-genai"])
        assert result.exit_code != 0
        assert "directory" in result.output.lower()
        assert "config" not in capture_run

    def test_unresolvable_model_id_rejected(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, monkeypatch
    ) -> None:
        # A -m that is neither a bundle dir nor an .onnx is treated as a model
        # id; an unresolvable one surfaces a clean UsageError, not a traceback.
        import winml.modelkit.loader as loader_mod

        monkeypatch.setenv("WINML_CACHE_DIR", str(tmp_path))

        def _boom(*_a: object, **_k: object):
            raise ValueError("nope")

        monkeypatch.setattr(loader_mod, "resolve_loader_config", _boom)
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

    def test_hf_model_id_autobuilds_and_dispatches(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, monkeypatch
    ) -> None:
        # -m is a model id (not a bundle dir): a genai bundle is built on the
        # fly into the cache and then benchmarked -- one command, no prior build.
        import winml.modelkit.loader as loader_mod
        import winml.modelkit.models.winml as winml_models

        monkeypatch.setenv("WINML_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(
            loader_mod, "resolve_loader_config", _fake_resolve_loader_config("qwen3")
        )
        monkeypatch.setattr(winml_models, "resolve_genai_bundle", lambda _mt: object())
        build_calls: dict = {}
        monkeypatch.setattr(
            winml_models, "build_genai_bundle", _fake_build_genai_bundle(build_calls)
        )

        result = runner.invoke(perf, ["-m", "Qwen/Qwen3-0.6B", "--runtime", "winml-genai"])

        assert result.exit_code == 0, result.output
        # Built once, pinned to the NPU HTP via QNN regardless of --device.
        assert build_calls["build"]["ep"] == "qnn"
        assert build_calls["build"]["device"] == "npu"
        assert build_calls["build"]["force_rebuild"] is False
        # Benchmarked the freshly built cache bundle.
        cfg = capture_run["config"]
        assert cfg.bundle_dir == build_calls["build"]["output_dir"]
        assert (cfg.bundle_dir / "genai_config.json").exists()
        # Omitting --device keeps the "respect the bundle" default.
        assert cfg.device == "config"
        assert cfg.ep is None

    def test_autobuild_reuses_cached_bundle(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, monkeypatch
    ) -> None:
        import winml.modelkit.models.winml as winml_models
        from winml.modelkit.cache import get_model_dir

        monkeypatch.setenv("WINML_CACHE_DIR", str(tmp_path))
        cached = get_model_dir("Qwen/Qwen3-0.6B", cache_dir=tmp_path) / "genai-bundle"
        cached.mkdir(parents=True)
        (cached / "genai_config.json").write_text("{}", encoding="utf-8")

        build_calls: dict = {}
        monkeypatch.setattr(
            winml_models, "build_genai_bundle", _fake_build_genai_bundle(build_calls)
        )

        result = runner.invoke(perf, ["-m", "Qwen/Qwen3-0.6B", "--runtime", "winml-genai"])

        assert result.exit_code == 0, result.output
        assert "build" not in build_calls  # cache hit: never rebuilt
        assert capture_run["config"].bundle_dir == cached

    def test_rebuild_forces_autobuild(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, monkeypatch
    ) -> None:
        import winml.modelkit.loader as loader_mod
        import winml.modelkit.models.winml as winml_models
        from winml.modelkit.cache import get_model_dir

        monkeypatch.setenv("WINML_CACHE_DIR", str(tmp_path))
        cached = get_model_dir("Qwen/Qwen3-0.6B", cache_dir=tmp_path) / "genai-bundle"
        cached.mkdir(parents=True)
        (cached / "genai_config.json").write_text("{}", encoding="utf-8")

        monkeypatch.setattr(
            loader_mod, "resolve_loader_config", _fake_resolve_loader_config("qwen3")
        )
        monkeypatch.setattr(winml_models, "resolve_genai_bundle", lambda _mt: object())
        build_calls: dict = {}
        monkeypatch.setattr(
            winml_models, "build_genai_bundle", _fake_build_genai_bundle(build_calls)
        )

        result = runner.invoke(
            perf, ["-m", "Qwen/Qwen3-0.6B", "--runtime", "winml-genai", "--rebuild"]
        )

        assert result.exit_code == 0, result.output
        assert build_calls["build"]["force_rebuild"] is True

    def test_ignore_cache_builds_in_tempdir(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, monkeypatch
    ) -> None:
        # --ignore-cache mirrors the winml runtime: build fresh in a throwaway
        # temp dir and never touch the managed cache. Both the assembled bundle
        # and its component build cache land outside WINML_CACHE_DIR, and the
        # managed bundle dir is never written.
        import winml.modelkit.loader as loader_mod
        import winml.modelkit.models.winml as winml_models
        from winml.modelkit.cache import get_model_dir

        monkeypatch.setenv("WINML_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(
            loader_mod, "resolve_loader_config", _fake_resolve_loader_config("qwen3")
        )
        monkeypatch.setattr(winml_models, "resolve_genai_bundle", lambda _mt: object())
        build_calls: dict = {}
        monkeypatch.setattr(
            winml_models, "build_genai_bundle", _fake_build_genai_bundle(build_calls)
        )

        result = runner.invoke(
            perf, ["-m", "Qwen/Qwen3-0.6B", "--runtime", "winml-genai", "--ignore-cache"]
        )

        assert result.exit_code == 0, result.output
        build = build_calls["build"]
        # Forced fresh build, isolated from the managed cache on both axes.
        assert build["force_rebuild"] is True
        assert not build["output_dir"].is_relative_to(tmp_path)
        assert not Path(build["cache_dir"]).is_relative_to(tmp_path)
        # The managed bundle dir is never populated.
        managed = get_model_dir("Qwen/Qwen3-0.6B", cache_dir=tmp_path) / "genai-bundle"
        assert not (managed / "genai_config.json").exists()
        # The benchmark ran against the temp bundle that was built.
        assert capture_run["config"].bundle_dir == build["output_dir"]

    def test_autobuild_honored_flags_not_warned_as_ignored(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, monkeypatch
    ) -> None:
        # The build-driving flags (--rebuild/--task) steer the auto-build, so
        # they must NOT appear in the "options are ignored" warning when a bundle
        # is built from a model id. A genuinely ignored flag (--memory) still is.
        import winml.modelkit.loader as loader_mod
        import winml.modelkit.models.winml as winml_models

        monkeypatch.setenv("WINML_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(
            loader_mod, "resolve_loader_config", _fake_resolve_loader_config("qwen3")
        )
        monkeypatch.setattr(winml_models, "resolve_genai_bundle", lambda _mt: object())
        monkeypatch.setattr(winml_models, "build_genai_bundle", _fake_build_genai_bundle({}))

        result = runner.invoke(
            perf,
            [
                "-m",
                "Qwen/Qwen3-0.6B",
                "--runtime",
                "winml-genai",
                "--rebuild",
                "--task",
                "text-generation",
                "--memory",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "--rebuild" not in result.output
        assert "--task" not in result.output
        assert "--memory" in result.output  # still ignored -> still warned

    def test_prebuilt_bundle_still_warns_build_flags(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        # A prebuilt bundle dir ignores the build-driving flags, so --rebuild is
        # reported as ignored (no auto-build happened).
        bundle = _make_bundle(tmp_path)
        result = runner.invoke(perf, ["-m", str(bundle), "--runtime", "winml-genai", "--rebuild"])
        assert result.exit_code == 0, result.output
        assert "--rebuild" in result.output

    def test_cache_hit_warns_dropped_build_input_flags(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, monkeypatch
    ) -> None:
        # On a cache hit the model-id-keyed bundle is reused as-is, so the
        # artifact-shaping flags (--precision/--task) were NOT applied to it and
        # must be reported as ignored -- unlike a fresh build, which honors them.
        # (--rebuild/--ignore-cache always force a build, so they never reach here.)
        import winml.modelkit.models.winml as winml_models
        from winml.modelkit.cache import get_model_dir

        monkeypatch.setenv("WINML_CACHE_DIR", str(tmp_path))
        cached = get_model_dir("Qwen/Qwen3-0.6B", cache_dir=tmp_path) / "genai-bundle"
        cached.mkdir(parents=True)
        (cached / "genai_config.json").write_text("{}", encoding="utf-8")

        build_calls: dict = {}
        monkeypatch.setattr(
            winml_models, "build_genai_bundle", _fake_build_genai_bundle(build_calls)
        )

        result = runner.invoke(
            perf,
            [
                "-m",
                "Qwen/Qwen3-0.6B",
                "--runtime",
                "winml-genai",
                "--precision",
                "w8a16",
                "--task",
                "text-generation",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "build" not in build_calls  # cache hit: reused, never built
        assert capture_run["config"].bundle_dir == cached
        # The reused bundle did not honor the build-input flags -> warned.
        assert "--precision" in result.output
        assert "--task" in result.output

    def test_autobuild_without_recipe_rejected(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, monkeypatch
    ) -> None:
        # A model id with no registered genai-bundle recipe cannot be auto-built:
        # surface a clear UsageError pointing at a prebuilt bundle directory.
        import winml.modelkit.loader as loader_mod
        import winml.modelkit.models.winml as winml_models

        monkeypatch.setenv("WINML_CACHE_DIR", str(tmp_path))
        monkeypatch.setattr(
            loader_mod, "resolve_loader_config", _fake_resolve_loader_config("bert")
        )
        monkeypatch.setattr(winml_models, "resolve_genai_bundle", lambda _mt: None)

        result = runner.invoke(
            perf, ["-m", "google-bert/bert-base-uncased", "--runtime", "winml-genai"]
        )

        assert result.exit_code != 0
        assert "recipe" in result.output.lower()
        assert "config" not in capture_run

    def test_winml_runtime_unaffected(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        # Default runtime must not route through the genai path.
        result = runner.invoke(perf, ["--help"])
        assert result.exit_code == 0
        assert "config" not in capture_run
