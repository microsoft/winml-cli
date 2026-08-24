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
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from rich.console import Console

from winml.modelkit.commands import _perf_genai as perf_genai
from winml.modelkit.commands import perf as perf_module
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
from winml.modelkit.commands.perf import _resolve_runtime, perf
from winml.modelkit.session import (
    GenaiNotInstalledError,
    GenaiSessionError,
    GenerationTiming,
)
from winml.modelkit.session.monitor.memory_tracker import MEMORY_PROFILE_FIELDS


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


def _timing(
    prefill_s: float,
    first_token_s: float,
    decode_s: list[float],
    *,
    input_tokens: int = 3,
    generator_create_s: float = 0.0,
    sequence_fetch_s: float = 0.0,
    detokenization_s: float = 0.0,
) -> GenerationTiming:
    """Build a GenerationTiming with ``1 + len(decode_s)`` generated tokens."""
    return GenerationTiming(
        input_tokens=input_tokens,
        generated_tokens=1 + len(decode_s),
        generator_create_s=generator_create_s,
        prefill_s=prefill_s,
        first_token_s=first_token_s,
        decode_s=list(decode_s),
        sequence_fetch_s=sequence_fetch_s,
        detokenization_s=detokenization_s,
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
        effective_device: str | None = None,
        effective_hardware_ep: str | None = None,
        load_timings_ms: dict[str, float] | None = None,
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
        self.effective_device = effective_device
        self.effective_hardware_ep = effective_hardware_ep
        self.load_timings_ms = load_timings_ms or {}
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def resolve_effective_route(self) -> None:
        pass

    def encode(self, text: str) -> list[int]:
        self.encoded_text = text
        return list(self._prompt_ids)

    def apply_chat_template(self, prompt: str, **_kwargs: object) -> str:
        self.template_applied = True
        if not self._chat_template:
            raise GenaiSessionError("bundle ships no chat template")
        return f"<chat>{prompt}</chat>"

    def generate_timed(
        self, prompt: object, config: object = None, **_kwargs: object
    ) -> GenerationTiming:
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
    """``resolve_genai_ep`` reuses the shared session resolve_device / available_eps_for_device.

    ``resolve_genai_ep`` imports them from ``..session`` at call time, so the
    tests patch ``winml.modelkit.session`` and assert the *device -> best
    available EP alias* mapping without probing real hardware.
    """

    def test_config_short_circuits_without_resolving(self, monkeypatch) -> None:
        # "config" means "respect the bundle": no device resolution happens.
        import winml.modelkit.session as session

        def _must_not_run(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("resolve_device must not be called for 'config'")

        monkeypatch.setattr(session, "resolve_device", _must_not_run)
        monkeypatch.setattr(session, "available_eps_for_device", _must_not_run)
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
            # gpu: prefer native GPU EPs over cross-device accelerators.
            ("gpu", "gpu", ["OpenVINOExecutionProvider", "DmlExecutionProvider"], "dml"),
            # cpu: prefer CPUExecutionProvider over cross-device accelerators.
            ("cpu", "cpu", ["OpenVINOExecutionProvider", "CPUExecutionProvider"], "cpu"),
            ("cpu", "cpu", ["CPUExecutionProvider"], "cpu"),
            # Edge case: only a cross-device EP available for this device.
            ("cpu", "cpu", ["OpenVINOExecutionProvider"], "openvino"),
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
        import winml.modelkit.session as session
        from winml.modelkit.session import EPDeviceTarget

        monkeypatch.setattr(
            session,
            "resolve_device",
            lambda _target: EPDeviceTarget(ep="auto", device=resolved_device),
        )
        monkeypatch.setattr(session, "available_eps_for_device", lambda _device: list(eps))
        assert resolve_genai_ep(device) == expected

    def test_no_available_ep_returns_none(self, monkeypatch) -> None:
        # A device that resolves to an empty EP list falls back to None
        # (respect config) rather than raising.
        import winml.modelkit.session as session
        from winml.modelkit.session import EPDeviceTarget

        monkeypatch.setattr(
            session, "resolve_device", lambda _target: EPDeviceTarget(ep="auto", device="npu")
        )
        monkeypatch.setattr(session, "available_eps_for_device", lambda _device: [])
        assert resolve_genai_ep("npu") is None

    def test_unavailable_device_propagates_valueerror(self, monkeypatch) -> None:
        # resolve_device raises for an unavailable device; genai fails fast
        # (matches the ONNX path) instead of silently respecting config.
        import winml.modelkit.session as session

        def _raise(_target: object) -> object:
            raise ValueError("Device 'npu' requested but no compatible EP is available.")

        monkeypatch.setattr(session, "resolve_device", _raise)
        with pytest.raises(ValueError, match="no compatible EP"):
            resolve_genai_ep("npu")


# ---------------------------------------------------------------------------
# Metric math
# ---------------------------------------------------------------------------


class TestMetricMath:
    def test_load_and_request_submetrics_are_canonical(self) -> None:
        cfg = GenaiPerfConfig(bundle_dir=Path("x"), warmup=0, iterations=1, max_new_tokens=2)
        # model_compute = 0.4 + 0.6 + 0.5 = 1.5s; response_eval = 1.1s.
        timing = _timing(
            0.4,
            0.6,
            [0.5],
            input_tokens=4,
            generator_create_s=0.05,
            sequence_fetch_s=0.02,
            detokenization_s=0.03,
        )
        session = _FakeSession(
            [timing],
            prompt_ids=[1, 2, 3, 4],
            load_timings_ms={
                "session_load_duration_ms": 1240.0,
                "ep_registration_duration_ms": 10.0,
                "bundle_prepare_duration_ms": 20.0,
                "native_load_duration_ms": 800.0,
                "config_create_duration_ms": 25.0,
                "model_create_duration_ms": 750.0,
                "tokenizer_create_duration_ms": 25.0,
            },
        )
        # session.load outer span = 1250 ms; template = 100 ms; tokenization = 150 ms.
        clock = iter([10.0, 11.25, 11.25, 11.35, 11.50])
        bench = GenaiPerfBenchmark(cfg, session=session, clock=lambda: next(clock))

        result = bench.run()

        assert session.loaded is True
        assert result.load["session_load_duration_ms"] == pytest.approx(1250.0)
        assert result.load["native_load_duration_ms"] == pytest.approx(800.0)
        assert result.load["model_create_duration_ms"] == pytest.approx(750.0)
        assert result.load["weight_upload_duration_ms"] is None
        assert result.load["weight_upload_estimate_duration_ms"] == pytest.approx(750.0)
        assert result.load["weight_upload_estimate_source"] == "model_create_duration"

        request = result.requests[0]
        assert request.kind == "timed"
        assert request.template_duration_ms == pytest.approx(100.0)
        assert request.tokenization_duration_ms == pytest.approx(150.0)
        assert request.generator_create_duration_ms == pytest.approx(50.0)
        assert request.model_ttft_duration_ms == pytest.approx(1000.0)
        assert request.request_ttft_duration_ms == pytest.approx(1300.0)
        assert request.model_compute_duration_ms == pytest.approx(1500.0)
        assert request.response_eval_duration_ms == pytest.approx(1100.0)
        assert request.request_duration_ms == pytest.approx(1850.0)
        assert request.prefill_tokens_per_second == pytest.approx(10.0)
        assert request.steady_state_decode_tokens_per_second == pytest.approx(2.0)
        assert request.response_eval_tokens_per_second == pytest.approx(1.8181818)

        assert result.aggregate["cold_start_ttft_duration_ms"] == pytest.approx(2550.0)
        assert result.aggregate["cold_start_total_duration_ms"] == pytest.approx(3100.0)
        assert result.aggregate["request_duration_ms"]["mean"] == pytest.approx(1850.0)
        assert result.aggregate["model_compute_duration_ms"]["mean"] == pytest.approx(1500.0)

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
        assert result.aggregate["model_ttft_duration_ms"]["mean"] == pytest.approx(1000.0)
        assert result.aggregate["prefill_duration_ms"]["mean"] == pytest.approx(400.0)
        # total = 0.4 + 0.6 + 3*0.4 = 2.2s
        assert result.aggregate["model_compute_duration_ms"]["mean"] == pytest.approx(2200.0)
        # decode: 3 steps over 1.2s -> 2.5 tok/s
        assert result.aggregate["steady_state_decode_tokens_per_second"]["mean"] == pytest.approx(
            2.5
        )
        # TPOT: mean of [0.4, 0.4, 0.4] = 0.4s
        assert result.aggregate["steady_state_tpot_ms"]["mean"] == pytest.approx(400.0)
        assert result.aggregate["response_eval_tokens_per_second"]["mean"] == pytest.approx(4 / 1.8)
        assert result.requests[0].model_ttft_duration_ms == pytest.approx(1000.0)
        assert result.requests[0].prefill_duration_ms == pytest.approx(400.0)
        assert result.requests[0].steady_state_tpot_ms == pytest.approx(400.0)
        assert result.requests[0].prefill_tokens_per_second == pytest.approx(12.5)

    def test_warmup_runs_excluded(self) -> None:
        cfg = GenaiPerfConfig(bundle_dir=Path("x"), warmup=1, iterations=2, max_new_tokens=4)
        # warmup TTFT 5000 ms must be excluded; timed runs: TTFT 1000 / 2000 ms.
        warmup = _timing(2.5, 2.5, [1.0])
        run1 = _timing(0.4, 0.6, [0.5])  # ttft 1.0s, total 1.5s
        run2 = _timing(0.8, 1.2, [1.0])  # ttft 2.0s, total 3.0s
        session = _FakeSession([warmup, run1, run2])
        bench = GenaiPerfBenchmark(cfg, session=session)

        result = bench.run()

        assert [s.kind for s in result.requests] == ["warmup", "timed", "timed"]
        timed_ttft = [s.model_ttft_duration_ms for s in result.requests if s.kind == "timed"]
        assert timed_ttft == pytest.approx([1000.0, 2000.0])
        assert result.aggregate["model_ttft_duration_ms"]["mean"] == pytest.approx(1500.0)
        # totals 1500 / 3000 ms -> mean 2250 ms
        assert result.aggregate["model_compute_duration_ms"]["mean"] == pytest.approx(2250.0)

    def test_single_token_generation_has_zero_decode_rate(self) -> None:
        cfg = GenaiPerfConfig(bundle_dir=Path("x"), warmup=0, iterations=1, max_new_tokens=1)
        # Single token: no steady-state decode phase.
        session = _FakeSession([_timing(1.0, 1.0, [])])
        bench = GenaiPerfBenchmark(cfg, session=session)

        result = bench.run()

        assert result.generated_tokens == 1
        assert result.aggregate["steady_state_decode_tokens_per_second"]["mean"] == 0.0
        assert result.aggregate["steady_state_tpot_ms"]["mean"] == 0.0
        assert result.aggregate["model_ttft_duration_ms"]["mean"] == pytest.approx(2000.0)

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

        assert [s.model_ttft_duration_ms for s in result.requests] == pytest.approx(
            [1000.0, 2000.0, 3000.0, 4000.0]
        )
        assert result.aggregate["model_ttft_duration_ms"]["min"] == pytest.approx(1000.0)
        assert result.aggregate["model_ttft_duration_ms"]["max"] == pytest.approx(4000.0)

    def test_memory_profile_uses_classic_perf_field_names(self, monkeypatch) -> None:
        cfg = GenaiPerfConfig(
            bundle_dir=Path("x"),
            warmup=0,
            iterations=1,
            max_new_tokens=2,
            memory=True,
        )
        state = {"rss": 100.0}

        class MemorySession(_FakeSession):
            def load(self) -> None:
                super().load()
                state["rss"] = 150.0

            def generate_timed(
                self, prompt: object, config: object = None, **kwargs: object
            ) -> GenerationTiming:
                state["rss"] = 180.0
                return super().generate_timed(prompt, config, **kwargs)

        session = MemorySession([_timing(0.4, 0.6, [0.5])])
        monkeypatch.setattr(
            "winml.modelkit.session.monitor.memory_tracker.get_rss_mb",
            lambda: state["rss"],
        )
        monkeypatch.setattr(
            "winml.modelkit.session.monitor.memory_tracker.get_vram_mb",
            lambda _adapter_luid: (None, None),
        )
        monkeypatch.setattr(perf_genai, "_resolve_adapter_luid", lambda *_args: None, raising=False)

        result = GenaiPerfBenchmark(cfg, session=session).run()

        assert result.memory_profile is not None
        assert result.memory_profile["rss_process_baseline_mb"] == 100.0
        assert result.memory_profile["rss_before_session_mb"] == 100.0
        assert result.memory_profile["rss_after_session_mb"] == 150.0
        assert result.memory_profile["rss_peak_during_inference_mb"] == 180.0
        assert result.memory_profile["rss_peak_delta_mb"] == 80.0
        assert set(result.memory_profile) == MEMORY_PROFILE_FIELDS

    def test_memory_profile_includes_vram_only_for_proven_adapter(self, monkeypatch) -> None:
        cfg = GenaiPerfConfig(
            bundle_dir=Path("x"),
            warmup=0,
            iterations=1,
            max_new_tokens=2,
            memory=True,
        )
        state: dict[str, object] = {
            "rss": 100.0,
            "vram": (10.0, 20.0),
        }

        class MemorySession(_FakeSession):
            def load(self) -> None:
                super().load()
                state["rss"] = 150.0
                state["vram"] = (30.0, 50.0)

            def generate_timed(
                self, prompt: object, config: object = None, **kwargs: object
            ) -> GenerationTiming:
                state["rss"] = 180.0
                state["vram"] = (40.0, 70.0)
                return super().generate_timed(prompt, config, **kwargs)

        session = MemorySession(
            [_timing(0.4, 0.6, [0.5])],
            effective_device="gpu",
            effective_hardware_ep="DmlExecutionProvider",
        )
        monkeypatch.setattr(
            "winml.modelkit.session.monitor.memory_tracker.get_rss_mb",
            lambda: float(state["rss"]),
        )
        monkeypatch.setattr(
            "winml.modelkit.session.monitor.memory_tracker.get_vram_mb",
            lambda adapter_luid: state["vram"] if adapter_luid else (None, None),
        )
        monkeypatch.setattr(
            perf_genai,
            "_resolve_adapter_luid",
            lambda device, ep: "luid" if (device, ep) == ("gpu", "DmlExecutionProvider") else None,
            raising=False,
        )

        result = GenaiPerfBenchmark(cfg, session=session).run()

        assert result.memory_profile is not None
        assert result.memory_profile["vram_local_process_baseline_mb"] is None
        assert result.memory_profile["vram_shared_process_baseline_mb"] is None
        assert result.memory_profile["vram_local_before_session_mb"] == 10.0
        assert result.memory_profile["vram_shared_before_session_mb"] == 20.0
        assert result.memory_profile["vram_local_after_session_mb"] == 30.0
        assert result.memory_profile["vram_shared_after_session_mb"] == 50.0
        assert result.memory_profile["vram_local_peak_during_inference_mb"] == 40.0
        assert result.memory_profile["vram_shared_peak_during_inference_mb"] == 70.0
        assert result.memory_profile["vram_local_peak_delta_mb"] == 30.0
        assert result.memory_profile["vram_shared_peak_delta_mb"] == 50.0
        assert set(result.memory_profile) == MEMORY_PROFILE_FIELDS


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
            model_id="Qwen/Qwen3-0.6B",
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
            [_timing(0.4, 0.6, [0.4, 0.4, 0.4])],
            prompt_ids=[1, 2, 3],
            effective_ep="qnn",
            effective_device="npu",
        )
        bench = GenaiPerfBenchmark(cfg, session=session)
        return bench.run()

    def test_to_dict_shape(self) -> None:
        d = self._result().to_dict()

        assert set(d) == {
            "schema_version",
            "benchmark_info",
            "load",
            "requests",
            "aggregate",
        }
        assert d["schema_version"] == 2
        info = d["benchmark_info"]
        assert info["runtime"] == "winml-genai"
        assert info["model_id"] == "Qwen/Qwen3-0.6B"
        assert info["running_model_path"] == "bundle"
        assert info["bundle_dir"] == "bundle"
        assert info["ep"] == "qnn"
        assert info["device"] == "npu"
        assert info["effective_device"] == "npu"
        assert info["max_new_tokens"] == 4
        assert info["prompt_tokens"] == 3
        assert info["generated_tokens"] == 4
        assert info["compile"] is True
        assert info["compile_timeout"] == 120
        assert info["monitor"] is False
        assert info["apply_template"] is True
        assert info["prompt"] == "Benchmark this exact prompt"
        assert set(d["load"]) == {
            "session_load_duration_ms",
            "ep_registration_duration_ms",
            "bundle_prepare_duration_ms",
            "native_load_duration_ms",
            "config_create_duration_ms",
            "model_create_duration_ms",
            "tokenizer_create_duration_ms",
            "weight_upload_duration_ms",
            "weight_upload_estimate_duration_ms",
            "weight_upload_estimate_source",
        }
        assert len(d["requests"]) == 1
        assert set(d["requests"][0]) == {
            "kind",
            "index",
            "prompt_tokens",
            "generated_tokens",
            "template_duration_ms",
            "tokenization_duration_ms",
            "generator_create_duration_ms",
            "prefill_duration_ms",
            "first_token_duration_ms",
            "decode_token_durations_ms",
            "sequence_fetch_duration_ms",
            "detokenization_duration_ms",
            "request_ttft_duration_ms",
            "model_ttft_duration_ms",
            "response_eval_duration_ms",
            "model_compute_duration_ms",
            "request_duration_ms",
            "prefill_tokens_per_second",
            "steady_state_decode_tokens_per_second",
            "response_eval_tokens_per_second",
            "steady_state_tpot_ms",
        }
        stats_keys = {
            "mean",
            "std",
            "min",
            "max",
            "p50",
            "p90",
            "p95",
            "p99",
        }
        assert d["aggregate"]["warmup_excluded"] is True
        assert d["aggregate"]["timed_request_count"] == 1
        assert set(d["aggregate"]["request_duration_ms"]) == stats_keys
        assert set(d["aggregate"]["model_compute_duration_ms"]) == stats_keys
        assert set(d["aggregate"]["model_ttft_duration_ms"]) == stats_keys
        assert set(d["aggregate"]["request_ttft_duration_ms"]) == stats_keys
        assert set(d["aggregate"]["prefill_duration_ms"]) == stats_keys
        assert set(d["aggregate"]["response_eval_duration_ms"]) == stats_keys
        assert set(d["aggregate"]["steady_state_tpot_ms"]) == stats_keys
        assert set(d["aggregate"]["prefill_tokens_per_second"]) == stats_keys
        assert set(d["aggregate"]["steady_state_decode_tokens_per_second"]) == stats_keys
        assert set(d["aggregate"]["response_eval_tokens_per_second"]) == stats_keys
        assert "accuracy" not in d

    def test_to_dict_includes_optional_memory_and_hw_monitor(self) -> None:
        cfg = GenaiPerfConfig(bundle_dir=Path("bundle"), iterations=1, warmup=0)
        result = GenaiBenchmarkResult(
            config=cfg,
            memory_profile={"rss_total_delta_mb": 12.5},
            hw_monitor={"monitor": "HWMonitor", "cpu": {"mean_pct": 1.0}},
        )

        d = result.to_dict()

        assert d["memory"] == {"rss_total_delta_mb": 12.5}
        assert d["hw_monitor"] == {"monitor": "HWMonitor", "cpu": {"mean_pct": 1.0}}
        assert "hardware" not in d

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

    def test_to_dict_includes_generation_monitor_metrics(self, monkeypatch) -> None:
        metrics = {
            "device_kind": "npu",
            "npu": {"mean_pct": 42.0, "sample_count": 4},
            "cpu": {"process_mean_pct": 250.0, "sample_count": 4},
            "ram": {"mean_mb": 2048.0},
        }

        class FakeMonitor:
            @classmethod
            def is_available(cls) -> bool:
                return True

            def __init__(self, **_kwargs: object) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args: object) -> None:
                pass

            def to_dict(self) -> dict:
                return metrics

        cfg = GenaiPerfConfig(
            bundle_dir=Path("bundle"),
            ep="qnn",
            device="npu",
            iterations=1,
            warmup=0,
            monitor=True,
        )
        session = _FakeSession([_timing(0.4, 0.6, [0.4])], effective_ep="qnn")
        bench = GenaiPerfBenchmark(cfg, session=session)
        monkeypatch.setattr(bench, "_build_hw_monitor", lambda: FakeMonitor())

        data = bench.run().to_dict()

        assert data["benchmark_info"]["monitor"] is True
        assert data["hw_monitor"] == metrics


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

    def test_build_hw_monitor_uses_cpu_when_effective_adapter_is_unproven(
        self, monkeypatch
    ) -> None:
        captured: dict = {}

        class FakeMonitor:
            @classmethod
            def is_available(cls) -> bool:
                return True

            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(
            "winml.modelkit.session.monitor.hw_monitor.HWMonitor",
            FakeMonitor,
        )
        cfg = GenaiPerfConfig(bundle_dir=Path("bundle"), device="config", ep=None)

        monitor = GenaiPerfBenchmark(cfg, session=_FakeSession([]))._build_hw_monitor()

        assert isinstance(monitor, FakeMonitor)
        assert captured == {"poll_interval_ms": 200, "device": "cpu", "ep_name": None}

    def test_build_hw_monitor_uses_effective_route_not_requested_values(self, monkeypatch) -> None:
        captured: dict = {}

        class FakeMonitor:
            @classmethod
            def is_available(cls) -> bool:
                return True

            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        monkeypatch.setattr(
            "winml.modelkit.session.monitor.hw_monitor.HWMonitor",
            FakeMonitor,
        )
        cfg = GenaiPerfConfig(bundle_dir=Path("bundle"), device="npu", ep="qnn")
        session = _FakeSession(
            [],
            effective_device="gpu",
            effective_hardware_ep="DmlExecutionProvider",
        )

        monitor = GenaiPerfBenchmark(cfg, session=session)._build_hw_monitor()

        assert isinstance(monitor, FakeMonitor)
        assert captured == {
            "poll_interval_ms": 200,
            "device": "gpu",
            "ep_name": "DmlExecutionProvider",
        }


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

    def test_display_genai_report_prefers_adapter_block_over_gpu_aggregate(self) -> None:
        result = self._result()
        result.hw_monitor = {
            "device_kind": "gpu",
            "adapter": {"mean_pct": 91.2, "peak_pct": 98.8, "sample_count": 5},
            "gpu": {"mean_pct": 1.1, "peak_pct": 2.2, "sample_count": 11},
            "cpu": {"mean_pct": 12.3, "peak_pct": 34.5, "sample_count": 5},
            "ram": {"used_mb": 1024.0, "peak_mb": 2048.0},
        }
        console = Console(file=StringIO(), width=200, force_terminal=False, record=True)

        display_genai_report(result, console)

        out = console.export_text()
        assert "GPU: 91.2% avg, 98.8% peak" in out
        assert "GPU: 1.1% avg, 2.2% peak" not in out

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
        # via the shared session resolve_device / available_eps_for_device path.
        import winml.modelkit.session as session
        from winml.modelkit.session import EPDeviceTarget

        monkeypatch.setattr(
            session, "resolve_device", lambda _t: EPDeviceTarget(ep="auto", device="npu")
        )
        monkeypatch.setattr(
            session, "available_eps_for_device", lambda _d: ["QNNExecutionProvider"]
        )
        bundle = _make_bundle(tmp_path)
        result = runner.invoke(
            perf, ["-m", str(bundle), "--runtime", "winml-genai", "--device", "npu"]
        )
        assert result.exit_code == 0, result.output
        cfg = capture_run["config"]
        assert cfg.model_id == str(bundle)
        assert cfg.ep == "qnn"
        assert cfg.device == "npu"
        assert cfg.bundle_dir == bundle

    def test_device_auto_resolves_best_ep(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, monkeypatch
    ) -> None:
        # Explicit --device auto matches the ONNX path: pick the best device +
        # its best available EP and force the whole pipeline onto it.
        import winml.modelkit.session as session
        from winml.modelkit.session import EPDeviceTarget

        monkeypatch.setattr(
            session, "resolve_device", lambda _t: EPDeviceTarget(ep="auto", device="gpu")
        )
        monkeypatch.setattr(
            session, "available_eps_for_device", lambda _d: ["DmlExecutionProvider"]
        )
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

    def test_auto_runtime_default_dispatches_genai_bundle(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        # Omitting --runtime detects the bundle marker and routes to genai.
        # Omitting --device then respects the bundle's own routing.
        bundle = _make_bundle(tmp_path)
        result = runner.invoke(perf, ["-m", str(bundle)])
        assert result.exit_code == 0, result.output
        cfg = capture_run["config"]
        assert cfg.device == "config"
        assert cfg.ep is None

    def test_auto_runtime_dispatches_normalized_bundle_path(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, monkeypatch
    ) -> None:
        bundle = _make_bundle(tmp_path)
        monkeypatch.setattr(
            perf_module.cli_utils,
            "normalize_model_arg",
            lambda _model: str(bundle),
        )

        result = runner.invoke(perf, ["-m", "~/bundle"])

        assert result.exit_code == 0, result.output
        assert capture_run["config"].bundle_dir == bundle

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
        # "config" is a winml-genai-only sentinel; auto resolves an ONNX path to
        # winml and rejects it with a clear message rather than a generic error.
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

    def test_duration_op_tracing_conflict_not_fatal_for_genai(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        # The --duration + --op-tracing conflict is a WinML-path constraint. For
        # winml-genai both flags are ignored (non-fatal), so passing both must
        # warn and continue rather than aborting with the conflict UsageError.
        bundle = _make_bundle(tmp_path)
        result = runner.invoke(
            perf,
            [
                "-m",
                str(bundle),
                "--runtime",
                "winml-genai",
                "--duration",
                "5",
                "--op-tracing",
                "basic",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "not valid with --op-tracing" not in result.output
        # Both stay non-fatal ignored options for genai.
        assert "--duration" in result.output
        assert "--op-tracing" in result.output

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

    def test_prompt_file_is_read_as_utf8(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        bundle = _make_bundle(tmp_path)
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("hello from a long prompt", encoding="utf-8")

        result = runner.invoke(
            perf,
            [
                "-m",
                str(bundle),
                "--runtime",
                "winml-genai",
                "--prompt-file",
                str(prompt_file),
            ],
        )

        assert result.exit_code == 0, result.output
        assert capture_run["config"].prompt == "hello from a long prompt"

    def test_prompt_and_prompt_file_are_mutually_exclusive(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        bundle = _make_bundle(tmp_path)
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("file prompt", encoding="utf-8")

        result = runner.invoke(
            perf,
            [
                "-m",
                str(bundle),
                "--runtime",
                "winml-genai",
                "--prompt",
                "argv prompt",
                "--prompt-file",
                str(prompt_file),
            ],
        )

        assert result.exit_code != 0
        assert "mutually exclusive" in result.output
        assert "config" not in capture_run

    def test_invalid_utf8_prompt_file_is_a_click_error(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        bundle = _make_bundle(tmp_path)
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_bytes(b"\xff\xfe")

        result = runner.invoke(
            perf,
            [
                "-m",
                str(bundle),
                "--runtime",
                "winml-genai",
                "--prompt-file",
                str(prompt_file),
            ],
        )

        assert result.exit_code != 0
        assert "Could not read prompt file" in result.output
        assert "Traceback" not in result.output
        assert "config" not in capture_run

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

    def test_memory_and_monitor_flags_forwarded(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict
    ) -> None:
        bundle = _make_bundle(tmp_path)
        result = runner.invoke(
            perf,
            ["-m", str(bundle), "--runtime", "winml-genai", "--memory", "--monitor"],
        )
        assert result.exit_code == 0, result.output
        assert "--memory" not in result.output
        assert "--monitor" not in result.output
        cfg = capture_run["config"]
        assert cfg.memory is True
        assert cfg.monitor is True

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
        assert cfg.model_id == "Qwen/Qwen3-0.6B"
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

    def test_autobuild_honored_flags_not_warned_as_ignored(
        self, runner: CliRunner, tmp_path: Path, capture_run: dict, monkeypatch
    ) -> None:
        # The build-driving flags (--rebuild/--task) steer the auto-build, so
        # they must NOT appear in the "options are ignored" warning when a bundle
        # is built from a model id. Runtime genai flags such as --memory are
        # honored by the benchmark and must not be reported as ignored either.
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
        assert "--memory" not in result.output

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
        # (--rebuild always forces a build, so it never reaches here.)
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

    def test_runtime_help_shows_auto_default(self, runner: CliRunner, capture_run: dict) -> None:
        result = runner.invoke(perf, ["--help"])
        assert result.exit_code == 0
        assert "[auto|winml|winml-genai]" in result.output
        assert "default: auto" in result.output
        assert "config" not in capture_run


class TestAutoRuntime:
    def test_selects_genai_for_bundle_folder(self, tmp_path: Path) -> None:
        bundle = _make_bundle(tmp_path)
        assert _resolve_runtime("auto", str(bundle)) == "winml-genai"

    @pytest.mark.parametrize("model_kind", ["plain-folder", "onnx-file", "model-id"])
    def test_selects_winml_without_bundle_marker(self, tmp_path: Path, model_kind: str) -> None:
        if model_kind == "plain-folder":
            model = tmp_path / "model"
            model.mkdir()
            value = str(model)
        elif model_kind == "onnx-file":
            model = tmp_path / "model.onnx"
            model.write_bytes(b"onnx")
            value = str(model)
        else:
            value = "organization/model"

        assert _resolve_runtime("auto", value) == "winml"

    @pytest.mark.parametrize("runtime", ["winml", "winml-genai"])
    def test_preserves_explicit_runtime(self, runtime: str) -> None:
        assert _resolve_runtime(runtime, "organization/model") == runtime
