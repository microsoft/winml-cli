# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for CGC text generation benchmarking in ``winml perf`` without native dependencies.

A fake session with the :class:`~winml.modelkit.session.WinMLCGCSession` surface
returns clock-generated timings, so the shared schema 3 engine runs for real
while no Windows ML or DXCore code runs. The session is tested in
``tests/unit/session/test_cgc_session.py``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from io import StringIO
from itertools import count
from typing import TYPE_CHECKING, Any

import click
import pytest
from rich.console import Console

import winml.modelkit
from tests.unit.test_helpers import make_timing
from winml.modelkit.commands import _generation_metrics
from winml.modelkit.commands import _perf_cgc as perf_cgc
from winml.modelkit.commands._perf_cgc import (
    CGCBenchmarkResult,
    CGCPerfBenchmark,
    CGCPerfConfig,
    run_cgc_perf,
)
from winml.modelkit.commands._perf_gguf import (
    GGUFBenchmarkResult,
    GGUFPerfBenchmark,
    GGUFPerfConfig,
)
from winml.modelkit.commands._task_generation_perf import TaskBenchmarkResult, display_task_report
from winml.modelkit.session import CGCLoadError, CGCNotInstalledError, CGCSessionError


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from pathlib import Path

    from winml.modelkit.session import TaskGenerationTiming


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------

_ADAPTER = {"name": "Fake GPU", "luid": "0x00000000_0x0000BEEF", "selection": "high-performance"}
# Per-call spans of a loaded WinMLCGCSession.
_LOAD_TIMINGS_MS = {
    "runtime_create_duration_ms": 1.0,
    "target_create_duration_ms": 2.0,
    "model_create_duration_ms": 3.0,
    "set_model_duration_ms": 4.0,
    "session_create_duration_ms": 5.0,
}
_PROMPT_IDS = [1, 2, 3, 4]
_TOKENIZER_EOS = (151645,)
_VRAM_MB = (256.0, 32.0)
# Console text only a CGC report shows: its header rows, load split and GPU memory.
_CGC_ONLY_TEXT = (
    "Adapter:",
    "Format:",
    "Capacity:",
    "Program cache:",
    "set_model:",
    "First session (GPU compile)",
    "GPU local/shared",
)
# benchmark_info keys of a llama.cpp report, captured by running
# list(GGUFBenchmarkResult(GGUFPerfConfig(model_path=...)).to_dict()["benchmark_info"])
# on origin/main 15228997, before the engine was shared.
_LLAMA_CPP_INFO_KEYS = [
    "runtime",
    "backend",
    "api",
    "timing_source",
    "model_format",
    "experimental",
    "model_id",
    "running_model_path",
    "ep",
    "device",
    "effective_device",
    "iterations",
    "warmup",
    "max_new_tokens",
    "ignore_eos",
    "apply_template",
    "chat_template_applied",
    "prompt",
    "prompt_tokens",
    "generated_tokens",
    "stop_reason",
    "context_length",
    "packages",
    "llama_backends",
    "timestamp",
]


class _FakeCGCSession:
    """``WinMLCGCSession`` stand-in: the Task generation session protocol plus CGC read-backs.

    Each call is recorded with the thread it ran on.

    Args:
        capacity: Logical capacity Windows ML reports after :meth:`load`;
            ``None`` when it reports none.
        vocabulary: Tokenizer vocabulary size; ``None`` when not reported.
        chat_template_applied: Whether the model's chat template renders.
        interrupt: ``(call, n)``: the ``n``-th call named ``call`` raises
            ``KeyboardInterrupt``, as a Ctrl+C would.
    """

    def __init__(
        self,
        *,
        capacity: int | None = 135,
        vocabulary: int | None = 151936,
        chat_template_applied: bool = True,
        interrupt: tuple[str, int] | None = None,
    ) -> None:
        self.adapter_name = _ADAPTER["name"]
        self.adapter_luid = _ADAPTER["luid"]
        self.adapter_selection = _ADAPTER["selection"]
        self.vocabulary_size = vocabulary
        self.chat_template_applied: bool | None = None
        self.load_timings_ms: dict[str, float] = dict(_LOAD_TIMINGS_MS)
        self.context_length: int | None = None
        self.effective_device: str | None = None
        self.calls: list[tuple[str, int]] = []
        self.closed = False
        self._capacity = capacity
        self._chat_template = chat_template_applied
        self._interrupt = interrupt
        self._eos_token_ids = _TOKENIZER_EOS

    @property
    def call_names(self) -> list[str]:
        return [name for name, _thread in self.calls]

    @property
    def task_default_eos_token_ids(self) -> tuple[int, ...]:
        self._record("task_default_eos_token_ids")
        return self._eos_token_ids

    def open_tokenizer(self) -> None:
        self._record("open_tokenizer")

    def encode_prompt(self, prompt: str, *, apply_template: bool = True) -> list[int]:
        self._record("encode_prompt")
        self.chat_template_applied = apply_template and self._chat_template
        return list(_PROMPT_IDS)

    def load(
        self, *, context_length: int | None = None, eos_token_ids: Sequence[int] | None = None
    ) -> None:
        self._record("load")
        if eos_token_ids is not None:
            self._eos_token_ids = tuple(eos_token_ids)
        self.context_length = self._capacity
        self.effective_device = "gpu"

    def generate_timed(
        self,
        prompt_token_ids: Iterable[int],
        *,
        max_new_tokens: int,
        clock: Callable[[], float] = time.perf_counter,
    ) -> TaskGenerationTiming:
        self._record("generate_timed")
        return make_timing(
            clock, tokens=range(max_new_tokens), input_tokens=len(list(prompt_token_ids))
        )

    def close(self) -> None:
        self._record("close")
        self.closed = True

    def _record(self, call: str) -> None:
        self.calls.append((call, threading.get_ident()))
        if self._interrupt == (call, self.call_names.count(call)):
            raise KeyboardInterrupt


def _config(tmp_path: Path, **overrides: Any) -> CGCPerfConfig:
    values: dict[str, Any] = {
        "model_path": tmp_path / "checkpoint",
        "model_format": "safetensors",
        "output_path": tmp_path / "report.json",
        "warmup": 1,
        "iterations": 2,
        "max_new_tokens": 3,
    }
    values.update(overrides)
    return CGCPerfConfig(**values)


def _run(
    tmp_path: Path, session: _FakeCGCSession | None = None, **overrides: Any
) -> tuple[CGCBenchmarkResult, _FakeCGCSession]:
    """Run the CGC benchmark on an injected fake session with a 1 ms tick clock."""
    fake = _FakeCGCSession() if session is None else session
    ticks = count(step=0.001)
    benchmark = CGCPerfBenchmark(
        _config(tmp_path, **overrides), session=fake, clock=lambda: next(ticks)
    )
    return benchmark.run(), fake


def _create_sessions_as(
    monkeypatch: pytest.MonkeyPatch, fake: _FakeCGCSession
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    """Make the binding's ``WinMLCGCSession`` return ``fake``; return its constructor calls."""
    constructed: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def create(*args: Any, **kwargs: Any) -> _FakeCGCSession:
        constructed.append((args, kwargs))
        return fake

    monkeypatch.setattr(perf_cgc, "WinMLCGCSession", create)
    return constructed


def _patch_memory(
    monkeypatch: pytest.MonkeyPatch,
    vram_mb: tuple[float | None, float | None] = _VRAM_MB,
) -> list[str | None]:
    """Replace the RSS and PDH readers; return the adapter LUIDs VRAM was read on."""
    luids: list[str | None] = []

    def read_vram(adapter_luid: str | None) -> tuple[float | None, float | None]:
        luids.append(adapter_luid)
        return vram_mb

    monkeypatch.setattr(_generation_metrics, "_get_vram_mb", read_vram)
    monkeypatch.setattr(_generation_metrics, "_get_rss_mb", lambda: 100.0)
    return luids


def _console_text(result: TaskBenchmarkResult) -> str:
    buffer = StringIO()
    display_task_report(result, Console(file=buffer, width=200, color_system=None))
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_identifies_the_cgc_backend(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="winml.modelkit.commands._task_generation_perf"):
            result, _fake = _run(tmp_path)
        info = result.to_dict()["benchmark_info"]

        assert result.to_dict()["schema_version"] == 3
        assert (info["backend"], info["model_format"], info["runtime"], info["api"]) == (
            "cgc",
            "safetensors",
            "winml-runtime",
            "windowsml.tasks",
        )
        assert (info["device"], info["effective_device"], info["ep"], info["experimental"]) == (
            "auto",
            "gpu",
            None,
            True,
        )
        assert info["adapter"] == _ADAPTER
        assert (info["vocabulary_size"], info["program_cache_dir"], info["context_length"]) == (
            151936,
            None,
            135,
        )
        assert info["packages"]["winml-cli"] == winml.modelkit.__version__
        assert set(info["packages"]) == {"windowsml", "winml-cli"}
        assert "llama_backends" not in info
        assert any(record.getMessage().startswith("CGC perf:") for record in caplog.records)

    def test_benchmark_info_keys_match_gguf_except_backend_fields(self, tmp_path: Path) -> None:
        gguf = GGUFBenchmarkResult(GGUFPerfConfig(model_path=tmp_path / "m.gguf")).to_dict()[
            "benchmark_info"
        ]
        cgc = CGCBenchmarkResult(_config(tmp_path)).to_dict()["benchmark_info"]

        assert set(cgc) - {"adapter", "vocabulary_size", "program_cache_dir"} == set(gguf) - {
            "llama_backends"
        }

    def test_benchmark_info_key_order_puts_backend_fields_after_packages(
        self, tmp_path: Path
    ) -> None:
        # Characterization: both bindings keep the llama.cpp report order and add
        # their own fields where llama_backends sits, right after packages.
        gguf = GGUFBenchmarkResult(GGUFPerfConfig(model_path=tmp_path / "m.gguf")).to_dict()[
            "benchmark_info"
        ]
        cgc = CGCBenchmarkResult(_config(tmp_path)).to_dict()["benchmark_info"]
        at = _LLAMA_CPP_INFO_KEYS.index("llama_backends")

        assert _LLAMA_CPP_INFO_KEYS[at - 1] == "packages"
        assert list(gguf) == _LLAMA_CPP_INFO_KEYS
        assert list(cgc) == [
            *_LLAMA_CPP_INFO_KEYS[:at],
            "adapter",
            "vocabulary_size",
            "program_cache_dir",
            *_LLAMA_CPP_INFO_KEYS[at + 1 :],
        ]

    def test_load_metrics_split_set_model_and_first_session(self, tmp_path: Path) -> None:
        result, _fake = _run(tmp_path)
        load = result.to_dict()["load"]

        assert list(load) == [
            "session_load_duration_ms",
            "runtime_create_duration_ms",
            "target_create_duration_ms",
            "model_create_duration_ms",
            "set_model_duration_ms",
            "session_create_duration_ms",
            "native_load_duration_ms",
            "weight_upload_duration_ms",
            "weight_upload_estimate_duration_ms",
            "weight_upload_estimate_source",
        ]
        assert {key: load[key] for key in _LOAD_TIMINGS_MS} == _LOAD_TIMINGS_MS
        assert load["native_load_duration_ms"] == (
            _LOAD_TIMINGS_MS["model_create_duration_ms"]
            + _LOAD_TIMINGS_MS["set_model_duration_ms"]
            + _LOAD_TIMINGS_MS["session_create_duration_ms"]
        )
        assert load["weight_upload_duration_ms"] is None
        assert load["weight_upload_estimate_duration_ms"] is None
        assert load["weight_upload_estimate_source"] == "unavailable"

    def test_memory_reports_vram_on_the_pinned_adapter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        luids = _patch_memory(monkeypatch)
        result, _fake = _run(tmp_path, memory=True)
        memory = result.to_dict()["memory"]

        # Baseline, after load and after the benchmark, each on the pinned GPU.
        assert luids == [_ADAPTER["luid"]] * 3
        assert memory["vram_local_after_compile_mb"] == _VRAM_MB[0]
        assert memory["vram_shared_after_compile_mb"] == _VRAM_MB[1]
        vram = {key: value for key, value in memory.items() if key.startswith("vram_")}
        assert len(vram) == 14
        assert None not in vram.values()

    def test_missing_chat_template_is_reported(self, tmp_path: Path) -> None:
        result, _fake = _run(tmp_path, _FakeCGCSession(chat_template_applied=False))

        assert result.to_dict()["benchmark_info"]["chat_template_applied"] is False
        assert "(no chat template; raw prompt)" in _console_text(result)

    def test_unreported_capacity_is_null_and_shown_as_unavailable(self, tmp_path: Path) -> None:
        result, _fake = _run(tmp_path, _FakeCGCSession(capacity=None, vocabulary=None))
        info = result.to_dict()["benchmark_info"]

        assert info["context_length"] is None
        assert info["vocabulary_size"] is None
        assert "Capacity: unavailable" in _console_text(result)


# ---------------------------------------------------------------------------
# Load checks
# ---------------------------------------------------------------------------


class TestLoadChecks:
    def test_context_length_needs_a_capacity_read_back(self, tmp_path: Path) -> None:
        fake = _FakeCGCSession(capacity=None)

        with pytest.raises(
            CGCLoadError,
            match=re.escape(
                "--context-length 256 could not be verified: Windows ML did not report the "
                "logical capacity."
            ),
        ):
            _run(tmp_path, fake, context_length=256)

        assert "load" in fake.call_names
        assert "generate_timed" not in fake.call_names

    def test_context_length_mismatch_keeps_the_shared_wording(self, tmp_path: Path) -> None:
        fake = _FakeCGCSession(capacity=512)

        with pytest.raises(
            CGCLoadError,
            match="--context-length 256 resolved to an effective sequence capacity of 512",
        ):
            _run(tmp_path, fake, context_length=256)

        assert "generate_timed" not in fake.call_names


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    def test_every_session_call_runs_on_the_calling_thread(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeCGCSession()
        _create_sessions_as(monkeypatch, fake)

        # --ignore-eos adds the native EOS read-back to the calls.
        CGCPerfBenchmark(_config(tmp_path, ignore_eos=True)).run()

        assert set(fake.call_names) == {
            "open_tokenizer",
            "encode_prompt",
            "load",
            "task_default_eos_token_ids",
            "generate_timed",
            "close",
        }
        assert fake.call_names[-1] == "close"
        assert {ident for _, ident in fake.calls} == {threading.get_ident()}

    def test_new_session_pins_the_requested_gpu_and_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        model = tmp_path / "model.gguf"
        cache = tmp_path / "programs"
        constructed = _create_sessions_as(monkeypatch, _FakeCGCSession())
        config = _config(
            tmp_path,
            model_path=model,
            model_format="gguf",
            device_luid=_ADAPTER["luid"],
            program_cache_dir=cache,
        )

        result = CGCPerfBenchmark(config).run()

        assert constructed == [
            (
                (model,),
                {
                    "model_format": "gguf",
                    "device_luid": _ADAPTER["luid"],
                    "program_cache_dir": cache,
                },
            )
        ]
        assert result.to_dict()["benchmark_info"]["program_cache_dir"] == str(cache)

    def test_missing_gguf_shard_fails_before_windows_ml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entry = tmp_path / "m-00001-of-00002.gguf"
        entry.write_bytes(b"GGUF")
        constructed = _create_sessions_as(monkeypatch, _FakeCGCSession())
        config = _config(tmp_path, model_path=entry, model_format="gguf")

        with pytest.raises(click.ClickException, match=re.escape("split shard(s) not found")):
            run_cgc_perf(config, console=Console(file=StringIO()), json_mode=False)

        assert constructed == []
        assert not config.output_path.exists()

    @pytest.mark.parametrize(
        "interrupt",
        [("load", 1), ("generate_timed", 2)],
        ids=["first-create-session", "mid-generation"],
    )
    def test_interrupt_releases_the_session_and_writes_no_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interrupt: tuple[str, int]
    ) -> None:
        fake = _FakeCGCSession(interrupt=interrupt)
        _create_sessions_as(monkeypatch, fake)
        config = _config(tmp_path)

        with pytest.raises(KeyboardInterrupt):
            run_cgc_perf(config, console=Console(file=StringIO()), json_mode=False)

        assert fake.closed
        assert fake.call_names[-1] == "close"
        assert not config.output_path.exists()


# ---------------------------------------------------------------------------
# Console view and entry point
# ---------------------------------------------------------------------------


class TestConsoleAndEntryPoint:
    @pytest.mark.parametrize(
        ("vram_mb", "gpu_memory"),
        [
            (_VRAM_MB, "256.0/32.0 MiB after load  |  256.0/32.0 MiB after inference"),
            (
                (None, None),
                "unavailable/unavailable MiB after load  |  "
                "unavailable/unavailable MiB after inference",
            ),
        ],
        ids=["reported", "unreported"],
    )
    def test_console_report_names_backend_adapter_and_compile(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        vram_mb: tuple[float | None, float | None],
        gpu_memory: str,
    ) -> None:
        _patch_memory(monkeypatch, vram_mb)
        result, _fake = _run(tmp_path, memory=True)
        text = _console_text(result)
        lines = [line.strip() for line in text.splitlines()]
        model_line = next(index for index, line in enumerate(lines) if line.startswith("Model:"))

        assert "winml-runtime (cgc, experimental)" in text
        assert lines[model_line + 1 : model_line + 5] == [
            "Adapter: Fake GPU (0x00000000_0x0000BEEF, high-performance)",
            "Format: safetensors",
            "Capacity: 135 tokens",
            "Program cache: off (cold compile every run)",
        ]
        # set_model and the first create_session, each from its own span.
        assert (
            f"set_model: {_LOAD_TIMINGS_MS['set_model_duration_ms']:.2f} ms  |  "
            "First session (GPU compile): "
            f"{_LOAD_TIMINGS_MS['session_create_duration_ms']:.2f} ms"
        ) in text
        assert f"GPU local/shared: {gpu_memory}" in text
        # Every marker the llama.cpp check below looks for is in a CGC report.
        assert [expected for expected in _CGC_ONLY_TEXT if expected not in text] == []

    def test_llama_cpp_report_shows_no_cgc_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The same session data through the llama.cpp binding.
        _patch_memory(monkeypatch)
        ticks = count(step=0.001)
        config = GGUFPerfConfig(
            model_path=tmp_path / "m.gguf", memory=True, warmup=1, iterations=2, max_new_tokens=3
        )
        result = GGUFPerfBenchmark(
            config, session=_FakeCGCSession(), clock=lambda: next(ticks)
        ).run()
        text = _console_text(result)

        assert "llama.cpp, experimental" in text
        assert "Memory:" in text
        assert [expected for expected in _CGC_ONLY_TEXT if expected in text] == []

    @pytest.mark.parametrize(
        "error", [CGCLoadError, CGCNotInstalledError, CGCSessionError, FileNotFoundError]
    )
    def test_session_errors_become_click_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: type[Exception]
    ) -> None:
        def fail(_self: CGCPerfBenchmark) -> CGCBenchmarkResult:
            raise error("boom")

        monkeypatch.setattr(CGCPerfBenchmark, "run", fail)
        config = _config(tmp_path)

        with pytest.raises(click.ClickException, match=r"^boom$"):
            run_cgc_perf(config, console=Console(file=StringIO()), json_mode=False)

        assert not config.output_path.exists()

    def test_json_mode_prints_only_the_report(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _create_sessions_as(monkeypatch, _FakeCGCSession())
        config = _config(tmp_path)

        run_cgc_perf(config, console=Console(file=StringIO()), json_mode=True)

        printed = json.loads(capsys.readouterr().out)
        assert printed["benchmark_info"]["backend"] == "cgc"
        assert printed == json.loads(config.output_path.read_text(encoding="utf-8"))
