# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the shared Task generation perf runner without native dependencies.

A minimal binding over a fake Task generation session drives the shared engine,
so the runner sees real results and real failures. The llama.cpp binding is
tested in ``test_perf_gguf.py``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import StringIO
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import pytest
from rich.console import Console

from tests.unit.test_helpers import make_timing
from winml.modelkit.commands._task_generation_perf import (
    RUNTIME_TYPE,
    TaskBenchmarkResult,
    TaskPerfBenchmark,
    run_task_perf,
)
from winml.modelkit.session import TaskGenerationTiming, TaskSessionError


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence


# ---------------------------------------------------------------------------
# A minimal binding
# ---------------------------------------------------------------------------


class _BindingError(TaskSessionError):
    """The session error the fake binding declares."""


class _BindingLoadError(_BindingError):
    """A load failure of the fake binding."""


class _OtherEngineError(TaskSessionError):
    """A Task session error of an engine the fake binding does not own."""


@dataclass
class _Config:
    model_path: Path
    device: str = "gpu"
    requested_device: str = "auto"
    prompt: str = "Name three primary colors."
    apply_template: bool = True
    max_new_tokens: int = 3
    iterations: int = 2
    warmup: int = 1
    memory: bool = False
    prompt_ids: tuple[int, ...] | None = None
    context_length: int | None = None
    ignore_eos: bool = False


@dataclass
class _Result(TaskBenchmarkResult):
    BACKEND = "fake-engine"


class _FakeSession:
    """Task generation session whose requests raise ``error`` when one is given."""

    def __init__(self, error: BaseException | None = None) -> None:
        self._error = error
        self.chat_template_applied: bool | None = None
        self.load_timings_ms: dict[str, float] = {}
        self.context_length: int | None = None
        self.effective_device: str | None = None
        self.task_default_eos_token_ids: tuple[int, ...] = (2,)
        self.closed = False

    def open_tokenizer(self) -> None:
        pass

    def encode_prompt(self, prompt: str, *, apply_template: bool = True) -> list[int]:
        self.chat_template_applied = apply_template
        return [1, 2, 3, 4]

    def load(
        self, *, context_length: int | None = None, eos_token_ids: Sequence[int] | None = None
    ) -> None:
        self.context_length = context_length
        self.effective_device = "gpu"

    def generate_timed(
        self,
        prompt_token_ids: Iterable[int],
        *,
        max_new_tokens: int,
        clock: Callable[[], float],
    ) -> TaskGenerationTiming:
        if self._error is not None:
            raise self._error
        return make_timing(
            clock, tokens=range(max_new_tokens), input_tokens=len(list(prompt_token_ids))
        )

    def close(self) -> None:
        self.closed = True


class _Benchmark(TaskPerfBenchmark[_Config, _Result]):
    """The fake binding: it owns and closes ``fake``, as a real binding owns its session."""

    SESSION_ERROR = _BindingError
    LOAD_ERROR = _BindingLoadError
    LOG_LABEL = "Fake"

    def __init__(self, config: _Config, fake: _FakeSession) -> None:
        ticks = count(step=0.001)
        super().__init__(config, clock=lambda: next(ticks))
        self.fake = fake

    def _new_session(self) -> _FakeSession:
        return self.fake

    def _new_memory_tracker(self, session: Any) -> None:
        return None

    def _new_result(self, session: Any, **fields: Any) -> _Result:
        return _Result(packages={"fake-runtime": "1.0"}, model_format="fake-format", **fields)

    def _load_metrics(self, session_load_ms: float) -> dict[str, float | str | None]:
        return {"session_load_duration_ms": session_load_ms}


class _Run:
    """One :func:`run_task_perf` call on the fake binding, and what it left behind."""

    def __init__(self, tmp_path: Path, error: BaseException | None = None) -> None:
        self.session = _FakeSession(error)
        self.benchmark = _Benchmark(_Config(model_path=tmp_path / "model"), self.session)
        self.report = tmp_path / "out" / "report.json"
        self.resolved: list[Path] = []
        self.console_output = StringIO()

    def resolve_output_path(self) -> Path:
        self.resolved.append(self.report)
        return self.report

    def __call__(self, *, json_mode: bool) -> _Result:
        return run_task_perf(
            self.benchmark,
            resolve_output_path=self.resolve_output_path,
            console=Console(file=self.console_output, width=1000, color_system=None),
            json_mode=json_mode,
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class TestRunTaskPerf:
    def test_console_mode_shows_the_report_then_saves_it(self, tmp_path: Path) -> None:
        run = _Run(tmp_path)
        result = run(json_mode=False)

        output = run.console_output.getvalue()
        assert run.resolved == [run.report]
        assert json.loads(run.report.read_text(encoding="utf-8")) == result.to_dict()
        assert f"{RUNTIME_TYPE} ({_Result.BACKEND}, experimental)" in output
        assert output.rstrip().endswith(f"Results saved to: {run.report}")
        assert run.session.closed

    def test_json_mode_prints_the_saved_report_instead_of_the_console_view(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        run = _Run(tmp_path)
        result = run(json_mode=True)

        saved = json.loads(run.report.read_text(encoding="utf-8"))
        assert json.loads(capsys.readouterr().out) == saved == result.to_dict()
        assert run.console_output.getvalue().strip() == f"Results saved to: {run.report}"

    @pytest.mark.parametrize(
        "error",
        [
            _BindingError("the session failed"),
            _BindingLoadError("the model did not load"),
            FileNotFoundError("model not found"),
        ],
        ids=lambda error: type(error).__name__,
    )
    def test_binding_failures_become_click_errors_without_a_report(
        self, tmp_path: Path, error: Exception
    ) -> None:
        run = _Run(tmp_path, error)

        with pytest.raises(click.ClickException) as raised:
            run(json_mode=False)

        assert raised.value.message == str(error)
        assert raised.value.__cause__ is error
        assert run.resolved == []
        assert not run.report.parent.exists()
        assert run.session.closed

    @pytest.mark.parametrize(
        "error",
        [KeyboardInterrupt(), _OtherEngineError("another engine failed"), RuntimeError("bug")],
        ids=lambda error: type(error).__name__,
    )
    def test_other_failures_propagate_without_a_report(
        self, tmp_path: Path, error: BaseException
    ) -> None:
        run = _Run(tmp_path, error)

        with pytest.raises(type(error)) as raised:
            run(json_mode=False)

        assert raised.value is error
        assert run.resolved == []
        assert not run.report.parent.exists()
        assert run.console_output.getvalue() == ""
        assert run.session.closed
