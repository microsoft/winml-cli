# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the E2E transient-download retry helper (tests/e2e/_retry.py).

Pure unit tests — no hardware or network required — so they live under
``tests/unit`` and run in the normal suite (the helper itself lives under
``tests/e2e`` because only the e2e harness uses it).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tests.e2e import _retry


@dataclass
class _FakeResult:
    """Minimal stand-in for click.testing.Result."""

    exit_code: int
    output: str


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_retry.time, "sleep", lambda _s: None)


@pytest.mark.parametrize(
    "output",
    [
        "Error: Evaluation failed: 408 Client Error: Request Time-out for url: https://...",
        "Read timed out.",
        "Connection aborted",
        "Max retries exceeded with url",
        "503 Server Error: Service Unavailable",
    ],
)
def test_is_transient_output_true(output: str) -> None:
    assert _retry.is_transient_output(output)


@pytest.mark.parametrize(
    "output",
    [
        "",
        None,
        "AssertionError: metric out of range",
        "Dataset schema mismatch: missing column 'label'",
        "404 Client Error: Not Found",
    ],
)
def test_is_transient_output_false(output: str | None) -> None:
    assert not _retry.is_transient_output(output)


def test_success_first_try_no_retry() -> None:
    calls = {"n": 0}

    def invoke() -> _FakeResult:
        calls["n"] += 1
        return _FakeResult(exit_code=0, output="ok")

    result = _retry.invoke_with_retry(invoke, retries=4, base_delay=0)
    assert result.exit_code == 0
    assert calls["n"] == 1


def test_retries_transient_then_succeeds() -> None:
    calls = {"n": 0}

    def invoke() -> _FakeResult:
        calls["n"] += 1
        if calls["n"] < 3:
            return _FakeResult(exit_code=1, output="408 Client Error: Request Time-out")
        return _FakeResult(exit_code=0, output="ok")

    result = _retry.invoke_with_retry(invoke, retries=4, base_delay=0)
    assert result.exit_code == 0
    assert calls["n"] == 3


def test_exhausts_retries_returns_last_transient() -> None:
    calls = {"n": 0}

    def invoke() -> _FakeResult:
        calls["n"] += 1
        return _FakeResult(exit_code=1, output="Request Time-out for url: https://...")

    result = _retry.invoke_with_retry(invoke, retries=3, base_delay=0)
    assert result.exit_code == 1
    assert calls["n"] == 3


def test_non_transient_failure_not_retried() -> None:
    calls = {"n": 0}

    def invoke() -> _FakeResult:
        calls["n"] += 1
        return _FakeResult(exit_code=1, output="AssertionError: metric out of range")

    result = _retry.invoke_with_retry(invoke, retries=4, base_delay=0)
    assert result.exit_code == 1
    assert calls["n"] == 1


def test_retry_count_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WINMLCLI_E2E_DATASET_RETRIES", "2")
    calls = {"n": 0}

    def invoke() -> _FakeResult:
        calls["n"] += 1
        return _FakeResult(exit_code=1, output="connection reset")

    result = _retry.invoke_with_retry(invoke, base_delay=0)
    assert result.exit_code == 1
    assert calls["n"] == 2


def test_retrying_cli_runner_retries_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    """RetryingCliRunner retries a real click command that flakes transiently."""
    import click

    monkeypatch.setattr(_retry.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    @click.command()
    def flaky() -> None:
        calls["n"] += 1
        if calls["n"] < 2:
            raise click.ClickException("Evaluation failed: 408 Client Error: Request Time-out")

    runner = _retry.RetryingCliRunner()
    result = runner.invoke(flaky, [])
    assert result.exit_code == 0
    assert calls["n"] == 2


def test_retrying_cli_runner_does_not_retry_real_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    import click

    monkeypatch.setattr(_retry.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    @click.command()
    def broken() -> None:
        calls["n"] += 1
        raise click.ClickException("Dataset schema mismatch: missing column 'label'")

    runner = _retry.RetryingCliRunner()
    result = runner.invoke(broken, [])
    assert result.exit_code != 0
    assert calls["n"] == 1
