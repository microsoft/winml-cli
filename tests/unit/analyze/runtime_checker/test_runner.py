# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Unit tests for ResilientRunner process lifecycle and recovery behavior."""

from __future__ import annotations

from typing import Any

from winml.modelkit.analyze.runtime_checker.runner import ResilientRunner


class _FakeFuture:
    def __init__(self, outcome: Any):
        self._outcome = outcome
        self.cancel_called = False

    def result(self, timeout: float | None = None) -> dict[str, Any]:
        _ = timeout
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome

    def cancel(self) -> bool:
        self.cancel_called = True
        return True


class _FakeProcess:
    def __init__(self) -> None:
        self.alive = True
        self.killed = False
        self.closed = False
        self.join_calls: list[float | None] = []

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.join_calls.append(timeout)

    def kill(self) -> None:
        self.killed = True
        self.alive = False

    def close(self) -> None:
        self.closed = True


class _FakeExecutor:
    def __init__(self, futures: list[_FakeFuture], processes: list[_FakeProcess] | None = None):
        self._futures = futures
        self._processes = dict(enumerate(processes or []))
        self.shutdown_calls: list[tuple[bool, bool]] = []
        self.submit_calls = 0

    def submit(self, fn, *args):
        _ = fn
        _ = args
        self.submit_calls += 1
        return self._futures.pop(0)

    def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
        self.shutdown_calls.append((wait, cancel_futures))


class TestResilientRunner:
    def test_run_recreates_executor_lazily_after_terminal_failure(self, monkeypatch):
        failure_future = _FakeFuture(TimeoutError("timed out"))
        success_payload = {
            "result": {"success": True, "reason": None},
            "stdout": None,
            "stderr": None,
        }
        success_future = _FakeFuture(success_payload)

        first_executor = _FakeExecutor([failure_future])
        second_executor = _FakeExecutor([success_future])
        created_executors = [first_executor, second_executor]
        create_calls: list[int] = []

        def _fake_new_executor(self) -> _FakeExecutor:
            _ = self
            create_calls.append(1)
            return created_executors[len(create_calls) - 1]

        monkeypatch.setattr(ResilientRunner, "_new_executor", _fake_new_executor)

        runner = ResilientRunner(max_retries=1, timeout_sec=0.001)

        first_result = runner.run(lambda: None)
        assert first_result["result"]["success"] is False
        assert runner._executor_needs_recreate is True
        assert runner.executor is first_executor
        assert len(create_calls) == 1

        second_result = runner.run(lambda: None)
        assert second_result == success_payload
        assert runner._executor_needs_recreate is False
        assert runner.executor is second_executor
        assert len(create_calls) == 2

    def test_shutdown_executor_two_phase_kills_survivors(self, monkeypatch):
        worker = _FakeProcess()
        executor = _FakeExecutor([], processes=[worker])

        def _fake_new_executor(self) -> _FakeExecutor:
            _ = self
            return executor

        monkeypatch.setattr(ResilientRunner, "_new_executor", _fake_new_executor)

        runner = ResilientRunner()
        runner._shutdown_executor_two_phase(cancel_futures=True, graceful_timeout_sec=0.0)

        assert executor.shutdown_calls == [(False, True), (True, True)]
        assert worker.killed is True
        assert worker.closed is False
        assert worker.join_calls == [runner._FORCED_KILL_JOIN_TIMEOUT_SEC]
