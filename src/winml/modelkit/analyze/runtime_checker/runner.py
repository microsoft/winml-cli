# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# python >= 3.8
import concurrent.futures as cf
import multiprocessing as mp
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


class _OutputCapturingWrapper:
    """Picklable wrapper that optionally captures stdout/stderr from a function.

    Captures both Python-level (sys.stdout/stderr) and OS-level (file descriptor)
    output to properly capture C/C++ library output (like ONNX Runtime).

    Always returns a dict with 'result', 'stdout', 'stderr' for consistent interface.
    When capture is disabled, stdout and stderr are None.

    This is a module-level class (not a nested function) so it can be pickled
    for multiprocessing.
    """

    def __init__(self, fn: Callable[[Any, Any], Any], capture: bool = True) -> None:
        self.fn = fn
        self.capture = capture

    def __call__(self, *args: Any) -> dict[str, Any]:
        # TODO: ORT logs like the following are not always captured, why?
        # [W:onnxruntime:, qnn_model_wrapper.cc:263
        # onnxruntime::qnn::QnnModelWrapper::CreateQnnNode]
        # QNN.backendValidateOpConfig() failed for node `n1` of type `Reshape`
        # with error code 3110
        if not self.capture:
            # No capture: execute directly and return None for stdout/stderr
            result = self.fn(*args)
            return {"result": result, "stdout": None, "stderr": None}

        # Capture mode: redirect both Python-level and OS-level streams
        import os
        import tempfile

        # Create temporary files for capturing OS-level output
        stdout_fd, stdout_path_str = tempfile.mkstemp(suffix=".out", text=True)
        stderr_fd, stderr_path_str = tempfile.mkstemp(suffix=".err", text=True)
        stdout_path = Path(stdout_path_str)
        stderr_path = Path(stderr_path_str)

        # Save original file descriptors
        old_stdout_fd = os.dup(1)  # Duplicate stdout fd
        old_stderr_fd = os.dup(2)  # Duplicate stderr fd

        # Save Python-level streams
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            # Redirect OS-level file descriptors to temp files
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)

            # Redirect Python-level streams to the same temp files
            sys.stdout = os.fdopen(os.dup(1), "w", encoding="utf-8", errors="replace")
            sys.stderr = os.fdopen(os.dup(2), "w", encoding="utf-8", errors="replace")

            # Execute the function
            result = self.fn(*args)

            # Flush all streams to ensure all output is written
            sys.stdout.flush()
            sys.stderr.flush()
            os.fsync(1)
            os.fsync(2)

        finally:
            # Restore Python-level streams first
            try:
                sys.stdout.close()
            except Exception:
                pass
            try:
                sys.stderr.close()
            except Exception:
                pass

            # Restore OS-level file descriptors
            os.dup2(old_stdout_fd, 1)
            os.dup2(old_stderr_fd, 2)
            os.close(old_stdout_fd)
            os.close(old_stderr_fd)

            # Restore original Python streams
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            # Close temp file descriptors
            os.close(stdout_fd)
            os.close(stderr_fd)

            # Read captured output from temp files
            try:
                with stdout_path.open(encoding="utf-8", errors="replace") as f:
                    stdout_content = f.read()
            except Exception as e:
                stdout_content = f"[Error reading stdout: {e}]"

            try:
                with stderr_path.open(encoding="utf-8", errors="replace") as f:
                    stderr_content = f.read()
            except Exception as e:
                stderr_content = f"[Error reading stderr: {e}]"

            # Clean up temp files
            try:
                stdout_path.unlink()
            except Exception:
                pass
            try:
                stderr_path.unlink()
            except Exception:
                pass

        return {"result": result, "stdout": stdout_content, "stderr": stderr_content}


class ResilientRunner:
    """Execute functions with automatic recovery from worker crashes and retries.

    The main goal of runnuing tests in child processes instead of the parent
    process is to avoid crashing the parent process. E.g. this input crashes
    the compiler targetting QNN EP:

    @onnxscript.script()
    def op_func(data: INT64[2, 3, 2, 2]):
        return opset.Reshape(
            allowzero=0, data=data, shape=opset.Constant(value=[2, 3, 2, 1, 2])
        )
    """

    def __init__(
        self,
        *,
        max_retries: int = 1,
        timeout_sec: float | None = None,
        capture_output: bool = False,
    ) -> None:
        """Initialize the resilient runner.

        Args:
            fn: The function to execute
            max_retries: Maximum number of retry attempts
            timeout_sec: Timeout for each execution attempt
            capture_output: If True, capture stdout/stderr from subprocess and return it
        """
        self.capture_output = capture_output
        self.max_retries = max_retries
        self.timeout_sec = timeout_sec
        self.ctx = mp.get_context("spawn")  # avoid fork-related instability
        self.executor = self._new_executor()
        self._executor_needs_recreate = False

    _GRACEFUL_SHUTDOWN_TIMEOUT_SEC = 2.0
    _FORCED_KILL_JOIN_TIMEOUT_SEC = 0.2

    def _new_executor(self) -> cf.ProcessPoolExecutor:
        """Create a new single-worker process pool executor."""
        return cf.ProcessPoolExecutor(max_workers=1, mp_context=self.ctx)

    @staticmethod
    def _snapshot_processes(executor: cf.ProcessPoolExecutor) -> list[Any]:
        """Best-effort snapshot of worker process handles from the executor."""
        try:
            processes = getattr(executor, "_processes", None)
            if not processes:
                return []
            return [proc for proc in processes.values() if proc is not None]
        except Exception:
            return []

    @staticmethod
    def _is_process_alive(proc: Any) -> bool:
        """Check process liveness without propagating process-state errors."""
        try:
            return bool(proc.is_alive())
        except Exception:
            return False

    @staticmethod
    def _join_process(proc: Any, timeout: float | None = None) -> None:
        """Best-effort process join that never raises."""
        try:
            proc.join(timeout=timeout)
        except Exception as e:
            print(f"Warning: failed to join process during executor shutdown: {e}", file=sys.stderr)

    @staticmethod
    def _kill_process(proc: Any) -> None:
        """Best-effort process kill that never raises."""
        try:
            proc.kill()
        except Exception as e:
            # Keep cleanup non-fatal, but surface the failure for diagnostics.
            print(f"Warning: failed to kill worker process: {e}", file=sys.stderr)

    @staticmethod
    def _close_process(proc: Any) -> None:
        """Best-effort process close that never raises."""
        try:
            proc.close()
        except Exception as ex:
            print(f"Warning: failed to close process: {ex}", file=sys.stderr)

    def _shutdown_executor_two_phase(
        self,
        *,
        cancel_futures: bool,
        graceful_timeout_sec: float | None = None,
    ) -> None:
        """Shutdown executor with graceful wait, then force-kill lingering workers."""
        executor = self.executor
        lingering = self._snapshot_processes(executor)

        try:
            executor.shutdown(wait=False, cancel_futures=cancel_futures)
        except Exception as exc:
            # Best-effort shutdown: keep cleanup flow non-raising, but surface failure.
            print(f"Executor shutdown failed during cleanup: {exc}", file=sys.stderr)

        timeout = (
            self._GRACEFUL_SHUTDOWN_TIMEOUT_SEC
            if graceful_timeout_sec is None
            else max(0.0, graceful_timeout_sec)
        )
        deadline = time.monotonic() + timeout

        for proc in lingering:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._join_process(proc, timeout=remaining)

        survivors = [proc for proc in lingering if self._is_process_alive(proc)]

        for proc in survivors:
            self._kill_process(proc)

        for proc in survivors:
            self._join_process(proc, timeout=self._FORCED_KILL_JOIN_TIMEOUT_SEC)

        for proc in lingering:
            self._close_process(proc)

    def run(self, fn: Callable[[Any, Any], Any] | None = None, *args: Any) -> dict[str, Any]:
        """Execute the function on a single input with automatic retry on failure.

        Args:
            fn: The function to execute
            args: Arguments to pass to the function

        Returns:
            A dict with keys:
            - 'result': The result of the function execution
            - 'stdout': Captured stdout (str if capture_output=True, None otherwise)
            - 'stderr': Captured stderr (str if capture_output=True, None otherwise)

        Raises:
            RuntimeError: If max retries exceeded or function fails
        """
        if fn is None:
            return {
                "result": {"success": None, "reason": None},
                "stdout": None,
                "stderr": None,
            }

        if self._executor_needs_recreate:
            self.executor = self._new_executor()
            self._executor_needs_recreate = False

        attempts = 0
        while True:
            attempts += 1
            future = self.executor.submit(
                _OutputCapturingWrapper(fn, capture=self.capture_output), *args
            )
            try:
                return future.result(timeout=self.timeout_sec)
            except Exception as e:
                try:
                    future.cancel()
                except Exception:
                    # Best-effort cleanup: ignore cancel failures so retry flow can continue.
                    pass

                self._shutdown_executor_two_phase(cancel_futures=True)

                if attempts >= self.max_retries:
                    self._executor_needs_recreate = True
                    # TODO: capture stdout/stderr on timeout/crashed inputs
                    return {
                        "result": {
                            "success": False,
                            "reason": (f"Timeout/crash/fail for {attempts} attempts: {e!s}"),
                        },
                        "stdout": None,
                        "stderr": None,
                    }
                self.executor = self._new_executor()
                continue

    def shutdown(self) -> None:
        """Shut down the executor cleanly."""
        self._shutdown_executor_two_phase(cancel_futures=False)

    def __enter__(self) -> "ResilientRunner":
        """Support context manager protocol."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Ensure executor is shut down when exiting context."""
        self.shutdown()
