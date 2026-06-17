# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Retry helpers for flaky E2E dataset downloads.

E2E tests invoke ``winml eval`` / ``winml perf`` in-process against real
HuggingFace datasets. Streaming parquet fetches from the xet CDN intermittently
fail with transient network errors (most commonly ``408 Request Time-out``),
which turns the daily e2e pipeline red even though the code under test is fine.

``RetryingCliRunner`` transparently retries an invocation when it fails with a
*transient* network error, so flaky blips don't fail the suite. The retry lives
entirely in the test harness — the shipped ``winml`` CLI behaviour is unchanged.

Tunable via environment variables:
    WINMLCLI_E2E_DATASET_RETRIES:     max attempts (default 4)
    WINMLCLI_E2E_DATASET_RETRY_DELAY: base backoff seconds (default 3.0)
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from click.testing import CliRunner


if TYPE_CHECKING:
    from collections.abc import Callable

    from click.testing import Result

_DEFAULT_RETRIES = 4
_DEFAULT_BASE_DELAY = 3.0

# Substrings (lower-cased) that mark a transient/retryable dataset-download
# failure in the captured CLI output. Kept narrow so genuine eval failures
# (assertion errors, bad metrics, schema mismatches) are never retried.
_TRANSIENT_MARKERS = (
    "request time-out",
    "time-out for url",
    "timed out",
    "read timed out",
    "connection reset",
    "connection aborted",
    "connection error",
    "temporarily unavailable",
    "max retries exceeded",
    "408 client error",
    "429 client error",
    "500 server error",
    "502 server error",
    "503 server error",
    "504 server error",
    "couldn't connect to",
    "failed to fetch",
)


def _retries() -> int:
    raw = os.environ.get("WINMLCLI_E2E_DATASET_RETRIES")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return _DEFAULT_RETRIES


def _base_delay() -> float:
    raw = os.environ.get("WINMLCLI_E2E_DATASET_RETRY_DELAY")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return _DEFAULT_BASE_DELAY


def is_transient_output(output: str | None) -> bool:
    """Return True if *output* looks like a transient dataset-download failure."""
    if not output:
        return False
    lowered = output.lower()
    return any(marker in lowered for marker in _TRANSIENT_MARKERS)


def invoke_with_retry(
    invoke: Callable[[], Result],
    *,
    retries: int | None = None,
    base_delay: float | None = None,
) -> Result:
    """Call *invoke* (a zero-arg invocation), retrying transient failures.

    A retry happens only when the returned ``Result`` has a non-zero exit code
    *and* its output matches a known transient network-error marker. Any other
    non-zero exit (real test failure) is returned immediately so genuine
    regressions are never masked.

    Args:
        invoke: Zero-arg callable returning a click ``Result``.
        retries: Max attempts (defaults to env/``_DEFAULT_RETRIES``).
        base_delay: Base backoff seconds (defaults to env/``_DEFAULT_BASE_DELAY``).

    Returns:
        The last ``Result`` produced by *invoke*.
    """
    attempts = retries if retries is not None else _retries()
    delay = base_delay if base_delay is not None else _base_delay()

    result = invoke()
    for attempt in range(1, attempts):
        if result.exit_code == 0 or not is_transient_output(result.output):
            return result
        sleep_for = delay * (2 ** (attempt - 1))
        print(
            f"[e2e-retry] transient dataset error (attempt {attempt}/{attempts}); "
            f"retrying in {sleep_for:.1f}s...",
        )
        time.sleep(sleep_for)
        result = invoke()
    return result


class RetryingCliRunner(CliRunner):
    """A ``CliRunner`` whose ``invoke`` retries transient dataset-download errors.

    Drop-in replacement for ``CliRunner`` in E2E fixtures. Non-transient
    failures behave exactly like the base runner (returned, not retried), so
    error-path tests are unaffected.
    """

    def invoke(self, *args: Any, **kwargs: Any) -> Result:
        return invoke_with_retry(lambda: super(RetryingCliRunner, self).invoke(*args, **kwargs))
