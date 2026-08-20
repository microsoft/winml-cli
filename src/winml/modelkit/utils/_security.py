# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Internal execution policy for code-loading entry points."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Iterator


_REMOTE_CODE_EXECUTION_ALLOWED: ContextVar[bool] = ContextVar(
    "remote_code_execution_allowed",
    default=True,
)


@contextmanager
def _disable_remote_code_execution() -> Iterator[None]:
    """Disable remote or user-provided code for the current invocation."""
    token = _REMOTE_CODE_EXECUTION_ALLOWED.set(False)
    try:
        yield
    finally:
        _REMOTE_CODE_EXECUTION_ALLOWED.reset(token)


def _require_remote_code_execution_allowed() -> None:
    """Reject code-loading paths disabled by the current execution policy."""
    if not _REMOTE_CODE_EXECUTION_ALLOWED.get():
        raise PermissionError(
            "Remote and user-provided code execution is disabled for HTTP CLI requests."
        )
