# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared helpers for timing log gating and formatting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..._env import env_flag_enabled


if TYPE_CHECKING:
    import logging
    from collections.abc import Callable


_TIMING_LOG_ENABLED = env_flag_enabled("WINMLCLI_TIMING_LOG")


def make_timing_logger(logger: logging.Logger) -> Callable[..., None]:
    """Create a per-module timing logger bound to a concrete logger.

    Returned callable signature: ``(event: str, **fields: object) -> None``.
    """

    def _log_timing(event: str, **fields: object) -> None:
        if not _TIMING_LOG_ENABLED:
            return

        parts = [f"{k}={v}" for k, v in fields.items() if v is not None]
        if parts:
            logger.info("[timing] %s %s", event, " ".join(parts))
        else:
            logger.info("[timing] %s", event)

    return _log_timing
