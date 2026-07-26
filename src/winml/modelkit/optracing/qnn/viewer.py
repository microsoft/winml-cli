# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Deprecated compatibility shim for the legacy QNN profile viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...session.monitor.qnn.viewer import (
    find_qnn_sdk as _find_qnn_sdk,
)
from ...session.monitor.qnn.viewer import (
    run_basic_viewer as _run_basic_viewer,
)
from ...session.monitor.qnn.viewer import (
    run_qhas_viewer as _run_qhas_viewer,
)
from .._compat import warn_deprecated


_SYMBOLS = {
    "find_qnn_sdk": _find_qnn_sdk,
    "run_basic_viewer": _run_basic_viewer,
    "run_qhas_viewer": _run_qhas_viewer,
}


def __getattr__(name: str) -> Any:
    value = _SYMBOLS.get(name)
    if value is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    warn_deprecated(
        f"qnn.viewer.{name}",
        f"winml.modelkit.session.monitor.qnn.viewer.{name}",
        stacklevel=2,
    )
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    find_qnn_sdk = _find_qnn_sdk
    run_basic_viewer = _run_basic_viewer
    run_qhas_viewer = _run_qhas_viewer


__all__ = ["find_qnn_sdk", "run_basic_viewer", "run_qhas_viewer"]
