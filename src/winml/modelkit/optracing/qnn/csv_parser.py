# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Deprecated compatibility shim for the legacy QNN CSV parser."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...session.monitor.qnn import parse_qnn_profiling_csv as _parse_current_csv
from .._compat import warn_deprecated


if TYPE_CHECKING:
    from pathlib import Path


def _parse_qnn_profiling_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    parsed = _parse_current_csv(csv_path)
    return [
        {
            "metadata": sample["metadata"],
            "samples": [
                {
                    "name": operator["op_path"],
                    "op_id": operator["op_id"],
                    "cycles": operator["cycles"],
                }
                for operator in sample["samples"]
            ],
        }
        for sample in parsed["samples"]
    ]


def __getattr__(name: str) -> Any:
    if name != "parse_qnn_profiling_csv":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    warn_deprecated(
        "qnn.csv_parser.parse_qnn_profiling_csv",
        "winml.modelkit.session.monitor.qnn.parse_qnn_profiling_csv",
        stacklevel=2,
    )
    return _parse_qnn_profiling_csv


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:
    parse_qnn_profiling_csv = _parse_qnn_profiling_csv


__all__ = ["parse_qnn_profiling_csv"]
