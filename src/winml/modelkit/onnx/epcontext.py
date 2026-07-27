# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Helpers for EPContext ONNX metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from onnx import helper

from .persistence import load_onnx


@dataclass(frozen=True)
class EPContextPartition:
    """EPContext partition metadata needed to bind QNN sidecars."""

    partition_name: str | None
    main_context: int | None
    main_context_present: bool


def _string_attr(value: Any) -> str | None:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return None
    if isinstance(value, str):
        return value
    return None


def _safe_partition_name(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.name != value or path.is_absolute() or path.drive or value in (".", ".."):
        return None
    return value


def _int_attr(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    return None


def epcontext_partitions(model_path: Path) -> list[EPContextPartition]:
    """Return EPContext partition metadata from standard node attributes."""
    model = load_onnx(model_path, load_weights=False, validate=False)
    partitions: list[EPContextPartition] = []
    for node in model.graph.node:
        if node.op_type != "EPContext":
            continue
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        partition_name = _safe_partition_name(_string_attr(attrs.get("partition_name")))
        partitions.append(
            EPContextPartition(
                partition_name=partition_name,
                main_context=_int_attr(attrs.get("main_context")),
                main_context_present="main_context" in attrs,
            )
        )
    return partitions


def select_main_epcontext_partition_name(model_path: Path) -> str | None:
    """Select the unique partition name that represents the main EPContext graph."""
    partitions = epcontext_partitions(model_path)
    if any(p.partition_name is None for p in partitions):
        return None
    safe_partitions = [p for p in partitions if p.partition_name is not None]
    main_partitions = [
        p.partition_name
        for p in safe_partitions
        if p.main_context == 1 and p.partition_name is not None
    ]
    if len(main_partitions) == 1:
        return main_partitions[0]
    if len(main_partitions) > 1:
        return None
    if len(safe_partitions) == 1 and not safe_partitions[0].main_context_present:
        return safe_partitions[0].partition_name
    return None
