# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Stable node-key helpers for test code."""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import onnx

from winml.modelkit.pattern.utils import make_stable_node_key


def stable_test_node_keys(nodes: list[onnx.NodeProto]) -> list[str]:
    """Build stable keys with the same fallback policy as production code."""
    return [make_stable_node_key(node, idx) for idx, node in enumerate(nodes)]
