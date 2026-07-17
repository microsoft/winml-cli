# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Regression pin: NPU EP checkers live in a single module.

After the duplication consolidation, ``check_patterns.py`` re-imports the
``OpenVINONPUChecker`` / ``QNNNPUChecker`` classes and ``get_ep_checker``
from ``check_ops.py`` rather than redefining them. Deleting the re-export
would allow the byte-identical duplicates to reappear.
"""

from __future__ import annotations

from winml.modelkit.analyze.pattern import check_patterns
from winml.modelkit.analyze.runtime_checker import check_ops


def test_openvino_checker_is_shared() -> None:
    assert check_patterns.OpenVINONPUChecker is check_ops.OpenVINONPUChecker


def test_qnn_checker_is_shared() -> None:
    assert check_patterns.QNNNPUChecker is check_ops.QNNNPUChecker


def test_get_ep_checker_is_shared() -> None:
    assert check_patterns.get_ep_checker is check_ops.get_ep_checker
