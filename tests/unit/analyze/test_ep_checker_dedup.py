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
from winml.modelkit.analyze.runtime_checker.ep_checker import _RulesPrefilterProtocol


def test_rules_prefilter_protocol_has_no_default_method_implementation() -> None:
    """The rules-prefilter contract should be a pure Protocol member."""
    method_name = "build_skip_check_result_for_rules_all_nodes_compile_run_pass"
    method = _RulesPrefilterProtocol.__dict__[method_name]
    assert getattr(method, "__isabstractmethod__", False)


def test_openvino_checker_is_shared() -> None:
    assert check_patterns.OpenVINONPUChecker is check_ops.OpenVINONPUChecker


def test_qnn_checker_is_shared() -> None:
    assert check_patterns.QNNNPUChecker is check_ops.QNNNPUChecker


def test_get_ep_checker_is_shared() -> None:
    assert check_patterns.get_ep_checker is check_ops.get_ep_checker
