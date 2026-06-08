# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for build.common — the optimize/analyze loop.

Focus: the ``allow_unsupported_nodes`` gate around the "unsupported nodes
persist" RuntimeError. Mock-based, no real optimize/analyze.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from winml.modelkit.build.common import run_optimize_analyze_loop
from winml.modelkit.config import WinMLBuildConfig


def _config() -> WinMLBuildConfig:
    """Minimal config with autoconf enabled and no optim flags."""
    config = WinMLBuildConfig.from_dict(
        {
            "loader": {"task": "depth-estimation"},
            "export": {"opset_version": 17},
            "optim": {},
            "quant": None,
            "compile": {"execution_provider": "openvino"},
        }
    )
    config.auto = True
    return config


def _errored_analysis():
    """AnalyzeResult that reports unsupported nodes and no autoconf opportunities."""
    from winml.modelkit.analyze import AnalyzeResult, LintResult
    from winml.modelkit.optim import WinMLOptimizationConfig

    opt = WinMLOptimizationConfig()  # empty -> autoconf converges immediately
    lint = LintResult(
        errors=1,
        warnings=0,
        info=0,
        passed=False,
        error_patterns=["Resize"],
        warning_patterns=[],
        information=[],
        optimization_config=opt,
    )
    return AnalyzeResult(lint=lint, optimization_config=opt)


def _patched_loop(tmp_path: Path):
    """Patch the loop's stage functions to avoid real optimize/analyze/copy."""
    return (
        patch(
            "winml.modelkit.build.common.optimize_onnx",
            side_effect=lambda **kw: Path(kw["output"]).write_text("mock"),
        ),
        patch(
            "winml.modelkit.build.common.analyze_onnx",
            return_value=_errored_analysis(),
        ),
        patch(
            "winml.modelkit.build.common.copy_onnx_model",
            side_effect=lambda src, dst: Path(dst).write_text("mock"),
        ),
    )


class TestAllowUnsupportedNodesGate:
    """The unsupported-nodes RuntimeError is gated by allow_unsupported_nodes."""

    def test_raises_by_default(self, tmp_path: Path) -> None:
        model = tmp_path / "in.onnx"
        model.write_text("mock")
        optimized = tmp_path / "out.onnx"
        p_opt, p_analyze, p_copy = _patched_loop(tmp_path)
        expect_raise = pytest.raises(RuntimeError, match="Unsupported nodes persist")

        with p_opt, p_analyze, p_copy, expect_raise:
            run_optimize_analyze_loop(
                model_path=model,
                optimized_path=optimized,
                config=_config(),
                ep="openvino",
                device="gpu",
                max_optim_iterations=1,
            )

    def test_warns_instead_of_raising_when_allowed(self, tmp_path: Path, caplog) -> None:
        model = tmp_path / "in.onnx"
        model.write_text("mock")
        optimized = tmp_path / "out.onnx"
        p_opt, p_analyze, p_copy = _patched_loop(tmp_path)

        with p_opt, p_analyze, p_copy, caplog.at_level("WARNING"):
            result = run_optimize_analyze_loop(
                model_path=model,
                optimized_path=optimized,
                config=_config(),
                ep="openvino",
                device="gpu",
                max_optim_iterations=1,
                allow_unsupported_nodes=True,
            )

        # No raise; loop returns its 5-tuple and logs a warning.
        assert result[0] == optimized
        assert "Unsupported nodes persist" in caplog.text
