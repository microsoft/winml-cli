"""Shared build pipeline utilities.

Provides the optimize-analyze loop reused by both build_hf_model() and
build_onnx_model().
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..onnx import copy_onnx_model
from ..optim import optimize_onnx
from ..analyze import analyze_onnx


if TYPE_CHECKING:
    from ..config import WinMLBuildConfig

logger = logging.getLogger(__name__)


def run_optimize_analyze_loop(
    model_path: Path,
    optimized_path: Path,
    config: WinMLBuildConfig,
    *,
    ep: str | None = None,
    device: str | None = None,
    max_optim_iterations: int = 0,
    **onnx_kwargs: Any,
) -> tuple[Path, float, int, int, dict]:
    """Optimize an ONNX model, analyze, and optionally re-optimize via autoconf.

    Flow:
        1. Optimize with ``config.optim`` flags
        2. Analyze the result (lint + autoconf discovery)
        3. For up to ``max_optim_iterations``: if autoconf found new flags,
           re-optimize and re-analyze
        4. Wrap up: persist flags, check black nodes, build manifest details

    Args:
        model_path: Path to the input ONNX model.
        optimized_path: Path where the optimized model should be written.
        config: Build configuration. ``config.optim`` provides optimization
            flags and may be mutated to include discovered autoconf flags.
        ep: Target execution provider for the analyzer.
        device: Target device for the analyzer.
        max_optim_iterations: Maximum autoconf re-optimization rounds.
            0 means optimize+analyze only (no autoconf re-optimization).
        **onnx_kwargs: Additional ONNX-level kwargs.

    Returns:
        ``(optimized_path, elapsed, analyze_count, black_node_count, details)``

    Raises:
        RuntimeError: If black nodes persist after analysis.
    """
    t0 = time.monotonic()

    # 1. Optimize
    optimize_onnx(
        model=model_path,
        output=optimized_path,
        **onnx_kwargs,
        **config.optim,
    )

    # 2. Analyze
    analysis = analyze_onnx(optimized_path, ep=ep, device=device)
    analyze_count = 1
    discovered_optim: dict[str, bool] = {}

    # 3. Autoconf re-optimization loop
    with tempfile.TemporaryDirectory() as tmp:
        iter_model = Path(tmp) / "iter.onnx"
        copied = False

        for _iteration in range(max_optim_iterations):
            if not analysis.autoconf:
                break

            logger.info(
                "Autoconf iteration %d: discovered %s",
                _iteration + 1, analysis.optimization_config.to_dict(),
            )

            if not copied:
                copy_onnx_model(optimized_path, iter_model)
                copied = True

            optimize_onnx(
                model=iter_model, output=iter_model,
                **onnx_kwargs,
                **analysis.optimization_config,
            )
            discovered_optim.update(analysis.optimization_config)

            analysis = analyze_onnx(iter_model, ep=ep, device=device)
            analyze_count += 1

        if copied:
            copy_onnx_model(iter_model, optimized_path)

    # 4. Wrap up
    if discovered_optim:
        config.optim.update(discovered_optim)
        logger.info("  [autoconf] final config: %s", discovered_optim)

    if analysis.autoconf:
        logger.warning(
            "Analysis still has autoconf suggestions: %s",
            analysis.optimization_config.to_dict(),
        )

    if analysis.has_errors:
        raise RuntimeError(
            f"Black nodes persist after {analyze_count} analyze "
            f"pass(es): {analysis.lint.error_patterns}"
        )

    details = {
        "lint": {
            "errors": analysis.lint.errors,
            "warnings": analysis.lint.warnings,
            "passed": analysis.lint.passed,
            "error_patterns": analysis.lint.error_patterns,
            "warning_patterns": analysis.lint.warning_patterns,
        },
        "autoconf": discovered_optim or {},
    }

    elapsed = time.monotonic() - t0
    return optimized_path, elapsed, analyze_count, analysis.lint.errors, details
