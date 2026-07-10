# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
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

from ..analyze import analyze_onnx
from ..onnx import copy_onnx_model, is_quantized_onnx
from ..optim import optimize_onnx


if TYPE_CHECKING:
    from ..config import WinMLBuildConfig
    from ..utils.constants import EPNameOrAlias

logger = logging.getLogger(__name__)


def ensure_pre_quantized_stamped(
    config: WinMLBuildConfig, onnx_path: Path, *, force: bool = False
) -> None:
    """Stamp ``config.skip_optimize`` (and clear ``config.quant``) once.

    Sets ``config.skip_optimize = True`` and clears ``config.quant`` if the
    input ONNX is already quantized.

    This is the **single defensive detection point** for the library entry
    points (``build_onnx_model``, ``build_hf_model``). When
    ``config.skip_optimize`` is already True (i.e. the unified CLI path
    via :func:`generate_onnx_build_config` already stamped the config), it
    still enforces ``config.quant = None`` without re-running
    ``is_quantized_onnx()``.

    Args:
        config: Build config to stamp in place.
        onnx_path: Path to the ONNX file under consideration.
        force: When True, stamp unconditionally without running
            ``is_quantized_onnx`` (used to honor the legacy
            ``skip_optimize=True`` kwarg from direct callers).
    """
    if config.skip_optimize:
        config.quant = None
        return
    if force:
        config.skip_optimize = True
        config.quant = None
        return

    if is_quantized_onnx(onnx_path):
        config.skip_optimize = True
        config.quant = None
        logger.info(
            "Pre-quantized model detected (QDQ or QOperator nodes present). "
            "Skipping optimize + quantize stages."
        )


def run_optimize_analyze_loop(
    model_path: Path,
    optimized_path: Path,
    config: WinMLBuildConfig,
    *,
    ep: EPNameOrAlias | None = None,
    device: str | None = None,
    max_optim_iterations: int = 0,
    allow_unsupported_nodes: bool = False,
    skip_optimize: bool = False,
    on_ep_start: Any = None,
    on_node_result: Any = None,
    on_iteration_start: Any = None,
    on_patterns_discovered: Any = None,
    on_reoptimize: Any = None,
    analyze_output_path: Path | None = None,
    **onnx_kwargs: Any,
) -> tuple[Path, float, int, int, dict]:
    """Optimize an ONNX model, analyze, and optionally re-optimize via autoconf.

    Flow:
        1. Optimize with ``config.optim`` flags (skipped if ``skip_optimize=True``)
        2. Analyze the result (lint + autoconf discovery)
        3. For up to ``max_optim_iterations``: if autoconf found new flags,
           re-optimize and re-analyze
        4. Wrap up: persist flags, check unsupported nodes, build manifest details

    Args:
        model_path: Path to the input ONNX model.
        optimized_path: Path where the optimized model should be written.
        config: Build configuration. ``config.optim`` provides optimization
            flags and may be mutated to include discovered autoconf flags.
        ep: Target execution provider for the analyzer.
        device: Target device for the analyzer.
        max_optim_iterations: Maximum autoconf re-optimization rounds.
            0 disables the autoconf re-optimize/analyze loop entirely
            (i.e. ``_run_analyze_loop`` is not invoked), in which case
            this function performs the initial ``optimize_onnx`` pass
            only (or, when ``skip_optimize=True``, just copies the input
            to ``optimized_path``).
        allow_unsupported_nodes: If True, log a warning instead of raising when
            unsupported nodes persist after analysis, letting the build proceed
            (the EP may still run them, e.g. via CPU fallback).
        analyze_output_path: Optional path to write the full analysis result as
            JSON. Written after every analyze pass; each pass overwrites the
            previous one so the file always reflects the most recent analysis.
        skip_optimize: When True, skip the initial ``optimize_onnx`` call and
            just copy the input model to ``optimized_path``. Used for
            pre-quantized models (QDQ or QOperator format) where ORT-based
            graph optimization would fail because the runtime lacks kernels
            for ops like ``ConvInteger`` on the host EP.
        **onnx_kwargs: Additional ONNX-level kwargs.

    Returns:
        ``(optimized_path, elapsed, analyze_count, unsupported_node_count, details)``

    Raises:
        RuntimeError: If unsupported nodes persist after analysis.
    """
    # Respect auto=False: flags are pre-configured, skip autoconf
    if not config.auto:
        max_optim_iterations = 0

    # Enforce the skip_optimize invariant: autoconf re-optimize would
    # crash on pre-quantized models for the same reason the initial
    # optimize was skipped (ORT lacks kernels for the integer ops on the
    # host EP). Drop iterations to 0 so callers can pass any value safely.
    if skip_optimize:
        max_optim_iterations = 0

    t0 = time.monotonic()

    # 1. Optimize (or skip for pre-quantized models)
    if skip_optimize:
        # Pre-quantized models (QOperator format with ConvInteger /
        # MatMulInteger) cannot pass through ORT graph optimization on
        # hosts that lack kernels for those integer ops. Simply forward
        # the input as the "optimized" artifact.
        if model_path.resolve() != optimized_path.resolve():
            copy_onnx_model(model_path, optimized_path)
    else:
        optimize_onnx(
            model=model_path,
            output=optimized_path,
            **onnx_kwargs,
            **config.optim,
        )
    current_path = optimized_path

    # Autoconf: analyze model, discover missing optimizations, re-optimize
    if max_optim_iterations > 0:
        analyze_iterations, analyze_black_nodes, analyze_details = _run_analyze_loop(
            optimized_path=optimized_path,
            ep=ep,
            device=device,
            max_optim_iterations=max_optim_iterations,
            allow_unsupported_nodes=allow_unsupported_nodes,
            config=config,
            on_ep_start=on_ep_start,
            on_node_result=on_node_result,
            on_iteration_start=on_iteration_start,
            on_patterns_discovered=on_patterns_discovered,
            on_reoptimize=on_reoptimize,
            analyze_output_path=analyze_output_path,
            **onnx_kwargs,
        )
    else:
        analyze_iterations, analyze_black_nodes, analyze_details = 0, 0, {}

    elapsed = time.monotonic() - t0

    return current_path, elapsed, analyze_iterations, analyze_black_nodes, analyze_details


def _run_analyze_loop(
    *,
    optimized_path: Path,
    ep: EPNameOrAlias | None,
    device: str | None,
    max_optim_iterations: int,
    config: WinMLBuildConfig,
    allow_unsupported_nodes: bool = False,
    on_ep_start: Any = None,
    on_node_result: Any = None,
    on_iteration_start: Any = None,
    on_patterns_discovered: Any = None,
    on_reoptimize: Any = None,
    analyze_output_path: Path | None = None,
    **kwargs: Any,
) -> tuple[int, int, dict]:
    """Run iterative analyzer autoconf loop in a temp folder.

    Each iteration applies ONLY the autoconf flags (not merged with original).
    A separate dict accumulates all discovered flags for persistence.
    """
    analyze_iterations = 0
    analyze_black_nodes = 0
    discovered_optim: dict[str, bool] = {}
    analysis = None
    _not_converged = False

    # 3. Autoconf re-optimization loop
    with tempfile.TemporaryDirectory() as tmp:
        iter_model = Path(tmp) / "iter.onnx"
        copy_onnx_model(optimized_path, iter_model)

        for _iteration in range(max_optim_iterations):
            # Notify: iteration starting
            if on_iteration_start is not None:
                on_iteration_start(
                    _iteration + 1,
                    max_optim_iterations,
                )

            analysis = analyze_onnx(
                iter_model,
                ep=ep,
                device=device,
                run_unknown_op=False,
                on_ep_start=on_ep_start,
                on_node_result=on_node_result,
                output_path=analyze_output_path,
            )
            analyze_iterations += 1

            optim_config = analysis.optimization_config
            if not optim_config:
                break

            logger.info(
                "Autoconf iteration %d: discovered %s",
                _iteration + 1,
                optim_config.to_dict(),
            )

            # Notify: patterns discovered
            if on_patterns_discovered is not None:
                on_patterns_discovered(optim_config)

            # Notify: re-optimizing with discovered flags
            if on_reoptimize is not None:
                on_reoptimize(optim_config)

            # Re-optimize with ONLY the autoconf flags (not merged with original)
            optimize_onnx(
                model=iter_model,
                output=iter_model,
                **kwargs,
                **optim_config,
            )
            discovered_optim.update(optim_config)
        else:
            logger.warning(
                "Autoconf did not converge after %d iteration(s)",
                max_optim_iterations,
            )
            _not_converged = True

        # Always analyze final state (validates after last optimize).
        # Pass a no-op on_node_result to suppress tqdm (which would
        # break the Rich Live display). No on_ep_start to avoid
        # duplicate EP bars.
        analysis = analyze_onnx(
            iter_model,
            ep=ep,
            device=device,
            run_unknown_op=False,
            on_node_result=lambda _: None,
            output_path=analyze_output_path,
        )

        copy_onnx_model(iter_model, optimized_path)

    # 4. Wrap up
    if discovered_optim:
        config.optim.update(discovered_optim)
        logger.info("  [autoconf] final config: %s", discovered_optim)

    # analysis is None only when max_optim_iterations == 0 (the loop body never
    # ran, so analyze_onnx was never called).
    final_optim_config = analysis.optimization_config if analysis else None
    if final_optim_config:
        logger.warning(
            "Analysis still has autoconf suggestions: %s",
            final_optim_config.to_dict(),
        )

    if analysis is not None and analysis.has_errors:
        message = (
            f"Unsupported nodes persist after {analyze_iterations} analyze "
            f"pass(es): {analysis.lint.error_patterns}"
        )
        if allow_unsupported_nodes:
            logger.warning(
                "%s. Continuing anyway (allow_unsupported_nodes=True); the EP may "
                "fall back to another device for these nodes.",
                message,
            )
        else:
            raise RuntimeError(message)

    analyze_black_nodes = analysis.lint.errors if analysis else 0

    # Build details for manifest
    details: dict = {}
    if analysis:
        details = {
            "lint": {
                "errors": analysis.lint.errors,
                "warnings": analysis.lint.warnings,
                "passed": analysis.lint.passed,
                "error_patterns": analysis.lint.error_patterns,
                "warning_patterns": analysis.lint.warning_patterns,
            },
            "autoconf": discovered_optim or {},
            "autoconf_not_converged": _not_converged,
        }

    return analyze_iterations, analyze_black_nodes, details
