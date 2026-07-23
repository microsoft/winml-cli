# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared build pipeline utilities.

Provides the optimize-analyze loop and the stage runner
(optimize -> quantize -> compile -> finalize) reused by both
:func:`build_hf_model` and :func:`build_onnx_model`.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..analyze import analyze_onnx
from ..compiler import compile_onnx
from ..onnx import copy_onnx_model, is_quantized_onnx
from ..optim import optimize_onnx
from ..quant import quantize_onnx


if TYPE_CHECKING:
    from ..config import WinMLBuildConfig

logger = logging.getLogger(__name__)


@dataclass
class StagesResult:
    """Outcome of :func:`run_build_stages`.

    Bundles the fields both build entry points need to persist into their
    build manifest and return via :class:`BuildResult`.
    """

    current_path: Path
    is_pre_quantized: bool
    stages_completed: list[str] = field(default_factory=list)
    stages_skipped: list[str] = field(default_factory=list)
    stage_timings: dict[str, float] = field(default_factory=dict)
    analyze_iterations: int = 0
    analyze_unsupported_nodes: int = 0
    analyze_details: dict = field(default_factory=dict)
    quant_result: Any = None


def run_build_stages(
    *,
    current_path: Path,
    optimized_path: Path,
    quantized_path: Path,
    compiled_path: Path,
    final_path: Path,
    config: WinMLBuildConfig,
    config_path: Path,
    ep: str | None = None,
    device: str | None = None,
    hack_max_optim_iterations: int = 3,
    skip_optimize: bool = False,
    onnx_kwargs: dict[str, Any] | None = None,
) -> StagesResult:
    """Run the shared build stages: optimize -> quantize -> compile -> finalize.

    Extracted from :func:`build_hf_model` and :func:`build_onnx_model` per
    the FIXMEs in ``build/hf.py`` and ``build/onnx.py``. Behaviour is
    identical to the previous inline blocks.

    Args:
        current_path: Path to the ONNX model at the start of the stages
            (post-export for HF, post-copy for ONNX). Mutated in place as
            the pipeline advances.
        optimized_path: Destination path for the optimize stage.
        quantized_path: Destination path for the quantize stage.
        compiled_path: Destination path for the compile stage.
        final_path: Destination for the finalize stage (``model.onnx``).
        config: Full build config; ``config.optim`` / ``config.quant`` /
            ``config.compile`` gate the individual stages.
        config_path: Where the resolved config is persisted after
            optimize (autoconf may have expanded ``config.optim``).
        ep, device: Passed through to :func:`run_optimize_analyze_loop`.
        hack_max_optim_iterations: Max analyzer autoconf rounds.
        skip_optimize: When True, bypasses optimize and only runs analyze
            (used for pre-quantized inputs).
        onnx_kwargs: ONNX-level kwargs forwarded to optimize/quantize.
    """
    onnx_kwargs = onnx_kwargs or {}
    result = StagesResult(
        current_path=current_path,
        is_pre_quantized=is_quantized_onnx(current_path) or skip_optimize,
    )

    # =========================================================================
    # OPTIMIZE + ANALYZE (or ANALYZE-ONLY for pre-quantized)
    # =========================================================================
    if result.is_pre_quantized:
        logger.info(
            "Pre-quantized model detected (QDQ nodes present). "
            "Skipping optimize + quantize, running analyze-only."
        )
        result.stages_skipped.append("optimize")
        (
            result.current_path,
            _,
            result.analyze_iterations,
            result.analyze_unsupported_nodes,
            result.analyze_details,
        ) = run_optimize_analyze_loop(
            model_path=result.current_path,
            optimized_path=optimized_path,
            config=config,
            ep=ep,
            device=device,
            **onnx_kwargs,
        )
    else:
        logger.info("Optimizing ONNX model...")
        (
            result.current_path,
            opt_elapsed,
            result.analyze_iterations,
            result.analyze_unsupported_nodes,
            result.analyze_details,
        ) = run_optimize_analyze_loop(
            model_path=result.current_path,
            optimized_path=optimized_path,
            config=config,
            ep=ep,
            device=device,
            max_optim_iterations=hack_max_optim_iterations,
            **onnx_kwargs,
        )
        result.stage_timings["optimize"] = opt_elapsed
        result.stages_completed.append("optimize")
        logger.info("Optimize done (%.1fs) -> %s", opt_elapsed, optimized_path)

    # Persist config AFTER autoconf — includes discovered optimization flags
    config_path.write_text(json.dumps(config.to_dict(), indent=2))
    logger.debug("Config persisted: %s", config_path)

    # =========================================================================
    # QUANTIZE (optional — config.quant=None means skip)
    # =========================================================================
    if result.is_pre_quantized:
        if "quantize" not in result.stages_skipped:
            result.stages_skipped.append("quantize")
        logger.info("Quantize skipped (pre-quantized model)")
    elif config.quant is not None:
        # Defensive fallback: catches the edge case where a direct caller
        # provides config.quant != None but the model already has QDQ nodes.
        if is_quantized_onnx(result.current_path):
            logger.warning(
                "Model already contains QDQ nodes, skipping quantization. "
                "Set config.quant=None to silence this warning."
            )
            result.stages_skipped.append("quantize")
        else:
            logger.info("Quantizing model...")
            t0 = time.monotonic()
            result.quant_result = quantize_onnx(
                model_path=result.current_path,
                output_path=quantized_path,
                config=config.quant,
                **onnx_kwargs,
            )
            if not result.quant_result.success:
                errors = (
                    ", ".join(result.quant_result.errors)
                    if result.quant_result.errors
                    else "Unknown"
                )
                raise RuntimeError(f"Quantization failed: {errors}")
            result.current_path = quantized_path
            result.stage_timings["quantize"] = time.monotonic() - t0
            result.stages_completed.append("quantize")
            logger.info(
                "Quantize done (%.1fs) -> %s",
                result.stage_timings["quantize"],
                quantized_path,
            )
    else:
        result.stages_skipped.append("quantize")
        logger.info("Quantize skipped (config.quant is None)")

    # =========================================================================
    # COMPILE (optional — config.compile=None means skip)
    # =========================================================================
    if config.compile is not None:
        logger.info("Compiling model...")
        t0 = time.monotonic()
        compile_result = compile_onnx(
            model_path=result.current_path,
            output_path=compiled_path,
            config=config.compile,
        )
        if hasattr(compile_result, "success") and not compile_result.success:
            errors = ", ".join(compile_result.errors) if compile_result.errors else "Unknown"
            raise RuntimeError(f"Compilation failed: {errors}")
        if compile_result.output_path and Path(compile_result.output_path) != compiled_path:
            copy_onnx_model(compile_result.output_path, compiled_path)
        if compiled_path.exists():
            result.current_path = compiled_path
        result.stage_timings["compile"] = time.monotonic() - t0
        result.stages_completed.append("compile")
        logger.info(
            "Compile done (%.1fs) -> %s", result.stage_timings["compile"], result.current_path
        )
    else:
        result.stages_skipped.append("compile")
        logger.info("Compile skipped (config.compile is None)")

    # =========================================================================
    # FINALIZE — Copy last stage output as model.onnx
    # =========================================================================
    if result.current_path != final_path:
        copy_onnx_model(result.current_path, final_path)
    result.current_path = final_path

    return result


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
    ep: str | None = None,
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
    ep: str | None,
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
