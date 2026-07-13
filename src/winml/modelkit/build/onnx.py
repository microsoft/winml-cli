# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Build pipeline for pre-exported ONNX models.

Provides build_onnx_model() which runs the same stages as build_hf_model()
minus Load and Export. Intended for users who already have an ONNX file and
want to optimize, quantize, and compile it for WinML deployment.

Pipeline: [Optimize] -> [Analyze<->Optimize] -> [Quantize] -> [Compile] -> [Finalize]
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..compiler import compile_onnx
from ..onnx import copy_onnx_model
from ..quant import quantize_onnx
from .common import ensure_pre_quantized_stamped, run_optimize_analyze_loop
from .hf import BuildResult


if TYPE_CHECKING:
    from ..config import WinMLBuildConfig
    from ..utils.constants import EPNameOrAlias

logger = logging.getLogger(__name__)


def build_onnx_model(
    onnx_path: Path | str,
    *,
    config: WinMLBuildConfig,
    output_dir: Path | str,
    rebuild: bool = False,
    ep: EPNameOrAlias | None = None,
    device: str | None = None,
    cache_key: str | None = None,
    **kwargs: Any,
) -> BuildResult:
    """Build from a pre-exported ONNX model.

    Pipeline: [Optimize] -> [Analyze<->Optimize] -> [Quantize] -> [Compile] -> [Finalize]

    Same stages as build_hf_model minus Load and Export. The config should
    have ``export=None`` for ONNX builds.

    Args:
        onnx_path: Path to input ONNX model.
        config: Build configuration (export should be None for ONNX builds).
        output_dir: Directory for output artifacts. Created if missing.
        rebuild: Force rebuild even if output exists.
        ep: Target execution provider for the analyzer (e.g., ``"qnn"``).
        device: Target device for the analyzer (e.g., ``"NPU"``).
        cache_key: Optional prefix for artifact filenames, enabling multiple
            task/config variants to coexist in one directory. When set, all
            artifact files are prefixed (e.g., ``"{cache_key}_model.onnx"``).
        **kwargs: Additional options:
            - ``hack_max_optim_iterations`` (int, default 3): Max analyzer
              iterations. 0 disables analyzer.
            - ``allow_unsupported_nodes`` (bool, default False): If True, warn
              instead of raising when unsupported nodes persist after analysis.
            - ``use_external_data`` (bool, default True): Whether to use ONNX
              external data format.

    Returns:
        BuildResult with paths to artifacts and build metadata.

    Raises:
        FileNotFoundError: If onnx_path doesn't exist.
        ValueError: If onnx_path is not a file or config validation fails.
        RuntimeError: If a pipeline stage fails.
    """
    hack_max_optim_iterations: int = kwargs.pop("hack_max_optim_iterations", 3)
    allow_unsupported_nodes: bool = kwargs.pop("allow_unsupported_nodes", False)
    onnx_kwargs = {
        "use_external_data": kwargs.get("use_external_data", True),
    }

    onnx_path = Path(onnx_path)
    output_dir = Path(output_dir)

    # =========================================================================
    # [0] VALIDATE & SETUP
    # =========================================================================
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if not onnx_path.is_file():
        raise ValueError(f"ONNX path is not a file: {onnx_path}")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {output_dir}")

    try:
        config.validate()
    except ValueError as e:
        raise ValueError(f"Config validation failed before build: {e}") from e

    start_time = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Artifact naming — optionally prefixed when cache_key is set so that
    # multiple task/config variants can coexist in one directory.
    def _name(base: str) -> str:
        return f"{cache_key}_{base}" if cache_key else base

    from ..utils.manifest import MANIFEST_FILENAME

    # Define output paths
    optimized_path = output_dir / _name("optimized.onnx")
    quantized_path = output_dir / _name("quantized.onnx")
    compiled_path = output_dir / _name("compiled.onnx")
    final_path = output_dir / _name("model.onnx")
    config_path = output_dir / _name("winml_build_config.json")
    manifest_path = output_dir / _name(MANIFEST_FILENAME)
    analyze_result_path = output_dir / _name("analyze_result.json")

    # Check for existing artifact (skip build if present and not rebuilding)
    if final_path.exists() and not rebuild:
        logger.info("Existing artifact found: %s", final_path)
        return BuildResult(
            output_dir=output_dir,
            final_onnx_path=final_path,
            config_path=config_path,
            reused=True,
            elapsed=time.monotonic() - start_time,
        )

    # Rebuild: clean old ONNX artifacts to prevent stale files
    if rebuild:
        pattern = f"{cache_key}_*.onnx" if cache_key else "*.onnx"
        for old in output_dir.glob(pattern):
            old.unlink()
            logger.debug("Removed old artifact: %s", old.name)
        data_pattern = f"{cache_key}_*.onnx.data" if cache_key else "*.onnx.data"
        for old in output_dir.glob(data_pattern):
            old.unlink()
            logger.debug("Removed old external data sidecar: %s", old.name)

    stages_completed: list[str] = []
    stages_skipped: list[str] = []
    stage_timings: dict[str, float] = {}

    # Copy input ONNX to output dir as starting point
    current_path = output_dir / onnx_path.name
    if current_path.resolve() != onnx_path.resolve():
        copy_onnx_model(onnx_path, current_path)

    # =========================================================================
    # [1] OPTIMIZE + ANALYZE (or SKIP-BOTH for pre-quantized)
    # FIXME: Stages [1]-[4] (optimize, quantize, compile, finalize) are
    # duplicated between build_onnx_model() and build_hf_model(). Extract
    # into a shared run_build_stages() function in common.py.
    # =========================================================================
    # Single defensive detection. No-op when the CLI path (via
    # ``generate_onnx_build_config``) already stamped ``config.skip_optimize``.
    # Direct callers who hand-built a config trigger the one detection here.
    skip_optimize_kwarg: bool = kwargs.pop("skip_optimize", False)
    ensure_pre_quantized_stamped(config, current_path, force=skip_optimize_kwarg)
    is_pre_quantized = config.skip_optimize

    if is_pre_quantized:
        logger.info("Skipping optimize + quantize stages (config.skip_optimize=True)")
        stages_skipped.append("optimize")
        # Skip the ORT-based graph optimization (no kernel for QOperator
        # ops like ConvInteger on the host EP). The autoconf re-optim/
        # analyze loop is disabled too -- ``run_optimize_analyze_loop``
        # forces ``max_optim_iterations=0`` when ``skip_optimize=True``,
        # so ``_run_analyze_loop`` is not invoked. The model still flows
        # through later stages (quantize-skip + compile) for validation.
        current_path, _, analyze_iters, analyze_unsupported, analyze_details = (
            run_optimize_analyze_loop(
                model_path=current_path,
                optimized_path=optimized_path,
                config=config,
                ep=ep,
                device=device,
                skip_optimize=True,
                **onnx_kwargs,
            )
        )
    else:
        logger.info("Optimizing ONNX model...")
        current_path, opt_elapsed, analyze_iters, analyze_unsupported, analyze_details = (
            run_optimize_analyze_loop(
                model_path=current_path,
                optimized_path=optimized_path,
                config=config,
                ep=ep,
                device=device,
                max_optim_iterations=hack_max_optim_iterations,
                allow_unsupported_nodes=allow_unsupported_nodes,
                analyze_output_path=analyze_result_path,
                **onnx_kwargs,
            )
        )
        stage_timings["optimize"] = opt_elapsed
        stages_completed.append("optimize")
        logger.info("Optimize done (%.1fs) -> %s", opt_elapsed, optimized_path)

    # Persist config AFTER autoconf — includes discovered optimization flags
    config_path.write_text(json.dumps(config.to_dict(), indent=2))
    logger.debug("Config persisted: %s", config_path)

    # =========================================================================
    # [2] QUANTIZE (optional — config.quant=None means skip)
    # =========================================================================
    # No defensive ``is_quantized_onnx`` re-check here: when the model is
    # pre-quantized, ``ensure_pre_quantized_stamped`` has already set
    # ``config.quant = None`` at stage [1], so this branch naturally
    # falls through to the ``quant is None`` skip path.
    quant_result = None
    if is_pre_quantized:
        # Already handled above -- skip quantize for pre-quantized models
        if "quantize" not in stages_skipped:
            stages_skipped.append("quantize")
        logger.info("Quantize skipped (pre-quantized model)")
    elif config.quant is not None:
        logger.info("Quantizing model...")
        t0 = time.monotonic()
        quant_result = quantize_onnx(
            model_path=current_path,
            output_path=quantized_path,
            config=config.quant,
            **onnx_kwargs,
        )
        if not quant_result.success:
            errors = ", ".join(quant_result.errors) if quant_result.errors else "Unknown"
            raise RuntimeError(f"Quantization failed: {errors}")
        current_path = quantized_path
        stage_timings["quantize"] = time.monotonic() - t0
        stages_completed.append("quantize")
        logger.info("Quantize done (%.1fs) -> %s", stage_timings["quantize"], quantized_path)
    else:
        stages_skipped.append("quantize")
        logger.info("Quantize skipped (config.quant is None)")

    # =========================================================================
    # [3] COMPILE (optional — config.compile=None means skip)
    # =========================================================================
    if config.compile is not None:
        logger.info("Compiling model...")
        t0 = time.monotonic()
        compile_result = compile_onnx(
            model_path=current_path,
            output_path=compiled_path,
            config=config.compile,
        )
        if hasattr(compile_result, "success") and not compile_result.success:
            errors = ", ".join(compile_result.errors) if compile_result.errors else "Unknown"
            raise RuntimeError(f"Compilation failed: {errors}")
        if compile_result.output_path and Path(compile_result.output_path) != compiled_path:
            copy_onnx_model(compile_result.output_path, compiled_path)
        if compiled_path.exists():
            current_path = compiled_path
        stage_timings["compile"] = time.monotonic() - t0
        stages_completed.append("compile")
        logger.info("Compile done (%.1fs) -> %s", stage_timings["compile"], current_path)
    else:
        stages_skipped.append("compile")
        logger.info("Compile skipped (config.compile is None)")

    # =========================================================================
    # [4] FINALIZE — Copy last stage output as model.onnx
    # =========================================================================
    if current_path != final_path:
        copy_onnx_model(current_path, final_path)

    elapsed = time.monotonic() - start_time
    logger.info("Build complete in %.1fs -> %s", elapsed, final_path)

    # =========================================================================
    # [5] BUILD MANIFEST — Machine-readable build provenance
    # =========================================================================
    from ..utils import ManifestStage, WinMLManifest

    manifest_stages: list[ManifestStage] = []
    stage_filenames = {
        "optimize": optimized_path.name,
        "quantize": quantized_path.name,
        "compile": compiled_path.name,
    }
    for stage_name in ["optimize", "quantize", "compile"]:
        if stage_name in stages_completed:
            stage = ManifestStage(
                name=stage_name,
                status="completed",
                filename=stage_filenames[stage_name],
                elapsed_seconds=round(stage_timings.get(stage_name, 0), 3),
            )
            # Thread QuantizeResult metrics into manifest
            if stage_name == "quantize" and quant_result is not None:
                stage.nodes_quantized = quant_result.nodes_quantized
                stage.nodes_skipped = quant_result.nodes_skipped
                stage.calibration_time_seconds = round(quant_result.calibration_time_seconds, 3)
                stage.qdq_insertion_time_seconds = round(quant_result.qdq_insertion_time_seconds, 3)
            manifest_stages.append(stage)
        elif stage_name in stages_skipped:
            manifest_stages.append(ManifestStage(name=stage_name, status="skipped"))

    manifest = WinMLManifest(
        source="onnx",
        input_onnx=str(onnx_path),
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        elapsed_seconds=round(elapsed, 3),
        final_artifact=final_path.name,
        stages=manifest_stages,
        analyze_iterations=analyze_iters,
        analyze_unsupported_node_count=analyze_unsupported,
        analyze_details=analyze_details,
    )
    manifest.save(manifest_path)

    return BuildResult(
        output_dir=output_dir,
        final_onnx_path=final_path,
        config_path=config_path,
        stages_completed=stages_completed,
        stages_skipped=stages_skipped,
        stage_timings=stage_timings,
        elapsed=elapsed,
        manifest_path=manifest_path,
    )
