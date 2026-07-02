# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Build API for HuggingFace models.

Core pipeline: Load -> Export -> Optimize -> [Analyze] -> [Quantize] -> [Compile]

This module is the PRIMARY owner of the build pipeline. CLI and
WinMLAutoModel.from_pretrained() are consumers of this API.

Design Principles:
    1. CONFIG-DRIVEN: All pipeline behavior from WinMLBuildConfig
    2. NO HARDCODED LOGIC: Shapes, tasks, model types all from config
    3. EXPLICIT OUTPUT: User controls where artifacts go
    4. INTERMEDIATE PRESERVATION: All stage outputs kept for debugging
    5. PORTABLE-FIRST: Produce device-agnostic ONNX before compile
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..compiler import compile_onnx
from ..export import export_onnx
from ..onnx import copy_onnx_model, is_quantized_onnx
from ..quant import quantize_onnx
from .common import run_optimize_analyze_loop


if TYPE_CHECKING:
    import torch.nn as nn

    from ..config import WinMLBuildConfig
    from ..utils.constants import EPNameOrAlias

logger = logging.getLogger(__name__)


# =============================================================================
# BUILD RESULT
# =============================================================================


@dataclass
class BuildResult:
    """Result of a build pipeline execution.

    Attributes:
        output_dir: Directory containing all artifacts.
        final_onnx_path: Path to final ONNX model (output_dir/model.onnx).
        config_path: Path to persisted config (output_dir/winml_build_config.json).
        stages_completed: Names of stages that ran (e.g., ["export", "optimize"]).
        stages_skipped: Names of stages that were skipped.
        stage_timings: Per-stage elapsed time in seconds.
        elapsed: Total build time in seconds.
        reused: True if existing artifact was found and no build ran.
    """

    output_dir: Path
    final_onnx_path: Path
    config_path: Path
    stages_completed: list[str] = field(default_factory=list)
    stages_skipped: list[str] = field(default_factory=list)
    stage_timings: dict[str, float] = field(default_factory=dict)
    elapsed: float = 0.0
    reused: bool = False
    manifest_path: Path | None = None


# =============================================================================
# BUILD API
# =============================================================================


def build_hf_model(
    config: WinMLBuildConfig,
    output_dir: Path,
    *,
    model_id: str | None = None,
    pytorch_model: nn.Module | None = None,
    rebuild: bool = False,
    trust_remote_code: bool = False,
    random_init: bool = False,
    cache_key: str | None = None,
    ep: EPNameOrAlias | None = None,
    device: str | None = None,
    model_type: str | None = None,
    **kwargs: Any,
) -> BuildResult:
    """Build an ONNX model from a HuggingFace model architecture.

    Pipeline: [Load] -> Export -> Optimize -> [Analyze] -> [Quantize] -> [Compile]

    The Analyze stage runs an iterative autoconf loop: after optimization,
    the static analyzer detects missed fusion opportunities and feeds them
    back for re-optimization until convergence.

    Args:
        config: Complete build configuration (from ``WinMLBuildConfig.from_dict()``).
            Must contain loader, export, optim sections. quant and compile
            are optional (``None`` = skip stage).
        output_dir: Directory for all build artifacts. Created if missing.
        model_id: HuggingFace model ID for pretrained weights.
            If ``None`` and ``pytorch_model`` is ``None``, instantiates model
            with random/init weights using ``config.loader.model_type``.
        pytorch_model: Pre-loaded PyTorch model. If provided, model_id is
            only used for labeling (not loading).
        rebuild: If True, overwrite existing artifacts and re-run pipeline.
        trust_remote_code: Whether to trust remote code when loading HF models.
        cache_key: Optional prefix for artifact filenames.
        ep: Target execution provider for the analyzer (e.g., ``"qnn"``).
            If ``None``, analyzer runs without EP filter (all-EP aggregation).
        device: Target device for the analyzer (e.g., ``"NPU"``).
            If ``None``, analyzer runs without device filter.
        **kwargs: Additional options extracted by key:
            - ``hack_max_optim_iterations`` (int, default 3): TEMPORARY HACK —
              Max analyzer iterations. 0 disables analyzer.
              TODO: Move to global env / build config.
            - ``allow_unsupported_nodes`` (bool, default False): If True, warn
              instead of raising when unsupported nodes persist after analysis.
            - ``use_external_data`` (bool, default True): Whether to use ONNX
              external data format. Default True for large model compatibility.
              TODO: Move to global env / build config.

    Returns:
        BuildResult with paths to artifacts and build metadata.

    Raises:
        ValueError: If config is invalid or missing required fields.
        RuntimeError: If a pipeline stage fails, or if unsupported nodes persist
            after analyzer convergence.
    """
    # TODO: Move hack_max_optim_iterations to global env config
    hack_max_optim_iterations: int = kwargs.pop("hack_max_optim_iterations", 3)
    allow_unsupported_nodes: bool = kwargs.pop("allow_unsupported_nodes", False)

    # ONNX-level kwargs forwarded to export, optimize, quantize stages
    onnx_kwargs = {
        "use_external_data": kwargs.get("use_external_data", True),
    }

    start_time = time.monotonic()

    # =========================================================================
    # [0] VALIDATE & SETUP
    # =========================================================================
    if model_id is not None and not model_id.strip():
        raise ValueError("model_id cannot be an empty string")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Output path exists but is not a directory: {output_dir}")

    try:
        config.validate()
    except ValueError as e:
        raise ValueError(f"Config validation failed before build: {e}") from e

    output_dir.mkdir(parents=True, exist_ok=True)

    # Artifact naming — optionally prefixed when cache_key is set so that
    # multiple task/config variants can coexist in one directory.
    def _name(base: str) -> str:
        return f"{cache_key}_{base}" if cache_key else base

    export_path = output_dir / _name("export.onnx")
    optimized_path = output_dir / _name("optimized.onnx")
    quantized_path = output_dir / _name("quantized.onnx")
    compiled_path = output_dir / _name("compiled.onnx")
    final_path = output_dir / _name("model.onnx")
    config_path = output_dir / _name("winml_build_config.json")
    manifest_path = output_dir / _name("build_manifest.json")
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

    # Rebuild: clean old ONNX artifacts to prevent stale files from skipped stages
    if rebuild:
        pattern = f"{cache_key}_*.onnx" if cache_key else "*.onnx"
        for old in output_dir.glob(pattern):
            old.unlink()
            logger.debug("Removed old artifact: %s", old.name)

    stages_completed: list[str] = []
    stages_skipped: list[str] = []
    stage_timings: dict[str, float] = {}
    model_label = model_id or "random-init"

    # =========================================================================
    # [1] LOAD — Get PyTorch model (skip if pre-loaded)
    # =========================================================================
    task = config.loader.task
    logger.info("Building model: %s (task=%s)", model_label, task)

    if pytorch_model is None:
        pytorch_model = _load_model(
            config,
            model_id,
            trust_remote_code,
            random_init=random_init,
            model_type=model_type,
        )

    # =========================================================================
    # [2] EXPORT — PyTorch -> ONNX
    # =========================================================================
    logger.info("Exporting to ONNX...")
    t0 = time.monotonic()
    # config.export is None only for the ONNX build path (build_onnx_model);
    # this is the HF path so the field must be populated.
    assert config.export is not None, "build_hf_model requires config.export"
    export_onnx(
        model=pytorch_model,
        output_path=export_path,
        export_config=config.export,
        model_id=model_label,
        task=task,
        verbose=False,
        **onnx_kwargs,
    )
    current_path = export_path
    stage_timings["export"] = time.monotonic() - t0
    stages_completed.append("export")
    logger.info("Export done (%.1fs) -> %s", stage_timings["export"], export_path)

    # =========================================================================
    # [3] OPTIMIZE — ONNX graph optimization + autoconf loop
    # FIXME: Stages [3]-[6] (optimize, quantize, compile, finalize) are
    # duplicated between build_hf_model() and build_onnx_model(). Extract
    # into a shared run_build_stages() function in common.py.
    # =========================================================================
    skip_optimize: bool = kwargs.pop("skip_optimize", False)
    # Defensive fallback: when called through the unified pipeline,
    # generate_*_build_config() already detects QDQ models and sets
    # config.quant=None. This is_quantized_onnx() check is redundant in that
    # path but kept for backward compatibility when build_hf_model()
    # is called directly with a hand-built config.
    is_pre_quantized = is_quantized_onnx(current_path) or skip_optimize

    if is_pre_quantized:
        logger.info(
            "Pre-quantized model detected (QDQ nodes present). "
            "Skipping optimize + quantize, running analyze-only."
        )
        stages_skipped.append("optimize")
        # Optimize+analyze only, no autoconf re-optimization
        current_path, _, analyze_iterations, analyze_unsupported_nodes, analyze_details = (
            run_optimize_analyze_loop(
                model_path=current_path,
                optimized_path=optimized_path,
                config=config,
                ep=ep,
                device=device,
                **onnx_kwargs,
            )
        )
    else:
        logger.info("Optimizing ONNX model...")
        (
            current_path,
            opt_elapsed,
            analyze_iterations,
            analyze_unsupported_nodes,
            analyze_details,
        ) = run_optimize_analyze_loop(
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
        stage_timings["optimize"] = opt_elapsed
        stages_completed.append("optimize")
        logger.info("Optimize done (%.1fs) -> %s", opt_elapsed, optimized_path)
        logger.info("Portable ONNX ready: %s", current_path)

    # Persist config AFTER autoconf — includes discovered optimization flags
    config_path.write_text(json.dumps(config.to_dict(), indent=2))
    logger.debug("Config persisted: %s", config_path)

    # =========================================================================
    # [4] QUANTIZE (optional — config.quant=None means skip)
    # =========================================================================
    quant_result = None
    if is_pre_quantized:
        if "quantize" not in stages_skipped:
            stages_skipped.append("quantize")
        logger.info("Quantize skipped (pre-quantized model)")
    elif config.quant is not None:
        # Defensive fallback: catches the edge case where a direct caller
        # provides config.quant != None but the model already has QDQ nodes
        # (e.g., hand-built config without running generate_*_build_config).
        if is_quantized_onnx(current_path):
            logger.warning(
                "Model already contains QDQ nodes, skipping quantization. "
                "Set config.quant=None to silence this warning."
            )
            stages_skipped.append("quantize")
        else:
            logger.info("Quantizing model...")
            t0 = time.monotonic()
            # A model-type-specific quant policy (e.g. the qwen3_transformer_only
            # w8a16 finalizer) is resolved and applied inside ``quantize_onnx``
            # from ``config.quant.model_type``. Ensure it carries the resolved
            # variant so hand-built configs (that skipped assemble_build_config)
            # still trigger the right policy; ``quantize_onnx`` no-ops for
            # model types without a registered finalizer.
            if config.quant.model_type is None:
                config.quant.model_type = config.loader.model_type
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
    # [5] COMPILE (optional — config.compile=None means skip)
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
    # [6] FINALIZE — Copy last stage output as model.onnx
    # =========================================================================
    if current_path != final_path:
        copy_onnx_model(current_path, final_path)

    elapsed = time.monotonic() - start_time
    logger.info("Build complete in %.1fs -> %s", elapsed, final_path)

    # =========================================================================
    # [7] BUILD MANIFEST — Machine-readable build provenance
    # =========================================================================
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model_id": model_label,
        "task": task,
        "cache_key": cache_key,
        "config_hash": cache_key.rsplit("_", 1)[-1] if cache_key and "_" in cache_key else None,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "stages": [],
        "final_artifact": final_path.name,
        "analyze_iterations": analyze_iterations,
        "analyze_unsupported_node_count": analyze_unsupported_nodes,
        "analyze_details": analyze_details,
    }

    stage_filenames = {
        "export": export_path.name,
        "optimize": optimized_path.name,
        "quantize": quantized_path.name,
        "compile": compiled_path.name,
    }
    for stage_name in ["export", "optimize", "quantize", "compile"]:
        if stage_name in stages_completed:
            entry: dict[str, Any] = {
                "name": stage_name,
                "status": "completed",
                "filename": stage_filenames[stage_name],
                "elapsed_seconds": round(stage_timings.get(stage_name, 0), 3),
            }
            # Thread QuantizeResult metrics into manifest
            if stage_name == "quantize" and quant_result is not None:
                entry["nodes_quantized"] = quant_result.nodes_quantized
                entry["nodes_skipped"] = quant_result.nodes_skipped
                entry["calibration_time_seconds"] = round(quant_result.calibration_time_seconds, 3)
                entry["qdq_insertion_time_seconds"] = round(
                    quant_result.qdq_insertion_time_seconds, 3
                )
            manifest["stages"].append(entry)
        elif stage_name in stages_skipped:
            manifest["stages"].append(
                {
                    "name": stage_name,
                    "status": "skipped",
                    "filename": None,
                    "elapsed_seconds": None,
                }
            )

    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.debug("Build manifest persisted: %s", manifest_path)

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


# =============================================================================
# INTERNAL HELPERS
# =============================================================================


def _load_model(
    config: WinMLBuildConfig,
    model_id: str | None,
    trust_remote_code: bool,
    random_init: bool = False,
    hf_config: Any | None = None,
    model_type: str | None = None,
) -> Any:
    """Load PyTorch model — pretrained or random weights.

    Args:
        config: Build config (loader fields used).
        model_id: HuggingFace model ID or local path.
        trust_remote_code: Whether to trust remote code.
        random_init: If True, build with random weights (no download).
        hf_config: Optional pre-loaded ``PretrainedConfig`` to reuse. When
            provided, skips the ``AutoConfig.from_pretrained`` round-trip in
            both the random-init path and the pretrained ``load_hf_model``
            path (PR #719 dedup pattern).
    """
    task = config.loader.task

    if random_init:
        from transformers import AutoConfig

        if hf_config is not None:
            pass
        elif model_id is not None:
            hf_config = AutoConfig.from_pretrained(model_id)
        else:
            logger.warning(
                "--random-init without --model falls back to AutoConfig.for_model() "
                "class defaults, which may differ from pretrained configs and cause "
                "export failures. "
                "Prefer passing model_id when --random-init=True"
            )
            model_type = config.loader.model_type
            if model_type is None:
                raise ValueError(
                    "Random-weight build requires 'model_type' in loader config.\n"
                    "Options:\n"
                    "  1. Provide --model <model_id> to use pretrained weights\n"
                    "  2. Ensure config has loader.model_type (e.g., 'bert', 'resnet')\n"
                    "  3. Regenerate config: winml config -m <model_id> -o config.json"
                )
            hf_config = AutoConfig.for_model(model_type)

        # Prefer explicit model_class from loader config (set by winml config),
        # fall back to resolve_hf_model_class for auto-detection.
        # Annotated Any: resolvers return bare `type`, but the actual classes are
        # HF model classes with extra methods (from_config, from_pretrained, etc.)
        # that bare `type` doesn't expose.
        model_class: Any = None
        if config.loader.model_class:
            from ..loader import resolve_hf_model_class

            try:
                model_class = resolve_hf_model_class(config.loader.model_class)
            except ImportError:
                logger.warning(
                    "Could not resolve model_class '%s', falling back to auto-detect",
                    config.loader.model_class,
                )

        if model_class is None:
            from ..loader.resolution import resolve_task

            model_class = resolve_task(hf_config, task=task).model_class

        model_label = model_id or config.loader.model_type
        logger.info("Creating random-weight model: %s (from %s)", model_class.__name__, model_label)
        return model_class.from_config(hf_config)

    if model_id is not None:
        from ..loader import load_hf_model

        effective_trust = trust_remote_code or config.loader.trust_remote_code
        pytorch_model, _, _ = load_hf_model(
            model_name_or_path=model_id,
            task=task,
            model_class=config.loader.model_class,
            trust_remote_code=effective_trust,
            hf_config=hf_config,
            model_type=model_type,
        )
        return pytorch_model

    raise ValueError("Impossible to load model: no model_id provided and random_init=False.")
